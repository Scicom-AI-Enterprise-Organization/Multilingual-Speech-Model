import time
import numpy as np
import os
import soundfile as sf
import copy
import itertools
import librosa
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from functools import partial
from multiprocess import Pool
from datasets import load_dataset
from tqdm import tqdm
import click

def old_chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def multiprocessing(strings, function, cores=6, returned=True):
    df_split = old_chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))

def check(indices_device_pair):
    rows, device = indices_device_pair
    filtered = []
    for r in tqdm(rows):
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.json")
        try:
            with open(filename) as fopen:
                json.load(fopen)
            continue
        except:
            pass
        filtered.append(r)

    return filtered

def loop(indices_device_pair):
    rows, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    import torch
    from titanet_vectors import load
    torch.set_float32_matmul_precision('high')

    model = load().cuda().eval()
    _ = model.to(torch.float16)

    def convert_to_vector(audio):
        audio, _ = librosa.load(audio, sr=16000)
        with torch.no_grad():
            a = torch.from_numpy(audio).unsqueeze(0).cuda()
            a = a.to(torch.float16)
            lens = torch.tensor([a.shape[-1]]).cuda()
            out_fp16 = model(a, lens)
            return out_fp16[1].cpu().numpy()

    for r in tqdm(rows):
        clone_from_audio = os.path.join("common-voice", r['language'], 'audio', r['audio_filename'])
        to_audio = os.path.join(r['output_folder'], f"{r['index']}-{r['retry']}.mp3")
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.json")

        if len(r['source_text']) < 5 or len(r['target_text']) < 5:
            continue
        
        try:
            clone_from_audio_v = convert_to_vector(clone_from_audio)
            to_audio_v = convert_to_vector(to_audio)

            similarity = float(cosine_similarity(clone_from_audio_v, to_audio_v)[0, 0])

            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            with open(filename, 'w') as fopen:
                json.dump(similarity, fopen)
        except:
            pass
        
@click.command()
@click.option('--output_folder')
@click.option('--output', default='similarity')
@click.option('--replication', default = 1)
@click.option('--retry', default = 2)
def main(output_folder, output, replication, retry):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices

    ds = load_dataset("Scicom-intl/Evaluation-Multilingual-VC", 'combine_filtered_whisper_large_v3')
    df = ds['train'].to_pandas()
    df['index'] = df.index
    rows = df.to_dict(orient='records')
    actual_rows = []
    for r in rows:
        for k in range(retry):
            r = copy.copy(r)
            r['retry'] = k
            r['output_folder'] = output_folder
            r['output'] = output
            actual_rows.append(r)
    
    filtered = multiprocessing(actual_rows, check, cores=20)
    print(len(filtered))
    if len(filtered):
        df_split = list(chunks(filtered, devices))

        loop_partial = partial(loop)

        with Pool(len(devices)) as pool:
            pooled = pool.map(loop_partial, df_split)
    
    # Aggregate similarities: language -> index -> [scores per retry]
    data = defaultdict(lambda: defaultdict(list))
    for r in actual_rows:
        try:
            filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.json")
            with open(filename) as fopen:
                similarity = json.load(fopen)
            data[r['language']][r['index']].append(similarity)
        except:
            pass

    # Print per-language averages (averaging retries per index first)
    all_lang_avgs = []
    for lang in sorted(data):
        index_avgs = [np.mean(scores) for scores in data[lang].values()]
        lang_avg = np.mean(index_avgs)
        all_lang_avgs.append(lang_avg)
        print(f"{lang}: {lang_avg:.4f} ({len(index_avgs)} samples)")

    print(f"\nGlobal average: {np.mean(all_lang_avgs):.4f}")


if __name__ == '__main__':
    main()

