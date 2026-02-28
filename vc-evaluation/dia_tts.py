import os
import soundfile as sf
import copy
import itertools
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
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")
        try:
            sf.read(filename)
            continue
        except Exception as e:
            pass
        filtered.append(r)

    return filtered

def loop(indices_device_pair):
    rows, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    import torch
    from dia.model import Dia

    torch.set_grad_enabled(False)
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

    for r in tqdm(rows):
        clone_from_text = f"[S1] {r['source_text']}"
        clone_from_audio = os.path.join("common-voice", r['language'], 'audio', r['audio_filename'])
        text_to_generate = f"[S1] {r['target_text']}"
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")

        try:
            sf.read(filename)
            continue
        except:
            pass

        try:
            with torch.no_grad():
                output = model.generate(
                    clone_from_text + text_to_generate,
                    audio_prompt=clone_from_audio,
                    use_torch_compile=True,
                    verbose=True,
                    cfg_scale=4.0,
                    temperature=1.8,
                    top_p=0.90,
                    cfg_filter_top_k=50,
                    max_tokens=1200,
                )
                os.makedirs(os.path.split(filename)[0], exist_ok = True)
                model.save_audio(filename, output)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
@click.command()
@click.option('--output')
@click.option('--replication', default = 1)
@click.option('--retry', default = 2)
def main(output, replication, retry):
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
            r['output'] = output
            actual_rows.append(r)
    
    filtered = multiprocessing(actual_rows, check, cores=20)
    print(len(filtered))
    if len(filtered):
        df_split = list(chunks(filtered, devices))

        loop_partial = partial(loop)

        with Pool(len(devices)) as pool:
            pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()
