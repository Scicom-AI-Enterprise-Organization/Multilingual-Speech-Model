import os
import soundfile as sf
import copy
import itertools
import librosa
from functools import partial
from multiprocess import Pool
from datasets import load_dataset
from tqdm import tqdm
import click

MODEL_NAME = os.environ.get('MODEL_NAME', 'Scicom-intl/Multilingual-TTS-1.7B-Base')

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
        except:
            pass
        filtered.append(r)

    return filtered

def loop(indices_device_pair):
    rows, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_qwen = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for r in tqdm(rows):
        clone_from_audio = os.path.join("common-voice", r['language'], 'audio', r['audio_filename'])
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")

        try:
            sf.read(filename)
            continue
        except:
            pass

        os.makedirs(os.path.split(filename)[0], exist_ok = True)

        y, sr = librosa.load(clone_from_audio, sr = 16000)
        with torch.no_grad():
            codes = codec.encode_code(torch.tensor(y)[None, None])
        tokens = ''.join([f'<|s_{i}|>' for i in codes[0, 0]])
        prompt = f"<|im_start|>{r['source_text']}<|speech_start|>{tokens}<|im_end|><|im_start|>{r['target_text']}<|speech_start|>"
        inputs = tokenizer(prompt,return_tensors="pt", add_special_tokens=True).to(model_qwen.device)

        with torch.no_grad():
            outputs = model_qwen.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.8,
                repetition_penalty=1.15,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        audio_tokens = re.findall(r'<\|s_(\d+)\|>', generated_text.split('<|speech_start|>')[-1])
        audio_tokens = [int(token) for token in audio_tokens]
        audio_codes = torch.tensor(audio_tokens)[None, None]

        with torch.no_grad():
            audio_waveform = codec.decode_code(audio_codes.cuda())
        
        os.makedirs(os.path.split(filename)[0], exist_ok = True)
        sf.write(filename, audio_waveform[0, 0].cpu().numpy(), 24000)
    
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
