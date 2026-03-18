import os
import soundfile as sf
import copy
import itertools
from functools import partial
from multiprocess import Pool
from datasets import load_dataset
from tqdm import tqdm
import click

COMMON_VOICE_TO_CHATTERBOX = {
    "ar": "ar",      # Arabic
    "da": "da",      # Danish
    "de": "de",      # German
    "el": "el",      # Greek
    "en": "en",      # English
    "es": "es",      # Spanish
    "fi": "fi",      # Finnish
    "fr": "fr",      # French
    "he": "he",      # Hebrew
    "hi": "hi",      # Hindi
    "it": "it",      # Italian
    "ja": "ja",      # Japanese
    "ko": "ko",      # Korean
    # "ms": None,    # Malay - NOT in Common Voice mapping
    "nl": "nl",      # Dutch
    "nn-NO": "no",   # Norwegian (nn-NO -> whisper "nn" -> chatterbox "no")
    "pl": "pl",      # Polish
    "pt": "pt",      # Portuguese
    "ru": "ru",      # Russian
    "sv-SE": "sv",   # Swedish
    "sw": "sw",      # Swahili
    "tr": "tr",      # Turkish
    "zh-CN": "zh",   # Chinese (Simplified)
    "zh-TW": "zh",   # Chinese (Traditional) - maps to same chatterbox "zh"
    # "zh-HK": None, # Cantonese -> whisper "yue", not supported in chatterbox
}

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
    from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    torch.set_grad_enabled(False)
    model = ChatterboxMultilingualTTS.from_pretrained('cuda')

    for r in tqdm(rows):
        text_to_generate = f"{r['target_text']}"
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")
        language_id = COMMON_VOICE_TO_CHATTERBOX[r['language']]
        generate_kwargs = {
            'exaggeration': 0.5,
            'temperature': 0.8,
            'cfg_weight': 0.5,
        }

        try:
            sf.read(filename)
            continue
        except:
            pass

        try:
            wav = model.generate(
                r['target_text'],
                language_id=language_id,
                **generate_kwargs
            )
            wav = wav.squeeze(0).numpy()
            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            sf.write(filename, wav, model.sr)
        except Exception as e:
            pass
        
    
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
        if r['language'] not in COMMON_VOICE_TO_CHATTERBOX:
            continue
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
