import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import click
import json
import librosa
import pandas as pd
from functools import partial
from multiprocess import Pool
from tqdm import tqdm

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end
        
def new_path(f):
    splitted = f.split('/')
    folder = f.split('/')[0]
    folder = folder + '_speech_categories'
    new_f = os.path.join(folder, '/'.join(splitted[1:]))
    new_f = new_f.replace('.mp3', '.json').replace('.wav', '.json')
    return new_f

def multiprocessing(strings, function, cores=6, returned=True):
    df_split = old_chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))

def loop(
    indices_device_pair,
    language,
):
    rows, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    from speech_categories_func import (
        predict_fluency,
        predict_quality,
        predict_accent,
        predict_sex_age,
        predict_gender,
        predict_emotion,
    )
    from speech_stats_func import (
        rate_apply,
        pitch_apply,
        snr_apply,
        squim_apply,
    )

    for r in tqdm(rows):    
        f = r['audio_filename']
        t = r['text']
        filename = new_path(f)
        if os.path.exists(filename):
            try:
                with open(filename) as fopen:
                    json.load(fopen)
                continue
            except:
                pass

        try:
            y, sr = librosa.load(f, sr = 16000)
            fluency = predict_fluency(y)
            quality = predict_quality(y)
            accent = predict_accent(y)
            sex_age = predict_sex_age(y)
            gender = predict_gender(y)
            emotion = predict_emotion(y)
            pitch = pitch_apply(y, sr)
            snr = snr_apply(y, sr)
            squim = squim_apply(y, sr)
            rate = rate_apply(t, language, snr['vad_duration'])
            new_r = {**r, **fluency, **quality, **accent, **sex_age, **gender, **emotion, **pitch, **snr, **squim, **rate}
            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            with open(filename, 'w') as fopen:
                json.dump(new_r, fopen)
        except Exception as e:
            print(e)

@click.command()
@click.option('--file')
@click.option('--replication', default = 1)
@click.option('--language', default = 'ms')
def main(file, replication, language):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices

    rows = pd.read_parquet(file).to_dict(orient='records')
    filtered = []
    for r in tqdm(rows): 
        f = r['audio_filename']
        filename = new_path(f)
        if os.path.exists(filename):
            try:
                with open(filename) as fopen:
                    json.load(fopen)
                continue
            except:
                pass
        filtered.append(r)

    if len(filtered):
        df_split = list(chunks(filtered, devices))

        loop_partial = partial(
            loop,
            language=language,
        )

        with Pool(len(devices)) as pool:
            pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()



    