import json
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocess import Pool
import itertools
import click

SPEAKER_RATE_BINS = ["very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast"]
SNR_BINS = ["very noisy", "quite noisy", "slightly noisy", "moderate ambient sound", "slightly clear", "quite clear", "very clear"]
REVERBERATION_BINS = ["very roomy sounding", "quite roomy sounding", "slightly roomy sounding", "moderate reverberation", "slightly confined sounding", "quite confined sounding", "very confined sounding"]
UTTERANCE_LEVEL_STD = ["very monotone", "quite monotone", "slightly monotone", "moderate intonation", "slightly expressive", "quite expressive", "very expressive"]
SI_SDR_BINS = ["extremely noisy", "very noisy", "noisy", "slightly noisy", "almost no noise", "very clear"]
PESQ_BINS = ["very bad speech quality", "bad speech quality", "slightly bad speech quality", "moderate speech quality", "great speech quality", "wonderful speech quality"]
SPEAKER_LEVEL_PITCH_BINS = ["very low pitch", "quite low pitch", "slightly low pitch", "moderate pitch", "slightly high pitch", "quite high pitch", "very high pitch"]

def chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)

def multiprocessing(strings, function, cores=6, returned=True):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))

def read(files):
    files, _ = files
    data = []
    for f in tqdm(files):
        with open(f) as fopen:
            d = json.load(fopen)
        s = f.split('/')[0].split('_speech_categories')[0]
        if s.endswith('_audio'):
            s = s.replace('_audio', '')
        d['source'] = s
        data.append(d)
    return data

def bins_to_text(values, text_bins):
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    lower, upper = np.percentile(values, [5, 95])
    values = np.clip(values, lower, upper)
    _, bin_edges = np.histogram(values, bins=len(text_bins))
    return bin_edges

def get_labels(values, bin_edges, labels):
    labels = np.array(labels)
    values = np.array(values, dtype=float)
    # NaN gets mapped to the lowest label rather than undefined behaviour
    values = np.where(np.isnan(values), bin_edges[0], values)
    index_bins = np.searchsorted(bin_edges, values, side="left")
    index_bins = np.clip(index_bins - 1, 0, len(labels) - 1)
    return labels[index_bins].tolist()

@click.command()
@click.option('--pattern')
@click.option('--output', default='output.parquet')
def main(pattern, output):
    files = glob(pattern)
    data = multiprocessing(files, read, cores=10)

    df = pd.DataFrame(data)

    bin_configs = [
        ('speaking_rate', SPEAKER_RATE_BINS, 'speaking_rate_label'),
        ('snr',           SNR_BINS,          'noise_label'),
        ('c50',           REVERBERATION_BINS,'reverberation_label'),
        ('pitch_std',     UTTERANCE_LEVEL_STD,'speech_monotony_label'),
        ('sdr',           SI_SDR_BINS,       'sdr_label'),
        ('pesq',          PESQ_BINS,         'pesq_label'),
    ]

    for key, bins, label_col in bin_configs:
        values = df[key].tolist()
        bin_edges = bins_to_text(values, bins)
        df[label_col] = get_labels(values, bin_edges, bins)

    # pitch_label must be binned per gender since male/female pitch ranges differ
    df['pitch_label'] = ''
    for gender, group in df.groupby('gender'):
        values = group['pitch_mean'].tolist()
        bin_edges = bins_to_text(values, SPEAKER_LEVEL_PITCH_BINS)
        df.loc[group.index, 'pitch_label'] = get_labels(values, bin_edges, SPEAKER_LEVEL_PITCH_BINS)

    df['fluency'] = df['fluency'].apply(json.dumps)
    df['quality'] = df['quality'].apply(lambda q: [p for p in q if p is not None])

    df.to_parquet(output)

if __name__ == '__main__':
    main()
