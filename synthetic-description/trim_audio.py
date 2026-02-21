import os
import click
import pandas as pd
from tqdm import tqdm

@click.command()
@click.option('--file')
@click.option('--small_file')
def main(file, small_file):
    df = pd.read_parquet(file)
    small_df = pd.read_parquet(small_file)

    audio_filename = set([df['audio_filename'].iloc[i] for i in range(len(df))])
    small_audio_filename = set([small_df['audio_filename'].iloc[i] for i in range(len(small_df))])

    for f in tqdm(audio_filename):
        if f not in small_audio_filename:
            os.remove(f)
