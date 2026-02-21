import pandas as pd
import os
import random
import click
import json
from collections import defaultdict, Counter

@click.command()
@click.option('--file')
@click.option('--max_row', default=20000)
def main(file, max_row):
    rows = pd.read_parquet(file).to_dict(orient='records')

    groups = defaultdict(list)
    for row in rows:
        speaker = row['audio_filename'].split('/')[0]
        groups[speaker].append(row)

    kept = set()
    for char_type, group_rows in groups.items():
        print(f'{char_type}: {len(group_rows)} rows', flush=True)
        sampled = random.sample(group_rows, max_row) if len(group_rows) > max_row else group_rows
        random.shuffle(sampled)
        output_file = file.replace('.parquet', f'_{char_type}.parquet')
        pd.DataFrame(sampled).to_parquet(output_file, index=False)
        print(f'saved {len(sampled)} rows to {output_file}')
        kept.update(row['audio_filename'] for row in sampled)
    
    with open(file.replace('.parquet', '_kept.json'), 'w') as fopen:
        json.dump(list(kept), fopen)

if __name__ == '__main__':
    main()
