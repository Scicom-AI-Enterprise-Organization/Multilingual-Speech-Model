import pandas as pd
import os
import random
import click
from collections import defaultdict, Counter

def detect_char_type(text):
    counts = Counter()
    has_hiragana_katakana = False
    has_hangul = False

    for ch in text:
        if ch.isspace() or not ch.isprintable():
            continue
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            has_hiragana_katakana = True
            counts['japanese'] += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            has_hangul = True
            counts['korean'] += 1
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
              0x20000 <= cp <= 0x2A6DF):
            counts['cjk_ideograph'] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            counts['arabic'] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts['devanagari'] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts['tamil'] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts['cyrillic'] += 1
        elif 0x0020 <= cp <= 0x024F:
            counts['latin'] += 1

    if not counts:
        return 'other'

    # Hiragana/Katakana are exclusively Japanese; CJK ideographs in such text are also Japanese
    if has_hiragana_katakana:
        counts['japanese'] = counts.get('japanese', 0) + counts.pop('cjk_ideograph', 0)
        return 'japanese'

    # Hangul is exclusively Korean
    if has_hangul:
        counts['korean'] = counts.get('korean', 0) + counts.pop('cjk_ideograph', 0)
        return 'korean'

    # Pure CJK ideographs without Hiragana/Katakana/Hangul → Chinese
    if 'cjk_ideograph' in counts:
        counts['chinese'] = counts.pop('cjk_ideograph')

    return max(counts, key=counts.get)

@click.command()
@click.option('--file')
@click.option('--max_row', default=10000)
def main(file, max_row):
    rows = pd.read_parquet(file).to_dict(orient='records')

    groups = defaultdict(list)
    for row in rows:
        char_type = detect_char_type(row['text'])
        groups[char_type].append(row)

    kept = set()
    for char_type, group_rows in groups.items():
        print(f'{char_type}: {len(group_rows)} rows', flush=True)
        sampled = random.sample(group_rows, max_row) if len(group_rows) > max_row else group_rows
        random.shuffle(sampled)
        output_file = file.replace('.parquet', f'_{char_type}.parquet')
        pd.DataFrame(sampled).to_parquet(output_file, index=False)
        print(f'saved {len(sampled)} rows to {output_file}')
        kept.update(row['audio_filename'] for row in sampled)

    deleted = 0
    for row in rows:
        f = row['audio_filename']
        if f not in kept and os.path.exists(f):
            os.remove(f)
            deleted += 1
    print(f'deleted {deleted} audio files')

if __name__ == '__main__':
    main()
