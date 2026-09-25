"""Pack the filtered Common Voice 22 subset into the ablation's three task packs.

Rows come from `malaysia-ai/Multilingual-TTS` config `common-voice-22` — 6,921,399 rows,
29.1% of raw CV22, already filtered and carrying `speaker`. The audio and the NeuCodec
tokens live in `malaysia-ai/common_voice_22_0`:

    metadata  audio_filename  audio_trim/{lang}/{split}/{id}.mp3
    tokens    audio_trim_neucodec-*.zip  -> audio_trim_neucodec/{lang}/{split}/{id}.json
    audio     audio-*.zip                -> audio/{lang}/{split}/{id}.mp3

Same three documents as preparation/multipacking_fleurs.py, with the path's language
segment as the locale tag:

    TTS  <|im_start|>{speaker}: {text}<|speech_start|>{<|s_N|> …}<|im_end|>
    STT  <|im_start|><|STT|>{<|s_N|> …}<|{lang}|>{text}<|im_end|>
    MEL  <|im_start|><|STT|><|mel_start|>{P × <|mel|>}<|mel_end|><|{lang}|>{text}<|im_end|>

NOTE: the repo publishes the tokens of the *trimmed* audio but only the *untrimmed*
mp3s, so the mel pack carries each clip's leading/trailing silence while the token packs
do not. Every optimizer sees the same data, so the sweep stays fair; a codec-vs-mel
reading has to account for it.

The language tokens must cover FLEURS and CV22 at once or the two corpora disagree on
token ids, so both packers take the same `--added-tokens-file`; build it with
`--stage tokens-file` before packing either.

Usage:
    python multipacking_cv22.py --stage meta      # metadata parquets only
    python multipacking_cv22.py --stage tokens    # NeuCodec zips (6.1GB) -> extract
    python multipacking_cv22.py --stage audio     # audio zips (286GB), keeping only packed rows
    python multipacking_cv22.py --task token --workers 96
    python multipacking_cv22.py --task mel --splits dev
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multipacking_fleurs import (
    BLOCK_SIZE,
    COLUMNS,
    HASHES,
    MEL_COLUMNS,
    BlockAccumulator,
    clean_text,
    log,
    probe_samples,
    snake_chunks,
    tokenizer_from,
)

META_REPO = 'malaysia-ai/Multilingual-TTS'
DATA_REPO = 'malaysia-ai/common_voice_22_0'
SUBSET = 'common-voice-22'
TASKS = ('tts', 'stt', 'mel')

G = {}


# ---------------------------------------------------------------- download

def download_meta(base):
    from huggingface_hub import snapshot_download

    snapshot_download(META_REPO, repo_type='dataset', local_dir=str(base / 'meta'),
                      allow_patterns=[f'{SUBSET}/*.parquet'], max_workers=8)
    return sorted((base / 'meta' / SUBSET).glob('*.parquet'))


def shard_meta(base, files, rows_per_shard):
    """Rewrite the 3 huge metadata parquets into worker-sized shards.

    The pack pool assigns whole files to workers, and 3 files means 3 workers for 6.9M
    rows. Sharding is streamed batch by batch, so no worker ever holds a 2.3M-row frame.
    """
    import pyarrow.parquet as pq

    out_dir = base / 'meta' / 'shards'
    if out_dir.exists() and any(out_dir.glob('*.parquet')):
        existing = sorted(out_dir.glob('*.parquet'))
        log(f'{len(existing)} metadata shards already present')
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    written, rows = [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=rows_per_shard,
                                     columns=['audio_filename', 'text', 'speaker']):
            path = out_dir / f'shard-{len(written):05d}.parquet'
            pq.write_table(pa_table(batch), path)
            written.append(path)
            rows += batch.num_rows
    log(f'{rows:,} rows -> {len(written)} shards of ~{rows_per_shard:,}')
    return written


def pa_table(batch):
    import pyarrow as pa

    return pa.Table.from_batches([batch])


def _zip_names(kind):
    from huggingface_hub import HfApi

    prefix = 'audio_trim_neucodec-' if kind == 'tokens' else 'audio-'
    return sorted(f for f in HfApi().list_repo_files(DATA_REPO, repo_type='dataset')
                  if f.startswith(prefix) and f.endswith('.zip'))


def download_archives(base, kind, wanted=None, workers=8):
    """Fetch each zip, extract (optionally only `wanted` members), drop the zip.

    The audio archives are 286GB for the whole of raw CV22 while the filtered rows need
    ~29% of it, so `wanted` keeps the extracted tree to what the packs actually read.
    """
    marker_dir = base / f'.{kind}_extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    names = _zip_names(kind)
    todo = [n for n in names if not (marker_dir / f'{n}.done').exists()]
    log(f'{kind}: {len(names)} archives, {len(todo)} to fetch')
    if not todo:
        return
    G['wanted'] = wanted
    tasks = [(n, str(base), str(marker_dir)) for n in todo]
    with get_context('fork').Pool(min(workers, len(tasks))) as pool:
        pool.map(_fetch_one, tasks, chunksize=1)
    G.pop('wanted', None)
    log(f'{kind}: done')


def _fetch_one(args):
    name, base, marker_dir = args
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            path = hf_hub_download(DATA_REPO, name, repo_type='dataset',
                                   local_dir=os.path.join(base, 'zips'))
            break
        except Exception as e:
            if attempt == 4:
                raise
            log(f'{name}: download failed ({e}); retrying')
            time.sleep(20)

    wanted = G.get('wanted')
    if wanted is None:
        subprocess.run(['unzip', '-q', '-o', path, '-d', base], check=True)
        kept = None
    else:
        kept = 0
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member in wanted:
                    zf.extract(member, base)
                    kept += 1
    os.remove(path)
    Path(marker_dir, f'{name}.done').touch()
    log(f'{name}: extracted' + (f' {kept} wanted members' if kept is not None else ''))


# ---------------------------------------------------------------- paths

def token_path(audio_filename):
    """audio_trim/{lang}/{split}/{id}.mp3 -> audio_trim_neucodec/{lang}/{split}/{id}.json"""
    folder, _, rest = str(audio_filename).partition('/')
    if not rest:
        return None
    return f'{folder}_neucodec/' + rest.rsplit('.', 1)[0] + '.json'


def audio_path(audio_filename):
    """The published mp3s sit under `audio/`, not `audio_trim/` (untrimmed originals)."""
    folder, _, rest = str(audio_filename).partition('/')
    return f'audio/{rest}' if rest else None


def language_of(audio_filename):
    parts = str(audio_filename).split('/')
    return parts[1] if len(parts) > 2 else None


def split_of(audio_filename):
    parts = str(audio_filename).split('/')
    return parts[2] if len(parts) > 3 else None


# ---------------------------------------------------------------- packing

def pack_worker(args):
    import contextlib

    import pandas as pd

    from chinidataset import ParquetWriter

    worker_id, files = args
    tokenizer = G['tokenizer']
    base = Path(G['base'])
    tasks = G['tasks']
    splits = set(G['splits'])
    suffix = G['suffix']
    mel_id = G.get('mel_id')
    languages = G['languages']

    mel_prefix = mel_positions = None
    if 'mel' in tasks:
        from mel_audio import mel_positions

        mel_prefix = tokenizer('<|im_start|><|STT|><|mel_start|>',
                               add_special_tokens=False)['input_ids']

    stats = {'docs': 0, 'missing': 0, 'empty_text': 0, 'ratio': 0, 'no_audio': 0, 'skipped_split': 0,
             **{f'{t}_blocks': 0 for t in tasks}, **{f'{t}_tokens': 0 for t in tasks}}

    with contextlib.ExitStack() as stack:
        acc = {}
        for task in tasks:
            out_dir = base / 'out' / f'cv22-{task}{suffix}' / f'{worker_id:05d}'
            shutil.rmtree(out_dir, ignore_errors=True)
            columns = MEL_COLUMNS if task == 'mel' else COLUMNS
            writer = stack.enter_context(
                ParquetWriter(out=str(out_dir), columns=columns, compression=None, hashes=HASHES,
                              size_limit=256 * 1024 * 1024))
            acc[task] = BlockAccumulator(writer, audio=task == 'mel')

        for f in files:
            df = pd.read_parquet(f, columns=['audio_filename', 'text', 'speaker'])
            for audio_filename, text, speaker in zip(df['audio_filename'], df['text'], df['speaker']):
                if split_of(audio_filename) not in splits:
                    stats['skipped_split'] += 1
                    continue
                language = language_of(audio_filename)
                if language not in languages:
                    stats['missing'] += 1
                    continue
                text = clean_text(text)
                if not text:
                    stats['empty_text'] += 1
                    continue
                try:
                    codes = read_codes_json(base / token_path(audio_filename))
                except Exception:
                    stats['missing'] += 1
                    continue
                if len(text.split()) > len(codes):
                    stats['ratio'] += 1
                    continue

                stats['docs'] += 1
                if 'tts' in acc or 'stt' in acc:
                    s_tokens = ''.join([f'<|s_{c}|>' for c in codes])
                    if 'tts' in acc:
                        voice = clean_text(speaker) or language
                        prompt = f'<|im_start|>{voice}: {text}<|speech_start|>{s_tokens}<|im_end|>'
                        acc['tts'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    if 'stt' in acc:
                        prompt = f'<|im_start|><|STT|>{s_tokens}<|{language}|>{text}<|im_end|>'
                        acc['stt'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                if 'mel' in acc:
                    rel = audio_path(audio_filename)
                    n_samples = probe_samples(base / rel)
                    if not n_samples:
                        stats['no_audio'] += 1
                    else:
                        tail = tokenizer(f'<|mel_end|><|{language}|>{text}<|im_end|>',
                                         add_special_tokens=False)['input_ids']
                        ids = mel_prefix + [mel_id] * mel_positions(n_samples) + tail
                        acc['mel'].add(ids, path=rel, samples=n_samples)

            if worker_id == 0:
                log(f'worker 0: finished {Path(f).name}, {stats["docs"]} docs')

        for task, a in acc.items():
            a.flush()
            stats[f'{task}_blocks'] = a.blocks
            stats[f'{task}_tokens'] = a.tokens

    return stats


def read_codes_json(path):
    with open(path) as f:
        return json.load(f)


def languages_of(files, splits):
    """Language codes present in the selected rows — the locale tags the packs will use."""
    import pandas as pd

    seen = set()
    for f in files:
        df = pd.read_parquet(f, columns=['audio_filename'])
        for name in df['audio_filename']:
            if split_of(name) in splits:
                lang = language_of(name)
                if lang:
                    seen.add(lang)
    return sorted(seen)


def build_shared_tokens(base, fleurs_dir, files, splits, out_file):
    """One token list covering FLEURS and CV22, so both corpora agree on ids.

    Appended after <|speech_start|> and the 65,536 <|s_N|>: <|STT|>, then every language
    tag from either corpus, then the mel tokens. Packing either corpus against a
    different list would silently shift the mel ids.
    """
    from mel_audio import MEL_TOKENS

    fleurs_locales = sorted({p.name.rsplit('-', 1)[0]
                             for p in Path(fleurs_dir).glob('*.parquet')}) if fleurs_dir else []
    cv22_languages = languages_of(files, splits)
    tags = sorted(set(fleurs_locales) | set(cv22_languages))
    tokens = ['<|STT|>'] + [f'<|{t}|>' for t in tags] + list(MEL_TOKENS)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(tokens, f, indent=2)
    log(f'{len(tokens)} added tokens -> {out_file} '
        f'({len(fleurs_locales)} FLEURS locales + {len(cv22_languages)} CV22 languages, '
        f'{len(tags)} distinct)')
    return tokens


def pack(base, files, tasks, splits, workers, added_tokens_file, suffix=''):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    tokenizer, added = tokenizer_from(added_tokens_file)
    languages = {t[2:-2] for t in added if t.startswith('<|') and t not in ('<|STT|>',)}

    out_root = base / 'out'
    for task in tasks:
        shutil.rmtree(out_root / f'cv22-{task}{suffix}', ignore_errors=True)
        (out_root / f'cv22-{task}{suffix}').mkdir(parents=True)

    G.update(tokenizer=tokenizer, base=str(base), tasks=tasks, splits=splits, suffix=suffix,
             languages=languages,
             mel_id=tokenizer.convert_tokens_to_ids('<|mel|>') if 'mel' in tasks else None)
    chunks = list(enumerate(snake_chunks(files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(chunks)) as pool:
        results = pool.map(pack_worker, chunks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    summary = {'files': len(files), 'splits': splits, **totals}
    for task in tasks:
        task_dir = out_root / f'cv22-{task}{suffix}'
        merge_index(task_dir)
        n = len(StreamingDataset(local=str(task_dir)))
        summary[f'{task}_blocks_indexed'] = n
        log(f'cv22-{task}{suffix}: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B packed tokens), '
            f'{totals[f"{task}_tokens"] / 1e9:.2f}B real tokens')
    log(f'packed in {time.time() - t0:.0f}s | ' + ' '.join(f'{k}={v}' for k, v in totals.items()))
    with open(out_root / f'summary-cv22{suffix or "-train"}.json', 'w') as f:
        json.dump(summary, f, indent=2)


def wanted_audio_members(files, splits):
    """Only the mp3s the packs will read — 29% of what the audio archives carry."""
    import pandas as pd

    wanted = set()
    for f in files:
        df = pd.read_parquet(f, columns=['audio_filename'])
        for name in df['audio_filename']:
            if split_of(name) in splits:
                rel = audio_path(name)
                if rel:
                    wanted.add(rel)
    return wanted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/root/share/cv22')
    parser.add_argument('--fleurs-data', default='/root/share/fleurs/data',
                        help='FLEURS metadata dir, read only to collect its locale tags')
    parser.add_argument('--added-tokens-file', default=None,
                        help='shared token list (default: <base>/out/added_tokens.json)')
    parser.add_argument('--splits', nargs='+', default=['train'], choices=['train', 'dev', 'test'])
    parser.add_argument('--task', choices=['tts', 'stt', 'mel', 'token', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--rows-per-shard', type=int, default=40000,
                        help='metadata shard size; the pack pool works one shard per worker')
    parser.add_argument('--download-workers', type=int, default=8)
    parser.add_argument('--stage',
                        choices=['meta', 'shard', 'tokens-file', 'tokens', 'audio', 'pack', 'all'],
                        default='all')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    tasks = {'all': TASKS, 'token': ('tts', 'stt')}.get(args.task, (args.task,))
    suffix = '' if args.splits == ['train'] else '-' + '-'.join(sorted(args.splits))
    added_file = Path(args.added_tokens_file) if args.added_tokens_file else base / 'out' / 'added_tokens.json'

    if args.stage in ('meta', 'all'):
        download_meta(base)
    files = sorted((base / 'meta' / SUBSET).glob('*.parquet'))
    if not files:
        raise SystemExit('no metadata — run --stage meta first')
    log(f'{len(files)} metadata parquets, splits {args.splits}, tasks {", ".join(tasks)}')

    if args.stage in ('shard', 'pack', 'all'):
        files = shard_meta(base, files, args.rows_per_shard)
    if args.stage in ('tokens-file', 'all') and not added_file.exists():
        build_shared_tokens(base, args.fleurs_data, files, args.splits, added_file)
    if args.stage in ('tokens', 'all'):
        download_archives(base, 'tokens', workers=args.download_workers)
    if args.stage == 'audio' or (args.stage == 'all' and 'mel' in tasks):
        log('collecting the mp3 members the packs need')
        download_archives(base, 'audio', wanted=wanted_audio_members(files, args.splits),
                          workers=args.download_workers)
    if args.stage in ('pack', 'all'):
        pack(base, files, tasks, args.splits, args.workers, added_file, suffix)


if __name__ == '__main__':
    main()
