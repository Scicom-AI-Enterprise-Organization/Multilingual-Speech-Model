"""Pack the full corpus into the three task packs, in disk-bounded waves.

Rows come from `malaysia-ai/Multilingual-TTS-language` — 118M rows over 1,552 subsets,
carrying everything all three tasks need in one table:

    audio_filename   {subset}_audio/{rest}.mp3
    text             raw transcription            -> the TTS document
    speaker          {subset}_audio_{n}           -> the TTS prompt's speaker slot
    language         GlotLID v3 label             -> the STT language tag
    post-normalized  rule-normalized text         -> what the STT documents predict

Tokens live beside the audio in `malaysia-ai/Multilingual-TTS`:

    tokens   {folder}_neucodec.zip   -> {folder}_neucodec/{rest}.json
    audio    {folder}_audio*.zip     -> {audio_filename}, fetched separately and
                                        budgeted by fetch_corpus_audio.py

    TTS  <|im_start|>{speaker}: {text}<|speech_start|>{<|s_N|> …}<|im_end|>
    STT  <|im_start|><|STT|>{<|s_N|> …}<|{language}|>{text}<|im_end|>
    MEL  <|im_start|><|STT|><|mel_start|>{P × <|mel|>}<|mel_end|><|{language}|>{text}<|im_end|>

The TTS and STT packs read the same token JSONs, so a wave extracts once and packs both.
Mel rows are packed only where the audio is on disk; the rest are counted as `no_audio`.

Waves exist for inodes as much as disk: the whole corpus is ~121.8M token JSONs against
~141M free inodes. `--stage clean` drops a wave's JSONs once it is packed.

Usage:
    python multipacking_corpus.py --stage meta
    python multipacking_corpus.py --num-waves 8 --wave 0 --stage tokens
    python multipacking_corpus.py --num-waves 8 --wave 0 --stage pack --task all --workers 96
    python multipacking_corpus.py --num-waves 8 --wave 0 --stage clean
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_retry import file_with_retry, snapshot_with_retry
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

META_REPO = 'malaysia-ai/Multilingual-TTS-language'
DATA_REPO = 'malaysia-ai/Multilingual-TTS'
TASKS = ('tts', 'stt', 'mel')

G = {}


# ---------------------------------------------------------------- paths

def token_path(audio_filename):
    folder, _, rest = str(audio_filename).partition('/')
    if not rest:
        return None
    return f'{folder}_neucodec/' + rest.rsplit('.', 1)[0] + '.json'


def subset_folder(meta_file):
    """The audio folder a subset's rows point into, from its first row."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(meta_file)
    if 'audio_filename' not in pf.schema_arrow.names:
        return None
    for batch in pf.iter_batches(batch_size=16, columns=['audio_filename']):
        if batch.num_rows:
            return str(batch.column(0)[0]).partition('/')[0]
    return None


# ---------------------------------------------------------------- stages

def download_meta(base):
    snapshot_with_retry(META_REPO, local_dir=str(base / 'meta'),
                        allow_patterns=['*/*.parquet'], max_workers=8)
    return sorted((base / 'meta').glob('*/*.parquet'))


def download_tokens(base, meta_files, workers):
    """One `{folder}_neucodec.zip` per subset in this wave, extracted then deleted."""
    from huggingface_hub import HfApi

    marker_dir = base / '.tokens_extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    folders = {subset_folder(f) for f in meta_files}
    folders.discard(None)
    available = set(HfApi().list_repo_files(DATA_REPO, repo_type='dataset'))
    wanted = sorted(f'{fo}_neucodec.zip' for fo in folders
                    if f'{fo}_neucodec.zip' in available)
    missing = sorted(fo for fo in folders if f'{fo}_neucodec.zip' not in available)
    todo = [z for z in wanted if not (marker_dir / f'{z}.done').exists()]
    log(f'{len(folders)} subsets in wave | {len(wanted)} have token zips | '
        f'{len(missing)} have none (their rows are skipped) | {len(todo)} to fetch')
    if missing[:3]:
        log(f'  no zip, e.g.: {missing[:3]}')
    if not todo:
        return
    tasks = [(z, str(base), str(marker_dir)) for z in todo]
    with get_context('fork').Pool(min(workers, len(tasks))) as pool:
        pool.map(_fetch_tokens_one, tasks, chunksize=1)


def _fetch_tokens_one(args):
    name, base, marker_dir = args
    path = file_with_retry(DATA_REPO, name, local_dir=os.path.join(base, 'zips'))
    r = subprocess.run(['unzip', '-q', '-o', path, '-d', base], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'unzip {name}: {r.stderr[-300:]}')
    os.remove(path)
    Path(marker_dir, f'{name}.done').touch()


def pack_worker(args):
    import pandas as pd

    from chinidataset import ParquetWriter

    worker_id, files = args
    tokenizer = G['tokenizer']
    base = Path(G['base'])
    audio_base = Path(G['audio_base'])
    tasks = G['tasks']
    wave = G['wave']
    mel_id = G.get('mel_id')
    languages = G['languages']

    mel_prefix = mel_positions = None
    if 'mel' in tasks:
        from mel_audio import mel_positions

        mel_prefix = tokenizer('<|im_start|><|STT|><|mel_start|>',
                               add_special_tokens=False)['input_ids']

    stats = {'docs': 0, 'missing_token': 0, 'empty_text': 0, 'ratio': 0, 'no_lang': 0,
             'no_audio': 0, **{f'{t}_blocks': 0 for t in tasks},
             **{f'{t}_tokens': 0 for t in tasks}}

    with contextlib.ExitStack() as stack:
        acc = {}
        for task in tasks:
            out_dir = base / 'out' / f'corpus-{task}' / wave / f'{worker_id:05d}'
            shutil.rmtree(out_dir, ignore_errors=True)
            columns = MEL_COLUMNS if task == 'mel' else COLUMNS
            writer = stack.enter_context(
                ParquetWriter(out=str(out_dir), columns=columns, compression=None,
                              hashes=HASHES, size_limit=256 * 1024 * 1024))
            acc[task] = BlockAccumulator(writer, audio=task == 'mel')

        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            if 'audio_filename' not in df.columns:
                stats['missing_token'] += len(df)
                continue
            norm = df['post-normalized'] if 'post-normalized' in df.columns else df['text']
            speakers = df['speaker'] if 'speaker' in df.columns else df['language']
            langs = df['language'] if 'language' in df.columns else [None] * len(df)
            for audio_filename, raw, normalized, speaker, language in zip(
                    df['audio_filename'], df['text'], norm, speakers, langs):
                tp = token_path(audio_filename)
                if tp is None:
                    stats['missing_token'] += 1
                    continue
                try:
                    with open(base / tp) as fopen:
                        codes = json.load(fopen)
                except Exception:
                    stats['missing_token'] += 1
                    continue

                tts_text = clean_text(raw, normalized)
                stt_text = clean_text(normalized, raw)
                if not tts_text and not stt_text:
                    stats['empty_text'] += 1
                    continue
                if len((stt_text or tts_text).split()) > len(codes):
                    stats['ratio'] += 1
                    continue
                tagged = language in languages
                if not tagged:
                    stats['no_lang'] += 1

                stats['docs'] += 1
                s_tokens = ''.join([f'<|s_{c}|>' for c in codes])
                if 'tts' in acc and tts_text:
                    voice = clean_text(speaker) or (language or 'unk')
                    prompt = f'<|im_start|>{voice}: {tts_text}<|speech_start|>{s_tokens}<|im_end|>'
                    acc['tts'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                if 'stt' in acc and stt_text and tagged:
                    prompt = f'<|im_start|><|STT|>{s_tokens}<|{language}|>{stt_text}<|im_end|>'
                    acc['stt'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                if 'mel' in acc and stt_text and tagged:
                    rel = str(audio_filename)
                    n_samples = probe_samples(audio_base / rel)
                    if not n_samples:
                        stats['no_audio'] += 1
                    else:
                        tail = tokenizer(f'<|mel_end|><|{language}|>{stt_text}<|im_end|>',
                                         add_special_tokens=False)['input_ids']
                        ids = mel_prefix + [mel_id] * mel_positions(n_samples) + tail
                        acc['mel'].add(ids, path=rel, samples=n_samples)

            if worker_id == 0:
                log(f'worker 0: {Path(f).parent.name} done, {stats["docs"]} docs')

        for task, a in acc.items():
            a.flush()
            stats[f'{task}_blocks'] = a.blocks
            stats[f'{task}_tokens'] = a.tokens

    return stats


def pack(base, audio_base, files, tasks, wave, workers, added_tokens_file):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    tokenizer, added = tokenizer_from(added_tokens_file)
    languages = {t[2:-2] for t in added if t.startswith('<|') and t != '<|STT|>'}

    out_root = base / 'out'
    for task in tasks:
        shutil.rmtree(out_root / f'corpus-{task}' / wave, ignore_errors=True)
        (out_root / f'corpus-{task}' / wave).mkdir(parents=True)

    G.update(tokenizer=tokenizer, base=str(base), audio_base=str(audio_base), tasks=tasks,
             wave=wave, languages=languages,
             mel_id=tokenizer.convert_tokens_to_ids('<|mel|>') if 'mel' in tasks else None)
    chunks = list(enumerate(snake_chunks(files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(chunks)) as pool:
        results = pool.map(pack_worker, chunks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    summary = {'wave': wave, 'files': len(files), **totals}
    for task in tasks:
        task_dir = out_root / f'corpus-{task}' / wave
        merge_index(task_dir)
        n = len(StreamingDataset(local=str(task_dir)))
        summary[f'{task}_blocks_indexed'] = n
        log(f'corpus-{task}/{wave}: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B packed tokens), '
            f'{totals[f"{task}_tokens"] / 1e9:.2f}B real tokens')
    log(f'packed in {time.time() - t0:.0f}s | ' + ' '.join(f'{k}={v}' for k, v in totals.items()))
    with open(out_root / f'summary-{wave}.json', 'w') as f:
        json.dump(summary, f, indent=2)


def clean_wave(base, meta_files):
    """Drop this wave's token JSONs once it is packed — inodes are the binding limit."""
    folders = {subset_folder(f) for f in meta_files}
    folders.discard(None)
    freed = 0
    for folder in folders:
        d = base / f'{folder}_neucodec'
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            freed += 1
        marker = base / '.tokens_extracted' / f'{folder}_neucodec.zip.done'
        if marker.exists():
            marker.unlink()
    log(f'removed {freed} token trees')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', default='/root/share/corpus')
    p.add_argument('--audio-base', default='/root/share/corpus')
    p.add_argument('--added-tokens-file', default='/root/share/vocab/added_tokens.json')
    p.add_argument('--task', choices=['tts', 'stt', 'mel', 'token', 'all'], default='all')
    p.add_argument('--num-waves', type=int, default=8)
    p.add_argument('--wave', type=int, default=0)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    p.add_argument('--download-workers', type=int, default=8)
    p.add_argument('--stage', choices=['meta', 'tokens', 'pack', 'clean', 'all'], default='all')
    args = p.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    tasks = {'all': TASKS, 'token': ('tts', 'stt')}.get(args.task, (args.task,))

    if args.stage in ('meta', 'all'):
        download_meta(base)
    all_files = sorted((base / 'meta').glob('*/*.parquet'))
    if not all_files:
        raise SystemExit('no metadata — run --stage meta first')

    subsets = sorted({f.parent.name for f in all_files})
    step = -(-len(subsets) // args.num_waves)
    wave_subsets = set(subsets[args.wave * step:(args.wave + 1) * step])
    files = [f for f in all_files if f.parent.name in wave_subsets]
    wave = f'wave-{args.wave}'
    log(f'{len(subsets)} subsets total | {wave}: {len(wave_subsets)} subsets / {len(files)} files '
        f'| tasks {", ".join(tasks)}')

    if args.stage in ('tokens', 'all'):
        download_tokens(base, files, args.download_workers)
    if args.stage in ('pack', 'all'):
        pack(base, Path(args.audio_base), files, tasks, wave, args.workers, args.added_tokens_file)
    if args.stage == 'clean':
        clean_wave(base, files)


if __name__ == '__main__':
    main()
