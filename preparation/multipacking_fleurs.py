"""Pack FLEURS-R NeuCodec tokens into TTS and STT multipacking blocks.

Source: malaysia-ai/fleurs-r-neucodec-all-languages — 102 locales,
`data/{locale}-{split}.parquet` metadata + `neucodec/{locale}-{split}-rank{N}.zip`
token archives holding `neucodec/{locale}/{split}/{id}.pt`
(`{'source': ..., 'codes': IntTensor[1, 1, T]}`).

Two packs out of one pass over the tokens, same rows and same text on both sides so
the optimizer sweep compares tasks, not corpora:

    TTS  <|im_start|>{speaker}: {text}<|speech_start|>{<|s_N|> …}<|im_end|>
    STT  <|im_start|><|STT|>{<|s_N|> …}<|{locale}|>{text}<|im_end|>

The speaker slot of the TTS prompt (preparation/multipacking-tts.ipynb) takes the repo's
`speaker` column — the TitaNet voice clusters added by fleurs-dataset/ — and falls back to
the locale when a subset predates that column. The locale is also the STT language tag: it
is ground truth, so no GlotLID pass is needed here (stt/README.md).

NOTE: the packs the published FLEURS ablation ran on were built on 2026-09-13, before the
speaker column existed, so their TTS prompts carry `{locale}: `. Re-packing now changes the
TTS side (the STT side is unaffected) — re-run both sweeps if you do.

Block format matches preparation/multipacking.py and stt/multipacking_stt.py:
ChiniDataset parquet, ~10,240-token attention-isolated blocks (per-doc position_ids
reset, attention_mask = per-doc lengths).

Tokenizer: Qwen3-1.7B-Base + <|speech_start|> + 65,536 <|s_N|> (TTS ids), then <|STT|>
and the sorted locale tokens appended AFTER them, so speech-token ids are identical in
both packs. The appended list lands in <out>/fleurs_stt_added_tokens.json — the STT
trainer must add the same tokens in the same order (`--added_tokens_file`).

Usage:
    python multipacking_fleurs.py --base-dir /share/multilingual-tts/fleurs --workers 96
    python multipacking_fleurs.py --stage download
    python multipacking_fleurs.py --task tts --locales 'en_us' 'ms_my'
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import fnmatch
import json
import shutil
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = 'malaysia-ai/fleurs-r-neucodec-all-languages'
BLOCK_SIZE = 1024 * 10
COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',
    'text': 'str',
}
HASHES = ['sha1', 'xxh64']
TASKS = ('tts', 'stt')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def match(name, patterns):
    return not patterns or any(fnmatch.fnmatch(name, p) for p in patterns)


# ---------------------------------------------------------------- download

def download(base, locales, splits, keep_zips=False):
    """Metadata parquets + token zips for the selected locales/splits, extracted once."""
    from huggingface_hub import snapshot_download

    meta_patterns = [f'data/{l}-{s}.parquet' for l in (locales or ['*']) for s in splits]
    snapshot_download(REPO, repo_type='dataset', local_dir=base, allow_patterns=meta_patterns,
                      max_workers=8)

    zip_patterns = [f'neucodec/{l}-{s}-rank*.zip' for l in (locales or ['*']) for s in splits]
    snapshot_download(REPO, repo_type='dataset', local_dir=base / 'zips', allow_patterns=zip_patterns,
                      max_workers=8)

    marker_dir = base / '.extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted((base / 'zips' / 'neucodec').glob('*.zip'))
    todo = [z for z in zips if not (marker_dir / f'{z.name}.done').exists()]
    log(f'{len(zips)} token zips on disk, {len(todo)} to extract')
    if not todo:
        return
    tasks = [(str(z), str(base), str(marker_dir), keep_zips) for z in todo]
    with get_context('fork').Pool(min(16, len(tasks))) as pool:
        pool.map(_extract_one, tasks, chunksize=1)
    log('extract done')


def _extract_one(args):
    z, base, marker_dir, keep = args
    # arcnames already start with neucodec/<locale>/<split>/, matching the parquet's
    # neucodec_path column, so everything extracts into the base dir as-is
    for attempt in range(3):
        r = subprocess.run(['unzip', '-q', '-o', z, '-d', base], capture_output=True, text=True)
        if r.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(f'unzip failed for {z} (rc={r.returncode}): {r.stderr.strip()[-500:]}')
        time.sleep(5)
    Path(marker_dir, f'{Path(z).name}.done').touch()
    if not keep:
        Path(z).unlink()


# ---------------------------------------------------------------- packing

# Globals inherited by fork()ed workers — set in pack() before the Pool starts.
G = {}


def make_block(docs):
    return {
        'input_ids': np.concatenate(docs).astype(np.uint32),
        'position_ids': np.concatenate([np.arange(len(d)) for d in docs]).astype(np.uint32),
        'attention_mask': np.array([len(d) for d in docs], dtype=np.uint32),
        'audio': '',
        'text': '',
    }


class BlockAccumulator:
    """Greedy 10,240-token packer around one ChiniDataset writer (one per task)."""

    def __init__(self, writer):
        self.writer = writer
        self.docs = []
        self.count = 0
        self.blocks = 0
        self.tokens = 0

    def add(self, ids):
        self.tokens += len(ids)
        if self.count + len(ids) > BLOCK_SIZE:
            if self.docs:
                self.writer.write(make_block(self.docs))
                self.blocks += 1
            self.docs = [ids]
            self.count = len(ids)
        else:
            self.docs.append(ids)
            self.count += len(ids)

    def flush(self):
        if self.docs:
            self.writer.write(make_block(self.docs))
            self.blocks += 1
            self.docs = []
            self.count = 0


def clean_text(*candidates):
    """First usable transcript; pandas nulls are floats, and NaN is truthy."""
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ''


def read_codes(path):
    """NeuCodec ids out of one .pt ({'source', 'codes': IntTensor[1, 1, T]})."""
    import torch

    obj = torch.load(path, map_location='cpu')
    codes = obj['codes'] if isinstance(obj, dict) else obj
    return codes.flatten().tolist()


def pack_worker(args):
    import contextlib

    import pandas as pd

    from chinidataset import ParquetWriter

    worker_id, files = args
    tokenizer = G['tokenizer']
    base = Path(G['base'])
    tasks = G['tasks']

    stats = {'docs': 0, 'missing': 0, 'empty_text': 0, 'ratio': 0,
             **{f'{t}_blocks': 0 for t in tasks}, **{f'{t}_tokens': 0 for t in tasks}}

    with contextlib.ExitStack() as stack:
        acc = {}
        for task in tasks:
            out_dir = base / 'out' / f'fleurs-{task}' / f'{worker_id:05d}'
            shutil.rmtree(out_dir, ignore_errors=True)
            writer = stack.enter_context(
                ParquetWriter(out=str(out_dir), columns=COLUMNS, compression=None, hashes=HASHES))
            acc[task] = BlockAccumulator(writer)

        for f in files:
            df = pd.read_parquet(f)
            normalized = df['normalized_text'] if 'normalized_text' in df.columns else df['sentence']
            speakers = df['speaker'] if 'speaker' in df.columns else df['locale']
            for neucodec_path, sentence, norm, locale, speaker in zip(
                    df['neucodec_path'], df['sentence'], normalized, df['locale'], speakers):
                text = clean_text(norm, sentence)
                if not text:
                    stats['empty_text'] += 1
                    continue
                try:
                    codes = read_codes(base / neucodec_path)
                except Exception:
                    stats['missing'] += 1
                    continue
                # more words than speech tokens means a broken alignment
                if len(text.split()) > len(codes):
                    stats['ratio'] += 1
                    continue

                s_tokens = ''.join([f'<|s_{c}|>' for c in codes])
                stats['docs'] += 1
                if 'tts' in acc:
                    voice = clean_text(speaker) or locale
                    prompt = f'<|im_start|>{voice}: {text}<|speech_start|>{s_tokens}<|im_end|>'
                    acc['tts'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                if 'stt' in acc:
                    prompt = f'<|im_start|><|STT|>{s_tokens}<|{locale}|>{text}<|im_end|>'
                    acc['stt'].add(tokenizer(prompt, add_special_tokens=False)['input_ids'])

            if worker_id == 0:
                log(f'worker 0: finished {Path(f).name}, {stats["docs"]} docs')

        for task, a in acc.items():
            a.flush()
            stats[f'{task}_blocks'] = a.blocks
            stats[f'{task}_tokens'] = a.tokens

    return stats


def build_tokenizer(locales):
    """Qwen3 + speech tokens (shared ids) + <|STT|>/locale tokens appended after them."""
    from transformers import AddedToken, AutoTokenizer

    log(f'building tokenizer (+65,537 speech tokens, <|STT|>, {len(locales)} locale tokens)')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    stt_tokens = ['<|STT|>'] + [f'<|{l}|>' for l in locales]
    tokenizer.add_tokens(extra + [AddedToken(t) for t in stt_tokens])
    return tokenizer, stt_tokens


def snake_chunks(files, workers):
    """Balance parquet files across workers by size (largest-first snake order)."""
    files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
    groups = [[] for _ in range(min(workers, len(files)))]
    for i, f in enumerate(files):
        cycle, pos = divmod(i, len(groups))
        idx = pos if cycle % 2 == 0 else len(groups) - 1 - pos
        groups[idx].append(str(f))
    return groups


def pack(base, files, tasks, workers):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    locales = sorted({f.name.rsplit('-', 1)[0] for f in files})
    tokenizer, stt_tokens = build_tokenizer(locales)

    out_root = base / 'out'
    for task in tasks:
        shutil.rmtree(out_root / f'fleurs-{task}', ignore_errors=True)
        (out_root / f'fleurs-{task}').mkdir(parents=True)
    with open(out_root / 'fleurs_stt_added_tokens.json', 'w') as f:
        json.dump(stt_tokens, f, indent=2)

    G.update(tokenizer=tokenizer, base=str(base), tasks=tasks)
    chunks = list(enumerate(snake_chunks(files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(chunks)) as pool:
        results = pool.map(pack_worker, chunks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    summary = {'locales': len(locales), 'files': len(files), **totals}
    for task in tasks:
        task_dir = out_root / f'fleurs-{task}'
        merge_index(task_dir)
        n = len(StreamingDataset(local=str(task_dir)))
        summary[f'{task}_blocks_indexed'] = n
        log(f'fleurs-{task}: {n} blocks (~{n * BLOCK_SIZE / 1e6:.0f}M packed tokens), '
            f'{totals[f"{task}_tokens"] / 1e6:.0f}M real tokens')
    log(f'packed in {time.time() - t0:.0f}s | ' + ' '.join(f'{k}={v}' for k, v in totals.items()))
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/share/multilingual-tts/fleurs')
    parser.add_argument('--locales', nargs='*', default=None, help='fnmatch patterns (default: all 102)')
    parser.add_argument('--splits', nargs='+', default=['train'], choices=['train', 'dev'])
    parser.add_argument('--task', choices=['tts', 'stt', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'pack', 'all'], default='all')
    parser.add_argument('--keep-zips', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == 'both' else (args.task,)

    if args.stage in ('download', 'all'):
        download(base, args.locales, args.splits, keep_zips=args.keep_zips)

    files = sorted(f for f in (base / 'data').glob('*.parquet')
                   if f.name.rsplit('-', 1)[1].removesuffix('.parquet') in args.splits
                   and match(f.name.rsplit('-', 1)[0], args.locales))
    if not files:
        raise SystemExit('no metadata parquets found — run --stage download first')
    log(f'{len(files)} metadata parquets, tasks: {", ".join(tasks)}')

    if args.stage in ('pack', 'all'):
        pack(base, files, tasks, args.workers)


if __name__ == '__main__':
    main()
