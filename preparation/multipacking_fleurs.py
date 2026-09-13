"""Pack FLEURS-R NeuCodec tokens into TTS and STT multipacking blocks.

Source: malaysia-ai/fleurs-r-neucodec-all-languages — 102 locales,
`data/{locale}-{split}.parquet` metadata + `neucodec/{locale}-{split}-rank{N}.zip`
token archives holding `neucodec/{locale}/{split}/{id}.pt`
(`{'source': ..., 'codes': IntTensor[1, 1, T]}`).

Two packs out of one pass over the tokens, same rows and same text on both sides so
the optimizer sweep compares tasks, not corpora:

    TTS  <|im_start|>{speaker}: {text}<|speech_start|>{<|s_N|> …}<|im_end|>
    STT  <|im_start|><|STT|>{<|s_N|> …}<|{locale}|>{text}<|im_end|>
    MEL  <|im_start|><|STT|><|mel_start|>{P × <|mel|>}<|mel_end|><|{locale}|>{text}<|im_end|>

The mel pack is the same STT task over the same rows with the audio carried as raw
whisper log-mel instead of NeuCodec tokens (mel_audio.py) — both run at 50 positions/s,
so the same utterance costs the same context either way and the codec's information loss
is the only thing that differs. Mel blocks store audio *paths*; run `--stage audio` first
to materialise the wavs, and give the trainer the same root as `--audio_dir`.

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
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# mel_audio lives at the repo root; only the mel task imports it, so the token arms
# still pack in a checkout that does not have it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# mel blocks store audio paths instead of speech tokens; the trainer reads the wavs and
# runs the STFT on GPU (mel_audio.py)
MEL_COLUMNS = dict(COLUMNS, audio='str', audio_samples='uint32[]')
TASKS = ('tts', 'stt', 'mel')


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


def download_audio(audio_base, locales, splits):
    """Materialise the wavs the mel pack points at: repo audio zip -> extract -> delete.

    The zips hold `audio/{locale}/{split}/{id}.wav`, exactly the `path` column, so they
    extract straight into the audio root and nothing has to be renamed. ~126GB for all
    102 locales, so it is marker-tracked per zip and safe to interrupt.
    """
    from huggingface_hub import HfApi

    marker_dir = audio_base / '.audio_extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f'{l}-{s}' for l in (locales or ['*']) for s in splits}
    names = sorted(
        f[len('audio/'):-len('.zip')]
        for f in HfApi().list_repo_files(REPO, repo_type='dataset')
        if f.startswith('audio/') and f.endswith('.zip')
    )
    names = [n for n in names if any(fnmatch.fnmatch(n, w) for w in wanted)]
    todo = [n for n in names if not (marker_dir / f'{n}.done').exists()]
    log(f'{len(names)} audio zips in scope, {len(todo)} to fetch+extract')
    if not todo:
        return
    tasks = [(n, str(audio_base), str(marker_dir)) for n in todo]
    with get_context('fork').Pool(min(12, len(tasks))) as pool:
        pool.map(_fetch_audio_one, tasks, chunksize=1)
    log('audio ready')


def _fetch_audio_one(args):
    name, audio_base, marker_dir = args
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            z = hf_hub_download(REPO, f'audio/{name}.zip', repo_type='dataset',
                                local_dir=os.path.join(audio_base, 'azips'))
            break
        except Exception as e:
            if attempt == 4:
                raise
            log(f'{name}: audio download failed ({e}); retrying')
            time.sleep(20)
    _extract_one((z, audio_base, marker_dir, True))


# ---------------------------------------------------------------- packing

# Globals inherited by fork()ed workers — set in pack() before the Pool starts.
G = {}


def make_block(docs, paths=None, samples=None):
    block = {
        'input_ids': np.concatenate(docs).astype(np.uint32),
        'position_ids': np.concatenate([np.arange(len(d)) for d in docs]).astype(np.uint32),
        'attention_mask': np.array([len(d) for d in docs], dtype=np.uint32),
        'audio': '',
        'text': '',
    }
    if paths is not None:
        block['audio'] = json.dumps(paths)
        block['audio_samples'] = np.array(samples, dtype=np.uint32)
    return block


class BlockAccumulator:
    """Greedy 10,240-token packer around one ChiniDataset writer (one per task).

    `audio` carries (path, samples) alongside each document for mel packs, so a block's
    utterance list stays in the order its `<|mel|>` placeholders appear.
    """

    def __init__(self, writer, audio=False):
        self.writer = writer
        self.audio = audio
        self.docs = []
        self.paths = []
        self.samples = []
        self.count = 0
        self.blocks = 0
        self.tokens = 0

    def _write(self):
        if self.audio:
            self.writer.write(make_block(self.docs, self.paths, self.samples))
        else:
            self.writer.write(make_block(self.docs))
        self.blocks += 1

    def add(self, ids, path=None, samples=None):
        self.tokens += len(ids)
        if self.count + len(ids) > BLOCK_SIZE:
            if self.docs:
                self._write()
            self.docs, self.paths, self.samples = [ids], [], []
            self.count = len(ids)
        else:
            self.docs.append(ids)
            self.count += len(ids)
        if self.audio:
            self.paths.append(path)
            self.samples.append(samples)

    def flush(self):
        if self.docs:
            self._write()
            self.docs, self.paths, self.samples = [], [], []
            self.count = 0


def clean_text(*candidates):
    """First usable transcript; pandas nulls are floats, and NaN is truthy."""
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ''


def probe_samples(path):
    """Length this wav will have at 16kHz mono, trimmed to whole stacked frames.

    Header only — `sf.info` reads the frame count without decoding, which is what keeps
    packing cheap now that the audio itself never enters the pack. FLEURS-R ships 24kHz,
    so this is where the 2/3 the loader will resample by gets applied. 0 means unreadable.
    """
    import soundfile as sf

    from mel_audio import SAMPLE_RATE, trim_to_position

    try:
        info = sf.info(str(path))
    except Exception:
        return 0
    return trim_to_position(int(info.frames * SAMPLE_RATE / info.samplerate))


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
    suffix = G['suffix']
    audio_base = Path(G['audio_base'])
    mel_id = G.get('mel_id')

    stats = {'docs': 0, 'missing': 0, 'empty_text': 0, 'ratio': 0, 'no_audio': 0,
             **{f'{t}_blocks': 0 for t in tasks}, **{f'{t}_tokens': 0 for t in tasks}}
    mel_prefix = mel_positions = None
    if 'mel' in tasks:
        from mel_audio import mel_positions

        mel_prefix = tokenizer('<|im_start|><|STT|><|mel_start|>',
                               add_special_tokens=False)['input_ids']

    with contextlib.ExitStack() as stack:
        acc = {}
        for task in tasks:
            out_dir = base / 'out' / f'fleurs-{task}{suffix}' / f'{worker_id:05d}'
            shutil.rmtree(out_dir, ignore_errors=True)
            columns = MEL_COLUMNS if task == 'mel' else COLUMNS
            writer = stack.enter_context(
                ParquetWriter(out=str(out_dir), columns=columns, compression=None, hashes=HASHES))
            acc[task] = BlockAccumulator(writer, audio=task == 'mel')

        for f in files:
            df = pd.read_parquet(f)
            normalized = df['normalized_text'] if 'normalized_text' in df.columns else df['sentence']
            speakers = df['speaker'] if 'speaker' in df.columns else df['locale']
            for neucodec_path, audio_path, sentence, norm, locale, speaker in zip(
                    df['neucodec_path'], df['path'], df['sentence'], normalized,
                    df['locale'], speakers):
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
                if 'mel' in acc:
                    # same row, same transcript — only the audio representation differs
                    n_samples = probe_samples(audio_base / audio_path)
                    if not n_samples:
                        stats['no_audio'] += 1
                    else:
                        positions = mel_positions(n_samples)
                        tail = tokenizer(f'<|mel_end|><|{locale}|>{text}<|im_end|>',
                                         add_special_tokens=False)['input_ids']
                        ids = mel_prefix + [mel_id] * positions + tail
                        acc['mel'].add(ids, path=str(audio_path), samples=n_samples)

            if worker_id == 0:
                log(f'worker 0: finished {Path(f).name}, {stats["docs"]} docs')

        for task, a in acc.items():
            a.flush()
            stats[f'{task}_blocks'] = a.blocks
            stats[f'{task}_tokens'] = a.tokens

    return stats


def build_tokenizer(locales, add_mel=False):
    """Qwen3 + speech tokens (shared ids) + <|STT|>/locale tokens appended after them.

    Mel tokens go last of all, so `<|s_N|>` and the language tokens keep the ids the
    token packs gave them and one tokenizer serves all three packs.
    """
    from transformers import AddedToken, AutoTokenizer

    from mel_audio import MEL_TOKENS

    log(f'building tokenizer (+65,537 speech tokens, <|STT|>, {len(locales)} locale tokens'
        + (f', {len(MEL_TOKENS)} mel tokens)' if add_mel else ')'))
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    stt_tokens = ['<|STT|>'] + [f'<|{l}|>' for l in locales]
    mel_tokens = list(MEL_TOKENS) if add_mel else []
    tokenizer.add_tokens(extra + [AddedToken(t) for t in stt_tokens + mel_tokens])
    return tokenizer, stt_tokens, mel_tokens


def snake_chunks(files, workers):
    """Balance parquet files across workers by size (largest-first snake order)."""
    files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
    groups = [[] for _ in range(min(workers, len(files)))]
    for i, f in enumerate(files):
        cycle, pos = divmod(i, len(groups))
        idx = pos if cycle % 2 == 0 else len(groups) - 1 - pos
        groups[idx].append(str(f))
    return groups


def pack(base, files, tasks, workers, suffix='', audio_base=None):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    locales = sorted({f.name.rsplit('-', 1)[0] for f in files})
    mel = 'mel' in tasks
    tokenizer, stt_tokens, mel_tokens = build_tokenizer(locales, add_mel=mel)

    out_root = base / 'out'
    for task in tasks:
        shutil.rmtree(out_root / f'fleurs-{task}{suffix}', ignore_errors=True)
        (out_root / f'fleurs-{task}{suffix}').mkdir(parents=True)
    # the locale list is the same for train and dev, so both splits share one token order
    with open(out_root / 'fleurs_stt_added_tokens.json', 'w') as f:
        json.dump(stt_tokens, f, indent=2)
    if mel:
        with open(out_root / 'fleurs_mel_added_tokens.json', 'w') as f:
            json.dump(stt_tokens + mel_tokens, f, indent=2)

    G.update(tokenizer=tokenizer, base=str(base), tasks=tasks, suffix=suffix,
             audio_base=str(audio_base or base),
             mel_id=tokenizer.convert_tokens_to_ids('<|mel|>') if mel else None)
    chunks = list(enumerate(snake_chunks(files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(chunks)) as pool:
        results = pool.map(pack_worker, chunks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    summary = {'locales': len(locales), 'files': len(files), 'splits': suffix or '-train', **totals}
    for task in tasks:
        task_dir = out_root / f'fleurs-{task}{suffix}'
        merge_index(task_dir)
        n = len(StreamingDataset(local=str(task_dir)))
        summary[f'{task}_blocks_indexed'] = n
        log(f'fleurs-{task}{suffix}: {n} blocks (~{n * BLOCK_SIZE / 1e6:.0f}M packed tokens), '
            f'{totals[f"{task}_tokens"] / 1e6:.0f}M real tokens')
    log(f'packed in {time.time() - t0:.0f}s | ' + ' '.join(f'{k}={v}' for k, v in totals.items()))
    with open(out_root / f'summary{suffix or "-train"}.json', 'w') as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/share/multilingual-tts/fleurs')
    parser.add_argument('--audio-base', default=None,
                        help='root the wavs live under (default: --base-dir). The mel pack stores '
                             'paths relative to it, and the trainer takes the same value as '
                             '--audio_dir')
    parser.add_argument('--locales', nargs='*', default=None, help='fnmatch patterns (default: all 102)')
    parser.add_argument('--splits', nargs='+', default=['train'], choices=['train', 'dev'])
    parser.add_argument('--task', choices=['tts', 'stt', 'mel', 'token', 'all'], default='token',
                        help="'token' = tts+stt (NeuCodec packs), 'mel' = the raw log-mel STT pack "
                             "(needs --stage audio first), 'all' = every arm of the ablation")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'audio', 'pack', 'all'], default='all',
                        help="'audio' materialises the wavs the mel pack points at")
    parser.add_argument('--keep-zips', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    tasks = {'all': TASKS, 'token': ('tts', 'stt')}.get(args.task, (args.task,))
    # dev packs sit beside the train ones rather than overwriting them
    suffix = '' if args.splits == ['train'] else '-' + '-'.join(sorted(args.splits))

    if args.stage in ('download', 'all'):
        download(base, args.locales, args.splits, keep_zips=args.keep_zips)
    audio_base = Path(args.audio_base) if args.audio_base else base
    if args.stage == 'audio' or (args.stage == 'all' and 'mel' in tasks):
        audio_base.mkdir(parents=True, exist_ok=True)
        download_audio(audio_base, args.locales, args.splits)

    files = sorted(f for f in (base / 'data').glob('*.parquet')
                   if f.name.rsplit('-', 1)[1].removesuffix('.parquet') in args.splits
                   and match(f.name.rsplit('-', 1)[0], args.locales))
    if not files:
        raise SystemExit('no metadata parquets found — run --stage download first')
    log(f'{len(files)} metadata parquets, tasks: {", ".join(tasks)}')

    if args.stage in ('pack', 'all'):
        pack(base, files, tasks, args.workers, suffix, audio_base)


if __name__ == '__main__':
    main()
