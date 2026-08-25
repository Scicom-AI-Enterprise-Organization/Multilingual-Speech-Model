"""Multipack STT documents (speech tokens -> transcription) into 10,240-token blocks.

Inverse of the TTS task, built from malaysia-ai/Multilingual-TTS-language. One
document per segment:

    <|im_start|><|STT|>{<|s_N|> speech tokens}<|{language}|>{normalized transcription}<|im_end|>

- speech tokens come from the `<subset>_neucodec.zip` files in
  malaysia-ai/Multilingual-TTS (`<subset>/<file>.mp3` -> `<subset>_neucodec/<file>.json`),
  the same JSONs the TTS multipacking uses,
- `language` is the GlotLID v3 label column (see langdetect_glotlid.py) turned into
  a `<|zsm_Latn|>`-style token,
- the transcription is the `post-normalized` column (postnormalizer.py output);
  when a subset predates that column it is normalized on the fly.

Tokenizer: Qwen3-1.7B-Base + <|speech_start|> + 65,536 <|s_N|> first (same order as
the TTS trainers, so speech-token IDs line up), then <|STT|> and the sorted language
tokens. The exact appended list is written to <out>/stt_added_tokens.json — trainers
must add the same tokens in the same order.

Output format matches preparation/multipacking.py: ChiniDataset parquet blocks with
attention-isolated documents (per-doc position_ids reset, attention_mask = per-doc
lengths).

Usage:
    python multipacking_stt.py --base-dir /share/stt --workers 96
    python multipacking_stt.py --base-dir /share/stt --subsets 'malaysian-*' 'emilia_zh'
    python multipacking_stt.py --base-dir /share/stt --stage download
    python multipacking_stt.py --base-dir /share/stt --stage upload

The full corpus is 118M rows / 1493 subsets — check disk before an unfiltered run
and use --subsets to scope.
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

META_REPO = 'malaysia-ai/Multilingual-TTS-language'
TOKENS_REPO = 'malaysia-ai/Multilingual-TTS'
UPLOAD_REPO = 'Scicom-intl/Multilingual-STT-multipacking-10k'
OUT_NAME = 'multipacking-stt'

BLOCK_SIZE = 1024 * 10
COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',
    'text': 'str',
}
HASHES = ['sha1', 'xxh64']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def match_subset(subset, patterns):
    return not patterns or any(fnmatch.fnmatch(subset, p) for p in patterns)


# ---------------------------------------------------------------- download

def download(base, subsets, delete_zips=True):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    meta_dir = base / 'meta'
    allow = [f'{p}/*.parquet' for p in subsets] if subsets else ['*/*.parquet']
    snapshot_download(META_REPO, repo_type='dataset', allow_patterns=allow, local_dir=meta_dir)

    data_dir = base / 'neucodec'
    marker_dir = data_dir / '.extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = base / 'zips'

    files = HfApi().list_repo_files(TOKENS_REPO, repo_type='dataset')
    wanted = sorted(
        f for f in files
        if f.endswith('_neucodec.zip') and match_subset(f[: -len('_neucodec.zip')], subsets)
    )
    todo = [f for f in wanted if not (marker_dir / f'{f}.done').exists()]
    log(f'{TOKENS_REPO}: {len(wanted)} neucodec zips, {len(todo)} to download+extract')
    if not todo:
        return

    zips = []
    for f in todo:
        for attempt in range(5):
            try:
                zips.append(hf_hub_download(TOKENS_REPO, f, repo_type='dataset', local_dir=zips_dir))
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log(f'download of {f} failed ({e}); retrying')
                time.sleep(30)

    with get_context('fork').Pool(min(8, len(zips))) as pool:
        pool.map(_extract_one, [(z, str(data_dir), str(marker_dir), delete_zips) for z in zips])


def _extract_one(args):
    z, data_dir, marker_dir, delete = args
    for attempt in range(3):
        r = subprocess.run(['unzip', '-q', '-o', z, '-d', data_dir], capture_output=True, text=True)
        if r.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(f'unzip failed for {z} (rc={r.returncode}): {r.stderr.strip()[-500:]}')
        log(f'unzip {Path(z).name} rc={r.returncode}, retrying: {r.stderr.strip()[-200:]}')
        time.sleep(5)
    Path(marker_dir, f'{Path(z).name}.done').touch()
    if delete:
        Path(z).unlink()
    log(f'extracted {Path(z).name}')


# ---------------------------------------------------------------- packing

# Globals inherited by fork()ed workers — set in pack() before the Pool starts.
G = {}


def token_json_path(data_dir, audio_filename):
    folder, _, rest = audio_filename.partition('/')
    if not rest:
        return None
    rest = rest.rsplit('.', 1)[0] + '.json'
    return os.path.join(data_dir, folder + '_neucodec', rest)


def make_block(docs):
    input_ids = np.concatenate(docs).astype(np.uint32)
    position_ids = np.concatenate([np.arange(len(d)) for d in docs]).astype(np.uint32)
    attention_mask = np.array([len(d) for d in docs], dtype=np.uint32)
    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'audio': '',
        'text': '',
    }


def _languages_one(f):
    import pandas as pd
    df = pd.read_parquet(f, columns=['language'])
    return set(df['language'].dropna().unique())


def collect_languages(base, files):
    """Distinct GlotLID labels across the metadata — defines the language tokens."""
    cache = base / 'languages.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    with get_context('fork').Pool(min(32, len(files))) as pool:
        sets = pool.map(_languages_one, files)
    languages = sorted(set().union(*sets) - {'und', ''})
    with open(cache, 'w') as f:
        json.dump(languages, f, indent=2)
    return languages


def build_tokenizer(languages):
    from transformers import AddedToken, AutoTokenizer

    log(f'building tokenizer (+65,537 speech tokens, <|STT|>, {len(languages)} language tokens)')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    stt_tokens = ['<|STT|>'] + [f'<|{l}|>' for l in languages]
    tokenizer.add_tokens(extra + [AddedToken(t) for t in stt_tokens])
    return tokenizer, stt_tokens


def pack_worker(args):
    import pandas as pd

    from chinidataset import ParquetWriter
    from postnormalizer import normalize

    worker_id, files = args
    tokenizer = G['tokenizer']
    data_dir = G['data_dir']
    languages = G['languages']

    out_dir = os.path.join(G['out_root'], f'{worker_id:05d}')
    shutil.rmtree(out_dir, ignore_errors=True)

    stats = {'blocks': 0, 'docs': 0, 'no_lang': 0, 'empty_text': 0, 'missing': 0, 'ratio': 0}
    docs = []
    count = 0
    with ParquetWriter(out=out_dir, columns=COLUMNS, compression=None, hashes=HASHES) as writer:
        for f in files:
            df = pd.read_parquet(f)
            has_norm = 'post-normalized' in df.columns
            audio_col = df['audio_filename'].tolist()
            lang_col = df['language'].tolist() if 'language' in df.columns else [None] * len(df)
            text_col = df['post-normalized'].tolist() if has_norm else df['text'].tolist()
            for audio_filename, language, text in zip(audio_col, lang_col, text_col):
                if not language or language == 'und' or language not in languages:
                    stats['no_lang'] += 1
                    continue

                if not has_norm:
                    text = normalize(text)
                if not text or not str(text).strip():
                    stats['empty_text'] += 1
                    continue
                text = str(text).strip()

                path = token_json_path(data_dir, audio_filename)
                try:
                    with open(path) as fopen:
                        token = json.load(fopen)
                except (TypeError, OSError, json.JSONDecodeError):
                    stats['missing'] += 1
                    continue

                # a transcript with more words than speech tokens is a bad alignment
                if len(text.split()) > len(token):
                    stats['ratio'] += 1
                    continue

                s_tokens = ''.join([f'<|s_{t}|>' for t in token])
                prompt = f'<|im_start|><|STT|>{s_tokens}<|{language}|>{text}<|im_end|>'

                ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
                stats['docs'] += 1

                if count + len(ids) > BLOCK_SIZE:
                    if docs:
                        writer.write(make_block(docs))
                        stats['blocks'] += 1
                    docs = [ids]
                    count = len(ids)
                else:
                    docs.append(ids)
                    count += len(ids)

            if worker_id == 0:
                log(f'worker 0: finished {Path(f).parent.name}/{Path(f).name}, {stats["blocks"]} blocks')

        if docs:
            writer.write(make_block(docs))
            stats['blocks'] += 1

    return stats


def snake_chunks(files, workers):
    """Balance parquet files across workers by size (largest-first snake order)."""
    files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
    groups = [[] for _ in range(min(workers, len(files)))]
    for i, f in enumerate(files):
        cycle, pos = divmod(i, len(groups))
        idx = pos if cycle % 2 == 0 else len(groups) - 1 - pos
        groups[idx].append(str(f))
    return groups


def pack(base, subsets, workers):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    files = [f for f in (base / 'meta').glob('*/*.parquet') if match_subset(f.parent.name, subsets)]
    if not files:
        raise SystemExit('no metadata parquets found — run --stage download first')
    log(f'{len(files)} metadata parquet files')

    languages = collect_languages(base, files)
    log(f'{len(languages)} languages')
    tokenizer, stt_tokens = build_tokenizer(languages)

    out_root = base / 'out' / OUT_NAME
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True)

    G.update(
        tokenizer=tokenizer,
        data_dir=str(base / 'neucodec'),
        languages=set(languages),
        out_root=str(out_root),
    )

    tasks = list(enumerate(snake_chunks(files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(tasks)) as pool:
        results = pool.map(pack_worker, tasks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    merge_index(out_root)

    packed = StreamingDataset(local=str(out_root))
    n = len(packed)
    log(
        f'stt: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens) in {time.time() - t0:.0f}s | '
        + ' '.join(f'{k}={v}' for k, v in totals.items())
    )

    with open(out_root / 'stt_added_tokens.json', 'w') as f:
        json.dump(stt_tokens, f, indent=2)
    with open(out_root / 'summary.json', 'w') as f:
        json.dump({'blocks': n, 'languages': len(languages), **totals}, f, indent=2)


def upload(base):
    from huggingface_hub import HfApi

    out_root = base / 'out' / OUT_NAME
    api = HfApi()
    api.create_repo(UPLOAD_REPO, repo_type='dataset', private=True, exist_ok=True)
    log(f'uploading {out_root} -> {UPLOAD_REPO}')
    api.upload_large_folder(repo_id=UPLOAD_REPO, repo_type='dataset', folder_path=str(out_root), num_workers=16)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/share/stt')
    parser.add_argument('--subsets', nargs='*', default=None,
                        help='fnmatch patterns of subset names (default: all 1493)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'pack', 'upload', 'all'], default='all')
    parser.add_argument('--keep-zips', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.stage in ('download', 'all'):
        download(base, args.subsets, delete_zips=not args.keep_zips)
    if args.stage in ('pack', 'all'):
        pack(base, args.subsets, args.workers)
    if args.stage == 'upload':
        upload(base)


if __name__ == '__main__':
    main()
