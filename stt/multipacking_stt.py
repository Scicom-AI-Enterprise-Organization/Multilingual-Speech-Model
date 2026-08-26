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
    python multipacking_stt.py --base-dir /share/stt --workers 96          # everything, one wave
    python multipacking_stt.py --base-dir /share/stt --subsets 'malaysian-*' 'emilia_zh'
    python multipacking_stt.py --base-dir /share/stt --stage download      # zips + metadata only

    # disk-bounded full run: ~177GB of token JSONs + ~270GB packed output do not
    # fit at once, so process in waves (upload + free between waves):
    python multipacking_stt.py --num-waves 2 --wave 0 --workers 96
    python multipacking_stt.py --stage upload          # push wave 0, then free neucodec/ + wave-0 shards
    python multipacking_stt.py --num-waves 2 --wave 1 --workers 96
    python multipacking_stt.py --stage merge           # root index.json across waves
    python multipacking_stt.py --stage upload          # -> Scicom-intl/Multilingual-STT-multipacking-10k

The language-token list is always computed over ALL selected metadata (cached in
languages.json), so token IDs are identical across waves.
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

def audio_folder(parquet_file):
    """Top-level audio folder a subset's rows point into (from the first row).

    Returns None for subsets without an `audio_filename` column (e.g.
    Malaysian-TTS-v2 ships `token_filename` and has no zip in the repo) and for
    empty parquets — their rows are skipped at pack time.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_file)
    if 'audio_filename' not in pf.schema_arrow.names:
        return None
    for batch in pf.iter_batches(batch_size=64, columns=['audio_filename']):
        if batch.num_rows:
            return str(batch.column(0)[0]).partition('/')[0]
    return None


def download_meta(base, subsets):
    from huggingface_hub import snapshot_download

    meta_dir = base / 'meta'
    allow = [f'{p}/*.parquet' for p in subsets] if subsets else ['*/*.parquet']
    snapshot_download(META_REPO, repo_type='dataset', allow_patterns=allow, local_dir=meta_dir)
    return meta_dir


def download_zips(base, meta_files, delete_zips=True):
    """Fetch the `<audio folder>_neucodec.zip` for every given metadata parquet.

    Zip names come from the audio folder inside `audio_filename`, which is NOT
    always the subset directory name (e.g. subset `emilia_zh` ->
    `emilia_zh_audio_neucodec.zip`).
    """
    from huggingface_hub import HfApi, hf_hub_download

    data_dir = base / 'neucodec'
    marker_dir = data_dir / '.extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = base / 'zips'

    folders = {audio_folder(f) for f in meta_files}
    folders.discard(None)
    available = set(HfApi().list_repo_files(TOKENS_REPO, repo_type='dataset'))
    wanted = sorted(f'{fo}_neucodec.zip' for fo in folders if f'{fo}_neucodec.zip' in available)
    no_zip = sorted(fo for fo in folders if f'{fo}_neucodec.zip' not in available)
    todo = [f for f in wanted if not (marker_dir / f'{f}.done').exists()]
    log(f'{TOKENS_REPO}: {len(wanted)} neucodec zips wanted, {len(todo)} to download+extract, '
        f'{len(no_zip)} audio folders have no zip yet (rows will be skipped)')
    if not todo:
        return

    # download+extract per zip in parallel — hundreds of zips sequentially is the
    # bottleneck, and overlapping the two stages keeps both the pipe and disk busy
    tasks = [(f, str(zips_dir), str(data_dir), str(marker_dir), delete_zips) for f in todo]
    with get_context('fork').Pool(min(8, len(tasks))) as pool:
        pool.map(_fetch_extract_one, tasks, chunksize=1)


def _fetch_extract_one(args):
    f, zips_dir, data_dir, marker_dir, delete = args
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            z = hf_hub_download(TOKENS_REPO, f, repo_type='dataset', local_dir=zips_dir)
            break
        except Exception as e:
            if attempt == 4:
                raise
            log(f'download of {f} failed ({e}); retrying')
            time.sleep(30)
    _extract_one((z, data_dir, marker_dir, delete))


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
            if 'audio_filename' not in df.columns:
                stats['missing'] += len(df)
                continue
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


def pack(base, all_files, wave_files, wave_name, workers):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    # language tokens come from ALL selected metadata, not just this wave, so the
    # tokenizer (and therefore token IDs) is identical across waves
    languages = collect_languages(base, all_files)
    log(f'{len(languages)} languages')
    tokenizer, stt_tokens = build_tokenizer(languages)

    out_root = base / 'out' / OUT_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    wave_dir = out_root / wave_name
    shutil.rmtree(wave_dir, ignore_errors=True)
    wave_dir.mkdir(parents=True)

    G.update(
        tokenizer=tokenizer,
        data_dir=str(base / 'neucodec'),
        languages=set(languages),
        out_root=str(wave_dir),
    )

    tasks = list(enumerate(snake_chunks(wave_files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(tasks)) as pool:
        results = pool.map(pack_worker, tasks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    merge_index(wave_dir)

    packed = StreamingDataset(local=str(wave_dir))
    n = len(packed)
    log(
        f'{wave_name}: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens) in {time.time() - t0:.0f}s | '
        + ' '.join(f'{k}={v}' for k, v in totals.items())
    )

    with open(out_root / 'stt_added_tokens.json', 'w') as f:
        json.dump(stt_tokens, f, indent=2)
    with open(wave_dir / f'summary-{wave_name}.json', 'w') as f:
        json.dump({'wave': wave_name, 'blocks': n, 'languages': len(languages), **totals}, f, indent=2)


def merge_waves(base):
    """Merge every wave's index into one root index.json (call after all waves)."""
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    out_root = base / 'out' / OUT_NAME
    merge_index(out_root)
    packed = StreamingDataset(local=str(out_root))
    n = len(packed)
    log(f'merged: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens)')
    with open(out_root / 'summary.json', 'w') as f:
        json.dump({'blocks': n}, f, indent=2)


def upload(base):
    from huggingface_hub import HfApi

    out_root = base / 'out' / OUT_NAME
    api = HfApi()
    api.create_repo(UPLOAD_REPO, repo_type='dataset', private=True, exist_ok=True)
    log(f'uploading {out_root} -> {UPLOAD_REPO}')
    # 6 workers: 16 already tripped the org's 3000 req/5min quota into 429 backoff storms
    api.upload_large_folder(repo_id=UPLOAD_REPO, repo_type='dataset', folder_path=str(out_root), num_workers=6)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/share/stt')
    parser.add_argument('--subsets', nargs='*', default=None,
                        help='fnmatch patterns of subset names (default: all)')
    parser.add_argument('--num-waves', type=int, default=1,
                        help='split subsets into N disk-bounded waves (JSONs+output of one wave on disk at a time)')
    parser.add_argument('--wave', type=int, default=0, help='which wave to process (0-based)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'pack', 'merge', 'upload', 'all'], default='all')
    parser.add_argument('--keep-zips', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.stage == 'merge':
        merge_waves(base)
        return
    if args.stage == 'upload':
        upload(base)
        return

    if args.stage in ('download', 'all'):
        download_meta(base, args.subsets)

    all_files = sorted(
        (f for f in (base / 'meta').glob('*/*.parquet') if match_subset(f.parent.name, args.subsets)),
        key=lambda f: (f.parent.name, f.name),
    )
    if not all_files:
        raise SystemExit('no metadata parquets found — run --stage download first')

    subsets_sorted = sorted({f.parent.name for f in all_files})
    step = -(-len(subsets_sorted) // args.num_waves)
    wave_subsets = set(subsets_sorted[args.wave * step:(args.wave + 1) * step])
    wave_files = [f for f in all_files if f.parent.name in wave_subsets]
    wave_name = f'wave-{args.wave}' if args.num_waves > 1 else 'data'
    log(f'{len(all_files)} metadata parquets total; {wave_name}: '
        f'{len(wave_subsets)} subsets / {len(wave_files)} files')

    if args.stage in ('download', 'all'):
        download_zips(base, wave_files, delete_zips=not args.keep_zips)
    if args.stage in ('pack', 'all'):
        pack(base, all_files, wave_files, wave_name, args.workers)


if __name__ == '__main__':
    main()
