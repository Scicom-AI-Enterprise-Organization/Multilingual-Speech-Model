"""Multipack voice-conversion (text, speech-token) pairs into 10,240-token training blocks.

Replaces the per-dataset notebooks (multipacking-emilia-yodas.ipynb,
multipacking-malaysian-*.ipynb) with one script, and writes ChiniDataset
parquet shards (https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset)
instead of mosaicml-streaming MDS.

Per dataset:
  1. download the `*_neucodec.zip` files from the HF repo and extract them
     (NeuCodec token JSONs, one per audio segment),
  2. load the (reference, target) permutation pairs + the
     `audio_length_ratio_text` reject list,
  3. pack `<|im_start|>{text}<|speech_start|>{tokens}<|im_end|>` pairs into
     ~10,240-token blocks across N worker processes (one ParquetWriter
     sub-folder each), then merge_index() into a single dataset.

Samples inside a block stay attention-isolated: per-document position_ids
reset to 0 and `attention_mask` holds the per-document lengths that the
trainers expand into a block-diagonal mask.

Usage:
    python multipacking.py all --base-dir /share/multipacking --workers 96
    python multipacking.py malaysian-tamil-emilia youtube-cantonese-emilia
    python multipacking.py all --stage download   # only fetch/extract zips
    python multipacking.py all --stage pack       # assume zips are extracted
    python multipacking.py all --upload           # push results to HF (private)
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')  # Xet CAS 401s intermittently on multi-GB pulls
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BLOCK_SIZE = 1024 * 10
COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',
    'text': 'str',
}
HASHES = ['sha1', 'xxh64']


@dataclass
class Spec:
    repo: str
    config: str | None                      # permutation-pairs config (None = default)
    reject_config: str | None               # audio_length_ratio_text-style config (None = no filter)
    neucodec_patterns: tuple                # zip files to pull from the repo
    out_name: str
    upload_repo: str
    # 'trim_neucodec': folder/x.mp3 -> folder_trim_neucodec/x.json (Emilia repos)
    # 'neucodec':      folder/x.mp3 -> folder_neucodec/x.json      (YouTube-Cantonese)
    path_style: str = 'trim_neucodec'
    skip_prefix: str | None = None          # drop rows whose top folder contains this


DATASETS = {
    'malaysian-tamil-emilia': Spec(
        repo='Scicom-intl/Malaysian-Tamil-Emilia',
        config='permutation_sample',
        reject_config='audio_length_ratio_text',
        neucodec_patterns=('audio_processed_trim_neucodec.zip',),
        out_name='multipacking-malaysian-tamil-emilia',
        upload_repo='Scicom-intl/Malaysian-Tamil-Emilia-multipacking-10k',
    ),
    'malaysian-chinese-emilia': Spec(
        repo='Scicom-intl/Malaysian-Chinese-Emilia',
        config='speaker_permutation_sample',
        reject_config='audio_length_ratio_text',
        neucodec_patterns=('malaysian-chinese_processed_trim_neucodec.zip',),
        out_name='multipacking-malaysian-chinese-emilia',
        upload_repo='Scicom-intl/Malaysian-Chinese-Emilia-multipacking-10k',
    ),
    'malaysian-emilia-dialects': Spec(
        repo='Scicom-intl/Malaysian-Emilia',
        config='dialects_v1_permutation_sample',
        reject_config='dialects_v1_audio_length_ratio_text',
        neucodec_patterns=('dialects_processed_trim_neucodec.zip',),
        out_name='multipacking-malaysian-emilia-dialects',
        upload_repo='Scicom-intl/Malaysian-Emilia-dialects-multipacking-10k',
    ),
    'malaysian-emilia': Spec(
        repo='Scicom-intl/Malaysian-Emilia',
        config=None,
        reject_config='audio_length_ratio_text',
        neucodec_patterns=('*_processed_trim_neucodec.zip',),
        out_name='multipacking-malaysian-emilia',
        upload_repo='Scicom-intl/Malaysian-Emilia-multipacking-10k',
        skip_prefix='malaysian-chinese',
    ),
    'youtube-cantonese-emilia': Spec(
        repo='Scicom-intl/YouTube-Cantonese-Emilia',
        config='permutation_sample',
        reject_config=None,  # repo ships no audio_length_ratio_text config
        neucodec_patterns=('output-audio-trim-*_neucodec.zip',),
        out_name='multipacking-youtube-cantonese-emilia',
        upload_repo='Scicom-intl/YouTube-Cantonese-Emilia-multipacking-10k',
        path_style='neucodec',
    ),
    'emilia-yodas': Spec(
        repo='Scicom-intl/Emilia-YODAS-Voice-Conversion',
        config=None,
        reject_config='audio_length_ratio_text',
        neucodec_patterns=('Emilia-YODAS_trim_neucodec-*.zip',),
        out_name='multipacking-emilia-yodas',
        upload_repo='Scicom-intl/Emilia-YODAS-multipacking-10k',
    ),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- download

def download_and_extract(spec, base, delete_zips=True):
    import fnmatch

    from huggingface_hub import HfApi, hf_hub_download

    zips_dir = base / 'zips' / spec.repo.split('/')[-1]
    data_dir = base / 'neucodec'
    marker_dir = data_dir / '.extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)

    files = HfApi().list_repo_files(spec.repo, repo_type='dataset')
    wanted = sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in spec.neucodec_patterns))
    todo = [f for f in wanted if not (marker_dir / f'{f}.done').exists()]
    log(f'{spec.repo}: {len(wanted)} neucodec zips, {len(todo)} to download+extract')
    if not todo:
        return

    zips = []
    for f in todo:
        for attempt in range(5):
            try:
                zips.append(hf_hub_download(spec.repo, f, repo_type='dataset', local_dir=zips_dir))
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
    # parallel unzips into a shared tree race on mkdir of common sub-dirs
    # ("checkdir error", exit 2) — a retry lands after the dirs exist
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

# Globals inherited by fork()ed workers — set in pack_dataset() before the Pool starts.
G = {}


def neucodec_json_path(data_dir, audio_path, style):
    folder, _, rest = audio_path.partition('/')
    suffix = '_trim_neucodec' if style == 'trim_neucodec' else '_neucodec'
    rest = rest.rsplit('.', 1)[0] + '.json'
    return os.path.join(data_dir, folder + suffix, rest)


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


def pack_worker(args):
    from chinidataset import ParquetWriter

    start, end, worker_id = args
    rows = G['rows']
    tokenizer = G['tokenizer']
    reject = G['reject']
    data_dir = G['data_dir']
    style = G['path_style']
    skip_prefix = G['skip_prefix']

    out_dir = os.path.join(G['out_root'], f'{worker_id:05d}')
    shutil.rmtree(out_dir, ignore_errors=True)

    stats = {'blocks': 0, 'docs': 0, 'reject': 0, 'missing': 0, 'ratio': 0}
    docs = []
    count = 0
    with ParquetWriter(out=out_dir, columns=COLUMNS, compression=None, hashes=HASHES) as writer:
        for i in range(start, end):
            row = rows[i]

            if skip_prefix and skip_prefix in row['reference_audio'].partition('/')[0]:
                continue

            if row['reference_audio'] in reject or row['target_audio'] in reject:
                stats['reject'] += 1
                continue

            try:
                with open(neucodec_json_path(data_dir, row['reference_audio'], style)) as f:
                    left = json.load(f)
                with open(neucodec_json_path(data_dir, row['target_audio'], style)) as f:
                    right = json.load(f)
            except (OSError, json.JSONDecodeError):
                stats['missing'] += 1
                continue

            left_text = row['reference_text'].strip()
            right_text = row['target_text'].strip()

            # a transcript with more words than speech tokens is a bad alignment
            if len(left_text.split()) > len(left) or len(right_text.split()) > len(right):
                stats['ratio'] += 1
                continue

            left_token = ''.join([f'<|s_{t}|>' for t in left])
            right_token = ''.join([f'<|s_{t}|>' for t in right])
            prompt = (
                f'<|im_start|>{left_text}<|speech_start|>{left_token}<|im_end|>'
                f'<|im_start|>{right_text}<|speech_start|>{right_token}<|im_end|>'
            )

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

            if worker_id == 0 and stats['docs'] % 50000 == 0:
                log(f'worker 0: {i - start + 1}/{end - start} rows, {stats["blocks"]} blocks')

        if docs:
            writer.write(make_block(docs))
            stats['blocks'] += 1

    return stats


def build_tokenizer():
    from transformers import AddedToken, AutoTokenizer

    log('building tokenizer (Qwen/Qwen3-1.7B-Base + 65,537 speech tokens)')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base')
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    tokenizer.add_tokens(extra)
    return tokenizer


def build_reject(spec):
    from datasets import load_dataset

    if not spec.reject_config:
        return set()
    df = load_dataset(spec.repo, spec.reject_config)['train'].to_pandas()
    reject = set(df.loc[~df['audio_length_ratio_text_accept'], 'audio_filename'])
    log(f'{spec.repo}/{spec.reject_config}: {len(reject)} rejected files')
    return reject


def pack_dataset(name, spec, base, tokenizer, workers):
    from datasets import load_dataset

    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    out_root = base / 'out' / spec.out_name
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True)

    reject = build_reject(spec)
    ds = load_dataset(spec.repo, spec.config) if spec.config else load_dataset(spec.repo)
    rows = ds['train'].to_list()
    del ds
    log(f'{name}: {len(rows)} pair rows')

    G.update(
        rows=rows,
        tokenizer=tokenizer,
        reject=reject,
        data_dir=str(base / 'neucodec'),
        path_style=spec.path_style,
        skip_prefix=spec.skip_prefix,
        out_root=str(out_root),
    )

    bounds = np.linspace(0, len(rows), workers + 1, dtype=int)
    tasks = [(int(bounds[i]), int(bounds[i + 1]), i) for i in range(workers) if bounds[i] < bounds[i + 1]]

    t0 = time.time()
    with get_context('fork').Pool(len(tasks)) as pool:
        results = pool.map(pack_worker, tasks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    merge_index(out_root)

    packed = StreamingDataset(local=str(out_root))
    n = len(packed)
    log(
        f'{name}: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens) in {time.time() - t0:.0f}s | '
        f'docs={totals["docs"]} rejected={totals["reject"]} missing={totals["missing"]} ratio={totals["ratio"]}'
    )

    summary = {'name': name, 'blocks': n, **totals}
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def upload(spec, base):
    from huggingface_hub import HfApi

    out_root = base / 'out' / spec.out_name
    api = HfApi()
    api.create_repo(spec.upload_repo, repo_type='dataset', private=True, exist_ok=True)
    log(f'uploading {out_root} -> {spec.upload_repo}')
    # modest worker count — the default (num CPUs) blows the org's 3000 req/5min API quota
    api.upload_large_folder(repo_id=spec.upload_repo, repo_type='dataset', folder_path=str(out_root), num_workers=16)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='+', help=f'"all" or any of: {", ".join(DATASETS)}')
    parser.add_argument('--base-dir', default='/share/multipacking')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'pack', 'upload', 'all'], default='all',
                        help='"upload" pushes existing out/ dirs without re-packing')
    parser.add_argument('--keep-zips', action='store_true', help='keep neucodec zips after extraction')
    parser.add_argument('--upload', action='store_true', help='upload packed datasets to HF (private)')
    args = parser.parse_args()

    names = list(DATASETS) if 'all' in args.datasets else args.datasets
    for n in names:
        if n not in DATASETS:
            parser.error(f'unknown dataset {n!r}; choose from: {", ".join(DATASETS)}')

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.stage in ('download', 'all'):
        for n in names:
            download_and_extract(DATASETS[n], base, delete_zips=not args.keep_zips)

    if args.stage in ('pack', 'all'):
        tokenizer = build_tokenizer()
        summaries = []
        for n in names:
            summaries.append(pack_dataset(n, DATASETS[n], base, tokenizer, args.workers))
            if args.upload:
                upload(DATASETS[n], base)
        log('=== done ===')
        for s in summaries:
            log(json.dumps(s))

    if args.stage == 'upload':
        for n in names:
            upload(DATASETS[n], base)
        log('=== upload done ===')


if __name__ == '__main__':
    main()
