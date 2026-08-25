"""Sync malaysia-ai/Multilingual-TTS-language with malaysia-ai/Multilingual-TTS.

For every source parquet missing in the -language repo:
  1. download it,
  2. add `language`   — GlotLID v3 label via langdetect_glotlid (und when no letters),
  3. add `post-normalized` — postnormalizer.normalize(text), auto script detection,
  4. upload to the same path in -language.

Then regenerate the README configs YAML (one config per subset directory) from the
final file listing, keeping everything else in the card untouched.

Resumable: files already present in -language are skipped on re-run.

Usage:
    python sync_language.py            # sync everything missing
    python sync_language.py --dry-run  # just report what would sync
"""

import argparse
import io
import json
import time
from multiprocessing import get_context
from pathlib import Path

SRC_REPO = 'malaysia-ai/Multilingual-TTS'
LANG_REPO = 'malaysia-ai/Multilingual-TTS-language'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parquet_files(api, repo):
    return {f for f in api.list_repo_files(repo, repo_type='dataset') if f.endswith('.parquet')}


def _normalize_chunk(texts):
    from postnormalizer import normalize
    return [normalize(t) for t in texts]


def annotate(df, lid, pool):
    texts = df['text'].tolist()
    preds = []
    for i in range(0, len(texts), 8192):
        preds.extend(lid.predict_batch(texts[i:i + 8192]))
    df['language'] = [p.label for p in preds]

    n = max(1, len(texts) // (pool._processes * 4))
    chunks = [texts[i:i + n] for i in range(0, len(texts), n)]
    normalized = []
    for part in pool.map(_normalize_chunk, chunks):
        normalized.extend(part)
    df['post-normalized'] = normalized
    return df


def regenerate_readme(api):
    from huggingface_hub import hf_hub_download

    subsets = sorted({f.split('/', 1)[0] for f in parquet_files(api, LANG_REPO)})
    readme = Path(hf_hub_download(LANG_REPO, 'README.md', repo_type='dataset')).read_text()

    head, _, rest = readme.partition('\n---\n')          # front matter / body
    pre_configs, _, _ = head.partition('\nconfigs:')     # keys above configs: stay

    lines = ['configs:']
    for s in subsets:
        lines.append(f'- config_name: {s}')
        lines.append('  data_files:')
        lines.append('  - split: train')
        lines.append(f'    path: {s}/train-*')
    new_readme = pre_configs + '\n' + '\n'.join(lines) + '\n---\n' + rest

    api.upload_file(
        path_or_fileobj=io.BytesIO(new_readme.encode()),
        path_in_repo='README.md',
        repo_id=LANG_REPO,
        repo_type='dataset',
        commit_message=f'Regenerate configs for {len(subsets)} subsets',
    )
    log(f'README regenerated with {len(subsets)} configs')


def main():
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    from langdetect_glotlid import LanguageDetector

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work-dir', default='sync_work')
    parser.add_argument('--procs', type=int, default=32, help='postnormalizer processes')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    api = HfApi()
    missing = sorted(parquet_files(api, SRC_REPO) - parquet_files(api, LANG_REPO))
    log(f'{len(missing)} parquet files to sync {SRC_REPO} -> {LANG_REPO}')
    if args.dry_run or not missing:
        for f in missing:
            print(' ', f)
        if not args.dry_run and not missing:
            regenerate_readme(api)
        return

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    lid = LanguageDetector()
    pool = get_context('fork').Pool(args.procs)

    for i, f in enumerate(missing):
        log(f'[{i + 1}/{len(missing)}] {f}')
        local = hf_hub_download(SRC_REPO, f, repo_type='dataset')
        df = pd.read_parquet(local)
        df = annotate(df, lid, pool)
        counts = df['language'].value_counts()
        log(f'  {len(df)} rows, dominant language: '
            + ', '.join(f'{l} {c}' for l, c in counts.head(3).items()))

        out = work / f.replace('/', '__')
        df.to_parquet(out, index=False)
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(out), path_in_repo=f,
                                repo_id=LANG_REPO, repo_type='dataset',
                                commit_message=f'Add {f} with language + post-normalized')
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log(f'  upload failed ({e}); retrying')
                time.sleep(30)
        out.unlink()

    pool.close()
    pool.join()
    regenerate_readme(api)

    still = parquet_files(api, SRC_REPO) - parquet_files(api, LANG_REPO)
    log(f'done — {len(still)} files still missing' + (f': {sorted(still)[:5]}' if still else ''))


if __name__ == '__main__':
    main()
