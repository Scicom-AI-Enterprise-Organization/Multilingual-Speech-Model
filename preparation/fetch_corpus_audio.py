"""Fetch a disk-budgeted slice of the corpus audio for the raw-mel pack.

`malaysia-ai/Multilingual-TTS` carries 4.438 TB of audio across 2,100 zips. The mel pack
stores audio *paths*, so whatever is fetched stays resident for the whole training run —
which makes this a budget decision, not a throughput one.

Selection is round-robin across subsets rather than largest-first: one shard from every
subset, then a second from every subset that has one, and so on until the budget is
spent. A budget that is 45% of the corpus therefore buys ~45% of *every* language rather
than all of the handful of biggest corpora.

    python fetch_corpus_audio.py --out /root/share/corpus --budget-tb 2.0
    python fetch_corpus_audio.py --out /root/share/corpus --budget-tb 2.0 --plan-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_retry import file_with_retry, log

REPO = 'malaysia-ai/Multilingual-TTS'
G = {}


def audio_zips():
    """Every audio zip with its size, grouped by the subset folder it belongs to."""
    from huggingface_hub import HfApi

    by_subset = defaultdict(list)
    for f in HfApi().list_repo_tree(REPO, repo_type='dataset', recursive=True):
        size = getattr(f, 'size', None)
        path = f.path
        if size is None or not path.endswith('.zip') or 'neucodec' in path or '/' in path:
            continue
        # `{folder}_audio.zip` or `{folder}_audio-<i>-<j>.zip`
        stem = path[:-4]
        if '_audio' not in stem:
            continue
        folder = stem.split('_audio')[0]
        by_subset[folder].append((path, size))
    for folder in by_subset:
        by_subset[folder].sort()
    return by_subset


def select(by_subset, budget_bytes):
    """Round-robin one shard per subset per pass, until the budget is spent."""
    chosen, total, depth = [], 0, 0
    while True:
        added = False
        for folder in sorted(by_subset):
            shards = by_subset[folder]
            if depth >= len(shards):
                continue
            path, size = shards[depth]
            if total + size > budget_bytes:
                continue
            chosen.append((path, size))
            total += size
            added = True
        if not added:
            break
        depth += 1
    return chosen, total


def _fetch_one(args):
    """Fetch+extract one zip. A zip that cannot be had is reported, not raised:
    aborting 1,171 good downloads over one bad file is worse than a short coverage gap."""
    name, out, marker_dir = args
    if (Path(marker_dir) / f'{name}.done').exists():
        return 0
    try:
        path = file_with_retry(REPO, name, local_dir=os.path.join(out, 'zips'))
        r = subprocess.run(['unzip', '-q', '-o', path, '-d', out], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'unzip rc={r.returncode}: {r.stderr[-200:]}')
        os.remove(path)
    except Exception as e:
        log(f'{name}: SKIPPED after retries — {type(e).__name__}: {str(e)[:120]}')
        Path(marker_dir, f'{name}.failed').touch()
        return 0
    Path(marker_dir, f'{name}.done').touch()
    log(f'{name}: extracted')
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='/root/share/corpus')
    p.add_argument('--budget-tb', type=float, default=2.0)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()

    out = Path(args.out)
    marker_dir = out / '.audio_extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)

    log('listing corpus audio zips')
    by_subset = audio_zips()
    corpus_bytes = sum(s for shards in by_subset.values() for _, s in shards)
    chosen, total = select(by_subset, args.budget_tb * 1e12)
    subsets = {c[0].split('_audio')[0] for c in chosen}
    log(f'corpus: {sum(len(v) for v in by_subset.values())} zips / {corpus_bytes/1e12:.3f} TB '
        f'over {len(by_subset)} subsets')
    log(f'budget {args.budget_tb} TB -> {len(chosen)} zips / {total/1e12:.3f} TB '
        f'({total/corpus_bytes:.1%} of the corpus) over {len(subsets)} subsets')

    with open(out / 'audio_selection.json', 'w') as f:
        json.dump({'budget_tb': args.budget_tb, 'selected_bytes': total,
                   'corpus_bytes': corpus_bytes, 'subsets': sorted(subsets),
                   'zips': [c[0] for c in chosen]}, f, indent=2)
    if args.plan_only:
        return

    todo = [c[0] for c in chosen
            if not (marker_dir / f'{c[0]}.done').exists()
            and not (marker_dir / f'{c[0]}.failed').exists()]
    log(f'{len(todo)} zips to fetch')
    tasks = [(n, str(out), str(marker_dir)) for n in todo]
    with get_context('fork').Pool(min(args.workers, max(1, len(tasks)))) as pool:
        done = sum(pool.map(_fetch_one, tasks, chunksize=1))
    failed = sorted(p.name[:-len('.failed')] for p in marker_dir.glob('*.failed'))
    have = len(list(marker_dir.glob('*.done')))
    log(f'CORPUS_AUDIO_READY {have}/{len(chosen)} zips on disk, {done} this pass, '
        f'{len(failed)} unrecoverable')
    if failed:
        log('  skipped: ' + ', '.join(failed[:5]) + ('…' if len(failed) > 5 else ''))


if __name__ == '__main__':
    main()
