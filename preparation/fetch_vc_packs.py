"""Fetch the six packed voice-conversion corpora — 3,646,600 blocks, ~37.34B tokens.

These are already ChiniDataset parquet, so there is nothing to pack: download and point
`--train_file` at them. They hold speech tokens and text only, so the language tags the
other packs append never move their ids.

    python fetch_vc_packs.py --out /root/share/packs
    python fetch_vc_packs.py --out /root/share/packs --packs Malaysian-Emilia
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_retry import log, snapshot_with_retry

PACKS = [
    ('Emilia-YODAS', 1_902_702, '19.48B'),
    ('Malaysian-Emilia', 775_589, '7.94B'),
    ('Malaysian-Emilia-dialects', 565_604, '5.79B'),
    ('Malaysian-Chinese-Emilia', 178_271, '1.83B'),
    ('YouTube-Cantonese-Emilia', 142_711, '1.46B'),
    ('Malaysian-Tamil-Emilia', 81_723, '0.84B'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='/root/share/packs')
    p.add_argument('--packs', nargs='*', help='subset of the six (default: all)')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    out_root = Path(args.out)
    for name, blocks, tokens in PACKS:
        if args.packs and name not in args.packs:
            continue
        out = out_root / name.lower()
        if (out / 'index.json').exists():
            log(f'{name}: already present, skipping')
            continue
        log(f'{name}: {blocks:,} blocks / {tokens} tokens -> {out}')
        snapshot_with_retry(f'Scicom-intl/{name}-multipacking-10k', local_dir=str(out),
                            max_workers=args.workers)
        log(f'{name}: done')
    log('VC_PACKS_READY')


if __name__ == '__main__':
    main()
