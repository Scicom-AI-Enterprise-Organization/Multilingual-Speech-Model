"""Turn the packs on disk into one `--train_file` string, with the step count to match.

The trainer takes `dir:weight,dir:weight,…` where weight = how many times that pack is
drawn per epoch. Getting those by hand across ~37 pack directories is how a mixture ends
up 3x off, so this reads every pack's real block count and prints:

  * blocks and tokens per pack and per task arm
  * the share of the mixture each arm actually gets after weighting
  * total optimizer steps for the launch script's batch geometry
  * `num_decay_steps` = 10% of those steps, for --lr_scheduler_kwargs
  * the `--train_file` string itself

Arms follow the training spec:

  TTS tokens   6 VC multipacking packs + the corpus TTS waves + non-verbal tags
  STT tokens   the corpus STT waves
  STT mel      the corpus mel waves

Non-verbal tags are 2.85M tokens against ~88B. `--nonverbal-weight` exists because at 1.0
the model sees a tag roughly never; it is the one weight that is a judgement call rather
than a measurement.

Usage:
    python build_mixture.py
    python build_mixture.py --nonverbal-weight 50 --mel-weight 2.0
    python build_mixture.py --emit-script scripts/1.7B-full.sh
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLOCK = 10240


def packs_for(corpus_out, packs_root, fleurs_out, cv22_out, nonverbal_out):
    """{arm: [(path, why)]} — every pack directory that exists, grouped by task."""
    arms = {'tts': [], 'stt': [], 'mel': []}

    for d in sorted(Path(packs_root).glob('*')):
        if (d / 'index.json').exists():
            arms['tts'].append((d, 'voice-conversion pairs'))

    for arm in ('tts', 'stt', 'mel'):
        root = Path(corpus_out) / f'corpus-{arm}'
        for d in sorted(root.glob('wave-*')):
            # a wave still being packed has neither index.json nor a summary; including it
            # would put a half-written pack in the mixture
            if (d / 'index.json').exists() and (Path(corpus_out) / f'summary-{d.name}.json').exists():
                arms[arm].append((d, 'corpus'))

    for out, label in ((fleurs_out, 'fleurs'), (cv22_out, 'cv22')):
        if not out:
            continue
        for d in sorted(Path(out).glob('*')):
            name = d.name
            if not d.is_dir() or not (d / 'index.json').exists():
                continue          # summary/vocab json files live beside the packs
            if name.endswith('-dev'):
                continue          # dev packs are for eval, never the train mixture
            arm = 'mel' if 'mel' in name else ('stt' if 'stt' in name else 'tts')
            arms[arm].append((d, label))

    if nonverbal_out and (Path(nonverbal_out) / 'index.json').exists():
        arms['tts'].append((Path(nonverbal_out), 'non-verbal tags'))

    return arms


# Greedy packing leaves a little slack in each block; measured at 97.0% of capacity on
# corpus wave 0 (470,830 blocks -> 4.678B tokens). Only used where no summary exists.
FILL = 0.97


def tokens_from_summary(path, arm, corpus_out):
    """Exact token count from the packer's own summary, or None.

    Reading rows to count tokens costs minutes per pack over NFS (random block reads are
    row-group decodes), and every packer already wrote the number.
    """
    path = Path(path)
    candidates = []
    if path.parent.name.startswith('corpus-'):          # corpus wave
        candidates.append((Path(corpus_out) / f'summary-{path.name}.json', f'{arm}_tokens'))
    for s in ('summary-train.json', 'summary.json', f'summary-{path.name}-train.json',
              'summary-cv22-train.json'):
        candidates.append((path.parent / s, f'{arm}_tokens'))
        candidates.append((path / s, f'{arm}_tokens'))
    for f, key in candidates:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if key in d:
            return int(d[key])
    return None


def count(path, arm, corpus_out):
    """(blocks, tokens, exact?) — blocks from the index, tokens from a summary if there."""
    from chinidataset import StreamingDataset
    n = len(StreamingDataset(local=str(path)))
    tok = tokens_from_summary(path, arm, corpus_out)
    if tok is not None:
        return n, tok, True
    return n, int(n * BLOCK * FILL), False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-out', default='/root/share/corpus/out')
    p.add_argument('--packs-root', default='/root/share/packs')
    p.add_argument('--fleurs-out', default='/root/share/fleurs/out')
    p.add_argument('--cv22-out', default='/root/share/cv22/out')
    p.add_argument('--nonverbal-out', default='/root/share/nonverbal/out/nonverbal-tts')
    p.add_argument('--nonverbal-weight', type=float, default=50.0,
                   help='2.85M tokens is 0.003%% of the mixture at weight 1')
    p.add_argument('--tts-weight', type=float, default=1.0)
    p.add_argument('--stt-weight', type=float, default=1.0)
    p.add_argument('--mel-weight', type=float, default=1.0)
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--per-device-batch', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=64)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--json-out', default=None)
    args = p.parse_args()

    arms = packs_for(args.corpus_out, args.packs_root, args.fleurs_out, args.cv22_out,
                     args.nonverbal_out)
    weight_of = {'tts': args.tts_weight, 'stt': args.stt_weight, 'mel': args.mel_weight}

    entries, totals, report = [], {}, {}
    for arm in ('tts', 'stt', 'mel'):
        blocks = tokens = weighted = 0
        report[arm] = []
        for path, why in arms[arm]:
            try:
                n, tok, exact = count(path, arm, args.corpus_out)
            except Exception as e:
                print(f'  !! {path}: {type(e).__name__} {e}', file=sys.stderr)
                continue
            w = args.nonverbal_weight if why == 'non-verbal tags' else weight_of[arm]
            entries.append(f'{path}:{w:g}')
            blocks += n
            tokens += tok
            weighted += n * w
            report[arm].append({'path': str(path), 'source': why, 'blocks': n,
                                'tokens': tok, 'tokens_exact': exact, 'weight': w})
        totals[arm] = {'blocks': blocks, 'tokens': tokens, 'weighted_blocks': weighted}

    approx = sum(1 for r in report.values() for e in r if not e['tokens_exact'])
    grand = sum(t['weighted_blocks'] for t in totals.values())
    print(f'{"arm":<6} {"packs":>6} {"blocks":>12} {"tokens":>16} {"weighted":>12} {"share":>7}')
    for arm in ('tts', 'stt', 'mel'):
        t = totals[arm]
        share = t['weighted_blocks'] / grand * 100 if grand else 0
        print(f'{arm:<6} {len(report[arm]):>6} {t["blocks"]:>12,} {t["tokens"]:>16,} '
              f'{t["weighted_blocks"]:>12,.0f} {share:>6.1f}%')
    print(f'{"total":<6} {sum(len(r) for r in report.values()):>6} '
          f'{sum(t["blocks"] for t in totals.values()):>12,} '
          f'{sum(t["tokens"] for t in totals.values()):>16,} {grand:>12,.0f}')

    if approx:
        print(f'({approx} packs have no summary; their tokens are blocks x {BLOCK} x {FILL})')

    global_batch = args.gpus * args.per_device_batch * args.grad_accum
    steps = int(args.epochs * grand / global_batch)
    decay = max(1, round(steps * 0.10))
    print(f'\nbatch: {args.per_device_batch} x {args.grad_accum} accum x {args.gpus} GPUs '
          f'= {global_batch} blocks = {global_batch * BLOCK / 1e6:.1f}M tokens/step')
    print(f'steps: {steps:,} for {args.epochs:g} epoch(s)')
    print(f'--lr_scheduler_kwargs \'{{"num_decay_steps": {decay}, "min_lr_ratio": 1e-1}}\'')
    print(f'\n--train_file "{",".join(entries)}"')

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'arms': report, 'totals': totals, 'steps': steps,
                       'num_decay_steps': decay, 'global_batch_blocks': global_batch,
                       'train_file': ','.join(entries)}, f, indent=2)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
