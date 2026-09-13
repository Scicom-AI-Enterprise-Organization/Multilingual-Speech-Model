"""Plot the FLEURS optimizer ablation — one figure per arm.

Each arm is a different prediction target, so their losses share no axis: the TTS arm
predicts NeuCodec speech tokens (~65k-way), both STT arms predict text. Putting them on
one chart would rank corpora, not optimizers, so every arm gets its own figure.

    python plot_fleurs_ablation.py                       # all arms found under --state-root
    python plot_fleurs_ablation.py --arms fleurs-stt-mel
    python plot_fleurs_ablation.py --out-dir docs/

Reads each run's `trainer_state.json` for the dev-loss curve and the sweep's
`summary.json` for the ranking, and writes `<arm>.png` plus a markdown table per arm.
"""

import argparse
import json
from pathlib import Path

ARMS = {
    'fleurs-tts': 'TTS — text → NeuCodec speech tokens',
    'fleurs-stt': 'STT — NeuCodec speech tokens → text',
    'fleurs-stt-mel': 'STT — whisper log-mel → text',
}

# one colour per optimizer family, so the LR variants of an optimizer read as a group
COLORS = {
    'adamw': '#4C72B0',
    'muon': '#DD8452',
    'shampoo': '#55A868',
    'soap': '#C44E52',
    'lion': '#8172B3',
    'ademamix': '#937860',
}


def optimizer_of(run, arm):
    return run[len(arm) + 1:].split('-')[0]


def load_runs(state_dir, runs_dir, arm):
    """(run name, [(step, dev loss)], final dev loss) for every finished run."""
    runs = []
    for marker in sorted(state_dir.glob('*.json')):
        if marker.name == 'summary.json':
            continue
        record = json.loads(marker.read_text())
        if record.get('status') != 'ok':
            continue
        state_file = runs_dir / record['run'] / 'trainer_state.json'
        curve, train = [], []
        if state_file.exists():
            history = json.loads(state_file.read_text()).get('log_history', [])
            curve = [(h['step'], h['eval_loss']) for h in history if 'eval_loss' in h]
            train = [(h['step'], h['loss']) for h in history if 'loss' in h]
        runs.append({
            'run': record['run'],
            'label': record['run'][len(arm) + 1:],
            'optimizer': optimizer_of(record['run'], arm),
            'curve': curve,
            'train': train,
            'final': record.get('final_eval_loss', record.get('last10_mean_loss')),
            'best': record.get('min_eval_loss'),
            'last10': record.get('last10_mean_loss'),
            'is_eval': 'final_eval_loss' in record,
        })
    return sorted(runs, key=lambda r: r['final'] if r['final'] is not None else float('inf'))


def plot_arm(arm, runs, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric = 'dev loss' if runs and runs[0]['is_eval'] else 'train loss'
    fig, (curves, bars) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})

    seen = set()
    for i, r in enumerate(runs):
        colour = COLORS.get(r['optimizer'], '#777777')
        points = r['curve'] or r['train']
        if not points:
            continue
        xs, ys = zip(*points)
        best = i == 0
        curves.plot(xs, ys, color=colour, linewidth=2.4 if best else 1.2,
                    alpha=1.0 if best else 0.55, zorder=3 if best else 2,
                    label=r['optimizer'] if r['optimizer'] not in seen else None)
        seen.add(r['optimizer'])
    if runs and runs[0]['curve']:
        xs, ys = zip(*runs[0]['curve'])
        curves.annotate(runs[0]['label'], (xs[-1], ys[-1]), textcoords='offset points',
                        xytext=(-6, 8), ha='right', fontsize=8, color='#222222')
    curves.set_xlabel('step')
    curves.set_ylabel(metric)
    curves.set_title(f'{ARMS.get(arm, arm)}\n{metric} through the 100-step run', fontsize=10)
    curves.grid(alpha=0.25, linewidth=0.6)
    curves.legend(fontsize=8, frameon=False, ncol=2)

    top = [r for r in runs if r['final'] is not None][:10][::-1]
    bars.barh([r['label'] for r in top], [r['final'] for r in top],
              color=[COLORS.get(r['optimizer'], '#777777') for r in top])
    lo = min(r['final'] for r in top)
    hi = max(r['final'] for r in top)
    bars.set_xlim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.02 * (hi - lo + 1e-6))
    bars.set_xlabel(f'final {metric}')
    bars.set_title(f'best {len(top)} configurations', fontsize=10)
    bars.tick_params(axis='y', labelsize=7)
    bars.grid(axis='x', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    path = out_dir / f'{arm}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fmt(value):
    return f'{value:.4f}' if isinstance(value, (int, float)) else '—'


def table(arm, runs):
    metric = 'dev loss' if runs and runs[0]['is_eval'] else 'train loss'
    lines = [f'### {ARMS.get(arm, arm)}', '',
             f'| rank | run | final {metric} | best {metric} | train last10 |',
             '|---|---|---|---|---|']
    for i, r in enumerate(runs, 1):
        lines.append(f"| {i} | `{r['label']}` | {fmt(r['final'])} | "
                     f"{fmt(r['best'])} | {fmt(r['last10'])} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state-root', default='/share/multilingual-tts/search_state')
    parser.add_argument('--runs-root', default='/share/multilingual-tts/runs')
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for arm in args.arms:
        state_dir = Path(args.state_root) / arm
        if not state_dir.is_dir():
            print(f'{arm}: no state dir, skipping')
            continue
        runs = load_runs(state_dir, Path(args.runs_root) / arm, arm)
        if not runs:
            print(f'{arm}: no finished runs yet')
            continue
        path = plot_arm(arm, runs, out_dir)
        metric = 'dev' if runs[0]['is_eval'] else 'train'
        print(f'{arm}: {len(runs)} runs, best {runs[0]["label"]} '
              f'({metric} {runs[0]["final"]:.4f}) -> {path}')
        sections.append((arm, runs))

    if sections:
        md = out_dir / 'fleurs-ablation.md'
        md.write_text('\n\n'.join(table(arm, runs) for arm, runs in sections) + '\n')
        print(f'tables written to {md}')


if __name__ == '__main__':
    main()
