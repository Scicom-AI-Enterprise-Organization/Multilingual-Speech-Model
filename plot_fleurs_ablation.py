"""Plot the FLEURS optimizer ablation — one figure per task.

Every run trains on one mixture (TTS audio tokens + STT audio tokens + STT raw mel) and
is scored on each task's own dev split. Those three losses share no axis — the TTS task
predicts 65k-way speech tokens while both STT tasks predict text — so each gets its own
figure rather than three lines on one chart.

    python plot_fleurs_ablation.py                    # every task found in the sweep
    python plot_fleurs_ablation.py --tasks mel
    python plot_fleurs_ablation.py --out-dir docs/

Writes `fleurs-ablation-<task>.png` per task plus `fleurs-ablation.md` with the ranking.
"""

import argparse
import json
from pathlib import Path

TASKS = {
    'tts': 'TTS — text → NeuCodec speech tokens',
    'stt': 'STT — NeuCodec speech tokens → text',
    'mel': 'STT — whisper log-mel → text',
}

# one colour per optimizer family, so an optimizer's LR variants read as a group
COLORS = {
    'adamw': '#4C72B0',
    'muon': '#DD8452',
    'shampoo': '#55A868',
    'soap': '#C44E52',
    'lion': '#8172B3',
    'ademamix': '#937860',
}


def load_runs(state_dir, runs_dir, prefix):
    """Finished runs with their per-task dev curves."""
    runs = []
    for marker in sorted(state_dir.glob('*.json')):
        if marker.name == 'summary.json':
            continue
        record = json.loads(marker.read_text())
        if record.get('status') != 'ok':
            continue
        curves = {}
        state_file = runs_dir / record['run'] / 'trainer_state.json'
        if state_file.exists():
            for entry in json.loads(state_file.read_text()).get('log_history', []):
                for key, value in entry.items():
                    if key.startswith('eval_') and key.endswith('_loss'):
                        task = key[len('eval_'):-len('_loss')] or 'dev'
                        curves.setdefault(task, []).append((entry['step'], value))
        label = record['run'][len(prefix) + 1:] if record['run'].startswith(prefix) else record['run']
        runs.append({
            'run': record['run'],
            'label': label,
            'optimizer': label.split('-')[0],
            'curves': curves,
            'dev': record.get('dev', {}),
            'mean_dev': record.get('final_eval_loss'),
            'last10': record.get('last10_mean_loss'),
        })
    return runs


def plot_task(task, runs, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scored = [r for r in runs if r['dev'].get(task) or r['curves'].get(task)]
    if not scored:
        return None
    scored.sort(key=lambda r: r['dev'].get(task, {}).get('final', float('inf')))

    fig, (curves, bars) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})

    seen = set()
    for i, r in enumerate(scored):
        points = r['curves'].get(task)
        if not points:
            continue
        xs, ys = zip(*points)
        best = i == 0
        colour = COLORS.get(r['optimizer'], '#777777')
        curves.plot(xs, ys, color=colour, linewidth=2.4 if best else 1.2,
                    alpha=1.0 if best else 0.5, zorder=3 if best else 2,
                    marker='o' if best else None, markersize=3.5,
                    label=r['optimizer'] if r['optimizer'] not in seen else None)
        seen.add(r['optimizer'])
    if scored[0]['curves'].get(task):
        xs, ys = zip(*scored[0]['curves'][task])
        curves.annotate(scored[0]['label'], (xs[-1], ys[-1]), textcoords='offset points',
                        xytext=(-8, 8), ha='right', fontsize=8, color='#222222')
    curves.set_xlabel('step')
    curves.set_ylabel('dev loss')
    curves.set_title(f'{TASKS.get(task, task)}\ndev loss, one mixed run per optimizer config',
                     fontsize=10)
    curves.grid(alpha=0.25, linewidth=0.6)
    curves.legend(fontsize=8, frameon=False, ncol=2)

    top = [r for r in scored if task in r['dev']][:10][::-1]
    if top:
        values = [r['dev'][task]['final'] for r in top]
        bars.barh([r['label'] for r in top], values,
                  color=[COLORS.get(r['optimizer'], '#777777') for r in top])
        lo, hi = min(values), max(values)
        bars.set_xlim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.02 * (hi - lo + 1e-6))
    bars.set_xlabel('final dev loss')
    bars.set_title(f'best {len(top)} configurations on this task', fontsize=10)
    bars.tick_params(axis='y', labelsize=7)
    bars.grid(axis='x', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    path = out_dir / f'fleurs-ablation-{task}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fmt(value):
    return f'{value:.4f}' if isinstance(value, (int, float)) else '—'


def table(runs, tasks):
    runs = sorted(runs, key=lambda r: r['mean_dev'] if r['mean_dev'] is not None else float('inf'))
    header = '| rank | run | ' + ' | '.join(f'{t} dev' for t in tasks) + ' | mean dev | train last10 |'
    lines = ['### FLEURS ablation — mixed training, per-task dev loss', '',
             header, '|---' * (len(tasks) + 4) + '|']
    for i, r in enumerate(runs, 1):
        cells = ' | '.join(fmt(r['dev'].get(t, {}).get('final')) for t in tasks)
        lines.append(f"| {i} | `{r['label']}` | {cells} | {fmt(r['mean_dev'])} | {fmt(r['last10'])} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state-dir', default='/share/multilingual-tts/search_state/fleurs')
    parser.add_argument('--runs-root', default='/share/multilingual-tts/runs/fleurs')
    parser.add_argument('--run-prefix', default='fleurs')
    parser.add_argument('--tasks', nargs='+', default=None, help='default: every task in the sweep')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        raise SystemExit(f'no sweep state at {state_dir}')
    runs = load_runs(state_dir, Path(args.runs_root), args.run_prefix)
    if not runs:
        raise SystemExit('no finished runs yet')

    found = {t for r in runs for t in set(r['dev']) | set(r['curves'])}
    tasks = args.tasks or [t for t in TASKS if t in found] + sorted(found - set(TASKS))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        path = plot_task(task, runs, out_dir)
        if path:
            best = min((r for r in runs if task in r['dev']),
                       key=lambda r: r['dev'][task]['final'], default=None)
            note = f" | best {best['label']} {best['dev'][task]['final']:.4f}" if best else ''
            print(f'{task}: {len(runs)} runs -> {path}{note}')

    md = out_dir / 'fleurs-ablation.md'
    md.write_text(table(runs, tasks) + '\n')
    print(f'table written to {md}')


if __name__ == '__main__':
    main()
