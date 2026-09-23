"""
plot_fleurs_ablation.py
───────────────────────
Figures for the FLEURS-R + Common Voice 22 optimizer ablation.

Two outputs, in the same house style as vc-evaluation/plot_results.py:

  fleurs-ablation-heatmap.png   runs x tasks table, colour per column
  fleurs-ablation-curves.png    dev-loss curve per task, one line per run

Every run trains on one mixture of all six packs, so the tasks share a model but not an
axis: TTS predicts 65k-way speech tokens, the STT tasks predict text. Columns are
normalised independently and the curves get one panel each.

To change style:  edit the STYLE dict.
To re-run:        python plot_fleurs_ablation.py --state-dir <sweep state> --out-dir docs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
#  TASKS  (column order — edit to reorder)
# ══════════════════════════════════════════════════════════════════════════════
TASKS = [
    ('fleurs_tts', 'FLEURS\nTTS'),
    ('fleurs_stt', 'FLEURS\nSTT'),
    ('fleurs_mel', 'FLEURS\nSTT-mel'),
    ('cv22_tts',   'CV22\nTTS'),
    ('cv22_stt',   'CV22\nSTT'),
    ('cv22_mel',   'CV22\nSTT-mel'),
]

PANEL_TITLES = {
    'fleurs_tts': 'FLEURS · TTS — text → speech tokens',
    'fleurs_stt': 'FLEURS · STT — speech tokens → text',
    'fleurs_mel': 'FLEURS · STT — raw log-mel → text',
    'cv22_tts':   'Common Voice 22 · TTS — text → speech tokens',
    'cv22_stt':   'Common Voice 22 · STT — speech tokens → text',
    'cv22_mel':   'Common Voice 22 · STT — raw log-mel → text',
}

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    title_color     = '#1a1a2e',
    label_color     = '#2c2c3e',
    tick_color      = '#555577',
    caption_color   = '#777799',
    anno_bg         = '#f5f5f5',
    missing_bg      = '#e8e8ee',
    missing_text    = '#aaaacc',
    best_color      = '#b07800',      # marks the winner of a column
    main_title_size = 15,
    title_size      = 11,
    caption_size    = 8.5,
    cell_value_size = 8.0,
    run_label_size  = 8.5,
    task_label_size = 8.5,
    axis_fontsize   = 9.5,
    tick_fontsize   = 8.5,
    cell_w          = 1.15,           # inches per task column
    cell_h          = 0.42,           # inches per run row
    mean_col_w      = 1.15,
    run_label_w     = 2.9,
    dpi             = 300,
    heatmap_file    = 'fleurs-ablation-heatmap.png',
    curves_file     = 'fleurs-ablation-curves.png',
    best_file       = 'fleurs-ablation-best-run.png',
)

# one colour per optimizer family
FAMILY = {
    'soap':    dict(color='#203882', edge='#101d55'),   # leader
    'muon':    dict(color='#2e9e4f', edge='#1d7a3a'),
    'shampoo': dict(color='#b07800', edge='#7d5600'),
    'adamw':   dict(color='#e05c3a', edge='#a63c22'),   # diverges at 1e-3
}

# low loss → teal, high loss → warm
LOSS_COLORS = ['#1f7a6a', '#7fb3a4', '#e8dfc0', '#e0a05c', '#c4462f']


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def short_name(run, prefix):
    n = run[len(prefix) + 1:] if run.startswith(prefix) else run
    return n.replace('-wd0.01', '').replace('lr0.001-', '').replace('mlr', 'mlr ').replace('lr', 'lr ')


def load_runs(state_dir, runs_dir, prefix):
    """Finished + diverged runs, each with its per-task finals and dev curves."""
    runs = []
    for marker in sorted(state_dir.glob('*.json')):
        if marker.name == 'summary.json':
            continue
        d = json.loads(marker.read_text())
        curves = {}
        state_file = runs_dir / d['run'] / 'trainer_state.json'
        if state_file.exists():
            for entry in json.loads(state_file.read_text()).get('log_history', []):
                for key, value in entry.items():
                    if key.startswith('eval_') and key.endswith('_loss'):
                        curves.setdefault(key[5:-5], []).append((entry['step'], value))
        runs.append(dict(
            run=d['run'],
            label=short_name(d['run'], prefix),
            family=short_name(d['run'], prefix).split('-')[0].split(' ')[0],
            status=d.get('status'),
            mean=d.get('final_eval_loss'),
            finals={k: v['final'] for k, v in d.get('dev', {}).items()},
            curves=curves,
        ))
    ok = sorted((r for r in runs if r['status'] == 'ok'), key=lambda r: r['mean'])
    bad = [r for r in runs if r['status'] != 'ok']
    return ok + bad


# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP TABLE
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap(runs, tasks, out_dir, s):
    cmap = LinearSegmentedColormap.from_list('loss', LOSS_COLORS, N=256)
    cols = [t for t, _ in tasks] + ['mean']
    n_rows, n_cols = len(runs), len(cols)

    table_w = s['run_label_w'] + len(tasks) * s['cell_w'] + s['mean_col_w']
    table_h = 0.95 + n_rows * s['cell_h']
    fig_w, fig_h = table_w + 0.9, table_h + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    ax = fig.add_axes([s['run_label_w'] / fig_w, 0.9 / fig_h,
                       (len(tasks) * s['cell_w'] + s['mean_col_w']) / fig_w,
                       n_rows * s['cell_h'] / fig_h])
    ax.set_facecolor(s['bg_color'])

    # column-wise normalisation: a TTS loss of 7 and a mel loss of 2.4 are both good
    values = {c: [r['finals'].get(c) if c != 'mean' else r['mean'] for r in runs] for c in cols}
    for c in cols:
        present = [v for v in values[c] if isinstance(v, (int, float))]
        lo, hi = (min(present), max(present)) if present else (0, 1)
        best = min(present) if present else None
        for row, v in enumerate(values[c]):
            col = cols.index(c)
            y = n_rows - row - 1
            if not isinstance(v, (int, float)):
                fc, txt, tc, weight = s['missing_bg'], '—', s['missing_text'], 'normal'
            else:
                fc = cmap(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))
                txt = f'{v:.3f}'
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                tc = '#0a0f14' if lum > 0.5 else '#ffffff'
                weight = 'bold'
            ax.add_patch(plt.Rectangle([col, y], 1, 1, facecolor=fc,
                                       edgecolor=s['bg_color'], linewidth=1.2))
            ax.text(col + 0.5, y + 0.5, txt, ha='center', va='center',
                    fontsize=s['cell_value_size'], color=tc, fontweight=weight,
                    fontfamily='monospace')
            if best is not None and v == best:
                ax.add_patch(plt.Rectangle([col + 0.04, y + 0.06], 0.92, 0.88, fill=False,
                                           edgecolor=s['best_color'], linewidth=1.8))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([lbl for _, lbl in tasks] + ['MEAN'],
                       fontsize=s['task_label_size'], color=s['tick_color'], linespacing=1.4)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([r['label'] for r in reversed(runs)],
                       fontsize=s['run_label_size'], color=s['label_color'],
                       fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(len(tasks), color=s['best_color'], linewidth=1.5)

    fig.text(0.5, 1 - 0.30 / fig_h, 'Optimizer ablation — dev loss per task',
             ha='center', va='top', fontsize=s['main_title_size'],
             color=s['title_color'], fontweight='bold')
    fig.text(0.5, 1 - 0.58 / fig_h,
             'one mixed run per configuration · Qwen3-1.7B-Base · 21M tokens/step · 200 steps  •  '
             'lower is better, colour scaled within each column  •  gold box = best on that task',
             ha='center', va='top', fontsize=s['caption_size'],
             color=s['caption_color'], style='italic')

    out = out_dir / s['heatmap_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CURVES
# ══════════════════════════════════════════════════════════════════════════════

def draw_curves(runs, tasks, out_dir, s):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(wspace=0.22, hspace=0.34, left=0.055, right=0.985,
                        top=0.855, bottom=0.115)

    for ax, (task, _) in zip(axes.ravel(), tasks):
        ax.set_facecolor(s['bg_color'])
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
        ax.grid(True, color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        scored = [r for r in runs if r['curves'].get(task)]
        best = min((r for r in scored if isinstance(r['finals'].get(task), float)),
                   key=lambda r: r['finals'][task], default=None)
        for r in scored:
            xs, ys = zip(*r['curves'][task])
            is_best = best is not None and r['run'] == best['run']
            fam = FAMILY.get(r['family'], dict(color='#777777'))
            ax.plot(xs, ys, color=fam['color'],
                    linewidth=2.6 if is_best else 1.3,
                    alpha=1.0 if is_best else 0.45,
                    marker='o' if is_best else None, markersize=4,
                    zorder=5 if is_best else 3)
        if best is not None:
            xs, ys = zip(*best['curves'][task])
            ax.annotate(f"{best['label']}   {ys[-1]:.3f}",
                        xy=(xs[-1], ys[-1]), xytext=(-8, 14),
                        textcoords='offset points', ha='right',
                        fontsize=8.2, color=s['label_color'], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.28', fc=s['anno_bg'],
                                  ec='none', alpha=0.85))

        ax.text(0.98, 0.96, '↓ lower is better', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#e05c3a', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc=s['anno_bg'], ec='#e05c3a',
                          alpha=0.9, linewidth=1.0))
        ax.set_title(PANEL_TITLES.get(task, task), fontsize=s['title_size'],
                     color=s['title_color'], fontweight='bold', pad=8)
        ax.set_xlabel('step', fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=4)
        ax.set_ylabel('dev loss', fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=4)
        ax.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'])

    handles = [mlines.Line2D([], [], color=FAMILY[f]['color'], linewidth=2.4, label=f)
               for f in FAMILY if any(r['family'] == f for r in runs)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9.5,
               facecolor='#ffffff', edgecolor='#cccccc', labelcolor=s['label_color'],
               framealpha=0.95, bbox_to_anchor=(0.5, 0.012))

    fig.suptitle('Optimizer ablation — dev loss through the run, per task',
                 fontsize=s['main_title_size'], color=s['title_color'],
                 fontweight='bold', y=0.975)
    fig.text(0.5, 0.925,
             'every run trains on one mixture of all six packs  •  each task scored on its own dev split  •  '
             'bold line = best final loss on that task',
             ha='center', fontsize=s['caption_size'], color=s['caption_color'], style='italic')

    out = out_dir / s['curves_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BEST RUN — does one model learn all three tasks?
# ══════════════════════════════════════════════════════════════════════════════

SPEECH_VOCAB = 65536        # <|s_N|> candidates the TTS/STT-token tasks predict into
TEXT_VOCAB = 217448         # full vocabulary the mel task predicts into

TOKEN_TASKS = [('fleurs_tts', 'FLEURS TTS'), ('fleurs_stt', 'FLEURS STT'),
               ('cv22_tts', 'CV22 TTS'), ('cv22_stt', 'CV22 STT')]
MEL_TASKS = [('fleurs_mel', 'FLEURS STT-mel'), ('cv22_mel', 'CV22 STT-mel')]
SHADES = ['#203882', '#3f66c0', '#2e9e4f', '#63c17f']


def draw_best_run(runs, out_dir, s):
    best = next((r for r in runs if r['status'] == 'ok'), None)
    if best is None:
        return None

    fig, (ax_tok, ax_mel) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(wspace=0.22, left=0.06, right=0.98, top=0.80, bottom=0.13)

    for ax, group, baseline, blabel in (
            (ax_tok, TOKEN_TASKS, np.log(SPEECH_VOCAB), f'uniform over {SPEECH_VOCAB:,} speech tokens'),
            (ax_mel, MEL_TASKS, None, None)):
        ax.set_facecolor(s['bg_color'])
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
        ax.grid(True, color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        for i, (task, label) in enumerate(group):
            pts = best['curves'].get(task)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SHADES[i % len(SHADES)], linewidth=2.4, marker='o',
                    markersize=4.5, label=f'{label}   {ys[-1]:.3f}', zorder=4)
        if baseline is not None:
            ax.axhline(baseline, color='#c4462f', linewidth=1.4, linestyle='--', alpha=0.8)
            ax.text(0.015, baseline, f'  {blabel}  ({baseline:.2f} nats)', transform=ax.get_yaxis_transform(),
                    va='bottom', fontsize=8, color='#c4462f', style='italic')
        ax.set_xlabel('step', fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=4)
        ax.set_ylabel('dev loss (nats/token)', fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=4)
        ax.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'])
        ax.legend(fontsize=9, facecolor='#ffffff', edgecolor='#cccccc',
                  labelcolor=s['label_color'], loc='upper right', framealpha=0.95)

    ax_tok.set_title('Predicting speech tokens', fontsize=s['title_size'],
                     color=s['title_color'], fontweight='bold', pad=8)
    ax_mel.set_title('Predicting text from raw log-mel', fontsize=s['title_size'],
                     color=s['title_color'], fontweight='bold', pad=8)
    ax_mel.text(0.02, 0.03, 'mel placeholders are masked out of the labels — this is text-only loss',
                transform=ax_mel.transAxes, ha='left', va='bottom', fontsize=8,
                color=s['caption_color'], style='italic')

    fig.suptitle(f"One model, three tasks — {best['label']} (best run)",
                 fontsize=s['main_title_size'], color=s['title_color'], fontweight='bold', y=0.965)
    fig.text(0.5, 0.895,
             'all six dev losses fall monotonically from one mixed run  •  '
             'the token tasks are still descending at step 200',
             ha='center', fontsize=s['caption_size'], color=s['caption_color'], style='italic')

    out = out_dir / s['best_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE
# ══════════════════════════════════════════════════════════════════════════════

def write_table(runs, tasks, out_dir, title):
    def fmt(v):
        return f'{v:.4f}' if isinstance(v, (int, float)) else '—'

    lines = [title, '',
             '| rank | run | ' + ' | '.join(lbl.replace('\n', ' ') for _, lbl in tasks) + ' | mean |',
             '|---' * (len(tasks) + 3) + '|']
    rank = 0
    for r in runs:
        if r['status'] == 'ok':
            rank += 1
            head = str(rank)
        else:
            head = '—'
        cells = ' | '.join(fmt(r['finals'].get(t)) for t, _ in tasks)
        mean = fmt(r['mean']) if r['status'] == 'ok' else f"_{r['status']}_"
        lines.append(f"| {head} | `{r['label']}` | {cells} | {mean} |")
    out = out_dir / 'fleurs-ablation-results.md'
    out.write_text('\n'.join(lines) + '\n')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state-dir', default='/share/multilingual-tts/search_state/fleurs-b2040')
    p.add_argument('--runs-root', default='/share/multilingual-tts/runs/fleurs-b2040')
    p.add_argument('--run-prefix', default='fleurs-b2040')
    p.add_argument('--out-dir', default='docs')
    p.add_argument('--title', default='### FLEURS-R + Common Voice 22 — optimizer ablation '
                                      '(2048-block batch, 200 steps)')
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        raise SystemExit(f'no sweep state at {state_dir}')
    runs = load_runs(state_dir, Path(args.runs_root), args.run_prefix)
    if not runs:
        raise SystemExit('no runs found')

    seen = {t for r in runs for t in set(r['finals']) | set(r['curves'])}
    tasks = [(t, lbl) for t, lbl in TASKS if t in seen]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'{len(runs)} runs, {len(tasks)} tasks')
    print('heatmap ->', draw_heatmap(runs, tasks, out_dir, STYLE))
    print('curves  ->', draw_curves(runs, tasks, out_dir, STYLE))
    print('best    ->', draw_best_run(runs, out_dir, STYLE))
    print('table   ->', write_table(runs, tasks, out_dir, args.title))


if __name__ == '__main__':
    main()
