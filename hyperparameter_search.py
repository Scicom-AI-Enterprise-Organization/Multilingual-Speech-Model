"""Optimizer + LR sweep harness driving qwen3_optimizer_search.py.

Replaces the old hyperparameter_search.py / hyperparameter_search_extra.py grid
scripts. Sweeps optimizers (adamw, muon, shampoo, soap, lion, ademamix) over
per-optimizer LR grids, resumes by run name, parses each run's loss curve from
trainer_state.json, and prints a ranked summary.

Usage:
    python hyperparameter_search.py --train-file <chinidataset dir>            # all optimizers
    python hyperparameter_search.py --train-file ... --optimizers muon shampoo soap
    python hyperparameter_search.py --train-file ... --dry-run                 # print commands only
    python hyperparameter_search.py --train-file ... --grid-json my_grid.json  # custom grids

shampoo/soap/lion/ademamix need `pip install pytorch_optimizer`.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# Per-optimizer grids. lr = AdamW side (and everything for full-model optimizers),
# matrix_lr = the 2D-hidden-weight sub-optimizer in hybrid modes, wd = weight decay.
# Centered on the aggressive-LR result that won the original search
# (adamw 1e-3 / muon 1e-2 / decay 0.01). Lion wants ~3-10x lower LR and higher decay
# than AdamW; SOAP/Shampoo hidden-matrix LRs follow their papers' LLM settings.
DEFAULT_GRIDS = {
    'adamw': [
        {'lr': lr, 'wd': 0.01} for lr in (5e-4, 1e-3, 2e-3)
    ],
    'muon': [
        {'lr': 1e-3, 'matrix_lr': mlr, 'wd': 0.01} for mlr in (5e-3, 1e-2, 2e-2)
    ],
    'shampoo': [
        {'lr': 1e-3, 'matrix_lr': mlr, 'wd': 0.01} for mlr in (5e-4, 1e-3, 3e-3)
    ],
    'soap': [
        {'lr': 1e-3, 'matrix_lr': mlr, 'wd': 0.01} for mlr in (1e-3, 3e-3, 1e-2)
    ],
    'lion': [
        {'lr': lr, 'wd': 0.1} for lr in (1e-4, 3e-4)
    ],
    'ademamix': [
        {'lr': lr, 'wd': 0.01} for lr in (5e-4, 1e-3)
    ],
}

COMMAND = """
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="{wandb_project}" \
WANDB_NAME="{run_name}" \
{python} -m torch.distributed.run --nproc_per_node {nproc} \
-m qwen3_optimizer_search \
--model_name_or_path "{model}" \
--optimizer {optimizer} {matrix_lr_arg} {added_tokens_arg} \
--learning_rate {lr} \
--weight_decay {wd} \
--num_decay_steps {num_decay_steps} \
--min_lr_ratio 0.1 \
--per_device_train_batch_size {batch_size} \
--gradient_accumulation_steps {grad_accum} \
--output_dir {output_dir} \
--bf16 --do_train --max_steps {steps} \
--train_file "{train_file}" {eval_args} {audio_args} \
--logging_steps 1 \
--warmup_steps {warmup} \
--block_size 10240 \
--save_strategy no \
--gradient_checkpointing true \
--torch_dtype float32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 20 \
--remove_unused_columns false
""".strip()


def parse_result(output_dir):
    """Pull the loss curve out of a finished run; rank by mean of the last 10 steps."""
    state_file = Path(output_dir) / 'trainer_state.json'
    with open(state_file) as f:
        state = json.load(f)
    history = state.get('log_history', [])
    losses = [h['loss'] for h in history if 'loss' in h]
    if not losses:
        raise ValueError(f'no loss entries in {state_file}')
    result = {
        'final_loss': losses[-1],
        'last10_mean_loss': statistics.mean(losses[-10:]),
        'min_loss': min(losses),
        'steps_logged': len(losses),
    }
    # a mixed run reports one dev loss per task (eval_tts_loss, eval_stt_loss, …); a
    # single dev set reports plain eval_loss
    curves = {}
    for entry in history:
        for k, v in entry.items():
            if k.startswith('eval_') and k.endswith('_loss'):
                curves.setdefault(k[len('eval_'):-len('_loss')] or 'dev', []).append(v)
    if curves:
        result['dev'] = {name: {'final': vals[-1], 'min': min(vals), 'points': len(vals)}
                         for name, vals in sorted(curves.items())}
        # the ranking scalar: the mean across tasks, so no single task's scale decides the
        # order on its own (speech-token loss runs several nats above text loss)
        result['final_eval_loss'] = statistics.mean(v['final'] for v in result['dev'].values())
        result['min_eval_loss'] = statistics.mean(v['min'] for v in result['dev'].values())
        result['eval_tasks'] = sorted(curves)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-file', required=True, help='chinidataset multipacking dir')
    parser.add_argument('--optimizers', nargs='+', default=None,
                        help=f'subset of the grid (default: everything in it). '
                             f'Built-in grids: {", ".join(DEFAULT_GRIDS)}')
    parser.add_argument('--grid-json', help='JSON file overriding DEFAULT_GRIDS')
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B-Base')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--num-decay-steps', type=int, default=243)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=32)
    parser.add_argument('--validation-file',
                        help='dev-split pack to validate on; runs are then ranked on dev loss')
    parser.add_argument('--eval-steps', type=int, default=25,
                        help='evaluate every N steps (and at the end of the run)')
    parser.add_argument('--max-eval-blocks', type=int, default=256,
                        help='cap the dev pass at this many evenly-strided blocks')
    parser.add_argument('--audio-dir',
                        help='root the pack\'s audio paths resolve against — switches the trainer '
                             'to the raw log-mel front end (mel packs only)')
    parser.add_argument('--added-tokens-file',
                        help='JSON list of tokens appended after the speech tokens '
                             '(STT packs; must match the file the data was packed with)')
    parser.add_argument('--run-prefix', default='search',
                        help='run-name prefix, so sweeps over different datasets do not collide')
    parser.add_argument('--output-root', default='gfs/01be5b33/optimizer-search')
    parser.add_argument('--state-dir', default='search_state', help='per-run done markers + results')
    parser.add_argument('--wandb-project', default='Multilingual-TTS')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-on-fail', action='store_true',
                        help='abort the sweep on the first failed run (default: record and continue)')
    args = parser.parse_args()

    grids = DEFAULT_GRIDS
    if args.grid_json:
        with open(args.grid_json) as f:
            grids = json.load(f)

    # a --grid-json defines the sweep: without --optimizers, run exactly what it holds
    args.optimizers = args.optimizers or list(grids)
    unknown = set(args.optimizers) - set(grids)
    if unknown:
        parser.error(f'no grid for: {", ".join(sorted(unknown))}')

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for opt in args.optimizers:
        for cfg in grids[opt]:
            matrix_lr = cfg.get('matrix_lr')
            parts = [opt, f"lr{cfg['lr']}"]
            if matrix_lr is not None:
                parts.append(f"mlr{matrix_lr}")
            parts.append(f"wd{cfg['wd']}")
            run_name = f'{args.run_prefix}-' + '-'.join(parts)
            cmd = COMMAND.format(
                # launch with the interpreter running the harness, so a venv that
                # inherits torch from system site-packages (no venv/bin/torchrun)
                # still launches the right python
                python=sys.executable,
                wandb_project=args.wandb_project,
                run_name=run_name,
                nproc=args.nproc,
                model=args.model,
                optimizer=opt,
                matrix_lr_arg=f'--matrix_lr {matrix_lr}' if matrix_lr is not None else '',
                added_tokens_arg=(f'--added_tokens_file {args.added_tokens_file}'
                                  if args.added_tokens_file else ''),
                eval_args=(
                    f'--validation_file "{args.validation_file}" --do_eval true '
                    f'--eval_strategy steps --eval_steps {args.eval_steps} '
                    f'--per_device_eval_batch_size 1 --prediction_loss_only true '
                    f'--max_eval_blocks {args.max_eval_blocks}'
                    if args.validation_file else '--do_eval false'),
                audio_args=(f'--audio_dir "{args.audio_dir}"' if args.audio_dir else ''),
                lr=cfg['lr'],
                wd=cfg['wd'],
                num_decay_steps=args.num_decay_steps,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                output_dir=f'{args.output_root}/{run_name}',
                steps=args.steps,
                train_file=args.train_file,
                warmup=args.warmup,
            )
            runs.append((run_name, cfg, cmd))

    print(f'{len(runs)} runs planned')
    results = {}
    for i, (run_name, cfg, cmd) in enumerate(runs):
        marker = state_dir / f'{run_name}.json'
        if marker.exists():
            with open(marker) as f:
                results[run_name] = json.load(f)
            print(f'[{i + 1}/{len(runs)}] {run_name}: already done, skipping')
            continue

        print(f'[{i + 1}/{len(runs)}] {run_name}')
        print(cmd)
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, shell=True)
        record = {'run': run_name, 'config': cfg}
        if proc.returncode != 0:
            record['status'] = 'failed'
            record['returncode'] = proc.returncode
            print(f'{run_name} FAILED with exit code {proc.returncode}')
            with open(marker, 'w') as f:
                json.dump(record, f, indent=2)
            results[run_name] = record
            if args.stop_on_fail:
                break
            continue

        record['status'] = 'ok'
        record.update(parse_result(f'{args.output_root}/{run_name}'))
        with open(marker, 'w') as f:
            json.dump(record, f, indent=2)
        results[run_name] = record

    if args.dry_run:
        return

    ok = [r for r in results.values() if r.get('status') == 'ok']
    # dev loss is the metric whenever every finished run measured one; train loss alone
    # would rank an optimizer that memorised the 100 steps above one that generalised
    on_eval = bool(ok) and all('final_eval_loss' in r for r in ok)
    key = 'final_eval_loss' if on_eval else 'last10_mean_loss'
    ok = sorted(ok, key=lambda r: r[key])
    failed = [r for r in results.values() if r.get('status') == 'failed']

    tasks = sorted({t for r in ok for t in r.get('eval_tasks', [])})
    if on_eval and len(tasks) > 1:
        print(f"\n=== ranked by mean dev loss over {', '.join(tasks)} ===")
    else:
        print(f"\n=== ranked by {'dev loss' if on_eval else 'mean train loss over last 10 steps'} ===")
    for rank, r in enumerate(ok, 1):
        line = f"{rank:2d}. {r['run']}: "
        if on_eval:
            line += f"dev={r['final_eval_loss']:.4f} "
            if len(tasks) > 1:
                line += '(' + ' '.join(f"{t}={r['dev'][t]['final']:.4f}" for t in tasks
                                       if t in r.get('dev', {})) + ') '
        print(line + f"train_last10={r['last10_mean_loss']:.4f}")
    for r in failed:
        print(f" X. {r['run']}: FAILED (exit {r['returncode']})")

    with open(state_dir / 'summary.json', 'w') as f:
        json.dump({'ranked_by': key, 'ranked': ok, 'failed': failed}, f, indent=2)
    print(f"\nsummary written to {state_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
