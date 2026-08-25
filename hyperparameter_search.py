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
torchrun --nproc_per_node {nproc} \
-m qwen3_optimizer_search \
--model_name_or_path "{model}" \
--optimizer {optimizer} {matrix_lr_arg} \
--learning_rate {lr} \
--weight_decay {wd} \
--num_decay_steps {num_decay_steps} \
--min_lr_ratio 0.1 \
--per_device_train_batch_size {batch_size} \
--gradient_accumulation_steps {grad_accum} \
--output_dir {output_dir} \
--bf16 --do_train --do_eval false --max_steps {steps} \
--train_file "{train_file}" \
--logging_steps 1 \
--warmup_steps {warmup} \
--block_size 10240 \
--save_steps 500 \
--save_total_limit 10 \
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
    losses = [h['loss'] for h in state.get('log_history', []) if 'loss' in h]
    if not losses:
        raise ValueError(f'no loss entries in {state_file}')
    return {
        'final_loss': losses[-1],
        'last10_mean_loss': statistics.mean(losses[-10:]),
        'min_loss': min(losses),
        'steps_logged': len(losses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-file', required=True, help='chinidataset multipacking dir')
    parser.add_argument('--optimizers', nargs='+', default=list(DEFAULT_GRIDS),
                        help=f'subset of: {", ".join(DEFAULT_GRIDS)}')
    parser.add_argument('--grid-json', help='JSON file overriding DEFAULT_GRIDS')
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B-Base')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--num-decay-steps', type=int, default=243)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=32)
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

    unknown = set(args.optimizers) - set(grids)
    if unknown:
        parser.error(f'no grid for: {", ".join(sorted(unknown))}')

    state_dir = Path(args.state_dir)
    state_dir.mkdir(exist_ok=True)

    runs = []
    for opt in args.optimizers:
        for cfg in grids[opt]:
            matrix_lr = cfg.get('matrix_lr')
            parts = [opt, f"lr{cfg['lr']}"]
            if matrix_lr is not None:
                parts.append(f"mlr{matrix_lr}")
            parts.append(f"wd{cfg['wd']}")
            run_name = 'search-' + '-'.join(parts)
            cmd = COMMAND.format(
                wandb_project=args.wandb_project,
                run_name=run_name,
                nproc=args.nproc,
                model=args.model,
                optimizer=opt,
                matrix_lr_arg=f'--matrix_lr {matrix_lr}' if matrix_lr is not None else '',
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

    ok = sorted(
        (r for r in results.values() if r.get('status') == 'ok'),
        key=lambda r: r['last10_mean_loss'],
    )
    failed = [r for r in results.values() if r.get('status') == 'failed']

    print('\n=== ranked by mean loss over last 10 steps ===')
    for rank, r in enumerate(ok, 1):
        print(f"{rank:2d}. {r['run']}: last10={r['last10_mean_loss']:.4f} "
              f"final={r['final_loss']:.4f} min={r['min_loss']:.4f}")
    for r in failed:
        print(f" X. {r['run']}: FAILED (exit {r['returncode']})")

    with open(state_dir / 'summary.json', 'w') as f:
        json.dump({'ranked': ok, 'failed': failed}, f, indent=2)
    print(f"\nsummary written to {state_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
