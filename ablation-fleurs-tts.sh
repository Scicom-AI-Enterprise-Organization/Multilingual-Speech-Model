#!/bin/bash
# Optimizer search (README "Optimizer search") on the FLEURS-R NeuCodec TTS pack —
# text -> speech tokens, 102 locales, packed by preparation/multipacking_fleurs.py.
#
# Same protocol as the published search: Qwen3-1.7B-Base, 100 steps, warmup 50,
# FP32-BF16 mixed precision, WSD LR, one grid per optimizer, ranked by mean train
# loss over the last 10 steps. Only the global token size differs: the published search
# ran 10,240 x 256 x 8 GPUs = 21M tokens/step, which the FLEURS packs (~177M tokens
# total, see fleurs/out/summary.json) would replay 12x per run. 1 x 6 x 8 GPUs = 48
# blocks = 491k tokens/step keeps a run at ~27% of one epoch, ~30 min on 8xH20.
#
#   bash ablation-fleurs-tts.sh                        # full grid (16 runs)
#   bash ablation-fleurs-tts.sh --optimizers muon soap # subset
#   bash ablation-fleurs-tts.sh --dry-run              # print commands only
set -e
cd "$(dirname "$0")"
unset LD_LIBRARY_PATH PYTHONPATH        # login shells poison the venv's torch (see CLAUDE.md)
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/share/multilingual-tts/hf}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

BASE="${BASE:-/share/multilingual-tts}"

"$BASE/venv/bin/python" hyperparameter_search.py \
  --train-file "$BASE/fleurs/out/fleurs-tts" \
  --run-prefix fleurs-tts \
  --output-root "$BASE/runs/fleurs-tts" \
  --state-dir "$BASE/search_state/fleurs-tts" \
  --wandb-project "${WANDB_PROJECT:-Multilingual-TTS}" \
  --nproc "${NPROC:-8}" \
  --batch-size "${BS:-1}" \
  --grad-accum "${ACCUM:-6}" \
  --steps "${STEPS:-100}" \
  --warmup "${WARMUP:-50}" \
  --num-decay-steps "${DECAY:-243}" \
  "$@"
