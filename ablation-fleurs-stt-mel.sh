#!/bin/bash
# Optimizer search (README "Optimizer search") on the FLEURS-R STT raw-mel pack —
# whisper log-mel -> text, 102 locales, packed by preparation/multipacking_fleurs.py.
#
# Trains on the train-split pack and ranks runs on the **dev-split pack**, so the sweep
# measures what an optimizer generalises to rather than how fast it fits 100 steps.
# Dev loss is only comparable within an arm: it is the loss of predicting text from raw mel.
#
# Same protocol as the published search: Qwen3-1.7B-Base, 100 steps, warmup 50,
# FP32-BF16 mixed precision, WSD LR, one grid per optimizer. Only the global token size
# differs: the published search ran 10,240 x 256 x 8 GPUs = 21M tokens/step, which the
# FLEURS packs (~177M tokens) would replay 12x per run. 1 x 6 x 8 GPUs = 48 blocks =
# 491k tokens/step keeps a run at ~27% of one epoch.
#
#   bash ablation-fleurs-stt-mel.sh                        # full grid (16 runs)
#   bash ablation-fleurs-stt-mel.sh --optimizers muon soap # subset
#   bash ablation-fleurs-stt-mel.sh --dry-run              # print commands only
set -e
cd "$(dirname "$0")"
unset LD_LIBRARY_PATH PYTHONPATH        # login shells poison the venv's torch (see CLAUDE.md)
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/share/multilingual-tts/hf}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

BASE="${BASE:-/share/multilingual-tts}"
# the mel pack stores audio paths; this is the root they resolve against
AUDIO="${AUDIO:-/share/fleurs-work}"

"$BASE/venv/bin/python" hyperparameter_search.py \
  --train-file "$BASE/fleurs/out/fleurs-mel" \
  --validation-file "$BASE/fleurs/out/fleurs-mel-dev" \
  --run-prefix fleurs-stt-mel \
  --output-root "$BASE/runs/fleurs-stt-mel" \
  --state-dir "$BASE/search_state/fleurs-stt-mel" \
  --wandb-project "${WANDB_PROJECT:-Multilingual-TTS}" \
  --nproc "${NPROC:-8}" \
  --batch-size "${BS:-1}" \
  --grad-accum "${ACCUM:-6}" \
  --steps "${STEPS:-100}" \
  --warmup "${WARMUP:-50}" \
  --num-decay-steps "${DECAY:-243}" \
  --eval-steps "${EVAL_STEPS:-25}" \
  --max-eval-blocks "${EVAL_BLOCKS:-256}" \
  --added-tokens-file "$BASE/fleurs/out/fleurs_mel_added_tokens.json" \
  --audio-dir "$AUDIO" \
  "$@"
