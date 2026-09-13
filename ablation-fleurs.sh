#!/bin/bash
# Optimizer search (README "Optimizer search") over the FLEURS-R ablation mixture.
#
# One run trains on all three packs at once — TTS audio tokens, STT audio tokens and STT
# raw mel, at their natural proportions — so every optimizer is judged on the model the
# project actually wants: one that speaks, listens through the codec, and listens through
# raw mel. The three tasks are then scored on their own dev splits, reported separately as
# eval_tts_loss / eval_stt_loss / eval_mel_loss, and plotted separately by
# plot_fleurs_ablation.py. They share no axis: the TTS task predicts 65k-way speech
# tokens, the two STT tasks predict text.
#
# Runs are ranked on the mean of the three dev losses, so no single task's scale decides
# the order alone.
#
# Same protocol as the published search: Qwen3-1.7B-Base, 100 steps, warmup 50, FP32-BF16
# mixed precision, WSD LR, one grid per optimizer. Only the global token size differs: the
# published search ran 10,240 x 256 x 8 GPUs = 21M tokens/step, which this corpus would
# replay many times over inside one run. 1 x 6 x 8 GPUs = 48 blocks = 491k tokens/step.
#
#   bash ablation-fleurs.sh                        # full grid (16 runs)
#   bash ablation-fleurs.sh --optimizers muon soap # subset
#   bash ablation-fleurs.sh --dry-run              # print commands only
set -e
cd "$(dirname "$0")"
unset LD_LIBRARY_PATH PYTHONPATH        # login shells poison the venv's torch (see CLAUDE.md)
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/share/multilingual-tts/hf}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

BASE="${BASE:-/share/multilingual-tts}"
PACKS="$BASE/fleurs/out"
# the mel pack stores audio paths; this is the root they resolve against
AUDIO="${AUDIO:-/share/fleurs-work}"

"$BASE/venv/bin/python" hyperparameter_search.py \
  --train-file "$PACKS/fleurs-tts,$PACKS/fleurs-stt,$PACKS/fleurs-mel" \
  --validation-file "tts=$PACKS/fleurs-tts-dev,stt=$PACKS/fleurs-stt-dev,mel=$PACKS/fleurs-mel-dev" \
  --added-tokens-file "$PACKS/fleurs_mel_added_tokens.json" \
  --audio-dir "$AUDIO" \
  --run-prefix fleurs \
  --output-root "$BASE/runs/fleurs" \
  --state-dir "$BASE/search_state/fleurs" \
  --wandb-project "${WANDB_PROJECT:-Multilingual-TTS}" \
  --nproc "${NPROC:-8}" \
  --batch-size "${BS:-1}" \
  --grad-accum "${ACCUM:-6}" \
  --steps "${STEPS:-100}" \
  --warmup "${WARMUP:-50}" \
  --num-decay-steps "${DECAY:-243}" \
  --eval-steps "${EVAL_STEPS:-25}" \
  --max-eval-blocks "${EVAL_BLOCKS:-256}" \
  "$@"
