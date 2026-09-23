#!/bin/bash
# Optimizer search (README "Optimizer search") at the published global batch, over the
# FLEURS-R + filtered Common Voice 22 mixture.
#
# One run trains on all six packs at once — TTS audio tokens, STT audio tokens and STT
# raw mel, from both corpora — so every optimizer is judged on the model the project
# actually wants: one that speaks, listens through the codec, and listens through raw
# mel. Each task is then scored on its own dev split and reported separately
# (eval_fleurs_tts_loss, eval_cv22_mel_loss, …), and plotted per task by
# plot_fleurs_ablation.py. Losses never share an axis across tasks: TTS predicts
# 65k-way speech tokens, the STT tasks predict text.
#
# Batch is the published one: 256 blocks/GPU x 8 GPUs = 2048 x 10,240 = 21M tokens/step.
# It is reached with micro-batch 4 and 64 accumulation steps rather than 8 x 32 — at
# micro-batch 8 a rank peaks at 143.1GB of the 143.8GB card, which leaves nothing for
# the tenant sharing GPUs 6-7; micro-batch 4 peaks ~11GB lower and costs ~1% more time.
#
# 200 steps rather than 100: in the published search the winning run only separates
# after ~step 50 and is still descending at 100 (hyperparameter-search.png).
#
# Runs are namespaced by PREFIX (default fleurs-b2048, the 2048-block batch) so a sweep
# at one batch size never reuses another's resume markers — the run names are otherwise
# identical strings and the harness would skip every run as already done.
#
#   bash ablation-fleurs.sh                        # the 6-config grid
#   bash ablation-fleurs.sh --optimizers muon      # subset
#   bash ablation-fleurs.sh --dry-run              # print commands only
set -e
cd "$(dirname "$0")/.."   # run from the repo root
unset LD_LIBRARY_PATH PYTHONPATH        # login shells poison the venv's torch (see CLAUDE.md)
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/share/multilingual-tts/hf}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

BASE="${BASE:-/share/multilingual-tts}"
F="$BASE/fleurs/out"                       # FLEURS packs
C="${CV22:-/share/cv22/out}"               # Common Voice 22 packs
# mel packs store audio paths; this root holds both corpora's audio (symlinked per
# language, which never collide: FLEURS uses en_us-style tags, CV22 plain en)
AUDIO="${AUDIO:-/share/audio-root}"

"$BASE/venv/bin/python" hyperparameter_search.py \
  --train-file "$F/fleurs-tts,$F/fleurs-stt,$F/fleurs-mel,$C/cv22-tts,$C/cv22-stt,$C/cv22-mel" \
  --validation-file "fleurs_tts=$F/fleurs-tts-dev,fleurs_stt=$F/fleurs-stt-dev,fleurs_mel=$F/fleurs-mel-dev,cv22_tts=$C/cv22-tts-dev,cv22_stt=$C/cv22-stt-dev,cv22_mel=$C/cv22-mel-dev" \
  --added-tokens-file "${TOKENS:-/share/cv22/out/added_tokens.json}" \
  --audio-dir "$AUDIO" \
  --grid-json "${GRID:-scripts/ablation-fleurs-grid.json}" \
  --run-prefix "${PREFIX:-fleurs-b2048}" \
  --output-root "$BASE/runs/${PREFIX:-fleurs-b2048}" \
  --state-dir "$BASE/search_state/${PREFIX:-fleurs-b2048}" \
  --wandb-project "${WANDB_PROJECT:-Multilingual-TTS}" \
  --nproc "${NPROC:-8}" \
  --batch-size "${BS:-4}" \
  --grad-accum "${ACCUM:-64}" \
  --steps "${STEPS:-200}" \
  --warmup "${WARMUP:-50}" \
  --num-decay-steps "${DECAY:-243}" \
  --eval-steps "${EVAL_STEPS:-25}" \
  --max-eval-blocks "${EVAL_BLOCKS:-256}" \
  "$@"
