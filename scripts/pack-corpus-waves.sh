#!/bin/bash
# Pack the corpus into TTS + STT + mel, one wave at a time.
#
# Each wave: fetch that wave's token zips, extract, pack all three tasks from the one
# extraction, then delete the JSONs before the next wave starts. Waves are sized by
# inodes rather than bytes — the whole corpus is ~121.8M token JSONs against ~98M free.
#
# Resumable: a wave whose pack summary exists is skipped.
#
#   bash scripts/pack-corpus-waves.sh            # all 8
#   WAVES="0 1" bash scripts/pack-corpus-waves.sh
set -e
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH PYTHONPATH
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/root/share/hf}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

BASE="${BASE:-/root/share/corpus}"
VOCAB="${VOCAB:-/root/share/vocab/added_tokens.json}"
NUM_WAVES="${NUM_WAVES:-8}"
WAVES="${WAVES:-0 1 2 3 4 5 6 7}"
WORKERS="${WORKERS:-96}"
PY=venv/bin/python

if [ ! -d "$BASE/meta" ]; then
  echo "=== building corpus metadata (once) ==="
  $PY preparation/multipacking_corpus.py --base-dir "$BASE" --stage meta
fi

for w in $WAVES; do
  if [ -f "$BASE/out/summary-wave-$w.json" ]; then
    echo "wave $w already packed, skipping"
    continue
  fi
  echo "=== wave $w/$NUM_WAVES: fetching token zips ==="
  $PY preparation/multipacking_corpus.py --base-dir "$BASE" --num-waves "$NUM_WAVES" \
      --wave "$w" --stage tokens --download-workers 8
  echo "=== wave $w: packing tts + stt + mel ==="
  $PY preparation/multipacking_corpus.py --base-dir "$BASE" --num-waves "$NUM_WAVES" \
      --wave "$w" --stage pack --task all --workers "$WORKERS" --added-tokens-file "$VOCAB"
  echo "=== wave $w: dropping this wave's token JSONs ==="
  $PY preparation/multipacking_corpus.py --base-dir "$BASE" --num-waves "$NUM_WAVES" \
      --wave "$w" --stage clean
  df -h / | tail -1
done
echo CORPUS_PACKS_DONE
