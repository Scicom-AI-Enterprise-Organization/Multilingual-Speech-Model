#!/bin/bash
# Runs after the 8 corpus waves finish: re-pack FLEURS and CV22 against the current
# vocabulary, then print the mixture.
#
# Why both, when the corpus covers most sources: the corpus metadata carries the Common
# Voice 22 rows but NOT their tokens (those live in malaysia-ai/common_voice_22_0), so the
# corpus packer skips them — ~677K `missing_token` per wave, ~5.4M against CV22's 6.9M
# filtered rows. The CV22 packs are therefore the only CV22 coverage, and they were built
# against the old vocabulary, so their language-tag and mel ids would land on the wrong
# tokens. FLEURS is not in the corpus at all.
#
# Waits for the wave loop rather than assuming it is done.
set -e
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH PYTHONPATH
set -a; . ./.env; set +a
export HF_HOME="${HF_HOME:-/root/share/hf}"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false

WAVE_LOG="${WAVE_LOG:-/root/share/multilingual-tts/pack-corpus.log}"
VOCAB="${VOCAB:-/root/share/vocab/added_tokens_v2.json}"
FLEURS_AUDIO="${FLEURS_AUDIO:-/root/share/fleurs-work/audio}"
WORKERS="${WORKERS:-96}"
MAX_WAIT="${MAX_WAIT:-57600}"          # 16h, then give up rather than hang forever
PY=venv/bin/python

waited=0
while ! grep -q CORPUS_PACKS_DONE "$WAVE_LOG" 2>/dev/null; do
  # A gone python process is not proof of failure: the loop exits the last wave, runs clean,
  # and only then echoes the marker. Polling inside that window once bailed out on a finished
  # run. Give the marker a grace period and re-check before giving up.
  if ! pgrep -f 'multipacking_corpu[s]' >/dev/null && [ "$waited" -gt 600 ]; then
    sleep 60
    grep -q CORPUS_PACKS_DONE "$WAVE_LOG" 2>/dev/null && break
    pgrep -f 'multipacking_corpu[s]' >/dev/null && continue
    echo "wave loop is not running and never printed CORPUS_PACKS_DONE - stopping"; exit 1
  fi
  [ "$waited" -ge "$MAX_WAIT" ] && { echo "timed out waiting for the waves"; exit 1; }
  sleep 120; waited=$((waited + 120))
done
echo "=== waves done after ${waited}s of waiting; re-packing against $VOCAB ==="

echo "=== FLEURS: tts + stt + mel, train ==="
$PY preparation/multipacking_fleurs.py --base-dir /root/share/fleurs --stage pack \
    --task all --workers "$WORKERS" --added-tokens-file "$VOCAB" --audio-base "$FLEURS_AUDIO"
echo "=== FLEURS: dev ==="
$PY preparation/multipacking_fleurs.py --base-dir /root/share/fleurs --stage pack \
    --task all --splits dev --workers "$WORKERS" --added-tokens-file "$VOCAB" \
    --audio-base "$FLEURS_AUDIO"

echo "=== CV22: tts + stt + mel, train ==="
$PY preparation/multipacking_cv22.py --base-dir /root/share/cv22 --stage pack \
    --task all --workers "$WORKERS" --added-tokens-file "$VOCAB"
echo "=== CV22: dev ==="
$PY preparation/multipacking_cv22.py --base-dir /root/share/cv22 --stage pack \
    --task all --splits dev --workers "$WORKERS" --added-tokens-file "$VOCAB"

echo "=== mixture ==="
$PY preparation/build_mixture.py --json-out /root/share/mixture.json
echo REPACK_DONE
