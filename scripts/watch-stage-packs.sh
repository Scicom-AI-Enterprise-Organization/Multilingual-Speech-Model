#!/bin/bash
# Keep local disk from filling while the 8 corpus waves run: every few minutes, move any
# wave that has finished (its summary.json exists) onto the NFS filer and symlink it back.
# Exits once the wave loop prints CORPUS_PACKS_DONE.
cd "$(dirname "$0")/.."
LOG="${LOG:-/root/share/multilingual-tts/pack-corpus.log}"
INTERVAL="${INTERVAL:-300}"
while true; do
  bash scripts/stage-packs-to-nfs.sh || echo "staging pass failed, will retry"
  if grep -q CORPUS_PACKS_DONE "$LOG" 2>/dev/null; then
    echo "wave loop finished; final staging pass done"
    exit 0
  fi
  sleep "$INTERVAL"
done
