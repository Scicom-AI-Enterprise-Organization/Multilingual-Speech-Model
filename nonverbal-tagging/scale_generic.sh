#!/bin/bash
# Generic per-shard tagging loop.
# Usage: scale_generic.sh <src_repo> <out_repo> <meta_glob> <zip1> [zip2 ...]
set -o pipefail
SRC=$1; OUT_REPO=$2; META_GLOB=$3; shift 3
export HF_HUB_ENABLE_HF_TRANSFER=1

n=0
for ZIP in "$@"; do
  printf -v N "%05d" "$n"
  echo "=== SHARD $n ($ZIP) START $(date -u +%H:%M:%S) ==="
  AUDIO=/root/data/audio_shard
  OUT=/root/out/gshard$n
  rm -rf "$AUDIO" "$OUT"
  mkdir -p "$AUDIO" "$OUT"

  hf download "$SRC" --repo-type dataset --include "$ZIP" \
    --local-dir /root/data/src || { echo "SHARD $n DOWNLOAD FAILED"; n=$((n+1)); continue; }
  unzip -q -o "/root/data/src/$ZIP" -d "$AUDIO/" || { echo "SHARD $n UNZIP FAILED"; n=$((n+1)); continue; }
  rm -f "/root/data/src/$ZIP"
  df -h / | tail -1

  python3 /root/pipeline/mine_events.py --audio-dir "$AUDIO" \
    --out "$OUT/events.jsonl" --thresh-scale 0.3 2>&1 | tail -2 \
    || { echo "SHARD $n MINE FAILED"; n=$((n+1)); continue; }

  python3 /root/pipeline/verify_and_tag.py --events "$OUT/events.jsonl" \
    --out-dir "$OUT" --meta-parquet-glob "$META_GLOB" 2>&1 | tail -20 \
    || { echo "SHARD $n VERIFY FAILED"; n=$((n+1)); continue; }

  hf upload "$OUT_REPO" "$OUT/tagged.parquet" "data/tagged-$N.parquet" --repo-type dataset
  hf upload "$OUT_REPO" "$OUT/events.jsonl" "events/events-$N.jsonl" --repo-type dataset
  [ -d "$OUT/qa_crops" ] && hf upload "$OUT_REPO" "$OUT/qa_crops" "qa_crops/shard-$n" --repo-type dataset
  rm -rf "$AUDIO"
  echo "=== SHARD $n DONE $(date -u +%H:%M:%S) ==="
  n=$((n+1))
done
echo "REPO DONE $OUT_REPO"
