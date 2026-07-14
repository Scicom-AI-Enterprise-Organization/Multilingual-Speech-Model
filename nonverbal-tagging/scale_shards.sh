#!/bin/bash
# Process Malaysian-Emilia shards 1..9: download -> mine -> verify+tag -> upload -> clean.
# Each shard's outputs upload to HF before the next starts, so progress is durable.
set -o pipefail
REPO=Scicom-intl/Malaysian-Emilia-Nonverbal-Tags
export HF_HUB_ENABLE_HF_TRANSFER=1

# shard 0: re-render with the two-format (tagged_text + nv_text) script.
# Audio and events.jsonl are already on disk; skips mining.
echo "=== SHARD 0 RERENDER START $(date -u +%H:%M:%S) ==="
OUT0=/root/out/shard0r
rm -rf "$OUT0"; mkdir -p "$OUT0"
if python3 /root/pipeline/verify_and_tag.py --events /root/out/events.jsonl \
     --out-dir "$OUT0" 2>&1 | tail -20; then
  hf upload "$REPO" "$OUT0/tagged.parquet" "data/tagged-00000.parquet" --repo-type dataset
  echo "=== SHARD 0 DONE $(date -u +%H:%M:%S) ==="
else
  echo "SHARD 0 RERENDER FAILED"
fi
rm -rf /root/data/audio

for i in $(seq 1 9); do
  echo "=== SHARD $i START $(date -u +%H:%M:%S) ==="
  ZIP=output-audio_trim-$i-0.zip
  AUDIO=/root/data/audio_shard
  OUT=/root/out/shard$i
  rm -rf "$AUDIO" "$OUT"
  mkdir -p "$AUDIO" "$OUT"

  hf download Scicom-intl/Malaysian-Emilia --repo-type dataset \
    --include "$ZIP" --local-dir /root/data/malaysian-emilia || { echo "SHARD $i DOWNLOAD FAILED"; continue; }
  unzip -q -o "/root/data/malaysian-emilia/$ZIP" -d "$AUDIO/" || { echo "SHARD $i UNZIP FAILED"; continue; }
  rm -f "/root/data/malaysian-emilia/$ZIP"
  df -h / | tail -1

  python3 /root/pipeline/mine_events.py --audio-dir "$AUDIO" \
    --out "$OUT/events.jsonl" --thresh-scale 0.3 2>&1 | tail -2 \
    || { echo "SHARD $i MINE FAILED"; continue; }

  python3 /root/pipeline/verify_and_tag.py --events "$OUT/events.jsonl" \
    --out-dir "$OUT" 2>&1 | tail -20 \
    || { echo "SHARD $i VERIFY FAILED"; continue; }

  printf -v N "%05d" "$i"
  hf upload "$REPO" "$OUT/tagged.parquet" "data/tagged-$N.parquet" --repo-type dataset
  hf upload "$REPO" "$OUT/events.jsonl" "events/events-$N.jsonl" --repo-type dataset
  [ -d "$OUT/qa_crops" ] && hf upload "$REPO" "$OUT/qa_crops" "qa_crops/shard-$i" --repo-type dataset
  rm -rf "$AUDIO"
  echo "=== SHARD $i DONE $(date -u +%H:%M:%S) ==="
done
echo "ALL SHARDS DONE"
