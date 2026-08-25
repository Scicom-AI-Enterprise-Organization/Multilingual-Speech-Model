#!/bin/bash
# Drive the full disk-bounded STT pack: 8 waves (~194 subsets each), pack -> upload
# -> free shards between waves, first-half token JSONs freed before wave 4, final
# index merge + upload at the end. Resumable: finished waves carry a .uploaded marker.
set -e
cd /share/stt
unset LD_LIBRARY_PATH PYTHONPATH
set -a; . ./.env; set +a
export HF_HOME=/share/stt/hf
PY=/share/multipacking/venv/bin/python
OUT=out/multipacking-stt

for w in 0 1 2 3 4 5 6 7; do
  if [ -f $OUT/wave-$w/.uploaded ]; then echo "wave $w already uploaded, skipping"; continue; fi
  if [ "$w" -eq 4 ]; then
    echo "freeing first-half token JSONs"
    find neucodec -mindepth 1 -maxdepth 1 -type d ! -name '.extracted' -exec rm -rf {} +
    df -h /share | tail -1
  fi
  echo "=== wave $w ==="
  $PY multipacking_stt.py --num-waves 8 --wave $w --workers 96
  $PY multipacking_stt.py --stage upload
  find $OUT/wave-$w -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  touch $OUT/wave-$w/.uploaded
  df -h /share | tail -1
done

$PY multipacking_stt.py --stage merge
$PY multipacking_stt.py --stage upload
echo ALL_WAVES_DONE
