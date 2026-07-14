#!/bin/bash
# Rent a US 4090/5090, health-check CUDA (cuInit), delete + retry if broken.
# Blacklists known-bad host IPs. Writes final result to rent_result.json
SCRATCH=/private/tmp/claude-2078641114/-Users-husein-z-Documents-Multilingual-TTS/2fa1c2e7-1d62-4d50-b946-2f0de72fb4af/scratchpad
LOG=$SCRATCH/rent_healthy.log
RESULT=$SCRATCH/rent_result.json
BLACKLIST=$SCRATCH/bad_hosts.txt
KEY=/Users/husein.z/.runpod/ssh/runpodctl-ssh-key
rm -f "$RESULT"
touch "$BLACKLIST"
grep -q 216.249.100.66 "$BLACKLIST" || echo 216.249.100.66 >> "$BLACKLIST"

try_create() {
  local gpu="$1" cloud="$2" scope="$3"
  if [ "$scope" = "US" ]; then
    runpodctl pod create --name nonverbal-tagging --gpu-id "$gpu" \
      --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
      --container-disk-in-gb 100 --country-code US --ports "22/tcp" \
      --cloud-type "$cloud" 2>&1
  else
    runpodctl pod create --name nonverbal-tagging --gpu-id "$gpu" \
      --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
      --container-disk-in-gb 100 --data-center-ids "$scope" --ports "22/tcp" \
      --cloud-type "$cloud" 2>&1
  fi
}

for machine_attempt in $(seq 1 12); do
  echo "=== machine attempt $machine_attempt $(date +%H:%M:%S) ===" >> "$LOG"
  POD_JSON=""
  for stock_attempt in $(seq 1 40); do
    # primary tier: 4090/5090, country-wide and DC-targeted
    while IFS='|' read -r gpu cloud scope; do
      out=$(try_create "$gpu" "$cloud" "$scope")
      if [[ "$out" != *error* ]]; then POD_JSON="$out"; echo "got $gpu $cloud $scope" >> "$LOG"; break 2; fi
    done <<'TIERS'
NVIDIA GeForce RTX 4090|SECURE|US
NVIDIA GeForce RTX 4090|SECURE|US-IL-1
NVIDIA GeForce RTX 4090|SECURE|US-NC-1
NVIDIA GeForce RTX 4090|SECURE|US-CA-2
NVIDIA GeForce RTX 4090|COMMUNITY|US
NVIDIA GeForce RTX 5090|SECURE|US
NVIDIA GeForce RTX 5090|SECURE|US-IL-1
NVIDIA GeForce RTX 5090|COMMUNITY|US
TIERS
    # fallback tier after ~10 min dry: equivalent-class US GPUs
    if [ "$stock_attempt" -ge 10 ]; then
      while IFS='|' read -r gpu cloud scope; do
        out=$(try_create "$gpu" "$cloud" "$scope")
        if [[ "$out" != *error* ]]; then POD_JSON="$out"; echo "got FALLBACK $gpu $cloud $scope" >> "$LOG"; break 2; fi
      done <<'TIERS'
NVIDIA L40S|SECURE|US
NVIDIA RTX 6000 Ada Generation|SECURE|US
NVIDIA RTX A5000|SECURE|US
NVIDIA L40S|COMMUNITY|US
TIERS
    fi
    echo "no stock, attempt $stock_attempt" >> "$LOG"
    sleep 45
  done
  [ -z "$POD_JSON" ] && { echo "NO STOCK AFTER RETRIES" >> "$LOG"; exit 1; }

  POD_ID=$(echo "$POD_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
  echo "created pod $POD_ID" >> "$LOG"

  IP=""; PORT=""
  for i in $(seq 1 30); do
    DET=$(runpodctl pod get "$POD_ID" 2>/dev/null)
    IP=$(echo "$DET" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('ip',''))" 2>/dev/null)
    PORT=$(echo "$DET" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('port',''))" 2>/dev/null)
    [ -n "$IP" ] && [ -n "$PORT" ] && break
    sleep 10
  done
  if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "no ssh details, deleting $POD_ID" >> "$LOG"
    runpodctl pod delete "$POD_ID" >/dev/null 2>&1
    continue
  fi
  echo "ssh $IP:$PORT" >> "$LOG"

  if grep -q "$IP" "$BLACKLIST"; then
    echo "blacklisted host $IP, deleting $POD_ID" >> "$LOG"
    runpodctl pod delete "$POD_ID" >/dev/null 2>&1
    sleep 30
    continue
  fi

  HEALTH=""
  for i in $(seq 1 20); do
    HEALTH=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" "root@$IP" \
      'python3 -c "
from ctypes import CDLL
rc = CDLL(\"libcuda.so.1\").cuInit(0)
import torch
print(\"HEALTH\", rc, torch.cuda.is_available())
"' 2>/dev/null | grep HEALTH)
    [ -n "$HEALTH" ] && break
    sleep 15
  done
  echo "health: $HEALTH" >> "$LOG"

  if [[ "$HEALTH" == *"HEALTH 0 True"* ]]; then
    echo "{\"pod_id\": \"$POD_ID\", \"ip\": \"$IP\", \"port\": $PORT}" > "$RESULT"
    echo "SUCCESS $POD_ID $IP:$PORT" >> "$LOG"
    exit 0
  fi
  echo "$IP" >> "$BLACKLIST"
  echo "unhealthy machine $IP, deleting $POD_ID" >> "$LOG"
  runpodctl pod delete "$POD_ID" >/dev/null 2>&1
  sleep 20
done
echo "GAVE UP: 12 unhealthy machines" >> "$LOG"
exit 1
