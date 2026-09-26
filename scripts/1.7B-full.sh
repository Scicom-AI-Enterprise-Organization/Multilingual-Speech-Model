#!/bin/bash
# Full three-task run: TTS tokens + STT tokens + STT raw mel, one model, one mixture.
#
# Reads the mixture build_mixture.py wrote, so --train_file and num_decay_steps come from
# the packs that actually exist rather than from numbers typed here:
#
#   venv/bin/python preparation/build_mixture.py --json-out /root/share/mixture.json
#
# Batch: 4 x 64 accum x 8 GPUs = 2048 blocks = 21.0M tokens/step -- the published global
# batch, at the micro-batch the 1.7B ablation ran. Do NOT raise it to 8 x 32: a rank
# already peaks near the card limit there, and SOAP's eigendecomposition falls back to
# float32 (linalg_eigh has no bf16 CUDA kernel) and allocates on top.
#
# Optimizer: soap, matrix_lr 1e-3 -- the three-task sweep's winner. --learning_rate is the
# AdamW side of the hybrid only.
#
# Dev losses are reported per task (eval_tts_loss / eval_stt_loss / eval_mel_loss): a
# single mixture-wide loss hides an arm that has stopped learning.
#
#   bash scripts/1.7B-full.sh                  # launch
#   DRY=1 bash scripts/1.7B-full.sh            # print the command and stop
set -e
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH PYTHONPATH
set -a; . ./.env; set +a

MIX="${MIX:-/root/share/mixture.json}"
VOCAB="${VOCAB:-/root/share/vocab/added_tokens_v2.json}"
AUDIO="${AUDIO:-/root/share/audio-root}"
OUT="${OUT:-/root/share/runs/Multilingual-Speech-Qwen3-1.7B-soap}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
NPROC="${NPROC:-8}"
F="${F:-/root/share/fleurs/out}"

[ -f "$MIX" ] || { echo "no mixture at $MIX -- run preparation/build_mixture.py first"; exit 1; }
TRAIN_FILE=$(venv/bin/python -c "import json,sys;print(json.load(open('$MIX'))['train_file'])")
DECAY=$(venv/bin/python -c "import json,sys;print(json.load(open('$MIX'))['num_decay_steps'])")
STEPS=$(venv/bin/python -c "import json,sys;print(json.load(open('$MIX'))['steps'])")
echo "mixture: $(echo "$TRAIN_FILE" | tr ',' '\n' | wc -l) packs | $STEPS steps | decay $DECAY"

# refuse to start into someone else's memory rather than dying 20 minutes in
venv/bin/python - <<'PY'
import sys, torch
need = 100 * 2**30
bad = [(i, torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())
       if torch.cuda.mem_get_info(i)[0] < need]
for i, free in bad:
    print(f'gpu {i}: only {free/2**30:.1f} GiB free')
if bad:
    print('not enough free GPU memory -- something else is holding the cards')
    sys.exit(1)
print(f'all {torch.cuda.device_count()} GPUs have room')
PY

CMD=(venv/bin/python -m torch.distributed.run --nproc_per_node "$NPROC" -m qwen3_mel_adamw
  --model_name_or_path "$MODEL"
  --stt_tokens_file "$VOCAB"
  --audio_dir "$AUDIO"
  --train_file "$TRAIN_FILE"
  --validation_file "tts=$F/fleurs-tts-dev,stt=$F/fleurs-stt-dev,mel=$F/fleurs-mel-dev"
  --max_eval_blocks 256
  --optimizer soap --matrix_lr 1e-3 --learning_rate 1e-4
  --per_device_train_batch_size 4 --gradient_accumulation_steps 64
  --per_device_eval_batch_size 4
  --output_dir "$OUT"
  --bf16 --do_train --do_eval --num_train_epochs 1
  --eval_strategy steps --eval_steps 100
  --logging_steps 1 --warmup_steps 100 --block_size 10240
  --save_steps 250 --save_total_limit 5
  --gradient_checkpointing true --torch_dtype bfloat16
  --ddp_find_unused_parameters false
  --dataloader_num_workers 8 --dataloader_prefetch_factor 4
  --remove_unused_columns false
  --lr_scheduler_type warmup_stable_decay
  --lr_scheduler_kwargs "{\"num_decay_steps\": $DECAY, \"min_lr_ratio\": 1e-1}")

if [ -n "$DRY" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-Multilingual-TTS}"
export WANDB_NAME="${WANDB_NAME:-Qwen3-1.7B-3task-soap}"
export TOKENIZERS_PARALLELISM=false
exec "${CMD[@]}"
