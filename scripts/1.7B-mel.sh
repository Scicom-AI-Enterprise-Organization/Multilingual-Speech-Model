# Three-task iteration: TTS audio tokens + STT audio tokens + STT raw mel.
#
# --train_file takes 'dir:weight' entries; weight = epochs of that dataset per training
# epoch, so the mix is tuned here and nothing is repacked. The mel pack is far smaller
# than the token packs, so it is weighted up rather than left to drown.
#
# --stt_tokens_file pins <|STT|> + the language tokens to the ids the STT pack froze:
#   huggingface-cli download --repo-type dataset \
#     Scicom-intl/Multilingual-STT-multipacking-10k stt_added_tokens.json
#
# --optimizer/--matrix_lr default to the three-task FLEURS sweep's winner (soap 1e-3, which
# beat muon 1e-2 and diverged AdamW at 1e-3). --learning_rate is the AdamW side of the
# hybrid: the 2D hidden weights run on SOAP at --matrix_lr, embeddings/lm_head/norms on
# AdamW at --learning_rate. Logged `learning_rate` is the SOAP group.
#
# num_decay_steps must be ~10% of the total optimizer steps for this mixture -- 243 below is
# a leftover from the 100-step sweep. Steps = sum(blocks x weight) / (8 GPUs x 8 x 32).
#
# --audio_dir is the extracted audio tree the mel pack's paths point into: that pack
# stores paths, not audio, and the dataloader workers decode + resample on the fly.
#
# Fewer, shallower dataloader prefetches than the token-only scripts: a mel block holds
# ~185s of audio, which is ~12MB of float32 waveform once decoded (vs 40KB of token ids).
# More workers too, since they now do file reads + resampling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Multilingual-TTS" \
WANDB_NAME="Qwen3-1.7B-mel-soap" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m qwen3_mel_adamw \
--model_name_or_path "Qwen/Qwen3-1.7B-Base" \
--stt_tokens_file "gfs/01be5b33/stt_added_tokens.json" \
--audio_dir "gfs/01be5b33/audio" \
--optimizer soap \
--matrix_lr 1e-3 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 32 \
--output_dir gfs/01be5b33/Multilingual-TTS-Qwen3-1.7B-mel-soap \
--bf16 --do_train --do_eval false --num_train_epochs 1 \
--train_file "gfs/01be5b33/multipacking-tts:1.0,gfs/01be5b33/multipacking-stt:1.0,gfs/01be5b33/multipacking-stt-mel:2.0" \
--logging_steps 1 \
--learning_rate 1e-4 \
--warmup_steps 100 \
--block_size 10240 \
--save_steps 500 \
--save_total_limit 10 \
--gradient_checkpointing true \
--torch_dtype float32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 8 \
--dataloader_prefetch_factor 4 \
--remove_unused_columns false \
--lr_scheduler_type "warmup_stable_decay" \
--lr_scheduler_kwargs '{"num_decay_steps": 243, "min_lr_ratio": 1e-1}'
