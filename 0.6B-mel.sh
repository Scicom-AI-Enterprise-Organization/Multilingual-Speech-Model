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
# num_decay_steps should be ~10% of the total optimizer steps for this mixture.
#
# --audio_dir is the extracted audio tree the mel pack's paths point into: that pack
# stores paths, not audio, and the dataloader workers decode + resample on the fly.
#
# Fewer, shallower dataloader prefetches than the token-only scripts: a mel block holds
# ~185s of audio, which is ~12MB of float32 waveform once decoded (vs 40KB of token ids).
# More workers too, since they now do file reads + resampling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Multilingual-TTS" \
WANDB_NAME="Qwen3-0.6B-float32-mel-adamw" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m qwen3_mel_adamw \
--model_name_or_path "Qwen/Qwen3-0.6B-Base" \
--stt_tokens_file "gfs/01be5b33/stt_added_tokens.json" \
--audio_dir "gfs/01be5b33/audio" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 32 \
--output_dir gfs/01be5b33/Multilingual-TTS-Qwen3-0.6B-float32-mel-adamw \
--bf16 --do_train --do_eval false --num_train_epochs 1 \
--train_file "gfs/01be5b33/multipacking-tts:1.0,gfs/01be5b33/multipacking-stt:1.0,gfs/01be5b33/multipacking-stt-mel:2.0" \
--logging_steps 1 \
--learning_rate 1e-3 \
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
