export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Multilingual-TTS-Expressive" \
WANDB_NAME="Qwen3-1.7B-float32-1.0" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m qwen3_muonadamw \
--model_name_or_path "Qwen/Qwen3-1.7B-Base" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 16 \
--output_dir gfs/01be5b33/Multilingual-TTS-Qwen3-1.7B-float32-expressive \
--bf16 --do_train --do_eval false --num_train_epochs 3 \
--train_file "gfs/01be5b33/combine-multipacking-expressive" \
--logging_steps 1 \
--learning_rate 1e-3 \
--warmup_steps 50 \
--block_size 10240 \
--save_steps 50 \
--save_total_limit 10 \
--gradient_checkpointing true \
--torch_dtype float32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 20 \
--remove_unused_columns false \
--lr_scheduler_type "warmup_stable_decay" \
--lr_scheduler_kwargs '{"num_decay_steps": 243, "min_lr_ratio": 1e-1, "lr": 1e-4, "lr_muon": 1e-3, "decay": 0.01}'