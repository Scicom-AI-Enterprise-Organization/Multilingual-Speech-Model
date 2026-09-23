export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Multilingual-TTS" \
WANDB_NAME="Qwen3-1.7B-float32" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m qwen3_adamw_old \
--model_name_or_path "Qwen/Qwen3-1.7B-Base" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--output_dir gfs/01be5b33/Multilingual-TTS-Qwen3-1.7B \
--bf16 --do_train --do_eval false --num_train_epochs 3 \
--train_file "gfs/01be5b33/combine-multipacking" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 50 \
--block_size 10240 \
--save_steps 500 \
--save_total_limit 10 \
--gradient_checkpointing true \
--torch_dtype float32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 20 \
--remove_unused_columns false