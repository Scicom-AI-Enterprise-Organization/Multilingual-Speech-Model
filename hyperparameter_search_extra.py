import os
import json

lr = [1e-3]
lr_muon = [1e-2]
decay = [0.01]

command = """
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Multilingual-TTS" \
WANDB_NAME="Qwen3-1.7B-float32-muonadamw-wsdlr-search-{lr_}-{lr_muon_}-{decay_}" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m qwen3_muonadamw_search \
--model_name_or_path "Qwen/Qwen3-1.7B-Base" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 32 \
--output_dir gfs/01be5b33/Multilingual-TTS-Qwen3-1.7B-float32-muonadamw-wsdlr-search-{lr_}-{lr_muon_}-{decay_} \
--bf16 --do_train --do_eval false --max_steps 100 \
--train_file "gfs/01be5b33/combine-multipacking" \
--logging_steps 1 \
--learning_rate {lr_} \
--warmup_steps 50 \
--block_size 10240 \
--save_steps 500 \
--save_total_limit 10 \
--gradient_checkpointing true \
--torch_dtype float32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 20 \
--remove_unused_columns false \
--lr_scheduler_type "warmup_stable_decay" \
--lr_scheduler_kwargs '{{"num_decay_steps": 243, "min_lr_ratio": 1e-1, "lr": {lr_}, "lr_muon": {lr_muon_}, "decay": {decay_}}}'
""".strip()

commands = []
for lr_ in lr:
    for lr_muon_ in lr_muon:
        for decay_ in decay:
            cmd = command.format(lr_=lr_, lr_muon_=lr_muon_, decay_=decay_)
            commands.append(cmd)

for i, cmd in enumerate(commands):
    filename_done = f'donehyperparameter_extra_{i}.json'
    try:
        with open(filename_done) as fopen:
            json.load(fopen)
            continue
    except:
        pass

    print(f"Running experiment {i+1}/{len(commands)}")
    print(cmd)
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"Experiment {i} failed with exit code {exit_code}")
        break
    
    with open(filename_done, 'w') as fopen:
        json.dump('done', fopen)