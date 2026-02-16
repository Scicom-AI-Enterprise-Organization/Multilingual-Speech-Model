# Multilingual-TTS

Building actual open source including dataset multilingual TTS more than 150 languages with Voice Conversion.

## Dataset 

### Source

1. https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS
2. https://huggingface.co/datasets/Scicom-intl/Emilia-YODAS-Voice-Conversion
3. https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia

### Size

1. Use [neucodec](https://github.com/neuphonic/neucodec) as speech tokenizer, 50 TPS, output in 24k sample rate.
2. Multi-speaker multilingual Voice Conversion, **up to 35.88B tokens**.
3. Multi-speaker multilingual TTS more than 150 languages, **up to 14.64B tokens**.

### Preparation

All steps to reproduce in [preparation](preparation).

## Ablation

### One Epoch

1. Use approximate of 10240 * 256 * 8 GPUs global token size, ~20,971,520 tokens.
3. Warmup step is 100.
4. Compare AdamW with WSD learning rate, Muon + AdamW with WSD learning rate, where WSD number decay step is 10% of the dataset.
5. Only done on Qwen3 1.7B Base.
6. AdamW performed better.

<img src="one-epoch.png" width="50%">

### Hyperparameter search

But we not satisfied with one epoch ablation due to learning rates are not aggressive enough.

1. Use approximate of 10240 * 256 * 8 GPUs global token size, ~20,971,520 tokens.
2. 100 steps only.
3. Warmup step is 50.
4. Only done on Qwen3 1.7B Base.
5. Permute search on LR for AdamW, LR for Muon and decay rate, [hyperparameter_search.py](hyperparameter_search.py)
6. We run aggresive LR, [hyperparameter_search_extra.py](hyperparameter_search_extra.py), turns out its the best.
7. We compare using AdamW only using the same aggresive LR as (6), [1.7B-adamw-aggresive.sh](1.7B-adamw-aggresive.sh).
8. Adding Muon performed better.

<img src="hyperparameter-search.png" width="50%">

## Continue Pretraining Base

### 1.7B

```bash
bash 1.7B.sh
```

### 4B

```bash
bash 4B.sh
```

## WanDB

All experiments at https://wandb.ai/aies-scicom-scicom-ai/Multilingual-TTS
