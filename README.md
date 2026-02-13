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
3. Multi-speaker multilingual TTS more than 150 languages

### Preparation

All steps to reproduce in [preparation](preparation).

## Ablation

### One Epoch

1. Use approximate of 10240 * 256 * 8 GPUs global token size, ~20,971,520 tokens.
3. Warmup step is 100.
4. Compare AdamW with WSD learning rate, Muon + AdamW with WSD learning rate, where WSD number decay step is 10% of the dataset.
5. Only done on Qwen3 1.7B Base.

<img src="one-epoch.png" width="50%">

### Hyperparameter search

1. Use approximate of 10240 * 256 * 8 GPUs global token size, ~20,971,520 tokens.
2. 100 steps only.
3. Warmup step is 50.
4. Only done on Qwen3 1.7B Base.
5. Permute search on LR for AdamW, LR for Muon and decay rate, [hyperparameter_search.py](hyperparameter_search.py)
