# Multilingual-TTS

Open-source multilingual TTS with Voice Cloning support for 150+ languages, built on [Neucodec](https://github.com/neuphonic/neucodec) as the speech tokenizer at 50 TPS.

## Models

| Model | Link | Purpose |
|---|---|---|
| Multilingual-TTS-0.6B-Base | [🤗](https://huggingface.co/Scicom-intl/Multilingual-TTS-0.6B-Base) | Base |
| Multilingual-TTS-1.7B-Base | [🤗](https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base) | Base |
| Multilingual-Expressive-TTS-0.6B | [🤗](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-0.6B) | Post-training TTS |
| Multilingual-Expressive-TTS-1.7B | [🤗](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-1.7B) | Post-training TTS |
| Multilingual-VC-0.6B | [🤗](https://huggingface.co/Scicom-intl/Multilingual-Expressive-VC-0.6B) | Post-training VC |
| Multilingual-VC-1.7B | [🤗](https://huggingface.co/Scicom-intl/Multilingual-Expressive-VC-1.7B) | Post-training VC |

## Evaluation

### [TTS Evaluation](tts-evaluation/README.md)

CER and MOS across 76 languages, compared against: Dia TTS, Orpheus, Chatterbox (23 languages), Fish Audio S2 Pro, Qwen3 TTS (11 languages).

### [VC Evaluation](vc-evaluation/README.md)

Speaker similarity and CER across 76 languages, compared against: Dia TTS, Orpheus, Chatterbox (23 languages), Fish Audio S2 Pro.

## Dataset

### Base

**Sources**

1. https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS
2. https://huggingface.co/datasets/Scicom-intl/Emilia-YODAS-Voice-Conversion
3. https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia

**Size**

1. Multi-speaker multilingual Voice Cloning — **up to 35.88B tokens**
2. Multi-speaker multilingual TTS, 150+ languages — **up to 25.35B tokens**

**Preparation:** [preparation](preparation)

### Expressive TTS

**Sources**

1. https://huggingface.co/datasets/Scicom-intl/ExpressiveSpeech

**Size**

1. Multi-speaker multilingual Expressive TTS — **up to 1.15B tokens**

**Preparation:** [synthetic-description](synthetic-description)

### Voice Cloning

**Sources**

1. https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia
2. https://huggingface.co/datasets/Scicom-intl/Multilingual-TTS-Voice-Conversion

**Size**

1. Multi-speaker multilingual Voice Cloning — **up to 47.65B tokens**

## Ablation

### One Epoch

1. Global token size: 10240 × 256 × 8 GPUs ≈ 20,971,520 tokens
2. Warmup: 100 steps
3. FP32-BF16 mixed precision
4. Compared AdamW with WSD LR vs Muon + AdamW with WSD LR (decay = 10% of dataset)
5. Run on Qwen3 1.7B Base only
6. **AdamW performed better**

<img src="one-epoch.png" width="50%">

### Hyperparameter Search

One-epoch results used conservative learning rates, so we ran a focused search:

1. Global token size: 10240 × 256 × 8 GPUs ≈ 20,971,520 tokens
2. 100 steps, warmup 50 steps
3. FP32-BF16 mixed precision
4. Run on Qwen3 1.7B Base only
5. Grid search over AdamW LR, Muon LR, and decay rate — [hyperparameter_search.py](hyperparameter_search.py)
6. Aggressive LR sweep — [hyperparameter_search_extra.py](hyperparameter_search_extra.py) — turned out best
7. AdamW-only with the same aggressive LR — [1.7B-adamw-aggresive.sh](1.7B-adamw-aggresive.sh)
8. **Adding Muon performed better**

<img src="hyperparameter-search.png" width="50%">

## Training

### Base

```bash
# 0.6B
bash 0.6B.sh

# 1.7B
bash 1.7B.sh
```

### Expressive TTS

```bash
bash 1.7B-expressive.sh
```

## WandB

- Base experiments: https://wandb.ai/aies-scicom-scicom-ai/Multilingual-TTS
- Post TTS experiments: https://wandb.ai/aies-scicom-scicom-ai/Multilingual-TTS-Expressive

## Acknowledgements

1. [Malaysia-AI](https://huggingface.co/malaysia-ai) for the large-scale TTS dataset: https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS
2. [Scitix](https://www.scitix.ai/) for H100 node access
