# Multilingual-TTS

Building actual open source including dataset multilingual TTS more than 150 languages with Voice Conversion.

## Dataset 

### Source

1. https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS
2. https://huggingface.co/datasets/Scicom-intl/Emilia-YODAS-Voice-Conversion
3. https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia

### Size

1. Use [neucodec](https://github.com/neuphonic/neucodec) as speech tokenizer, 50 TPS, output in 24k sample rate.
2. Multi-speaker multilingual Voice Conversion, **up to 38.96B tokens**.
3. Multi-speaker multilingual TTS more than 150 languages, **up to 12.07B tokens**.

### Preparation

All steps to reproduce in [preparation](preparation).

## Ablation

1. Use approximate of 10240 * 128 * 8 GPUs global token size, ~10,485,760 tokens.
2. 1 epoch.
3. learning rate 1e-4.
4. Warmup step is 100.
5. Compare AdamW with constant learning rate, AdamW with linear decay learning rate, AdamW with WSD learning rate, Muon + AdamW with WSD learning rate, where WSD number decay step is 10% of the dataset.
6. Only done on Qwen3 1.7B Base.