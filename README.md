# Multilingual-TTS

Building actual open source including dataset Multilingual TTS more than 150 languages with Voice Conversion.

## Dataset size

1. Use [neucodec](https://github.com/neuphonic/neucodec) as speech tokenizer, 50 TPS, output in 24k sample rate.
2. Multi-speaker multilingual Voice Conversion, **up to 38.96B tokens**.
3. Multi-speaker multilingual TTS more than 150 languages, **up to 12.07B tokens**.

## Ablation

1. Use approximate of 10240 * 128 global token size.
2. 1 epoch.
3. learning rate 1e-4.
4. Warmup step is 100.
5. Compare AdamW with linear learning rate, AdamW with WSD, Muon + AdamW with WSD, where WSD number decay step is 10% of the dataset.
6. Only done on Qwen3 1.7B Base.
