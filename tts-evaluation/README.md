# TTS Evaluation

Benchmarking multilingual TTS models across 76 languages using Character Error Rate (CER) and MOS based on UTMOSv2.

![Benchmark scatter](scatter_results.png)

## Models

| Model | Description |
|-------|-------------|
| **Dia TTS** | [Nari Labs Dia TTS](https://github.com/nari-labs/dia) |
| **Multilingual TTS 0.6B** | [Scicom-intl/Multilingual-Expressive-TTS-0.6B](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-0.6B) |
| **Multilingual TTS 1.7B** | [Scicom-intl/Multilingual-Expressive-TTS-1.7B](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-1.7B) |
| **Orpheus** | [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) |
| **Chatterbox** | [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) — 23 languages only |
| **FishSpeech2** | [FishSpeech 2](https://github.com/fishaudio/fish-speech) |
| **Qwen3 TTS** | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — 11 languages only |

## Setup

```bash
pip install -r requirements.txt
```

## Run Generations

Each prompt is generated **twice** and the scores are averaged to reduce variance. We also upload all the generations done by us at [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC)

### Dia TTS

```bash
python3 dia_tts.py --output 'dia-tts'
```

### Scicom Multilingual TTS

```bash
# 0.6B
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'

# 1.7B
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'
```

### Orpheus

```bash
python3 orpheus.py --output 'orpheus'
```

### Chatterbox

```bash
python3 chatterbox.py --output 'chatterbox'
```

### FishSpeech2

```bash
python3 fishspeech2.py --output 'fishspeech2'
```

### Qwen3 TTS

```bash
python3 qwen3_tts.py --output 'qwen3_tts'
```

## Evaluate

### CER

```bash
python3 calculate_cer.py --output_folder "dia-tts"               --output "dia-tts-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
python3 calculate_cer.py --output_folder "orpheus"               --output "orpheus-cer"
python3 calculate_cer.py --output_folder "chatterbox"            --output "chatterbox-cer"
python3 calculate_cer.py --output_folder "fishspeech2"           --output "fishspeech2-cer"
python3 calculate_cer.py --output_folder "qwen3_tts"             --output "qwen3_tts-cer"
```

## Results

Summary across all evaluated languages. Full per-language heatmap below.

| Model | Languages | CER ↓ |
|-------|:---------:|:-----:|
| Dia TTS | 76 | 0.8131 |
| Multilingual TTS 0.6B | 76 | 0.2384 |
| Multilingual TTS 1.7B | 76 | **0.2362** |
| Orpheus | 76 | 0.6075 |
| Chatterbox | 23 | 0.1698 |
| FishSpeech2 | 76 | 0.2370 |
| Qwen3 TTS | 11 | **0.1064** |

> Chatterbox covers 23 languages only; Qwen3 TTS covers 11 languages only. Their averages are not directly comparable to 76-language models.

### Full Breakdown (76 languages)

![Benchmark heatmap](benchmark_results.png)
