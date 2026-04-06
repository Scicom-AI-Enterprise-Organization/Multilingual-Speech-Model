# VC Evaluation

Benchmarking multilingual voice cloning (VC) models across 76 languages using speaker similarity and Character Error Rate (CER).

![Benchmark scatter](scatter_results.png)

## Models

| Model | Description |
|-------|-------------|
| **Dia TTS** | [Nari Labs Dia TTS](https://github.com/nari-labs/dia) |
| **Multilingual TTS 0.6B** | [Scicom-intl/Multilingual-TTS-0.6B-Base](https://huggingface.co/Scicom-intl/Multilingual-TTS-0.6B-Base) |
| **Multilingual TTS 1.7B** | [Scicom-intl/Multilingual-TTS-1.7B-Base](https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base) |
| **Orpheus** | [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) |
| **Chatterbox** | [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) — 23 languages only |
| **Fish Audio S2 Pro** | [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) |

## Setup

Download the evaluation audio:

```bash
wget https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/resolve/main/vc_audio.zip
unzip vc_audio.zip -d common-voice
rm vc_audio.zip
```

## Run Generations

Each prompt is generated **twice** and the scores are averaged to reduce variance. We also upload all the generations done by us at [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC),

- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/blob/main/dia-tts.zip
- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/resolve/main/multilingual-tts-0.6b.zip
- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/blob/main/multilingual-tts-1.7b.zip
- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/blob/main/orpheus.zip
- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/blob/main/chatterbox.zip
- https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/blob/main/fishspeech2.zip

### Dia TTS

```bash
python3 dia_tts.py --output 'dia-tts'
```

### Scicom Multilingual TTS

```bash
# 0.6B
MODEL_NAME="Scicom-intl/Multilingual-TTS-0.6B-Base" python3 multilingual_tts.py --output 'multilingual-tts-0.6b'

# 1.7B
MODEL_NAME="Scicom-intl/Multilingual-TTS-1.7B-Base" python3 multilingual_tts.py --output 'multilingual-tts-1.7b'
```

### Orpheus

```bash
python3 orpheus.py --output 'orpheus'
```

### Chatterbox

```bash
python3 chatterbox.py --output 'chatterbox'
```

### Fish Audio S2 Pro

```bash
python3 fishspeech2.py --output 'fishspeech2'
```

### OmniVoice

```bash
python3 omnivoice_vc.py --output 'omnivoice'
```

## Evaluate

### Speaker Similarity

```bash
python3 calculate_similarity.py --output_folder "dia-tts"              --output "dia-tts-similarity"
python3 calculate_similarity.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-similarity"
python3 calculate_similarity.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-similarity"
python3 calculate_similarity.py --output_folder "orpheus"              --output "orpheus-similarity"
python3 calculate_similarity.py --output_folder "chatterbox"           --output "chatterbox-similarity"
python3 calculate_similarity.py --output_folder "fishspeech2"          --output "fishspeech2-similarity"
```

### CER

```bash
python3 calculate_cer.py --output_folder "dia-tts"              --output "dia-tts-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
python3 calculate_cer.py --output_folder "orpheus"              --output "orpheus-cer"
python3 calculate_cer.py --output_folder "chatterbox"           --output "chatterbox-cer"
python3 calculate_cer.py --output_folder "fishspeech2"          --output "fishspeech2-cer"
```

## Results

Summary across all evaluated languages. Full per-language heatmap below.

| Model | Languages | Similarity ↑ | CER ↓ |
|-------|:---------:|:------------:|:-----:|
| Dia TTS | 76 | 0.3416 | 0.6867 |
| Multilingual TTS 0.6B | 76 | 0.4868 | 0.3502 |
| Multilingual TTS 1.7B | 76 | **0.5036** | 0.3007 |
| Orpheus | 76 | 0.4002 | 0.6771 |
| Chatterbox | 23 | **0.6704** | **0.1099** |
| Fish Audio S2 Pro | 76 | 0.6097 | **0.2283** |

> Chatterbox covers 23 languages only; its averages are not directly comparable to 76-language models.

### Full Breakdown (76 languages)

![Benchmark heatmap](benchmark_results.png)
