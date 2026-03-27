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
| **Fish Audio S2 Pro** | [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) |
| **Qwen3 TTS** | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — 11 languages only |

## Setup

```bash
pip install -r requirements.txt
```

## Run Generations

Each prompt is generated **twice** and the scores are averaged to reduce variance. We also upload all the generations done by us at [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC)

```bash
python3 dia_tts.py --output 'dia-tts'

MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'

python3 orpheus.py   --output 'orpheus'
python3 chatterbox.py --output 'chatterbox'
python3 fishspeech2.py --output 'fishspeech2'
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

### MOS

We evaluate using [Scicom-AI-Enterprise-Organization/faster-UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2) with 5 repetitions.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "dia-tts"               --output "dia-tts-mos"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "orpheus"               --output "orpheus-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "chatterbox"            --output "chatterbox-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "fishspeech2"           --output "fishspeech2-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "qwen3_tts"             --output "qwen3_tts-mos" --replication 5
```

## Results

Summary across all evaluated languages. Full per-language heatmap below.

| Model | Languages | CER ↓ | MOS ↑ |
|-------|:---------:|:-----:|:-----:|
| Dia TTS | 76 | 0.8131 | 1.8575 |
| Multilingual TTS 0.6B | 76 | 0.2384 | 3.2273 |
| Multilingual TTS 1.7B | 76 | **0.2362** | **3.2330** |
| Orpheus | 76 | 0.6075 | 2.7267 |
| Chatterbox | 23 | 0.1698 | 2.8405 |
| Fish Audio S2 Pro | 76 | 0.2370 | 2.9698 |
| Qwen3 TTS | 11 | **0.1064** | 2.6073 |

> Chatterbox covers 23 languages only; Qwen3 TTS covers 11 languages only. Their averages are not directly comparable to 76-language models.

### Full Breakdown (76 languages)

![Benchmark heatmap](benchmark_results.png)