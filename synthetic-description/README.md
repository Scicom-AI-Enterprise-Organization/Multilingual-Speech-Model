# Synthetic Description

## How to

This example use https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/viewer/haqkiem-TTS

1. Download the dataset,

```bash
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS/train-00000-of-00001.parquet -O haqkiem.parquet
unzip haqkiem-TTS_audio.zip
```

2. Calculate speech statistics and categories,

```bash
python3 speech_categories.py --file 'haqkiem.parquet' --language 'ms'
```