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

- Single GPU replica required at least 22GB of VRAM.

3. Calculate bins for speech statistics and add the columns,

```bash
python3 calculate_bins.py --pattern 'haqkiem-TTS_audio_speech_categories/*.json' --output 'bins.json'
python3 add_column.py --pattern 'haqkiem-TTS_audio_speech_categories/*.json' --output 'categorized_haqkiem.parquet'
```

### Larger scale

1. Check [download.sh](download.sh) and [process.sh](process.sh) how we scale to bigger dataset
2. Calculate global statistics,

```bash
python3 calculate_bins.py --pattern '*_speech_categories/*.json' --output 'bins.json'
```