# Synthetic Description

Generate natural-language descriptions of speech audio by computing acoustic statistics and categories, then prompting an LLM to summarise them.

The output dataset is published at [Scicom-intl/ExpressiveSpeech](https://huggingface.co/datasets/Scicom-intl/ExpressiveSpeech).

## Pipeline

```
audio + transcript
      │
      ▼
speech_categories.py   ← categories (emotion, gender, fluency, accent, quality)
      │                   speech_stats_func  ← pitch, SNR, speaking rate, SQUIM
      ▼
calculate_bins.py      ← bin continuous stats, merge into a single parquet
      │
      ▼
synthetic.py           ← LLM generates a natural-language description per row
```

## Quickstart

Uses [malaysia-ai/Multilingual-TTS · haqkiem-TTS](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/viewer/haqkiem-TTS) as an example.

**1. Download the dataset**

```bash
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS/train-00000-of-00001.parquet -O haqkiem.parquet
unzip haqkiem-TTS_audio.zip
```

**2. Compute speech statistics and categories**

> Requires a single GPU with at least 22 GB VRAM.

```bash
python3 speech_categories.py --file 'haqkiem.parquet' --language 'ms'
```

**3. Calculate bins and merge**

```bash
python3 calculate_bins.py --pattern 'haqkiem-TTS_audio_speech_categories/*.json' --output 'output.parquet'
```

**4. Generate synthetic descriptions**

```bash
python3 synthetic.py --file 'output.parquet' --folder 'output'
```

Uses [Qwen/Qwen2.5-72B-Instruct](https://deepinfra.com/Qwen/Qwen2.5-72B-Instruct) via DeepInfra by default. Any OpenAI-compatible endpoint works — set `API_KEY`, `BASE_URL`, and `MODEL_NAME`.

## Larger Scale

See [download.sh](download.sh) and [process.sh](process.sh) for batch processing scripts.

**Bin across all speakers**

```bash
python3 calculate_bins.py --pattern '*_speech_categories/*.json' --output 'output.parquet'
```

**Serve a local LLM with vLLM**

```bash
hf download mesolitica/Qwen2.5-72B-Instruct-FP8 --local-dir=./Qwen2.5-72B-Instruct-FP8
vllm serve "Qwen2.5-72B-Instruct-FP8" --tensor-parallel 8 --max-model-len 4096
```

```bash
API_KEY="-" BASE_URL="http://localhost:8000/v1" MODEL_NAME="Qwen2.5-72B-Instruct-FP8" \
python3 synthetic.py --file 'output.parquet' --folder 'output-synthetic'
```

## Pretrained Models

### `speech_categories_func.py`

| Model | HuggingFace ID | Purpose |
|-------|----------------|---------|
| WhisperFluency | [tiantiaf/whisper-large-v3-speech-flow](https://huggingface.co/tiantiaf/whisper-large-v3-speech-flow) | Fluency / disfluency detection |
| WhisperQuality | [tiantiaf/whisper-large-v3-voice-quality](https://huggingface.co/tiantiaf/whisper-large-v3-voice-quality) | Voice quality tagging |
| WhisperAccent | [tiantiaf/whisper-large-v3-narrow-accent](https://huggingface.co/tiantiaf/whisper-large-v3-narrow-accent) | English accent classification |
| emotion2vec | [iic/emotion2vec_plus_large](https://huggingface.co/iic/emotion2vec_plus_large) | Emotion recognition (via FunASR) |
| ECAPA Gender | [JaesungHuh/ecapa-gender](https://huggingface.co/JaesungHuh/ecapa-gender) | Binary gender (male / female) |
| Wav2Vec2 Age/Gender | [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) | Age regression + gender (female / male / child) |

| Function | Output |
|----------|--------|
| `predict_fluency(audio)` | Per-3 s segment fluency label; disfluency types: Block, Prolongation, Sound Repetition, Word Repetition, Interjection |
| `predict_quality(audio)` | Active voice-quality tags (shrill, husky, raspy, monotone, …) |
| `predict_accent(audio)` | Dominant English accent region |
| `predict_sex_age(audio)` | Sex label (female / male / child) and estimated age |
| `predict_gender(audio)` | Binary gender via ECAPA |
| `predict_emotion(audio)` | Top emotion label |

### `speech_stats_func.py`

| Model | HuggingFace ID | Purpose |
|-------|----------------|---------|
| G2P T5 | [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100) | Multilingual grapheme-to-phoneme |
| ByT5 tokenizer | [google/byt5-small](https://huggingface.co/google/byt5-small) | Tokenizer for G2P model |
| Brouhaha | [ylacombe/brouhaha-best](https://huggingface.co/ylacombe/brouhaha-best) | SNR and C50 over voiced regions |
| SQUIM Objective | `torchaudio.pipelines.SQUIM_OBJECTIVE` | Reference-free STOI / PESQ / SDR |

| Function | Output |
|----------|--------|
| `rate_apply(text, lang, audio_length)` | Phoneme list and speaking rate (phonemes/s); falls back to word tokens for `multilingual`/`urdu` |
| `pitch_apply(audio, sr)` | Pitch mean and std (Hz) via [PENN](https://github.com/interactiveaudiolab/penn) FCNF0++ |
| `snr_apply(audio, sr)` | Mean SNR (dB), C50 (dB), and VAD duration (s) via Brouhaha |
| `squim_apply(audio, sr)` | SDR, PESQ, and STOI via torchaudio SQUIM |
