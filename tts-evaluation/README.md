# TTS Evaluation

## How to run generations

### Dia TTS

### Scicom Multilingual TTS

#### 0.6B

```bash
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'
```

#### 1.7B

```bash
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'
```

### Orpheus

### Chatterbox

### FishSpeech2

## How to calculate CER

### Scicom Multilingual TTS

#### 0.6B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
```

```
```

#### 1.7B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
```

```
```