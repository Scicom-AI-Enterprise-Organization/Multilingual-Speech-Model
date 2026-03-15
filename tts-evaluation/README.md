# TTS Evaluation

## How to run generations

### Scicom Multilingual TTS

```bash
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'
```