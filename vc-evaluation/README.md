# VC Evaluation

## How to

1. Download the audio,

```bash
wget https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/resolve/main/vc_audio.zip
unzip vc_audio.zip -d common-voice
rm vc_audio.zip
```

2. Run Dia TTS,

```bash
python3 dia_tts.py --output 'dia-tts'
```

3. Run Scicom Multilingual TTS,

```bash
MODEL_NAME="Scicom-intl/Multilingual-TTS-0.6B-Base" python3 multilingual_tts.py --output 'multilingual-tts-0.6b'
MODEL_NAME="Scicom-intl/Multilingual-TTS-1.7B-Base" python3 multilingual_tts.py --output 'multilingual-tts-1.7b'
MODEL_NAME="Scicom-intl/Multilingual-TTS-4B-Base" python3 multilingual_tts.py --output 'multilingual-tts-4b'
```