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