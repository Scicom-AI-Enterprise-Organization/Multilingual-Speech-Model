import re
import unicodedata
import torch
import penn
import numpy as np
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from huggingface_hub import hf_hub_download

# lang from https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
languages = {
    'en': 'eng-us',
    'ms': 'ind',
    'tamil': 'tam',
    'korean': 'kor',
    'chinese': 'zho-s',
    'latin': 'lat-clas',
    'vietnamese': 'vie-c',
    'turkish': 'tur',
    'japanese': 'jpn',
    'dutch': 'dut',
    'french': 'fra',
    'german': 'ger',
    'italian': 'ita',
    'polish': 'pol',
    'portuguese': 'por-po',
    'spanish': 'spa',
    'hungarian': 'hun',
    'russian': 'rus',
}

split_by_lang = {
    'en': 'space',
    'ms': 'space',
    'tamil': 'space',
    'korean': 'char',
    'chinese': 'char',
    'latin': 'space',
    'vietnamese': 'space',
    'turkish': 'space',
    'japanese': 'char',
    'dutch': 'space',
    'french': 'space',
    'german': 'space',
    'italian': 'space',
    'polish': 'space',
    'portuguese': 'space',
    'spanish': 'space',
    'hungarian': 'space',
    'russian': 'space'
}

# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

device = torch.device('cuda')

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

max_audio_length = 15 * SQUIM_OBJECTIVE.sample_rate

def clean_text(text):
    """
    Remove bracketed annotations like [Laughter], [Music], etc.
    and remove multilingual punctuation (ASCII, CJK, Arabic, Thai, etc.).
    """
    # remove bracketed content: [Laughter], [Music], etc.
    text = re.sub(r'\[[^\]]*\]', '', text)
    # remove parenthesized content: (laughs), （笑）, etc.
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'（[^）]*）', '', text)

    # remove characters whose Unicode category is punctuation (P*) or symbol (S*)
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('P') or cat.startswith('S'):
            continue
        cleaned.append(ch)
    text = ''.join(cleaned)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

model = None
tokenizer = None
model_brouhaha = None
pipeline = None
model_squim = None

def is_cjk(ch):
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or   # CJK Extension A
        0x20000 <= cp <= 0x2A6DF or # CJK Extension B
        0x3040 <= cp <= 0x309F or   # Hiragana
        0x30A0 <= cp <= 0x30FF or   # Katakana
        0xAC00 <= cp <= 0xD7AF or   # Hangul Syllables
        0x1100 <= cp <= 0x11FF      # Hangul Jamo
    )

def split_multilingual(text):
    """Split text by character type: CJK characters are individual tokens,
    non-CJK sequences are split by whitespace."""
    tokens = []
    current_word = []
    for ch in text:
        if ch.isspace():
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
        elif is_cjk(ch):
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
            tokens.append(ch)
        else:
            current_word.append(ch)
    if current_word:
        tokens.append(''.join(current_word))
    return tokens

def rate_apply(text, lang, audio_length):
    global model, tokenizer

    text = clean_text(text)

    if lang in ['multilingual', 'urdu']:
        tokens = split_multilingual(text)
        speaking_rate = len(tokens) / audio_length
        return {'phonemes': tokens, 'speaking_rate': speaking_rate}

    if model is None:
        model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
        tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    split_by = split_by_lang.get(lang, 'space')
    if split_by == 'space':
        words = text.split()
    else:
        words = list(text)

    lang_code = languages.get(lang, lang)
    words = [f'<{lang_code}>: ' + i for i in words]

    out = tokenizer(words, padding=True, add_special_tokens=False, return_tensors='pt')
    preds = model.generate(**out, num_beams=1, max_length=128)
    phones = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)

    speaking_rate = len(phones) / audio_length
    return {'phonemes': phones, 'speaking_rate': speaking_rate}

def pitch_apply(audio, sr, penn_batch_size=4096):
    pitch, periodicity = penn.from_audio(
        torch.from_numpy(audio[None, :]).float(),
        sr,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=penn_batch_size,
        center=center,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=0
    )
    return {'pitch_mean': float(pitch.mean().cpu()), 'pitch_std': float(pitch.std().cpu())}

def snr_apply(audio, sr, batch_size=32, ratio = 16000/270):
    global model_brouhaha, pipeline

    if model_brouhaha is None:
        model_brouhaha = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
        model_brouhaha = model_brouhaha.to(device)
    
    if pipeline is None:
        pipeline = RegressiveActivityDetectionPipeline(segmentation=model_brouhaha, batch_size=batch_size)
        pipeline = pipeline.to(device)

    res = pipeline(
        {
            "sample_rate": sr,
            "waveform": torch.from_numpy(audio[None, :]).to(device).float()
        }
    )
    mask = np.full(res["snr"].shape, False)
    for (segment, _) in res["annotation"].itertracks():
        start = int(segment.start * ratio)
        end = int(segment.end * ratio)
        mask[start:end] = True
    mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

    vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))
    return {'snr': float(res["snr"][mask].mean()), 'c50': float(res["c50"][mask].mean()), 'vad_duration': vad_duration}


def squim_apply(audio, sr):
    global model_squim
    if model_squim is None:
        model_squim = SQUIM_OBJECTIVE.get_model().to(device)
    
    waveform = torchaudio.functional.resample(torch.from_numpy(audio[None, :]).to(device).float(), sr, SQUIM_OBJECTIVE.sample_rate)
    with torch.no_grad():
        stoi_sample, pesq_sample, sdr_sample = model_squim(waveform)
    
    return {'sdr': float(sdr_sample.cpu()[0]), 'pesq': float(pesq_sample.cpu()[0]), 'stoi': float(stoi_sample.cpu()[0])}