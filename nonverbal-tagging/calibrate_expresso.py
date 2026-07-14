#!/usr/bin/env python3
"""Detector recall calibration on Expresso: 'laughing' style clips are known
positives, 'default'/'confused' read styles are near-guaranteed negatives.
Reports detection rate at the mining thresholds so we know what recall the
Malaysian-Emilia run is operating at."""
import numpy as np
import librosa
import torch
from datasets import load_dataset

SR = 32000
N_PER_CLASS = 150

from panns_inference import SoundEventDetection
from panns_inference.config import labels as AS_LABELS

LAUGH_COLS = [AS_LABELS.index(m) for m in
              ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]]

sed = SoundEventDetection(checkpoint_path="/root/models/Cnn14_DecisionLevelMax.pth",
                          device="cuda")

import io
import soundfile as _sf
from datasets import Audio

ds = load_dataset("ylacombe/expresso", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

pos_scores, neg_scores = [], []
for ex in ds:
    style = ex["style"]
    if style == "laughing" and len(pos_scores) < N_PER_CLASS:
        bucket = pos_scores
    elif style in ("default", "confused") and len(neg_scores) < N_PER_CLASS:
        bucket = neg_scores
    else:
        if len(pos_scores) >= N_PER_CLASS and len(neg_scores) >= N_PER_CLASS:
            break
        continue
    y, in_sr = _sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if in_sr != SR:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=SR)
    y = y[: SR * 30]
    with torch.no_grad():
        fw = sed.inference(y[None, :])
    track = fw[0][:, LAUGH_COLS].max(axis=1)
    bucket.append(float(track.max()))

pos = np.array(pos_scores)
neg = np.array(neg_scores)
print(f"laughing clips: {len(pos)}, negative clips: {len(neg)}")
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    print(f"thresh {th:.1f}: recall {(pos >= th).mean():.3f}  false-pos {(neg >= th).mean():.3f}")
