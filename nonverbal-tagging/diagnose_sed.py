#!/usr/bin/env python3
"""What does the SED model actually output on known laughing audio?"""
import io
import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset, Audio

SR = 32000
from panns_inference import SoundEventDetection
from panns_inference.config import labels as AS_LABELS

LAUGH = ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]
LAUGH_COLS = [AS_LABELS.index(m) for m in LAUGH]

sed = SoundEventDetection(checkpoint_path="/root/models/Cnn14_DecisionLevelMax.pth",
                          device="cpu")

ds = load_dataset("ylacombe/expresso", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

n = 0
laugh_maxes = []
for ex in ds:
    if ex["style"] != "laughing":
        continue
    y, in_sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if in_sr != SR:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=SR)
    fw = sed.inference(y[None, : SR * 30])[0]  # (T, 527)
    clipmax = fw.max(axis=0)  # (527,)
    top = np.argsort(clipmax)[::-1][:6]
    if n < 5:
        print(f"clip {n} ({ex['id'] if 'id' in ex else ''}) top classes:",
              [(AS_LABELS[i], round(float(clipmax[i]), 3)) for i in top])
    laugh_maxes.append(float(clipmax[LAUGH_COLS].max()))
    n += 1
    if n >= 50:
        break

lm = np.array(laugh_maxes)
print(f"\nlaughter-family clip-max over {n} laughing clips:")
for q in (10, 25, 50, 75, 90, 99):
    print(f"  p{q}: {np.percentile(lm, q):.4f}")
print(f"  max: {lm.max():.4f}")
