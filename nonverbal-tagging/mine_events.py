#!/usr/bin/env python3
"""Stage A+B: framewise sound-event detection over audio segments.

Runs PANNs Cnn14_DecisionLevelMax (framewise AudioSet posteriors, ~100 fps)
on every audio file, extracts per-family events (onset/offset/peak) with
threshold + median filter + gap merge, writes events.jsonl.

Segments are already VAD-trimmed podcast clips, so clip-level gating and
framewise SED collapse into one pass.
"""
import argparse
import json
import os
import warnings
from glob import glob

import numpy as np
import torch
import librosa
from scipy.ndimage import median_filter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

SR = 32000  # PANNs input rate

# AudioSet label -> our tag family. Family score = max over member labels.
FAMILIES = {
    "laughter": ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "cough": ["Cough", "Throat clearing"],
    "sigh": ["Sigh"],
    "crying": ["Crying, sobbing", "Whimper"],
    "screaming": ["Screaming"],
    "sneeze": ["Sneeze"],
    "sniff": ["Sniff"],
    "burping": ["Burping, eructation"],
    "humming": ["Humming"],
}
# extraction params per family: (enter_thresh, peak_thresh, min_dur_s)
PARAMS = {
    "laughter": (0.20, 0.50, 0.30),
    "cough": (0.20, 0.50, 0.15),
    "sigh": (0.15, 0.40, 0.20),
    "crying": (0.20, 0.50, 0.40),
    "screaming": (0.25, 0.60, 0.20),
    "sneeze": (0.20, 0.50, 0.10),
    "sniff": (0.20, 0.50, 0.10),
    "burping": (0.25, 0.60, 0.10),
    "humming": (0.20, 0.50, 0.30),
}
MERGE_GAP_S = 0.30


class AudioDataset(Dataset):
    def __init__(self, files, max_dur=30.0):
        self.files = files
        self.max_samples = int(max_dur * SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        try:
            y, _ = librosa.load(f, sr=SR, mono=True)
        except Exception:
            return f, np.zeros(SR, dtype=np.float32), 0
        if len(y) < SR // 10:
            return f, np.zeros(SR, dtype=np.float32), 0
        y = y[: self.max_samples]
        return f, y.astype(np.float32), len(y)


def collate(batch):
    files, ys, lens = zip(*batch)
    T = max(max(lens), SR)
    padded = np.zeros((len(ys), T), dtype=np.float32)
    for i, y in enumerate(ys):
        padded[i, : len(y)] = y
    return files, padded, lens


def extract_events(score, fps, enter, peak, min_dur):
    """score: (T,) family posterior track -> [(onset_s, offset_s, peak_conf)]"""
    score = median_filter(score, size=5)
    active = score >= enter
    events = []
    start = None
    for t in range(len(active)):
        if active[t] and start is None:
            start = t
        elif not active[t] and start is not None:
            events.append((start, t))
            start = None
    if start is not None:
        events.append((start, len(active)))
    # merge close events
    merged = []
    for s, e in events:
        if merged and (s - merged[-1][1]) / fps < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        dur = (e - s) / fps
        pk = float(score[s:e].max())
        if dur >= min_dur and pk >= peak:
            out.append((s / fps, e / fps, pk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="/root/out/events.jsonl")
    ap.add_argument("--checkpoint", default="/root/models/Cnn14_DecisionLevelMax.pth")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresh-scale", type=float, default=1.0,
                    help="multiply all enter/peak thresholds by this factor")
    ap.add_argument("--stats-out", default="",
                    help="optional jsonl of per-file max posterior per family")
    args = ap.parse_args()

    global PARAMS
    if args.thresh_scale != 1.0:
        PARAMS = {k: (e * args.thresh_scale, p * args.thresh_scale, d)
                  for k, (e, p, d) in PARAMS.items()}
        print(f"thresholds scaled x{args.thresh_scale}: {PARAMS}")

    from panns_inference import SoundEventDetection
    from panns_inference.config import labels as AS_LABELS

    idx = {}
    for fam, members in FAMILIES.items():
        idx[fam] = [AS_LABELS.index(m) for m in members if m in AS_LABELS]
        missing = [m for m in members if m not in AS_LABELS]
        if missing:
            print(f"warning: {fam} missing labels {missing}")

    exts = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(args.audio_dir, "**", e), recursive=True))
    files = sorted(files)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} audio files")

    # resume support
    done = set()
    if os.path.exists(args.out + ".done"):
        with open(args.out + ".done") as f:
            done = set(l.strip() for l in f)
        files = [f for f in files if f not in done]
        print(f"resuming: {len(files)} left")

    sed = SoundEventDetection(checkpoint_path=args.checkpoint, device="cuda")

    ds = AudioDataset(files)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=collate, prefetch_factor=4)

    n_events = 0
    with open(args.out, "a") as fout, open(args.out + ".done", "a") as fdone, torch.no_grad():
        for bfiles, batch, lens in tqdm(dl):
            framewise = sed.inference(batch)  # (B, T_frames, 527) numpy
            T_frames = framewise.shape[1]
            fps = T_frames / (batch.shape[1] / SR)
            for i, f in enumerate(bfiles):
                if lens[i] == 0:
                    fdone.write(f + "\n")
                    continue
                n_valid = int(T_frames * lens[i] / batch.shape[1])
                fw = framewise[i, :n_valid]
                for fam, cols in idx.items():
                    if not cols:
                        continue
                    track = fw[:, cols].max(axis=1)
                    enter, peak, min_dur = PARAMS[fam]
                    if track.max() < peak:
                        continue
                    for onset, offset, pk in extract_events(track, fps, enter, peak, min_dur):
                        fout.write(json.dumps({
                            "file": f, "family": fam,
                            "onset": round(onset, 3), "offset": round(offset, 3),
                            "peak": round(pk, 4), "dur_s": round(lens[i] / SR, 2),
                        }) + "\n")
                        n_events += 1
                fdone.write(f + "\n")
            fout.flush()
            fdone.flush()
    print(f"DONE, {n_events} events")


if __name__ == "__main__":
    main()
