#!/usr/bin/env python3
"""Stage C+D: CLAP verification of mined events + inline tag placement.

C: crop each event, score with CLAP zero-shot against class prompts vs
   speech/music/noise negatives; keep events whose positive prob passes.
D: for files with >=1 verified event, run faster-whisper with word
   timestamps, insert <|sfx:family|> + onomatopoeia at the nearest word
   gap; events overlapping speech >50% are kept in metadata but not tagged.

Output: tagged.parquet + qa_crops/<family>/*.wav for listening.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from tqdm import tqdm

CLAP_SR = 48000
CTX_S = 0.20  # context around crop

PROMPTS = {
    "laughter": ["a person laughing", "someone giggling loudly"],
    "cough": ["a person coughing", "someone clearing their throat"],
    "sigh": ["a person sighing"],
    "crying": ["a person crying and sobbing"],
    "screaming": ["a person screaming"],
    "sneeze": ["a person sneezing"],
    "sniff": ["a person sniffing their nose"],
    "burping": ["a person burping"],
    "humming": ["a person humming a tune"],
}
NEGATIVES = [
    "a person talking", "clear speech of a person",
    "music playing", "background noise", "silence",
]
ONO = {
    "laughter": "Haha", "cough": "Ahem", "sigh": "Hah", "crying": "Huhu",
    "screaming": "Ahh", "sneeze": "Achoo", "sniff": "Sff", "burping": "Burp",
    "humming": "Hmm",
}
# script-matched onomatopoeia by whisper-detected language; fall back to Latin
ONO_LANG = {
    "zh": {"laughter": "哈哈", "cough": "咳咳", "sigh": "唉", "crying": "呜呜",
           "screaming": "啊", "sneeze": "阿嚏", "sniff": "哼", "burping": "嗝",
           "humming": "嗯"},
    "ta": {"laughter": "ஹஹ", "cough": "அஹம்", "sigh": "ஆ", "crying": "ஊஊ",
           "screaming": "ஆஹ்", "sneeze": "அச்சூ"},
}


def ono_for(family, lang, long_laugh=False):
    base = ONO_LANG.get(lang, {}).get(family, ONO[family])
    if family == "laughter" and long_laugh:
        return base + base[len(base) // 2:] if lang in ONO_LANG else "Hahaha"
    return base


def build_clap():
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().cuda()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    fam_names = list(PROMPTS.keys())
    texts, owner = [], []
    for fam in fam_names:
        for p in PROMPTS[fam]:
            texts.append(p)
            owner.append(fam)
    for p in NEGATIVES:
        texts.append(p)
        owner.append("_neg")
    with torch.no_grad():
        t = proc(text=texts, return_tensors="pt", padding=True)
        tf = model.get_text_features(**{k: v.cuda() for k, v in t.items()})
        tf = getattr(tf, "pooler_output", tf)  # transformers>=5 returns an output object
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return model, proc, tf, owner


def clap_prob(model, proc, tfeat, owner, wav, family):
    with torch.no_grad():
        try:
            inp = proc(audio=[wav], sampling_rate=CLAP_SR, return_tensors="pt")
        except Exception:
            inp = proc(audios=[wav], sampling_rate=CLAP_SR, return_tensors="pt")
        af = model.get_audio_features(**{k: v.cuda() for k, v in inp.items()})
        af = getattr(af, "pooler_output", af)
        af = af / af.norm(dim=-1, keepdim=True)
        sims = (af @ tfeat.T).squeeze(0) * 100.0  # logit scale ~ CLAP temp
        probs = sims.softmax(dim=0).cpu().numpy()
    fam_prob = sum(p for p, o in zip(probs, owner) if o == family)
    neg_prob = sum(p for p, o in zip(probs, owner) if o == "_neg")
    return float(fam_prob), float(neg_prob)


NV_LABEL = {
    "laughter": "[Laughter]", "cough": "[Cough]", "sigh": "[Sigh]",
    "crying": "[Crying]", "screaming": "[Screaming]", "sneeze": "[Sneeze]",
    "sniff": "[Sniff]", "burping": "[Burping]", "humming": "[Humming]",
}


def place_tags(words, events, lang=""):
    """words: [(start, end, text)]; events: verified.

    Returns two renderings from the same events:
    - tagged_text (Higgs style): <|sfx:x|> + onomatopoeia, only for events NOT
      overlapping speech >50% (a discrete event the model should *produce*).
    - nv_text (Emilia-NV / NVSpeech style): bare [Label] word-level tokens for
      ALL events at their occurrence position, overlapped or not.
    """
    tokens = [w[2].strip() for w in words]
    sfx_ins = defaultdict(list)
    nv_ins = defaultdict(list)
    tagged_events = []
    for ev in events:
        onset, offset = ev["onset"], ev["offset"]
        ov = 0.0
        for s, e, _ in words:
            ov += max(0.0, min(e, offset) - max(s, onset))
        frac = ov / max(1e-6, offset - onset)
        pos = len(words)
        for i, (s, _, _) in enumerate(words):
            if s >= onset:
                pos = i
                break
        nv_ins[pos].append(NV_LABEL[ev["family"]])
        ev["overlap_frac"] = round(frac, 3)
        if frac > 0.5:
            ev["placed"] = False
        else:
            n_ono = ono_for(ev["family"], lang, long_laugh=(offset - onset) > 1.2)
            sfx_ins[pos].append(f"<|sfx:{ev['family']}|>{n_ono}")
            ev["placed"] = True
        tagged_events.append(ev)

    def render(ins):
        out = []
        for i, tok in enumerate(tokens):
            if i in ins:
                out.extend(ins[i])
            out.append(tok)
        if len(words) in ins:
            out.extend(ins[len(words)])
        return " ".join(out)

    return render(sfx_ins), render(nv_ins), tagged_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="/root/out/events.jsonl")
    ap.add_argument("--out-dir", default="/root/out")
    ap.add_argument("--meta-parquet-glob",
                    default="/root/data/malaysian-emilia/audio_length_ratio_text/*.parquet")
    ap.add_argument("--clap-thresh", type=float, default=0.55)
    ap.add_argument("--qa-per-class", type=int, default=40)
    ap.add_argument("--limit-files", type=int, default=0)
    args = ap.parse_args()

    events = [json.loads(l) for l in open(args.events)]
    by_file = defaultdict(list)
    for ev in events:
        by_file[ev["file"]].append(ev)
    files = sorted(by_file)
    if args.limit_files:
        files = files[: args.limit_files]
    print(f"{len(events)} events across {len(by_file)} files; verifying {len(files)} files")

    model, proc, tfeat, owner = build_clap()

    # ---- Stage C: CLAP verify ----
    verified_by_file = {}
    qa_saved = defaultdict(int)
    os.makedirs(f"{args.out_dir}/qa_crops", exist_ok=True)
    for f in tqdm(files, desc="clap-verify"):
        try:
            y, _ = librosa.load(f, sr=CLAP_SR, mono=True)
        except Exception:
            continue
        keep = []
        for ev in by_file[f]:
            s = max(0, int((ev["onset"] - CTX_S) * CLAP_SR))
            e = min(len(y), int((ev["offset"] + CTX_S) * CLAP_SR))
            if e - s < CLAP_SR // 10:
                continue
            fam_p, neg_p = clap_prob(model, proc, tfeat, owner, y[s:e], ev["family"])
            ev["clap_fam"] = round(fam_p, 4)
            ev["clap_neg"] = round(neg_p, 4)
            if fam_p >= args.clap_thresh:
                keep.append(ev)
                fam = ev["family"]
                if qa_saved[fam] < args.qa_per_class:
                    d = f"{args.out_dir}/qa_crops/{fam}"
                    os.makedirs(d, exist_ok=True)
                    sf.write(f"{d}/{qa_saved[fam]:03d}_p{fam_p:.2f}_{os.path.basename(f)}.wav",
                             y[s:e], CLAP_SR)
                    qa_saved[fam] += 1
        if keep:
            verified_by_file[f] = keep
    n_verified = sum(len(v) for v in verified_by_file.values())
    print(f"verified: {n_verified} events in {len(verified_by_file)} files")

    del model
    torch.cuda.empty_cache()

    # ---- Stage D: whisper word timestamps + tag insertion ----
    # faster-whisper (CTranslate2) may lack sm_120 kernels on RTX 5090; fall
    # back to transformers whisper (plain torch, works on Blackwell).
    wm, hf_asr = None, None
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel("large-v3", device="cuda", compute_type="float16")
        list(wm.transcribe(np.zeros(16000, dtype=np.float32))[0])  # smoke test
        print("using faster-whisper")
    except Exception as e:
        print(f"faster-whisper unavailable on this GPU ({e}); using transformers whisper")
        wm = None
        from transformers import pipeline as hf_pipeline
        hf_asr = hf_pipeline(
            "automatic-speech-recognition", model="openai/whisper-large-v3",
            torch_dtype=torch.float16, device="cuda",
            chunk_length_s=30, return_timestamps="word",
        )

    def transcribe_words(path):
        """-> (words [(start, end, text)], full_text, language)"""
        if wm is not None:
            segs, info = wm.transcribe(path, word_timestamps=True, vad_filter=False)
            words, texts = [], []
            for seg in segs:
                texts.append(seg.text)
                for w in seg.words or []:
                    words.append((w.start, w.end, w.word))
            return words, "".join(texts).strip(), info.language
        out = hf_asr(path)
        words = [(c["timestamp"][0], c["timestamp"][1] or c["timestamp"][0] + 0.2, c["text"])
                 for c in out.get("chunks", []) if c["timestamp"][0] is not None]
        return words, out["text"].strip(), ""

    meta = {}
    try:
        from glob import glob as _g
        dfs = [pd.read_parquet(p, columns=["audio_filename_trim", "text"])
               for p in _g(args.meta_parquet_glob)]
        m = pd.concat(dfs)
        meta = dict(zip(m["audio_filename_trim"].map(os.path.basename), m["text"]))
        print(f"meta transcripts: {len(meta)}")
    except Exception as e:
        print(f"meta load failed ({e}); continuing without original text")

    rows = []
    for f in tqdm(sorted(verified_by_file), desc="whisper-tag"):
        try:
            words, full_text, lang = transcribe_words(f)
        except Exception as e:
            print(f"whisper failed {f}: {e}")
            continue
        tagged, nv_text, evs = place_tags(words, sorted(verified_by_file[f], key=lambda x: x["onset"]), lang=lang)
        rows.append({
            "file": f,
            "orig_text": meta.get(os.path.basename(f), ""),
            "whisper_text": full_text,
            "tagged_text": tagged,
            "nv_text": nv_text,
            "language": lang,
            "events": json.dumps(evs),
            "n_placed": sum(1 for e in evs if e.get("placed")),
        })

    df = pd.DataFrame(rows)
    out = f"{args.out_dir}/tagged.parquet"
    df.to_parquet(out)
    print(f"wrote {out}: {len(df)} rows, {int(df['n_placed'].sum())} placed tags")
    stats = defaultdict(int)
    for evs in df["events"]:
        for ev in json.loads(evs):
            stats[ev["family"] + ("_placed" if ev.get("placed") else "_overlapped")] += 1
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
