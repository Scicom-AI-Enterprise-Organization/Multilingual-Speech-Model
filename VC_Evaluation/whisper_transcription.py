
import os
import json
import librosa
import jiwer
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict
from datasets import load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
from queue import Empty
from collections import defaultdict


DEVICE_MAP = [0, 1, 2]
WHISPER_MODEL = "openai/whisper-large-v3"
DATASET_DIR = "vc_dataset_filtered"
OUTPUT_ROOT = "vc_outputs/qwen_1.7b"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "eval")
METRICS_PATH = os.path.join(OUTPUT_DIR, "whisper_eval_metrics.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "avg.json")

PROBE_START_CLIPS = 4
SAFETY_MARGIN = 0.8
SAVE_EVERY = 10
MAX_SAMPLES = 30 * 16000  


@dataclass
class EvalItem:
    sample_id: str
    wav_path: str
    target: str
    lang: str
    run: str


def compute_metrics(results: list) -> dict:
    by_lang_run = defaultdict(list)
    by_lang = defaultdict(list)
    all_cer = []

    for r in results:
        score = r["score"]
        by_lang_run[(r["lang"], r["run"])].append(score)
        by_lang[r["lang"]].append(score)
        all_cer.append(score)

    per_lang_per_run = defaultdict(dict)
    for (lang, run), scores in by_lang_run.items():
        per_lang_per_run[lang][run] = {
            "metric": "cer",
            "mean": round(sum(scores) / len(scores), 4),
            "num_samples": len(scores),
        }

    per_lang_avg = {}
    for lang, scores in by_lang.items():
        per_lang_avg[lang] = {
            "metric": "cer",
            "mean": round(sum(scores) / len(scores), 4),
            "num_samples": len(scores),
        }

    overall_avg = {}
    if all_cer:
        overall_avg["cer"] = round(sum(all_cer) / len(all_cer), 4)

    return {
        "per_lang_per_run": dict(per_lang_per_run),
        "per_lang_avg": per_lang_avg,
        "overall_avg": overall_avg,
    }


def save_results(results_list, lock):
    with lock:
        results = list(results_list)
        metrics = compute_metrics(results)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output = {
            "summary": metrics,
            "per_sample": results,
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Aggregated averages only
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def find_batch_size(model, processor, gpu_id: int) -> int:
    import torch
    dummy = np.zeros(30 * 16000, dtype=np.float32)
    n_clips = PROBE_START_CLIPS
    last_good = PROBE_START_CLIPS

    print(f"[GPU {gpu_id}] Probing VRAM to find batch size...")

    while True:
        try:
            audios = [dummy] * n_clips
            inputs = processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=False,
            ).to("cuda:0")
            inputs["input_features"] = inputs["input_features"].to(torch.float16)

            with torch.no_grad():
                model.generate(**inputs)

            last_good = n_clips
            print(f"[GPU {gpu_id}]   {n_clips} clips → OK")
            n_clips *= 2
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}]   {n_clips} clips → OOM")
            break

    batch_size = max(1, int(last_good * SAFETY_MARGIN))
    print(f"[GPU {gpu_id}] Auto batch size: {batch_size} clips per batch")
    return batch_size


def transcribe_batch(
    batch: List[EvalItem],
    audios: List[np.ndarray],
    model,
    processor,
    long_pipe,
) -> Dict[str, str]:
    import torch

    short_items, short_audios = [], []
    long_items, long_audios = [], []

    for item, audio in zip(batch, audios):
        if len(audio) <= MAX_SAMPLES:
            short_items.append(item)
            short_audios.append(audio)
        else:
            long_items.append(item)
            long_audios.append(audio)

    results = {}

    if short_items:
        inputs = processor(
            short_audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=False,
        ).to("cuda:0")
        inputs["input_features"] = inputs["input_features"].to(torch.float16)
        with torch.no_grad():
            ids = model.generate(**inputs)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for item, hyp in zip(short_items, hyps):
            results[item.sample_id] = hyp

    if long_items:
        for item, audio in zip(long_items, long_audios):
            print(f"  [long clip] {item.sample_id}: {len(audio)/16000:.1f}s — using chunked inference")
            hyp = long_pipe({"array": audio, "sampling_rate": 16000})["text"]
            results[item.sample_id] = hyp

    return results


def process_lang(gpu_id, lang, model, processor, long_pipe, results_list, lock, batch_size):
    ds_path = os.path.join(DATASET_DIR, lang)
    if not os.path.exists(ds_path):
        print(f"[GPU {gpu_id}] {lang}: dataset not found, skipping")
        return

    try:
        ds = load_from_disk(ds_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] {lang}: failed to load dataset - {e}")
        return

    out_lang_dir = os.path.join(OUTPUT_ROOT, lang)
    if not os.path.exists(out_lang_dir):
        print(f"[GPU {gpu_id}] {lang}: output dir not found, skipping")
        return

    run_dirs = [
        d for d in os.listdir(out_lang_dir)
        if d.startswith("run") and os.path.isdir(os.path.join(out_lang_dir, d))
    ]
    if not run_dirs:
        print(f"[GPU {gpu_id}] {lang}: no run dirs found, skipping")
        return

    items: List[EvalItem] = []
    for row in ds:
        sample_id = row["audio_filename"].replace(".flac", "")
        target = row["target_text"]
        for run_dir in sorted(run_dirs):
            wav_path = os.path.join(out_lang_dir, run_dir, f"{sample_id}.wav")
            if not os.path.exists(wav_path):
                continue
            items.append(EvalItem(
                sample_id=sample_id,
                wav_path=wav_path,
                target=target,
                lang=lang,
                run=run_dir,
            ))

    if not items:
        print(f"[GPU {gpu_id}] {lang}: no items to evaluate")
        return

    print(f"[GPU {gpu_id}] {lang}: {len(items)} items, batch_size={batch_size}")

    pbar = tqdm(
        total=len(items),
        desc=f"[GPU {gpu_id}] {lang}",
        position=gpu_id,
        leave=True,
    )

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            audios = [librosa.load(item.wav_path, sr=16000)[0] for item in batch]
            hyp_map = transcribe_batch(batch, audios, model, processor, long_pipe)

            with lock:
                for item in batch:
                    hyp = hyp_map.get(item.sample_id, "")
                    err = jiwer.cer(item.target.lower(), hyp.lower())
                    results_list.append({
                        "lang": item.lang,
                        "run": item.run,
                        "sample_id": item.sample_id,
                        "metric": "cer",
                        "score": round(err, 4),
                        "target": item.target,
                        "hyp": hyp,
                    })

        except Exception as e:
            print(f"[GPU {gpu_id}] {lang} batch failed: {e}")

        pbar.update(len(batch))

        batch_idx = i // batch_size
        if batch_idx % SAVE_EVERY == 0:
            save_results(results_list, lock)

    pbar.close()
    save_results(results_list, lock)
    print(f"[GPU {gpu_id}] {lang}: done")


def worker(gpu_id, lang_queue, results_list, lock):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)

    long_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        chunk_length_s=30,
        stride_length_s=5,
    )

    batch_size = find_batch_size(model, processor, gpu_id)
    print(f"[GPU {gpu_id}] Ready, waiting for work...")

    while True:
        try:
            lang = lang_queue.get_nowait()
        except Empty:
            break

        print(f"[GPU {gpu_id}] picked up: {lang}")
        process_lang(gpu_id, lang, model, processor, long_pipe, results_list, lock, batch_size)

    print(f"[GPU {gpu_id}] queue empty, shutting down.")


def eval_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    langs = [
        d for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
        and d != "eval"
    ]
    print(f"Found {len(langs)} languages: {langs}")

    manager = mp.Manager()
    results_list = manager.list()
    lock = manager.Lock()

    lang_queue = manager.Queue()
    for lang in langs:
        lang_queue.put(lang)

    print(f"Spawning {len(DEVICE_MAP)} GPU workers...")
    processes = []
    for gpu_id in DEVICE_MAP:
        p = mp.Process(target=worker, args=(gpu_id, lang_queue, results_list, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save_results(results_list, lock)

    results = list(results_list)
    metrics = compute_metrics(results)

    print(f"\nSaved full results to {METRICS_PATH}")
    print(f"Saved averages to {SUMMARY_PATH}")

    print("\n=== Per Language Per Run ===")
    for lang, runs in metrics["per_lang_per_run"].items():
        for run, m in sorted(runs.items()):
            print(f"  {lang} {run}: {m['metric'].upper()} = {m['mean']:.4f} ({m['num_samples']} samples)")

    print("\n=== Per Language Avg (all runs) ===")
    for lang, m in metrics["per_lang_avg"].items():
        print(f"  {lang}: {m['metric'].upper()} = {m['mean']:.4f} ({m['num_samples']} samples)")

    print("\n=== Overall Avg ===")
    for metric, val in metrics["overall_avg"].items():
        print(f"  {metric.upper()} = {val:.4f}")


if __name__ == "__main__":
    eval_all()