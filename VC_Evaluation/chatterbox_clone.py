import os
import multiprocessing as mp
mp.set_start_method('spawn', force=True)  

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker

import json
import torch
import torchaudio as ta
from dataclasses import dataclass
from typing import List, Tuple
from functools import partial
from multiprocess import Pool
from tqdm import tqdm
from datasets import load_from_disk
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

DATASET_DIR = "vc_dataset_filtered"
OUTPUT_DIR = "vc_outputs/chatterbox_multilingual"
FAILED_JSON = "failed_samples.json"

NUM_RUNS = 3
MAX_RETRIES = 3
PROCESSES_PER_GPU = 5  

@dataclass
class TTSItem:
    sample_id: str
    target_text: str
    ref_audio_path: str
    language_id: str
    lang: str  


def chunks(items: list, devices: list):
    chunk_size = len(items) // len(devices)
    remainder = len(items) % len(devices)
    start = 0
    for i, device in enumerate(devices):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (items[start:end], device)
        start = end


def load_failed_samples():
    if os.path.exists(FAILED_JSON):
        with open(FAILED_JSON) as f:
            return json.load(f)
    return {}


def save_failed_sample(lang: str, sample_id: str):
    failed = load_failed_samples()
    failed.setdefault(lang, []).append(sample_id)
    with open(FAILED_JSON, "w") as f:
        json.dump(failed, f, indent=2)


def process_item(model, item: TTSItem, run_dir: str, proc_id: str, run: int):
    out_path = os.path.join(run_dir, f"{item.sample_id}.wav")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            wav = model.generate(
                text=item.target_text,
                language_id=item.language_id,
                audio_prompt_path=item.ref_audio_path,
            )
            ta.save(out_path, wav, model.sr)
            return None
        except Exception as e:
            print(f"[{proc_id}] {item.lang} run{run} {item.sample_id} attempt {attempt}: failed - {e}")
    return item.sample_id


def loop(items_device_pair: Tuple[List[TTSItem], int]):
    import traceback
    items, gpu_id = items_device_pair
    proc_id = f"P{gpu_id}-{os.getpid()}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        if hasattr(torch, "compile"):
            try:
                model.t3 = torch.compile(model.t3, mode="reduce-overhead")
                model.s3gen = torch.compile(model.s3gen, mode="reduce-overhead")
                print(f"[{proc_id}] Model compiled for faster inference")
            except Exception as e:
                print(f"[{proc_id}] Model compilation failed: {e}")
    except Exception:
        print(f"[{proc_id}] Failed to load model:\n{traceback.format_exc()}")
        raise

    for run in range(1, NUM_RUNS + 1):
        for item in tqdm(items, desc=f"[{proc_id}] run{run}"):
            run_dir = os.path.join(OUTPUT_DIR, item.lang, f"run{run}")
            os.makedirs(run_dir, exist_ok=True)

            out_path = os.path.join(run_dir, f"{item.sample_id}.wav")
            if os.path.exists(out_path):
                continue

            failed_id = process_item(model, item, run_dir, proc_id, run)
            if failed_id:
                print(f"[{proc_id}] {item.lang} sample {failed_id}: all retries failed")
                save_failed_sample(item.lang, failed_id)


def build_all_items() -> List[TTSItem]:
    with open("languages.json") as f:
        allowed_languages = set(json.load(f)["supported"])

    supported_langs = [
        lang for lang in SUPPORTED_LANGUAGES
        if lang in allowed_languages
        and os.path.exists(os.path.join(DATASET_DIR, lang))
    ]
    print(f"Languages to process: {supported_langs}")

    failed = load_failed_samples()
    all_items: List[TTSItem] = []

    for lang in supported_langs:
        lang_dir = os.path.join(DATASET_DIR, lang)
        audio_dir = os.path.join(lang_dir, "audio")
        try:
            ds = load_from_disk(lang_dir)
        except Exception as e:
            print(f"Skipping {lang}: failed to load dataset - {e}")
            continue

        for row in ds:
            sample_id = row["audio_filename"].replace(".flac", "")

            # Skip permanently failed samples
            if lang in failed and sample_id in failed[lang]:
                continue

            # Skip if ALL runs already done
            all_runs_done = all(
                os.path.exists(os.path.join(OUTPUT_DIR, lang, f"run{run}", f"{sample_id}.wav"))
                for run in range(1, NUM_RUNS + 1)
            )
            if all_runs_done:
                continue

            all_items.append(TTSItem(
                sample_id=sample_id,
                target_text=row["target_text"],
                ref_audio_path=os.path.join(audio_dir, row["audio_filename"]),
                language_id=lang,
                lang=lang,
            ))

    return all_items


if __name__ == "__main__":
    all_items = build_all_items()
    print(f"Total pending items: {len(all_items)}")

    if not all_items:
        print("Nothing to do.")
    else:
        gpu_count = 3
        devices = PROCESSES_PER_GPU * list(range(gpu_count))  
        splits = list(chunks(all_items, devices))

        print(f"Launching {len(devices)} workers across {gpu_count} GPUs ({PROCESSES_PER_GPU} per GPU)")
        for i, (split, gpu_id) in enumerate(splits):
            print(f"  Worker {i} -> GPU {gpu_id}: {len(split)} items")

        with Pool(len(devices)) as pool:
            pool.map(loop, splits)

    print("All done.")
    failed = load_failed_samples()
    if failed:
        print(f"Failed samples saved to {FAILED_JSON}:")
        for lang, samples in failed.items():
            print(f"  {lang}: {len(samples)} failed")
    else:
        print("No failed samples.")
