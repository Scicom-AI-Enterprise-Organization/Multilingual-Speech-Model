import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker

import os
import json
import torch
import torchaudio as ta
import multiprocessing as mp
from dataclasses import dataclass
from typing import List
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


def chunks(data, devices):
    """Split data evenly across devices."""
    chunk_size = len(data) // len(devices)
    remainder = len(data) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (data[start:end], devices[i])
        start = end


def load_failed_samples():
    if os.path.exists(FAILED_JSON):
        with open(FAILED_JSON) as f:
            return json.load(f)
    return {}


def save_failed_sample(lang, sample_id, run, lock):
    """Save failed sample per run without deleting other runs."""
    with lock:
        failed = load_failed_samples()
        failed.setdefault(lang, {}).setdefault(f"run{run}", []).append(sample_id)
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
            print(
                f"[{proc_id}] {item.language_id} run{run} "
                f"{item.sample_id} attempt {attempt} failed: {e}"
            )

    return item.sample_id


def process_chunk(items: List[TTSItem], gpu_id: int, run: int, lock):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc_id = f"P{gpu_id}"

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as e:
        print(f"[{proc_id}] Model load failed: {e}")
        return

    print(f"[{proc_id}] Loaded model on GPU {gpu_id} for run{run} ({len(items)} samples)")

    for item in items:
        out_lang_dir = os.path.join(OUTPUT_DIR, item.language_id)
        run_dir = os.path.join(out_lang_dir, f"run{run}")
        os.makedirs(run_dir, exist_ok=True)

        out_path = os.path.join(run_dir, f"{item.sample_id}.wav")
        if os.path.exists(out_path):
            continue

        failed_id = process_item(model, item, run_dir, proc_id, run)
        if failed_id:
            print(f"[{proc_id}] Failed completely: {failed_id}")
            save_failed_sample(item.language_id, failed_id, run, lock)

    print(f"[{proc_id}] run{run} completed")


if __name__ == "__main__":

    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("No GPUs detected.")

    print(f"Detected GPUs: {gpu_ids}")
    print(f"Processes per GPU: {PROCESSES_PER_GPU}")

    devices = gpu_ids * PROCESSES_PER_GPU
    print(f"Total processes launching per run: {len(devices)}")

    with open("languages.json") as f:
        allowed_languages = set(json.load(f)["supported"])

    supported_langs = [
        lang for lang in SUPPORTED_LANGUAGES
        if lang in allowed_languages
        and os.path.exists(os.path.join(DATASET_DIR, lang))
    ]
    print(f"Languages to process: {supported_langs}")

    all_items = []
    for lang in supported_langs:
        lang_dir = os.path.join(DATASET_DIR, lang)
        audio_dir = os.path.join(lang_dir, "audio")
        ds = load_from_disk(lang_dir)

        for row in ds:
            sample_id = row["audio_filename"].replace(".flac", "")
            item = TTSItem(
                sample_id=sample_id,
                target_text=row["target_text"],
                ref_audio_path=os.path.join(audio_dir, row["audio_filename"]),
                language_id=lang,
            )
            all_items.append(item)

    print(f"Total samples collected: {len(all_items)}")
    if not all_items:
        print("Nothing to process.")
        exit(0)

    lock = mp.Manager().Lock()

    for run in range(1, NUM_RUNS + 1):
        print(f"Starting run {run}...")
        item_splits = list(chunks(all_items, devices))
        processes = []

        for items_chunk, gpu_id in item_splits:
            if not items_chunk:
                continue
            p = mp.Process(
                target=process_chunk,
                args=(items_chunk, gpu_id, run, lock),
                daemon=True,
            )
            p.start()
            processes.append(p)
            print(f"Started process on GPU {gpu_id} for run{run} with {len(items_chunk)} samples")

        for p in processes:
            p.join()

    print("All runs completed.")

    failed = load_failed_samples()
    if failed:
        print(f"Failed samples saved to {FAILED_JSON}:")
        for lang, runs in failed.items():
            for run, samples in runs.items():
                print(f"  {lang} {run}: {len(samples)} failed")
    else:
        print("No failed samples.")
