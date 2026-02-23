import torch  
import soundfile as sf  
import os  
import json  
import multiprocessing as mp  
from dataclasses import dataclass, field  
from typing import List, Dict, Any, Optional  
  
QWEN_LANGUAGE_MAP = {  
    "zh-TW": "Chinese", "zh-CN": "Chinese", "zh-HK": "Chinese",  
    "en": "English", "ja": "Japanese", "ko": "Korean",  
    "de": "German", "fr": "French", "ru": "Russian",  
    "pt": "Portuguese", "es": "Spanish", "it": "Italian",  
}  
  
DATASET_DIR = "vc_dataset_filtered"  
OUTPUT_DIR = "vc_outputs/qwen_1.7b"  
FAILED_JSON = "failed_samples.json"  
NUM_RUNS = 3  
MAX_RETRIES = 3  
BATCH_SIZE = 12  
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  
NUM_BUCKETS = 5  
TOKENS_PER_SECOND = 12  
MAX_TOKEN_BUFFER_FACTOR = 1.5  
MAX_TOKENS_CAP = 1024 
  
@dataclass  
class TTSItem:  
    sample_id: str  
    target_text: str  
    ref_audio_path: str  
    ref_text: str  
    out_path: str  
    text_length: int = 0  

  
def load_failed_samples():  
    if os.path.exists(FAILED_JSON):  
        with open(FAILED_JSON) as f:  
            return json.load(f)  
    return {}  
  
def save_failed_sample(lang, sample_id, lock):  
    with lock:  
        failed = load_failed_samples()  
        if lang not in failed:  
            failed[lang] = []  
        if sample_id not in failed[lang]:  
            failed[lang].append(sample_id)  
            for run in range(1, NUM_RUNS + 1):  
                run_path = os.path.join(OUTPUT_DIR, lang, f"run{run}", f"{sample_id}.wav")  
                if os.path.exists(run_path):  
                    os.remove(run_path)  
                    print(f"[CLEANUP] Deleted {run_path}")  
        with open(FAILED_JSON, "w") as f:  
            json.dump(failed, f, indent=2)  
  
def process_batch(model, batch: List[TTSItem], qwen_lang: str, run_dir: str, lang: str, gpu_id: int, run: int, lock, max_new_tokens: int):   
    sample_ids = [b.sample_id for b in batch]  
    texts = [b.target_text for b in batch]  
    ref_audios = [b.ref_audio_path for b in batch]  
    ref_texts = [b.ref_text for b in batch]  
    out_paths = [b.out_path for b in batch]  
  
    for attempt in range(1, MAX_RETRIES + 1):  
        try:  
            wavs, sr = model.generate_voice_clone(  
                text=texts,  
                language=[qwen_lang] * len(batch),  
                ref_audio=ref_audios,  
                ref_text=ref_texts,  
                max_new_tokens=max_new_tokens,  
            )  
            for wav, out_path in zip(wavs, out_paths):  
                sf.write(out_path, wav, sr)  
            return set()  
        except Exception as e:  
            print(f"[GPU {gpu_id}] {lang} run{run} batch attempt {attempt}: failed - {e}")  
    return set(sample_ids)
  
def create_length_buckets(items: List[TTSItem], num_buckets: int) -> List[List[TTSItem]]:   
    if not items:  
        return []  
    sorted_items = sorted(items, key=lambda x: x.text_length)  
    bucket_size = len(sorted_items) // num_buckets + 1  
    buckets = [sorted_items[i:i + bucket_size] for i in range(0, len(sorted_items), bucket_size)]  
    return buckets  
  
def process_language_with_bucketing(model, ds, audio_dir, qwen_lang, out_lang_dir, lang, gpu_id, run, lock):  
    run_dir = os.path.join(out_lang_dir, f"run{run}")  
    os.makedirs(run_dir, exist_ok=True)  
    failed = load_failed_samples()  
    pending_items: List[TTSItem] = []  
    for row in ds:  
        sample_id = row["audio_filename"].replace(".flac", "")  
        if lang in failed and sample_id in failed[lang]:  
            continue  
        out_path = os.path.join(run_dir, f"{sample_id}.wav")  
        if os.path.exists(out_path):  
            continue  
        item = TTSItem(  
            sample_id=sample_id,  
            target_text=row["target_text"],  
            ref_audio_path=os.path.join(audio_dir, row["audio_filename"]),  
            ref_text=row["source_text"],  
            out_path=out_path,  
        )  
        pending_items.append(item)  
    if not pending_items:  
        print(f"[GPU {gpu_id}] {lang} run{run}: nothing to process")  
        return 0, 0  
  
    input_texts = [model._build_assistant_text(it.target_text) for it in pending_items]  
    input_ids = model._tokenize_texts(input_texts)  # List[Tensor]    
    token_lengths = [tid.shape[1] for tid in input_ids]  
    for item, length in zip(pending_items, token_lengths):  
        item.text_length = length  
  
    # Estimate max_new_tokens from longest tokenized input  
    max_input_len = max(token_lengths) if token_lengths else 0  
    max_new_tokens = min(int(max_input_len * TOKENS_PER_SECOND * MAX_TOKEN_BUFFER_FACTOR), MAX_TOKENS_CAP)  
    print(f"[GPU {gpu_id}] {lang} run{run}: max_input_len={max_input_len} -> max_new_tokens={max_new_tokens}")  
  
  
    # Bucket by token length  
    buckets = create_length_buckets(pending_items, NUM_BUCKETS)  
    print(f"[GPU {gpu_id}] {lang} run{run}: {len(pending_items)} items split into {len(buckets)} buckets")  
    total_processed = 0  
    total_failed = 0  
    from tqdm import tqdm  
    for bucket_idx, bucket in enumerate(buckets):  
        bucket = sorted(bucket, key=lambda x: x.text_length)  
        pbar = tqdm(  
            total=len(bucket),  
            desc=f"[GPU {gpu_id}] {lang} run{run} bucket{bucket_idx+1}/{len(buckets)}",  
            position=gpu_id,  
            leave=False,  
        )  
        for i in range(0, len(bucket), BATCH_SIZE):  
            batch = bucket[i:i + BATCH_SIZE]  
            failed_ids = process_batch(model, batch, qwen_lang, run_dir, lang, gpu_id, run, lock, max_new_tokens)  
            for sid in failed_ids:  
                print(f"[GPU {gpu_id}] {lang} sample {sid}: all retries failed")  
                save_failed_sample(lang, sid, lock)  
                total_failed += 1  
            processed = len(batch) - len(failed_ids)  
            total_processed += processed  
            pbar.update(len(batch))  
            if i % (BATCH_SIZE * 10) == 0:  
                failed = load_failed_samples()  
        pbar.close()  
    return total_processed, total_failed  
  
def process_languages(gpu_id, languages, lock):  
    from qwen_tts import Qwen3TTSModel  
    from datasets import load_from_disk  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    model = Qwen3TTSModel.from_pretrained(  
        MODEL_ID,  
        device_map="cuda:0",  
        dtype=torch.bfloat16,  
        # attn_implementation="flash_attention_2",   
    )  
    for lang in languages:  
        qwen_lang = QWEN_LANGUAGE_MAP[lang]  
        lang_dir = os.path.join(DATASET_DIR, lang)  
        audio_dir = os.path.join(lang_dir, "audio")  
        if not os.path.exists(lang_dir):  
            print(f"[GPU {gpu_id}] {lang}: dataset not found, skipping")  
            continue  
        try:  
            ds = load_from_disk(lang_dir)  
        except Exception as e:  
            print(f"[GPU {gpu_id}] {lang}: failed to load dataset - {e}")  
            continue  
        out_lang_dir = os.path.join(OUTPUT_DIR, lang)  
        os.makedirs(out_lang_dir, exist_ok=True)  
        for run in range(1, NUM_RUNS + 1):  
            processed, failed_count = process_language_with_bucketing(  
                model, ds, audio_dir, qwen_lang, out_lang_dir, lang, gpu_id, run, lock  
            )  
            print(f"[GPU {gpu_id}] {lang} run{run}: completed ({processed} processed, {failed_count} failed)")  
        print(f"[GPU {gpu_id}] {lang}: all runs completed")  
  
if __name__ == "__main__":  
    with open("languages.json") as f:  
        supported_languages = set(json.load(f)["supported"])  
    supported_langs = [  
        lang for lang in QWEN_LANGUAGE_MAP  
        if lang in supported_languages  
        and os.path.exists(os.path.join(DATASET_DIR, lang))  
    ]  
    print(f"Languages to process: {supported_langs}")  
    gpu0_langs = supported_langs[0::3]  
    gpu1_langs = supported_langs[1::3]  
    gpu2_langs = supported_langs[2::3]  
    print(f"GPU 0: {gpu0_langs}")  
    print(f"GPU 1: {gpu1_langs}")  
    print(f"GPU 2: {gpu2_langs}")  
    lock = mp.Manager().Lock()  
    processes = []  
    for gpu_id, langs in enumerate([gpu0_langs, gpu1_langs, gpu2_langs]):  
        if langs:  
            p = mp.Process(target=process_languages, args=(gpu_id, langs, lock))  
            p.start()  
            processes.append(p)  
    for p in processes:  
        p.join()  
    print("All done.")  
    failed = load_failed_samples()  
    if failed:  
        print(f"Failed samples saved to {FAILED_JSON}:")  
        for lang, samples in failed.items():  
            print(f"  {lang}: {len(samples)} failed")  
    else:  
        print("No failed samples.")







# import torch
# import soundfile as sf
# import os
# import json
# import multiprocessing as mp

# QWEN_LANGUAGE_MAP = {
#     "zh-TW": "Chinese",
#     "zh-CN": "Chinese",
#     "zh-HK": "Chinese",
#     "en": "English",
#     "ja": "Japanese",
#     "ko": "Korean",
#     "de": "German",
#     "fr": "French",
#     "ru": "Russian",
#     "pt": "Portuguese",
#     "es": "Spanish",
#     "it": "Italian",
# }

# DATASET_DIR = "vc_dataset_filtered"
# OUTPUT_DIR = "vc_outputs/qwen_1.7b"
# FAILED_JSON = "failed_samples.json"
# NUM_RUNS = 3
# MAX_RETRIES = 3
# BATCH_SIZE = 15
# MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


# def load_failed_samples():
#     if os.path.exists(FAILED_JSON):
#         with open(FAILED_JSON) as f:
#             return json.load(f)
#     return {}


# def save_failed_sample(lang, sample_id, lock):
#     with lock:
#         failed = load_failed_samples()
#         if lang not in failed:
#             failed[lang] = []
#         if sample_id not in failed[lang]:
#             failed[lang].append(sample_id)
#             for run in range(1, NUM_RUNS + 1):
#                 run_path = os.path.join(OUTPUT_DIR, lang, f"run{run}", f"{sample_id}.wav")
#                 if os.path.exists(run_path):
#                     os.remove(run_path)
#                     print(f"[CLEANUP] Deleted {run_path}")
#         with open(FAILED_JSON, "w") as f:
#             json.dump(failed, f, indent=2)


# def process_batch(model, batch, qwen_lang, run_dir, lang, gpu_id, run, lock):
#     sample_ids = [b["sample_id"] for b in batch]
#     texts = [b["target_text"] for b in batch]
#     ref_audios = [b["ref_audio_path"] for b in batch]
#     ref_texts = [b["ref_text"] for b in batch]
#     out_paths = [b["out_path"] for b in batch]

#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             wavs, sr = model.generate_voice_clone(
#                 text=texts,
#                 language=[qwen_lang] * len(batch),
#                 ref_audio=ref_audios,
#                 ref_text=ref_texts,
#                 max_new_tokens=700,
#             )

#             for wav, out_path in zip(wavs, out_paths):
#                 sf.write(out_path, wav, sr)

#             return set()

#         except Exception as e:
#             print(f"[GPU {gpu_id}] {lang} run{run} batch attempt {attempt}: failed - {e}")

#     return set(sample_ids)


# def process_languages(gpu_id, languages, lock):
#     from tqdm import tqdm
#     from qwen_tts import Qwen3TTSModel
#     from datasets import load_from_disk

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     model = Qwen3TTSModel.from_pretrained(
#         MODEL_ID,
#         device_map="cuda:0",
#         dtype=torch.bfloat16,
#     )

#     for lang in languages:
#         qwen_lang = QWEN_LANGUAGE_MAP[lang]
#         lang_dir = os.path.join(DATASET_DIR, lang)
#         audio_dir = os.path.join(lang_dir, "audio")

#         if not os.path.exists(lang_dir):
#             print(f"[GPU {gpu_id}] {lang}: dataset not found, skipping")
#             continue

#         try:
#             ds = load_from_disk(lang_dir)
#         except Exception as e:
#             print(f"[GPU {gpu_id}] {lang}: failed to load dataset - {e}")
#             continue

#         out_lang_dir = os.path.join(OUTPUT_DIR, lang)

#         for run in range(1, NUM_RUNS + 1):
#             run_dir = os.path.join(out_lang_dir, f"run{run}")
#             os.makedirs(run_dir, exist_ok=True)

#             pbar = tqdm(
#                 total=len(ds),
#                 desc=f"[GPU {gpu_id}] {lang} run{run}",
#                 position=gpu_id,
#                 leave=True,
#             )

#             batch = []
#             failed = load_failed_samples()

#             for i, row in enumerate(ds):
#                 sample_id = row["audio_filename"].replace(".flac", "")

#                 if lang in failed and sample_id in failed[lang]:
#                     pbar.update(1)
#                     continue

#                 out_path = os.path.join(run_dir, f"{sample_id}.wav")
#                 if os.path.exists(out_path):
#                     pbar.update(1)
#                     continue

#                 batch.append({
#                     "sample_id": sample_id,
#                     "target_text": row["target_text"],
#                     "ref_audio_path": os.path.join(audio_dir, row["audio_filename"]),
#                     "ref_text": row["source_text"],
#                     "out_path": out_path,
#                 })

#                 if len(batch) == BATCH_SIZE:
#                     failed_ids = process_batch(model, batch, qwen_lang, run_dir, lang, gpu_id, run, lock)
#                     for sid in failed_ids:
#                         print(f"[GPU {gpu_id}] {lang} sample {sid}: all retries failed, adding to failed list")
#                         save_failed_sample(lang, sid, lock)
#                     pbar.update(len(batch))
#                     batch = []
#                     failed = load_failed_samples()

#             if batch:
#                 failed_ids = process_batch(model, batch, qwen_lang, run_dir, lang, gpu_id, run, lock)
#                 for sid in failed_ids:
#                     print(f"[GPU {gpu_id}] {lang} sample {sid}: all retries failed, adding to failed list")
#                     save_failed_sample(lang, sid, lock)
#                 pbar.update(len(batch))

#             pbar.close()
#             print(f"[GPU {gpu_id}] {lang} run{run}: completed")

#         print(f"[GPU {gpu_id}] {lang}: all runs completed")


# if __name__ == "__main__":
#     with open("languages.json") as f:
#         supported_languages = set(json.load(f)["supported"])

#     supported_langs = [
#         lang for lang in QWEN_LANGUAGE_MAP
#         if lang in supported_languages
#         and os.path.exists(os.path.join(DATASET_DIR, lang))
#     ]

#     print(f"Languages to process: {supported_langs}")

#     gpu0_langs = supported_langs[0::3]
#     gpu1_langs = supported_langs[1::3]
#     gpu2_langs = supported_langs[2::3]

#     print(f"GPU 0: {gpu0_langs}")
#     print(f"GPU 1: {gpu1_langs}")
#     print(f"GPU 2: {gpu2_langs}")

#     lock = mp.Manager().Lock()

#     processes = []
#     for gpu_id, langs in enumerate([gpu0_langs, gpu1_langs, gpu2_langs]):
#         if langs:
#             p = mp.Process(target=process_languages, args=(gpu_id, langs, lock))
#             p.start()
#             processes.append(p)

#     for p in processes:
#         p.join()

#     print("All done.")
#     failed = load_failed_samples()
#     if failed:
#         print(f"Failed samples saved to {FAILED_JSON}:")
#         for lang, samples in failed.items():
#             print(f"  {lang}: {len(samples)} failed")
#     else:
#         print("No failed samples.")