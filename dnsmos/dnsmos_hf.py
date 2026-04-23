"""
DNSMOS evaluation pipeline for HuggingFace dataset malaysia-ai/Multilingual-TTS.

Outer loop (main process): iterate zip files sequentially (Producer-Consumer pattern)
  - Stage 0: Download + unzip
  - Stage 1: Preprocess audio -> float32 numpy   (few workers, spawned per zip)
  - Stage 2: ONNX inference / MOS scoring         (many workers, persistent)
  - Stage 3: Append results to JSONL              (1 worker,    persistent)
  - Cleanup: delete extracted dir to reclaim storage
"""
import multiprocessing as mp
import os
import json
import random
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download

SAMPLING_RATE = 16000
INPUT_LENGTH  = 9.01
SUPPORTED_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

BATCH_END    = '__BATCH_END__'    # end of one zip; workers stay alive
PIPELINE_END = '__PIPELINE_END__' # full shutdown

# Polyfit coefficients — module-level so they are created once per worker process, not per call
_P_OVR_PERS = np.poly1d([-0.00533021,  0.005101,    1.18058466, -0.11236046])
_P_SIG_PERS = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
_P_BAK_PERS = np.poly1d([-0.04976499,  0.44276479, -0.1644611,   0.96883132])
_P_OVR_NRML = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
_P_SIG_NRML = np.poly1d([-0.08397278,  1.22083953,  0.0052439])
_P_BAK_NRML = np.poly1d([-0.13166888,  1.60915514, -0.39604546])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_processed(output_path):
    """Return the set of filepaths already written to the output JSONL."""
    processed = set()
    p = Path(output_path)
    if not p.exists():
        return processed
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if 'filepath' in r:
                    processed.add(r['filepath'])
            except json.JSONDecodeError:
                pass
    return processed


def load_completed_zips(cache_path):
    """Return the set of zip names (HF repo paths) that have been fully processed."""
    completed = set()
    p = Path(cache_path)
    if not p.exists():
        return completed
    with open(p) as f:
        for line in f:
            name = line.strip()
            if name:
                completed.add(name)
    return completed


def mark_zip_complete(cache_path, zip_name):
    """Append zip_name to the completed-zips cache file."""
    with open(cache_path, 'a') as f:
        f.write(zip_name + '\n')


def _get_polyfit_val(sig, bak, ovr, personalized=True):
    if personalized:
        return _P_SIG_PERS(sig), _P_BAK_PERS(bak), _P_OVR_PERS(ovr)
    return _P_SIG_NRML(sig), _P_BAK_NRML(bak), _P_OVR_NRML(ovr)


def download_and_extract(zip_name, repo_id, token, base_dir):
    """Download one zip from HF into base_dir, extract it there, then delete the zip.

    e.g. zip_name="folder/a.zip" -> downloads base_dir/.../a.zip, extracts to base_dir/a/, deletes zip.
    Returns (extract_dir, None) on success or (None, exception) on failure.
    """
    try:
        zip_stem    = Path(zip_name).stem               # "a" from "folder/a.zip"
        extract_dir = Path(base_dir) / zip_stem         # base_dir/a/
        extract_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # hf_hub_download may nest subdirs under base_dir; we capture the returned path
        local_zip = hf_hub_download(
            repo_id=repo_id,
            filename=zip_name,
            repo_type="dataset",
            token=token,
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(str(extract_dir))
        # Remove zip file and any empty parent dirs hf_hub_download may have created
        local_zip_path = Path(local_zip)
        local_zip_path.unlink()
        parent = local_zip_path.parent
        while parent.resolve() != Path(base_dir).resolve():
            try:
                parent.rmdir()   # only succeeds if empty
                parent = parent.parent
            except OSError:
                break
        return extract_dir, None
    except Exception as e:
        return None, e

# TO-BE-DELETED
def load_subset_metadata(subset_name, all_files, repo_id, token, base_dir):
    """Download parquet files for the subset and return {audio_filename: {text, speaker}}.

    The subset name is derived from the zip path's top-level folder
    (e.g. "norwegian-100h" from "norwegian-100h/train.zip").
    Returns an empty dict if no matching parquet is found.
    """
    parquet_files = [f for f in all_files if f.endswith('.parquet') and subset_name in f]
    if not parquet_files:
        tqdm.write(f"[Metadata] No parquet found for '{subset_name}', text/speaker will be empty.")
        return {}

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    dfs = []
    for pq_path in parquet_files:
        local_pq = hf_hub_download(
            repo_id=repo_id, filename=pq_path, repo_type="dataset",
            token=token, local_dir=str(base_dir), local_dir_use_symlinks=False,
        )
        try:
            dfs.append(pd.read_parquet(local_pq, columns=['audio_filename', 'text', 'speaker']))
        finally:
            local_pq_path = Path(local_pq)
            local_pq_path.unlink(missing_ok=True)
            parent = local_pq_path.parent
            while parent.resolve() != Path(base_dir).resolve():
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

    if not dfs:
        return {}

    df = pd.concat(dfs, ignore_index=True)
    records = df.to_dict('records')
    del df  # free DataFrame before building the return dict
    return {
        str(r['audio_filename']): {
            'text':    str(r.get('text',    '')),
            'speaker': str(r.get('speaker', '')),
        }
        for r in records
    }


# ---------------------------------------------------------------------------
# Stage 1 – Preprocess worker (spawned fresh per zip, low CPU)
# ---------------------------------------------------------------------------
def preprocess_worker(file_shard, compute_queue, c_pre):
    """Preprocess a shard of audio files and push float32 numpy arrays downstream.

    metadata: {audio_filename (relative to extract_dir) -> {text, speaker}}
    extract_dir: base path used to compute the relative lookup key.
    """
    for audio_path in file_shard:
        try:
            aud, input_fs = sf.read(audio_path)
            if aud.ndim > 1:
                aud = np.mean(aud, axis=1)
            if input_fs != SAMPLING_RATE:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE, res_type='soxr_hq')
            else:
                audio = aud
            del aud  # free original read buffer; audio holds resampled (or same) data

            actual_audio_len = len(audio)
            len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
            if len(audio) < len_samples:
                repeats = -(-len_samples // len(audio))  # ceiling division
                audio = np.tile(audio, repeats)[:len_samples].copy()  # .copy() breaks tile's base ref

            compute_queue.put({
                'audio':     np.ascontiguousarray(audio, dtype=np.float32),  # no-op if already float32 C-contiguous
                'filepath':  audio_path,
                'audio_len': float(actual_audio_len),
            })
            del audio  # queued item owns its own array now
            with c_pre.get_lock():
                c_pre.value += 1
        except Exception as e:
            tqdm.write(f"[Preprocess] Skipping {audio_path}: {e}")


# ---------------------------------------------------------------------------
# Stage 2 – Inference worker (persistent across zips, high CPU)
# https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
# ---------------------------------------------------------------------------
def compute_worker(model_path, compute_queue, save_queue, c_comp):
    """Load ONNX once. Forward BATCH_END/PIPELINE_END; exit on PIPELINE_END."""
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    session  = ort.InferenceSession(model_path, sess_options=sess_opts)
    fs       = SAMPLING_RATE
    len_samp = int(INPUT_LENGTH * fs)

    while True:
        item = compute_queue.get()

        if item in (BATCH_END, PIPELINE_END):
            save_queue.put(item)
            if item == PIPELINE_END:
                break
            continue

        audio            = item['audio']
        filepath         = item['filepath']
        actual_audio_len = item['audio_len']
        del item  # release the dict; audio/filepath/actual_audio_len hold all needed refs
        text    = ""
        speaker = ""
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop       = fs

        # Loop over hops one at a time — avoids stacking all hops into a giant array.
        # For a 10-min file this would be ~591 hops × 563 KB = ~340 MB if batched.
        n_valid = 0
        sig_r_sum = bak_r_sum = ovr_r_sum = 0.0
        sig_sum   = bak_sum   = ovr_sum   = 0.0
        for idx in range(num_hops):
            seg = audio[int(idx * hop) : int((idx + INPUT_LENGTH) * hop)]
            if len(seg) < len_samp:
                continue
            out = session.run(None, {'input_1': seg[np.newaxis, :]})[0][0]  # (3,)
            s_r, b_r, o_r = float(out[0]), float(out[1]), float(out[2])
            s, b, o = _get_polyfit_val(s_r, b_r, o_r)
            sig_r_sum += s_r  
            bak_r_sum += b_r  
            ovr_r_sum += o_r
            sig_sum   += s    
            bak_sum   += b     
            ovr_sum   += o
            n_valid   += 1
        del audio

        if n_valid == 0:
            with c_comp.get_lock():
                c_comp.value += 1
            continue

        save_queue.put({
            'filepath':   filepath,
            'text':       text,
            'speaker':    speaker,
            'len_in_sec': float(actual_audio_len / fs),
            'sr':         fs,
            'num_hops':   n_valid,
            'OVRL_raw': ovr_r_sum / n_valid, 'SIG_raw': sig_r_sum / n_valid, 'BAK_raw': bak_r_sum / n_valid,
            'OVRL':     ovr_sum   / n_valid, 'SIG':     sig_sum   / n_valid, 'BAK':     bak_sum   / n_valid,
        })
        with c_comp.get_lock():
            c_comp.value += 1


# ---------------------------------------------------------------------------
# Stage 3 – Save worker (persistent across zips, low CPU)
# ---------------------------------------------------------------------------

def save_worker(save_queue, output_path, n_compute, c_save, batch_done):
    """Append results to JSONL. Sets batch_done after each BATCH_END round; exits on PIPELINE_END."""
    batch_end_seen    = 0
    pipeline_end_seen = 0
    with open(output_path, 'a') as f:
        while True:
            item = save_queue.get()
            if item == BATCH_END:
                batch_end_seen += 1
                if batch_end_seen == n_compute:
                    batch_end_seen = 0
                    batch_done.set()          # unblock main process -> cleanup + next zip
            elif item == PIPELINE_END:
                pipeline_end_seen += 1
                if pipeline_end_seen == n_compute:
                    break
            else:
                f.write(json.dumps(item) + '\n')
                f.flush()
                with c_save.get_lock():
                    c_save.value += 1

skip_zips = [
    
]
# skip_zips = [
#     'assamese-asr-dataset_audio.zip', 
#     'indian-english-nptel-v0_audio.zip', 
#     'kazakh_speech_dataset_ksd_audio.zip', 
#     'ghana-english-asr-2700hrs_audio.zip', 
#     'gemini-flash-2.0-speech_audio.zip', 
#     'urdu-tts-speaker3_audio.zip', 
#     'zeroth_korean_ipa_audio.zip', 
#     'kazakh_speech_mfa_punctuation_audio.zip', 
#     'afrikaans-speech-dataset_audio.zip', 
#     'elevenlabs_ru_audio.zip', 
#     'VietSpeech_audio.zip', 
#     'IndicTTS_audio.zip', 
#     'voxbox_audio.zip', 
#     'hungarian-speech-dataset_audio.zip', 
#     'Enigma-Dataset_audio.zip', 
#     'IndicTTS_v2_audio.zip', 
#     'Vaani_audio.zip', 
#     'FalAR_audio.zip', 
#     'mls_dutch_audio.zip', 
#     'arknights_voices_audio.zip', 
#     'Czech-Speech-Monospeaker-Honza_audio.zip', 
#     'NorthTTS_audio_audio.zip', 
#     'google_audio_audio.zip', 
#     'WolneLektury-TTS-Polish_audio.zip', 
#     'malaysian-emilia-v2_audio.zip', 
#     'AISHELL3_audio.zip', 
#     'IndicTTS_English_audio.zip', 
#     'TTS-Hungarian_audio.zip', 
#     'ToneWebinars_audio.zip', 
#     'hausa-tts-22k_audio.zip', 
#     'common-voice-22_audio.zip', 
#     'singlish-speaker_audio.zip', 
#     'emilia_zh_audio.zip', 
#     'JSS_audio.zip', 
#     'mgb2-arabic_audio.zip', 
#     'malay-audiobook_audio.zip', 
#     'WaxalNLP_audio.zip', 
#     'EA-UD-DI_audio.zip', 
#     'viVoice_audio.zip', 
#     'everyayah-phonemes_audio.zip', 
#     'KSS_audio.zip', 
#     'japanese-Eroge-Voice-V2_audio.zip', 
#     'japanese-anime-speech-v2_audio.zip', 
#     'malaysian-chinese-emilia_audio.zip', 
#     'uzbekvoice_audio.zip', 
#     'CommonVoice22_Sidon_audio.zip', 
#     'Malaysian-TTS-v2_audio.zip', 
#     'macedonian_audio.zip', 
#     'MsceneSpeech_audio.zip', 
#     'YouTube-Cantonese_audio.zip', 
#     'khursanirevo_chatter_audio.zip', 
#     'voices_jp_audio.zip', 
#     'ftspeech_audio.zip', 
#     'nepali-slr_audio.zip', 
#     'egyptian-arabic-400k_audio.zip', 
#     'coral-v3_audio.zip', 
#     'Thai-dialect-corpus_audio.zip', 
#     'sova_rudevices_audio.zip', 
#     'indicvoices_r_audio.zip', 
#     'Hindi-1482Hrs_audio.zip', 
#     'openslr-140-hq-Kazakh_audio.zip', 
#     'marathi-speech-dataset_audio.zip', 
#     'omnilingual-asr-corpus_audio.zip', 
#     'MASC-Arabic_audio.zip', 
#     'anv_data_ke_audio.zip', 
#     'kazakh-stt_audio.zip', 
#     'naijavoices-dataset_audio.zip', 
#     'cml-tts_audio.zip', 
#     'KeSpeech_audio.zip', 
#     'libritts_r_filtered_audio.zip',
#     '700h-tr-turkish-text-to-speech_audio.zip',
#     '9jalingo-hausa_audio.zip',
#     '9jalingo-igbo_audio.zip',
#     'Alexis_audio.zip',
#     'CommonPhoneDataset_audio-0-0.zip',
#     'CommonPhoneDataset_audio-2-0.zip',
#     'CommonPhoneDataset_audio-3-0.zip',
#     'CommonPhoneDataset_audio-5-0.zip',
#     'CommonPhoneDataset_audio.zip',
#     'DarijaTTS-clean_audio.zip',
#     'Dataset-Text-To-Speech-Indonesia_audio.zip',
#     'Enigma-Dataset_audio-1-0.zip',
#     'IndicTTS_Manipuri_audio.zip',
#     'IndicTTS_Punjabi_audio.zip',
#     'IndicTTS_Tamil_audio.zip',
#     'IndicTTS_Telugu_MultiSpeaker_audio.zip',
#     'Japanese-Anime-Speech-v2_audio-0-0.zip',
#     'Japanese-Anime-Speech-v2_audio-1-0.zip',
#     'Japanese-Anime-Speech-v2_audio.zip',
#     'Lahaja_audio.zip',
#     'MASC-Arabic_audio-4-0.zip',
#     'Nanchang_Dialect_Conversational_Speech_Corpus_audio.zip',
#     'NepaliONE-tts_audio.zip',
#     'SPRING_INX_Malayalam_R1_audio.zip',
#     'StoryTTS_audio.zip',
#     'Tibetan-0310_audio.zip',
#     'ToneWebinars_audio-2-0.zip',
#     'WaxalNLP-3-1.zip',
#     'WaxalNLP-4-2.zip',
#     'YouTube-Cantonese_audio-3-0.zip',
#     'YouTube-Cantonese_audio-4-1.zip',
#     'Zhengzhou_Dialect_Conversational_Speech_Corpus_audio.zip',
#     'afrispeech_afrikaans_audio.zip',
#     'afvoices_audio.zip',
#     'amharic_cleaned_testset_verified_audio.zip',
#     'anv_data_ke_mas_audio.zip',
#     'arknights_voices_en_audio.zip',
#     'assamese_speech_dataset1_audio.zip',
#     'azerbaijani-audiobooks_audio.zip',
#     'azerbaijani-tts-dataset_audio.zip',
#     'biggest-ru-book_audio.zip',
#     'bulgarian_tts_audio.zip',
#     'camoes_SI_audio.zip',
#     'egyptian-arabic-400k_audio-2-0.zip',
#     'egyptian-arabic-400k_audio-6-0.zip',
#     'expresso_audio.zip',
#     'gemini-flash-2.0-speech_data_audio.zip',
#     'google-marathi_audio.zip',
#     'indicvoices_r_Marathi_audio.zip',
#     'indicvoices_r_Punjabi_audio.zip',
#     'indicvoices_r_Tamil_audio.zip',
#     'libritts_r_filtered_clean.zip',
#     'marathi_asr_dataset_audio.zip',
#     'samromur_children_audio.zip',
#     'shrutilipi_sanskrit_audio.zip',
#     'turkish_male_audio.zip',
#     'ukrainian-speech-dataset_audio-5-0.zip',
#     'urdu-voice-dataset_audio.zip',
#     'uzbekvoice-2k-each-accent_audio-2-0.zip',
#     'uzbekvoice-2k-each-accent_audio-6-0.zip',
#     'za-african-next-voices_audio-4-0.zip',
#     'za-african-next-voices_audio-5-0.zip',
#     'za-african-next-voices_audio-9-0.zip'
# ]
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNSMOS HF pipeline — malaysia-ai/Multilingual-TTS")
    parser.add_argument("--repo",         type=str, default="malaysia-ai/Multilingual-TTS",
                        help="HuggingFace dataset repo id")
    parser.add_argument("--token",        type=str, default=None,
                        help="HuggingFace auth token (or set HF_TOKEN env var)")
    parser.add_argument("--base_dir",     type=str, default="./",
                        help="Directory where zips are downloaded and extracted (extracted folder deleted after each zip)")
    parser.add_argument("--model",        type=str, default="sig_bak_ovr.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--output",       type=str, default="results_hf.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--n_preprocess", type=int, default=1,
                        help="Stage-1 workers per zip (low CPU)")
    parser.add_argument("--n_compute",    type=int, default=None,
                        help="Stage-2 workers, persistent (high CPU); default: cpu_count - n_preprocess - 3")
    parser.add_argument("--zip_cache",   type=str, default=None,
                        help="File recording fully-completed zip names (auto-derived from --output if omitted)")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Randomly sample N zip files to process (default: process all)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for --sample (default: 42)")
    parser.add_argument("--slice",       type=str, default=None,
                        help="Slice the zip file list before processing, e.g. '0:5' or '10:20' (Python slice syntax)")
    args = parser.parse_args()

    if args.zip_cache is None:
        args.zip_cache = str(Path(args.output).with_suffix('')) + '_completed_zips.txt'

    token = args.token or os.environ.get("HF_TOKEN")

    if args.n_compute is None:
        args.n_compute = max(1, os.cpu_count() - args.n_preprocess - 3)

    # --- List zip files from HF repo, exclude neucodec paths ---
    tqdm.write(f"Listing files in {args.repo} ...")
    api       = HfApi()
    all_files = list(api.list_repo_files(repo_id=args.repo, repo_type="dataset", token=token))
    zip_files = [
        f for f in all_files
        if f.endswith('.zip') and 'neucodec' not in f.lower() and f not in skip_zips
    ]
    if not zip_files:
        tqdm.write("No zip files found after neucodec and skip_zips filter.")
        raise SystemExit(1)
    tqdm.write(f"{len(zip_files)} zip file(s) to process (neucodec excluded).")
    tqdm.write(f"Workers — preprocess: {args.n_preprocess}, compute: {args.n_compute}, save: 1")

    # --- Optional random sampling (applied to full list BEFORE cache filter for determinism) ---
    if args.sample is not None:
        random.seed(args.seed)
        n = min(args.sample, len(zip_files))
        zip_files = random.sample(zip_files, n)
        tqdm.write(f"Sampled {n} zip(s) from full list (seed={args.seed}) — same seed always picks the same zips.")

    # --- Optional slice (applied after sample if both are given) ---
    if args.slice is not None:
        parts = args.slice.split(':')
        if len(parts) not in (2, 3):
            tqdm.write(f"Invalid --slice value '{args.slice}'. Expected format: 'start:stop' or 'start:stop:step'.")
            raise SystemExit(1)
        try:
            sl = slice(*[int(p) if p else None for p in parts])
        except ValueError:
            tqdm.write(f"Invalid --slice value '{args.slice}'. All parts must be integers.")
            raise SystemExit(1)
        zip_files = zip_files[sl]
        tqdm.write(f"Sliced to {len(zip_files)} zip(s) using [{args.slice}].")

    # --- Zip-level cache: skip already-completed zips within the (possibly sampled) list ---
    completed_zips = load_completed_zips(args.zip_cache)
    if completed_zips:
        tqdm.write(f"Zip cache: {len(completed_zips)} zip(s) already complete, skipping.")
    zip_files = [z for z in zip_files if z not in completed_zips]
    if not zip_files:
        tqdm.write("All zips already complete. Nothing to do.")
        raise SystemExit(0)
    tqdm.write(f"{len(zip_files)} zip(s) remaining after cache filter.")
    
    # Randomize zip order 
    random.shuffle(zip_files)

    # --- Resume: load already-processed filepaths (for partial zips) ---
    processed = load_processed(args.output)
    if processed:
        tqdm.write(f"Resuming: {len(processed)} file(s) already processed, will be skipped.")

    # --- Shared counters and sync primitives ---
    c_pre      = mp.Value('i', 0)
    c_comp     = mp.Value('i', 0)
    c_save     = mp.Value('i', 0)
    batch_done = mp.Event()

    compute_queue = mp.Queue(maxsize=args.n_compute * 8)
    save_queue    = mp.Queue(maxsize=args.n_compute * 8)

    # --- Stage 2: start persistent inference workers ---
    compute_procs = [
        mp.Process(
            target=compute_worker,
            args=(args.model, compute_queue, save_queue, c_comp),
            daemon=True,
        )
        for _ in range(args.n_compute)
    ]
    for p in compute_procs:
        p.start()

    # --- Stage 3: start persistent save worker ---
    save_proc = mp.Process(
        target=save_worker,
        args=(save_queue, args.output, args.n_compute, c_save, batch_done),
        daemon=True,
    )
    save_proc.start()

    # --- Outer loop: one zip at a time ---
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    zip_bar = tqdm(total=len(zip_files), desc="Zips     ", position=0, leave=True)

    # Prefetch: one background thread downloads the next zip while the current is being processed.
    # This hides download latency almost entirely.
    dl_executor = ThreadPoolExecutor(max_workers=1)
    _dl = lambda z: download_and_extract(z, args.repo, token, str(base_dir))
    # Pre-submit the first zip so the executor is warm from the start
    current_future = dl_executor.submit(_dl, zip_files[0])

    for i, zip_name in enumerate(zip_files):

        # Kick off the next download immediately (runs while we process current)
        if i + 1 < len(zip_files):
            next_future = dl_executor.submit(_dl, zip_files[i + 1])
        else:
            next_future = None

        # Stage 0: wait for current zip to finish downloading/extracting
        tqdm.write(f"[Download] {zip_name}")
        extract_dir, err = current_future.result()
        current_future = next_future

        if err:
            tqdm.write(f"[Download] Failed {zip_name}: {err}")
            zip_bar.update(1)
            continue

        audio_files = [
            str(p) for p in extract_dir.rglob('*')
            if p.suffix.lower() in SUPPORTED_EXT
        ]

        # Skip already-processed files (resume support)
        audio_files = [f for f in audio_files if f not in processed]

        if not audio_files:
            tqdm.write(f"[Skip] {zip_name} — all files already processed.")
            shutil.rmtree(str(extract_dir), ignore_errors=True)
            mark_zip_complete(args.zip_cache, zip_name)
            zip_bar.update(1)
            continue

        n_files = len(audio_files)

        # Load text/speaker metadata from parquet for this subset
        # subset_name = Path(zip_name).parts[0] if len(Path(zip_name).parts) > 1 else Path(zip_name).stem
        # metadata = load_subset_metadata(subset_name, all_files, args.repo, token, str(base_dir))

        # Per-zip progress bars (leave=False so they clear after each zip)
        bar_pre  = tqdm(total=n_files, desc="  Stage 1 Preprocess", position=1, leave=False)
        bar_comp = tqdm(total=n_files, desc="  Stage 2 Inference ", position=2, leave=False)
        bar_save = tqdm(total=n_files, desc="  Stage 3 Save      ", position=3, leave=False)

        # Reset per-zip counters
        with c_pre.get_lock():  c_pre.value  = 0
        with c_comp.get_lock(): c_comp.value = 0
        with c_save.get_lock(): c_save.value = 0
        batch_done.clear()

        # Stage 1: spawn preprocess workers for this zip
        n_pre  = min(args.n_preprocess, n_files)
        shards = [audio_files[i::n_pre] for i in range(n_pre)]
        pre_procs = [
            mp.Process(target=preprocess_worker,
                       args=(shard, compute_queue, c_pre),
                       daemon=True)
            for shard in shards
        ]
        for p in pre_procs:
            p.start()

        # Release metadata from main process — child processes already received their pickled copy
        # del metadata

        # Join Stage 1 in a background thread so the main process can keep updating bars
        stage1_done = threading.Event()
        def _join_stage1():
            for p in pre_procs:
                p.join()
                p.close()  # explicitly release internal pipe fd; without this fds accumulate across zips
            stage1_done.set()
        threading.Thread(target=_join_stage1, daemon=True).start()

        counters = [c_pre, c_comp, c_save]
        bars     = [bar_pre, bar_comp, bar_save]
        prev     = [0, 0, 0]

        def _tick():
            for i, (bar, counter) in enumerate(zip(bars, counters)):
                cur = counter.value
                inc = cur - prev[i]
                if inc:
                    bar.update(inc)
                    prev[i] = cur

        # Phase 1: poll while Stage 1 workers are preprocessing
        while not stage1_done.wait(timeout=0.1):
            _tick()

        # Stage 1 done — inject one BATCH_END per compute worker
        for _ in range(args.n_compute):
            compute_queue.put(BATCH_END)

        # Phase 2: poll while Stage 2/3 drain the remaining work
        while not batch_done.wait(timeout=0.1):
            _tick()
        _tick()  # final flush after batch_done fires

        bar_pre.close()
        bar_comp.close()
        bar_save.close()

        # Cleanup: remove extracted dir to reclaim storage
        shutil.rmtree(str(extract_dir), ignore_errors=True)
        tqdm.write(f"[Cleanup] {extract_dir} removed.")
        mark_zip_complete(args.zip_cache, zip_name)
        zip_bar.update(1)

    zip_bar.close()
    dl_executor.shutdown(wait=True)

    # --- Graceful shutdown of persistent workers ---
    for _ in range(args.n_compute):
        compute_queue.put(PIPELINE_END)
    for p in compute_procs:
        p.join()
    save_proc.join()

    tqdm.write(f"Pipeline complete. Results written to {args.output}")
