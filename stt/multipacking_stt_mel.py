"""Multipack raw-mel STT documents (audio -> transcription) into 10,240-token blocks.

Same task as multipacking_stt.py, different modality: instead of NeuCodec
`<|s_N|>` tokens the document carries the audio itself, as `<|mel|>` placeholders the
trainer swaps for projected whisper log-mel frames (see ../mel_audio.py).

    <|im_start|><|STT|><|mel_start|>{P x <|mel|>}<|mel_end|><|{language}|>{text}<|im_end|>

P = samples // 320, i.e. 50 positions/s -- the NeuCodec rate, so a mel document and a
token document cost the same context for the same audio and the two tasks are
comparable at equal sequence length.

The pack stores only *paths* (relative to the extracted audio tree) plus the sample
count each placeholder run was derived from. The dataset reads the files and computes mel
on the fly, so packing never decodes audio -- it only reads headers -- and the pack stays
tiny. The audio tree has to be present on the training box at `--audio_dir`.
Utterances are trimmed to a multiple of 320 samples so every one contributes a whole
number of stacked frames.

Token ids must not move: this pack appends only `<|mel_start|> <|mel|> <|mel_end|>`,
after the 65,536 `<|s_N|>` tokens and after `<|STT|>` + the language tokens that
multipacking_stt.py fixed. Those come from that pack's `stt_added_tokens.json` -- never
recomputed here, because a narrower `--subsets` selection would yield a shorter language
list and silently shift every id after it.

Usage:
    # audio zips are the full corpus -- scope with --subsets and check df -h first
    python multipacking_stt_mel.py --base-dir /root/share/stt-mel --subsets 'malaysian-*' 'emilia_zh'
    python multipacking_stt_mel.py --base-dir /root/share/stt-mel --subsets '...' --stage download
    python multipacking_stt_mel.py --base-dir /root/share/stt-mel --stage upload
"""

import os
import sys

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import fnmatch
import json
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mel_audio import (
    MEL_TOKENS,
    SAMPLE_RATE,
    build_speech_tokenizer,
    mel_positions,
    trim_to_position,
)
from multipacking_stt import (
    TOKENS_REPO,
    _extract_one,
    audio_folder,
    download_meta,
    log,
    match_subset,
    snake_chunks,
)

UPLOAD_REPO = 'Scicom-intl/Multilingual-STT-mel-multipacking-10k'
STT_PACK_REPO = 'Scicom-intl/Multilingual-STT-multipacking-10k'
OUT_NAME = 'multipacking-stt-mel'

BLOCK_SIZE = 1024 * 10
COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',            # json list of paths, relative to the audio tree
    'audio_samples': 'uint32[]',
}
HASHES = ['sha1', 'xxh64']

G = {}


# ---------------------------------------------------------------- audio probe

def probe_samples(path):
    """Length this file will have at 16kHz mono, trimmed to whole stacked frames.

    Header only -- `sf.info` reads the frame count without decoding, which is what keeps
    packing cheap now that the audio itself never enters the pack. Floored before the
    trim so the value can never exceed what the resampler actually produces.
    """
    import soundfile as sf

    info = sf.info(path)
    return trim_to_position(int(info.frames * SAMPLE_RATE / info.samplerate))


# ---------------------------------------------------------------- download

def audio_zip_patterns(folder):
    # naming is inconsistent across subsets: `<folder>_audio.zip`, `AISHELL3-audio.zip`,
    # and sharded `ASR-Tamil-cleaned_audio-0-0.zip` all occur
    return (f'{folder}_audio.zip', f'{folder}_audio-*.zip',
            f'{folder}-audio.zip', f'{folder}-audio-*.zip')


def download_audio_zips(base, meta_files, delete_zips=True):
    from huggingface_hub import HfApi

    data_dir = base / 'audio'
    marker_dir = data_dir / '.extracted'
    marker_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = base / 'zips'

    folders = {audio_folder(f) for f in meta_files}
    folders.discard(None)

    api = HfApi()
    sizes = {f.path: (f.size or 0) for f in api.list_repo_tree(
        TOKENS_REPO, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
    wanted = sorted(
        path for path in sizes
        if any(fnmatch.fnmatch(path, pat) for fo in folders for pat in audio_zip_patterns(fo))
    )
    todo = [f for f in wanted if not (marker_dir / f'{Path(f).name}.done').exists()]
    no_zip = sorted(
        fo for fo in folders
        if not any(fnmatch.fnmatch(p, pat) for p in wanted for pat in audio_zip_patterns(fo))
    )
    gb = sum(sizes.get(f, 0) for f in todo) / 1e9
    log(f'{TOKENS_REPO}: {len(wanted)} audio zips wanted, {len(todo)} to download+extract '
        f'(~{gb:.1f}GB compressed), {len(no_zip)} folders have no audio zip (rows skipped)')
    if not todo:
        return

    tasks = [(f, str(zips_dir), str(data_dir), str(marker_dir), delete_zips) for f in todo]
    with get_context('fork').Pool(min(8, len(tasks))) as pool:
        pool.map(_fetch_extract_one, tasks, chunksize=1)


def _fetch_extract_one(args):
    f, zips_dir, data_dir, marker_dir, delete = args
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            z = hf_hub_download(TOKENS_REPO, f, repo_type='dataset', local_dir=zips_dir)
            break
        except Exception as e:
            if attempt == 4:
                raise
            log(f'download of {f} failed ({e}); retrying')
            time.sleep(30)
    _extract_one((z, data_dir, marker_dir, delete))


# ---------------------------------------------------------------- tokens

def load_stt_tokens(base, explicit=None):
    """The <|STT|> + language token list the token-STT pack froze. Never recomputed."""
    if explicit:
        with open(explicit) as f:
            return json.load(f), explicit
    cached = base / 'stt_added_tokens.json'
    if not cached.exists():
        from huggingface_hub import hf_hub_download

        log(f'fetching stt_added_tokens.json from {STT_PACK_REPO}')
        got = hf_hub_download(STT_PACK_REPO, 'stt_added_tokens.json', repo_type='dataset')
        shutil.copyfile(got, cached)
    with open(cached) as f:
        return json.load(f), str(cached)


# ---------------------------------------------------------------- packing

def audio_rel_path(audio_dir, audio_filename):
    """The path to store: relative to the audio tree, so the tree can be moved."""
    if os.path.exists(os.path.join(audio_dir, audio_filename)):
        return audio_filename
    # a few zips are flat, without the leading folder the metadata carries
    _, _, rest = audio_filename.partition('/')
    if rest and os.path.exists(os.path.join(audio_dir, rest)):
        return rest
    return None


def make_block(docs, paths, samples):
    return {
        'input_ids': np.concatenate(docs).astype(np.uint32),
        'position_ids': np.concatenate([np.arange(len(d)) for d in docs]).astype(np.uint32),
        'attention_mask': np.array([len(d) for d in docs], dtype=np.uint32),
        'audio': json.dumps(paths),
        'audio_samples': np.array(samples, dtype=np.uint32),
    }


def pack_worker(args):
    import pandas as pd

    from chinidataset import ParquetWriter
    from postnormalizer import normalize

    worker_id, files = args
    tokenizer = G['tokenizer']
    audio_dir = G['audio_dir']
    languages = G['languages']
    mel_id = G['mel_id']
    min_samples = G['min_seconds'] * SAMPLE_RATE
    max_samples = G['max_seconds'] * SAMPLE_RATE

    prefix = tokenizer('<|im_start|><|STT|><|mel_start|>', add_special_tokens=False)['input_ids']

    out_dir = os.path.join(G['out_root'], f'{worker_id:05d}')
    shutil.rmtree(out_dir, ignore_errors=True)

    stats = {'blocks': 0, 'docs': 0, 'seconds': 0, 'no_lang': 0, 'empty_text': 0,
             'missing': 0, 'duration': 0, 'ratio': 0, 'probe_error': 0}
    docs, paths, samples = [], [], []
    count = 0
    with ParquetWriter(out=out_dir, columns=COLUMNS, compression=None, hashes=HASHES,
                       size_limit=256 * 1024 * 1024) as writer:
        for f in files:
            df = pd.read_parquet(f)
            if 'audio_filename' not in df.columns:
                stats['missing'] += len(df)
                continue
            has_norm = 'post-normalized' in df.columns
            audio_col = df['audio_filename'].tolist()
            lang_col = df['language'].tolist() if 'language' in df.columns else [None] * len(df)
            text_col = df['post-normalized'].tolist() if has_norm else df['text'].tolist()
            for audio_filename, language, text in zip(audio_col, lang_col, text_col):
                if not language or language == 'und' or language not in languages:
                    stats['no_lang'] += 1
                    continue

                if not has_norm:
                    text = normalize(text)
                if not text or not str(text).strip():
                    stats['empty_text'] += 1
                    continue
                text = str(text).strip()

                rel = audio_rel_path(audio_dir, audio_filename)
                if rel is None:
                    stats['missing'] += 1
                    continue

                try:
                    n_samples = probe_samples(os.path.join(audio_dir, rel))
                except Exception:
                    stats['probe_error'] += 1
                    continue

                if not min_samples <= n_samples <= max_samples:
                    stats['duration'] += 1
                    continue

                positions = mel_positions(n_samples)
                # same alignment sanity check the token pack uses: at 50 positions/s,
                # more words than positions means the transcript is not this audio
                if len(text.split()) > positions:
                    stats['ratio'] += 1
                    continue

                suffix = tokenizer(f'<|mel_end|><|{language}|>{text}<|im_end|>',
                                   add_special_tokens=False)['input_ids']
                # spliced rather than tokenized as text: a 30s clip is 1,500 placeholders
                # and re-tokenizing that string per row dominates the pack
                ids = prefix + [mel_id] * positions + suffix
                stats['docs'] += 1
                stats['seconds'] += n_samples / SAMPLE_RATE

                if count + len(ids) > BLOCK_SIZE:
                    if docs:
                        writer.write(make_block(docs, paths, samples))
                        stats['blocks'] += 1
                    docs, paths, samples = [ids], [rel], [n_samples]
                    count = len(ids)
                else:
                    docs.append(ids)
                    paths.append(rel)
                    samples.append(n_samples)
                    count += len(ids)

            if worker_id == 0:
                log(f'worker 0: finished {Path(f).parent.name}/{Path(f).name}, '
                    f'{stats["blocks"]} blocks, {stats["seconds"] / 3600:.1f}h')

        if docs:
            writer.write(make_block(docs, paths, samples))
            stats['blocks'] += 1

    return stats


def pack(base, wave_files, wave_name, workers, args):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    stt_tokens, src = load_stt_tokens(base, args.stt_tokens)
    languages = [t[2:-2] for t in stt_tokens if t != '<|STT|>']
    log(f'{len(languages)} language tokens from {src}')

    log('building tokenizer (+65,537 speech tokens, <|STT|>, '
        f'{len(languages)} language tokens, {len(MEL_TOKENS)} mel tokens)')
    tokenizer = build_speech_tokenizer(
        'Qwen/Qwen3-1.7B-Base', stt_tokens_file=src, add_mel_tokens=True)

    out_root = base / 'out' / OUT_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    wave_dir = out_root / wave_name
    shutil.rmtree(wave_dir, ignore_errors=True)
    wave_dir.mkdir(parents=True)

    G.update(
        tokenizer=tokenizer,
        audio_dir=str(base / 'audio'),
        languages=set(languages),
        # a token_to_id lookup, not an encode — the rust-tokenizer/fork deadlock the
        # preparation/ CLAUDE.md warns about needs an encode in the parent
        mel_id=tokenizer.convert_tokens_to_ids('<|mel|>'),
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        out_root=str(wave_dir),
    )

    tasks = list(enumerate(snake_chunks(wave_files, workers)))
    t0 = time.time()
    with get_context('fork').Pool(len(tasks)) as pool:
        results = pool.map(pack_worker, tasks)
    G.clear()

    totals = {k: sum(r[k] for r in results) for k in results[0]}
    merge_index(wave_dir)

    packed = StreamingDataset(local=str(wave_dir))
    n = len(packed)
    log(f'{wave_name}: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens, '
        f'{totals["seconds"] / 3600:.1f}h audio) in {time.time() - t0:.0f}s | '
        + ' '.join(f'{k}={v}' for k, v in totals.items() if k != 'seconds'))
    if totals['missing'] > totals['docs']:
        log('WARNING: more rows missing audio than packed — this is a path-convention '
            'mismatch between the metadata and the extracted zips, not missing data')

    with open(out_root / 'mel_added_tokens.json', 'w') as f:
        json.dump(stt_tokens + MEL_TOKENS, f, indent=2)
    with open(wave_dir / f'summary-{wave_name}.json', 'w') as f:
        json.dump({'wave': wave_name, 'blocks': n, 'languages': len(languages), **totals}, f, indent=2)


def merge_waves(base):
    from chinidataset import StreamingDataset
    from chinidataset.util import merge_index

    out_root = base / 'out' / OUT_NAME
    merge_index(out_root)
    n = len(StreamingDataset(local=str(out_root)))
    log(f'merged: {n} blocks (~{n * BLOCK_SIZE / 1e9:.2f}B tokens)')
    with open(out_root / 'summary.json', 'w') as f:
        json.dump({'blocks': n}, f, indent=2)


def upload(base):
    from huggingface_hub import HfApi

    out_root = base / 'out' / OUT_NAME
    api = HfApi()
    for attempt in range(10):
        try:
            api.create_repo(UPLOAD_REPO, repo_type='dataset', private=True, exist_ok=True)
            break
        except Exception as e:
            if attempt == 9:
                raise
            log(f'create_repo rate-limited ({e}); waiting 120s')
            time.sleep(120)
    log(f'uploading {out_root} -> {UPLOAD_REPO}')
    for attempt in range(5):
        try:
            api.upload_large_folder(repo_id=UPLOAD_REPO, repo_type='dataset',
                                    folder_path=str(out_root), num_workers=6)
            return
        except Exception as e:
            if attempt == 4:
                raise
            log(f'upload_large_folder failed ({e}); resuming in 300s')
            time.sleep(300)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/root/share/stt-mel')
    parser.add_argument('--subsets', nargs='*', default=None,
                        help='fnmatch patterns of subset names (default: all — the full audio corpus)')
    parser.add_argument('--stt-tokens', default=None,
                        help='stt_added_tokens.json from the token-STT pack '
                             f'(default: fetch from {STT_PACK_REPO})')
    parser.add_argument('--min-seconds', type=float, default=0.3)
    parser.add_argument('--max-seconds', type=float, default=30.0)
    parser.add_argument('--num-waves', type=int, default=1)
    parser.add_argument('--wave', type=int, default=0)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument('--stage', choices=['download', 'pack', 'merge', 'upload', 'all'], default='all')
    parser.add_argument('--keep-zips', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.stage == 'merge':
        merge_waves(base)
        return
    if args.stage == 'upload':
        upload(base)
        return

    if args.stage in ('download', 'all'):
        download_meta(base, args.subsets)

    all_files = sorted(
        (f for f in (base / 'meta').glob('*/*.parquet') if match_subset(f.parent.name, args.subsets)),
        key=lambda f: (f.parent.name, f.name),
    )
    if not all_files:
        raise SystemExit('no metadata parquets found — run --stage download first')

    subsets_sorted = sorted({f.parent.name for f in all_files})
    step = -(-len(subsets_sorted) // args.num_waves)
    wave_subsets = set(subsets_sorted[args.wave * step:(args.wave + 1) * step])
    wave_files = [f for f in all_files if f.parent.name in wave_subsets]
    wave_name = f'wave-{args.wave}' if args.num_waves > 1 else 'data'
    log(f'{len(all_files)} metadata parquets total; {wave_name}: '
        f'{len(wave_subsets)} subsets / {len(wave_files)} files')

    if args.stage in ('download', 'all'):
        download_audio_zips(base, wave_files, delete_zips=not args.keep_zips)
    if args.stage in ('pack', 'all'):
        pack(base, wave_files, wave_name, args.workers, args)


if __name__ == '__main__':
    main()
