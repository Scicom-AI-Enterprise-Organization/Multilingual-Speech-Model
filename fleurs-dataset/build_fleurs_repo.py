"""Complete and enrich malaysia-ai/fleurs-r-neucodec-all-languages.

Three gaps this closes:

  audio/    only 15 of 204 zips were ever pushed — the upload stopped at `bg_bg-dev`,
            so 94 of 102 locales ship tokens and transcripts but no source audio
  speaker   FLEURS (and FLEURS-R) ship `gender` and nothing else about the voice, so
            the TTS prompt has no speaker to condition on
  viewer    the dataset card has no `configs:`, so the HF viewer reports
            "No (supported) data files found" and the repo previews nothing

Stages, all resumable and safe to re-run:

  audio    for every (locale, split): pull `google/fleurs-r data/{locale}/audio/{split}.tar.gz`,
           rebuild `audio/{locale}-{split}.zip` (24kHz PCM16 mono, arcnames
           `audio/{locale}/{split}/{id}.wav` — the convention the 15 existing zips use and
           the one `data/*.parquet` `path` points at), upload it when the repo lacks it,
           and embed every utterance with TitaNet-L on the way past. Wavs are deleted as
           soon as the zip and the vectors exist, so peak disk stays ~3GB per worker.
  cluster  agglomerative clustering (cosine, average linkage) of the vectors within each
           (locale, gender) — gender is a hard constraint, so a cluster never mixes voices
           FLEURS already labels apart. Tiny clusters fold into their nearest centroid;
           ids are `{locale}_spkNN`, numbered by descending cluster size.
  parquet  write the `speaker` column into `data/{locale}-{split}.parquet` (keeping the
           HF features metadata in sync) and re-upload.
  card     README.md with `configs:` so the viewer loads data/*.parquet, plus column docs.

Usage:
    python build_fleurs_repo.py --stage all --workers 16
    python build_fleurs_repo.py --stage audio --locales en_us ms_my
    python build_fleurs_repo.py --stage cluster --threshold 0.4
    python build_fleurs_repo.py --stage card --dry-run
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import argparse
import json
import shutil
import tarfile
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = 'malaysia-ai/fleurs-r-neucodec-all-languages'
UPSTREAM = 'google/fleurs-r'
SPLITS = ('train', 'dev')
EMB_DIM = 192


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def api():
    from huggingface_hub import HfApi
    return HfApi()


def fetch_metadata(base):
    """The repo's own data/*.parquet — the row set every stage is keyed on."""
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type='dataset', local_dir=base, allow_patterns=['data/*.parquet'],
                      max_workers=8)
    return sorted((base / 'data').glob('*.parquet'))


def locale_split(parquet_path):
    locale, _, split = parquet_path.stem.rpartition('-')
    return locale, split


# ---------------------------------------------------------------- audio + vectors

G = {}


def _load_model():
    import torch

    from titanet_vectors import load

    torch.set_grad_enabled(False)
    torch.set_num_threads(G.get('threads', 2))
    model = load().eval()
    return model


def _embed_wavs(model, files):
    """TitaNet-L vector per utterance; FLEURS-R is 24kHz, TitaNet wants 16kHz."""
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    vectors = np.zeros((len(files), EMB_DIM), dtype=np.float32)
    for i, f in enumerate(files):
        audio, sr = sf.read(f, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            # 24000 -> 16000 is exactly 2/3, so polyphase resampling is exact and fast
            audio = resample_poly(audio, 2, 3) if sr == 24000 else resample_poly(audio, 16000, sr)
        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        lens = torch.tensor([tensor.shape[-1]])
        vectors[i] = model(tensor, lens)[1][0].numpy()
    return vectors


def audio_worker(task):
    locale, split, need_zip = task
    base = Path(G['base'])
    work = Path(G['work']) / f'{locale}-{split}'
    emb_file = base / 'emb' / f'{locale}-{split}.npz'
    zip_marker = base / 'uploaded' / f'{locale}-{split}.done'

    need_emb = not emb_file.exists()
    need_zip = need_zip and not zip_marker.exists()
    if not need_emb and not need_zip:
        return {'locale': locale, 'split': split, 'skipped': True}

    from huggingface_hub import hf_hub_download

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        tar_path = None
        for attempt in range(5):
            try:
                tar_path = hf_hub_download(UPSTREAM, f'data/{locale}/audio/{split}.tar.gz',
                                           repo_type='dataset', local_dir=str(work / 'src'))
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log(f'{locale}-{split}: download failed ({e}); retrying')
                time.sleep(20)

        wav_dir = work / 'wav'
        wav_dir.mkdir()
        with tarfile.open(tar_path) as tf:
            # members are `{split}/{id}.wav`; flatten into one directory
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith('.wav'):
                    continue
                member.name = os.path.basename(member.name)
                tf.extract(member, wav_dir, filter='data')
        os.remove(tar_path)
        files = sorted(wav_dir.glob('*.wav'))

        result = {'locale': locale, 'split': split, 'wavs': len(files), 'skipped': False}

        if need_zip:
            zip_path = work / f'{locale}-{split}.zip'
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.write(f, arcname=f'audio/{locale}/{split}/{f.name}')
            result['zip_mb'] = round(zip_path.stat().st_size / 1e6, 1)
            for attempt in range(5):
                try:
                    api().upload_file(path_or_fileobj=str(zip_path), repo_id=REPO, repo_type='dataset',
                                      path_in_repo=f'audio/{locale}-{split}.zip',
                                      commit_message=f'audio: {locale}-{split}')
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    log(f'{locale}-{split}: upload failed ({e}); retrying in 60s')
                    time.sleep(60)
            zip_marker.parent.mkdir(parents=True, exist_ok=True)
            zip_marker.touch()
            zip_path.unlink()

        if need_emb:
            model = G.get('model')
            if model is None:
                model = G['model'] = _load_model()
            t0 = time.time()
            vectors = _embed_wavs(model, files)
            emb_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb_file, ids=np.array([f.name for f in files]), vectors=vectors)
            result['embed_s'] = round(time.time() - t0)

        log(f'{locale}-{split}: ' + ' '.join(f'{k}={v}' for k, v in result.items() if k not in ('locale', 'split')))
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def stage_audio(base, work, parquets, workers, threads, dry_run):
    existing = set(api().list_repo_files(REPO, repo_type='dataset'))
    tasks = []
    for p in parquets:
        locale, split = locale_split(p)
        need_zip = f'audio/{locale}-{split}.zip' not in existing
        tasks.append((locale, split, need_zip))
    missing = sum(1 for t in tasks if t[2])
    log(f'{len(tasks)} (locale, split) pairs; {missing} audio zips missing from the repo')
    if dry_run:
        for t in tasks[:10]:
            log(f'would process {t[0]}-{t[1]} (upload zip: {t[2]})')
        return

    G.update(base=str(base), work=str(work), threads=threads)
    with get_context('fork').Pool(workers) as pool:
        results = pool.map(audio_worker, tasks, chunksize=1)
    done = [r for r in results if not r.get('skipped')]
    log(f'audio stage done: {len(done)} pairs processed, {len(results) - len(done)} already complete')


# ---------------------------------------------------------------- speaker clustering

def cluster_locale(vectors, genders, threshold, min_size):
    """Cluster within gender; returns labels as ints ordered by descending size."""
    from sklearn.cluster import AgglomerativeClustering

    norm = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
    labels = np.full(len(vectors), -1, dtype=np.int64)
    next_label = 0
    for gender in sorted(set(genders)):
        idx = np.flatnonzero(genders == gender)
        if len(idx) == 1:
            labels[idx] = next_label
            next_label += 1
            continue
        sub = norm[idx]
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                        metric='cosine', linkage='average')
        sub_labels = model.fit_predict(sub)

        # fold clusters too small to be a real speaker into the nearest big centroid
        sizes = {l: int((sub_labels == l).sum()) for l in set(sub_labels)}
        big = [l for l, n in sizes.items() if n >= min_size]
        if big:
            centroids = np.stack([sub[sub_labels == l].mean(axis=0) for l in big])
            centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
            for l, n in sizes.items():
                if n >= min_size:
                    continue
                members = np.flatnonzero(sub_labels == l)
                nearest = np.argmax(sub[members] @ centroids.T, axis=1)
                for m, nb in zip(members, nearest):
                    sub_labels[m] = big[nb]

        for l in sorted(set(sub_labels)):
            labels[idx[sub_labels == l]] = next_label
            next_label += 1

    # renumber by descending cluster size so spk00 is the dominant voice
    order = sorted(set(labels.tolist()), key=lambda l: -int((labels == l).sum()))
    remap = {l: i for i, l in enumerate(order)}
    return np.array([remap[l] for l in labels], dtype=np.int64)


def stage_cluster(base, parquets, threshold, min_size):
    import pandas as pd
    from sklearn.metrics import silhouette_score

    by_locale = {}
    for p in parquets:
        locale, split = locale_split(p)
        by_locale.setdefault(locale, []).append((split, p))

    out_dir = base / 'speakers'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for locale, items in sorted(by_locale.items()):
        ids, vectors, genders, splits = [], [], [], []
        for split, p in sorted(items):
            emb_file = base / 'emb' / f'{locale}-{split}.npz'
            if not emb_file.exists():
                log(f'{locale}-{split}: no vectors yet, skipping locale')
                ids = None
                break
            df = pd.read_parquet(p, columns=['filename', 'gender'])
            gender_of = dict(zip(df['filename'], df['gender']))
            data = np.load(emb_file)
            for name, vector in zip(data['ids'], data['vectors']):
                name = str(name)
                if name not in gender_of:      # wav with no metadata row
                    continue
                ids.append(name)
                vectors.append(vector)
                genders.append(str(gender_of[name] or 'UNKNOWN'))
                splits.append(split)
        if not ids:
            continue

        vectors = np.stack(vectors)
        genders = np.array(genders)
        labels = cluster_locale(vectors, genders, threshold, min_size)
        n_spk = len(set(labels.tolist()))
        sil = None
        if 1 < n_spk < len(labels):
            norm = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
            sil = round(float(silhouette_score(norm, labels, metric='cosine')), 3)
        speakers = {name: f'{locale}_spk{int(l):02d}' for name, l in zip(ids, labels)}
        sizes = sorted((int((labels == l).sum()) for l in set(labels.tolist())), reverse=True)
        with open(out_dir / f'{locale}.json', 'w') as f:
            json.dump(speakers, f)
        summary[locale] = {'utterances': len(ids), 'speakers': n_spk, 'sizes': sizes,
                           'silhouette': sil, 'genders': sorted(set(genders.tolist()))}
        log(f'{locale}: {len(ids)} utts -> {n_spk} speakers {sizes[:6]} silhouette={sil}')

    with open(base / 'speaker_summary.json', 'w') as f:
        json.dump({'threshold': threshold, 'min_size': min_size, 'locales': summary}, f, indent=2)
    counts = [v['speakers'] for v in summary.values()]
    if counts:
        log(f'{len(counts)} locales | speakers per locale: min {min(counts)} '
            f'median {int(np.median(counts))} max {max(counts)}')


# ---------------------------------------------------------------- parquet + card

FEATURE_ENTRY = {'dtype': 'string', '_type': 'Value'}


def stage_parquet(base, parquets, dry_run):
    import pyarrow as pa
    import pyarrow.parquet as pq

    changed = []
    for p in parquets:
        locale, split = locale_split(p)
        speaker_file = base / 'speakers' / f'{locale}.json'
        if not speaker_file.exists():
            log(f'{locale}: no speaker map, skipping {p.name}')
            continue
        with open(speaker_file) as f:
            speakers = json.load(f)

        table = pq.read_table(p)
        if 'speaker' in table.column_names:
            table = table.drop_columns(['speaker'])
        names = table.column('filename').to_pylist()
        column = [speakers.get(n) for n in names]
        missing = sum(1 for c in column if c is None)
        if missing:
            log(f'{p.name}: {missing}/{len(column)} rows without a speaker (left null)')
        # keep speaker next to gender rather than appended at the end
        position = table.column_names.index('gender') + 1
        table = table.add_column(position, pa.field('speaker', pa.string()), pa.array(column, pa.string()))

        metadata = dict(table.schema.metadata or {})
        hf_key = b'huggingface'
        if hf_key in metadata:
            info = json.loads(metadata[hf_key])
            features = info.get('info', {}).get('features')
            if isinstance(features, dict):
                features['speaker'] = FEATURE_ENTRY
                metadata[hf_key] = json.dumps(info).encode()
                table = table.replace_schema_metadata(metadata)

        if not dry_run:
            pq.write_table(table, p, compression='snappy')
        changed.append(p)
        if len(changed) <= 3:
            log(f'{p.name}: speaker column written ({len(set(c for c in column if c))} distinct)')

    log(f'{len(changed)} parquet files updated')
    if dry_run or not changed:
        return
    for attempt in range(5):
        try:
            api().upload_folder(folder_path=str(base / 'data'), path_in_repo='data', repo_id=REPO,
                                repo_type='dataset', allow_patterns=['*.parquet'],
                                commit_message='add speaker column (TitaNet-L clusters)')
            break
        except Exception as e:
            if attempt == 4:
                raise
            log(f'parquet upload failed ({e}); retrying in 60s')
            time.sleep(60)
    log('parquet upload done')


CARD = '''---
license: cc-by-4.0
language:
- multilingual
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: FLEURS-R NeuCodec All Languages
size_categories:
- 100K<n<1M
configs:
{configs}---

# FLEURS-R NeuCodec All Languages

FLEURS-R metadata, source audio and precomputed NeuCodec speech tokens for {locales}
locales, plus a `speaker` label FLEURS itself does not ship.

## Layout

- `data/{{locale}}-{{split}}.parquet` — metadata, one row per utterance (this is what the
  viewer shows).
- `audio/{{locale}}-{{split}}.zip` — source FLEURS-R audio, 24kHz mono PCM16 WAV, members
  named `audio/{{locale}}/{{split}}/{{id}}.wav` (the `path` column).
- `neucodec/{{locale}}-{{split}}-rank{{N}}.zip` — NeuCodec tokens, members named
  `neucodec/{{locale}}/{{split}}/{{id}}.pt` (the `neucodec_path` column), each a
  `{{'source': str, 'codes': IntTensor[1, 1, T]}}` torch file at 50 tokens/s.

## Columns

| column | what |
|---|---|
| `id` | FLEURS utterance id |
| `path` | audio member path inside `audio/{{locale}}-{{split}}.zip` |
| `neucodec_path` | token member path inside the matching `neucodec/` zip |
| `filename` | bare `{{id}}.wav` |
| `sentence` | raw transcription |
| `normalized_text` | lowercased, punctuation-stripped transcription |
| `characters` | space-separated character transcription |
| `locale`, `split` | e.g. `en_us`, `train` |
| `gender` | FLEURS speaker gender label |
| `speaker` | **derived** voice identity, `{{locale}}_spkNN` (see below) |
| `num_samples` | sample count of the original 16kHz FLEURS audio |

## Speaker labels

FLEURS and FLEURS-R ship `gender` but no speaker identity. `speaker` here is derived:
every utterance is embedded with TitaNet-L ([titanet-vectors-fp16](https://github.com/Scicom-AI-Enterprise-Organization/titanet-vectors-fp16)),
and the vectors are clustered per locale with agglomerative clustering
(cosine distance, average linkage, threshold {threshold}) **within each gender**, so a
cluster never merges voices FLEURS already labels apart. Clusters smaller than
{min_size} utterances are folded into their nearest centroid, and ids are numbered by
descending cluster size, so `{{locale}}_spk00` is the most-recorded voice of that locale.

They are clusters, not ground truth: treat them as consistent voice tags for conditioning,
not as verified speaker identities.

{speaker_stats}

## Loading

```python
from datasets import load_dataset

ds = load_dataset("malaysia-ai/fleurs-r-neucodec-all-languages")             # all locales
ds = load_dataset("malaysia-ai/fleurs-r-neucodec-all-languages", "en_us")    # one locale
```

Audio and tokens are the zips above — `path` and `neucodec_path` name the member inside
`audio/{{locale}}-{{split}}.zip` and `neucodec/{{locale}}-{{split}}-rank*.zip`.

## Source

[google/fleurs-r](https://huggingface.co/datasets/google/fleurs-r) (FLEURS-R, CC BY 4.0),
the restored variant of [google/fleurs](https://huggingface.co/datasets/google/fleurs).
'''


def build_configs(locales):
    """`default` over every locale, plus one config per locale for the viewer dropdown."""
    lines = ['- config_name: default', '  data_files:']
    for split in SPLITS:
        lines += [f'  - split: {split}', f'    path: data/*-{split}.parquet']
    for locale in locales:
        lines += [f'- config_name: {locale}', '  data_files:']
        for split in SPLITS:
            lines += [f'  - split: {split}', f'    path: data/{locale}-{split}.parquet']
    return '\n'.join(lines) + '\n'


def stage_card(base, dry_run):
    summary_file = base / 'speaker_summary.json'
    stats_block = ''
    threshold, min_size, n_locales = 0.4, 5, 102
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        threshold = summary['threshold']
        min_size = summary['min_size']
        locales = summary['locales']
        n_locales = len(locales)
        counts = sorted(v['speakers'] for v in locales.values())
        utts = sum(v['utterances'] for v in locales.values())
        sils = [v['silhouette'] for v in locales.values() if v['silhouette'] is not None]
        stats_block = (
            f'Across {n_locales} locales / {utts:,} utterances: '
            f'{min(counts)}–{max(counts)} speakers per locale (median {counts[len(counts) // 2]}), '
            f'mean silhouette {np.mean(sils):.2f}.'
        )
    locale_names = sorted({locale_split(p)[0] for p in (base / 'data').glob('*.parquet')})
    card = CARD.format(configs=build_configs(locale_names), locales=len(locale_names) or n_locales,
                       threshold=threshold, min_size=min_size, speaker_stats=stats_block)
    out = base / 'README.md'
    out.write_text(card)
    log(f'card written to {out} ({len(card)} chars)')
    if dry_run:
        print(card)
        return
    api().upload_file(path_or_fileobj=str(out), path_in_repo='README.md', repo_id=REPO,
                      repo_type='dataset', commit_message='card: viewer configs + speaker docs')
    log('card uploaded')


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='/share/fleurs-work')
    parser.add_argument('--work-dir', default='/share/fleurs-work/scratch',
                        help='per-task scratch for tarballs/wavs/zips (deleted as it goes)')
    parser.add_argument('--stage', choices=['audio', 'cluster', 'parquet', 'card', 'all'], default='all')
    parser.add_argument('--locales', nargs='*', help='subset of locales (default: all in the repo)')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2, help='torch threads per worker')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='cosine distance cut for clustering — 0.4 maximised silhouette '
                             '(0.55-0.61) across a 8-locale sweep of 0.3/0.4/0.5/0.6')
    parser.add_argument('--min-size', type=int, default=5, help='clusters smaller than this are folded in')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    parquets = fetch_metadata(base)
    parquets = [p for p in parquets if locale_split(p)[1] in args.splits
                and (not args.locales or locale_split(p)[0] in args.locales)]
    log(f'{len(parquets)} metadata parquets in scope')

    if args.stage in ('audio', 'all'):
        stage_audio(base, Path(args.work_dir), parquets, args.workers, args.threads, args.dry_run)
    if args.stage in ('cluster', 'all'):
        stage_cluster(base, parquets, args.threshold, args.min_size)
    if args.stage in ('parquet', 'all'):
        stage_parquet(base, parquets, args.dry_run)
    if args.stage in ('card', 'all'):
        stage_card(base, args.dry_run)


if __name__ == '__main__':
    main()
