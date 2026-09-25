"""Build the one added-token list every pack in a training run must share.

Language tags and the mel tokens are appended after the 65,537 speech tokens, in order.
Two packs built against different lists therefore disagree on ids for everything after
`<|s_65535|>` — including `<|mel|>`. One run, one list.

Sources of tags:

    FLEURS-R              locale per file name        e.g. en_us, cmn_hans_cn   (102)
    Common Voice 22       language segment in path    e.g. en, de              (130)
    Multilingual-TTS      GlotLID v3 `language`       e.g. zsm_Latn, yor_Latn  (~1.5k subsets)

The voice-conversion packs need no tag: they are speech tokens and text only, so their
ids are unaffected by anything appended here.

Usage:
    python build_vocab.py --out /root/share/vocab/added_tokens.json \\
        --fleurs-data /root/share/fleurs/data \\
        --cv22-meta /root/share/cv22/meta/common-voice-22 \\
        --stt-meta /root/share/stt/meta
    python build_vocab.py --out ... --stt-meta ... --download-stt-meta   # fetch it first
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STT_META_REPO = 'malaysia-ai/Multilingual-TTS-language'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fleurs_locales(data_dir):
    if not data_dir or not Path(data_dir).is_dir():
        return []
    return sorted({p.name.rsplit('-', 1)[0] for p in Path(data_dir).glob('*.parquet')})


def cv22_languages(meta_dir):
    """The language segment of `audio_trim/{lang}/{split}/{id}.mp3`."""
    if not meta_dir or not Path(meta_dir).is_dir():
        return []
    import pyarrow.parquet as pq

    seen = set()
    for f in sorted(Path(meta_dir).glob('*.parquet')):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=100_000, columns=['audio_filename']):
            for name in batch.column(0):
                parts = str(name).split('/')
                if len(parts) > 2:
                    seen.add(parts[1])
    return sorted(seen)


def download_stt_meta(meta_dir, attempts=8):
    """Fetch the metadata, surviving the CDN dropping a connection mid-file.

    snapshot_download resumes from what is already on disk, so a retry costs only the
    files that were still in flight.
    """
    from huggingface_hub import snapshot_download

    for attempt in range(attempts):
        try:
            log(f'fetching {STT_META_REPO} metadata -> {meta_dir} (attempt {attempt + 1})')
            snapshot_download(STT_META_REPO, repo_type='dataset', local_dir=str(meta_dir),
                              allow_patterns=['*/*.parquet'], max_workers=8)
            return
        except Exception as e:
            if attempt == attempts - 1:
                raise
            log(f'  download failed ({type(e).__name__}: {e}); retrying in 30s')
            time.sleep(30)


def _labels_one(path):
    import pandas as pd

    try:
        return set(pd.read_parquet(path, columns=['language'])['language'].dropna().unique())
    except Exception:
        return set()


def stt_languages(meta_dir, workers):
    """Distinct GlotLID labels across every subset."""
    if not meta_dir or not Path(meta_dir).is_dir():
        return []
    files = sorted(Path(meta_dir).glob('*/*.parquet'))
    if not files:
        return []
    log(f'scanning {len(files)} metadata parquets for GlotLID labels')
    with get_context('fork').Pool(min(workers, len(files))) as pool:
        sets = pool.map(_labels_one, files, chunksize=4)
    return sorted(set().union(*sets) - {'und', '', None})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fleurs-data', default='/root/share/fleurs/data')
    parser.add_argument('--cv22-meta', default='/root/share/cv22/meta/common-voice-22')
    parser.add_argument('--stt-meta', default='/root/share/stt/meta')
    parser.add_argument('--download-stt-meta', action='store_true')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = parser.parse_args()

    from mel_audio import MEL_TOKENS

    if args.download_stt_meta:
        Path(args.stt_meta).mkdir(parents=True, exist_ok=True)
        download_stt_meta(Path(args.stt_meta))

    fleurs = fleurs_locales(args.fleurs_data)
    cv22 = cv22_languages(args.cv22_meta)
    stt = stt_languages(args.stt_meta, args.workers)
    tags = sorted(set(fleurs) | set(cv22) | set(stt))
    tokens = ['<|STT|>'] + [f'<|{t}|>' for t in tags] + list(MEL_TOKENS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(tokens, f, indent=2)
    summary = {'fleurs_locales': len(fleurs), 'cv22_languages': len(cv22),
               'stt_glotlid_labels': len(stt), 'distinct_tags': len(tags),
               'added_tokens': len(tokens)}
    with open(out.with_suffix('.summary.json'), 'w') as f:
        json.dump({**summary, 'tags': tags}, f, indent=2)
    log(f'{len(tokens)} added tokens -> {out}')
    log('  ' + ' | '.join(f'{k}={v}' for k, v in summary.items()))


if __name__ == '__main__':
    main()
