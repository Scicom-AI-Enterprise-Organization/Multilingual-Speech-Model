"""Pack the non-verbal-tag datasets into TTS blocks.

12,740 rows across three `Scicom-intl/*-Nonverbal-Tags` repos, whose `tagged_text`
carries inline `<|sfx:family|>` markers placed by the nonverbal-tagging pipeline:

    <|im_start|>{language}: {tagged_text}<|speech_start|>{<|s_N|> …}<|im_end|>

Those repos hold text and audit crops only, so the speech tokens come from the source
Emilia repos the rows were mined from. `file` is the mining pod's absolute path, and the
folder ending in `_trim` names both the zip and the member prefix:

    /root/data/audio/malaysian-podcast_processed_trim/A/B_25.mp3
      -> Scicom-intl/Malaysian-Emilia :: malaysian-podcast_processed_trim_neucodec.zip
      -> member malaysian-podcast_processed_trim_neucodec/A/B_25.json

Six zips, 4.73 GB total, so they are downloaded whole and read in place — nothing is
extracted, which matters because one of them holds 255K members.

## The vocabulary must GROW, never shift

`<|sfx:*|>` tokens are not in the 2,337-token `added_tokens.json` the corpus, FLEURS and
CV22 packs were built against, and the mel tokens are at the END of that list. Inserting
anything before them would renumber `<|mel|>` and silently invalidate every mel block
already written. So this writes `added_tokens_v2.json` = the existing list **unchanged**,
with the sfx tags appended after it, and verifies the prefix before writing. Packs built
against v1 stay valid; the trainer takes v2.

## Weighting

~4M tokens against ~88B in the full mixture is 0.005%. At weight 1.0 the model will
effectively never see a tag. Weight this pack up hard in `--train_file` (50-100x) or
leave it out — packing it is not the same as training on it.

Usage:
    python multipacking_nonverbal.py --stage tokens
    python multipacking_nonverbal.py --stage pack
    python multipacking_nonverbal.py --keep-families laughter,cough,sigh
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_retry import file_with_retry
from multipacking_fleurs import (
    COLUMNS,
    HASHES,
    BlockAccumulator,
    clean_text,
    log,
    tokenizer_from,
)

# nonverbal repo -> the Emilia repo its `file` paths point into
SOURCES = {
    'Scicom-intl/Malaysian-Emilia-Nonverbal-Tags': 'Scicom-intl/Malaysian-Emilia',
    'Scicom-intl/Malaysian-Tamil-Emilia-Nonverbal-Tags': 'Scicom-intl/Malaysian-Tamil-Emilia',
    'Scicom-intl/Malaysian-Chinese-Emilia-Nonverbal-Tags': 'Scicom-intl/Malaysian-Chinese-Emilia',
}

SFX_RE = re.compile(r'<\|sfx:([^|]+)\|>')

# `burping` is ~14% of all placed tags and the pipeline's own notes call it "mouth/plosive
# false-accepts sneaking past CLAP" (nonverbal-tagging/CLAUDE.md). Training on it teaches
# the model to emit a burp tag on plosives, so it is dropped unless asked for.
DEFAULT_DROP = ('burping',)


def rows(nv_repo):
    """The repo's data/*.parquet as one frame."""
    import pandas as pd
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(nv_repo, repo_type='dataset')
             if f.startswith('data/') and f.endswith('.parquet')]
    frames = [pd.read_parquet(file_with_retry(nv_repo, f, repo_type='dataset'))
              for f in sorted(files)]
    return pd.concat(frames, ignore_index=True)


def token_member(file_path):
    """(zip name, member name) for a mining-pod audio path, or None."""
    parts = str(file_path).split('/')
    folder = next((p for p in parts if p.endswith('_trim')), None)
    if folder is None:
        return None
    rest = parts[parts.index(folder) + 1:]
    if not rest:
        return None
    member = '/'.join([f'{folder}_neucodec'] + rest)
    if not member.endswith('.mp3'):
        return None
    return f'{folder}_neucodec.zip', member[:-4] + '.json'


def stage_tokens(base):
    """Download the source zips the rows point into (whole, not extracted)."""
    wanted = defaultdict(set)
    for nv_repo, src_repo in SOURCES.items():
        df = rows(nv_repo)
        for f in df['file']:
            m = token_member(f)
            if m:
                wanted[(src_repo, m[0])].add(m[1])
    total = 0
    for (src_repo, zip_name), members in sorted(wanted.items()):
        path = file_with_retry(src_repo, zip_name, repo_type='dataset')
        size = os.path.getsize(path) / 1e9
        log(f'{src_repo.split("/")[-1]}/{zip_name}: {len(members):,} members wanted, {size:.2f} GB')
        total += len(members)
    log(f'{total:,} token members across {len(wanted)} zips')
    (base / 'nonverbal').mkdir(parents=True, exist_ok=True)
    return wanted


def load_codes(wanted):
    """{(src_repo, member): codes} read straight out of the cached zips."""
    out = {}
    missing = 0
    for (src_repo, zip_name), members in sorted(wanted.items()):
        path = file_with_retry(src_repo, zip_name, repo_type='dataset')
        with zipfile.ZipFile(path) as z:
            have = z.NameToInfo
            for name in members:
                if name not in have:
                    missing += 1
                    continue
                try:
                    out[(src_repo, name)] = json.loads(z.read(name))
                except Exception:
                    missing += 1
        log(f'{zip_name}: {len(out):,} loaded so far, {missing} missing')
    return out, missing


def write_vocab(v1_path, v2_path, sfx_tokens):
    """v2 = v1 unchanged + the sfx tags appended, so no existing id moves."""
    with open(v1_path) as f:
        v1 = json.load(f)
    new = [t for t in sfx_tokens if t not in set(v1)]
    v2 = v1 + new
    assert v2[:len(v1)] == v1, 'v2 must keep v1 as an exact prefix'
    with open(v2_path, 'w') as f:
        json.dump(v2, f, ensure_ascii=False, indent=1)
    log(f'vocab: {len(v1)} -> {len(v2)} tokens ({len(new)} sfx appended), '
        f'every pre-existing id unchanged')
    return v2


def stage_pack(base, out_dir, v1_path, v2_path, keep, drop):
    wanted = defaultdict(set)
    frames = {}
    for nv_repo, src_repo in SOURCES.items():
        frames[nv_repo] = rows(nv_repo)
        for f in frames[nv_repo]['file']:
            m = token_member(f)
            if m:
                wanted[(src_repo, m[0])].add(m[1])

    codes_by_member, missing_members = load_codes(wanted)

    # which families survive the filter, in a stable order
    families = Counter()
    for df in frames.values():
        for t in df['tagged_text'].fillna(''):
            families.update(SFX_RE.findall(str(t)))
    chosen = [f for f in sorted(families) if (f in keep if keep else f not in drop)]
    log(f'families present: {dict(families)}')
    log(f'families kept: {chosen}')
    sfx_tokens = [f'<|sfx:{f}|>' for f in chosen]
    dropped_tokens = [f'<|sfx:{f}|>' for f in sorted(families) if f not in chosen]

    write_vocab(v1_path, v2_path, sfx_tokens)
    tokenizer, _ = tokenizer_from(v2_path)

    from chinidataset import ParquetWriter, StreamingDataset
    from chinidataset.util import merge_index

    stats = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    with ParquetWriter(out=str(out_dir / '00000'), columns=COLUMNS, compression=None,
                       hashes=HASHES, size_limit=256 * 1024 * 1024) as writer:
        acc = BlockAccumulator(writer)
        for nv_repo, df in frames.items():
            src_repo = SOURCES[nv_repo]
            for file_path, tagged, language, n_placed in zip(
                    df['file'], df['tagged_text'].fillna(''), df['language'],
                    df.get('n_placed', [1] * len(df))):
                stats['rows'] += 1
                if not int(n_placed or 0):
                    stats['no_tag_placed'] += 1
                    continue
                m = token_member(file_path)
                if m is None:
                    stats['bad_path'] += 1
                    continue
                codes = codes_by_member.get((src_repo, m[1]))
                if codes is None:
                    stats['missing_token'] += 1
                    continue

                text = clean_text(tagged)
                for t in dropped_tokens:          # a dropped family becomes plain speech
                    text = text.replace(t, '')
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    stats['empty_text'] += 1
                    continue
                if not SFX_RE.search(text):
                    stats['no_tag_left'] += 1
                    continue
                if len(text.split()) > len(codes):
                    stats['ratio'] += 1
                    continue

                s_tokens = ''.join(f'<|s_{c}|>' for c in codes)
                voice = clean_text(language) or 'unk'
                prompt = f'<|im_start|>{voice}: {text}<|speech_start|>{s_tokens}<|im_end|>'
                acc.add(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                stats['docs'] += 1
                stats['tags'] += len(SFX_RE.findall(text))
        acc.flush()
        stats['blocks'] = acc.blocks
        stats['tokens'] = acc.tokens

    merge_index(out_dir)
    stats['missing_members'] = missing_members
    stats['blocks_indexed'] = len(StreamingDataset(local=str(out_dir)))
    summary = dict(sorted(stats.items()))
    summary['families'] = chosen
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', default='/root/share/nonverbal')
    p.add_argument('--out-dir', default=None, help='default <base>/out/nonverbal-tts')
    p.add_argument('--added-tokens-file', default='/root/share/vocab/added_tokens.json',
                   help='the v1 list every existing pack was built against; never modified')
    p.add_argument('--out-tokens-file', default='/root/share/vocab/added_tokens_v2.json',
                   help='written: v1 + the sfx tags, appended')
    p.add_argument('--keep-families', default=None,
                   help='comma-separated allow-list; overrides --drop-families')
    p.add_argument('--drop-families', default=','.join(DEFAULT_DROP))
    p.add_argument('--stage', choices=['tokens', 'pack', 'all'], default='all')
    args = p.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else base / 'out' / 'nonverbal-tts'
    keep = {s.strip() for s in args.keep_families.split(',') if s.strip()} if args.keep_families else None
    drop = {s.strip() for s in args.drop_families.split(',') if s.strip()}

    if args.stage in ('tokens', 'all'):
        stage_tokens(base)
    if args.stage in ('pack', 'all'):
        stage_pack(base, out_dir, args.added_tokens_file, args.out_tokens_file,
                   keep, drop)


if __name__ == '__main__':
    main()
