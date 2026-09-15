"""Build a low-resource-language TTS test set from malaysia-ai/Multilingual-TTS.

50 of the lowest-resource languages in the corpus x 25 utterances each, every row
carrying the audio, the NeuCodec speech tokens and the normalized transcription:

    audio (mp3 bytes) + tokens (<|s_N|> ids) + post-normalized text

Sources
-------
- malaysia-ai/Multilingual-TTS-language : metadata + GlotLID `language` +
  `post-normalized` columns (one parquet dir per subset)
- malaysia-ai/Multilingual-TTS          : `<folder>.zip` (mp3) and
  `<folder>_neucodec.zip` (token JSON) per audio folder

Nothing is bulk-downloaded: parquet reads are column-pruned and zip members are
pulled with HTTP range requests against the zip's central directory, so building
the whole test set moves a few hundred MB instead of the corpus' ~40TB.

Stages (all resumable, each writes into --work-dir):

    python build_testset.py --stage scan     # per-parquet language histogram
    python build_testset.py --stage select   # -> selection.json  (languages + rows)
    python build_testset.py --stage fetch    # -> fetched/        (mp3 + token json)
    python build_testset.py --stage push     # -> malaysia-ai/Low-Language-TTS
    python build_testset.py --stage verify   # re-download and re-check the invariants
    python build_testset.py --stage all

Why "lowest-resource" is not "rarest GlotLID label": measured on this corpus, of
the 500 rarest labels only 3 survived row-level verification. That tail is the
detector firing on short lines of other languages (a three-word Hausa line lands
on `wnc_Latn`) plus `und_<script>` false positives. A language is taken seriously
only when some subset was *collected for it* -- >= ATTEST_ROWS rows and >=
ATTEST_SHARE of that subset (attested_languages()). 177 of the 2,100 labels clear
that bar, and they rank into a clean low-resource tail: Wayuu, Berom, Iban,
Baoule, Skolt Sami, Karamojong. Rows are then verified individually (verify()).
"""

import os

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

META_REPO = 'malaysia-ai/Multilingual-TTS-language'
AUDIO_REPO = 'malaysia-ai/Multilingual-TTS'
PUSH_REPO = 'malaysia-ai/Low-Language-TTS'

N_LANGUAGES = 50
N_PER_LANGUAGE = 25
CANDIDATES_PER_LANGUAGE = 60      # over-select rows: some members are missing from the zips
SELECT_EXTRA_LANGUAGES = 10       # over-select languages too, for the same reason
MAX_FILES_PER_LANGUAGE = 4        # parquet files read per candidate language
MAX_ROWS_PER_SPEAKER = 4          # speaker diversity inside one language
CANDIDATE_POOL = 150              # rarest attested languages put through verification
ATTEST_ROWS = 50                  # a language is real when some subset was collected for it:
ATTEST_SHARE = 0.4                # >= ATTEST_ROWS rows AND >= ATTEST_SHARE of that subset
MAX_SUBSETS_PER_LANGUAGE = 3      # attested subsets read per candidate language

MIN_CHARS = 40                    # GlotLID is unreliable below ~40 chars
MIN_WORDS = 5
MIN_PROB = 0.90                   # re-predicted probability on post-normalized text
MIN_MARGIN = 0.50                 # top1 - top2: no sibling-language ambiguity
MIN_TOKENS = 50                   # 1s of NeuCodec at 50 tokens/s
MAX_TOKENS = 1500                 # 30s

_local = threading.local()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fs():
    if not hasattr(_local, 'fs'):
        _local.fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
    return _local.fs


def api():
    return HfApi(token=os.environ.get('HF_TOKEN'))


def retry(fn, what, tries=4, wait=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            log(f'{what} failed ({type(e).__name__}: {e}); retry {attempt + 1}/{tries - 1}')
            _local.fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
            time.sleep(wait * (attempt + 1))


# ------------------------------------------------------------------ stage: scan

def counts_key(parquet_file):
    """Filename for one parquet's histogram.

    Hashed, not derived from the path: subset names collide both on separators
    (`ATC_combined__Tabys`) and on case (`MSC` vs `msc`, `Sagalee` vs `sagalee`
    -- indistinguishable on macOS), and a collision silently drops a subset from
    the histogram. The parquet path is stored inside the file instead.
    """
    return hashlib.sha1(parquet_file.encode()).hexdigest()[:16] + '.json'


def scan(work, workers):
    """Per-parquet language histogram, reading only the `language` column."""
    out = work / 'counts'
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in api().list_repo_files(META_REPO, repo_type='dataset')
                   if f.endswith('.parquet'))
    todo = [f for f in files if not (out / counts_key(f)).exists()]
    log(f'{META_REPO}: {len(files)} parquet, {len(todo)} to scan')

    def one(f):
        def go():
            with fs().open(f'datasets/{META_REPO}/{f}', 'rb') as fh:
                tb = pq.ParquetFile(fh).read(columns=['language'])
            (out / counts_key(f)).write_text(json.dumps(
                {'file': f, 'rows': tb.num_rows,
                 'counts': Counter(tb.column('language').to_pylist())}))
        retry(go, f)

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(one, f) for f in todo]):
            fut.result()
            done += 1
            if done % 100 == 0:
                log(f'  scanned {done}/{len(todo)}')
    log('scan done')


def load_counts(work):
    """{language: total} and {language: {parquet file: count}} from the scan."""
    totals, per_file = Counter(), defaultdict(Counter)
    for p in (work / 'counts').glob('*.json'):
        d = json.loads(p.read_text())
        for lang, c in d['counts'].items():
            totals[lang] += c
            per_file[lang][d['file']] = c
    return totals, per_file


# ---------------------------------------------------------------- stage: select

def zip_index(repo_files):
    """audio folder -> (audio zips, neucodec zip). Handles `folder-0-1.zip` shards."""
    audio, neucodec = defaultdict(list), {}
    for f in repo_files:
        if not f.endswith('.zip'):
            continue
        stem = f[:-len('.zip')]
        if stem.endswith('_neucodec'):
            neucodec[stem[:-len('_neucodec')]] = f
        else:
            audio[re.sub(r'-\d+-\d+$', '', stem)].append(f)
    return audio, neucodec


def subset_histograms(per_file):
    """subset directory -> Counter of languages in it."""
    sub = defaultdict(Counter)
    for lang, files in per_file.items():
        for f, c in files.items():
            sub[f.split('/')[0]][lang] += c
    return sub


def attested_languages(sub):
    """language -> [(subset, rows, share)] of the subsets that were collected for it.

    This is the load-bearing filter. Ranking the global histogram and taking the
    rarest labels returns almost nothing usable -- of the 500 rarest labels only
    3 survived row-level verification, because that tail is GlotLID firing on
    short lines of other languages. A language that *dominates a whole subset*,
    on the other hand, is a language somebody actually recorded a corpus for.
    """
    out = defaultdict(list)
    for s, counts in sub.items():
        total = sum(counts.values())
        for lang, n in counts.items():
            if lang.startswith('und') or n < ATTEST_ROWS or n / total < ATTEST_SHARE:
                continue
            out[lang].append((s, n, round(n / total, 3)))
    for lang in out:
        out[lang].sort(key=lambda t: -t[1])
    return out


def verify(rows, lid):
    """Keep rows whose transcript is long enough to classify and whose stored
    GlotLID label is reproduced at high probability/margin."""
    texts = [r['post-normalized'] for r in rows]
    preds = []
    for i in range(0, len(texts), 4096):
        preds.extend(lid.predict_batch(texts[i:i + 4096]))
    keep = []
    for r, p in zip(rows, preds):
        if p.label != r['language'] or p.prob < MIN_PROB or p.margin < MIN_MARGIN:
            continue
        r['lid_prob'], r['lid_margin'] = round(float(p.prob), 4), round(float(p.margin), 4)
        keep.append(r)
    return keep


def read_rows(f, wanted, cap):
    """Rows of `f` whose language is in `wanted`, at most `cap` per language.

    Streamed row group by row group: the biggest parquet here is 356MB and a
    couple of dozen of those materialised at once would not fit in RAM.
    """
    cols = ['audio_filename', 'text', 'speaker', 'language', 'post-normalized']
    subset = f.split('/')[0]

    def go():
        out, taken = [], Counter()
        with fs().open(f'datasets/{META_REPO}/{f}', 'rb') as fh:
            pf = pq.ParquetFile(fh)
            if not set(cols) <= set(pf.schema_arrow.names):
                return []
            for batch in pf.iter_batches(batch_size=50_000, columns=cols):
                idx = []
                for i, l in enumerate(batch.column('language').to_pylist()):
                    if l in wanted and taken[l] < cap:
                        taken[l] += 1
                        idx.append(i)
                if idx:
                    for r in batch.take(idx).to_pylist():
                        r['subset'] = subset
                        out.append(r)
        return out
    return retry(go, f)


# Thai/Lao/Khmer/Burmese/Tibetan/Han/Kana are written without spaces, so a word
# count is meaningless there and characters carry the signal instead.
NO_SPACE_SCRIPT = re.compile(
    r'[\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u1780-\u17ff'
    r'\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]')


def enough_text(text):
    """Long enough for GlotLID to be trusted."""
    if len(text) < MIN_CHARS:
        return False
    if len(text.split()) >= MIN_WORDS:
        return True
    return len(NO_SPACE_SCRIPT.findall(text)) >= len(text) / 2


def eligible(row, audio_zips, neucodec_zips):
    name = row.get('audio_filename') or ''
    if '/' not in name or not enough_text((row.get('post-normalized') or '').strip()):
        return False
    folder = name.split('/')[0]
    return folder in neucodec_zips and folder in audio_zips


def select(work, workers, refresh=False):
    """Pick the lowest-resource attested languages that still have 25 clean rows."""
    from langdetect_glotlid import LanguageDetector

    totals, per_file = load_counts(work)
    audio_zips, neucodec_zips = zip_index(api().list_repo_files(AUDIO_REPO, repo_type='dataset'))
    attested = attested_languages(subset_histograms(per_file))
    log(f'{len(totals)} GlotLID labels over {sum(totals.values()):,} rows; '
        f'{len(attested)} of them are attested by a subset of their own')

    ranked = sorted(attested, key=lambda l: totals[l])
    pool = ranked[:CANDIDATE_POOL]
    log(f'verifying the {len(pool)} lowest-resource '
        f'({totals[pool[0]]:,}..{totals[pool[-1]]:,} corpus rows)')

    want_files = defaultdict(set)   # parquet file -> candidate languages to pull from it
    for lang in pool:
        subsets = {s for s, _, _ in attested[lang][:MAX_SUBSETS_PER_LANGUAGE]}
        for f in per_file[lang]:
            if f.split('/')[0] in subsets:
                want_files[f].add(lang)
    log(f'{len(want_files)} parquet files to read')

    # cached before any filtering, so re-tuning the gates costs no re-download
    cache = work / 'candidates.json'
    if cache.exists() and not refresh:
        raw = defaultdict(list, json.loads(cache.read_text()))
        log(f'reusing {cache} ({sum(len(v) for v in raw.values()):,} rows) '
            f'-- pass --refresh to re-read the parquet')
    else:
        raw = defaultdict(list)
        with ThreadPoolExecutor(workers) as ex:
            # 20x the needed rows is plenty of headroom for the gates below, and
            # keeps the cache (and peak RSS) to a few hundred MB
            futs = [ex.submit(read_rows, f, langs, CANDIDATES_PER_LANGUAGE * 20)
                    for f, langs in want_files.items()]
            for n, fut in enumerate(as_completed(futs), 1):
                for r in fut.result():
                    raw[r['language']].append(r)
                if n % 25 == 0 or n == len(futs):
                    log(f'  read {n}/{len(futs)} parquet, '
                        f'{sum(len(v) for v in raw.values()):,} rows')
        cache.write_text(json.dumps(raw, ensure_ascii=False))

    by_lang = {l: [r for r in rows if eligible(r, audio_zips, neucodec_zips)]
               for l, rows in raw.items()}
    log(f'{sum(len(v) for v in by_lang.values()):,} of {sum(len(v) for v in raw.values()):,} '
        f'rows are long enough and have audio + tokens')

    lid = LanguageDetector()
    chosen, short = {}, []
    for lang in pool:
        # a few languages lose most of their rows at fetch time (missing token
        # JSONs upstream), so carry spares and trim to N_LANGUAGES in push
        if len(chosen) >= N_LANGUAGES + SELECT_EXTRA_LANGUAGES:
            break
        picked = pick_rows(verify(by_lang.get(lang, []), lid))
        if len(picked) >= N_PER_LANGUAGE:
            chosen[lang] = {
                'corpus_rows': totals[lang],
                'attested': attested[lang][:MAX_SUBSETS_PER_LANGUAGE],
                'candidates': picked,
            }
            log(f'  + {lang}: {totals[lang]:,} corpus rows, {len(picked)} candidates, '
                f'{len(set(r["subset"] for r in picked))} subset(s)')
        else:
            short.append((lang, len(picked)))

    log(f'{len(short)} candidate languages fell short: '
        + ', '.join(f'{l}({n})' for l, n in short[:20]))
    if len(chosen) < N_LANGUAGES:
        log(f'WARNING: only {len(chosen)}/{N_LANGUAGES} languages qualified — '
            f'raise CANDIDATE_POOL and re-run')
    (work / 'selection.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=1))
    log(f'selection.json: {len(chosen)} languages, '
        f'{sum(len(v["candidates"]) for v in chosen.values())} candidate rows')


def pick_rows(rows):
    """Most confident rows first, deduped by text and spread over speakers.

    The speaker cap is relaxed when it would disqualify the language: a lot of the
    tail lives in single-speaker corpora (one Bible reader, one radio archive), and
    dropping those would bias the test set towards the languages that happen to have
    many speakers -- exactly the ones that are not low-resource.
    """
    rows.sort(key=lambda r: (-r['lid_prob'], -r['lid_margin']))
    for cap in (MAX_ROWS_PER_SPEAKER, MAX_ROWS_PER_SPEAKER * 2, N_PER_LANGUAGE, len(rows) or 1):
        picked, per_speaker, texts = [], Counter(), set()
        for r in rows:
            t = ' '.join(r['post-normalized'].lower().split())
            if t in texts or per_speaker[r['speaker']] >= cap:
                continue
            texts.add(t)
            per_speaker[r['speaker']] += 1
            picked.append(r)
            if len(picked) >= CANDIDATES_PER_LANGUAGE:
                break
        if len(picked) >= N_PER_LANGUAGE:
            break
    return picked


# ----------------------------------------------------------------- stage: fetch

def _local_header_data(blob, info):
    """Member bytes out of an over-fetched range starting at the local header.

    The range is deliberately longer than the member, so the payload MUST be cut
    to compress_size: a stored member otherwise comes back with the next entry's
    header glued to its tail (valid-looking mp3, unparseable JSON).
    """
    n, m = struct.unpack('<HH', blob[26:30])
    start = 30 + n + m
    data = blob[start:start + info.compress_size]
    if len(data) != info.compress_size:
        raise ValueError(f'short read: {len(data)} of {info.compress_size} bytes')
    if info.compress_type == 8:
        data = zlib.decompress(data, -15)
    elif info.compress_type:
        raise ValueError(f'unsupported zip compression {info.compress_type}')
    if len(data) != info.file_size:
        raise ValueError(f'size mismatch: {len(data)} != {info.file_size}')
    return data


def read_members(repo, zip_path, names, workers):
    """{name: bytes} for `names` inside a remote zip, via range requests."""
    def cd():
        with fs().open(f'datasets/{repo}/{zip_path}', 'rb') as fh:
            z = zipfile.ZipFile(fh)
            return {n: z.getinfo(n) for n in names if n in z.NameToInfo}
    infos = retry(cd, f'central directory of {zip_path}')

    def one(item):
        name, info = item
        end = info.header_offset + 30 + 512 + info.compress_size

        def go():
            blob = fs().cat_file(f'datasets/{repo}/{zip_path}',
                                 start=info.header_offset, end=end)
            return _local_header_data(blob, info)
        return name, retry(go, f'{zip_path}:{name}')

    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(one, it) for it in infos.items()]):
            name, data = fut.result()
            out[name] = data
    return out


def token_member(audio_filename):
    """`folder/rest.mp3` -> `folder_neucodec/rest.json` (same mapping as
    stt/multipacking_stt.py:token_json_path)."""
    folder, _, rest = audio_filename.partition('/')
    return f'{folder}_neucodec/' + rest.rsplit('.', 1)[0] + '.json'


def fetch(work, workers):
    selection = json.loads((work / 'selection.json').read_text())
    out = work / 'fetched'
    out.mkdir(parents=True, exist_ok=True)
    audio_zips, neucodec_zips = zip_index(api().list_repo_files(AUDIO_REPO, repo_type='dataset'))

    # group every candidate row by the audio folder it lives in
    by_folder = defaultdict(list)
    for lang, info in selection.items():
        for i, r in enumerate(info['candidates']):
            r['lang'], r['idx'] = lang, i
            by_folder[r['audio_filename'].split('/')[0]].append(r)
    log(f'{len(by_folder)} audio folders to pull from')

    # a folder is marked done rather than inferred from file counts: some members
    # are simply absent upstream, so "every row present" never becomes true and
    # the folder would be re-pulled on every run
    marker_dir = out / '.done'
    marker_dir.mkdir(exist_ok=True)
    for n, (folder, rows) in enumerate(sorted(by_folder.items()), 1):
        marker = marker_dir / f'{folder}.{len(rows)}'
        if marker.exists():
            continue
        log(f'[{n}/{len(by_folder)}] {folder}: {len(rows)} rows '
            f'({len(audio_zips[folder])} audio zip(s))')
        tok_names = {token_member(r['audio_filename']): r for r in rows}
        toks = read_members(AUDIO_REPO, neucodec_zips[folder], set(tok_names), workers)
        log(f'  {len(toks)}/{len(tok_names)} token json')

        have = {r['audio_filename'] for name, r in tok_names.items() if name in toks}
        auds = {}
        for zp in sorted(audio_zips[folder]):
            missing = have - set(auds)
            if not missing:
                break
            auds.update(read_members(AUDIO_REPO, zp, missing, workers))
        log(f'  {len(auds)}/{len(have)} audio')

        for name, r in tok_names.items():
            if name not in toks or r['audio_filename'] not in auds:
                continue
            ext = r['audio_filename'].rsplit('.', 1)[-1]
            (out / f'{r["lang"]}__{r["idx"]}.{ext}').write_bytes(auds[r['audio_filename']])
            (out / f'{r["lang"]}__{r["idx"]}.json').write_text(json.dumps(
                {**r, 'tokens': json.loads(toks[name])}, ensure_ascii=False))
        marker.touch()
    log('fetch done')


# ------------------------------------------------------------------ stage: push

def flatten_tokens(tok):
    while isinstance(tok, list) and tok and isinstance(tok[0], list):
        tok = tok[0] if len(tok) == 1 else [t for sub in tok for t in sub]
    return [int(t) for t in tok]


def audio_info(blob):
    """(duration seconds, sampling rate) of an encoded clip, None if undecodable."""
    import io

    import soundfile as sf
    try:
        info = sf.info(io.BytesIO(blob))
        return info.frames / info.samplerate, info.samplerate
    except Exception:
        return None


def build_rows(work):
    selection = json.loads((work / 'selection.json').read_text())
    out, dropped = [], Counter()
    for lang in sorted(selection, key=lambda l: selection[l]['corpus_rows']):
        picked = []
        for i in range(len(selection[lang]['candidates'])):
            meta_p = work / 'fetched' / f'{lang}__{i}.json'
            if not meta_p.exists():
                dropped['no member'] += 1
                continue
            meta = json.loads(meta_p.read_text())
            audio_p = next((p for p in (work / 'fetched').glob(f'{lang}__{i}.*')
                            if p.suffix != '.json'), None)
            tokens = flatten_tokens(meta['tokens'])
            if audio_p is None or not audio_p.stat().st_size:
                dropped['empty audio'] += 1
                continue
            if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
                dropped['token length'] += 1
                continue
            if len(meta['post-normalized'].split()) > len(tokens):
                dropped['more words than tokens'] += 1
                continue
            audio_bytes = audio_p.read_bytes()
            info = audio_info(audio_bytes)
            if info is None:
                dropped['undecodable audio'] += 1
                continue
            # tokens and mp3 come from different zips: if they disagree on length
            # the path convention mapped this row onto someone else's clip
            if abs(info[0] - len(tokens) / 50) > max(0.4, 0.2 * info[0]):
                dropped['audio/token length mismatch'] += 1
                continue
            picked.append({
                'language': lang,
                'audio': {'path': f'{lang}/{audio_p.name}', 'bytes': audio_bytes},
                'audio_filename': meta['audio_filename'],
                'text': meta['text'],
                'post-normalized': meta['post-normalized'],
                'tokens': tokens,
                'duration': round(info[0], 3),
                'sampling_rate': info[1],
                'speaker': meta['speaker'],
                'subset': meta['subset'],
                'lid_prob': meta['lid_prob'],
                'lid_margin': meta['lid_margin'],
            })
            if len(picked) >= N_PER_LANGUAGE:
                break
        # a short language is dropped, not padded: a spare language with all 25
        # rows is worth more than a ragged one
        if len(picked) < N_PER_LANGUAGE:
            log(f'  dropping {lang}: only {len(picked)}/{N_PER_LANGUAGE} rows survived')
            dropped['incomplete language'] += len(picked)
            continue
        if len(set(r['language'] for r in out)) >= N_LANGUAGES:
            continue
        out.extend(picked)
    log(f'{len(out)} rows over {len(set(r["language"] for r in out))} languages; '
        f'dropped {dict(dropped)}')
    return out


def push(work, dry_run=False):
    from datasets import Audio, Dataset

    rows = build_rows(work)
    ds = Dataset.from_list(rows).cast_column('audio', Audio())
    out = work / 'low-language-tts.parquet'
    ds.to_parquet(str(out))
    log(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')
    if dry_run:
        return
    ds.push_to_hub(PUSH_REPO, split='test', token=os.environ.get('HF_TOKEN'),
                   commit_message=f'{len(rows)} utterances over '
                                  f'{len(set(r["language"] for r in rows))} low-resource languages')
    log(f'pushed to {PUSH_REPO}')
    write_card(work, rows)


def write_card(work, rows):
    """Keep the front matter push_to_hub generated, replace the body."""
    import io

    from huggingface_hub import hf_hub_download

    selection = json.loads((work / 'selection.json').read_text())
    per_lang = Counter(r['language'] for r in rows)
    hours = sum(r['duration'] for r in rows) / 3600
    table = '\n'.join(
        f'| `{l}` | {per_lang[l]} | {selection[l]["corpus_rows"]:,} | '
        f'{sum(r["duration"] for r in rows if r["language"] == l) / 60:.1f} | '
        + ', '.join(sorted({r['subset'] for r in rows if r['language'] == l})) + ' |'
        for l in sorted(per_lang, key=lambda l: selection[l]['corpus_rows']))

    body = rf"""# Low-Language-TTS

A held-out TTS/ASR test set for the **long tail** of
[malaysia-ai/Multilingual-TTS](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS):
{len(per_lang)} of the lowest-resource languages in that corpus,
{N_PER_LANGUAGE} utterances each ({len(rows)} rows, {hours:.2f} hours).

Every row carries the three things needed to score a NeuCodec speech-token model
without touching the parent corpus:

| column | what |
|---|---|
| `audio` | the original clip, exactly as stored upstream (mostly mp3) |
| `tokens` | NeuCodec speech tokens at 50 tokens/s — the `<\|s_N\|>` ids the Multilingual-TTS models emit |
| `post-normalized` | normalized transcription (`postnormalizer.py` output, the reference text for CER) |
| `text` | raw upstream transcript |
| `language` | GlotLID v3 label, e.g. `mos_Latn` |
| `speaker`, `subset`, `audio_filename` | provenance back into the parent corpus |
| `duration`, `sampling_rate` | decoded clip length in seconds, and its sample rate |
| `lid_prob`, `lid_margin` | language-ID confidence of the row (see below) |

## How the languages were picked

Ranking the corpus' GlotLID histogram and taking the rarest labels does **not**
work, and this was measured: of the 500 rarest labels, exactly **3** survived
row-level verification. Short transcripts are misclassified constantly (a
three-word Hausa line lands on `wnc_Latn`) and `und_<script>` labels are
punctuation false positives, so that tail is detector noise, not languages.

A language is instead taken seriously when **some subset of the corpus was
collected for it** — at least {ATTEST_ROWS} rows *and* at least {ATTEST_SHARE:.0%} of that
subset. 177 of the 2,100 labels clear that bar, and they rank into a genuine
low-resource tail. The {N_LANGUAGES} with the fewest rows in the parent corpus that can
still fill {N_PER_LANGUAGE} verified utterances are what you see here.

Individual rows are then verified too, because a subset's minority rows are not
its language. A row is kept only when

- the normalized transcript is >= {MIN_CHARS} characters and >= {MIN_WORDS} words (characters
  instead of words for scripts written without spaces), long enough for GlotLID
  to be reliable,
- re-running GlotLID v3 reproduces the stored label with probability
  >= {MIN_PROB} and top1-top2 margin >= {MIN_MARGIN},
- NeuCodec tokens exist for the clip and decode to {MIN_TOKENS}-{MAX_TOKENS} tokens (1-30s),
- the transcript has no more words than speech tokens,
- the decoded audio length and `len(tokens) / 50` agree to within 20% — audio and
  tokens live in different zips upstream, and this is what catches a mispairing.

Rows are deduplicated by text and spread over speakers ({MAX_ROWS_PER_SPEAKER} per speaker,
relaxed only when a language would otherwise not reach {N_PER_LANGUAGE} — much of the tail
is single-speaker corpora). On long, confident rows the stored and re-predicted
labels agree 97-99% of the time, so the language tag here is considerably more
trustworthy than the parent corpus' raw `language` column.

## Languages

`corpus rows` is how many rows that language has in the whole 121.8M-row parent corpus.

| language | rows | corpus rows | minutes | subset(s) |
|---|---:|---:|---:|---|
{table}

## Usage

```python
from datasets import load_dataset

ds = load_dataset('{PUSH_REPO}', split='test')
row = ds[0]
row['audio']['array'], row['tokens'], row['post-normalized'], row['language']
```

Built by `low-language-testset/build_testset.py` in the Multilingual-Speech-Model repo,
from [malaysia-ai/Multilingual-TTS](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS)
(audio + NeuCodec tokens) and
[malaysia-ai/Multilingual-TTS-language](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS-language)
(language + normalized transcription).
"""

    readme = pathlib.Path(hf_hub_download(PUSH_REPO, 'README.md', repo_type='dataset',
                                          token=os.environ.get('HF_TOKEN'))).read_text()
    front = readme.split('---\n')[1] if readme.startswith('---\n') else ''
    card = f'---\n{front}---\n\n{body}' if front else body
    HfApi(token=os.environ.get('HF_TOKEN')).upload_file(
        path_or_fileobj=io.BytesIO(card.encode()), path_in_repo='README.md',
        repo_id=PUSH_REPO, repo_type='dataset', commit_message='Dataset card')
    (work / 'README.md').write_text(card)
    log('card written')


# ---------------------------------------------------------------- stage: verify

def verify_pushed():
    """Re-download what was published and re-check every invariant."""
    from datasets import Audio, load_dataset

    ds = load_dataset(PUSH_REPO, split='test', token=os.environ.get('HF_TOKEN'))
    ds = ds.cast_column('audio', Audio(decode=False))
    langs = Counter(ds['language'])
    log(f'{len(ds)} rows, {len(langs)} languages, '
        f'{sum(ds["duration"]) / 3600:.2f} hours')
    log(f'rows per language: min {min(langs.values())}, max {max(langs.values())}')

    bad = Counter()
    for r in ds:
        if not r['audio']['bytes']:
            bad['no audio bytes'] += 1
            continue
        info = audio_info(r['audio']['bytes'])
        if info is None:
            bad['undecodable'] += 1
            continue
        if abs(info[0] - len(r['tokens']) / 50) > max(0.4, 0.2 * info[0]):
            bad['audio/token length mismatch'] += 1
        if not r['post-normalized'].strip():
            bad['empty transcription'] += 1
        if min(r['tokens']) < 0 or max(r['tokens']) > 65535:
            bad['token id out of NeuCodec range'] += 1
    dupes = len(ds) - len({(r['audio_filename']) for r in ds})
    if dupes:
        bad['duplicate audio_filename'] = dupes
    log(f'problems: {dict(bad) or "none"}')
    return not bad


# ------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', default='all',
                   choices=['scan', 'select', 'fetch', 'push', 'verify', 'all'])
    p.add_argument('--work-dir', default='work')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--refresh', action='store_true',
                   help='select stage: re-read the parquet instead of reusing candidates.json')
    p.add_argument('--dry-run', action='store_true', help='push stage: write parquet, no upload')
    args = p.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'stt'))
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stages = (['scan', 'select', 'fetch', 'push', 'verify']
              if args.stage == 'all' else [args.stage])
    for s in stages:
        log(f'=== stage {s}')
        if s == 'scan':
            scan(work, args.workers)
        elif s == 'select':
            select(work, args.workers, args.refresh)
        elif s == 'fetch':
            fetch(work, args.workers)
        elif s == 'push':
            push(work, args.dry_run)
        elif s == 'verify':
            verify_pushed()


if __name__ == '__main__':
    main()
