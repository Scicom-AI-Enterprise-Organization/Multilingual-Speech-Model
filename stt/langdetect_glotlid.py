"""
Text language detector for the ``malaysia-ai/Multilingual-TTS`` dataset.

Goal
----
Given the ``text`` (transcript) column of any subset, decide *what language it
is* -- "like fastText, but better".

Model choice
------------
We use **GlotLID v3** (``cis-lmu/glotlid``) rather than the classic
``lid.176.bin``:

* It *is* a fastText model -- same millisecond-per-sentence speed, no GPU, tiny
  RAM footprint -- so it is a drop-in upgrade, not a heavier architecture.
* It covers **2102 language+script labels** vs 176. This dataset is dominated by
  low-resource languages (Yoruba, Hausa, Igbo, Bambara, Pulaar, Kabyle, Darija,
  Breton, dozens of Indic / dialectal-Arabic sets) that ``lid.176`` cannot even
  emit -- it would silently mislabel them as the nearest high-resource language.
* Labels carry the **script** too (e.g. ``yor_Latn``, ``arb_Arab``), which pairs
  well with the Unicode script detection already in ``postnormalizer.py``.

Trust signals
-------------
Raw probability is not enough for closely related pairs (zsm/ind, bos/hrv/srp,
nob/nno, Arabic dialects), so every prediction also carries:

* ``margin`` -- top-1 minus top-2 probability.  A confident-looking 0.9 with a
  0.05 margin means "one of two sibling languages", not "certain".
* ``group``  -- macro-language group (``ind``/``zsm`` -> ``msa``, ``bos``/``hrv``/
  ``srp`` -> ``hbs``, Arabic dialects -> ``ara`` ...).  Subset-level audits report
  dominance at both label and group level, so sibling confusion doesn't read as
  "mixed-language data".
* optional **ensemble** -- pass ``ensemble=`` (path, or ``"repo_id[:file]"``) to
  also load a second fastText LID (e.g. OpenLID); rows where the two models
  disagree on the language get their confidence capped below 0.5.

Install
-------
    pip install fasttext-numpy2 huggingface_hub pyarrow pandas
    # (classic ``fasttext`` also works if you are on numpy<2)
    # no aiohttp/fsspec needed: remote parquet is read via the HF hub (lazy
    # range-reads for sampling, cached downloads for full scans) or stdlib urllib.

Quick use
---------
    from langdetect_glotlid import LanguageDetector
    lid = LanguageDetector()                 # downloads model.bin on first use
    lid.predict_one("mo náwó mo nára gbogbo ètò ni mo sì ṣe")
    # -> Prediction(label='yor_Latn', lang='yor', script='Latn', prob=0.99, margin=0.99)

Model resolution: explicit ``model_path`` > ``$GLOTLID_MODEL`` env var >
download ``cis-lmu/glotlid``.

CLI
---
    python langdetect_glotlid.py "some text"                  # one-off
    echo -e "line1\nline2" | python langdetect_glotlid.py     # one line per row
    # audit one subset (dominant language, margins, off-language examples);
    # --sample reads only the first N rows via HTTP range requests -- no full download:
    python langdetect_glotlid.py --audit --sample 2000 \
        --parquet hf://datasets/malaysia-ai/Multilingual-TTS/9jalingo-yoruba/train-00000-of-00001.parquet
    # audit EVERY subset of the dataset -> JSONL report (resumable -- rerun to continue):
    python langdetect_glotlid.py --audit-all --out lid_audit.jsonl
    # score every row and write an annotated parquet:
    python langdetect_glotlid.py --parquet <url_or_path> --out scored.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "LanguageDetector", "Prediction", "clean", "GROUPS",
    "score_parquet", "audit_parquet", "audit_dataset", "list_subsets",
]

DEFAULT_DATASET = "malaysia-ai/Multilingual-TTS"

# ---------------------------------------------------------------------------
# Macro-language groups for confusable siblings
# ---------------------------------------------------------------------------
# GlotLID emits fine-grained ISO 639-3 codes; for QC purposes a subset that is
# 60% ``ind`` / 40% ``zsm`` is *not* mixed-language -- it is Malay-family text
# the model can't split reliably.  Group-level dominance makes that visible.

GROUPS: dict[str, str] = {}
for _macro, _members in {
    "msa": ("zsm", "ind", "min", "bjn"),                       # Malay family
    "hbs": ("bos", "hrv", "srp", "cnr"),                       # Serbo-Croatian
    "nor": ("nob", "nno"),                                     # Norwegian
    "fas": ("pes", "prs"),                                     # Persian/Dari
    "swa": ("swh", "swc"),                                     # Swahili
    "uzb": ("uzn", "uzs"),                                     # Uzbek
    "pus": ("pbt", "pst"),                                     # Pashto
    "ara": ("arb", "ars", "arz", "acm", "acq", "aeb", "afb",   # Arabic dialects
            "ajp", "apc", "apd", "ary", "ayn", "ayp"),
}.items():
    for _m in _members:
        GROUPS[_m] = _macro


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
# fastText refuses newlines in ``predict``.  On top of that, these transcripts
# carry alignment / disfluency markers (``<um>``, ``<unk>``, ``[noise]`` ...)
# that are not part of any language and only add noise -- strip them first.

_TAG_RE = re.compile(r"<[^>]{0,20}>")           # <um>, <unk>, <sil>, <laugh> ...
_BRACKET_RE = re.compile(r"\[[^\]]{0,20}\]")    # [noise], [music] ...
_WS_RE = re.compile(r"\s+")


def clean(text: str | None) -> str:
    """Normalize a transcript into a single clean line for fastText."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return _WS_RE.sub(" ", text).strip()


def _scoreable(text: str) -> bool:
    """A row is only worth scoring if it contains at least one letter --
    digits-/punctuation-only transcripts have no language."""
    return bool(text) and any(c.isalpha() for c in text)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    label: str        # full GlotLID label, e.g. "yor_Latn"
    lang: str         # ISO 639-3 part,     e.g. "yor"
    script: str       # ISO 15924 part,     e.g. "Latn"
    prob: float       # model confidence in [0, 1]
    margin: float = 0.0  # top1 prob - top2 prob; small margin = sibling-language ambiguity

    @property
    def group(self) -> str:
        """Macro-language group ("msa" for ind/zsm, "hbs" for bos/hrv/srp, ...)."""
        return GROUPS.get(self.lang, self.lang)

    @property
    def ok(self) -> bool:
        return bool(self.label) and self.label != "und"


_UND = ("und", "und", "", 0.0, 0.0)


def _split_label(raw: str, prob: float, margin: float = 0.0) -> Prediction:
    """Turn a raw fastText label ("__label__yor_Latn") into a Prediction."""
    lbl = raw.replace("__label__", "")
    if "_" in lbl:
        lang, script = lbl.rsplit("_", 1)
    else:
        lang, script = lbl, ""
    return Prediction(label=lbl, lang=lang, script=script,
                      prob=min(float(prob), 1.0), margin=float(margin))


def _resolve_model_spec(spec: str, default_file: str = "model.bin") -> str:
    """Resolve a model spec: local path, or "repo_id[:filename]" on the HF hub."""
    import os
    if os.path.exists(spec):
        return spec
    if "/" in spec:
        from huggingface_hub import hf_hub_download
        repo, _, fn = spec.partition(":")
        return hf_hub_download(repo_id=repo, filename=fn or default_file)
    raise FileNotFoundError(f"model spec not found (not a file, not repo_id): {spec!r}")


class LanguageDetector:
    """Thin, fast wrapper around the GlotLID (and optionally OpenLID) fastText model."""

    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str = "cis-lmu/glotlid",
        filename: str = "model.bin",
        ensemble: str | None = None,   # path or "repo_id[:file]" of a 2nd LID model
    ):
        import os
        import fasttext
        # silence fasttext's harmless "load_model does not return ..." warning
        try:
            fasttext.FastText.eprint = lambda *a, **k: None
        except Exception:
            pass

        model_path = model_path or os.environ.get("GLOTLID_MODEL")
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            model_path = _resolve_model_spec(model_path)
        self.model = fasttext.load_model(model_path)

        self.ensemble = None
        if ensemble is not None:
            self.ensemble = fasttext.load_model(_resolve_model_spec(ensemble))

    # -- single text -------------------------------------------------------
    def predict_one(self, text: str) -> Prediction:
        text = clean(text)
        if not _scoreable(text):
            return Prediction(*_UND)
        labels, probs = self.model.predict(text, k=2)
        margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])
        pred = _split_label(labels[0], probs[0], margin)
        if self.ensemble is not None:
            e_labels, _ = self.ensemble.predict(text, k=1)
            if _split_label(e_labels[0], 0.0).lang != pred.lang:
                # disagreement on the language -> mark low trust, keep GlotLID label
                pred.prob = min(pred.prob, 0.49)
        return pred

    def topk(self, text: str, k: int = 3) -> list[Prediction]:
        text = clean(text)
        if not _scoreable(text):
            return [Prediction(*_UND)]
        labels, probs = self.model.predict(text, k=k)
        return [_split_label(l, p) for l, p in zip(labels, probs)]

    # -- batch (fastText predicts a whole list in one C++ call) ------------
    def predict_batch(self, texts: Sequence[str | None]) -> list[Prediction]:
        cleaned = [clean(t) for t in texts]
        idx = [i for i, t in enumerate(cleaned) if _scoreable(t)]
        out: list[Prediction] = [Prediction(*_UND) for _ in texts]
        if not idx:
            return out
        batch = [cleaned[i] for i in idx]
        labels, probs = self.model.predict(batch, k=2)
        for j, i in enumerate(idx):
            p = probs[j]
            margin = float(p[0] - p[1]) if len(p) > 1 else float(p[0])
            out[i] = _split_label(labels[j][0], p[0], margin)
        if self.ensemble is not None:
            e_labels, _ = self.ensemble.predict(batch, k=1)
            for j, i in enumerate(idx):
                if _split_label(e_labels[j][0], 0.0).lang != out[i].lang:
                    out[i].prob = min(out[i].prob, 0.49)
        return out


# ---------------------------------------------------------------------------
# Parquet access (local path, hf:// or https://huggingface.co URL, generic URL)
# ---------------------------------------------------------------------------

_HF_URL = re.compile(
    r"^(?:hf://datasets/|https://huggingface\.co/datasets/)"
    r"(?P<repo>[^/]+/[^/]+)/(?:(?:resolve|blob)/(?P<rev>[^/]+)/)?(?P<path>.+)$"
)


def _parse_hf_url(source: str):
    """Return (repo_id, revision, path_in_repo) for an HF dataset URL, else None."""
    if not source.startswith(("hf://", "https://huggingface.co/")):
        return None
    m = _HF_URL.match(source)
    if m is None:
        raise ValueError(f"unrecognized HuggingFace dataset URL: {source!r}")
    if m["path"].startswith(("tree/", "tree%2F")):
        raise ValueError("that is a directory listing URL; pass a file "
                         "(.../resolve/main/<subset>/<file>.parquet)")
    return m["repo"], m["rev"] or "main", m["path"]


def _resolve_source(source: str) -> str:
    """Turn a remote parquet into a local path, downloading if needed.

    Local paths are returned as-is.  HF dataset URLs go through the hub cache
    (deduped, resumable).  Any other http(s) URL is streamed to a temp file
    with stdlib ``urllib`` -- no ``aiohttp``/``requests`` required.
    """
    if not source.startswith(("http://", "https://", "hf://")):
        return source

    hf = _parse_hf_url(source)
    if hf is not None:
        repo_id, revision, filename = hf
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename,
                               revision=revision, repo_type="dataset")

    # Generic URL: stream to a temp file.
    import os
    import tempfile
    import urllib.request
    fd, tmp = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    with urllib.request.urlopen(source) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return tmp


def _open_parquet(source: str):
    """ParquetFile over `source`; HF URLs are opened lazily (HTTP range reads),
    so sampling the first N rows never downloads the whole file."""
    import pyarrow.parquet as pq
    hf = _parse_hf_url(source) if source.startswith(("hf://", "https://huggingface.co/")) else None
    if hf is not None:
        repo_id, revision, path = hf
        from huggingface_hub import HfFileSystem
        f = HfFileSystem().open(f"datasets/{repo_id}@{revision}/{path}", "rb")
        return pq.ParquetFile(f)
    return pq.ParquetFile(_resolve_source(source))


def _read_text_column(source: str, text_col: str = "text",
                      sample: int | None = None) -> tuple[list, int]:
    """Return (texts, total_rows).  With ``sample``, reads only the first
    ``sample`` rows (lazy range-reads for HF URLs -- no full download)."""
    import pyarrow.parquet as pq
    if sample is None:
        tbl = pq.read_table(_resolve_source(source), columns=[text_col])
        return tbl.column(text_col).to_pylist(), tbl.num_rows
    pf = _open_parquet(source)
    total = pf.metadata.num_rows
    texts: list = []
    for batch in pf.iter_batches(batch_size=min(sample, 4096), columns=[text_col]):
        texts.extend(batch.column(0).to_pylist())
        if len(texts) >= sample:
            break
    return texts[:sample], total


# ---------------------------------------------------------------------------
# Scoring / auditing
# ---------------------------------------------------------------------------

def score_parquet(
    detector: LanguageDetector,
    source: str,
    text_col: str = "text",
    batch: int = 8192,
    sample: int | None = None,
):
    """Return a pandas DataFrame of the text column annotated with predictions.

    ``df.attrs["rows_total"]`` holds the file's full row count (== len(df)
    unless ``sample`` was used).
    """
    import pandas as pd
    texts, total = _read_text_column(source, text_col, sample=sample)
    preds: list[Prediction] = []
    for i in range(0, len(texts), batch):
        preds.extend(detector.predict_batch(texts[i : i + batch]))
    df = pd.DataFrame(
        {
            text_col: texts,
            "lid_label": [p.label for p in preds],
            "lid_lang": [p.lang for p in preds],
            "lid_script": [p.script for p in preds],
            "lid_group": [p.group for p in preds],
            "lid_prob": [p.prob for p in preds],
            "lid_margin": [p.margin for p in preds],
        }
    )
    df.attrs["rows_total"] = total
    return df


def audit_parquet(
    detector: LanguageDetector,
    source: str,
    text_col: str = "text",
    threshold: float = 0.5,
    top: int = 8,
    sample: int | None = None,
) -> dict:
    """Summarize what language(s) a subset actually contains.

    This is the robust QC signal: regardless of how the folder is named, it
    tells you the *dominant* detected language (at label and macro-group
    level), how confident the model is, and the off-language tail with
    examples.  ``low_margin_rows`` counts rows where top-1 and top-2 labels
    were nearly tied -- sibling-language ambiguity that raw prob hides.
    """
    df = score_parquet(detector, source, text_col, sample=sample)
    n = len(df)
    conf = df[df.lid_prob >= threshold]
    dist = Counter(conf.lid_label)
    gdist = Counter(conf.lid_group)
    dominant, dom_n = dist.most_common(1)[0] if dist else ("und", 0)
    gdominant, gdom_n = gdist.most_common(1)[0] if gdist else ("und", 0)
    off = conf[conf.lid_group != gdominant]
    examples = [
        {"label": r.lid_label, "text": str(r[text_col])[:120]}
        for _, r in off.head(3).iterrows()
    ]
    return {
        "source": source,
        "rows_total": int(df.attrs.get("rows_total", n)),
        "rows_scored": n,
        "mean_prob": round(float(df.lid_prob.mean()), 4) if n else 0.0,
        "confident_rows": int(len(conf)),
        "low_confidence_rows": int((df.lid_prob < threshold).sum()),
        "low_margin_rows": int((df.lid_margin < 0.1).sum()),
        "dominant_label": dominant,
        "dominant_share": round(dom_n / n, 4) if n else 0.0,
        "dominant_group": gdominant,
        "dominant_group_share": round(gdom_n / n, 4) if n else 0.0,
        "distribution": dist.most_common(top),
        "offlang_examples": examples,
    }


# ---------------------------------------------------------------------------
# Whole-dataset sweep
# ---------------------------------------------------------------------------

def list_subsets(repo_id: str = DEFAULT_DATASET) -> dict[str, list[str]]:
    """Map subset folder -> its parquet files, for every subset in the dataset."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    subsets: dict[str, list[str]] = {}
    for f in files:
        if f.endswith(".parquet") and "/" in f:
            subsets.setdefault(f.split("/", 1)[0], []).append(f)
    return dict(sorted(subsets.items()))


def audit_dataset(
    detector: LanguageDetector,
    repo_id: str = DEFAULT_DATASET,
    out_path: str = "lid_audit.jsonl",
    text_col: str = "text",
    sample: int = 1024,
    threshold: float = 0.5,
    limit: int | None = None,
) -> str:
    """Audit every subset of ``repo_id`` -> one JSONL record per subset.

    Samples the first ``sample`` rows of each subset's first parquet via HTTP
    range reads (no full downloads).  Resumable: subsets already present in
    ``out_path`` are skipped, so rerun the same command to continue.
    """
    import os
    subsets = list_subsets(repo_id)
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["subset"])
                except Exception:
                    pass
    todo = [s for s in subsets if s not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(subsets)} subsets, {len(done)} already audited, "
          f"{len(todo)} to do -> {out_path}", file=sys.stderr)
    with open(out_path, "a") as out:
        for i, name in enumerate(todo, 1):
            files = subsets[name]
            url = f"hf://datasets/{repo_id}/{files[0]}"
            try:
                rec = audit_parquet(detector, url, text_col, threshold, sample=sample)
                rec = {"subset": name, "files_total": len(files), **rec}
                status = (f"{rec['dominant_label']} "
                          f"({rec['dominant_share']:.0%}, {rec['rows_scored']} rows)")
            except Exception as e:  # keep sweeping; record the failure
                rec = {"subset": name, "files_total": len(files),
                       "error": f"{type(e).__name__}: {e}"}
                status = rec["error"][:80]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {name}: {status}", file=sys.stderr)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GlotLID text language detector")
    ap.add_argument("text", nargs="*", help="text to classify (else read stdin)")
    ap.add_argument("--model", help="local model.bin path or 'repo_id[:file]' "
                                    "(default: $GLOTLID_MODEL or cis-lmu/glotlid)")
    ap.add_argument("--ensemble", help="2nd LID model (e.g. OpenLID) path or 'repo_id[:file]'")
    ap.add_argument("--parquet", help="parquet path/URL to score")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out", help="output path (annotated parquet, or JSONL for --audit-all)")
    ap.add_argument("--audit", action="store_true", help="print subset language summary")
    ap.add_argument("--audit-all", action="store_true",
                    help="audit every subset of --dataset -> JSONL (resumable)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--sample", type=int,
                    help="only score the first N rows (lazy read, no full download)")
    ap.add_argument("--limit", type=int, help="audit-all: stop after N subsets")
    ap.add_argument("-k", type=int, default=1, help="show top-k for text/stdin mode")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    lid = LanguageDetector(model_path=args.model, ensemble=args.ensemble)

    if args.audit_all:
        audit_dataset(lid, args.dataset, args.out or "lid_audit.jsonl",
                      args.text_col, sample=args.sample or 1024,
                      threshold=args.threshold, limit=args.limit)
        return 0

    if args.parquet:
        if args.audit:
            print(json.dumps(
                audit_parquet(lid, args.parquet, args.text_col,
                              args.threshold, sample=args.sample),
                indent=2, ensure_ascii=False))
        else:
            df = score_parquet(lid, args.parquet, args.text_col, sample=args.sample)
            if args.out:
                df.to_parquet(args.out, index=False)
                print(f"wrote {len(df)} rows -> {args.out}")
            else:
                print(df.head(20).to_string())
        return 0

    def emit(text: str):
        if args.k > 1:
            for p in lid.topk(text, k=args.k):
                print(f"{p.label}\t{p.prob:.4f}\t{text[:60]}")
        else:
            p = lid.predict_one(text)
            print(f"{p.label}\t{p.prob:.4f}\tmargin={p.margin:.4f}\t{text[:80]}")

    if args.text:
        emit(" ".join(args.text))
    else:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line:
                emit(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
