# fleurs-dataset

Maintenance tooling for
[malaysia-ai/fleurs-r-neucodec-all-languages](https://huggingface.co/datasets/malaysia-ai/fleurs-r-neucodec-all-languages)
— FLEURS-R metadata + source audio + precomputed NeuCodec tokens for 102 locales, the
corpus the [FLEURS optimizer ablation](../README.md#fleurs-r--common-voice-22--one-mixture-three-audio-tokenizers-scored-per-task)
trains on.

## What was wrong

| gap | detail |
|---|---|
| `audio/` incomplete | only 15 of 204 zips were ever pushed — the upload died after `bg_bg-dev`, so 94 locales had tokens and transcripts but no source audio |
| no speaker | FLEURS and FLEURS-R ship `gender` and nothing else about the voice, so the TTS prompt had nothing to condition on (the multipacking fell back to the locale) |
| viewer dead | the card had no `configs:`, so the HF viewer answered `No (supported) data files found` and the repo previewed nothing |

`data/` (204 parquets) and `neucodec/` (816 zips) were already complete.

## `build_fleurs_repo.py`

```bash
python build_fleurs_repo.py --stage all --workers 12          # everything
python build_fleurs_repo.py --stage audio --locales en_us     # one locale
python build_fleurs_repo.py --stage cluster --threshold 0.4   # re-cluster only
python build_fleurs_repo.py --stage card --dry-run            # print the card
```

| stage | does |
|---|---|
| `audio` | per (locale, split): pull `google/fleurs-r data/{locale}/audio/{split}.tar.gz`, rebuild `audio/{locale}-{split}.zip`, upload it when the repo lacks it, and embed every utterance with TitaNet-L on the way past. Wavs are deleted as soon as the zip and vectors exist (~3GB peak per worker) |
| `cluster` | agglomerative clustering (cosine, average linkage) of the vectors **within each (locale, gender)** → `{locale}_spkNN` |
| `parquet` | writes the `speaker` column into `data/{locale}-{split}.parquet`, keeping the HF features metadata in sync, and re-uploads |
| `card` | README.md with `configs:` (viewer) + column and speaker docs |

Every stage is resumable: uploaded zips are marked in `uploaded/`, vectors cached in
`emb/{locale}-{split}.npz`, speaker maps in `speakers/{locale}.json`.

### Regenerated zips are byte-identical

FLEURS-R audio is 24kHz mono PCM16 WAV. The 15 zips already in the repo hold exactly the
upstream `tar.gz` bytes under `audio/{locale}/{split}/{id}.wav` — verified by md5 against
`google/fleurs-r` before regenerating the rest, so old and new zips are the same artifact.

### Speaker labels are derived, not ground truth

FLEURS has no speaker ids. Each utterance gets a TitaNet-L vector
([titanet-vectors-fp16](https://github.com/Scicom-AI-Enterprise-Organization/titanet-vectors-fp16),
the same embedder `vc-evaluation/calculate_similarity.py` scores voice cloning with), and
vectors are clustered per locale with gender as a hard constraint — a cluster can never
merge voices FLEURS already labels apart. Clusters below `--min-size` utterances fold into
their nearest centroid; ids are numbered by descending size, so `{locale}_spk00` is the
most-recorded voice. Per-locale cluster counts and silhouette scores land in
`speaker_summary.json`.

The 0.4 threshold came from sweeping 0.3/0.4/0.5/0.6 over 8 locales: 0.3 shatters voices
into dozens of fragments (silhouette 0.31–0.51), 0.6 starts merging distinct speakers
(af_za 1370-utterance blob, silhouette drops to 0.25 on ar_eg), and 0.4 sits at the
silhouette peak (0.55–0.61) while erring toward splitting a speaker rather than merging
two — the safer direction for a conditioning tag. Example at 0.4: af_za 1,222 utterances
→ 6 speakers (silhouette 0.72), bg_bg 3,358 → 19 (0.62).

Use them as consistent voice tags for conditioning, not as verified identities.

## Running it (reference box)

Same shared GPU box the multipacking runs on (see [preparation](../preparation/CLAUDE.md)),
work dir `/share/fleurs-work` (venv, `emb/`, `speakers/`, `data/`). `unset LD_LIBRARY_PATH`
first, and uploads need a **write** `HF_TOKEN` — the one in the repo `.env` is read-only/stale.

The TitaNet vectors under `emb/` are the expensive artifact (~1.5h of 12-worker CPU for
279k utterances); re-clustering at another threshold reuses them and takes seconds.
