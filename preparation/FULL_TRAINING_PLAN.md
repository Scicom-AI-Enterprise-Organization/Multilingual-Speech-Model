# Full training data — build plan

Target: one model that does TTS, STT from speech tokens, and STT from raw log-mel. One
mixture, one vocabulary, three tasks.

## Sources

| task | source | what it gives |
|---|---|---|
| TTS tokens | 6 × `Scicom-intl/*-multipacking-10k` | 37.34 B tokens, already packed |
| TTS tokens | [`malaysia-ai/Multilingual-TTS`](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS) | 121.8 M rows / 1,552 subsets → ~36 B tokens |
| TTS tokens | `Scicom-intl/*-Nonverbal-Tags` × 3 | 12,740 rows with inline `<\|sfx:laughter\|>` tags |
| STT tokens | [`malaysia-ai/Multilingual-TTS-language`](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS-language) | same rows + GlotLID `language` + `post-normalized` → ~40 B tokens |
| STT mel | the same corpus, **raw audio** | 4.438 TB of audio zips; budgeted subset → ~14 B tokens |
| all three | [`malaysia-ai/fleurs-r-neucodec-all-languages`](https://huggingface.co/datasets/malaysia-ai/fleurs-r-neucodec-all-languages) | 102 locales, restored audio, derived speaker labels |

Sizes measured from the Hub, not estimated:

```
malaysia-ai/Multilingual-TTS            4.637 TB
  audio zips          2,100 files       4.438 TB    mel arm only
  neucodec token zips 1,531 files         190 GB    TTS-token AND STT-token read these
  metadata parquet    1,561 files          10 GB
malaysia-ai/Multilingual-TTS-language                 12 GB
```

**One extraction serves two packs.** The TTS-token and STT-token packs read the same token
JSONs; only the metadata differs (`Multilingual-TTS` has `speaker`, `-language` has
`language` + `post-normalized`). Extract a wave once, pack both, delete, repeat.

## What is and is not already covered

- **Common Voice 22 rows are in the corpus metadata, but its tokens are not.** The rows
  point at `audio_trim/...`, and `Multilingual-TTS` holds **zero** `audio_trim*` zips —
  they live in `malaysia-ai/common_voice_22_0`. The corpus packer will skip those 6.9 M
  rows for want of a zip, so the existing `cv22-*` packs are not duplicates; they fill
  exactly that hole. They do need re-packing against the unified vocabulary (§ vocabulary).
- **FLEURS-R is not in the corpus.** The corpus has 21 `fleurs*` subsets, but those are
  third-party derivatives (per-language ASR slices). `fleurs-r-neucodec-all-languages` is
  the restored-audio release with our own speaker clusters. Some transcript overlap is
  possible; the audio and tokens differ.
- The 6 voice-conversion packs need **no** re-packing: they contain speech tokens and text
  only, so appending language tags leaves their ids untouched.

## Vocabulary

Language tags and mel tokens are appended after the 65,537 speech tokens. Two packs built
against different lists disagree on every id after `<\|s_65535\|>`, including `<\|mel\|>`.

**One run, one list.** `preparation/build_vocab.py` unions

```
FLEURS locales (102)  ∪  CV22 languages (130)  ∪  GlotLID labels (~1.5 k subsets)
→  <|STT|> + <|tag|> … + <|mel_start|> <|mel|> <|mel_end|>
```

Everything packed after this point takes `--added-tokens-file`. FLEURS and CV22 are
re-packed against it; the VC packs are unaffected.

## Stages

| # | stage | output | time | disk |
|---|---|---|---:|---:|
| 0a | download the 6 VC packs | 37.3 B tok | ~1.5 h | 262 GB |
| 0b | build the unified vocabulary | ~1.6 k tags | ~30 m | — |
| 1 | token zips → extract, in waves | 121.8 M JSONs | ~3 h | ~200 GB/wave |
| 2 | pack TTS-token | ~36 B tok | ~4 h | 253 GB |
| 3 | pack STT-token | ~40 B tok | ~4 h | 280 GB |
| 4 | pack non-verbal tags | ~4 M tok | ~20 m | < 1 GB |
| 5 | re-pack FLEURS (+ CV22) | 3.9 B tok | ~20 m | 46 GB |
| 6 | mel audio, budgeted subset | — | ~8 h | **2 TB** |
| 7 | pack STT-mel | ~14 B tok | ~3 h | 100 GB |

Stages 2 and 3 run per wave off stage 1's extraction.

## Mel budget: 2 TB

Mel packs store audio *paths*, so their audio stays resident for the whole run. At the
corpus's 16.4 GB per 1,000 h:

```
2 TB  ≈ 122,000 h  ≈ 45% of the corpus  ≈ 14 B mel tokens
```

Selection takes a share of each subset's shards rather than whole subsets, so language
coverage is preserved instead of spending the budget on the few largest corpora. Local
disk, not NFS: local reads measured 4,167 files/s against the NAS's 1,492, and the
dataloader hits one small file per utterance.

## Resulting mixture

```
TTS tokens   37.3 B  VC packs
           + ~36 B   Multilingual-TTS
           + 0.2 B   FLEURS + CV22 + non-verbal tags
STT tokens   ~40 B   Multilingual-TTS-language
             + 1.8 B  FLEURS + CV22
STT mel      ~14 B   budgeted subset
             + 2.0 B  FLEURS + CV22
                      ────────
                      ~131 B tokens per epoch
```

Weights are a launch flag (`--train_file "dir:weight,…"`), so ratios change without
re-packing.

## Risks

- **Inodes.** 121.8 M token JSONs against ~152 M free. Waves with deletion after each
  pack; never extract the whole corpus at once.
- **Subsets without token zips.** The packer skips rows whose `{folder}_neucodec.zip` is
  absent (CV22 is one). The skip count per wave goes in the summary so the real coverage
  is known.
- **Disk.** Projected ~1.5 TB of packs and JSONs plus 2 TB of mel audio, against 3.2 TB
  free. Wave deletion keeps the transient peak under control.

## Not data — needed before training starts

1. `qwen3_mel_adamw.py` is AdamW-only. Port `build_optimizer` so it can run
   `--optimizer muon --matrix_lr 1e-2`, which beat AdamW by 3.2 nats in the ablation.
2. Re-enable checkpointing (the sweep ran `--save_strategy no`).
3. Set `num_decay_steps` to ~10% of the real step count; in the 200-step sweep the WSD
   schedule never entered decay.
