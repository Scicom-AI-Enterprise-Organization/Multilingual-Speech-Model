# Plan: tag-aware Whisper (NVSpeech-style distillation)

Turn the ~13k-event mined seed into a paralinguistic-aware ASR, then pseudo-label raw audio at
scale. Target: 100k+ tagged utterances for the pretraining-data scale-up.

Status: **planned** (seed datasets complete, see README results table).

## Step 1 — Build the fine-tune dataset

Target format = the `nv_text` column: transcript with inline `[Laughter]` / `[Cough]` / … tokens.

| Ingredient | Source | Purpose |
|---|---|---|
| Positives (~12.7k utts) | `*-Nonverbal-Tags` `data/*.parquet` (`nv_text`) | the mined seed |
| Plain negatives (large sample) | event-free segments from the same corpora (~1.5M available) | most audio has zero tags — prevents over-tagging |
| **Hard negatives (~7k)** | SED candidates that **CLAP rejected** (`events/*.jsonl` minus verified) | acoustically-confusable moments trained as no-tag — strongest anti-hallucination signal |
| Synthetic mixes | VocalSound (21k real laughs/coughs/sneezes/sighs, ~3k speakers) overlaid on clean segments at known offsets | unlimited positives with exact placement; patches weak classes (sigh, crying) |

Pre-work:

- **Audit or drop `burping`** before training — false burps in the seed teach Whisper to
  hallucinate burps at scale (worse than losing the class). Listen to `qa_crops/`.
- Mix ms / en / ta / zh to match deployment.
- Hold out one shard per dataset for eval.

Seed-size note: NVSpeech's human seed was ~48k utterances. Our 12.7k alone is thin; VocalSound
synthesis + hard negatives close the gap for a v1.

## Step 2 — Fine-tune whisper-large-v3-turbo

Script: [`whisper_finetune.py`](whisper_finetune.py) (transformers Seq2SeqTrainer; consumes jsonl
manifests `{"audio_path", "text", "language"}` where `text` = nv_text-style target; per-sample
language prefixes for the ms/en/ta/zh mix; `--lora` for the 24 GB path; reports `tag_f1` /
`tag_precision` / `tag_recall` / `cer_notags`, with a pre-training baseline eval for the
regression check).

- 809M params — cheap to fine-tune, and fast enough (CT2) to pseudo-label thousands of hours later.
- Tags emitted as plain text tokens in the transcript (no vocab surgery for v1).
- Start with **LoRA on a 4090 (~$5)**; escalate to full FT on an H100 only if F1 disappoints.
- Success gates:
  1. tag F1 + placement accuracy on held-out events
  2. **no CER regression** on untagged speech (`cer_notags` vs the baseline eval)

## Step 3 — Pseudo-label at scale

- Convert to CT2 / faster-whisper.
- Run over **raw pre-trim audio** (`audio_processed-*.zip` in Malaysian-Tamil-Emilia, the
  Malaysian-Emilia sources) + Emilia-YODAS — discrete laughs live between/outside the VAD-trimmed
  segments the seed was mined from.
- Precision filter: keep tags where Whisper and PANNs agree.
- Output feeds the pretraining mix (with tag dropout, per the Higgs/NVSpeech recipe) and
  expressive post-training.

## Cheap parallel wins (independent of steps 1–3)

- `thresh-scale 0.1` re-mine of the existing shards (~$12): likely 2–4× more seed.
- `[Breathing]` / `[Pause]` tags from forced-alignment gaps — no model needed, immediately
  usable in TTS data.

## Sequencing

1. Dataset builder (develop + validate locally, no GPU)
2. LoRA fine-tune + eval gates (4090)
3. CT2 conversion + pseudo-label sweep (4090/L40S, disk-heavy)
4. Merge into pretraining-data scale-up
