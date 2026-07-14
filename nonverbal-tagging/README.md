# Non-verbal Tagging

Mine non-verbal vocalization events (laughter, cough, sigh, sneeze, burping, humming, crying,
screaming) from Emilia-style segmented audio and produce transcripts with inline tags, for
expressive-TTS post-training.

## Output datasets

| Source | Tagged dataset |
|---|---|
| [Malaysian-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia) | [Malaysian-Emilia-Nonverbal-Tags](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia-Nonverbal-Tags) |
| [Malaysian-Tamil-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Tamil-Emilia) | [Malaysian-Tamil-Emilia-Nonverbal-Tags](https://huggingface.co/datasets/Scicom-intl/Malaysian-Tamil-Emilia-Nonverbal-Tags) |
| [Malaysian-Chinese-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Chinese-Emilia) | [Malaysian-Chinese-Emilia-Nonverbal-Tags](https://huggingface.co/datasets/Scicom-intl/Malaysian-Chinese-Emilia-Nonverbal-Tags) |

Each row carries two renderings of the same verified events:

- `tagged_text` — Higgs-TTS style: `<|sfx:laughter|>Haha` (script-matched onomatopoeia: 哈哈 for zh,
  ஹஹ for ta), discrete events only (speech overlap ≤ 50%)
- `nv_text` — Emilia-NV / NVSpeech style: bare `[Laughter]` word-level tokens for ALL events,
  including vocalizations overlapping speech

## Pipeline

```
audio segments (mp3, VAD-trimmed)
   │  mine_events.py     PANNs Cnn14_DecisionLevelMax framewise SED (AudioSet posteriors ~100 fps)
   ▼                     thresholds scaled to 0.3x (high recall) — see calibration note
candidate events (onset/offset/peak)          -> events/events-*.jsonl
   │  verify_and_tag.py  stage C: CLAP (laion/clap-htsat-unfused) zero-shot, class prompts vs
   ▼                     speech/music/noise negatives, keep fam-prob >= 0.55
verified events
   │  verify_and_tag.py  stage D: faster-whisper large-v3 word timestamps; insert tag at nearest
   ▼                     word gap; overlap > 50% -> metadata only (nv_text still tags it)
tagged transcripts                            -> data/tagged-*.parquet + qa_crops/ for auditing
```

**Calibration note** (`calibrate_expresso.py`, `diagnose_sed.py`): AudioSet laughter posteriors are
heavily suppressed on speech-adjacent laughter — on Expresso `laughing` clips the median clip-max is
0.007 (Speech class scores 0.5–0.8). Naive thresholds (~0.5) detect nothing; precision must come
from the CLAP gate, not the SED threshold.

## Usage (RunPod)

```bash
bash rent_healthy_pod.sh           # rents a US 4090/5090, health-checks cuInit, blacklists broken hosts
# scp/rsync this directory to /root/pipeline on the pod, then:
bash /root/pipeline/setup_pod.sh   # deps (pip --break-system-packages), PANNs ckpt, CLAP+whisper cache
umask 077; echo "HF_TOKEN=hf_..." > /root/.env_hf   # sourced by scale_generic.sh for downloads+uploads

# one shard-set, generic over any Emilia-style repo:
bash /root/pipeline/scale_generic.sh \
  Scicom-intl/Malaysian-Tamil-Emilia \
  Scicom-intl/Malaysian-Tamil-Emilia-Nonverbal-Tags \
  "/root/data/meta-tamil/audio_length_ratio_text/*.parquet" \
  audio_processed_trim-0-0.zip audio_processed_trim-1-0.zip ...
# resuming from shard K (keeps parquet numbering aligned): START_N=K + pass only the remaining zips
```

`scale_generic.sh` exports `HF_HUB_DISABLE_XET=1` (Xet CAS 401s intermittently on multi-GB files)
and retries downloads 3×. After any run, verify the HF `data/` listing matches the shard count —
upload failures are logged but non-fatal.

Each shard: download zip → unzip → mine → verify+tag → upload parquet/events/qa_crops to HF →
wipe audio. Progress is durable per shard. ~40–60 min per 10 GB shard on an RTX 4090
(mining is dataloader-bound; CLAP+whisper only run on event-bearing files).

## Results (July 2026 runs, all shards)

| Dataset | rows | verified events | laughter (nv / discrete) | cost |
|---|---|---|---|---|
| Malaysian-Emilia (10 shards, ~3,000 h) | 8,702 | 8,985 | 5,745 / 2,376 | ~$7 |
| Malaysian-Tamil-Emilia (7 shards) | 2,383 | 2,539 | 2,185 / 907 | ~$2.5 |
| Malaysian-Chinese-Emilia (21 shards) | 1,655 | 1,694 | 1,189 / 445 | ~$2.5 |
| **Combined seed** | **12,740** | **13,218** | **9,119 / 3,728** | **~$12** |

Tamil podcasts are ~4× more laugh-dense per audio-hour than Malay; Chinese sits between.

## Known limitations / next steps

See [plan.md](plan.md) for the full tag-aware-Whisper distillation plan (the volume path).

- Yield is precision-first and low-recall (~3 placed tags / 1k trimmed segments): the trimmed
  segments are post-VAD post-DNSMOS (which removes laughter), and PANNs misses most
  speech-overlapped events. The intended path to volume is NVSpeech-style distillation: fine-tune
  Whisper on these verified events to emit `[Label]` tokens inline, then pseudo-label raw
  (pre-trim) audio at scale.
- `burping` is polluted by mouth/plosive false-accepts — audit `qa_crops/` before training on it.
- `sigh`/`crying` recall is poor (weak AudioSet classes); needs dedicated detectors.
- `[Breathing]` and interjections are not mined; breathing is cheap via forced-alignment gaps.
