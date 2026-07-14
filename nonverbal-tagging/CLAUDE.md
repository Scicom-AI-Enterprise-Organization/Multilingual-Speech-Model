# CLAUDE.md — nonverbal-tagging

Guidance for working on the **non-verbal tagging pipeline** (laughter/cough/sigh/… events mined
from Emilia-style audio into inline-tagged transcripts for expressive TTS). Read `README.md` for
the pipeline overview and `plan.md` for the next phase (tag-aware Whisper distillation); this
file is the operational knowledge.

## Architecture in one line

`mine_events.py` (PANNs framewise SED, recall) → `verify_and_tag.py` (CLAP zero-shot gate,
precision; then faster-whisper word timestamps, placement) → per-shard parquet upload to HF.

## Numbers that drive every design decision

- **AudioSet posteriors on speech-laugh are ~0.007 (median clip-max)** — measured on Expresso
  `laughing` via `calibrate_expresso.py`. Never raise SED thresholds back toward 0.5 "to improve
  precision"; precision belongs to the CLAP stage. Mining runs at `--thresh-scale 0.3`.
- CLAP gate passes ~50% of candidates; median accepted confidence ≥ 0.95.
- Yield on quality-trimmed segments ≈ 3 placed tags / 1k segments. This is a **seed dataset**;
  volume comes from the planned NVSpeech-style step: fine-tune Whisper on these events to emit
  `[Label]` tokens, pseudo-label raw (pre-trim) audio.
- Completed runs (July 2026): Malay 8,985 events / Tamil 2,539 / Chinese 1,694 → combined 13,218
  verified (9,119 laughter) across the three `*-Nonverbal-Tags` HF repos. Tamil is ~4× more
  laugh-dense per hour than Malay.

## Output schema (`data/tagged-*.parquet`)

`file, orig_text, whisper_text, tagged_text, nv_text, language, events(JSON), n_placed`

- `tagged_text`: `<|sfx:family|>` + onomatopoeia (script-matched by whisper language: zh 哈哈,
  ta ஹஹ, else Latin). Only events with speech-overlap ≤ 50%.
- `nv_text`: Emilia-NV style bare `[Laughter]` tokens, ALL events including overlapped.
- `events[*]`: family, onset, offset, peak (SED), clap_fam/clap_neg, overlap_frac, placed.
- `events/events-*.jsonl` in the HF repos = pre-CLAP candidates → re-threshold offline without
  re-running GPU inference.

## Gotchas (every one of these bit us)

- **transformers ≥ 5**: `ClapModel.get_text/audio_features` return `BaseModelOutputWithPooling`
  → take `.pooler_output`; processor kwarg is `audio=` (`audios=` raises ValueError).
- **datasets ≥ 4** requires torchcodec for audio decode → use `.cast_column("audio",
  Audio(decode=False))` + soundfile on the bytes.
- **PANNs SED checkpoint** comes from zenodo only (no HF mirror; slow ~300 MB). Cached at
  `/root/models/Cnn14_DecisionLevelMax.pth` by `setup_pod.sh`.
- **RunPod image (Ubuntu 24.04)**: `pip install --break-system-packages`; torch is a system pkg.
- **HF Xet downloads 401 intermittently on multi-GB files** (CAS reconstruction), even when
  `hf auth whoami` says logged in and the repo is public; small files pass. Fix: export
  `HF_TOKEN` explicitly + `HF_HUB_DISABLE_XET=1` (plain CDN path) + retry ×3 — all baked into
  `scale_generic.sh` (sources `/root/.env_hf`). Keep `HF_HOME=/root/hf` on the local disk,
  never `/workspace` (network volume on RunPod).
- **Health-check every pod**: `CDLL("libcuda.so.1").cuInit(0)` must return 0 — nvidia-smi works on
  broken hosts (seen: community 5090 at 216.249.100.66, cuInit=999). `rent_healthy_pod.sh` does
  rent → check → blacklist-and-retry automatically.
- **`pkill -f` over SSH self-matches** the ssh command string → use `pgrep -f "patter[n]"`.
- Two CUDA processes on one pod can abort (`cuInit` conflict) — run calibration on CPU or after
  mining finishes.
- Local ssh wrappers launched with `nohup ... &` get reaped by the harness — the **remote**
  process survives; always verify with `pgrep` on the pod, not by the local exit status.
- Use `claude-ping` (`~/Documents/claude-ping`, config via `CLAUDE_PING_CONFIG` — do NOT edit its
  checked-in claude-ping.json, that belongs to neucodec-44k) for persistent SSH to pods.

## HF layout & viewer

Output repos need explicit `configs:` in README frontmatter (default → `data/*.parquet`,
events → `events/*.jsonl`) or the mixed parquet/jsonl/wav layout breaks the dataset viewer.
Upload per shard, never batch at the end — pods are ephemeral.

- **`hf upload` failures are non-fatal in the loop** — a shard can log DONE with nothing on HF
  (happened to Tamil shard 2 during an HF outage). After every run, diff the HF `data/` tree
  against the expected shard count before deleting the pod.
- Output dirs are namespaced per output repo (`/root/out/<repo>-<n>`); with the old shared
  `gshardN` naming, dataset B's failed shard `rm -rf`'d dataset A's not-yet-uploaded output.
- Resume/partial runs: `START_N=<k>` keeps parquet numbering aligned when passing a zip subset.

## Class-quality caveats

- `burping` ≈ mouth/plosive false-accepts sneaking past CLAP — audit `qa_crops/` before training.
- `sigh`/`crying`: weak AudioSet classes, low recall — needs dedicated detectors.
- `[Breathing]` + interjections not mined yet; breathing is cheap via alignment gaps.

## Cost/pace reference (RTX 4090, $0.69/hr)

~10 GB zip shard (~128k segments ≈ 300 h): mine ~35 min (dataloader-bound, GPU ~60%),
CLAP+whisper ~15 min (event files only), total ~40–60 min ≈ $0.60/shard.
