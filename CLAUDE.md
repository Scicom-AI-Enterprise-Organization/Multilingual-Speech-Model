# CLAUDE.md

Guidance for working in **Multilingual-TTS** — the training/data repo for the
`Scicom-intl/Multilingual-TTS-*` models (Qwen3 backbones continued-pretrained to emit NeuCodec
speech tokens `<|s_NNNN|>` at 50 tokens/s). The *serving* stack lives in a separate repo
(`TTS-API-Neucodec`, has its own CLAUDE.md).

## Repo map

| Path | What it is |
|---|---|
| `*.sh` (`1.7B.sh`, `0.6B-vc.sh`, `1.7B-expressive.sh`, …) | torchrun launch scripts per model/stage; pair with `qwen3_*.py` trainers (AdamW vs Muon+AdamW, WSD LR) |
| `qwen3_adamw.py`, `qwen3_muonadamw*.py` | trainers; `_post` = post-training variant |
| `preparation/` | multipacking notebooks: (text, speech-token) pairs → 10,240-token MDS blocks. Samples are **attention-isolated** (per-doc position_ids reset + length-based block-diagonal mask). Prompt: `<|im_start|>{speaker}: {text}<|speech_start|>{tokens}<|im_end|>` |
| `synthetic-description/` | expressive-TTS descriptions: acoustic stats + classifiers → bins → LLM summary (→ `Scicom-intl/ExpressiveSpeech`). Expressive prompt adds `<|description|>` |
| `nonverbal-tagging/` | non-verbal event mining (laughter/cough/…) → `*-Nonverbal-Tags` HF datasets. **Own CLAUDE.md with all pipeline + RunPod gotchas — read it before touching pods or HF bulk transfers** |
| `dnsmos/` | DNSMOS quality-filter pipeline (score → threshold → re-upload) |
| `tts-evaluation/`, `vc-evaluation/` | 76-language CER/MOS and speaker-similarity benchmarks vs Dia/Orpheus/Chatterbox/Fish/Qwen3-TTS |
| `vc-rl/grpo_async_vllm.py` | async GRPO trainer (7 DDP ranks + 1 dedicated vLLM rank, NCCL weight sync). **Reward is still `example_reward_fn` placeholder** — the intended reward is TitaNet similarity + (1−CER) + DNSMOS, reusable from `vc-evaluation/` + `dnsmos/` |
| `hyperparameter_search*.py` | LR search harness (see README ablation section) |

## Facts that shape decisions

- **VC speaker similarity is the weak metric** (0.505 vs Chatterbox 0.670) while CER is
  competitive — that's why vc-rl exists; wiring real rewards into GRPO is the known next step.
- Training data samples never attend across packed samples. Multi-turn continuation
  (streaming-coherence) and VC both rely on the *prompt-level* multi-turn format
  `...<|im_end|><|im_start|>...` — if adding continuation training data, pack contiguous
  same-recording segments as ONE document (single mask entry) instead.
- vLLM serving quirk: `--max-num-seqs` must stay low (~64); the ~217K speech-token vocab makes
  sampler warmup memory-heavy.
- Datasets are HF-hosted under `Scicom-intl/` (public); tokens/`.env` has `HF_TOKEN`,
  `RUNPOD_API_KEY`, `WANDB_API_KEY` — never commit or echo it.
- README ablations: AdamW beat Muon+AdamW at 1-epoch scale; hyperparameter search results and
  plots are in the root README.

## Working conventions

- GPU jobs run on RunPod pods; use `~/Documents/claude-ping` for persistent SSH
  (set `CLAUDE_PING_CONFIG` to a per-project JSON — its checked-in config belongs to another
  project). Health-check CUDA (`cuInit==0`) before trusting any community pod.
- On RunPod keep code/HF cache/venvs on local disk (`/root`), never `/workspace`
  (network volume). Set `HF_HOME=/root/hf`.
- For multi-GB HF transfers, `HF_HUB_DISABLE_XET=1` + retries (Xet CAS 401s intermittently).
- Upload results per shard/step, verify the HF tree afterwards, and only then delete pods.
