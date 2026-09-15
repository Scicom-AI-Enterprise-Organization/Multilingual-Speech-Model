# CLAUDE.md

Guidance for working in **Multilingual-Speech-Model** (renamed from Multilingual-TTS —
the work went beyond TTS) — the training/data repo for the `Scicom-intl/Multilingual-TTS-*`
models (Qwen3 backbones continued-pretrained to emit NeuCodec speech tokens `<|s_NNNN|>` at
50 tokens/s, and to read them back as text). The *serving* stack lives in a separate repo
(`TTS-API-Neucodec`, has its own CLAUDE.md).

`README.md` is the slim current index; the whole TTS-only V1 record — released checkpoints,
76-language benchmarks, ablations, dataset and training recipes — lives in `README_V1.md`.
Put new work in `README.md` or an area README, not back into V1.

## Repo map

| Path | What it is |
|---|---|
| `*.sh` (`1.7B.sh`, `0.6B-vc.sh`, `1.7B-expressive.sh`, …) | torchrun launch scripts per model/stage; pair with `qwen3_*.py` trainers (AdamW vs Muon+AdamW, WSD LR) |
| `qwen3_adamw.py`, `qwen3_muonadamw*.py` | trainers; `_post` = post-training variant |
| `dryrun_pack.py` | synthetic packs for all three tasks (real tokenizer ids, real audio files, random content) — smoke-tests the trainer end to end with no dataset. See README_V1 "TTS + STT + raw mel" |
| `mel_audio.py` + `qwen3_mel_adamw.py` | **raw-mel audio input** (no whisper encoder — only its mel front end): log-mel 100 fps → stack 2 frames → LayerNorm+MLP → the embeddings behind `<|mel|>` placeholders. Trains all three tasks in one run (TTS tokens / STT tokens / STT mel) via weighted `--train_file "dir:weight,…"`. Tests: `stt/test_mel_pipeline.py` |
| `preparation/` | multipacking: (text, speech-token) pairs → 10,240-token blocks. `multipacking.py` (VC pairs, all 6 datasets) writes **ChiniDataset parquet**; the TTS/expressive notebooks still write MDS. Samples are **attention-isolated** (per-doc position_ids reset + length-based block-diagonal mask). Prompt: `<|im_start|>{speaker}: {text}<|speech_start|>{tokens}<|im_end|>` (VC pairs omit `{speaker}: `). **Own CLAUDE.md — read it before touching path conventions or the remote box** |
| `synthetic-description/` | expressive-TTS descriptions: acoustic stats + classifiers → bins → LLM summary (→ `Scicom-intl/ExpressiveSpeech`). Expressive prompt adds `<|description|>` |
| `nonverbal-tagging/` | non-verbal event mining (laughter/cough/…) → `*-Nonverbal-Tags` HF datasets. **Own CLAUDE.md with all pipeline + RunPod gotchas — read it before touching pods or HF bulk transfers** |
| `stt/` | inverse-TTS (STT) data prep from `malaysia-ai/Multilingual-TTS-language`: `<|im_start|><|STT|>{speech tokens}<|{lang}|>{normalized text}<|im_end|>` blocks via `multipacking_stt.py`; ships the GlotLID detector + rule postnormalizer that produced the `language`/`post-normalized` columns. New `<|STT|>`/language tokens append AFTER the speech tokens — keep the order from `stt_added_tokens.json`. `multipacking_stt_mel.py` packs the **raw-mel** variant of the same task, storing audio *paths* (the dataset reads + resamples them on the fly; trainers need `--audio_dir`) |
| `dnsmos/` | DNSMOS quality-filter pipeline (score → threshold → re-upload) |
| `tts-evaluation/`, `vc-evaluation/` | 76-language CER/MOS and speaker-similarity benchmarks vs Dia/Orpheus/Chatterbox/Fish/Qwen3-TTS |
| `low-language-testset/` | long-tail test set: the 50 lowest-resource languages of `malaysia-ai/Multilingual-TTS`, 25 utterances each (audio + NeuCodec tokens + normalized text) → `malaysia-ai/Low-Language-TTS`. Reads remote parquet column-pruned and pulls zip members by range request — nothing is bulk-downloaded. **Own README — read it before changing the language-selection gates** |
| `vc-rl/grpo_async_vllm.py` | async GRPO trainer (7 DDP ranks + 1 dedicated vLLM rank, NCCL weight sync). **Reward is still `example_reward_fn` placeholder** — the intended reward is TitaNet similarity + (1−CER) + DNSMOS, reusable from `vc-evaluation/` + `dnsmos/` |
| `hyperparameter_search.py` + `qwen3_optimizer_search.py` | optimizer/LR sweep harness — adamw, muon, shampoo, soap, lion, ademamix (hybrid ones share Muon's 2D-hidden/AdamW split); resumes via `search_state/`, ranks by last-10-step loss (see README_V1 ablation section) |

## Facts that shape decisions

- **VC speaker similarity is the weak metric** (0.505 vs Chatterbox 0.670) while CER is
  competitive — that's why vc-rl exists; wiring real rewards into GRPO is the known next step.
- Training data samples never attend across packed samples. Multi-turn continuation
  (streaming-coherence) and VC both rely on the *prompt-level* multi-turn format
  `...<|im_end|><|im_start|>...` — if adding continuation training data, pack contiguous
  same-recording segments as ONE document (single mask entry) instead.
- Raw mel is **stacked, not pooled**, down to 50 positions/s (= the NeuCodec rate, so mel-STT and
  token-STT cost the same context for the same audio). A reshape of 2 frames is lossless and is
  exactly `Conv1d(128, H, kernel_size=2, stride=2)`; an average pool throws half the temporal
  detail away. Qwen2-Audio can pool only because its pool sits *after* 32 encoder layers that
  already made neighbouring frames redundant — there is no encoder here.
- Token order is now `<|speech_start|>` → 65,536 `<|s_N|>` → `<|STT|>` + languages → `<|mel_start|>
  <|mel|> <|mel_end|>`, appended and never reordered. The mel pack **reads** the language list from
  the token pack's `stt_added_tokens.json` and never recomputes it: a narrower `--subsets` selection
  would produce a shorter list and silently shift every id after it.
- The mel pack stores **paths only** (`audio` = json list, relative to `--audio_dir`, plus
  `audio_samples`); the dataset reads the files and resamples in the dataloader workers and the
  STFT runs on the GPU inside the model forward. So packing never decodes audio — `sf.info`
  headers only — but the extracted audio tree must be on the training box. `audio_samples` is
  authoritative: the decoded waveform is forced to it, or a resampler that rounds differently
  shifts every later utterance in the block onto the wrong placeholders.
- **`chinidataset.StreamingDataset` returns the *same dict object* for a repeated index**, so
  `data.pop(...)` pops off the cached row permanently. Harmless for the token-only trainers (they
  pop `audio`/`text` they never read), fatal the moment a trainer reads a column back: the mel
  trainer lost its audio on the second read of a block and would have trained on random `<|mel|>`
  embeddings. Copy first — `mel_audio.unpack_block()` does, and a test pins it. A local harness
  that does `dict(ds[idx])` passes while the trainer fails, which is how it hid.
- `ddp_find_unused_parameters false` + a micro-batch that drew no mel document stalls the
  all-reduce: the ranks that did draw one wait forever. `MelProjector.zero_probe()` keeps the
  projector in the graph on those steps — same fix as malaya's
  `session/audiollm/qwen_audio_stage2.py` (`dummy_audio`), minus the encoder, since `WhisperMel`
  has no parameters and the projector is the whole parameterised audio path.
- vLLM serving quirk: `--max-num-seqs` must stay low (~64); the ~217K speech-token vocab makes
  sampler warmup memory-heavy.
- Datasets are HF-hosted under `Scicom-intl/` (public); tokens/`.env` has `HF_TOKEN`,
  `RUNPOD_API_KEY`, `WANDB_API_KEY` — never commit or echo it.
- README_V1 ablations: AdamW beat Muon+AdamW at 1-epoch scale; hyperparameter search results
  and plots are in `README_V1.md` (the root README is now just an index).

## Working conventions

- GPU jobs run on RunPod pods; use `~/Documents/claude-ping` for persistent SSH
  (set `CLAUDE_PING_CONFIG` to a per-project JSON — its checked-in config belongs to another
  project). Health-check CUDA (`cuInit==0`) before trusting any community pod.
- On RunPod keep code/HF cache/venvs on local disk (`/root`), never `/workspace`
  (network volume). Set `HF_HOME=/root/hf`.
- For multi-GB HF transfers, `HF_HUB_DISABLE_XET=1` + retries (Xet CAS 401s intermittently).
- The mel pack needs `soundfile` + `soxr` and pulls `*_audio.zip` (not `*_neucodec.zip`) — that is the
  *audio* corpus, so always scope it with `--subsets` and check `df -h` first. The extracted tree
  has to stay: the trainer reads it every step, unlike the token packs which are self-contained.
- Upload results per shard/step, verify the HF tree afterwards, and only then delete pods.
