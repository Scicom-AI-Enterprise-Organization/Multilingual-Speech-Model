# CLAUDE.md — preparation/

Multipacking pipelines: (text, NeuCodec token) pairs → ~10,240-token training blocks.
`multipacking.py` (voice-conversion pairs, ChiniDataset parquet) is the maintained tool;
the `multipacking-tts.ipynb` / `multipacking-expressivetts.ipynb` notebooks are the
TTS/expressive variants and still write mosaicml MDS.

## Facts that shape decisions

- **Format split**: `multipacking.py` writes ChiniDataset parquet, and the `qwen3_*.py`
  trainers read it via `chinidataset.StreamingDataset` (ported 2026-08-25). HF repos
  created before that date under `Scicom-intl/*-multipacking-10k` are mosaicml MDS —
  those need `streaming.LocalDataset`. Don't mix readers, and don't upload parquet
  shards into an existing MDS repo.
- **NeuCodec JSON path conventions** (audio path in the pair configs → JSON on disk):
  - Emilia-style repos: `folder/rest.mp3` → `folder_trim_neucodec/rest.json`
    (zips named `*_trim_neucodec*.zip`, arcnames already carry the folder).
  - **YouTube-Cantonese is different**: `shard/id/id_N.mp3` → `shard_neucodec/id/id_N.json`
    — no `_trim` in the folder name even though the tokens came from trimmed audio
    (`path_style='neucodec'` in the spec). It also has **no**
    `audio_length_ratio_text` reject config.
- Every pair row is dropped if either side is in the reject list, its JSON is missing,
  or `len(text.split()) > len(speech_tokens)`. Drop counts land in
  `out/<name>/summary.json` — sanity-check `missing` is near zero; a huge `missing`
  count means a path-convention mismatch, not missing data.
- Packing keeps documents attention-isolated: `position_ids` reset per document,
  `attention_mask` = per-document lengths (block-diagonal mask reconstructed at
  training time). A single document longer than 10,240 tokens is emitted as its own
  oversized block (same behavior as the old notebooks) — trainers truncate.
- The tokenizer is `Qwen/Qwen3-1.7B-Base` + 65,537 `AddedToken`s; building it takes
  minutes and tokenization is the throughput bottleneck. It's built **once in the
  parent** and inherited by fork — workers must not rebuild it, and nothing may *use*
  a fast tokenizer in the parent before the Pool forks (rust tokenizer + fork deadlock).
- Workers share `rows`/`tokenizer`/`reject` via the module-global `G` + fork COW —
  don't refactor into `pool.map` args (pickling multi-GB row lists is what made the
  old notebooks slow to start).

## Running it (reference box)

- Box: `ssh to the shared GPU box (host/port in the claude-ping config)` (key `scicom` at repo root,
  gitignored — never commit it). 164 cores / 1.6TB RAM; disk is >90% full, check
  `df -h /share` before a run (a full `all` run peaks around ~330GB: 262GB packed
  parquet + ~50GB extracted JSONs + transient zips). If space runs short mid-run,
  it's safe to `rm -rf` the `neucodec/<folder>` trees of datasets that already
  finished packing — but then also remove their `neucodec/.extracted/*.done`
  markers, or a later re-run will think the JSONs are still on disk.
- Work dir `/share/multipacking/`: `venv/` (uv, CPU torch + chinidataset),
  `zips/` (deleted after extract), `neucodec/` (extracted JSONs, millions of small
  files), `out/<dataset>/`, `hf/` (HF_HOME), `run.log`.
- **`unset LD_LIBRARY_PATH` before running python there.** Login shells (tmux/`claude-ping run`)
  export the DSW image's `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:…`
  (system torch 2.9 NVIDIA build), which poisons the venv's torch with mismatched
  `libtorch` symbols (`torch._C has no attribute '_dlpack_exchange_api'`). Non-login
  `claude-ping exec` doesn't set it — so imports "work" in exec and then crash in tmux.
- Drive it with `~/Documents/claude-ping` (set `CLAUDE_PING_CONFIG` to a per-project
  JSON): `claude-ping run --session <name> "cd /share/multipacking && unset LD_LIBRARY_PATH PYTHONPATH && HF_HOME=/share/multipacking/hf venv/bin/python multipacking.py all --workers 96"`,
  then `claude-ping watch --session <name> --interval 120s` in the background.
  `scp` is not supported by that box — use `claude-ping sync`, and keep
  `venv,zips,neucodec,out,hf` in `sync_excludes` or rsync `--delete` wipes them.
- Uploads (`--upload`) create **private** `*-multipacking-10k` repos via
  `upload_large_folder`; needs `HF_TOKEN` in the env (claude-ping `env-sync`).
  Verify the HF tree before deleting anything on the box.
