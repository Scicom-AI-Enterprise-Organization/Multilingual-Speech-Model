# DNSMOS Evaluation Pipeline

Quality filtering pipeline for the `malaysia-ai/Multilingual-TTS` dataset using 
[DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)

## Overview

The pipeline runs in two stages:

1. **Score** (`dnsmos_hf.py`) — Downloads audio zips from HuggingFace, runs 
ONNX inference, and writes scores to a JSONL file.
2. **Postprocess** (`postprocess.ipynb`) — Filters by score threshold, joins 
with metadata (text/speaker), and re-uploads to HuggingFace by subset.

## Files

| File | Description |
|------|-------------|
| `dnsmos_hf.py` | Multi-stage scoring pipeline for the HF dataset |
| `postprocess.ipynb` | Filtering, metadata join, and HF upload notebook |

## Setup
```bash
pip install onnxruntime librosa tqdm pandas huggingface_hub datasets
wget https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
```

## Usage

### 1. Score the HuggingFace dataset
```bash
python dnsmos_hf.py \
    --repo malaysia-ai/Multilingual-TTS \
    --sample 10
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--repo` | `malaysia-ai/Multilingual-TTS` | HF dataset repo |
| `--output` | `results_hf.jsonl` | Output JSONL file |
| `--base_dir` | `./` | Working directory for downloads |
| `--model` | `sig_bak_ovr.onnx` | Path to ONNX model |
| `--n_preprocess` | `1` | Stage-1 worker count (I/O bound) |
| `--n_compute` | `cpu_count - 4` | Stage-2 worker count (CPU bound) |
| `--zip_cache` | auto | File tracking completed zips (for resume) |
| `--sample` | all | Process only N randomly sampled zips |
| `--seed` | `42` | Random seed for `--sample` |

The pipeline is resumable — completed zips are tracked in a cache file and 
already-scored files are skipped on restart.

### 2. Score a local folder

```bash
python filter.py \
    --folder ./audio \
    --output results.jsonl \
    --n_compute 8
```

### 3. Postprocess and upload

Open `postprocess.ipynb` and run all cells. The notebook:

1. Loads `results_hf.jsonl`
2. Filters rows with `OVRL >= 3.2`
3. Joins each row with `text` and `speaker` from the original HF dataset
4. Uploads the filtered dataset to HuggingFace, one configuration per subset:

```python
ds_subset.push_to_hub(
    "org/dataset-name",
    config_name=subset_name,   # e.g. "700h-tr-turkish-text-to-speech"
    private=True,
)
```

## Output Schema

Each line in the JSONL output contains:

| Field | Description |
|-------|-------------|
| `filepath` | Relative audio file path |
| `len_in_sec` | Audio duration in seconds |
| `sr` | Sample rate (always 16000) |
| `num_hops` | Number of scored windows |
| `OVRL_raw` | Raw overall MOS score |
| `SIG_raw` | Raw signal MOS score |
| `BAK_raw` | Raw background MOS score |
| `OVRL` | Polyfit-calibrated overall score |
| `SIG` | Polyfit-calibrated signal score |
| `BAK` | Polyfit-calibrated background score |

After postprocessing, `text`, `speaker`, and `subset` columns are added and 
OVRL-filtered rows are kept.
