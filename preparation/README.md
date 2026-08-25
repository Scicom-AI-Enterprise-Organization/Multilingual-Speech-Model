# preparation

Packs (text, NeuCodec speech-token) pairs into ~10,240-token multipacked training blocks
for the `Scicom-intl/Multilingual-TTS-*` continued-pretraining runs.

## Voice-conversion multipacking — `multipacking.py`

One script replaces the old per-dataset notebooks (`multipacking-emilia-yodas.ipynb`,
`multipacking-malaysian-*.ipynb`). It writes
[ChiniDataset](https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset) parquet
shards instead of mosaicml-streaming MDS.

### Datasets

| name | source (config) | reject filter | upload target |
|---|---|---|---|
| `malaysian-tamil-emilia` | [Malaysian-Tamil-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Tamil-Emilia) (`permutation_sample`) | `audio_length_ratio_text` | `Scicom-intl/Malaysian-Tamil-Emilia-multipacking-10k` |
| `malaysian-chinese-emilia` | [Malaysian-Chinese-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Chinese-Emilia) (`speaker_permutation_sample`) | `audio_length_ratio_text` | `Scicom-intl/Malaysian-Chinese-Emilia-multipacking-10k` |
| `malaysian-emilia-dialects` | [Malaysian-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia) (`dialects_v1_permutation_sample`) | `dialects_v1_audio_length_ratio_text` | `Scicom-intl/Malaysian-Emilia-dialects-multipacking-10k` |
| `malaysian-emilia` | [Malaysian-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia) (default; `malaysian-chinese*` rows skipped) | `audio_length_ratio_text` | `Scicom-intl/Malaysian-Emilia-multipacking-10k` |
| `youtube-cantonese-emilia` | [YouTube-Cantonese-Emilia](https://huggingface.co/datasets/Scicom-intl/YouTube-Cantonese-Emilia) (`permutation_sample`) | — (repo has none) | `Scicom-intl/YouTube-Cantonese-Emilia-multipacking-10k` |
| `emilia-yodas` | [Emilia-YODAS-Voice-Conversion](https://huggingface.co/datasets/Scicom-intl/Emilia-YODAS-Voice-Conversion) (default) | `audio_length_ratio_text` | `Scicom-intl/Emilia-YODAS-multipacking-10k` |

### What it does

Per dataset:

1. **Download + extract** the `*_neucodec.zip` files from the HF repo (NeuCodec token
   JSONs, one per trimmed audio segment). Extraction is marker-tracked
   (`neucodec/.extracted/<zip>.done`) and zips are deleted afterwards unless
   `--keep-zips`, so re-runs skip finished work.
2. **Load** the (reference, target) permutation pairs and the reject list
   (`audio_length_ratio_text_accept == False` → drop).
3. **Pack**: each pair becomes one document
   `<|im_start|>{ref_text}<|speech_start|>{ref_tokens}<|im_end|><|im_start|>{tgt_text}<|speech_start|>{tgt_tokens}<|im_end|>`
   tokenized with `Qwen/Qwen3-1.7B-Base` + 65,537 added speech tokens. Documents are
   greedily packed into ~10,240-token blocks. N worker processes each write their own
   `ParquetWriter` sub-folder; `merge_index()` then unifies them into one dataset.

Pairs that fail `len(text.split()) > len(speech_tokens)` (bad alignment) or whose
NeuCodec JSON is missing are dropped, and per-dataset drop counts are reported in
`out/<name>/summary.json`.

### Output format

Each sample is one training block, attention-isolated per document:

| column | type | |
|---|---|---|
| `input_ids` | `uint32[]` | ≤10,240 packed token ids |
| `position_ids` | `uint32[]` | reset to 0 at each document boundary |
| `attention_mask` | `uint32[]` | per-document lengths — trainers expand to a block-diagonal mask |
| `audio`, `text` | `str` | empty; kept for schema parity with the old MDS datasets |

Read it back with:

```python
from chinidataset import StreamingDataset
ds = StreamingDataset(local='out/multipacking-emilia-yodas')
len(ds), ds[0]
```

> Note: the `qwen3_*.py` trainers read this format via
> `chinidataset.StreamingDataset`. The pre-2026-08 `*-multipacking-10k` HF repos
> are still in the old mosaicml MDS format and need `streaming.LocalDataset`.

### Usage

```bash
python multipacking.py all --base-dir /share/multipacking --workers 96
python multipacking.py malaysian-tamil-emilia youtube-cantonese-emilia
python multipacking.py all --stage download    # only fetch/extract zips
python multipacking.py all --stage pack        # zips already extracted
python multipacking.py all --upload            # push results to HF (private)
```

Run it on a big-CPU box — tokenizing with 65k added tokens is slow, so it scales with
cores (the reference runs used 64–96 workers). Set `HF_HOME` somewhere with space; the
script sets `HF_HUB_DISABLE_XET=1` itself.

## TTS / expressive multipacking (still notebooks)

- `multipacking-tts.ipynb`, `combine-multipacking-tts.ipynb` — single-utterance TTS
  format `<|im_start|>{speaker}: {text}<|speech_start|>...`
- `multipacking-expressivetts.ipynb`, `combine-multipacking-expressive.ipynb` —
  expressive format with `<|description|>`

These still write mosaicml-streaming MDS and have not been ported to ChiniDataset yet.
