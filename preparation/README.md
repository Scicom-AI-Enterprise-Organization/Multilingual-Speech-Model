# preparation

Packs (text, NeuCodec speech-token) pairs into ~10,240-token multipacked training blocks
for the `Scicom-intl/Multilingual-TTS-*` continued-pretraining runs.

## Voice-conversion multipacking — `multipacking.py`

One script replaces the old per-dataset notebooks (`multipacking-emilia-yodas.ipynb`,
`multipacking-malaysian-*.ipynb`). It writes
[ChiniDataset](https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset) parquet
shards instead of mosaicml-streaming MDS.

### Datasets

Sizes are from the 2026-08-24 run (blocks × 10,240 tokens; greedy packing, so true
counts sit slightly below):

| name | source (config) | reject filter | blocks | tokens | upload target |
|---|---|---|---|---|---|
| `malaysian-tamil-emilia` | [Malaysian-Tamil-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Tamil-Emilia) (`permutation_sample`) | `audio_length_ratio_text` | 81,723 | ~0.84B | `Scicom-intl/Malaysian-Tamil-Emilia-multipacking-10k` |
| `malaysian-chinese-emilia` | [Malaysian-Chinese-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Chinese-Emilia) (`speaker_permutation_sample`) | `audio_length_ratio_text` | 178,271 | ~1.83B | `Scicom-intl/Malaysian-Chinese-Emilia-multipacking-10k` |
| `malaysian-emilia-dialects` | [Malaysian-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia) (`dialects_v1_permutation_sample`) | `dialects_v1_audio_length_ratio_text` | 565,604 | ~5.79B | `Scicom-intl/Malaysian-Emilia-dialects-multipacking-10k` |
| `malaysian-emilia` | [Malaysian-Emilia](https://huggingface.co/datasets/Scicom-intl/Malaysian-Emilia) (default; `malaysian-chinese*` rows skipped) | `audio_length_ratio_text` | 775,589 | ~7.94B | `Scicom-intl/Malaysian-Emilia-multipacking-10k` |
| `youtube-cantonese-emilia` | [YouTube-Cantonese-Emilia](https://huggingface.co/datasets/Scicom-intl/YouTube-Cantonese-Emilia) (`permutation_sample`) | — (repo has none) | 142,711 | ~1.46B | `Scicom-intl/YouTube-Cantonese-Emilia-multipacking-10k` |
| `emilia-yodas` | [Emilia-YODAS-Voice-Conversion](https://huggingface.co/datasets/Scicom-intl/Emilia-YODAS-Voice-Conversion) (default) | `audio_length_ratio_text` | 1,902,702 | ~19.48B | `Scicom-intl/Emilia-YODAS-multipacking-10k` |
| **total** | | | **3,646,600** | **~37.3B** | |

#### Two packs share one repo — and do not overlap

`malaysian-emilia` and `malaysian-emilia-dialects` both read `Scicom-intl/Malaysian-Emilia`,
but they are **disjoint configs over disjoint audio folders** (full column census, not a
sample):

| pack | config | pair rows | audio folders |
|---|---|---:|---|
| `malaysian-emilia` | `default` | 8,664,602 | `parlimen-24k-chunk_processed` 4,075,911 · `malaysian-chinese_processed` 2,074,651 · `malaysian-podcast_processed` 1,438,798 · `sg-podcast_processed` 900,733 · `cartoon-24k_processed` 143,656 · `klasik_processed` 30,853 |
| `malaysian-emilia-dialects` | `dialects_v1_permutation_sample` | 4,963,410 | `dialects_processed` 4,963,410 |

`default` holds **no** `dialects_processed` rows, and every pair's reference and target sit
in the same folder, so neither pack can contain the other's audio.

The one real overlap is `malaysian-chinese_processed`: it is 2.07M rows of `default` **and**
the whole of the separate `malaysian-chinese-emilia` pack (same folder, same audio, from
`Scicom-intl/Malaysian-Chinese-Emilia`). That is what `skip_prefix='malaysian-chinese'`
drops — and the block count shows it took effect: 6.59M pairs after the skip → 775,589
blocks ≈ 8.5 pairs/block, in line with the dialects pack's 8.8; packing all 8.66M rows
would have produced ~985K blocks.

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

## FLEURS-R TTS/STT ablation packs — `multipacking_fleurs.py`

Builds the two packs the optimizer ablation compares (README "Optimizer search") from
[malaysia-ai/fleurs-r-neucodec-all-languages](https://huggingface.co/datasets/malaysia-ai/fleurs-r-neucodec-all-languages)
— FLEURS-R metadata plus precomputed NeuCodec tokens for 102 locales, small enough
(~640MB of token zips) to re-pack in minutes.

One pass over the tokens writes both directions of the task, so the two sweeps differ
only in what the model has to predict:

| pack | document | train blocks | dev blocks |
|---|---|---|---|
| `out/fleurs-tts` | `<\|im_start\|>{speaker}: {text}<\|speech_start\|>{speech tokens}<\|im_end\|>` | 18,047 | 2,237 |
| `out/fleurs-stt` | `<\|im_start\|><\|STT\|>{speech tokens}<\|{locale}\|>{text}<\|im_end\|>` | 17,990 | 2,213 |
| `out/fleurs-mel` | `<\|im_start\|><\|STT\|><\|mel_start\|>{P × <\|mel\|>}<\|mel_end\|><\|{locale}\|>{text}<\|im_end\|>` | — | — |

From the 2026-09-13 run: 248,117 train documents over 102 locales (0 missing token
files, 1 row dropped on `len(text.split()) > len(speech_tokens)`) and 31,378 dev
documents (0 missing). `--splits dev` writes `-dev` packs beside the train ones, which
is what the sweep validates on.

The mel pack is the STT task again with the audio carried as raw whisper log-mel instead
of codec tokens ([../mel_audio.py](../mel_audio.py)). Both representations run at 50
positions/s, so an utterance costs the same context either way. Mel blocks store audio
**paths** plus the 16kHz sample count each `<|mel|>` placeholder budget was derived from
— run `--stage audio` first to materialise the wavs (~126GB for all 102 locales,
marker-tracked per zip), and hand the trainer the same root as `--audio_dir`.

- The speaker slot takes the repo's `speaker` column — the TitaNet voice clusters added by
  [../fleurs-dataset](../fleurs-dataset) — falling back to the locale when absent. The
  locale is also the STT language tag: it is ground truth, so unlike [../stt](../stt) no
  GlotLID pass is needed. **The 2026-09-13 packs above predate the speaker column and carry
  `{locale}: `**; re-packing changes the TTS side only.
- Text is `normalized_text` (falling back to `sentence`) on both sides.
- `<\|STT\|>` + one token per locale are appended **after** the 65,537 speech tokens and
  listed in `out/fleurs_stt_added_tokens.json`; the mel tokens go after those again, in
  `out/fleurs_mel_added_tokens.json`. Each trainer takes the file matching its pack via
  `--added_tokens_file`, so `<|s_N|>` ids are identical in all three arms.
- Block format, greedy packing and attention isolation are identical to
  `multipacking.py` above.

```bash
python multipacking_fleurs.py --base-dir <base> --task token --workers 96   # tts + stt
python multipacking_fleurs.py --base-dir <base> --splits dev --task token   # the dev packs
python multipacking_fleurs.py --base-dir <base> --stage audio --audio-base <audio>
python multipacking_fleurs.py --base-dir <base> --task mel --audio-base <audio>
python multipacking_fleurs.py --base-dir <base> --task stt --locales 'en_us' 'ms_my'
```

## TTS / expressive multipacking (still notebooks)

- `multipacking-tts.ipynb`, `combine-multipacking-tts.ipynb` — single-utterance TTS
  format `<|im_start|>{speaker}: {text}<|speech_start|>...`
- `multipacking-expressivetts.ipynb`, `combine-multipacking-expressive.ipynb` —
  expressive format with `<|description|>`

These still write mosaicml-streaming MDS and have not been ported to ChiniDataset yet.
