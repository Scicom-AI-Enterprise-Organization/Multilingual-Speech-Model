# stt

Inverse-TTS (speech-to-text) data preparation from
[malaysia-ai/Multilingual-TTS-language](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS-language)
(118M rows / 1493 subsets: `audio_filename, text, speaker, language, post-normalized`).

## Document format

One document per audio segment, language predicted *after* the audio:

```
<|im_start|><|STT|>{<|s_N|> speech tokens}<|{language}|>{normalized transcription}<|im_end|>
```

- speech tokens: the same `<subset>_neucodec/<file>.json` NeuCodec tokens the TTS
  multipacking uses (`*_neucodec.zip` in
  [malaysia-ai/Multilingual-TTS](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS))
- `<|{language}|>`: the GlotLID v3 label as a token, e.g. `<|zsm_Latn|>`, `<|yor_Latn|>`
- transcription: the `post-normalized` column (normalized on the fly for subsets
  that predate it)

New special tokens (`<|STT|>` + one per language) are appended **after**
`<|speech_start|>` and the 65,536 `<|s_N|>` tokens so speech-token IDs stay aligned
with the TTS trainers; the exact appended list lands in
`<out>/stt_added_tokens.json` and trainers must add the same tokens in the same order.

## Pipeline — `multipacking_stt.py`

Same block format as [preparation/multipacking.py](../preparation/multipacking.py):
ChiniDataset parquet, ~10,240-token attention-isolated blocks
(`input_ids` / `position_ids` / `attention_mask` as per-doc lengths).

```bash
python multipacking_stt.py --base-dir /share/stt --workers 96          # everything
python multipacking_stt.py --base-dir /share/stt --subsets 'malaysian-*' 'emilia_zh'
python multipacking_stt.py --base-dir /share/stt --stage download      # zips + metadata only
python multipacking_stt.py --base-dir /share/stt --stage upload        # -> Scicom-intl/Multilingual-STT-multipacking-10k
```

Rows are dropped when `language` is missing/`und`, the normalized text is empty,
the token JSON is absent, or the transcript has more words than speech tokens;
counts land in `<out>/summary.json`. Subset sizes are heavily skewed
(`emilia_zh`, `urdu-tts-corpus`, `common-voice-22` dominate) so parquet files are
snake-balanced across workers by size. **The full corpus is large — check `df -h`
first and scope with `--subsets`.**

## Transcript tooling (copied from `scaling-discrete-speech-token-LLM`)

| file | what |
|---|---|
| `langdetect_glotlid.py` | GlotLID v3 language detector (2102 language+script labels) with margin/macro-group trust signals, subset auditing, and a CLI — this is what produced the `language` column |
| `postnormalizer.py` | rule-based multilingual post-normalizer (stdlib only) — this is what produced the `post-normalized` column |
| `test_postnormalizer.py` | 44 regression cases pinning the normalizer rules: `python test_postnormalizer.py` |
