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
python multipacking_stt.py --base-dir /root/share/stt --workers 96          # everything
python multipacking_stt.py --base-dir /root/share/stt --subsets 'malaysian-*' 'emilia_zh'
python multipacking_stt.py --base-dir /root/share/stt --stage download      # zips + metadata only
python multipacking_stt.py --base-dir /root/share/stt --stage upload        # -> Scicom-intl/Multilingual-STT-multipacking-10k
```

Rows are dropped when `language` is missing/`und`, the normalized text is empty,
the token JSON is absent, or the transcript has more words than speech tokens;
counts land in `<out>/summary.json`. Subset sizes are heavily skewed
(`emilia_zh`, `urdu-tts-corpus`, `common-voice-22` dominate) so parquet files are
snake-balanced across workers by size. **The full corpus is large — check `df -h`
first and scope with `--subsets`.**

## Raw mel variant — `multipacking_stt_mel.py`

Same task, different modality: instead of NeuCodec tokens the document carries the
audio itself, as placeholders the trainer swaps for projected whisper log-mel frames.

```
<|im_start|><|STT|><|mel_start|>{P × <|mel|>}<|mel_end|><|{language}|>{text}<|im_end|>
```

`P = samples // 320`, i.e. **50 positions/s** — the NeuCodec rate, so a mel document and
a token document cost the same context for the same audio and the two are comparable at
equal sequence length. There is **no whisper encoder**: only whisper's mel front end
survives, and 2 frames are *stacked* (not pooled) into the projection. See
[../mel_audio.py](../mel_audio.py) for why stacking rather than pooling.

The pack stores **paths, not audio**: an `audio` column (json list, relative to the
extracted audio tree) and `audio_samples`. Packing therefore never decodes anything — it
reads `sf.info` headers — and the dataset reads the files and resamples on the fly in the
dataloader workers, with the STFT on the GPU. The extracted tree has to be present on the
training box and is passed to the trainer as `--audio_dir`.

Utterances are trimmed to a multiple of 320 samples so each contributes a whole number of
stacked frames, and `audio_samples` is authoritative: the decoded waveform is forced to
it, because a resampler that rounds a few samples differently would shift every later
utterance in the block onto the wrong placeholders.

```bash
pip install soundfile soxr

# audio zips are the *audio* corpus — always scope, and check df -h first
python multipacking_stt_mel.py --base-dir /root/share/stt-mel --subsets 'malaysian-*' 'emilia_zh'
python multipacking_stt_mel.py --base-dir /root/share/stt-mel --stage upload
```

Extra drop reasons over the token pack: `duration` (outside `--min-seconds` /
`--max-seconds`, default 0.3–30s) and `probe_error`. A `missing` count larger than
`docs` means the extracted zip's layout does not match `audio_filename` — a
path-convention mismatch, not missing data; the pack warns about it.

**Language token ids are not recomputed here.** They are read from the token pack's
`stt_added_tokens.json` (fetched from `Scicom-intl/Multilingual-STT-multipacking-10k`,
or passed with `--stt-tokens`), because a narrower `--subsets` selection would produce a
shorter language list and silently shift every id after it. The pack writes
`mel_added_tokens.json` = those tokens + `<|mel_start|> <|mel|> <|mel_end|>`, which is
what `--stt_tokens_file` and the trainer reproduce.

## Tests

```bash
python test_mel_pipeline.py      # 12 cases, no GPU and no dataset
```

Pins the invariants that fail *silently*: the batched log-mel matching
`WhisperFeatureExtractor` utterance for utterance, no bleed between neighbours sharing
one STFT buffer, 50 positions/s, the codec round-trip preserving the packed sample
count, and `<|mel|>` placeholders landing on their own utterance's frames in order.

## Transcript tooling (copied from `scaling-discrete-speech-token-LLM`)

| file | what |
|---|---|
| `langdetect_glotlid.py` | GlotLID v3 language detector (2102 language+script labels) with margin/macro-group trust signals, subset auditing, and a CLI — this is what produced the `language` column |
| `postnormalizer.py` | rule-based multilingual post-normalizer (stdlib only) — this is what produced the `post-normalized` column |
| `test_postnormalizer.py` | 44 regression cases pinning the normalizer rules: `python test_postnormalizer.py` |
