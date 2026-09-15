# low-language-testset

Held-out TTS/ASR test set for the **long tail** of
[malaysia-ai/Multilingual-TTS](https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS):
50 of the lowest-resource languages in the corpus, 25 utterances each, published as
[malaysia-ai/Low-Language-TTS](https://huggingface.co/datasets/malaysia-ai/Low-Language-TTS)
(split `test`, 1,250 rows / 2.56 hours / 514 speakers / 59 subsets, Wayuu at 1,128 corpus
rows through Kasem at 61,277).

Every row carries **audio + NeuCodec tokens + normalized transcription**, so it can score
either direction without touching the parent corpus: TTS/VC (generate from `post-normalized`,
CER against it) and STT (`tokens` in, text out).

```bash
export HF_TOKEN=...                      # write access to malaysia-ai
python build_testset.py --stage all --work-dir work
```

| stage | what it does | cost |
|---|---|---|
| `scan` | per-parquet `language` histogram over the 1,561 metadata parquets of `Multilingual-TTS-language`, column-pruned | ~10 min, one JSON per parquet under `work/counts/` |
| `select` | rank attested languages, verify candidate rows, write `work/selection.json` | ~2 min / ~0.5GB read |
| `fetch` | pull the mp3 + token JSON of every candidate straight out of the remote zips | `work/fetched/` |
| `push` | parquet + dataset card → `malaysia-ai/Low-Language-TTS` | |
| `verify` | re-download the published split and re-check every invariant | |

Every stage is resumable — re-running skips whatever is already on disk.

## Facts that shape decisions

- **Nothing is bulk-downloaded.** Metadata parquets are read column-pruned over
  `HfFileSystem`, and audio/token zips are read by parsing the zip central directory
  remotely and range-requesting only the members we want (`read_members`). The parent
  corpus' zips are tens of TB; this pulls a few hundred MB.
- **The raw histogram tail is not "low-resource languages", it is GlotLID noise.**
  This was measured, not assumed: ranking all 2,100 labels by corpus rows and verifying
  the 500 rarest yielded **3** usable languages out of 500. Short transcripts get
  misclassified constantly (a three-word Hausa line lands on `wnc_Latn`), and
  `und_<script>` labels (`und_Lydi`, `und_Phnx`) are punctuation false positives.
- **What works instead: subset attestation.** A language counts as real when some
  subset was collected for it — >= 50 rows *and* >= 40% of that subset. 177 labels
  clear that bar and rank into a clean tail (Wayuu, Berom, Iban, Baoulé, Skolt Sami,
  Karamojong). It is also 20x cheaper to read: the subsets belonging to a low-resource
  language are small files, where the same languages scattered across `common-voice-22`
  cost 4GB to sift.
- **Rows are still verified one by one**, because a subset's minority rows are not its
  language: >= 40 chars and >= 5 words (characters instead of words for scripts written
  without spaces), and a re-run of GlotLID v3 that reproduces the stored label at
  prob >= 0.90 and margin >= 0.50. On long+confident rows the two labellings agree
  97-99%; on everything they agree 62-88%.
- **Cut every range-read zip member to `compress_size`.** The read fetches a range
  starting at the local file header and a bit past the end (the header length is not
  known in advance). Deflate members survive a sloppy tail because zlib stops at the
  end of the stream — **stored** members do not, and these zips store the token JSON
  uncompressed: every file came back with ~442 bytes of the next entry's header glued
  on, which still decodes as an mp3 and fails only on `json.loads`. `_local_header_data`
  now slices to `compress_size` and asserts the decompressed length equals `file_size`.
- **Audio and tokens come from two different zips** (`<folder>.zip`,
  `<folder>_neucodec.zip`), so a path-convention slip silently pairs a clip with someone
  else's tokens. `build_rows` decodes each clip and drops rows where the audio length and
  `len(tokens) / 50` disagree by more than 20%.
- The token path mapping is the same one `stt/multipacking_stt.py` uses:
  `folder/rest.mp3` -> `folder_neucodec/rest.json`. Some subsets have sharded audio
  (`folder-0-0.zip`, `folder-1-0.zip`); `fetch` walks the shards until it finds the member.
- **Do not derive the histogram cache filename from the parquet path.** Subset names
  collide both on separators (`ATC_combined__Tabys`) and on case (`MSC`/`msc`,
  `Sagalee`/`sagalee` — indistinguishable on macOS), and each collision silently drops a
  subset. `counts_key` hashes the path and stores it inside the file instead.
- Rows are deduplicated by text and capped at 4 per speaker — **relaxed** when the cap
  would disqualify the language, because much of the tail is single-speaker corpora and
  enforcing diversity there biases the set towards languages that are not low-resource.
- **Over-select languages, not just rows.** Some languages lose most of their rows at fetch
  time (token JSONs simply absent upstream — `hif_Latn` kept 10 of 55, `ewo_Latn` 24 of 25).
  `select` therefore keeps `N_LANGUAGES + SELECT_EXTRA_LANGUAGES` and `push` drops any
  language short of 25 and trims to the 50 lowest-resource complete ones — a spare language
  with all 25 rows beats a ragged one.
- `fetch` marks folders done with a marker file rather than counting files: with members
  missing upstream, "every row present" never becomes true and the folder is re-pulled on
  every run.

## Result (2026-09-14)

1,250 rows / 50 languages / 25 each / 2.56 hours / 514 speakers / 59 subsets, from Wayuu
(1,128 rows in the parent corpus) to Kasem (61,277). Verified after upload: no undecodable
audio, no duplicate `audio_filename`, no token outside the NeuCodec range, and the worst
disagreement between decoded duration and `len(tokens) / 50` across all 1,250 rows is
**0.02s** — the audio and the tokens really are the same clip.

## Tuning

Everything is a module constant at the top of `build_testset.py`: `N_LANGUAGES`,
`N_PER_LANGUAGE`, `CANDIDATE_POOL` (how deep into the tail to look), `ATTEST_ROWS` /
`ATTEST_SHARE` (what makes a language real), `MIN_PROB`, `MIN_MARGIN`, `MIN_CHARS`,
`MIN_WORDS`, `MAX_ROWS_PER_SPEAKER`. Widening the pool or
loosening the LID gates costs another `select` pass (`work/counts/` is reused).
