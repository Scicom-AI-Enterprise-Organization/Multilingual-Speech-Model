#!/bin/bash
# One --audio_dir that resolves every mel pack's paths.
#
# The three mel sources store paths in two shapes:
#   FLEURS  audio/{locale}/{split}/{id}.wav   -> base is the PARENT of audio/
#   CV22    audio/{lang}/...                  -> same
#   corpus  {subset}_audio/{rest}.mp3         -> base is /root/share/corpus
# so the root gets an `audio/` directory holding the FLEURS+CV22 locale trees, plus one
# symlink per corpus subset beside it. FLEURS locale tags (en_us) and CV22 language codes
# (en) never collide, and corpus subsets all end in _audio.
#
# Always re-points existing links (ln -sfn). The first version of this root was built
# before /share was renamed to /root/share, so all 234 of its links were dangling -- and a
# dangling audio root is silent: every mel row is counted as no_audio at pack time, or at
# train time the block keeps its random <|mel|> embeddings. Verify, never assume.
set -e
ROOT="${ROOT:-/root/share/audio-root}"
FLEURS="${FLEURS:-/root/share/fleurs-work/audio}"
CV22="${CV22:-/root/share/cv22/audio}"
CORPUS="${CORPUS:-/root/share/corpus}"
mkdir -p "$ROOT/audio"

n_f=0 n_c=0 n_s=0
for d in "$FLEURS"/*;  do [ -d "$d" ] && { ln -sfn "$d" "$ROOT/audio/$(basename "$d")"; n_f=$((n_f+1)); }; done
for d in "$CV22"/*;    do [ -d "$d" ] && { ln -sfn "$d" "$ROOT/audio/$(basename "$d")"; n_c=$((n_c+1)); }; done
for d in "$CORPUS"/*_audio; do [ -d "$d" ] && { ln -sfn "$d" "$ROOT/$(basename "$d")"; n_s=$((n_s+1)); }; done

broken=$(find "$ROOT" -maxdepth 2 -xtype l | wc -l)
echo "linked: fleurs $n_f, cv22 $n_c, corpus $n_s | dangling links now: $broken"
[ "$broken" = "0" ] || { echo "FAILED: $broken dangling links remain"; exit 1; }
