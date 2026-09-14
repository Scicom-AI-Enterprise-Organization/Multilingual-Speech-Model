#!/bin/bash
# Build the single audio root the mel packs resolve their paths against.
#
# Mel blocks store `audio/{language}/{split}/{id}.{wav,mp3}` and the trainer takes one
# --audio_dir, but the two corpora's audio lives in separate trees. Their language tags
# never collide (FLEURS uses en_us-style locales, CV22 plain en), so one directory of
# per-language symlinks serves both without copying ~250GB.
#
#   bash preparation/link_audio_root.sh [root] [fleurs_audio] [cv22_audio]
set -e
ROOT="${1:-/share/audio-root}"
FLEURS="${2:-/share/fleurs-work/audio}"
CV22="${3:-/share/cv22/audio}"

mkdir -p "$ROOT/audio"
linked=0
for src in "$FLEURS" "$CV22"; do
  [ -d "$src" ] || { echo "skipping $src (not present)"; continue; }
  for lang in "$src"/*/; do
    name=$(basename "$lang")
    target="$ROOT/audio/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
      echo "refusing to replace non-symlink $target" >&2; exit 1
    fi
    ln -sfn "${lang%/}" "$target"
    linked=$((linked + 1))
  done
done
echo "$linked language links under $ROOT/audio"
