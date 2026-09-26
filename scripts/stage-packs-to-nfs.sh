#!/bin/bash
# Move finished pack directories off local disk onto the NFS filer, leaving a symlink
# behind so every path the trainer sees stays the same.
#
# Safe because of what was measured (see preparation/FULL_TRAINING_PLAN.md): random block
# reads out of a ChiniDataset pack are CPU-bound on row-group decode, so NFS costs 1.3x,
# and the copy itself is big-file sequential (~276 MB/s). The audio tree must NOT be moved
# this way — 43.5M small files is where NFS is 32-117x slower.
#
#   bash scripts/stage-packs-to-nfs.sh                       # every completed wave + idle packs
#   DIRS="/root/share/packs" bash scripts/stage-packs-to-nfs.sh
set -e
NFS="${NFS:-/mnt/data/multilingual-speech}"
CORPUS="${CORPUS:-/root/share/corpus}"
mkdir -p "$NFS"

stage() {          # stage <local dir>
  local src="$1"
  [ -e "$src" ] || return 0
  [ -L "$src" ] && { echo "already staged: $src"; return 0; }
  local dst="$NFS/${src#/root/share/}"
  mkdir -p "$(dirname "$dst")"
  local sz; sz=$(du -sh "$src" | cut -f1)
  echo "staging $src ($sz) -> $dst"
  local t0=$SECONDS
  rm -rf "$dst.partial"
  cp -r "$src" "$dst.partial"
  # verify file count and byte total before dropping the local copy
  local a b
  a=$(find "$src" -type f | wc -l); b=$(find "$dst.partial" -type f | wc -l)
  [ "$a" = "$b" ] || { echo "FAILED $src: $a files local vs $b on nfs"; return 1; }
  a=$(du -sb "$src" | cut -f1); b=$(du -sb "$dst.partial" | cut -f1)
  [ "$a" = "$b" ] || { echo "FAILED $src: $a bytes local vs $b on nfs"; return 1; }
  rm -rf "$dst"; mv "$dst.partial" "$dst"
  rm -rf "$src"; ln -s "$dst" "$src"
  echo "  done in $((SECONDS-t0))s, $a bytes, symlinked"
}

if [ -n "$DIRS" ]; then
  for d in $DIRS; do stage "$d"; done
else
  # a wave is finished once its summary exists; only then are its pack dirs immutable
  for s in "$CORPUS"/out/summary-wave-*.json; do
    [ -e "$s" ] || continue
    w=$(basename "$s" .json); w=${w#summary-}
    for task in tts stt mel; do stage "$CORPUS/out/corpus-$task/$w"; done
  done
  # packs nothing is writing any more. Off by default: the FLEURS/CV22 packs are due to be
  # re-packed against the unified vocab, and re-packing into a symlinked dir is asking for it.
  if [ -n "$ALSO_IDLE" ]; then
    for d in /root/share/packs /root/share/cv22/out /root/share/fleurs/out; do stage "$d"; done
  fi
fi
df -h / | tail -1
