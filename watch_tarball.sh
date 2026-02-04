#!/usr/bin/env bash
set -euo pipefail

TARBALL_PATH="${1:-/workspace/energy-coding/out_ebt_openwebtext.tar.gz}"
INTERVAL_SEC="${2:-2}"

if [[ ! -e "$TARBALL_PATH" ]]; then
  echo "Tarball not found: $TARBALL_PATH"
  exit 1
fi

echo "Watching: $TARBALL_PATH"
echo "Interval: ${INTERVAL_SEC}s"

prev_size=""
while true; do
  if [[ ! -e "$TARBALL_PATH" ]]; then
    echo "Tarball no longer exists: $TARBALL_PATH"
    exit 1
  fi
  size_bytes=$(stat -c "%s" "$TARBALL_PATH")
  mtime=$(stat -c "%y" "$TARBALL_PATH")
  if [[ -n "$prev_size" ]]; then
    if [[ "$size_bytes" == "$prev_size" ]]; then
      echo "$(date -Iseconds) size=${size_bytes}B (no change) mtime=${mtime}"
    else
      echo "$(date -Iseconds) size=${size_bytes}B (growing) mtime=${mtime}"
    fi
  else
    echo "$(date -Iseconds) size=${size_bytes}B mtime=${mtime}"
  fi
  prev_size="$size_bytes"
  sleep "$INTERVAL_SEC"
done
