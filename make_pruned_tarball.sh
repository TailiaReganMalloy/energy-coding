#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/workspace/energy-coding/out_ebt_openwebtext}"
TAR_PATH="${2:-/workspace/energy-coding/out_ebt_openwebtext_pruned.tar.gz}"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "Output directory not found: $OUT_DIR"
  exit 1
fi

find_latest_ckpt() {
  local dir="$1"
  local latest=""
  while IFS= read -r fname; do
    local iter
    iter=$(printf '%s' "$fname" | sed -E 's/ckpt_iter_([0-9]+)\.pt/\1/')
    if [[ "$iter" =~ ^[0-9]+$ ]]; then
      latest="$fname"
    fi
  done < <(find "$dir" -maxdepth 1 -type f -name 'ckpt_iter_*.pt' -printf '%f\n' | sort -V)

  if [[ -n "$latest" ]]; then
    echo "$latest"
    return 0
  fi

  if [[ -f "$dir/ckpt.pt" ]]; then
    echo "ckpt.pt"
    return 0
  fi

  return 1
}

LATEST_CKPT=""
if LATEST_CKPT=$(find_latest_ckpt "$OUT_DIR"); then
  :
fi

# Always include loss files if present
EXTRA_FILES=("losses.pkl" "train_losses.csv" "val_losses.csv")

# Build tar from the project root so paths are clean.
PROJECT_ROOT="$(cd "$(dirname "$OUT_DIR")" && pwd)"
REL_OUT_DIR="$(basename "$OUT_DIR")"

pushd "$PROJECT_ROOT" >/dev/null

FILES_TO_TAR=()
if [[ -n "$LATEST_CKPT" ]]; then
  FILES_TO_TAR+=("${REL_OUT_DIR}/${LATEST_CKPT}")
fi
for f in "${EXTRA_FILES[@]}"; do
  if [[ -f "${REL_OUT_DIR}/${f}" ]]; then
    FILES_TO_TAR+=("${REL_OUT_DIR}/${f}")
  fi
done

# If no files selected, bail
if [[ ${#FILES_TO_TAR[@]} -eq 0 ]]; then
  echo "No files selected to tar. Nothing to do."
  exit 1
fi

tar -czf "$TAR_PATH" "${FILES_TO_TAR[@]}"

popd >/dev/null

echo "Wrote tarball: $TAR_PATH"
