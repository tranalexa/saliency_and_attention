#!/usr/bin/env bash
# Download experiment outputs from Modal volume into ./results/ (notebook layout).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="${MODAL_RESULTS_VOLUME:-saliency-results}"
DEST="${ROOT}/results"

cd "$ROOT"
mkdir -p "$DEST"

ARCHS=(resnet50 vit dinov2 mechanistic)
FOUND=0

for arch in "${ARCHS[@]}"; do
  if modal volume ls "$VOLUME" "/$arch" >/dev/null 2>&1; then
    echo "Downloading $arch -> $DEST/$arch/"
    rm -rf "$DEST/$arch"
    modal volume get "$VOLUME" "/$arch" "$DEST"
    FOUND=1
  else
    echo "Skip $arch (not on volume)"
  fi
done

if [[ "$FOUND" -eq 0 ]]; then
  echo "No results found on volume '$VOLUME'. Run an experiment first, e.g.:"
  echo "  modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual"
  exit 1
fi

echo "Done. Results in $DEST"
echo "Run: jupyter notebook notebooks/notebook_analysis.ipynb"
