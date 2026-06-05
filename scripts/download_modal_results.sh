#!/usr/bin/env bash
# Download experiment outputs from Modal volume into ./results/ (notebook layout).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="${MODAL_RESULTS_VOLUME:-saliency-results}"
DEST="${ROOT}/results"

cd "$ROOT"
mkdir -p "$DEST"

ARCHS=(resnet50 vit mechanistic)
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

if modal volume ls "$VOLUME" "/figures/subset_examples" >/dev/null 2>&1; then
  echo "Downloading figures/subset_examples -> $DEST/figures/subset_examples/"
  mkdir -p "$DEST/figures"
  modal volume get "$VOLUME" "/figures/subset_examples" "$DEST/figures"
  FOUND=1
fi
if modal volume ls "$VOLUME" "/figures/subset_sample_grid.png" >/dev/null 2>&1; then
  echo "Downloading figures/subset_sample_grid.png"
  mkdir -p "$DEST/figures"
  modal volume get "$VOLUME" "/figures/subset_sample_grid.png" "$DEST/figures/subset_sample_grid.png"
  FOUND=1
fi

if [[ "$FOUND" -eq 0 ]]; then
  echo "No results found on volume '$VOLUME'. Run an experiment first, e.g.:"
  echo "  modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual"
  exit 1
fi

echo ""
echo "Download complete (3 volume folders: resnet50, vit, mechanistic)."
echo "Parallel-methods layout is normal — not 'missing 1/3' per model:"
echo "  Class A (gradient, input_grad, ig): results/<arch>/seed42/"
echo "  Class B/C (gradcam, gbp, attention_rollout, …): results/<arch>/ top level"
echo "  mechanistic: activation scales + logit corr (separate from cascade npz)"
echo ""
python3 - <<'PY' "$DEST"
import sys
from pathlib import Path

dest = Path(sys.argv[1])
checks = [
    ("resnet50", ["qual_bundle.npz", "seed42/baseline_ig.npz", "baseline_gradcam.npz"]),
    ("vit", ["qual_bundle.npz", "seed42/baseline_ig.npz", "baseline_transformer_gradcam.npz"]),
    ("mechanistic", ["logit_corr_resnet.npy", "activation_scale_resnet_depth00.npy"]),
]
ok = True
for arch, files in checks:
    missing = [f for f in files if not (dest / arch / f).exists()]
    if missing:
        ok = False
        print(f"  {arch}: MISSING {missing}")
    else:
        print(f"  {arch}: OK ({len(files)} key artifacts)")
if ok:
    print("Sanity check passed — full trees downloaded.")
else:
    print("Some expected files missing; re-run Modal experiment or check volume path.")
PY

echo ""
echo "Done. Results in $DEST"
echo "Legacy ViT/ResNet files from old methods? Clean without re-running:"
echo "  python3 scripts/clean_stale_results.py --apply"
echo "  python3 scripts/clean_stale_results.py --apply --modal   # also trim Modal volume"
echo "Run: jupyter notebook notebooks/notebook_analysis.ipynb"
