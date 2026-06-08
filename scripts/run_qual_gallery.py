#!/usr/bin/env python3
"""Build and render qual-bundle heatmap galleries for random subset images.

Typical Modal workflow (one index at a time):

  python scripts/run_qual_gallery.py --print-plan

  for IDX in 149 303 328 329 414; do
    modal run modal/app.py --experiment all --qual-only \\
      --image-index-mode fixed --image-index "$IDX" --qual-force
    ./scripts/download_modal_results.sh
    python scripts/run_qual_gallery.py --capture-index "$IDX" --render
  done

Local workflow (needs IMAGENET_ROOT + GPU):

  export IMAGENET_ROOT=/path/to/imagenet
  python scripts/run_qual_gallery.py --build --render

Outputs:
  results/qual_gallery/idx149/{arch}/qual_bundle.npz
  results/figures/qual_gallery/idx149/cascade_grid_{arch}_{preset}.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import METHODS_BY_ARCH, run_qual_bundle_pipeline  # noqa: E402
from viz_utils import (  # noqa: E402
    CASCADE_DISPLAY_PRESETS,
    plot_cascade_paper_grid,
    select_depth_indices,
)

DEFAULT_GALLERY_ROOT = ROOT / "results" / "qual_gallery"
DEFAULT_FIGURES_ROOT = ROOT / "results" / "figures" / "qual_gallery"
PLAN_PATH = DEFAULT_GALLERY_ROOT / "gallery_plan.json"

ARCHS = {
    "resnet50": {"title": "ResNet-50"},
    "vit": {"title": "ViT-B/16"},
}
for _arch in ARCHS:
    ARCHS[_arch]["methods"] = list(METHODS_BY_ARCH[_arch])

# All heatmap styles to export per architecture.
GALLERY_HEATMAP_PRESETS: dict[str, dict] = {
    "overlay_jet": {
        "overlay": True,
        "heatmap_cmap": "jet",
        "display_percentile": 99.0,
    },
    "overlay_turbo": {
        "overlay": True,
        "heatmap_cmap": "turbo",
        "display_percentile": 99.0,
    },
    "overlay_hot": {
        "overlay": True,
        "heatmap_cmap": "hot",
        "display_percentile": 99.0,
    },
    "overlay_viridis": {
        "overlay": True,
        "heatmap_cmap": "viridis",
        "display_percentile": 99.0,
    },
    "masks_gray": {
        "overlay": False,
        "mask_cmap": "gray",
        "display_norm": "minmax",
    },
    **CASCADE_DISPLAY_PRESETS,
}


def _parse_archs(raw: str) -> list[str]:
    archs = [a.strip() for a in raw.split(",") if a.strip()]
    unknown = [a for a in archs if a not in ARCHS]
    if unknown:
        raise ValueError("Unknown arch(s): %s" % ", ".join(unknown))
    return archs


def _parse_indices(raw: str | None, num_images: int, max_index: int, seed: int) -> list[int]:
    if raw:
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        rng = random.Random(seed)
        k = min(num_images, max_index)
        indices = sorted(rng.sample(range(max_index), k=k))
    for idx in indices:
        if idx < 0 or idx >= max_index:
            raise ValueError("image index %d out of range [0, %d)" % (idx, max_index))
    return indices


def _load_manifest_row(manifest: list[dict], dataset_index: int) -> dict | None:
    for row in manifest:
        if int(row.get("dataset_index", -1)) == dataset_index:
            return row
    if 0 <= dataset_index < len(manifest):
        return manifest[dataset_index]
    return None


def _manifest_lookup(dataset_index: int) -> dict | None:
    manifest_path = ROOT / "results" / "diagnostics" / "subset_manifest_first500.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    return _load_manifest_row(manifest, dataset_index)


def write_plan(
    indices: list[int],
    *,
    gallery_root: Path,
    figures_root: Path,
    archs: list[str],
    seed: int,
    max_index: int,
) -> Path:
    gallery_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx in indices:
        row = _manifest_lookup(idx)
        entries.append(
            {
                "image_index": idx,
                "filename": row.get("filename") if row else None,
                "class_name": row.get("class_name") if row else None,
                "wnid": row.get("wnid") if row else None,
            }
        )
    plan = {
        "seed": seed,
        "max_index": max_index,
        "archs": archs,
        "gallery_root": str(gallery_root.relative_to(ROOT)),
        "figures_root": str(figures_root.relative_to(ROOT)),
        "heatmap_presets": list(GALLERY_HEATMAP_PRESETS.keys()),
        "images": entries,
    }
    plan_path = gallery_root / "gallery_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path


def print_plan_commands(indices: list[int]) -> None:
    joined = " ".join(str(i) for i in indices)
    print("Random qual indices:", ", ".join(str(i) for i in indices))
    print()
    print("# Activate venv first (modal CLI lives here, not system PATH):")
    print("source .venv/bin/activate")
    print("modal setup   # one-time, if not authenticated")
    print()
    print("# Run on Modal (repeat for each index):")
    print("set -e")
    print("for IDX in %s; do" % joined)
    print("  modal run modal/app.py --experiment all --qual-only \\")
    print("    --image-index-mode fixed --image-index \"$IDX\" --qual-force")
    print("  ./scripts/download_modal_results.sh")
    print("  python3 scripts/run_qual_gallery.py --capture-index \"$IDX\" --render")
    print("done")
    print()
    print("# Or local build + render (needs IMAGENET_ROOT and GPU):")
    print("export IMAGENET_ROOT=/path/to/imagenet")
    print("python3 scripts/run_qual_gallery.py --build --render \\")
    print("  --indices %s" % ",".join(str(i) for i in indices))


def capture_index(
    image_index: int,
    *,
    gallery_root: Path,
    archs: list[str],
    results_root: Path,
) -> None:
    dest_root = gallery_root / ("idx%03d" % image_index)
    for arch in archs:
        src = results_root / arch / "qual_bundle.npz"
        if not src.exists():
            raise FileNotFoundError("Missing %s — run qual-only first." % src)
        data_idx = int(__import__("numpy").load(src)["image_index"])
        if data_idx != image_index:
            raise ValueError(
                "%s has image_index=%d, expected %d.\n"
                "Qual bundle was not rebuilt for this index. Run qual-only first:\n"
                "  source .venv/bin/activate\n"
                "  modal run modal/app.py --experiment all --qual-only \\\n"
                "    --image-index-mode fixed --image-index %d --qual-force\n"
                "  ./scripts/download_modal_results.sh"
                % (src, data_idx, image_index, image_index)
            )
        dest = dest_root / arch / "qual_bundle.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print("Captured %s -> %s" % (src, dest))


def build_index(
    image_index: int,
    *,
    gallery_root: Path,
    archs: list[str],
    imagenet_root: Path,
    num_images: int,
    device: str,
) -> None:
    for arch in archs:
        results_dir = gallery_root / ("idx%03d" % image_index) / arch
        results_dir.mkdir(parents=True, exist_ok=True)
        idx = run_qual_bundle_pipeline(
            arch=arch,
            imagenet_root=imagenet_root,
            results_dir=results_dir,
            num_images=num_images,
            image_index=image_index,
            image_index_mode="fixed",
            force=True,
            device=device,
        )
        print("Built qual_bundle for %s image_index=%d -> %s" % (arch, idx, results_dir))


def render_index(
    image_index: int,
    *,
    gallery_root: Path,
    figures_root: Path,
    archs: list[str],
) -> list[Path]:
    idx_dir = gallery_root / ("idx%03d" % image_index)
    out_dir = figures_root / ("idx%03d" % image_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _manifest_lookup(image_index)
    label = row.get("class_name") if row else None
    saved: list[Path] = []

    for arch in archs:
        qual_path = idx_dir / arch / "qual_bundle.npz"
        if not qual_path.exists():
            print("Skip render %s: missing %s" % (arch, qual_path))
            continue
        data = __import__("numpy").load(qual_path, allow_pickle=True)
        methods = [m for m in ARCHS[arch]["methods"] if ("baseline_" + m) in data.files]
        order = list(data["order"])
        depth_indices = select_depth_indices(len(order) + 1, max_cols=None)
        title_bits = [ARCHS[arch]["title"], "image %d" % image_index]
        if label:
            title_bits.append(label)
        base_title = " — ".join(title_bits)

        for preset_name, preset in GALLERY_HEATMAP_PRESETS.items():
            out_path = out_dir / ("cascade_grid_%s_%s.png" % (arch, preset_name))
            plot_cascade_paper_grid(
                qual_path,
                methods,
                depth_indices=depth_indices,
                out_path=out_path,
                title=base_title,
                arch=arch,
                max_depth_cols=None,
                show_input_column=True,
                show=False,
                **preset,
            )
            saved.append(out_path)
            print("Wrote", out_path.relative_to(ROOT))

    meta = {
        "image_index": image_index,
        "filename": row.get("filename") if row else None,
        "class_name": row.get("class_name") if row else None,
        "figures": [str(p.relative_to(ROOT)) for p in saved],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20250604)
    parser.add_argument("--max-index", type=int, default=500)
    parser.add_argument("--indices", type=str, default=None, help="comma-separated dataset indices")
    parser.add_argument("--archs", type=str, default="resnet50,vit")
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--imagenet-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-plan", action="store_true", help="write plan + print Modal/local commands")
    parser.add_argument("--capture-index", type=int, default=None, metavar="IDX")
    parser.add_argument("--build", action="store_true", help="build qual bundles locally into gallery folders")
    parser.add_argument("--render", action="store_true", help="render all heatmap presets for captured/built bundles")
    args = parser.parse_args()

    archs = _parse_archs(args.archs)
    indices = _parse_indices(args.indices, args.num_images, args.max_index, args.seed)

    if args.print_plan or (not args.capture_index and not args.build and not args.render):
        plan_path = write_plan(
            indices,
            gallery_root=args.gallery_root,
            figures_root=args.figures_root,
            archs=archs,
            seed=args.seed,
            max_index=args.max_index,
        )
        print("Wrote plan:", plan_path.relative_to(ROOT))
        print_plan_commands(indices)
        if not args.capture_index and not args.build and not args.render:
            return

    if args.capture_index is not None:
        capture_index(
            args.capture_index,
            gallery_root=args.gallery_root,
            archs=archs,
            results_root=args.results_root,
        )
        if args.render:
            render_index(
                args.capture_index,
                gallery_root=args.gallery_root,
                figures_root=args.figures_root,
                archs=archs,
            )
        return

    if args.build:
        imagenet_root = args.imagenet_root or Path(os.environ.get("IMAGENET_ROOT", ""))
        if not imagenet_root or not Path(imagenet_root).exists():
            raise SystemExit("Set IMAGENET_ROOT or pass --imagenet-root for --build")
        for idx in indices:
            build_index(
                idx,
                gallery_root=args.gallery_root,
                archs=archs,
                imagenet_root=Path(imagenet_root),
                num_images=args.max_index,
                device=args.device,
            )

    if args.render:
        for idx in indices:
            render_index(
                idx,
                gallery_root=args.gallery_root,
                figures_root=args.figures_root,
                archs=archs,
            )
        print("Figures under", args.figures_root.relative_to(ROOT))


if __name__ == "__main__":
    main()
