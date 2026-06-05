"""
Export example ImageNet val images for the experiment subset (Modal volumes).

Reads ``image_indices.npy`` from the last cascade run on the results volume,
copies full-resolution JPEGs from the ImageNet volume, and writes a contact
sheet under ``/results/figures/subset_examples/``.

Usage (from repo root):

  modal run modal/export_subset_images.py
  modal run modal/export_subset_images.py --num-samples 20
  ./scripts/download_modal_results.sh   # or pull figures only (see script)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

APP_NAME = "saliency-export-subset-images"
IMAGENET_VOLUME = "saliency-imagenet"
RESULTS_VOLUME = "saliency-results"
IMAGENET_MOUNT = "/imagenet"
RESULTS_MOUNT = "/results"

REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
imagenet_vol = modal.Volume.from_name(IMAGENET_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-pytorch.txt"))
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
)


def _pick_sample_positions(n_total: int, num_samples: int) -> list[int]:
    if num_samples >= n_total:
        return list(range(n_total))
    if num_samples <= 1:
        return [0]
    step = (n_total - 1) / float(num_samples - 1)
    return sorted({int(round(i * step)) for i in range(num_samples)})


def _load_run_indices(results_mount: Path, arch: str) -> tuple[list[int], dict]:
    arch_dir = results_mount / arch
    indices_path = arch_dir / "image_indices.npy"
    if not indices_path.exists():
        raise FileNotFoundError(
            "Missing %s on results volume. Run cascade first." % indices_path
        )
    import numpy as np

    indices = [int(x) for x in np.load(indices_path).tolist()]
    config = {}
    config_path = arch_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    return indices, config


def _write_contact_sheet(
    image_dir: Path, rows: list[dict], grid_path: Path
) -> Path:
    import math

    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(rows)
    if n == 0:
        raise ValueError("No images to plot")
    cols = min(4, n)
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.2 * rows_n))
    axes_flat = (
        axes.flatten() if hasattr(axes, "flatten") else [axes]
    )
    for ax, row in zip(axes_flat, rows):
        img_path = image_dir / Path(row["exported_path"]).name
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(
            "#%d %s\n%s" % (row["dataset_index"], row["class_name"], row["filename"]),
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes_flat[len(rows) :]:
        ax.axis("off")
    fig.suptitle("ImageNet val subset (last run indices)", fontsize=11)
    fig.tight_layout()
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return grid_path


def _extra_qual_indices(results_mount: Path) -> list[int]:
    import numpy as np

    extra = []
    for arch in ("resnet50", "vit"):
        qual_path = results_mount / arch / "qual_bundle.npz"
        if qual_path.exists():
            extra.append(int(np.load(qual_path)["image_index"]))
    return extra


@app.function(
    image=export_image,
    timeout=3600,
    volumes={IMAGENET_MOUNT: imagenet_vol, RESULTS_MOUNT: results_vol},
)
def export_subset_images(
    arch: str = "resnet50",
    num_samples: int = 16,
    output_subdir: str = "subset_examples",
) -> list[str]:
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import SortedValDataset, validate_imagenet_root

    sys.path.insert(0, "/root/scripts")
    from build_subset_manifest import _wnid_to_primary_class_name

    imagenet_root = Path(IMAGENET_MOUNT)
    results_root = Path(RESULTS_MOUNT)
    validate_imagenet_root(imagenet_root)

    run_indices, config = _load_run_indices(results_root, arch)
    n_run = len(run_indices)
    print(
        "Loaded %d dataset indices from %s (num_images=%s, subset_order=%s)"
        % (
            n_run,
            arch,
            config.get("num_images", "?"),
            config.get("subset_order", "sorted"),
        )
    )

    positions = _pick_sample_positions(n_run, num_samples)
    for qual_idx in _extra_qual_indices(results_root):
        if qual_idx in run_indices:
            pos = run_indices.index(qual_idx)
            if pos not in positions:
                positions.append(pos)
    positions = sorted(set(positions))

    dataset_indices = [run_indices[p] for p in positions]
    val_dir = imagenet_root / "val"
    dataset = SortedValDataset(val_dir, root_for_meta=imagenet_root)
    try:
        wnid_to_name = _wnid_to_primary_class_name(imagenet_root)
    except FileNotFoundError:
        wnid_to_name = {}

    out_dir = results_root / "figures" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    written = []
    for pos, ds_idx in zip(positions, dataset_indices):
        if ds_idx < 0 or ds_idx >= len(dataset):
            raise IndexError(
                "dataset_index %d out of range (val has %d images)" % (ds_idx, len(dataset))
            )
        src = dataset.paths[ds_idx]
        wnid = src.parent.name
        class_name = wnid_to_name.get(wnid, wnid)
        stem = "idx%03d_ds%05d_%s" % (pos, ds_idx, src.stem)
        dest = out_dir / ("%s.JPEG" % stem)
        shutil.copy2(src, dest)
        written.append(str(dest.relative_to(results_root)))
        manifest_rows.append(
            {
                "subset_position": pos,
                "dataset_index": ds_idx,
                "filename": src.name,
                "wnid": wnid,
                "class_name": class_name,
                "class_index": int(dataset.labels[ds_idx]),
                "exported_path": str(dest.relative_to(results_root)),
            }
        )

    manifest_path = out_dir / "exported_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "source_arch": arch,
                "n_run_images": n_run,
                "run_dataset_indices": run_indices,
                "experiment_config": config,
                "samples": manifest_rows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    written.append(str(manifest_path.relative_to(results_root)))

    grid_path = _write_contact_sheet(
        out_dir, manifest_rows, results_root / "figures" / "subset_sample_grid.png"
    )
    written.append(str(grid_path.relative_to(results_root)))

    results_vol.commit()
    print("Wrote %d images + grid to %s" % (len(manifest_rows), out_dir))
    for rel in written:
        print(" ", rel)
    return written


@app.local_entrypoint()
def main(
    arch: str = "resnet50",
    num_samples: int = 16,
    output_subdir: str = "subset_examples",
):
    export_subset_images.remote(arch=arch, num_samples=num_samples, output_subdir=output_subdir)
