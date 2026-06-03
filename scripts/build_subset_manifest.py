#!/usr/bin/env python3
"""Build subset_manifest JSON aligned with SortedValDataset (subset_order=sorted).

Usage (from repo root, with ImageNet on disk or IMAGENET_ROOT):

  python scripts/build_subset_manifest.py --imagenet-root /path/to/imagenet
  python scripts/build_subset_manifest.py --num-images 500 --output results/diagnostics/subset_manifest_first500.json

On Modal volume (from laptop):

  modal run scripts/build_subset_manifest.py --imagenet-root /imagenet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import (  # noqa: E402
    SortedValDataset,
    ilsvrc_wnid_to_index,
    validate_imagenet_root,
)


def _wnid_to_primary_class_name(root: Path) -> dict[str, str]:
    """Map WNID -> short class label (first word of synset), from meta.bin or devkit."""
    meta_bin = root / "meta.bin"
    if meta_bin.exists():
        import torch

        wnid_to_classes, _ = torch.load(meta_bin, weights_only=True)
        return {
            str(wnid): str(classes[0]).strip()
            for wnid, classes in wnid_to_classes.items()
        }

    for meta_mat in (
        root / "ILSVRC2012_devkit_t12" / "data" / "meta.mat",
        *root.glob("**/data/meta.mat"),
    ):
        if meta_mat.is_file():
            from scipy.io import loadmat

            synsets = loadmat(str(meta_mat), squeeze_me=True)["synsets"]
            nums_children = list(zip(*synsets))[4]
            leaves = [synsets[i] for i, n in enumerate(nums_children) if n == 0]
            wnids = list(zip(*leaves))[1]
            words = list(zip(*leaves))[2]
            return {
                str(wnid): str(word).split(",")[0].strip()
                for wnid, word in zip(wnids, words)
            }

    raise FileNotFoundError(
        "ILSVRC class names not found under %s (need meta.bin or devkit data/meta.mat)."
        % root
    )


def build_sorted_subset_manifest(
    imagenet_root: Path,
    num_images: int = 500,
) -> list[dict]:
    """Entries for subset positions 0..num_images-1 in sorted val JPEG order."""
    root = Path(imagenet_root)
    val_dir = root / "val"
    if not val_dir.exists():
        raise FileNotFoundError("Missing val/ under %s" % root)

    dataset = SortedValDataset(val_dir, root_for_meta=root)
    wnid_to_idx = ilsvrc_wnid_to_index(root)
    wnid_to_name = _wnid_to_primary_class_name(root)
    n = min(num_images, len(dataset))
    if n < num_images:
        raise ValueError(
            "Requested %d images but sorted val has only %d." % (num_images, len(dataset))
        )

    entries = []
    for i in range(n):
        path = dataset.paths[i]
        wnid = path.parent.name
        if wnid not in wnid_to_idx:
            raise ValueError("Unknown WNID %s at sorted index %d (%s)" % (wnid, i, path.name))
        entries.append(
            {
                "dataset_index": i,
                "wnid": wnid,
                "filename": path.name,
                "class_index": int(dataset.labels[i]),
                "class_name": wnid_to_name.get(wnid, "unknown"),
            }
        )
    return entries


def _assert_sorted_filenames(entries: list[dict]) -> None:
    names = [e["filename"] for e in entries]
    if names != sorted(names):
        for j in range(1, len(names)):
            if names[j] < names[j - 1]:
                raise ValueError(
                    "Manifest filenames not sorted at index %d: %s then %s"
                    % (j, names[j - 1], names[j])
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path(os.environ.get("IMAGENET_ROOT", "")),
        help="ImageNet root containing val/ and meta.bin (default: IMAGENET_ROOT)",
    )
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "subset_manifest_first500.json",
    )
    args = parser.parse_args()
    if not args.imagenet_root:
        parser.error("Set --imagenet-root or IMAGENET_ROOT")

    validate_imagenet_root(args.imagenet_root)
    entries = build_sorted_subset_manifest(args.imagenet_root, num_images=args.num_images)
    _assert_sorted_filenames(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")

    print("Wrote %d entries to %s" % (len(entries), args.output))
    print("First: %s (%s)" % (entries[0]["filename"], entries[0]["class_name"]))
    print("Last:  %s (%s)" % (entries[-1]["filename"], entries[-1]["class_name"]))


if __name__ == "__main__":
    main()
