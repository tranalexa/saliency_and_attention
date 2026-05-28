"""
Download ImageNet ILSVRC2012 validation into Modal volume (no local copy).

Obtain download URLs from https://image-net.org after registration:
  - ILSVRC2012_img_val.tar
  - ILSVRC2012_devkit_t12.tar

Usage (from repo root):

  modal run modal/download_imagenet.py \\
    --val-tar-url "https://..." \\
    --devkit-tar-url "https://..."
"""
from __future__ import annotations

import modal

APP_NAME = "saliency-imagenet-download"
IMAGENET_VOLUME = "saliency-imagenet"
IMAGENET_MOUNT = "/imagenet"

app = modal.App(APP_NAME)
imagenet_vol = modal.Volume.from_name(IMAGENET_VOLUME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "ca-certificates", "tar")
    .pip_install("tqdm", "scipy", "numpy", "torch")
)


def _validate_download_url(url: str, label: str) -> None:
    """Require a full https URL from image-net.org, not just a filename."""
    if not url or not url.strip():
        raise ValueError("Missing --%s" % label)
    url = url.strip()
    if url.startswith(("http://", "https://")):
        return
    raise ValueError(
        "Invalid --%s: got %r\n"
        "You must pass the full download URL from https://image-net.org/download.php "
        "(ILSVRC 2012 section), not just the filename.\n"
        "On the download page: right-click the val / devkit link → Copy Link Address.\n"
        "Example shape: https://image-net.org/data/.../ILSVRC2012_img_val.tar"
        % (label, url)
    )


def _is_wnid_dirname(name: str) -> bool:
    import re

    return re.fullmatch(r"n\d{8}", name) is not None


def parse_devkit_meta(devkit_data_dir):
    """Parse ILSVRC2012 devkit into (wnid_to_classes, val_wnids).

    Matches torchvision.datasets.ImageNet.parse_devkit_archive exactly: default
    struct_as_record=True so each synset row is a numpy.void that unpacks as
    (ILSVRC2012_ID, WNID, words, gloss, num_children, ...). struct_as_record=False
    returns mat_struct objects that aren't iterable, which breaks zip(*synsets).
    """
    from scipy.io import loadmat

    meta_file = devkit_data_dir / "meta.mat"
    gt_file = devkit_data_dir / "ILSVRC2012_validation_ground_truth.txt"
    if not meta_file.exists() or not gt_file.exists():
        raise FileNotFoundError(
            "Devkit must contain data/meta.mat and data/ILSVRC2012_validation_ground_truth.txt"
        )

    synsets = loadmat(str(meta_file), squeeze_me=True)["synsets"]
    nums_children = list(zip(*synsets))[4]
    leaves = [synsets[i] for i, n in enumerate(nums_children) if n == 0]
    idcs, wnids, classes = list(zip(*leaves))[:3]
    classes = [tuple(str(c).split(", ")) for c in classes]
    idx_to_wnid = {int(idx): str(w) for idx, w in zip(idcs, wnids)}
    wnid_to_classes = {str(w): c for w, c in zip(wnids, classes)}

    val_idcs = [int(x) for x in gt_file.read_text().strip().splitlines()]
    val_wnids = [idx_to_wnid[i] for i in val_idcs]
    return wnid_to_classes, val_wnids


def _organize_val_flat_dir(val_workdir, devkit_data_dir):
    """Sort flat val JPEGs into synset subfolders (equivalent to valprep.sh)."""
    import shutil

    _, val_wnids = parse_devkit_meta(devkit_data_dir)

    jpgs = sorted(val_workdir.glob("*.JPEG"))
    if len(jpgs) != len(val_wnids):
        raise RuntimeError(
            "Expected %d val images, found %d (check val tar extraction)."
            % (len(val_wnids), len(jpgs))
        )

    for jpg, wnid in zip(jpgs, val_wnids):
        dest_dir = val_workdir / wnid
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(jpg), str(dest_dir / jpg.name))

    return len(jpgs), len(set(val_wnids))


def _write_meta_bin(imagenet_root, devkit_data_dir) -> None:
    import torch

    meta = parse_devkit_meta(devkit_data_dir)
    out = imagenet_root / "meta.bin"
    torch.save(meta, out)
    print("Wrote %s (%d classes)" % (out, len(meta[0])))


@app.function(
    image=download_image,
    timeout=86400,
    volumes={IMAGENET_MOUNT: imagenet_vol},
    cpu=4,
    memory=8192,
)
def download_imagenet_val(
    val_tar_url: str,
    devkit_tar_url: str,
    skip_if_exists: bool = True,
):
    import shutil
    import tarfile
    from pathlib import Path
    from urllib.request import urlretrieve

    from tqdm import tqdm

    imagenet_root = Path(IMAGENET_MOUNT)
    val_dir = imagenet_root / "val"
    staging = imagenet_root / "_staging"
    val_tar_path = staging / "ILSVRC2012_img_val.tar"
    devkit_tar_path = staging / "ILSVRC2012_devkit_t12.tar"
    val_flat = staging / "val_flat"
    devkit_dir = staging / "devkit"

    if skip_if_exists and val_dir.exists():
        class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]
        n_subdirs = len(class_dirs)
        n_files = sum(1 for _ in val_dir.rglob("*.JPEG"))
        if n_subdirs > 0 and n_files > 1000:
            bad_dirs = [p.name for p in class_dirs if not _is_wnid_dirname(p.name)]
            if bad_dirs:
                print(
                    "Existing val dir has non-WNID class folders (first few: %s); "
                    "re-downloading and rebuilding /val."
                    % sorted(bad_dirs)[:5]
                )
            else:
                meta_bin = imagenet_root / "meta.bin"
                if meta_bin.exists():
                    print("Skipping download: %d class dirs, %d images under %s" % (n_subdirs, n_files, val_dir))
                    return {"status": "skipped", "val_dir": str(val_dir), "n_images": n_files}
                return {
                    "status": "need_meta_bin",
                    "val_dir": str(val_dir),
                    "n_images": n_files,
                    "hint": "modal run modal/download_imagenet.py --backfill-meta-only --devkit-tar-url URL",
                }

    _validate_download_url(val_tar_url, "val-tar-url")
    _validate_download_url(devkit_tar_url, "devkit-tar-url")

    staging.mkdir(parents=True, exist_ok=True)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    def download(url: str, dest: Path):
        print("Downloading", dest.name, "...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
            urlretrieve(url, dest, reporthook=bar.update_to)
        print("Saved %.2f GB" % (dest.stat().st_size / 1e9))

    download(val_tar_url, val_tar_path)
    download(devkit_tar_url, devkit_tar_path)

    print("Extracting validation tar...")
    if val_flat.exists():
        shutil.rmtree(val_flat)
    val_flat.mkdir(parents=True)
    with tarfile.open(val_tar_path, "r") as tf:
        tf.extractall(path=val_flat)
    jpgs = list(val_flat.rglob("*.JPEG"))
    if not jpgs:
        raise RuntimeError("No .JPEG files found after extracting val tar.")
    val_workdir = jpgs[0].parent
    print("Found %d JPEGs in %s" % (len(jpgs), val_workdir))

    print("Extracting devkit tar...")
    if devkit_dir.exists():
        shutil.rmtree(devkit_dir)
    devkit_dir.mkdir(parents=True)
    with tarfile.open(devkit_tar_path, "r") as tf:
        tf.extractall(path=devkit_dir)
    data_dirs = list(devkit_dir.rglob("data"))
    devkit_data = None
    for d in data_dirs:
        if (d / "meta.mat").exists() and (d / "ILSVRC2012_validation_ground_truth.txt").exists():
            devkit_data = d
            break
    if devkit_data is None:
        raise RuntimeError("Could not find devkit data/ with meta.mat and ground truth file.")

    print("Organizing val into class folders...")
    n_images, _ = _organize_val_flat_dir(val_workdir, devkit_data)

    print("Writing meta.bin for ILSVRC class indices (DINOv2 / torchvision)...")
    _write_meta_bin(imagenet_root, devkit_data)

    shutil.move(str(val_workdir), str(val_dir))
    n_subdirs = sum(1 for p in val_dir.iterdir() if p.is_dir())
    print("Done: %d images, %d class folders at %s" % (n_images, n_subdirs, val_dir))

    shutil.rmtree(staging, ignore_errors=True)
    modal.Volume.from_name(IMAGENET_VOLUME).commit()
    return {"status": "ok", "val_dir": str(val_dir), "n_images": n_images, "n_classes": n_subdirs}


@app.function(
    image=download_image,
    timeout=3600,
    volumes={IMAGENET_MOUNT: imagenet_vol},
)
def backfill_meta_bin(devkit_tar_url: str):
    """Download devkit only and write /imagenet/meta.bin (for existing val/ layouts)."""
    import shutil
    import tarfile
    from pathlib import Path
    from urllib.request import urlretrieve

    _validate_download_url(devkit_tar_url, "devkit-tar-url")
    imagenet_root = Path(IMAGENET_MOUNT)
    staging = imagenet_root / "_meta_staging"
    devkit_tar_path = staging / "ILSVRC2012_devkit_t12.tar"
    devkit_dir = staging / "devkit"

    staging.mkdir(parents=True, exist_ok=True)
    print("Downloading devkit...")
    urlretrieve(devkit_tar_url.strip(), devkit_tar_path)

    if devkit_dir.exists():
        shutil.rmtree(devkit_dir)
    devkit_dir.mkdir(parents=True)
    with tarfile.open(devkit_tar_path, "r") as tf:
        tf.extractall(path=devkit_dir)
    data_dirs = list(devkit_dir.rglob("data"))
    devkit_data = None
    for d in data_dirs:
        if (d / "meta.mat").exists() and (d / "ILSVRC2012_validation_ground_truth.txt").exists():
            devkit_data = d
            break
    if devkit_data is None:
        raise RuntimeError("Could not find devkit data/ with meta.mat")

    _write_meta_bin(imagenet_root, devkit_data)
    shutil.rmtree(staging, ignore_errors=True)
    imagenet_vol.commit()
    return {"status": "ok", "meta_bin": str(imagenet_root / "meta.bin")}


@app.local_entrypoint()
def main(
    val_tar_url: str = "",
    devkit_tar_url: str = "",
    skip_if_exists: bool = True,
    backfill_meta_only: bool = False,
):
    """
    Download ImageNet val + devkit inside Modal into volume saliency-imagenet.

    Paste full https URLs from https://image-net.org/download.php (ILSVRC 2012), not filenames.
    """
    if backfill_meta_only:
        _validate_download_url(devkit_tar_url, "devkit-tar-url")
        result = backfill_meta_bin.remote(devkit_tar_url=devkit_tar_url)
        print(result)
        return
    _validate_download_url(val_tar_url, "val-tar-url")
    _validate_download_url(devkit_tar_url, "devkit-tar-url")
    result = download_imagenet_val.remote(
        val_tar_url=val_tar_url,
        devkit_tar_url=devkit_tar_url,
        skip_if_exists=skip_if_exists,
    )
    print(result)
