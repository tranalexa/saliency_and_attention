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
    .pip_install("tqdm", "scipy", "numpy")
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


def _organize_val_flat_dir(val_workdir, devkit_data_dir):
    """Sort flat val JPEGs into synset subfolders (equivalent to valprep.sh)."""
    import shutil

    import numpy as np
    from scipy.io import loadmat  # noqa: F401

    gt_file = devkit_data_dir / "ILSVRC2012_validation_ground_truth.txt"
    meta_file = devkit_data_dir / "meta.mat"
    if not gt_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            "Devkit must contain data/ILSVRC2012_validation_ground_truth.txt and data/meta.mat"
        )

    meta = loadmat(str(meta_file), squeeze_me=True, struct_as_record=False)
    synsets_raw = meta["synsets"]
    synsets = []
    for s in synsets_raw:
        if isinstance(s, np.ndarray):
            synsets.append(str(s[0]))
        else:
            synsets.append(str(s))
    labels = [int(x) for x in gt_file.read_text().strip().splitlines()]

    jpgs = sorted(val_workdir.glob("*.JPEG"))
    if len(jpgs) != len(labels):
        raise RuntimeError(
            "Expected %d val images, found %d (check val tar extraction)." % (len(labels), len(jpgs))
        )

    for jpg, label in zip(jpgs, labels):
        synset = synsets[label - 1]
        dest_dir = val_workdir / synset
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(jpg), str(dest_dir / jpg.name))

    return len(jpgs), len(set(synsets))


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
        n_subdirs = sum(1 for p in val_dir.iterdir() if p.is_dir())
        n_files = sum(1 for _ in val_dir.rglob("*.JPEG"))
        if n_subdirs > 0 and n_files > 1000:
            print("Skipping download: %d class dirs, %d images under %s" % (n_subdirs, n_files, val_dir))
            return {"status": "skipped", "val_dir": str(val_dir), "n_images": n_files}

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

    shutil.move(str(val_workdir), str(val_dir))
    n_subdirs = sum(1 for p in val_dir.iterdir() if p.is_dir())
    print("Done: %d images, %d class folders at %s" % (n_images, n_subdirs, val_dir))

    shutil.rmtree(staging, ignore_errors=True)
    modal.Volume.from_name(IMAGENET_VOLUME).commit()
    return {"status": "ok", "val_dir": str(val_dir), "n_images": n_images, "n_classes": n_subdirs}


@app.local_entrypoint()
def main(
    val_tar_url: str = "",
    devkit_tar_url: str = "",
    skip_if_exists: bool = True,
):
    """
    Download ImageNet val + devkit inside Modal into volume saliency-imagenet.

    Paste full https URLs from https://image-net.org/download.php (ILSVRC 2012), not filenames.
    """
    _validate_download_url(val_tar_url, "val-tar-url")
    _validate_download_url(devkit_tar_url, "devkit-tar-url")
    result = download_imagenet_val.remote(
        val_tar_url=val_tar_url,
        devkit_tar_url=devkit_tar_url,
        skip_if_exists=skip_if_exists,
    )
    print(result)
