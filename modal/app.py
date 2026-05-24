"""
Modal cloud runner for PyTorch sanity-check experiments.

Usage (from repo root):
  modal run modal/app.py --experiment resnet50 --num-images 500
  modal run modal/app.py --experiment all --num-images 10 --skip-qual
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "saliency-sanity-checks"
IMAGENET_VOLUME = "saliency-imagenet"
RESULTS_VOLUME = "saliency-results"
IMAGENET_MOUNT = "/imagenet"
RESULTS_MOUNT = "/results"
GPU_TYPE = "A10G"
TIMEOUT_SEC = 86400

REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

imagenet_vol = modal.Volume.from_name(IMAGENET_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-pytorch.txt"))
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
)

volume_mounts = {IMAGENET_MOUNT: imagenet_vol, RESULTS_MOUNT: results_vol}


def _run_pipeline(fn_name: str, results_subdir: str, num_images: int, batch_size: int, skip_qual: bool, **kwargs):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import (
        run_dinov2_pipeline,
        run_mechanistic_pipeline,
        run_resnet50_pipeline,
        run_vit_pipeline,
    )

    fns = {
        "resnet50": run_resnet50_pipeline,
        "vit": run_vit_pipeline,
        "dinov2": run_dinov2_pipeline,
        "mechanistic": run_mechanistic_pipeline,
    }
    fn = fns[fn_name]
    results_dir = Path(RESULTS_MOUNT) / results_subdir
    if fn_name == "mechanistic":
        fn(imagenet_root=Path(IMAGENET_MOUNT), results_dir=results_dir, num_images=num_images, batch_size=batch_size, device="cuda")
    elif fn_name == "vit":
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            skip_qual=skip_qual,
            **kwargs,
        )
    elif fn_name == "dinov2":
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            skip_qual=skip_qual,
        )
    else:
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            skip_qual=skip_qual,
        )
    results_vol.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_resnet50(num_images: int = 500, batch_size: int = 8, skip_qual: bool = False):
    _run_pipeline("resnet50", "resnet50", num_images, batch_size, skip_qual)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_vit(num_images: int = 500, batch_size: int = 8, skip_qual: bool = False):
    _run_pipeline("vit", "vit", num_images, batch_size, skip_qual)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_dinov2(num_images: int = 500, batch_size: int = 8, skip_qual: bool = False):
    _run_pipeline("dinov2", "dinov2", num_images, batch_size, skip_qual)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_mechanistic(num_images: int = 500, batch_size: int = 16, skip_qual: bool = False):
    _run_pipeline("mechanistic", "mechanistic", num_images, batch_size, skip_qual)


@app.local_entrypoint()
def main(
    experiment: str = "resnet50",
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
):
    """
    experiment: resnet50 | vit | dinov2 | mechanistic | all
    """
    experiments = {
        "resnet50": run_resnet50,
        "vit": run_vit,
        "dinov2": run_dinov2,
        "mechanistic": run_mechanistic,
    }
    if experiment == "all":
        for name in ["resnet50", "vit", "dinov2", "mechanistic"]:
            print("Launching", name)
            bs = 16 if name == "mechanistic" else batch_size
            experiments[name].remote(num_images=num_images, batch_size=bs, skip_qual=skip_qual)
        return
    if experiment not in experiments:
        raise ValueError("Unknown experiment: %s (use resnet50|vit|dinov2|mechanistic|all)" % experiment)
    bs = 16 if experiment == "mechanistic" else batch_size
    experiments[experiment].remote(num_images=num_images, batch_size=bs, skip_qual=skip_qual)
