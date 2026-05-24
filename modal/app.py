"""
Modal cloud runner for PyTorch sanity-check experiments.

Usage (from repo root):
  modal run modal/app.py --experiment resnet50 --num-images 500 --skip-qual
  modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods
  modal run modal/app.py --experiment resnet50 --num-images 10 --sequential
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

SALIENCY_ARCHS = ("resnet50", "vit", "dinov2")


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


def _run_single_method(
    arch: str,
    method: str,
    num_images: int,
    batch_size: int,
    skip_qual: bool,
):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import run_arch_method_pipeline

    run_arch_method_pipeline(
        arch=arch,
        method=method,
        imagenet_root=Path(IMAGENET_MOUNT),
        results_dir=Path(RESULTS_MOUNT) / arch,
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


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_saliency_method(
    arch: str,
    method: str,
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = True,
):
    _run_single_method(arch, method, num_images, batch_size, skip_qual)


def _methods_for_arch(arch: str) -> list[str]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiment_utils import ARCH_SALIENCY_METHODS

    return list(ARCH_SALIENCY_METHODS[arch])


def _launch_arch_parallel(
    arch: str,
    num_images: int,
    batch_size: int,
    skip_qual: bool,
) -> list:
    methods = _methods_for_arch(arch)
    print("Launching %d parallel method jobs for %s: %s" % (len(methods), arch, methods))
    handles = [
        run_saliency_method.spawn(
            arch=arch,
            method=method,
            num_images=num_images,
            batch_size=batch_size,
            skip_qual=skip_qual,
        )
        for method in methods
    ]
    return handles


def _wait_handles(handles: list) -> None:
    for handle in handles:
        handle.get()


@app.local_entrypoint()
def main(
    experiment: str = "resnet50",
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
    parallel_methods: bool = True,
    sequential: bool = False,
):
    """
    experiment: resnet50 | vit | dinov2 | mechanistic | all
  parallel_methods: one GPU per saliency method (default on)
  sequential: run full pipeline on a single GPU (opt-out of parallel_methods)
    """
    use_parallel_methods = parallel_methods and not sequential

    experiments = {
        "resnet50": run_resnet50,
        "vit": run_vit,
        "dinov2": run_dinov2,
        "mechanistic": run_mechanistic,
    }

    def launch_arch(name: str) -> list:
        if name == "mechanistic" or not use_parallel_methods:
            bs = 16 if name == "mechanistic" else batch_size
            print("Launching sequential job:", name)
            return [experiments[name].spawn(num_images=num_images, batch_size=bs, skip_qual=skip_qual)]
        return _launch_arch_parallel(name, num_images, batch_size, skip_qual)

    if experiment == "all":
        all_handles = []
        for name in ["resnet50", "vit", "dinov2", "mechanistic"]:
            all_handles.extend(launch_arch(name))
        _wait_handles(all_handles)
        return

    if experiment not in experiments:
        raise ValueError("Unknown experiment: %s (use resnet50|vit|dinov2|mechanistic|all)" % experiment)

    _wait_handles(launch_arch(experiment))
