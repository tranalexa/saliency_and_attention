"""
Modal cloud runner for PyTorch sanity-check experiments.

Usage (from repo root):
  modal run modal/app.py --experiment resnet50 --num-images 500 --skip-qual
  modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods
  modal run modal/app.py --experiment resnet50 --num-images 10 --sequential
  modal run modal/app.py --experiment resnet50 --qual-only --image-index-mode auto_ssim
  modal run modal/app.py --experiment vit --num-images 500 --seeds 42,1,2 --parallel-methods
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

def _preload_pretrained_weights() -> None:
    """Cache hub/timm weights on Modal at image build (per Meta README)."""
    import timm
    import torch

    timm.create_model("resnet50", pretrained=True)
    timm.create_model("vit_base_patch16_224", pretrained=True, img_size=224)
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc", layers=1, pretrained=True)
    print("Preloaded ResNet-50, ViT-B/16, dinov2_vitb14_lc.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-pytorch.txt"))
    .env({"PYTHONPATH": "/root/src"})
    .run_function(_preload_pretrained_weights)
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
)

volume_mounts = {IMAGENET_MOUNT: imagenet_vol, RESULTS_MOUNT: results_vol}

SALIENCY_ARCHS = ("resnet50", "vit", "dinov2")


def _parse_seeds(seeds: str) -> list[int]:
    parsed = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    if not parsed:
        raise ValueError("At least one seed is required in --seeds")
    return parsed


def _methods_for_arch(arch: str) -> list[str]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiment_utils import METHODS_BY_ARCH

    return list(METHODS_BY_ARCH[arch])


def _is_class_a_method(method: str) -> bool:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiment_utils import METHODS_CLASS_A

    return method in METHODS_CLASS_A


def _results_subdir_for_method(arch: str, method: str, seed: int) -> str:
    if _is_class_a_method(method):
        return "%s/seed%d" % (arch, seed)
    return arch


def _run_pipeline(
    fn_name: str,
    results_subdir: str,
    num_images: int,
    batch_size: int,
    skip_qual: bool,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
    **kwargs,
):
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
    pipe_kw = dict(
        force_recompute=force_recompute,
        target_mode=target_mode,
        seed=seed,
        ig_baseline=ig_baseline,
    )
    if fn_name == "mechanistic":
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            force_recompute=force_recompute,
            seed=seed,
        )
    elif fn_name == "vit":
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            skip_qual=skip_qual,
            **pipe_kw,
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
            **pipe_kw,
        )
    else:
        fn(
            imagenet_root=Path(IMAGENET_MOUNT),
            results_dir=results_dir,
            num_images=num_images,
            batch_size=batch_size,
            device="cuda",
            skip_qual=skip_qual,
            **pipe_kw,
        )
    results_vol.commit()


def _run_single_method(
    arch: str,
    method: str,
    num_images: int,
    batch_size: int,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import run_arch_method_pipeline

    run_arch_method_pipeline(
        arch=arch,
        method=method,
        imagenet_root=Path(IMAGENET_MOUNT),
        results_dir=Path(RESULTS_MOUNT) / _results_subdir_for_method(arch, method, seed),
        num_images=num_images,
        batch_size=batch_size,
        device="cuda",
        force_recompute=force_recompute,
        target_mode=target_mode,
        seed=seed,
        ig_baseline=ig_baseline,
    )
    results_vol.commit()


def _run_qual(
    arch: str,
    num_images: int,
    image_index: int,
    image_index_mode: str,
    force: bool,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import run_qual_bundle_pipeline

    idx = run_qual_bundle_pipeline(
        arch=arch,
        imagenet_root=Path(IMAGENET_MOUNT),
        results_dir=Path(RESULTS_MOUNT) / arch,
        num_images=num_images,
        device="cuda",
        image_index=image_index,
        image_index_mode=image_index_mode,
        force=force,
        target_mode=target_mode,
        seed=seed,
        ig_baseline=ig_baseline,
    )
    print("qual_bundle for %s written (image_index=%d)" % (arch, idx))
    results_vol.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_resnet50(
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    _run_pipeline(
        "resnet50", "resnet50", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, target_mode=target_mode, seed=seed,
        ig_baseline=ig_baseline,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_vit(
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    _run_pipeline(
        "vit", "vit", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, target_mode=target_mode, seed=seed,
        ig_baseline=ig_baseline,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_dinov2(
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    _run_pipeline(
        "dinov2", "dinov2", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, target_mode=target_mode, seed=seed,
        ig_baseline=ig_baseline,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_mechanistic(
    num_images: int = 500,
    batch_size: int = 16,
    skip_qual: bool = False,
    force_recompute: bool = False,
    seed: int = 42,
):
    _run_pipeline(
        "mechanistic", "mechanistic", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, seed=seed,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_qual_bundle(
    arch: str,
    num_images: int = 500,
    image_index: int = 0,
    image_index_mode: str = "fixed",
    force: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    _run_qual(
        arch,
        num_images,
        image_index,
        image_index_mode,
        force,
        target_mode,
        seed,
        ig_baseline,
    )


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
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
):
    _run_single_method(
        arch,
        method,
        num_images,
        batch_size,
        force_recompute=force_recompute,
        target_mode=target_mode,
        seed=seed,
        ig_baseline=ig_baseline,
    )


def _launch_arch_parallel(
    arch: str,
    num_images: int,
    batch_size: int,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seeds: list[int] | None = None,
    ig_baseline: str = "zero",
) -> list:
    seeds = seeds or [42]
    methods = _methods_for_arch(arch)
    handles = []
    for method in methods:
        method_seeds = seeds if _is_class_a_method(method) else [seeds[0]]
        for seed in method_seeds:
            print(
                "Launching %s/%s seed=%d (Class A multi-seed=%s)"
                % (arch, method, seed, _is_class_a_method(method))
            )
            handles.append(
                run_saliency_method.spawn(
                    arch=arch,
                    method=method,
                    num_images=num_images,
                    batch_size=batch_size,
                    force_recompute=force_recompute,
                    target_mode=target_mode,
                    seed=seed,
                    ig_baseline=ig_baseline,
                )
            )
    print(
        "Launched %d parallel method jobs for %s across seeds %s"
        % (len(handles), arch, seeds)
    )
    return handles


def _wait_handles(handles: list) -> None:
    for handle in handles:
        handle.get()


def _launch_qual(
    arch: str,
    num_images: int,
    image_index: int,
    image_index_mode: str,
    force: bool,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
) -> list:
    print("Launching qual_bundle job for", arch)
    return [
        run_qual_bundle.spawn(
            arch=arch,
            num_images=num_images,
            image_index=image_index,
            image_index_mode=image_index_mode,
            force=force,
            target_mode=target_mode,
            seed=seed,
            ig_baseline=ig_baseline,
        )
    ]


@app.local_entrypoint()
def main(
    experiment: str = "resnet50",
    num_images: int = 500,
    batch_size: int = 8,
    skip_qual: bool = False,
    parallel_methods: bool = True,
    sequential: bool = False,
    qual_only: bool = False,
    image_index: int = 0,
    image_index_mode: str = "fixed",
    qual_force: bool = False,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seed: int = 42,
    seeds: str = "42",
    ig_baseline: str = "zero",
):
    """
    experiment: resnet50 | vit | dinov2 | mechanistic | all
    parallel_methods: one GPU per saliency method (default on)
    sequential: run full pipeline on a single GPU (opt-out of parallel_methods)
    qual_only: only build qual_bundle.npz (no quant recompute)
    image_index_mode: fixed | auto_ssim (pick demo image from existing SSIM arrays)
    force_recompute: ignore cached spearman/baseline npy (full rerun)
    target_mode: dynamic (per-depth argmax) | frozen_baseline
    seeds: comma-separated RNG seeds; Class A methods run for each seed in seed subdirs
    ig_baseline: zero | mean (Integrated Gradients baseline)
    """
    if ig_baseline not in ("zero", "mean"):
        raise ValueError("ig_baseline must be 'zero' or 'mean'")

    seed_list = [seed] if seeds == "42" and seed != 42 else _parse_seeds(seeds)
    primary_seed = seed_list[0]
    use_parallel_methods = parallel_methods and not sequential

    experiments = {
        "resnet50": run_resnet50,
        "vit": run_vit,
        "dinov2": run_dinov2,
        "mechanistic": run_mechanistic,
    }

    def qual_archs_for_experiment(name: str) -> list[str]:
        if name == "all":
            return list(SALIENCY_ARCHS)
        if name in SALIENCY_ARCHS:
            return [name]
        return []

    if qual_only:
        archs = qual_archs_for_experiment(experiment)
        if not archs:
            raise ValueError(
                "qual_only requires experiment resnet50|vit|dinov2|all (not mechanistic)"
            )
        handles = []
        for arch in archs:
            handles.extend(
                _launch_qual(
                    arch, num_images, image_index, image_index_mode, qual_force,
                    target_mode=target_mode, seed=primary_seed,
                    ig_baseline=ig_baseline,
                )
            )
        _wait_handles(handles)
        return

    def launch_arch(name: str) -> list:
        if name == "mechanistic" or not use_parallel_methods:
            bs = 16 if name == "mechanistic" else batch_size
            run_seed = primary_seed
            if not use_parallel_methods and len(seed_list) > 1 and name != "mechanistic":
                print(
                    "Note: sequential mode uses seed=%d only; "
                    "use --parallel-methods for multi-seed Class A runs."
                    % run_seed
                )
            print("Launching sequential job:", name)
            if name == "mechanistic":
                return [
                    experiments[name].spawn(
                        num_images=num_images,
                        batch_size=bs,
                        skip_qual=skip_qual,
                        force_recompute=force_recompute,
                        seed=run_seed,
                    )
                ]
            return [
                experiments[name].spawn(
                    num_images=num_images,
                    batch_size=bs,
                    skip_qual=skip_qual,
                    force_recompute=force_recompute,
                    target_mode=target_mode,
                    seed=run_seed,
                    ig_baseline=ig_baseline,
                )
            ]

        method_handles = _launch_arch_parallel(
            name,
            num_images,
            batch_size,
            force_recompute,
            target_mode,
            seeds=seed_list,
            ig_baseline=ig_baseline,
        )
        if skip_qual or name not in SALIENCY_ARCHS:
            return method_handles
        _wait_handles(method_handles)
        return _launch_qual(
            name, num_images, image_index, image_index_mode, qual_force,
            target_mode=target_mode, seed=primary_seed,
            ig_baseline=ig_baseline,
        )

    if experiment == "all":
        all_handles = []
        for name in ["resnet50", "vit", "dinov2", "mechanistic"]:
            all_handles.extend(launch_arch(name))
        _wait_handles(all_handles)
        return

    if experiment not in experiments:
        raise ValueError("Unknown experiment: %s (use resnet50|vit|dinov2|mechanistic|all)" % experiment)

    _wait_handles(launch_arch(experiment))
