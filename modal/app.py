"""
Modal cloud runner for PyTorch sanity-check experiments.

Usage (from repo root):
  modal run modal/app.py --experiment resnet50 --num-images 500 --skip-qual
  modal run modal/app.py --experiment occlusion --num-images 100
  modal run modal/app.py --experiment vit_gradcam_diagnostic --num-images 50
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
    """Cache timm weights on Modal at image build."""
    import timm

    timm.create_model("resnet50", pretrained=True)
    timm.create_model("vit_base_patch16_224", pretrained=True, img_size=224)
    print("Preloaded ResNet-50 and ViT-B/16.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-pytorch.txt"))
    .env({"PYTHONPATH": "/root/src"})
    .run_function(_preload_pretrained_weights)
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")  # v2
    .add_local_dir(str(REPO_ROOT / "diagnostics"), remote_path="/root/diagnostics")
)

volume_mounts = {IMAGENET_MOUNT: imagenet_vol, RESULTS_MOUNT: results_vol}

SALIENCY_ARCHS = ("resnet50", "vit")


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
    ig_steps: int = 50,
    **kwargs,
):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import (
        run_mechanistic_pipeline,
        run_resnet50_pipeline,
        run_vit_pipeline,
    )

    fns = {
        "resnet50": run_resnet50_pipeline,
        "vit": run_vit_pipeline,
        "mechanistic": run_mechanistic_pipeline,
    }
    fn = fns[fn_name]
    results_dir = Path(RESULTS_MOUNT) / results_subdir
    pipe_kw = dict(
        force_recompute=force_recompute,
        target_mode=target_mode,
        seed=seed,
        ig_baseline=ig_baseline,
        ig_steps=ig_steps,
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
    ig_steps: int = 50,
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
        ig_steps=ig_steps,
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
    ig_steps: int = 50,
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
        ig_steps=ig_steps,
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
    ig_steps: int = 50,
):
    _run_pipeline(
        "resnet50", "resnet50", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, target_mode=target_mode, seed=seed,
        ig_baseline=ig_baseline, ig_steps=ig_steps,
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
    ig_steps: int = 50,
):
    _run_pipeline(
        "vit", "vit", num_images, batch_size, skip_qual,
        force_recompute=force_recompute, target_mode=target_mode, seed=seed,
        ig_baseline=ig_baseline, ig_steps=ig_steps,
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
    ig_steps: int = 50,
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
        ig_steps,
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
    ig_steps: int = 50,
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
        ig_steps=ig_steps,
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_occlusion(
    arch: str,
    num_images: int = 500,
    batch_size: int = 8,
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: str = "zero",
    ig_steps: int = 50,
    occlusion_patch_fractions: str = "0.10,0.20,0.30",
    blur_type: str = "box",
    blur_sigma: float = 8.0,
):
    import sys

    sys.path.insert(0, "/root/src")
    from experiment_utils import run_occlusion_pipeline

    patch_fractions = [float(x.strip()) for x in occlusion_patch_fractions.split(",") if x.strip()]
    run_occlusion_pipeline(
        arch=arch,
        imagenet_root=Path(IMAGENET_MOUNT),
        results_dir=Path(RESULTS_MOUNT) / arch,
        num_images=num_images,
        batch_size=batch_size,
        device="cuda",
        force_recompute=force_recompute,
        seed=seed,
        ig_baseline=ig_baseline,
        ig_steps=ig_steps,
        patch_fractions=patch_fractions,
        blur_type=blur_type,
        blur_sigma=blur_sigma,
    )
    results_vol.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SEC,
    volumes=volume_mounts,
)
def run_vit_gradcam_diagnostic(
    num_images: int = 50,
    batch_size: int = 8,
    seed: int = 42,
):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "/root/diagnostics/choose_vit_gradcam_layer.py",
            "--imagenet-root",
            IMAGENET_MOUNT,
            "--num-images",
            str(num_images),
            "--batch-size",
            str(batch_size),
            "--device",
            "cuda",
            "--seed",
            str(seed),
            "--output-dir",
            str(Path(RESULTS_MOUNT) / "diagnostics"),
        ],
        check=True,
    )
    results_vol.commit()


def _launch_arch_parallel(
    arch: str,
    num_images: int,
    batch_size: int,
    force_recompute: bool = False,
    target_mode: str = "dynamic",
    seeds: list[int] | None = None,
    ig_baseline: str = "zero",
    ig_steps: int = 50,
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
                    ig_steps=ig_steps,
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


def _launch_occlusion(
    num_images: int,
    batch_size: int,
    force_recompute: bool,
    primary_seed: int,
    ig_baseline: str,
    ig_steps: int,
    occlusion_arch: str,
    occlusion_patch_fractions: str,
    blur_type: str,
    blur_sigma: float,
) -> list:
    archs = list(SALIENCY_ARCHS) if occlusion_arch == "all" else [occlusion_arch]
    return [
        run_occlusion.spawn(
            arch=arch,
            num_images=num_images,
            batch_size=batch_size,
            force_recompute=force_recompute,
            seed=primary_seed,
            ig_baseline=ig_baseline,
            ig_steps=ig_steps,
            occlusion_patch_fractions=occlusion_patch_fractions,
            blur_type=blur_type,
            blur_sigma=blur_sigma,
        )
        for arch in archs
    ]


def _launch_qual(
    arch: str,
    num_images: int,
    image_index: int,
    image_index_mode: str,
    force: bool,
    target_mode: str = "dynamic",
    seed: int = 42,
    ig_baseline: str = "zero",
    ig_steps: int = 50,
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
            ig_steps=ig_steps,
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
    ig_steps: int = 50,
    occlusion_arch: str = "all",
    occlusion_patch_fractions: str = "0.10,0.20,0.30",
    blur_type: str = "box",
    blur_sigma: float = 8.0,
):
    """
    experiment: resnet50 | vit | mechanistic | occlusion | vit_gradcam_diagnostic | all
    parallel_methods: one GPU per saliency method (default on)
    sequential: run full pipeline on a single GPU (opt-out of parallel_methods)
    qual_only: only build qual_bundle.npz (no quant recompute)
    image_index_mode: fixed | auto_ssim | auto_ssim_shared
      (auto_ssim uses one index per arch; shared / all+auto_ssim picks the same index for every arch)
    force_recompute: ignore cached spearman/baseline npy (full rerun)
    target_mode: dynamic (per-depth argmax) | frozen_baseline
    seeds: comma-separated RNG seeds; Class A methods run for each seed in seed subdirs
    ig_baseline: zero | mean (Integrated Gradients baseline)
    ig_steps: Integrated Gradients interpolation steps
    occlusion_arch: resnet50 | vit | all
    occlusion_patch_fractions: comma-separated fractions of 196 arch-native tiles (default 0.10,0.20,0.30)
    blur_type: box (Binder default) | gaussian (legacy ablation)
    blur_sigma: Gaussian sigma only; ignored when blur_type=box
    """
    if ig_baseline not in ("zero", "mean"):
        raise ValueError("ig_baseline must be 'zero' or 'mean'")
    if blur_type not in ("box", "gaussian"):
        raise ValueError("blur_type must be 'box' or 'gaussian'")

    seed_list = [seed] if seeds == "42" and seed != 42 else _parse_seeds(seeds)
    primary_seed = seed_list[0]
    use_parallel_methods = parallel_methods and not sequential

    experiments = {
        "resnet50": run_resnet50,
        "vit": run_vit,
        "mechanistic": run_mechanistic,
    }

    if occlusion_arch not in (*SALIENCY_ARCHS, "all"):
        raise ValueError("occlusion_arch must be resnet50|vit|all")

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
                "qual_only requires experiment resnet50|vit|all (not mechanistic|occlusion)"
            )
        handles = []
        for arch in archs:
            handles.extend(
                _launch_qual(
                    arch, num_images, image_index, image_index_mode, qual_force,
                    target_mode=target_mode, seed=primary_seed,
                    ig_baseline=ig_baseline, ig_steps=ig_steps,
                )
            )
        _wait_handles(handles)
        return

    if experiment == "occlusion":
        handles = _launch_occlusion(
            num_images,
            batch_size,
            force_recompute,
            primary_seed,
            ig_baseline,
            ig_steps,
            occlusion_arch,
            occlusion_patch_fractions,
            blur_type,
            blur_sigma,
        )
        _wait_handles(handles)
        return

    if experiment == "vit_gradcam_diagnostic":
        _wait_handles(
            [
                run_vit_gradcam_diagnostic.spawn(
                    num_images=num_images,
                    batch_size=batch_size,
                    seed=primary_seed,
                )
            ]
        )
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
                    ig_steps=ig_steps,
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
            ig_steps=ig_steps,
        )
        if skip_qual or name not in SALIENCY_ARCHS:
            return method_handles
        _wait_handles(method_handles)
        return _launch_qual(
            name, num_images, image_index, image_index_mode, qual_force,
            target_mode=target_mode, seed=primary_seed,
            ig_baseline=ig_baseline, ig_steps=ig_steps,
        )

    if experiment == "all":
        if use_parallel_methods:
            cascade_handles = []
            for name in SALIENCY_ARCHS:
                cascade_handles.extend(
                    _launch_arch_parallel(
                        name,
                        num_images,
                        batch_size,
                        force_recompute,
                        target_mode,
                        seeds=seed_list,
                        ig_baseline=ig_baseline,
                        ig_steps=ig_steps,
                    )
                )
            cascade_handles.append(
                experiments["mechanistic"].spawn(
                    num_images=num_images,
                    batch_size=16,
                    skip_qual=True,
                    force_recompute=force_recompute,
                    seed=primary_seed,
                )
            )
            _wait_handles(cascade_handles)
            if not skip_qual:
                qual_handles = []
                for name in SALIENCY_ARCHS:
                    qual_handles.extend(
                        _launch_qual(
                            name,
                            num_images,
                            image_index,
                            image_index_mode,
                            qual_force,
                            target_mode=target_mode,
                            seed=primary_seed,
                            ig_baseline=ig_baseline,
                            ig_steps=ig_steps,
                        )
                    )
                _wait_handles(qual_handles)
        else:
            all_handles = []
            for name in ["resnet50", "vit", "mechanistic"]:
                all_handles.extend(launch_arch(name))
            _wait_handles(all_handles)
        occlusion_handles = _launch_occlusion(
            num_images,
            batch_size,
            force_recompute,
            primary_seed,
            ig_baseline,
            ig_steps,
            occlusion_arch,
            occlusion_patch_fractions,
            blur_type,
            blur_sigma,
        )
        _wait_handles(occlusion_handles)
        return

    if experiment not in experiments:
        raise ValueError(
            "Unknown experiment: %s "
            "(use resnet50|vit|mechanistic|occlusion|vit_gradcam_diagnostic|all)"
            % experiment
        )

    _wait_handles(launch_arch(experiment))
