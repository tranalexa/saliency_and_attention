"""Shared experiment logic for cascading sanity checks (notebooks + Modal)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
import timm

from attention_utils import get_raw_attention, get_rollout
from metrics_utils import (
    abs_grayscale_norm,
    compute_logit_correlation,
    compute_spearman,
    compute_ssim,
    diverging_norm,
)
from viz_utils import pick_qual_image_index
from randomize_utils import (
    get_resnet_block_names,
    get_vit_block_names,
    reset_layer,
    restore_checkpoint,
    save_checkpoint,
)

TargetMode = Literal["dynamic", "frozen_baseline"]
MapNorm = Literal["abs", "diverging"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Cross-architecture method sets (Adebayo-style cascading sanity checks).
SHARED_SALIENCY_METHODS = [
    "gradient",
    "smoothgrad",
    "input_grad",
    "ig",
    "gradcam",
]

# GBP / GBP-GC use Guided Backprop (ReLU-specific); ResNet only.
RESNET50_SALIENCY_METHODS = SHARED_SALIENCY_METHODS + ["gbp", "gbp_gc"]

# ViT / DINOv2: shared Captum methods + attention (not in original Adebayo paper).
VIT_SALIENCY_METHODS = SHARED_SALIENCY_METHODS + ["raw_attn", "rollout"]

ARCH_SALIENCY_METHODS = {
    "resnet50": RESNET50_SALIENCY_METHODS,
    "vit": VIT_SALIENCY_METHODS,
    "dinov2": VIT_SALIENCY_METHODS,
}

METHOD_BATCH_CAPS = {
    "ig": 2,
    "smoothgrad": 2,
    "gbp_gc": 2,
    "gbp": 4,
    "gradcam": 4,
}

IMAGENET_UPLOAD_HINT = (
    "ImageNet not found. Options:\n"
    "  1) Download inside Modal: modal run modal/download_imagenet.py "
    "--val-tar-url URL --devkit-tar-url URL\n"
    "  2) Upload from laptop: modal volume put saliency-imagenet /path/to/imagenet/val /val\n"
    "  3) Local: set IMAGENET_ROOT to a directory containing val/"
)


def validate_imagenet_root(imagenet_root: Path) -> None:
    """Ensure ImageNet validation data is available."""
    root = Path(imagenet_root)
    if not root.exists():
        raise FileNotFoundError(IMAGENET_UPLOAD_HINT)
    val_dir = root / "val"
    if not val_dir.exists() and not any(root.iterdir()):
        raise FileNotFoundError(IMAGENET_UPLOAD_HINT)


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_imagenet_subset(
    root: str | Path,
    num_images: int = 500,
    split: str = "val",
    image_size: int = 224,
    transform=None,
) -> Tuple[Subset, List[int]]:
    transform = transform or build_transform(image_size)
    root = Path(root)
    val_dir = root / split
    if val_dir.exists():
        try:
            dataset = datasets.ImageNet(root=str(root), split=split, transform=transform)
        except Exception:
            dataset = ImageFolder(str(val_dir), transform=transform)
    else:
        dataset = ImageFolder(str(root), transform=transform)
    indices = list(range(min(num_images, len(dataset))))
    return Subset(dataset, indices), indices


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_experiment_config(results_dir: Path, **fields) -> None:
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(fields, f, indent=2)


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _method_batch_size(method: str, batch_size: int) -> int:
    cap = METHOD_BATCH_CAPS.get(method, batch_size)
    return max(1, min(batch_size, cap))


def attribution_to_map(
    attr: torch.Tensor,
    out_size: int = 224,
    grid_size: int | None = None,
    norm: MapNorm = "abs",
) -> np.ndarray:
    if attr.dim() == 4:
        attr = attr.sum(dim=1)
    elif attr.dim() == 3 and grid_size is not None:
        tok = attr[0] if attr.shape[0] == 1 else attr
        if tok.shape[0] == grid_size * grid_size + 1:
            tok = tok[1:]
        if tok.shape[0] == grid_size * grid_size:
            if norm == "abs":
                arr = tok.abs().mean(dim=-1).reshape(grid_size, grid_size).detach().cpu().numpy()
            else:
                arr = tok.mean(dim=-1).reshape(grid_size, grid_size).detach().cpu().numpy()
            t = torch.from_numpy(arr).float()[None, None]
            up = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
            out = up.squeeze().numpy()
            return abs_grayscale_norm(out) if norm == "abs" else diverging_norm(out)
    arr = attr.squeeze().detach().cpu().numpy()
    if arr.ndim == 1:
        n_patches = grid_size * grid_size if grid_size is not None else None
        if n_patches is not None and arr.size == n_patches + 1:
            arr = arr[1:]
        if n_patches is not None and arr.size == n_patches:
            arr = arr.reshape(grid_size, grid_size)
        else:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError(
                    "Cannot reshape attribution of size %d to a square grid"
                    % arr.size
                )
            arr = arr.reshape(side, side)
    t = torch.from_numpy(arr.astype(np.float64)).float()[None, None]
    up = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    out = up.squeeze().numpy()
    if norm == "abs":
        return abs_grayscale_norm(np.abs(out))
    return diverging_norm(out)


def compute_ig(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    n_steps: int = 50,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = ig.attribute(inp, baselines=torch.zeros_like(inp), target=tgt, n_steps=n_steps)
        maps.append(attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def compute_gradient(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import Saliency

    sal = Saliency(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = sal.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def compute_smoothgrad(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    stdev: float = 0.15,
    n_samples: int = 25,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import NoiseTunnel, Saliency

    sal = Saliency(model)
    nt = NoiseTunnel(sal)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = nt.attribute(
            inp,
            target=tgt,
            nt_type="smoothgrad",
            stdevs=stdev,
            nt_samples=n_samples,
            nt_samples_batch_size=1,
        )
        maps.append(attribution_to_map(attr, norm=map_norm))
        _clear_cuda()
    return np.stack(maps)


def compute_input_grad(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import InputXGradient

    ixg = InputXGradient(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = ixg.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def compute_gbp_gc(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    layer: nn.Module,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    gbp_maps = compute_gbp(model, images, target_indices, map_norm=map_norm)
    gc_maps = compute_gradcam(
        model, images, target_indices, layer, map_norm=map_norm
    )
    return gbp_maps * gc_maps


def compute_gbp(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import GuidedBackprop

    gbp = GuidedBackprop(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = gbp.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def compute_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    layer: nn.Module,
    grid_size: int | None = None,
    map_norm: MapNorm = "abs",
) -> np.ndarray:
    from captum.attr import LayerGradCam

    cam = LayerGradCam(model, layer)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = cam.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr, grid_size=grid_size, norm=map_norm))
    return np.stack(maps)


@torch.no_grad()
def get_target_indices(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images).argmax(dim=1)


def load_all_images(
    loader: DataLoader, model: nn.Module, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_images, all_targets = [], []
    for images, _ in tqdm(loader, desc="images"):
        all_images.append(images)
        all_targets.append(get_target_indices(model, images.to(device)).cpu())
    return torch.cat(all_images), torch.cat(all_targets)


def _method_baseline_path(results_dir: Path, method: str) -> Path:
    return results_dir / ("baseline_%s.npz" % method)


def _method_baseline_exists(results_dir: Path, method: str) -> bool:
    path = _method_baseline_path(results_dir, method)
    if not path.exists():
        legacy = results_dir / "baseline_maps.npz"
        if legacy.exists():
            return ("baseline_" + method) in np.load(legacy).files
        return False
    data = np.load(path)
    return "baseline_map" in data.files and "baseline_map_div" in data.files


def load_baseline_maps(
    results_dir: Path, method: str, norm: MapNorm = "abs"
) -> np.ndarray:
    key = "baseline_map" if norm == "abs" else "baseline_map_div"
    per_method = _method_baseline_path(results_dir, method)
    if per_method.exists():
        data = np.load(per_method)
        if key in data.files:
            return data[key]
        if norm == "abs" and "baseline_map" in data.files:
            return data["baseline_map"]
    legacy = results_dir / "baseline_maps.npz"
    if legacy.exists():
        data = np.load(legacy)
        legacy_key = "baseline_" + method
        if legacy_key in data.files:
            return data[legacy_key]
    raise FileNotFoundError(
        "Baseline maps for method %r (norm=%s) not found under %s"
        % (method, norm, results_dir)
    )


def compute_baseline_maps(
    model: nn.Module,
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    compute_fn_div: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    batch_size: int,
    device: str,
    force_recompute: bool = False,
) -> None:
    if force_recompute:
        missing = list(methods)
    else:
        missing = [m for m in methods if not _method_baseline_exists(results_dir, m)]
    if not missing:
        return
    for method in tqdm(missing, desc="baseline"):
        batch_maps, batch_maps_div = [], []
        method_bs = _method_batch_size(method, batch_size)
        for start in range(0, len(all_images), method_bs):
            batch = all_images[start : start + method_bs].to(device)
            tgt = all_targets[start : start + method_bs].to(device)
            batch_maps.append(compute_fn(method, batch, tgt))
            batch_maps_div.append(compute_fn_div(method, batch, tgt))
            _clear_cuda()
        np.savez_compressed(
            _method_baseline_path(results_dir, method),
            baseline_map=np.concatenate(batch_maps, axis=0),
            baseline_map_div=np.concatenate(batch_maps_div, axis=0),
        )


def _batch_targets(
    model: nn.Module,
    batch: torch.Tensor,
    all_targets: torch.Tensor,
    start: int,
    target_mode: TargetMode,
    device: str,
) -> torch.Tensor:
    if target_mode == "dynamic":
        return get_target_indices(model, batch.to(device))
    return all_targets[start : start + batch.shape[0]].to(device)


def run_cascading_experiment(
    model: nn.Module,
    order: List[str],
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    compute_fn_div: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    batch_size: int,
    device: str,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
) -> None:
    original_sd = save_checkpoint(model)
    num_depths = len(order)
    n_images = all_images.shape[0]

    for method in methods:
        spearman_path = results_dir / ("%s_spearman.npy" % method)
        if spearman_path.exists() and not force_recompute:
            print("Skip", method)
            continue
        baseline_maps = load_baseline_maps(results_dir, method, norm="abs")
        baseline_maps_div = load_baseline_maps(results_dir, method, norm="diverging")
        spearman_all = np.full((num_depths, n_images), np.nan)
        ssim_all = np.full((num_depths, n_images), np.nan)
        spearman_div = np.full((num_depths, n_images), np.nan)
        ssim_div = np.full((num_depths, n_images), np.nan)
        for depth, _layer_name in enumerate(tqdm(order, desc=method)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            rand_maps, rand_maps_div = [], []
            method_bs = _method_batch_size(method, batch_size)
            for start in range(0, n_images, method_bs):
                batch = all_images[start : start + method_bs].to(device)
                tgt = _batch_targets(model, batch, all_targets, start, target_mode, device)
                rand_maps.append(compute_fn(method, batch, tgt))
                rand_maps_div.append(compute_fn_div(method, batch, tgt))
                _clear_cuda()
            rand_maps = np.concatenate(rand_maps, axis=0)
            rand_maps_div = np.concatenate(rand_maps_div, axis=0)
            for i in range(n_images):
                spearman_all[depth, i] = compute_spearman(baseline_maps[i], rand_maps[i])
                ssim_all[depth, i] = compute_ssim(baseline_maps[i], rand_maps[i])
                spearman_div[depth, i] = compute_spearman(
                    baseline_maps_div[i], rand_maps_div[i]
                )
                ssim_div[depth, i] = compute_ssim(baseline_maps_div[i], rand_maps_div[i])
        np.save(spearman_path, spearman_all)
        np.save(results_dir / ("%s_ssim.npy" % method), ssim_all)
        np.save(results_dir / ("%s_spearman_mean.npy" % method), np.nanmean(spearman_all, axis=1))
        np.save(results_dir / ("%s_ssim_mean.npy" % method), np.nanmean(ssim_all, axis=1))
        np.save(results_dir / ("%s_spearman_div.npy" % method), spearman_div)
        np.save(results_dir / ("%s_ssim_div.npy" % method), ssim_div)
        np.save(
            results_dir / ("%s_spearman_div_mean.npy" % method),
            np.nanmean(spearman_div, axis=1),
        )
        np.save(results_dir / ("%s_ssim_div_mean.npy" % method), np.nanmean(ssim_div, axis=1))


def _qual_has_all_methods(qual_path: Path, methods: Sequence[str]) -> bool:
    if not qual_path.exists():
        return False
    data = np.load(qual_path, allow_pickle=True)
    return all(("baseline_" + m) in data.files and ("cascade_" + m) in data.files for m in methods)


def build_qual_bundle(
    model: nn.Module,
    order: List[str],
    dataset: Subset,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    device: str,
    image_index: int = 0,
    force: bool = False,
    target_mode: TargetMode = "dynamic",
) -> None:
    qual_path = results_dir / "qual_bundle.npz"
    methods = list(methods)
    existing: dict = {}
    if qual_path.exists() and not force:
        data = np.load(qual_path, allow_pickle=True)
        if _qual_has_all_methods(qual_path, methods):
            if "image_index" in data.files and int(data["image_index"]) == image_index:
                return
        existing = {k: data[k] for k in data.files}
        if "image_index" in existing and int(existing["image_index"]) != image_index:
            force = True
            existing = {}

    methods_to_compute = methods if force else [m for m in methods if ("baseline_" + m) not in existing]
    if not methods_to_compute and existing and not force:
        return

    original_sd = save_checkpoint(model)
    img, _ = dataset[image_index]
    inp = img[None].to(device)
    baseline_tgt = get_target_indices(model, inp)
    out = dict(existing) if existing and not force else {}
    out.update(
        {
            "image": denormalize(inp).squeeze(0).permute(1, 2, 0).cpu().numpy(),
            "order": np.array(order, dtype=object),
            "image_index": np.int64(image_index),
            "target_index": np.int64(int(baseline_tgt[0].item())),
            "target_mode": np.array(target_mode),
        }
    )
    for method in methods_to_compute:
        out["baseline_" + method] = compute_fn(method, inp, baseline_tgt)[0]
        cascade = []
        for depth in range(len(order)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            if target_mode == "dynamic":
                tgt = get_target_indices(model, inp)
            else:
                tgt = baseline_tgt
            cascade.append(compute_fn(method, inp, tgt)[0])
        out["cascade_" + method] = np.stack(cascade)
    np.savez_compressed(qual_path, **out)


def _setup_arch_model(
    arch: str,
    imagenet_root: Path,
    num_images: int,
    image_size: int,
    device: str,
    ig_steps: int,
    smoothgrad_stdev: float,
    smoothgrad_samples: int,
) -> Tuple[nn.Module, List[str], Subset, Callable[[str, torch.Tensor, torch.Tensor], np.ndarray], List[str]]:
    """Load model, dataset subset, and saliency compute_fn for an architecture."""
    transform = build_transform(image_size)
    dataset, _ = load_imagenet_subset(imagenet_root, num_images, transform=transform)

    if arch == "resnet50":
        model = timm.create_model("resnet50", pretrained=True).to(device).eval()
        order = get_resnet_block_names(model)
        gradcam_layer = model.layer4[-1].conv3
        methods = list(RESNET50_SALIENCY_METHODS)
        compute_fn = make_saliency_compute_fn(
            model,
            gradcam_layer,
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
        )
    elif arch == "vit":
        model_name = "vit_base_patch16_224"
        grid_size = 14
        model = timm.create_model(model_name, pretrained=True, img_size=image_size).to(device).eval()
        order = get_vit_block_names(model)
        gradcam_layer = model.patch_embed.proj
        methods = list(VIT_SALIENCY_METHODS)
        compute_fn = make_saliency_compute_fn(
            model,
            gradcam_layer,
            grid_size=grid_size,
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
            attention_grid_size=grid_size,
        )
    elif arch == "dinov2":
        model_name = "vit_base_patch14_dinov2.lvd142m"
        grid_size = 16
        model = timm.create_model(
            model_name, pretrained=True, img_size=image_size, num_classes=1000
        ).to(device).eval()
        order = get_vit_block_names(model)
        gradcam_layer = model.patch_embed.proj
        methods = list(VIT_SALIENCY_METHODS)
        compute_fn = make_saliency_compute_fn(
            model,
            gradcam_layer,
            grid_size=grid_size,
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
            attention_grid_size=grid_size,
        )
    else:
        raise ValueError("Unknown arch: %s (use resnet50|vit|dinov2)" % arch)
    return model, order, dataset, compute_fn, methods


def run_qual_bundle_pipeline(
    arch: str,
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    device: str = "cuda",
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    image_index: int = 0,
    image_index_mode: str = "fixed",
    auto_ssim_method: str = "ig",
    force: bool = False,
    target_mode: TargetMode = "dynamic",
    seed: int = 42,
) -> int:
    """Build qual_bundle.npz for paper cascade figures. Returns image index used."""
    if arch not in ARCH_SALIENCY_METHODS:
        raise ValueError("Unknown arch: %s" % arch)
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    if image_index_mode == "auto_ssim":
        image_index = pick_qual_image_index(
            results_dir, method=auto_ssim_method, fallback=image_index
        )
    elif image_index_mode != "fixed":
        raise ValueError("image_index_mode must be 'fixed' or 'auto_ssim'")

    model, order, dataset, compute_fn, methods = _setup_arch_model(
        arch,
        imagenet_root,
        num_images,
        image_size,
        device,
        ig_steps,
        smoothgrad_stdev,
        smoothgrad_samples,
    )
    build_qual_bundle(
        model,
        order,
        dataset,
        methods,
        compute_fn,
        results_dir,
        device,
        image_index=image_index,
        force=force,
        target_mode=target_mode,
    )
    return image_index


def reduce_activation_scales(t: torch.Tensor) -> torch.Tensor:
    """Mean |activation| per channel/feature over batch and spatial/token dims."""
    t = t.abs().float()
    if t.dim() == 4:
        return t.mean(dim=(0, 2, 3))
    if t.dim() == 3:
        return t.mean(dim=(0, 1))
    if t.dim() == 2:
        return t.mean(dim=0)
    return t.mean().reshape(1)


def run_mechanistic(
    model_name: str,
    arch_tag: str,
    order_fn: Callable[[nn.Module], List[str]],
    act_reduce: Callable[[torch.Tensor], torch.Tensor],
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "cuda",
    model_kwargs: dict | None = None,
    force_recompute: bool = False,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(image_size)
    dataset, _ = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    create_kwargs = {"pretrained": True, **(model_kwargs or {})}
    model = timm.create_model(model_name, **create_kwargs).to(device).eval()
    order = order_fn(model)
    original_sd = save_checkpoint(model)

    logit_path = results_dir / ("logit_corr_%s.npy" % arch_tag)
    if force_recompute or not logit_path.exists():
        orig_logits = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="orig logits " + arch_tag):
                orig_logits.append(model(images.to(device)).detach().cpu().numpy())
        orig_logits = np.concatenate(orig_logits, axis=0)
        logit_corr = []
        for depth, _ in enumerate(tqdm(order, desc="logit " + arch_tag)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            depth_logits = []
            with torch.no_grad():
                for images, _ in loader:
                    depth_logits.append(model(images.to(device)).detach().cpu().numpy())
            logit_corr.append(compute_logit_correlation(orig_logits, np.concatenate(depth_logits)))
        np.save(logit_path, np.array(logit_corr))

    for depth in range(len(order)):
        act_path = results_dir / ("activation_scale_%s_depth%02d.npy" % (arch_tag, depth))
        if (act_path.exists() and not force_recompute) or depth + 1 >= len(order):
            continue
        hook_layer = order[depth + 1]
        restore_checkpoint(model, original_sd)
        for name in order[: depth + 1]:
            reset_layer(model, name)
        activations = []

        def hook_fn(_m, _i, out):
            activations.append(out.detach())

        handle = model.get_submodule(hook_layer).register_forward_hook(hook_fn)
        with torch.no_grad():
            for images, _ in loader:
                model(images.to(device))
        handle.remove()
        scales = act_reduce(torch.cat(activations, dim=0)).cpu().numpy()
        np.save(act_path, scales)

    with open(results_dir / ("randomization_order_%s.json" % arch_tag), "w") as f:
        json.dump(order, f)


def make_saliency_compute_fn(
    model: nn.Module,
    gradcam_layer: nn.Module,
    *,
    grid_size: int | None = None,
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    attention_grid_size: int | None = None,
    map_norm: MapNorm = "abs",
) -> Callable[[str, torch.Tensor, torch.Tensor], np.ndarray]:
    """Dispatch saliency method name -> batched attribution maps."""

    def compute_fn(method: str, batch: torch.Tensor, tgt: torch.Tensor) -> np.ndarray:
        if method == "gradient":
            return compute_gradient(model, batch, tgt, map_norm=map_norm)
        if method == "smoothgrad":
            return compute_smoothgrad(
                model,
                batch,
                tgt,
                stdev=smoothgrad_stdev,
                n_samples=smoothgrad_samples,
                map_norm=map_norm,
            )
        if method == "input_grad":
            return compute_input_grad(model, batch, tgt, map_norm=map_norm)
        if method == "gbp":
            return compute_gbp(model, batch, tgt, map_norm=map_norm)
        if method == "gradcam":
            return compute_gradcam(
                model, batch, tgt, gradcam_layer, grid_size=grid_size, map_norm=map_norm
            )
        if method == "gbp_gc":
            return compute_gbp_gc(model, batch, tgt, gradcam_layer, map_norm=map_norm)
        if method == "ig":
            return compute_ig(model, batch, tgt, ig_steps, map_norm=map_norm)
        if attention_grid_size is not None:
            if method == "raw_attn":
                return np.stack(
                    [
                        get_raw_attention(model, batch[i : i + 1], attention_grid_size)
                        for i in range(batch.shape[0])
                    ]
                )
            if method == "rollout":
                return np.stack(
                    [
                        get_rollout(model, batch[i : i + 1], attention_grid_size)
                        for i in range(batch.shape[0])
                    ]
                )
        raise ValueError("Unknown saliency method: %s" % method)

    return compute_fn


def _ensure_shared_metadata(
    results_dir: Path,
    image_indices: List[int],
    all_targets: torch.Tensor,
    order: List[str],
    arch: str,
    methods: Sequence[str],
    target_mode: TargetMode,
    seed: int,
    num_images: int,
) -> None:
    np.save(results_dir / "image_indices.npy", np.array(image_indices))
    np.save(results_dir / "target_indices.npy", all_targets.numpy())
    with open(results_dir / "randomization_order.json", "w") as f:
        json.dump(order, f, indent=2)
    resnet_randomization = "block" if arch == "resnet50" else "n/a"
    write_experiment_config(
        results_dir,
        arch=arch,
        methods=list(methods),
        target_mode=target_mode,
        resnet_randomization=resnet_randomization,
        seed=seed,
        num_images=num_images,
        randomization_order=order,
    )


def validate_dinov2_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_probe: int = 10,
    min_top1_acc: float = 0.05,
    min_mean_confidence: float = 0.05,
) -> None:
    """Fail fast if DINOv2 ImageNet head looks untrained."""
    correct, conf_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            conf_sum += probs.max(dim=1).values.sum().item()
            correct += (preds == labels).sum().item()
            n += images.shape[0]
            if n >= num_probe:
                break
    if n == 0:
        raise RuntimeError("DINOv2 validation probe: empty loader.")
    acc = correct / n
    mean_conf = conf_sum / n
    if acc < min_top1_acc or mean_conf < min_mean_confidence:
        raise RuntimeError(
            "DINOv2 classifier probe failed (acc=%.3f, mean_conf=%.3f). "
            "Load a trained ImageNet linear head before large runs."
            % (acc, mean_conf)
        )


def run_arch_method_pipeline(
    arch: str,
    method: str,
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    """Run cascading sanity check for a single architecture + method (parallel Modal workers)."""
    if arch not in ARCH_SALIENCY_METHODS:
        raise ValueError("Unknown arch: %s (use resnet50|vit|dinov2)" % arch)
    allowed = ARCH_SALIENCY_METHODS[arch]
    if method not in allowed:
        raise ValueError("Method %r not valid for arch %r (allowed: %s)" % (method, arch, allowed))

    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(image_size)
    dataset, image_indices = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if arch == "resnet50":
        model = timm.create_model("resnet50", pretrained=True).to(device).eval()
        order = get_resnet_block_names(model)
        gradcam_layer = model.layer4[-1].conv3
        fn_kw = dict(
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
        )
    elif arch == "vit":
        model_name = "vit_base_patch16_224"
        grid_size = 14
        model = timm.create_model(model_name, pretrained=True, img_size=image_size).to(device).eval()
        order = get_vit_block_names(model)
        gradcam_layer = model.patch_embed.proj
        fn_kw = dict(
            grid_size=grid_size,
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
            attention_grid_size=grid_size,
        )
    elif arch == "dinov2":
        model_name = "vit_base_patch14_dinov2.lvd142m"
        grid_size = 16
        model = timm.create_model(
            model_name, pretrained=True, img_size=image_size, num_classes=1000
        ).to(device).eval()
        if num_images >= 50:
            validate_dinov2_classifier(model, loader, device)
        order = get_vit_block_names(model)
        gradcam_layer = model.patch_embed.proj
        fn_kw = dict(
            grid_size=grid_size,
            ig_steps=ig_steps,
            smoothgrad_stdev=smoothgrad_stdev,
            smoothgrad_samples=smoothgrad_samples,
            attention_grid_size=grid_size,
        )
    else:
        raise ValueError("Unknown arch: %s" % arch)

    compute_fn = make_saliency_compute_fn(model, gradcam_layer, map_norm="abs", **fn_kw)
    compute_fn_div = make_saliency_compute_fn(model, gradcam_layer, map_norm="diverging", **fn_kw)

    all_images, all_targets = load_all_images(loader, model, device)
    _ensure_shared_metadata(
        results_dir,
        image_indices,
        all_targets,
        order,
        arch,
        [method],
        target_mode,
        seed,
        num_images,
    )
    compute_baseline_maps(
        model,
        all_images,
        all_targets,
        [method],
        compute_fn,
        compute_fn_div,
        results_dir,
        batch_size,
        device,
        force_recompute=force_recompute,
    )
    run_cascading_experiment(
        model,
        order,
        all_images,
        all_targets,
        [method],
        compute_fn,
        compute_fn_div,
        results_dir,
        batch_size,
        device,
        target_mode=target_mode,
        force_recompute=force_recompute,
    )


def run_resnet50_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    skip_qual: bool = False,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = list(RESNET50_SALIENCY_METHODS)

    transform = build_transform(image_size)
    dataset, image_indices = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = timm.create_model("resnet50", pretrained=True).to(device).eval()
    order = get_resnet_block_names(model)
    gradcam_layer = model.layer4[-1].conv3
    fn_kw = dict(
        ig_steps=ig_steps,
        smoothgrad_stdev=smoothgrad_stdev,
        smoothgrad_samples=smoothgrad_samples,
    )
    compute_fn = make_saliency_compute_fn(model, gradcam_layer, map_norm="abs", **fn_kw)
    compute_fn_div = make_saliency_compute_fn(model, gradcam_layer, map_norm="diverging", **fn_kw)

    all_images, all_targets = load_all_images(loader, model, device)
    _ensure_shared_metadata(
        results_dir, image_indices, all_targets, order, "resnet50", methods, target_mode, seed, num_images
    )
    compute_baseline_maps(
        model, all_images, all_targets, methods, compute_fn, compute_fn_div,
        results_dir, batch_size, device, force_recompute=force_recompute,
    )
    run_cascading_experiment(
        model, order, all_images, all_targets, methods, compute_fn, compute_fn_div,
        results_dir, batch_size, device, target_mode=target_mode, force_recompute=force_recompute,
    )
    if not skip_qual:
        build_qual_bundle(
            model, order, dataset, methods, compute_fn, results_dir, device, target_mode=target_mode
        )


def run_vit_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    model_name: str = "vit_base_patch16_224",
    grid_size: int = 14,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    skip_qual: bool = False,
    use_attention: bool = True,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = list(VIT_SALIENCY_METHODS) if use_attention else list(SHARED_SALIENCY_METHODS)
    arch_tag = "dinov2" if "dinov2" in model_name else "vit"

    transform = build_transform(image_size)
    dataset, image_indices = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model_kwargs = {"pretrained": True, "img_size": image_size}
    if "dinov2" in model_name:
        model_kwargs["num_classes"] = 1000
    model = timm.create_model(model_name, **model_kwargs).to(device).eval()
    if "dinov2" in model_name and num_images >= 50:
        validate_dinov2_classifier(model, loader, device)
    order = get_vit_block_names(model)
    gradcam_layer = model.patch_embed.proj
    fn_kw = dict(
        grid_size=grid_size,
        ig_steps=ig_steps,
        smoothgrad_stdev=smoothgrad_stdev,
        smoothgrad_samples=smoothgrad_samples,
        attention_grid_size=grid_size if use_attention else None,
    )
    compute_fn = make_saliency_compute_fn(model, gradcam_layer, map_norm="abs", **fn_kw)
    compute_fn_div = make_saliency_compute_fn(model, gradcam_layer, map_norm="diverging", **fn_kw)

    all_images, all_targets = load_all_images(loader, model, device)
    _ensure_shared_metadata(
        results_dir, image_indices, all_targets, order, arch_tag, methods, target_mode, seed, num_images
    )
    compute_baseline_maps(
        model, all_images, all_targets, methods, compute_fn, compute_fn_div,
        results_dir, batch_size, device, force_recompute=force_recompute,
    )
    run_cascading_experiment(
        model, order, all_images, all_targets, methods, compute_fn, compute_fn_div,
        results_dir, batch_size, device, target_mode=target_mode, force_recompute=force_recompute,
    )
    if not skip_qual:
        build_qual_bundle(
            model, order, dataset, methods, compute_fn, results_dir, device, target_mode=target_mode
        )


def run_dinov2_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    smoothgrad_stdev: float = 0.15,
    smoothgrad_samples: int = 25,
    skip_qual: bool = False,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    run_vit_pipeline(
        imagenet_root=imagenet_root,
        results_dir=results_dir,
        model_name="vit_base_patch14_dinov2.lvd142m",
        grid_size=16,
        num_images=num_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        ig_steps=ig_steps,
        smoothgrad_stdev=smoothgrad_stdev,
        smoothgrad_samples=smoothgrad_samples,
        skip_qual=skip_qual,
        use_attention=True,
        target_mode=target_mode,
        force_recompute=force_recompute,
        seed=seed,
    )


def run_mechanistic_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "cuda",
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_mechanistic(
        "resnet50", "resnet", get_resnet_block_names,
        reduce_activation_scales,
        imagenet_root, results_dir, num_images, image_size, batch_size, device,
        force_recompute=force_recompute,
    )
    run_mechanistic(
        "vit_base_patch16_224", "vit", get_vit_block_names,
        reduce_activation_scales,
        imagenet_root, results_dir, num_images, image_size, batch_size, device,
        model_kwargs={"img_size": image_size},
        force_recompute=force_recompute,
    )
    run_mechanistic(
        "vit_base_patch14_dinov2.lvd142m", "dinov2", get_vit_block_names,
        reduce_activation_scales,
        imagenet_root, results_dir, num_images, image_size, batch_size, device,
        model_kwargs={"img_size": image_size, "num_classes": 1000},
        force_recompute=force_recompute,
    )
