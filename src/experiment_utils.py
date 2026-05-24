"""Shared experiment logic for cascading sanity checks (notebooks + Modal)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

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
from metrics_utils import abs_grayscale_norm, compute_logit_correlation, compute_spearman, compute_ssim
from randomize_utils import (
    get_resnet_conv1_names,
    get_vit_block_names,
    reset_layer,
    restore_checkpoint,
    save_checkpoint,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Matches Adebayo et al. Inception/MNIST notebooks (pair-code/saliency).
RESNET50_SALIENCY_METHODS = [
    "gradient",
    "smoothgrad",
    "input_grad",
    "gbp",
    "gradcam",
    "gbp_gc",
    "ig",
    "ig_smoothgrad",
]

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


def attribution_to_map(attr: torch.Tensor, out_size: int = 224) -> np.ndarray:
    if attr.dim() == 4:
        attr = attr.sum(dim=1)
    arr = attr.squeeze().detach().cpu().numpy()
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        arr = arr.reshape(side, side)
    arr = np.abs(arr)
    t = torch.from_numpy(arr).float()[None, None]
    up = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return abs_grayscale_norm(up.squeeze().numpy())


def compute_ig(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = ig.attribute(inp, baselines=torch.zeros_like(inp), target=tgt, n_steps=n_steps)
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_gradient(
    model: nn.Module, images: torch.Tensor, target_indices: torch.Tensor
) -> np.ndarray:
    from captum.attr import Saliency

    sal = Saliency(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = sal.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_smoothgrad(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    stdev: float = 0.15,
    n_samples: int = 25,
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
            n_samples=n_samples,
        )
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_input_grad(
    model: nn.Module, images: torch.Tensor, target_indices: torch.Tensor
) -> np.ndarray:
    from captum.attr import InputXGradient

    ixg = InputXGradient(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = ixg.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_ig_smoothgrad(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    n_steps: int = 50,
    stdev: float = 0.15,
    n_samples: int = 25,
) -> np.ndarray:
    from captum.attr import IntegratedGradients, NoiseTunnel

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = nt.attribute(
            inp,
            baselines=torch.zeros_like(inp),
            target=tgt,
            n_steps=n_steps,
            nt_type="smoothgrad",
            stdevs=stdev,
            n_samples=n_samples,
        )
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_gbp_gc(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    layer: nn.Module,
) -> np.ndarray:
    gbp_maps = compute_gbp(model, images, target_indices)
    gc_maps = compute_gradcam(model, images, target_indices, layer)
    return gbp_maps * gc_maps


def compute_gbp(
    model: nn.Module, images: torch.Tensor, target_indices: torch.Tensor
) -> np.ndarray:
    from captum.attr import GuidedBackprop

    gbp = GuidedBackprop(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = gbp.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr))
    return np.stack(maps)


def compute_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    layer: nn.Module,
) -> np.ndarray:
    from captum.attr import LayerGradCam

    cam = LayerGradCam(model, layer)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = cam.attribute(inp, target=tgt)
        maps.append(attribution_to_map(attr))
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


def compute_baseline_maps(
    model: nn.Module,
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    baseline_path = results_dir / "baseline_maps.npz"
    if baseline_path.exists():
        return
    baseline = {}
    for method in tqdm(methods, desc="baseline"):
        batch_maps = []
        for start in range(0, len(all_images), batch_size):
            batch = all_images[start : start + batch_size].to(device)
            tgt = all_targets[start : start + batch_size].to(device)
            batch_maps.append(compute_fn(method, batch, tgt))
        baseline["baseline_" + method] = np.concatenate(batch_maps, axis=0)
    np.savez_compressed(baseline_path, **baseline)


def run_cascading_experiment(
    model: nn.Module,
    order: List[str],
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    original_sd = save_checkpoint(model)
    num_depths = len(order)
    n_images = all_images.shape[0]
    baseline_path = results_dir / "baseline_maps.npz"

    for method in methods:
        spearman_path = results_dir / ("%s_spearman.npy" % method)
        if spearman_path.exists():
            print("Skip", method)
            continue
        baseline_maps = np.load(baseline_path)["baseline_" + method]
        spearman_all = np.full((num_depths, n_images), np.nan)
        ssim_all = np.full((num_depths, n_images), np.nan)
        for depth, _layer_name in enumerate(tqdm(order, desc=method)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            rand_maps = []
            for start in range(0, n_images, batch_size):
                batch = all_images[start : start + batch_size].to(device)
                tgt = all_targets[start : start + batch_size].to(device)
                rand_maps.append(compute_fn(method, batch, tgt))
            rand_maps = np.concatenate(rand_maps, axis=0)
            for i in range(n_images):
                spearman_all[depth, i] = compute_spearman(baseline_maps[i], rand_maps[i])
                ssim_all[depth, i] = compute_ssim(baseline_maps[i], rand_maps[i])
        np.save(spearman_path, spearman_all)
        np.save(results_dir / ("%s_ssim.npy" % method), ssim_all)
        np.save(results_dir / ("%s_spearman_mean.npy" % method), np.nanmean(spearman_all, axis=1))
        np.save(results_dir / ("%s_ssim_mean.npy" % method), np.nanmean(ssim_all, axis=1))


def build_qual_bundle(
    model: nn.Module,
    order: List[str],
    dataset: Subset,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
    results_dir: Path,
    device: str,
) -> None:
    qual_path = results_dir / "qual_bundle.npz"
    if qual_path.exists():
        return
    original_sd = save_checkpoint(model)
    img, _ = dataset[0]
    inp = img[None].to(device)
    tgt = get_target_indices(model, inp)
    out = {
        "image": denormalize(inp).squeeze(0).permute(1, 2, 0).cpu().numpy(),
        "order": np.array(order, dtype=object),
    }
    for method in methods:
        out["baseline_" + method] = compute_fn(method, inp, tgt)[0]
        cascade = []
        for depth in range(len(order)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            cascade.append(compute_fn(method, inp, tgt)[0])
        out["cascade_" + method] = np.stack(cascade)
    np.savez_compressed(qual_path, **out)


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
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(image_size)
    dataset, _ = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    order = order_fn(model)
    original_sd = save_checkpoint(model)

    logit_path = results_dir / ("logit_corr_%s.npy" % arch_tag)
    if not logit_path.exists():
        orig_logits = []
        for images, _ in tqdm(loader, desc="orig logits " + arch_tag):
            orig_logits.append(model(images.to(device)).cpu().numpy())
        orig_logits = np.concatenate(orig_logits, axis=0)
        logit_corr = []
        for depth, _ in enumerate(tqdm(order, desc="logit " + arch_tag)):
            restore_checkpoint(model, original_sd)
            for name in order[: depth + 1]:
                reset_layer(model, name)
            depth_logits = []
            for images, _ in loader:
                depth_logits.append(model(images.to(device)).cpu().numpy())
            logit_corr.append(compute_logit_correlation(orig_logits, np.concatenate(depth_logits)))
        np.save(logit_path, np.array(logit_corr))

    for depth in range(len(order)):
        act_path = results_dir / ("activation_scale_%s_depth%02d.npy" % (arch_tag, depth))
        if act_path.exists() or depth + 1 >= len(order):
            continue
        hook_layer = order[depth + 1]
        restore_checkpoint(model, original_sd)
        for name in order[: depth + 1]:
            reset_layer(model, name)
        activations = []

        def hook_fn(_m, _i, out):
            activations.append(out.detach())

        handle = model.get_submodule(hook_layer).register_forward_hook(hook_fn)
        for images, _ in loader:
            model(images.to(device))
        handle.remove()
        scales = act_reduce(torch.cat(activations, dim=0)).cpu().numpy()
        np.save(act_path, scales)

    with open(results_dir / ("randomization_order_%s.json" % arch_tag), "w") as f:
        json.dump(order, f)


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
) -> None:
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = list(RESNET50_SALIENCY_METHODS)

    transform = build_transform(image_size)
    dataset, image_indices = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    np.save(results_dir / "image_indices.npy", np.array(image_indices))

    model = timm.create_model("resnet50", pretrained=True).to(device).eval()
    order = get_resnet_conv1_names(model)
    with open(results_dir / "randomization_order.json", "w") as f:
        json.dump(order, f, indent=2)
    gradcam_layer = model.layer4[-1].conv3

    def compute_fn(method, batch, tgt):
        if method == "gradient":
            return compute_gradient(model, batch, tgt)
        if method == "smoothgrad":
            return compute_smoothgrad(
                model, batch, tgt, stdev=smoothgrad_stdev, n_samples=smoothgrad_samples
            )
        if method == "input_grad":
            return compute_input_grad(model, batch, tgt)
        if method == "gbp":
            return compute_gbp(model, batch, tgt)
        if method == "gradcam":
            return compute_gradcam(model, batch, tgt, gradcam_layer)
        if method == "gbp_gc":
            return compute_gbp_gc(model, batch, tgt, gradcam_layer)
        if method == "ig":
            return compute_ig(model, batch, tgt, ig_steps)
        if method == "ig_smoothgrad":
            return compute_ig_smoothgrad(
                model,
                batch,
                tgt,
                n_steps=ig_steps,
                stdev=smoothgrad_stdev,
                n_samples=smoothgrad_samples,
            )
        raise ValueError("Unknown saliency method: %s" % method)

    all_images, all_targets = load_all_images(loader, model, device)
    np.save(results_dir / "target_indices.npy", all_targets.numpy())
    compute_baseline_maps(model, all_images, all_targets, methods, compute_fn, results_dir, batch_size, device)
    run_cascading_experiment(model, order, all_images, all_targets, methods, compute_fn, results_dir, batch_size, device)
    if not skip_qual:
        build_qual_bundle(model, order, dataset, methods, compute_fn, results_dir, device)


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
    skip_qual: bool = False,
    use_attention: bool = True,
) -> None:
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = ["ig", "gradcam", "raw_attn", "rollout"] if use_attention else ["ig", "gradcam"]

    transform = build_transform(image_size)
    dataset, image_indices = load_imagenet_subset(imagenet_root, num_images, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    np.save(results_dir / "image_indices.npy", np.array(image_indices))

    model_kwargs = {"pretrained": True, "img_size": image_size}
    if "dinov2" in model_name:
        model_kwargs["num_classes"] = 1000
    model = timm.create_model(model_name, **model_kwargs).to(device).eval()
    order = get_vit_block_names(model)
    with open(results_dir / "randomization_order.json", "w") as f:
        json.dump(order, f, indent=2)
    gradcam_layer = model.blocks[-1].norm1

    def compute_fn(method, batch, tgt):
        if method == "ig":
            return compute_ig(model, batch, tgt, ig_steps)
        if method == "gradcam":
            return compute_gradcam(model, batch, tgt, gradcam_layer)
        if method == "raw_attn":
            return np.stack([get_raw_attention(model, batch[i : i + 1], grid_size) for i in range(batch.shape[0])])
        return np.stack([get_rollout(model, batch[i : i + 1], grid_size) for i in range(batch.shape[0])])

    all_images, all_targets = load_all_images(loader, model, device)
    np.save(results_dir / "target_indices.npy", all_targets.numpy())
    compute_baseline_maps(model, all_images, all_targets, methods, compute_fn, results_dir, batch_size, device)
    run_cascading_experiment(model, order, all_images, all_targets, methods, compute_fn, results_dir, batch_size, device)
    if not skip_qual:
        build_qual_bundle(model, order, dataset, methods, compute_fn, results_dir, device)


def run_dinov2_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    skip_qual: bool = False,
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
        skip_qual=skip_qual,
        use_attention=True,
    )


def run_mechanistic_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "cuda",
) -> None:
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_mechanistic(
        "resnet50", "resnet", get_resnet_conv1_names,
        lambda t: t.abs().mean(dim=(0, 2, 3)),
        imagenet_root, results_dir, num_images, image_size, batch_size, device,
    )
    run_mechanistic(
        "vit_base_patch16_224", "vit", get_vit_block_names,
        lambda t: t.abs().mean(dim=(0, 1)),
        imagenet_root, results_dir, num_images, image_size, batch_size, device,
    )
