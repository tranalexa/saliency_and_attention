"""Shared experiment logic for cascading sanity checks (notebooks + Modal)."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
import timm

from attention_utils import (
    get_raw_attention,
    validate_raw_attention,
)
from metrics_utils import (
    abs_grayscale_norm,
    visualize_image_grayscale,
    characterize_cascade_curve,
    characterize_sensitivity_thresholds,
    compute_curve_auc,
    compute_logit_correlation,
    compute_sensitivity_ratio,
    compute_spearman,
    compute_ssim,
    compute_ssim_abs,
    diverging_norm,
    prepare_map_for_metric,
)
from viz_utils import pick_qual_image_index, pick_qual_image_index_shared
from randomize_utils import (
    get_resnet_block_names,
    get_vit_block_names,
    reset_layer,
    restore_checkpoint,
    save_checkpoint,
)

TargetMode = Literal["dynamic", "frozen_baseline"]
MapNorm = Literal["raw", "abs", "diverging"]
IgBaseline = Literal["zero", "mean"]
BlurType = Literal["box", "gaussian"]
SubsetOrder = Literal["imagefolder", "sorted"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class A — portable gradient methods (valid cross-architecture comparison).
METHODS_CLASS_A = ["gradient", "input_grad", "ig"]

METHODS_CLASS_B = {
    "resnet50": ["gradcam"],
    "vit": ["transformer_gradcam"],
}

METHODS_CLASS_C = {
    "resnet50": ["gbp"],
    "vit": ["raw_attn"],
}

METHODS_REFERENCE = {
    "resnet50": [],
    "vit": [],
}

METHODS_BY_ARCH = {
    arch: METHODS_CLASS_A + METHODS_CLASS_B[arch] + METHODS_CLASS_C[arch] + METHODS_REFERENCE[arch]
    for arch in ("resnet50", "vit")
}

# Backward-compatible aliases.
SHARED_SALIENCY_METHODS = list(METHODS_CLASS_A)
RESNET50_SALIENCY_METHODS = METHODS_BY_ARCH["resnet50"]
VIT_SALIENCY_METHODS = METHODS_BY_ARCH["vit"]
ARCH_SALIENCY_METHODS = METHODS_BY_ARCH

METHODS_SCORED = {
    arch: METHODS_CLASS_A + METHODS_CLASS_B[arch] + METHODS_CLASS_C[arch]
    for arch in METHODS_BY_ARCH
}

PRIMARY_SPEARMAN_VARIANT = {
    "ig": "signed_rms",
    "input_grad": "signed_rms",
}

GRADCAM_TARGET_BY_ARCH = {
    "resnet50": "layer4[-1]",
    "vit": "blocks[-2]",
}

MECHANISTIC_ARCH_TAG = {
    "resnet50": "resnet",
    "vit": "vit",
}

METHOD_BATCH_CAPS = {
    "ig": 2,
    "gbp": 4,
    "gradcam": 4,
    "transformer_gradcam": 4,
}


class CascadeContext:
    """Mutable cascade depth state consumed by architecture-specific methods."""

    def __init__(
        self,
        arch: str = "",
        order: List[str] | None = None,
        grid_size: int | None = None,
    ):
        self.arch = arch
        self.order = list(order or [])
        self.depth = 0
        self.grid_size = grid_size

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


META_BIN = "meta.bin"


def _parse_ilsvrc_wnid_to_index_from_meta_mat(meta_mat_path: Path) -> dict[str, int]:
    """Leaf synset WNID -> class index (0..999), matching torchvision ImageNet meta.

    Uses default struct_as_record=True so each synset row is a numpy.void that
    unpacks as (ILSVRC2012_ID, WNID, words, gloss, num_children, ...).
    struct_as_record=False would return mat_struct objects that aren't iterable,
    breaking zip(*synsets).
    """
    from scipy.io import loadmat

    synsets = loadmat(str(meta_mat_path), squeeze_me=True)["synsets"]
    nums_children = list(zip(*synsets))[4]
    leaves = [synsets[i] for i, n in enumerate(nums_children) if n == 0]
    wnids = list(zip(*leaves))[1]
    return {str(wnid): i for i, wnid in enumerate(wnids)}


def ilsvrc_wnid_to_index(root: str | Path) -> dict[str, int]:
    """Map val-folder WNID names to ILSVRC class indices used by ImageNet heads."""
    root = Path(root)
    meta_path = root / META_BIN
    if meta_path.exists():
        wnid_to_classes, _ = torch.load(meta_path, weights_only=True)
        return {wnid: i for i, wnid in enumerate(wnid_to_classes.keys())}

    for meta_mat in (
        root / "ILSVRC2012_devkit_t12" / "data" / "meta.mat",
        *root.glob("**/data/meta.mat"),
    ):
        if meta_mat.is_file():
            return _parse_ilsvrc_wnid_to_index_from_meta_mat(meta_mat)

    raise FileNotFoundError(
        "ILSVRC class mapping not found under %s. Need %s (re-run modal/download_imagenet.py) "
        "or devkit data/meta.mat on the volume."
        % (root, META_BIN)
    )


class ILSVRCValDataset(Dataset):
    """ImageFolder val layout with ILSVRC class indices (not alphabetical WNID order)."""

    def __init__(
        self,
        val_dir: str | Path,
        transform=None,
        wnid_to_idx: dict[str, int] | None = None,
        root_for_meta: str | Path | None = None,
    ):
        val_dir = Path(val_dir)
        root_for_meta = Path(root_for_meta or val_dir.parent)
        self.wnid_to_idx = wnid_to_idx or ilsvrc_wnid_to_index(root_for_meta)
        self.inner = ImageFolder(str(val_dir), transform=transform)
        missing = set(self.inner.classes) - set(self.wnid_to_idx)
        if missing:
            raise ValueError(
                "Unknown WNIDs in val (first few): %s" % sorted(missing)[:5]
            )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        path, _ = self.inner.samples[index]
        wnid = Path(path).parent.name
        image = self.inner.loader(path)
        if self.inner.transform is not None:
            image = self.inner.transform(image)
        return image, self.wnid_to_idx[wnid]


def _sorted_val_image_paths(val_dir: Path) -> list[Path]:
    """All val JPEGs sorted by filename (standard ILSVRC val tar order)."""
    paths = list(val_dir.rglob("*.JPEG")) + list(val_dir.rglob("*.jpg"))
    return sorted(paths, key=lambda p: p.name)


class SortedValDataset(Dataset):
    """ILSVRC val images in sorted JPEG filename order."""

    def __init__(
        self,
        val_dir: str | Path,
        transform=None,
        root_for_meta: str | Path | None = None,
    ):
        from PIL import Image

        val_dir = Path(val_dir)
        root_for_meta = Path(root_for_meta or val_dir.parent)
        self.wnid_to_idx = ilsvrc_wnid_to_index(root_for_meta)
        self.transform = transform
        self.loader = Image.open
        self.paths = _sorted_val_image_paths(val_dir)
        if not self.paths:
            raise FileNotFoundError("No val JPEGs found under %s" % val_dir)
        missing = {p.parent.name for p in self.paths} - set(self.wnid_to_idx)
        if missing:
            raise ValueError(
                "Unknown WNIDs in val (first few): %s" % sorted(missing)[:5]
            )
        self.labels = [self.wnid_to_idx[p.parent.name] for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = self.loader(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def load_imagenet_subset(
    root: str | Path,
    num_images: int = 500,
    split: str = "val",
    image_size: int = 224,
    transform=None,
    subset_order: SubsetOrder = "sorted",
) -> Tuple[Subset, List[int]]:
    """
    Load the first ``num_images`` of ImageNet validation.

    ``sorted`` (default): ``ILSVRC2012_val_*.JPEG`` paths under ``val/`` sorted by filename.

    ``imagefolder``: PyTorch ImageFolder order (alphabetical WNID folders).
    """
    transform = transform or build_transform(image_size)
    root = Path(root)
    val_dir = root / split
    if val_dir.exists() and subset_order == "sorted":
        dataset = SortedValDataset(val_dir, transform=transform, root_for_meta=root)
    elif val_dir.exists():
        try:
            dataset = datasets.ImageNet(root=str(root), split=split, transform=transform)
        except Exception:
            dataset = ILSVRCValDataset(val_dir, transform=transform, root_for_meta=root)
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


def _attribution_batch(images: torch.Tensor, device: str) -> torch.Tensor:
    """Copy inputs before saliency attribution (never reuse occlusion working tensors)."""
    return images.clone().to(device)


def _apply_patch_replacement(
    current: torch.Tensor,
    blurred: torch.Tensor,
    ys: slice,
    xs: slice,
) -> None:
    """Replace one spatial patch; supports (C,H,W) or (1,C,H,W)."""
    if current.dim() == 4:
        current[:, :, ys, xs] = blurred[:, :, ys, xs]
    elif current.dim() == 3:
        current[:, ys, xs] = blurred[:, ys, xs]
    else:
        raise ValueError("Expected 3D or 4D image tensor, got shape %s" % (current.shape,))


def attribution_to_map(
    attr: torch.Tensor,
    out_size: int = 224,
    grid_size: int | None = None,
    norm: MapNorm = "abs",
) -> np.ndarray:
    if attr.dim() == 4:
        # Preserve signed channel evidence; abs is applied later by metric variants.
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
            if norm == "raw":
                return out.astype(np.float64)
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
    if norm == "raw":
        return out.astype(np.float64)
    if norm == "abs":
        return abs_grayscale_norm(np.abs(out))
    return diverging_norm(out)


def gbp_attribution_to_map(
    attr: torch.Tensor,
    out_size: int = 224,
    norm: MapNorm = "raw",
) -> np.ndarray:
    """GBP: PAIR VisualizeImageGrayscale (positive max over channels), then optional norm."""
    arr = attr.squeeze().detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    out = visualize_image_grayscale(arr)
    if out.shape != (out_size, out_size):
        t = torch.from_numpy(out.astype(np.float64)).float()[None, None]
        out = (
            F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
        )
    if norm == "raw":
        return out.astype(np.float64)
    if norm == "abs":
        return abs_grayscale_norm(out)
    return diverging_norm(out)


def compute_ig(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    n_steps: int = 50,
    map_norm: MapNorm = "raw",
    ig_baseline: IgBaseline = "zero",
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    # NOTE: n_steps=50 is the default. The completeness check
    # (ig_completeness_pass_rate in experiment_config.json) monitors whether
    # this is sufficient. If pass rate < 0.8, re-run with --ig-steps 100 or higher.
    ig = IntegratedGradients(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        if ig_baseline == "zero":
            # NOTE: baseline is torch.zeros_like(input_tensor), applied to the
            # normalized tensor. This corresponds to the ImageNet mean pixel
            # (0.485, 0.456, 0.406) in [0,1] image space - a gray image, not a
            # black image. This is the standard IG baseline in the literature
            # and has a clean "absence of signal" interpretation in normalized space.
            baseline = torch.zeros_like(inp)
        else:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=inp.device).view(
                1, 3, 1, 1
            )
            baseline = imagenet_mean.expand_as(inp)
        attr = ig.attribute(inp, baselines=baseline, target=tgt, n_steps=n_steps)
        maps.append(attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def _ig_baseline_tensor(inp: torch.Tensor, ig_baseline: IgBaseline) -> torch.Tensor:
    if ig_baseline == "zero":
        return torch.zeros_like(inp)
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=inp.device).view(1, 3, 1, 1)
    return imagenet_mean.expand_as(inp)


def check_ig_completeness(
    ig_map: np.ndarray,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    model: nn.Module,
    target_class: int,
    device: torch.device,
    tol: float = 0.05,
    n_steps: int = 50,
) -> bool:
    """
    Checks IG completeness axiom: sum(IG) ~= f(x) - f(baseline).

    Returns True if the check passes, False if it fails. Logs a warning on
    failure and does not raise.
    """
    model.eval()
    with torch.no_grad():
        f_x = model(input_tensor.to(device))[0, target_class].item()
        f_baseline = model(baseline_tensor.to(device))[0, target_class].item()
    expected_diff = f_x - f_baseline
    actual_sum = float(ig_map.sum())
    if abs(expected_diff) < 1e-6:
        return True
    rel_error = abs(actual_sum - expected_diff) / abs(expected_diff)
    if rel_error > tol:
        logging.warning(
            "IG completeness violation: sum(IG)=%.4f, f(x)-f(baseline)=%.4f, "
            "relative error=%.3f > tol=%.3f. Consider increasing n_steps (currently %d).",
            actual_sum,
            expected_diff,
            rel_error,
            tol,
            n_steps,
        )
        return False
    return True


def check_ig_baseline_completeness(
    model: nn.Module,
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    device: str,
    ig_baseline: IgBaseline,
    n_steps: int,
    max_images: int = 10,
) -> float:
    """Check IG completeness on the first few baseline images and return pass rate."""
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)
    device_obj = torch.device(device)
    n_check = min(max_images, int(all_images.shape[0]))
    if n_check == 0:
        return float("nan")

    passed = 0
    for i in range(n_check):
        inp = all_images[i : i + 1].to(device)
        tgt = int(all_targets[i].item())
        baseline = _ig_baseline_tensor(inp, ig_baseline)
        attr = ig.attribute(inp, baselines=baseline, target=tgt, n_steps=n_steps)
        ok = check_ig_completeness(
            attr.squeeze(0).detach().cpu().numpy(),
            inp,
            baseline,
            model,
            tgt,
            device_obj,
            n_steps=n_steps,
        )
        passed += int(ok)
        _clear_cuda()

    pass_rate = passed / n_check
    if pass_rate < 0.8:
        suggested_steps = max(100, n_steps * 2)
        logging.warning(
            "IG completeness failing on >20%% of images. Consider increasing n_steps "
            "from %d to at least %d.",
            n_steps,
            suggested_steps,
        )
    return float(pass_rate)


def compute_gradient(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "raw",
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


def compute_input_grad(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "raw",
) -> np.ndarray:
    from captum.attr import InputXGradient

    ixg = InputXGradient(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = ixg.attribute(inp, target=tgt)
        map_1ch = attribution_to_map(attr, norm=map_norm)
        if map_norm == "raw":
            assert (map_1ch < 0).any(), (
                "input_grad map has no negative values. abs() was applied before "
                "channel aggregation. Check aggregation step."
            )
        maps.append(map_1ch)
    return np.stack(maps)


def compute_gbp(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    map_norm: MapNorm = "raw",
) -> np.ndarray:
    from captum.attr import GuidedBackprop

    gbp = GuidedBackprop(model)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1]
        tgt = int(target_indices[i].item())
        attr = gbp.attribute(inp, target=tgt)
        maps.append(gbp_attribution_to_map(attr, norm=map_norm))
    return np.stack(maps)


def compute_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_indices: torch.Tensor,
    layer: nn.Module,
    grid_size: int | None = None,
    map_norm: MapNorm = "raw",
) -> np.ndarray:
    from captum.attr import LayerGradCam

    cam = LayerGradCam(model, layer)
    maps = []
    for i in range(images.shape[0]):
        inp = images[i : i + 1].requires_grad_(True)
        tgt = int(target_indices[i].item())
        attr = cam.attribute(inp, target=tgt)
        attr = F.relu(attr)
        maps.append(attribution_to_map(attr, grid_size=grid_size, norm=map_norm))
    return np.stack(maps)


def vit_reshape_transform(
    tensor: torch.Tensor, height: int = 14, width: int = 14
) -> torch.Tensor:
    """Reshape ViT tokens to a spatial feature map for pytorch-grad-cam."""
    result = tensor[:, 1:, :]
    result = result.reshape(result.shape[0], height, width, result.shape[2])
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def compute_transformer_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int,
    arch: str,
    out_size: int = 224,
) -> np.ndarray:
    """
    GradCAM for ViT using pytorch-grad-cam.

    Target layer: model.blocks[-2].
    Reshape function converts token sequence to spatial feature map.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if arch == "vit":
        reshape_transform = vit_reshape_transform
    else:
        raise ValueError("Unknown arch for transformer gradcam: %s" % arch)

    targets = [ClassifierOutputTarget(int(target_class))] * images.shape[0]
    model.eval()

    target_layer = model.blocks[-2]
    with GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        grayscale_cam = cam(input_tensor=images, targets=targets)

    maps = []
    for i in range(grayscale_cam.shape[0]):
        heatmap_t = torch.tensor(grayscale_cam[i]).float()[None, None]
        up = F.interpolate(
            heatmap_t, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
        maps.append(up.squeeze().numpy().astype(np.float64))
    return np.stack(maps)


def check_and_flag_zero_gradcam(
    cam_map: np.ndarray,
    depth: int,
    image_idx: int,
    method: str = "transformer_gradcam",
) -> bool:
    """
    Return True if a GradCAM map is effectively zero and log it as unreliable.
    """
    if float(np.asarray(cam_map).max()) < 1e-6:
        logging.warning(
            "%s: zero map at depth=%d, image=%d. ReLU zeroed all activations - "
            "likely negative gradients due to class shift under dynamic target mode. "
            "Marking as unreliable.",
            method,
            depth,
            image_idx,
        )
        return True
    return False


@torch.no_grad()
def get_target_indices(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images).argmax(dim=1)


def load_all_images(
    loader: DataLoader, model: nn.Module, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_images, all_targets, all_gt_labels = [], [], []
    for images, gt_labels in tqdm(loader, desc="images"):
        all_images.append(images)
        all_targets.append(get_target_indices(model, images.to(device)).cpu())
        all_gt_labels.append(gt_labels)
    return torch.cat(all_images), torch.cat(all_targets), torch.cat(all_gt_labels)


def _method_baseline_path(results_dir: Path, method: str) -> Path:
    return results_dir / ("baseline_%s.npz" % method)


def _baseline_search_dirs(
    results_dir: Path, method: str, seed: int = 42
) -> list[Path]:
    """Class A baselines from parallel Modal jobs live under ``arch/seed{N}/``."""
    dirs = [results_dir]
    if method in METHODS_CLASS_A:
        seed_dir = results_dir / ("seed%d" % seed)
        if seed_dir not in dirs:
            dirs.append(seed_dir)
    return dirs


def _method_baseline_exists(
    results_dir: Path, method: str, seed: int = 42
) -> bool:
    for directory in _baseline_search_dirs(results_dir, method, seed):
        path = _method_baseline_path(directory, method)
        if not path.exists():
            legacy = directory / "baseline_maps.npz"
            if legacy.exists() and ("baseline_" + method) in np.load(legacy).files:
                return True
            continue
        data = np.load(path)
        if "baseline_map_raw" in data.files or "baseline_map" in data.files:
            return True
    return False


def load_baseline_maps(
    results_dir: Path, method: str, norm: MapNorm = "raw", seed: int = 42
) -> np.ndarray:
    if norm == "raw":
        key = "baseline_map_raw"
    elif norm == "abs":
        key = "baseline_map"
    else:
        key = "baseline_map_div"
    for directory in _baseline_search_dirs(results_dir, method, seed):
        per_method = _method_baseline_path(directory, method)
        if per_method.exists():
            data = np.load(per_method)
            if key in data.files:
                return data[key]
            if norm == "raw" and "baseline_map" in data.files:
                return data["baseline_map"]
        legacy = directory / "baseline_maps.npz"
        if legacy.exists():
            data = np.load(legacy)
            legacy_key = "baseline_" + method
            if legacy_key in data.files:
                return data[legacy_key]
    raise FileNotFoundError(
        "Baseline maps for method %r (norm=%s) not found under %s (seed=%d)"
        % (method, norm, results_dir, seed)
    )


def _primary_metric_file(method: str) -> str:
    variant = PRIMARY_SPEARMAN_VARIANT.get(method, "abs_rms")
    if variant == "signed_rms":
        return "%s_spearman_rms.npy" % method
    return "%s_spearman_abs_rms.npy" % method


def _load_logit_corr(results_dir: Path, arch: str) -> np.ndarray | None:
    tag = MECHANISTIC_ARCH_TAG.get(arch, arch)
    path = results_dir.parent / "mechanistic" / ("logit_corr_%s.npy" % tag)
    if path.exists():
        return np.load(path)
    return None


def _fractional_depths(num_depths: int) -> np.ndarray:
    if num_depths <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, num_depths, dtype=np.float64)


def _save_curve_characterization(
    results_dir: Path,
    method: str,
    arch: str,
    similarity_mean: np.ndarray,
    num_depths: int,
    logit_corr: np.ndarray | None,
) -> None:
    depths = _fractional_depths(num_depths)
    sim = np.asarray(similarity_mean, dtype=np.float64)
    sim_for_ratio = sim
    depths_for_ratio = depths

    primary_variant = PRIMARY_SPEARMAN_VARIANT.get(method, "abs_rms")
    stats = characterize_cascade_curve(sim_for_ratio, depths_for_ratio)
    stats["normalized_auc"] = compute_curve_auc(sim_for_ratio, depths_for_ratio)
    if logit_corr is not None:
        logit = np.asarray(logit_corr, dtype=np.float64)
        if logit.shape != depths.shape:
            logging.warning(
                "Skipping sensitivity ratio for %s/%s: logit_corr has shape %s "
                "but cascade depths have shape %s. Re-run mechanistic after the "
                "depth-0 cascade fix.",
                arch,
                method,
                logit.shape,
                depths.shape,
            )
            logit_corr = None
        else:
            logit_corr = logit
    if logit_corr is not None:
        logit = np.asarray(logit_corr, dtype=np.float64)
        logit_depths = depths
        d_arch = characterize_cascade_curve(logit, logit_depths)["d_half"]
        stats["d_arch"] = float(d_arch)
        stats["sensitivity_ratio"] = compute_sensitivity_ratio(
            stats["d_half"], logit, logit_depths
        )
        stats["threshold_sweep"] = characterize_sensitivity_thresholds(
            sim_for_ratio, logit, logit_depths
        )
    with open(results_dir / ("%s_curve_stats.json" % method), "w") as f:
        json.dump(stats, f, indent=2)
    if logit_corr is not None:
        ratio_payload = {
            "method": method,
            "arch": arch,
            "primary_metric_variant": primary_variant,
            "sensitivity_ratio": stats.get("sensitivity_ratio"),
            "d_half": stats.get("d_half"),
            "d_arch": stats.get("d_arch"),
            "threshold_sweep": stats.get("threshold_sweep"),
        }
        with open(results_dir / ("%s_sensitivity_ratio.json" % method), "w") as f:
            json.dump(ratio_payload, f, indent=2)


def compute_baseline_maps(
    model: nn.Module,
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    compute_fn: Callable[[str, torch.Tensor, torch.Tensor], np.ndarray],
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
        batch_maps = []
        method_bs = _method_batch_size(method, batch_size)
        for start in range(0, len(all_images), method_bs):
            batch = _attribution_batch(
                all_images[start : start + method_bs], device
            )
            tgt = all_targets[start : start + method_bs].to(device)
            batch_maps.append(compute_fn(method, batch, tgt))
            _clear_cuda()
        baseline_raw = np.concatenate(batch_maps, axis=0)
        if method == "input_grad":
            assert (baseline_raw < 0).any(), (
                "input_grad map has no negative values after channel aggregation. "
                "This suggests abs() was applied before aggregation, destroying sign. "
                "Check the aggregation step."
            )
        np.savez_compressed(
            _method_baseline_path(results_dir, method),
            baseline_map_raw=baseline_raw,
        )


def _map_summary_stats(arr: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanstd(a))


def _log_input_grad_ig_stats(results_dir: Path) -> None:
    try:
        input_grad = load_baseline_maps(results_dir, "input_grad", norm="raw")[0]
        ig = load_baseline_maps(results_dir, "ig", norm="raw")[0]
    except FileNotFoundError:
        return
    ixg_stats = _map_summary_stats(input_grad)
    ig_stats = _map_summary_stats(ig)
    print(
        "input_grad stats image0 min/max/std: %.6g %.6g %.6g"
        % ixg_stats
    )
    print("ig stats image0 min/max/std: %.6g %.6g %.6g" % ig_stats)
    assert not np.allclose(input_grad, ig), (
        "input_grad and ig maps are identical for image 0. Check method dispatch: "
        "InputXGradient and IntegratedGradients should run different computations."
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


def _reset_through_depth(model: nn.Module, order: Sequence[str], depth: int) -> None:
    """Depth 0 is the pretrained model; depth d randomizes order[:d]."""
    for name in order[:depth]:
        reset_layer(model, name)


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
    arch: str,
    cascade_context: CascadeContext,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
) -> None:
    original_sd = save_checkpoint(model)
    num_depths = len(order) + 1
    n_images = all_images.shape[0]
    logit_corr = _load_logit_corr(results_dir, arch)

    metric_variants = {
        "spearman_rms": "signed_rms",
        "spearman_abs_rms": "abs_rms",
        "spearman_maxabs": "abs_maxabs",
        "ssim_rms": "signed_rms",
        "ssim_abs_rms": "abs_rms",
        "ssim_maxabs": "abs_maxabs",
    }

    for method in methods:
        skip_path = results_dir / _primary_metric_file(method)
        if skip_path.exists() and not force_recompute:
            print("Skip", method)
            continue
        baseline_raw = load_baseline_maps(results_dir, method, norm="raw")
        arrays = {
            key: np.full((num_depths, n_images), np.nan) for key in metric_variants
        }
        zero_map_rates = (
            np.full(num_depths, np.nan, dtype=np.float64)
            if method == "transformer_gradcam"
            else None
        )
        for depth in tqdm(range(num_depths), desc=method):
            cascade_context.depth = depth
            restore_checkpoint(model, original_sd)
            _reset_through_depth(model, order, depth)
            rand_maps = []
            method_bs = _method_batch_size(method, batch_size)
            for start in range(0, n_images, method_bs):
                batch = _attribution_batch(
                    all_images[start : start + method_bs], device
                )
                tgt = _batch_targets(model, batch, all_targets, start, target_mode, device)
                rand_maps.append(compute_fn(method, batch, tgt))
                _clear_cuda()
            rand_maps = np.concatenate(rand_maps, axis=0)

            if method == "transformer_gradcam" and zero_map_rates is not None:
                zero_flags = [
                    check_and_flag_zero_gradcam(rand_maps[i], depth, i, method=method)
                    for i in range(n_images)
                ]
                zero_map_rates[depth] = float(np.mean(zero_flags))

            if method == "raw_attn":
                entropy_vals = []
                for start in range(0, n_images, method_bs):
                    batch = _attribution_batch(
                        all_images[start : start + method_bs], device
                    )
                    for i in range(batch.shape[0]):
                        _map, entropy = get_raw_attention(
                            model,
                            batch[i : i + 1],
                            cascade_context.grid_size,
                            return_entropy=True,
                        )
                        entropy_vals.append(entropy)
                np.save(
                    results_dir / ("raw_attn_entropy_depth%02d.npy" % depth),
                    np.array(entropy_vals, dtype=np.float64),
                )

            for i in range(n_images):
                for metric_key, variant in metric_variants.items():
                    base = prepare_map_for_metric(baseline_raw[i], variant)
                    rand = prepare_map_for_metric(rand_maps[i], variant)
                    if metric_key.startswith("spearman"):
                        arrays[metric_key][depth, i] = compute_spearman(base, rand)
                    elif variant.startswith("abs_"):
                        arrays[metric_key][depth, i] = compute_ssim_abs(base, rand)
                    else:
                        arrays[metric_key][depth, i] = compute_ssim(base, rand)

            primary_key = (
                "spearman_rms"
                if PRIMARY_SPEARMAN_VARIANT.get(method) == "signed_rms"
                else "spearman_abs_rms"
            )
            if depth == 0:
                depth0_spearman = float(np.nanmean(arrays[primary_key][depth]))
                assert depth0_spearman > 0.99, (
                    "Depth-0 Spearman=%.4f for %s. Must be ~1.0. "
                    "Baseline and cascade maps are being normalized differently "
                    "or depth 0 is not the pretrained model."
                    % (depth0_spearman, method)
                )

        for metric_key, arr in arrays.items():
            np.save(results_dir / ("%s_%s.npy" % (method, metric_key)), arr)
            np.save(
                results_dir / ("%s_%s_mean.npy" % (method, metric_key)),
                np.nanmean(arr, axis=1),
            )
        if zero_map_rates is not None:
            np.save(results_dir / "transformer_gradcam_zero_map_rate.npy", zero_map_rates)

        primary_key = (
            "spearman_rms"
            if PRIMARY_SPEARMAN_VARIANT.get(method) == "signed_rms"
            else "spearman_abs_rms"
        )
        if method in METHODS_CLASS_A or method in METHODS_CLASS_B.get(arch, []):
            _save_curve_characterization(
                results_dir,
                method,
                arch,
                np.nanmean(arrays[primary_key], axis=1),
                num_depths,
                logit_corr,
            )


NUM_PATCHES_BINDER = 30  # Binder et al. Section A.1
PATCH_SIZE_BINDER = 15  # Binder et al. Section A.1 main results
PATCH_SIZE_BINDER_ALT = 8  # Binder appendix robustness check


def normalize_images(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _box_blur(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Uniform box blur in [0, 1] image space (Binder et al. Section A.1)."""
    channels = tensor.shape[1]
    weight = torch.ones(
        channels,
        1,
        kernel_size,
        kernel_size,
        device=tensor.device,
        dtype=tensor.dtype,
    ) / (kernel_size * kernel_size)
    return F.conv2d(tensor, weight, padding=kernel_size // 2, groups=channels)


def make_blurred_inputs(
    images: torch.Tensor,
    blur_kernel_size: int = 31,
    blur_type: BlurType = "box",
    blur_sigma: float = 8.0,
) -> torch.Tensor:
    """Blur normalized inputs by temporarily returning to image space."""
    denormed = denormalize(images)
    if blur_type == "box":
        blurred = _box_blur(denormed, blur_kernel_size)
    elif blur_type == "gaussian":
        blurred = transforms.functional.gaussian_blur(
            denormed,
            kernel_size=[blur_kernel_size, blur_kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )
    else:
        raise ValueError("Unknown blur_type: %s (use box|gaussian)" % blur_type)
    return normalize_images(blurred)


def _patch_slices(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> list[tuple[slice, slice]]:
    ys = list(range(0, max(height - patch_size + 1, 1), stride))
    xs = list(range(0, max(width - patch_size + 1, 1), stride))
    if not ys or ys[-1] != max(height - patch_size, 0):
        ys.append(max(height - patch_size, 0))
    if not xs or xs[-1] != max(width - patch_size, 0):
        xs.append(max(width - patch_size, 0))
    return [
        (slice(y, min(y + patch_size, height)), slice(x, min(x + patch_size, width)))
        for y in ys
        for x in xs
    ]


def _rank_patches_by_saliency(
    saliency_map: np.ndarray,
    patch_slices: Sequence[tuple[slice, slice]],
) -> list[tuple[slice, slice]]:
    saliency = np.asarray(saliency_map, dtype=np.float64)
    scores = []
    for ys, xs in patch_slices:
        patch = saliency[ys, xs]
        score = np.abs(patch).mean()
        scores.append(float(score))
    order = np.argsort(scores)[::-1]
    return [patch_slices[int(i)] for i in order]


def _patch_score_stats(
    saliency_map: np.ndarray,
    patch_slices: Sequence[tuple[slice, slice]],
) -> tuple[float, float, float]:
    saliency = np.asarray(saliency_map, dtype=np.float64)
    scores = []
    for ys, xs in patch_slices:
        patch = saliency[ys, xs]
        score = np.abs(patch).mean()
        scores.append(float(score))
    scores = np.array(scores, dtype=np.float64)
    return float(scores.min()), float(scores.max()), float(scores.std())


def _target_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, target_class]
    return probs.detach().cpu().numpy()


def _blurred_deletion_curve(
    model: nn.Module,
    image: torch.Tensor,
    blurred: torch.Tensor,
    target_class: int,
    ranked_patches: Sequence[tuple[slice, slice]],
    num_patches: int,
    eval_batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidences, curve) where confidences[0] is unoccluded and curve excludes step 0."""
    image = image.detach().clone()
    blurred = blurred.detach().clone()
    if image.dim() == 3:
        image = image.unsqueeze(0)
        blurred = blurred.unsqueeze(0)
    current = image.clone()
    pending = [current.clone()]
    scores: list[np.ndarray] = []

    def flush() -> None:
        if pending:
            batch = torch.stack(pending).to(device)
            scores.append(_target_probabilities(model, batch, target_class))
            pending.clear()

    for ys, xs in ranked_patches[:num_patches]:
        _apply_patch_replacement(current, blurred, ys, xs)
        pending.append(current.clone())
        if len(pending) >= eval_batch_size:
            flush()
    flush()
    confidences = np.concatenate(scores, axis=0).astype(np.float64)
    return confidences, confidences[1:]


def _occlusion_output_suffix(num_patches: int, n_grid: int) -> str:
    return "_full" if num_patches >= n_grid else ""


def run_blurred_occlusion_faithfulness(
    model: nn.Module,
    all_images: torch.Tensor,
    all_targets: torch.Tensor,
    methods: Sequence[str],
    results_dir: Path,
    batch_size: int,
    device: str,
    patch_size: int = PATCH_SIZE_BINDER,
    stride: int | None = None,
    num_patches: int = NUM_PATCHES_BINDER,
    blur_kernel_size: int | None = None,
    blur_type: BlurType = "box",
    blur_sigma: float = 8.0,
    force_recompute: bool = False,
    seed: int = 42,
) -> None:
    """Run fixed-target blurred-patch deletion (Binder et al. Section A.1)."""
    if stride is None:
        stride = patch_size
    if blur_kernel_size is None:
        blur_kernel_size = patch_size

    results_dir.mkdir(parents=True, exist_ok=True)
    images_snapshot = all_images.clone()
    clean_images = all_images.clone()
    n_images, _channels, height, width = clean_images.shape
    patch_slices = _patch_slices(height, width, patch_size, stride)
    n_grid = len(patch_slices)
    effective_patches = min(num_patches, n_grid)
    output_suffix = _occlusion_output_suffix(effective_patches, n_grid)
    blurred_images = make_blurred_inputs(
        clean_images.to(device),
        blur_kernel_size=blur_kernel_size,
        blur_type=blur_type,
        blur_sigma=blur_sigma,
    ).cpu()

    for method in methods:
        curve_path = results_dir / (
            "occlusion_%s_curve%s.npy" % (method, output_suffix)
        )
        auc_path = results_dir / ("occlusion_%s_auc%s.npy" % (method, output_suffix))
        if curve_path.exists() and auc_path.exists() and not force_recompute:
            print("Skip occlusion", method)
            continue

        saliency_maps = load_baseline_maps(
            results_dir, method, norm="raw", seed=seed
        )
        curves = np.full((n_images, effective_patches), np.nan, dtype=np.float64)
        aucs = np.full(n_images, np.nan, dtype=np.float64)
        method_bs = _method_batch_size(method, batch_size)

        for i in tqdm(range(n_images), desc="occlusion " + method):
            target_class = int(all_targets[i].item())
            ranked = _rank_patches_by_saliency(saliency_maps[i], patch_slices)
            if i == 0:
                patch_min, patch_max, patch_std = _patch_score_stats(
                    saliency_maps[i], patch_slices
                )
                original_std = float(denormalize(clean_images[i : i + 1]).std().item())
                blurred_std = float(denormalize(blurred_images[i : i + 1]).std().item())
                print(
                    "occlusion %s image0 patch_score min/max/std: %.6g %.6g %.6g"
                    % (method, patch_min, patch_max, patch_std)
                )
                print(
                    "occlusion %s image0 original_std=%.6g blurred_std=%.6g fixed_target=%d"
                    % (method, original_std, blurred_std, target_class)
                )
                if blurred_std >= original_std:
                    logging.warning(
                        "Blurred image std %.6g is not lower than original std %.6g "
                        "for %s image0.",
                        blurred_std,
                        original_std,
                        method,
                    )
            _confidences, curve = _blurred_deletion_curve(
                model,
                clean_images[i].clone(),
                blurred_images[i].clone(),
                target_class,
                ranked,
                effective_patches,
                method_bs,
                device,
            )
            curves[i] = curve
            aucs[i] = float(np.mean(curve))
            _clear_cuda()

        if not torch.equal(all_images, images_snapshot):
            raise RuntimeError(
                "Occlusion modified the input batch; attribution tensors must stay clean."
            )

        np.save(curve_path, curves)
        np.save(auc_path, aucs)
        if not output_suffix:
            np.save(
                results_dir / ("occlusion_%s_auc_mean.npy" % method),
                np.nanmean(aucs),
            )

    occlusion_config = {
        "patch_size": patch_size,
        "stride": stride,
        "num_patches": effective_patches,
        "blur_kernel_size": blur_kernel_size,
        "blur_type": blur_type,
        "target_policy": "fixed_step0_target",
        "score": "softmax_probability",
        "curve": "target_probability_steps_1_to_N",
        "auc": "mean_confidence_steps_1_to_N",
        "methods": list(methods),
    }
    if blur_type == "gaussian":
        occlusion_config["blur_sigma"] = blur_sigma
    with open(results_dir / "occlusion_config.json", "w") as f:
        json.dump(occlusion_config, f, indent=2)


run_occlusion = run_blurred_occlusion_faithfulness


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
    cascade_context: CascadeContext,
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
            "qual_norm": np.array("abs_maxabs"),
        }
    )
    for method in methods_to_compute:
        restore_checkpoint(model, original_sd)
        _reset_through_depth(model, order, 0)
        attr_inp = inp.clone()
        if target_mode == "dynamic":
            tgt = get_target_indices(model, attr_inp)
        else:
            tgt = baseline_tgt
        raw = compute_fn(method, attr_inp, tgt)[0]
        out["baseline_" + method] = prepare_map_for_metric(raw, "abs_maxabs")
        if method == "raw_attn" and cascade_context.grid_size is not None:
            _raw_attn_map, per_head = get_raw_attention(
                model,
                attr_inp,
                cascade_context.grid_size,
                return_per_head=True,
            )
            out["raw_attn_per_head"] = per_head
        cascade = []
        for depth in range(len(order) + 1):
            cascade_context.depth = depth
            restore_checkpoint(model, original_sd)
            _reset_through_depth(model, order, depth)
            attr_inp = inp.clone()
            if target_mode == "dynamic":
                tgt = get_target_indices(model, attr_inp)
            else:
                tgt = baseline_tgt
            cascade.append(
                prepare_map_for_metric(
                    compute_fn(method, attr_inp, tgt)[0], "abs_maxabs"
                )
            )
        out["cascade_" + method] = np.stack(cascade)
    np.savez_compressed(qual_path, **out)


def _build_arch_runtime(
    arch: str,
    device: str,
    ig_steps: int = 50,
    image_size: int = 224,
    ig_baseline: IgBaseline = "zero",
) -> Tuple[nn.Module, List[str], CascadeContext, Callable, List[str], int | None]:
    """Load model and saliency dispatch for an architecture."""
    if arch == "resnet50":
        model = timm.create_model("resnet50", pretrained=True).to(device).eval()
        order = get_resnet_block_names(model)
        grid_size = None
        cascade_context = CascadeContext(arch=arch, order=order, grid_size=grid_size)
        compute_fn = make_saliency_compute_fn(
            model,
            arch,
            cascade_context,
            gradcam_layer=model.layer4[-1],
            ig_steps=ig_steps,
            ig_baseline=ig_baseline,
        )
        return model, order, cascade_context, compute_fn, list(METHODS_BY_ARCH[arch]), grid_size

    if arch == "vit":
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, img_size=image_size
        ).to(device).eval()
        order = get_vit_block_names(model)
        grid_size = 14
        cascade_context = CascadeContext(arch=arch, order=order, grid_size=grid_size)
        compute_fn = make_saliency_compute_fn(
            model,
            arch,
            cascade_context,
            ig_steps=ig_steps,
            attention_grid_size=grid_size,
            ig_baseline=ig_baseline,
        )
        return model, order, cascade_context, compute_fn, list(METHODS_BY_ARCH[arch]), grid_size

    raise ValueError("Unknown arch: %s (use resnet50|vit)" % arch)


def _setup_arch_model(
    arch: str,
    imagenet_root: Path,
    num_images: int,
    image_size: int,
    device: str,
    ig_steps: int,
    ig_baseline: IgBaseline = "zero",
) -> Tuple[nn.Module, List[str], Subset, Callable[[str, torch.Tensor, torch.Tensor], np.ndarray], List[str], CascadeContext]:
    """Load model, dataset subset, and saliency compute_fn for an architecture."""
    dataset, _ = load_imagenet_subset(
        imagenet_root,
        num_images,
        image_size=image_size,
        transform=build_transform(image_size),
    )

    model, order, cascade_context, compute_fn, methods, _grid = _build_arch_runtime(
        arch,
        device,
        ig_steps=ig_steps,
        image_size=image_size,
        ig_baseline=ig_baseline,
    )
    return model, order, dataset, compute_fn, methods, cascade_context


def run_qual_bundle_pipeline(
    arch: str,
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    device: str = "cuda",
    ig_steps: int = 50,
    image_index: int = 0,
    image_index_mode: str = "fixed",
    auto_ssim_method: str = "ig",
    force: bool = False,
    target_mode: TargetMode = "dynamic",
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
) -> int:
    """Build qual_bundle.npz for paper cascade figures. Returns image index used."""
    if arch not in ARCH_SALIENCY_METHODS:
        raise ValueError("Unknown arch: %s" % arch)
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    if image_index_mode in ("auto_ssim", "auto_ssim_shared"):
        peer_dirs = [
            results_dir.parent / name
            for name in ARCH_SALIENCY_METHODS
            if (results_dir.parent / name).is_dir()
        ]
        use_shared = image_index_mode == "auto_ssim_shared" or len(peer_dirs) > 1
        if use_shared and len(peer_dirs) > 1:
            image_index = pick_qual_image_index_shared(
                peer_dirs, method=auto_ssim_method, fallback=image_index
            )
        else:
            image_index = pick_qual_image_index(
                results_dir, method=auto_ssim_method, fallback=image_index
            )
    elif image_index_mode != "fixed":
        raise ValueError(
            "image_index_mode must be 'fixed', 'auto_ssim', or 'auto_ssim_shared'"
        )

    model, order, dataset, compute_fn, methods, cascade_context = _setup_arch_model(
        arch,
        imagenet_root,
        num_images,
        image_size,
        device,
        ig_steps,
        ig_baseline=ig_baseline,
    )
    build_qual_bundle(
        model,
        order,
        dataset,
        methods,
        compute_fn,
        results_dir,
        device,
        cascade_context,
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
    dataset, _ = load_imagenet_subset(
        imagenet_root,
        num_images,
        image_size=image_size,
        transform=build_transform(image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    create_kwargs = {"pretrained": True, **(model_kwargs or {})}
    model = timm.create_model(model_name, **create_kwargs).to(device).eval()
    order = order_fn(model)
    original_sd = save_checkpoint(model)

    logit_path = results_dir / ("logit_corr_%s.npy" % arch_tag)
    pretrained_logits_path = results_dir / ("pretrained_logits_%s.npy" % arch_tag)
    if force_recompute or not logit_path.exists():
        restore_checkpoint(model, original_sd)
        orig_logits = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="orig logits " + arch_tag):
                orig_logits.append(model(images.to(device)).detach().cpu().numpy())
        orig_logits = np.concatenate(orig_logits, axis=0)
        np.save(pretrained_logits_path, orig_logits)
        logit_corr = []
        for depth in tqdm(range(len(order) + 1), desc="logit " + arch_tag):
            restore_checkpoint(model, original_sd)
            _reset_through_depth(model, order, depth)
            depth_logits = []
            with torch.no_grad():
                for images, _ in loader:
                    depth_logits.append(model(images.to(device)).detach().cpu().numpy())
            logit_corr.append(compute_logit_correlation(orig_logits, np.concatenate(depth_logits)))
        assert logit_corr[0] > 0.5, (
            "Logit correlation at depth 0 = %.3f. Expected > 0.5. "
            "Check that pretrained_logits are computed before any randomization occurs."
            % logit_corr[0]
        )
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
    arch: str,
    cascade_context: CascadeContext,
    *,
    gradcam_layer: nn.Module | None = None,
    grid_size: int | None = None,
    ig_steps: int = 50,
    attention_grid_size: int | None = None,
    ig_baseline: IgBaseline = "zero",
    map_norm: MapNorm = "raw",
) -> Callable[[str, torch.Tensor, torch.Tensor], np.ndarray]:
    """Dispatch saliency method name -> batched attribution maps."""

    def compute_fn(method: str, batch: torch.Tensor, tgt: torch.Tensor) -> np.ndarray:
        if method == "gradient":
            return compute_gradient(model, batch, tgt, map_norm=map_norm)
        if method == "input_grad":
            return compute_input_grad(model, batch, tgt, map_norm=map_norm)
        if method == "gbp":
            return compute_gbp(model, batch, tgt, map_norm=map_norm)
        if method == "gradcam":
            if gradcam_layer is None:
                raise ValueError("gradcam requested but gradcam_layer is None")
            return compute_gradcam(
                model, batch, tgt, gradcam_layer, grid_size=grid_size, map_norm=map_norm
            )
        if method == "transformer_gradcam":
            return np.stack(
                [
                    compute_transformer_gradcam(
                        model, batch[i : i + 1], int(tgt[i].item()), arch=arch
                    )[0]
                    for i in range(batch.shape[0])
                ]
            )
        if method == "ig":
            return compute_ig(
                model,
                batch,
                tgt,
                ig_steps,
                map_norm=map_norm,
                ig_baseline=ig_baseline,
            )
        if attention_grid_size is not None:
            if method == "raw_attn":
                return np.stack(
                    [
                        get_raw_attention(
                            model,
                            batch[i : i + 1],
                            attention_grid_size,
                            apply_rms_norm=False,
                        )
                        for i in range(batch.shape[0])
                    ]
                )
        raise ValueError("Unknown saliency method: %s" % method)

    return compute_fn


def _load_existing_target_mode(results_dir: Path) -> TargetMode:
    """Preserve cascade target_mode when occlusion updates shared metadata."""
    config_path = results_dir / "experiment_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f).get("target_mode")
            if existing in ("dynamic", "frozen_baseline"):
                return existing
        except (json.JSONDecodeError, OSError):
            pass
    return "dynamic"


def _ensure_shared_metadata(
    results_dir: Path,
    image_indices: List[int],
    all_targets: torch.Tensor,
    order: List[str],
    arch: str,
    target_mode: TargetMode,
    seed: int,
    num_images: int,
    ig_steps: int = 50,
    ig_baseline: IgBaseline = "zero",
    ig_completeness_pass_rate: float | None = None,
    gt_labels: torch.Tensor | None = None,
    subset_order: SubsetOrder = "sorted",
) -> None:
    np.save(results_dir / "image_indices.npy", np.array(image_indices))
    np.save(results_dir / "target_indices.npy", all_targets.numpy())
    if gt_labels is not None:
        gt = gt_labels.numpy()
        np.save(results_dir / "ground_truth_indices.npy", gt)
        np.save(
            results_dir / "correctly_classified.npy",
            (all_targets.numpy() == gt),
        )
    with open(results_dir / "randomization_order.json", "w") as f:
        json.dump(order, f, indent=2)
    resnet_randomization = "block" if arch == "resnet50" else "n/a"
    config_path = results_dir / "experiment_config.json"
    if ig_completeness_pass_rate is None and config_path.exists():
        try:
            with open(config_path) as f:
                ig_completeness_pass_rate = json.load(f).get("ig_completeness_pass_rate")
        except (json.JSONDecodeError, OSError):
            ig_completeness_pass_rate = None
    write_experiment_config(
        results_dir,
        arch=arch,
        methods_class_a=list(METHODS_CLASS_A),
        methods_class_b=list(METHODS_CLASS_B[arch]),
        methods_class_c=list(METHODS_CLASS_C[arch]),
        methods_reference=list(METHODS_REFERENCE[arch]),
        ig_baseline=ig_baseline,
        ig_steps=ig_steps,
        ig_completeness_pass_rate=ig_completeness_pass_rate,
        normalization_primary="rms",
        normalization_legacy="maxabs",
        primary_metric_ig="spearman_div_rms",
        primary_metric_input_grad="spearman_div_rms",
        primary_metric_others="spearman_abs_rms",
        gradcam_target=GRADCAM_TARGET_BY_ARCH[arch],
        target_mode=target_mode,
        resnet_randomization=resnet_randomization,
        seed=seed,
        num_images=num_images,
        randomization_order=order,
        subset_order=subset_order,
    )


def _run_arch_pipeline(
    arch: str,
    imagenet_root: Path,
    results_dir: Path,
    methods: Sequence[str],
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    skip_qual: bool = False,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
    subset_order: SubsetOrder = "sorted",
) -> None:
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset, image_indices = load_imagenet_subset(
        imagenet_root,
        num_images,
        image_size=image_size,
        transform=build_transform(image_size),
        subset_order=subset_order,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model, order, cascade_context, compute_fn, _all_methods, _grid = _build_arch_runtime(
        arch,
        device,
        ig_steps=ig_steps,
        image_size=image_size,
        ig_baseline=ig_baseline,
    )

    scored_methods = [m for m in methods if m in METHODS_SCORED[arch]]
    if arch == "vit" and "raw_attn" in scored_methods:
        first_img, _ = dataset[0]
        validate_raw_attention(model, first_img[None].to(device), name=arch)
    all_images, all_targets, gt_labels = load_all_images(loader, model, device)
    ig_completeness_pass_rate = None
    if "ig" in scored_methods:
        ig_completeness_pass_rate = check_ig_baseline_completeness(
            model,
            all_images,
            all_targets,
            device,
            ig_baseline,
            ig_steps,
        )
    _ensure_shared_metadata(
        results_dir,
        image_indices,
        all_targets,
        order,
        arch,
        target_mode,
        seed,
        num_images,
        ig_steps=ig_steps,
        ig_baseline=ig_baseline,
        ig_completeness_pass_rate=ig_completeness_pass_rate,
        gt_labels=gt_labels,
        subset_order=subset_order,
    )
    compute_baseline_maps(
        model,
        all_images,
        all_targets,
        scored_methods,
        compute_fn,
        results_dir,
        batch_size,
        device,
        force_recompute=force_recompute,
    )
    if "input_grad" in scored_methods and "ig" in scored_methods:
        _log_input_grad_ig_stats(results_dir)
    run_cascading_experiment(
        model,
        order,
        all_images,
        all_targets,
        scored_methods,
        compute_fn,
        results_dir,
        batch_size,
        device,
        arch,
        cascade_context,
        target_mode=target_mode,
        force_recompute=force_recompute,
    )
    if not skip_qual:
        build_qual_bundle(
            model,
            order,
            dataset,
            methods,
            compute_fn,
            results_dir,
            device,
            cascade_context,
            target_mode=target_mode,
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
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
) -> None:
    """Run cascading sanity check for a single architecture + method (parallel Modal workers)."""
    if arch not in ARCH_SALIENCY_METHODS:
        raise ValueError("Unknown arch: %s (use resnet50|vit)" % arch)
    allowed = ARCH_SALIENCY_METHODS[arch]
    if method not in allowed:
        raise ValueError("Method %r not valid for arch %r (allowed: %s)" % (method, arch, allowed))

    _run_arch_pipeline(
        arch=arch,
        imagenet_root=imagenet_root,
        results_dir=results_dir,
        methods=[method],
        num_images=num_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        ig_steps=ig_steps,
        skip_qual=True,
        target_mode=target_mode,
        force_recompute=force_recompute,
        seed=seed,
        ig_baseline=ig_baseline,
    )


def run_resnet50_pipeline(
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    skip_qual: bool = False,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
) -> None:
    _run_arch_pipeline(
        arch="resnet50",
        imagenet_root=imagenet_root,
        results_dir=results_dir,
        methods=METHODS_BY_ARCH["resnet50"],
        num_images=num_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        ig_steps=ig_steps,
        skip_qual=skip_qual,
        target_mode=target_mode,
        force_recompute=force_recompute,
        seed=seed,
        ig_baseline=ig_baseline,
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
    skip_qual: bool = False,
    use_attention: bool = True,
    target_mode: TargetMode = "dynamic",
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
) -> None:
    methods = METHODS_BY_ARCH["vit"] if use_attention else METHODS_CLASS_A
    _run_arch_pipeline(
        arch="vit",
        imagenet_root=imagenet_root,
        results_dir=results_dir,
        methods=methods,
        num_images=num_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        ig_steps=ig_steps,
        skip_qual=skip_qual,
        target_mode=target_mode,
        force_recompute=force_recompute,
        seed=seed,
        ig_baseline=ig_baseline,
    )


def run_occlusion_pipeline(
    arch: str,
    imagenet_root: Path,
    results_dir: Path,
    num_images: int = 500,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cuda",
    ig_steps: int = 50,
    force_recompute: bool = False,
    seed: int = 42,
    ig_baseline: IgBaseline = "zero",
    patch_size: int = PATCH_SIZE_BINDER,
    stride: int | None = None,
    num_patches: int = NUM_PATCHES_BINDER,
    blur_kernel_size: int | None = None,
    blur_type: BlurType = "box",
    blur_sigma: float = 8.0,
    subset_order: SubsetOrder = "sorted",
) -> None:
    """Run the blurred-occlusion faithfulness axis for one scoped architecture."""
    if arch not in ARCH_SALIENCY_METHODS:
        raise ValueError("Unknown arch: %s (use resnet50|vit)" % arch)
    set_seed(seed)
    validate_imagenet_root(imagenet_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset, image_indices = load_imagenet_subset(
        imagenet_root,
        num_images,
        image_size=image_size,
        transform=build_transform(image_size),
        subset_order=subset_order,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model, order, _cascade_context, _compute_fn, methods, _grid = _build_arch_runtime(
        arch,
        device,
        ig_steps=ig_steps,
        image_size=image_size,
        ig_baseline=ig_baseline,
    )
    if arch == "vit" and "raw_attn" in methods:
        first_img, _ = dataset[0]
        validate_raw_attention(model, first_img[None].to(device), name=arch)

    missing = [
        m for m in methods if not _method_baseline_exists(results_dir, m, seed=seed)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing baseline npz for: %s. Run cascade first." % missing
        )

    all_images, all_targets, gt_labels = load_all_images(loader, model, device)
    _ensure_shared_metadata(
        results_dir,
        image_indices,
        all_targets,
        order,
        arch,
        _load_existing_target_mode(results_dir),
        seed,
        num_images,
        ig_steps=ig_steps,
        ig_baseline=ig_baseline,
        gt_labels=gt_labels,
        subset_order=subset_order,
    )
    run_blurred_occlusion_faithfulness(
        model,
        all_images,
        all_targets,
        methods,
        results_dir,
        batch_size,
        device,
        patch_size=patch_size,
        stride=stride,
        num_patches=num_patches,
        blur_kernel_size=blur_kernel_size,
        blur_type=blur_type,
        blur_sigma=blur_sigma,
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
