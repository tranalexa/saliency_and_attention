"""Visualization helpers for cascading saliency sanity-check figures."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from metrics_utils import abs_grayscale_norm

METHOD_DISPLAY_NAMES = {
    "gradient": "Gradient",
    "smoothgrad": "SmoothGrad",
    "input_grad": "Input-Grad",
    "ig": "Integrated\nGradients",
    "gradcam": "GradCAM",
    "gbp": "Guided\nBackProp",
    "gbp_gc": "GBP-GC",
    "raw_attn": "Raw\nAttention",
    "rollout": "Rollout",
}


ARCH_FAMILY_CNN = "cnn"
ARCH_FAMILY_TRANSFORMER = "transformer"

# Results folder name -> architecture family for column labels.
ARCH_TO_FAMILY = {
    "resnet50": ARCH_FAMILY_CNN,
    "vit": ARCH_FAMILY_TRANSFORMER,
    "dinov2": ARCH_FAMILY_TRANSFORMER,
}


def infer_arch_family(order: Sequence[str]) -> str:
    """Infer CNN (ResNet) vs transformer (ViT/DINOv2) from randomization order."""
    for name in order:
        if re.match(r"blocks\.\d+$", str(name)):
            return ARCH_FAMILY_TRANSFORMER
    for name in order:
        if re.match(r"layer\d+\.\d+\.conv1$", str(name)):
            return ARCH_FAMILY_CNN
    if order and str(order[0]).startswith("layer"):
        return ARCH_FAMILY_CNN
    return "unknown"


def resolve_arch_family(arch: Optional[str], order: Sequence[str]) -> str:
    if arch and arch in ARCH_TO_FAMILY:
        return ARCH_TO_FAMILY[arch]
    return infer_arch_family(order)


def format_cascade_column_label(layer_name: str, arch_family: str) -> str:
    """Human-readable column header; CNN vs transformer prefixes differ."""
    name = str(layer_name)
    if arch_family == ARCH_FAMILY_TRANSFORMER:
        m = re.match(r"blocks\.(\d+)$", name)
        if m:
            return "TF block %s" % m.group(1)
        if name.startswith("head"):
            return "TF classifier"
        if name == "fc":
            return "Classifier"
    if arch_family == ARCH_FAMILY_CNN:
        m = re.match(r"layer(\d+)\.(\d+)\.conv1$", name)
        if m:
            return "CNN S%s-B%s" % (m.group(1), m.group(2))
        if name == "fc":
            return "CNN classifier"
    return short_layer_label(name)


def short_layer_label(name: str, max_len: int = 9) -> str:
    """Shorten module name for column headers (fallback)."""
    if name == "fc" or name.startswith("head"):
        return name[:max_len]
    m = re.match(r"blocks\.(\d+)$", name)
    if m:
        return "blk%s" % m.group(1)
    parts = name.split(".")
    if parts[0].startswith("layer") and len(parts) >= 2:
        return "%s.%s" % (parts[0], parts[1])
    label = parts[-1] if parts else name
    if len(label) > max_len:
        label = label[:max_len]
    return label


def prepare_map_for_display(map_2d: np.ndarray) -> np.ndarray:
    """Absolute grayscale normalize to [0, 1] for imshow."""
    return abs_grayscale_norm(np.asarray(map_2d))


def select_depth_indices(n_depths: int, max_cols: int = 8) -> List[int]:
    """Subsample cascade depth indices for readable figures (inclusive endpoints)."""
    if n_depths <= 0:
        return []
    if n_depths <= max_cols:
        return list(range(n_depths))
    # max_cols includes baseline column separately; we want up to max_cols-1 depth columns
    n_show = min(max_cols - 1, n_depths)
    if n_show <= 1:
        return [0]
    if n_show == 2:
        return [0, n_depths - 1]
    raw = np.linspace(0, n_depths - 1, n_show)
    indices = sorted(set(int(round(x)) for x in raw))
    if indices[0] != 0:
        indices = [0] + indices
    if indices[-1] != n_depths - 1:
        indices.append(n_depths - 1)
    return indices


def pick_qual_image_index(
    results_dir: Path,
    method: str = "ig",
    fallback: int = 0,
) -> int:
    """Pick image with largest SSIM drop (baseline vs fully randomized)."""
    path = Path(results_dir) / ("%s_ssim.npy" % method)
    if not path.exists():
        for alt in ("gradient", "input_grad", "gradcam"):
            alt_path = Path(results_dir) / ("%s_ssim.npy" % alt)
            if alt_path.exists():
                path = alt_path
                break
        else:
            return fallback
    ssim = np.load(path)
    if ssim.ndim != 2 or ssim.shape[1] == 0:
        return fallback
    drop = ssim[0] - ssim[-1]
    if np.all(np.isnan(drop)):
        return fallback
    return int(np.nanargmax(drop))


def _method_display(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def plot_cascade_paper_grid(
    qual_path: Union[str, Path],
    methods: Sequence[str],
    depth_indices: Optional[Sequence[int]] = None,
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    arch: Optional[str] = None,
    max_depth_cols: int = 8,
    dpi: int = 150,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Adebayo-style grid: rows = saliency methods, cols = baseline + cascade depths.
    """
    qual_path = Path(qual_path)
    if not qual_path.exists():
        return None
    data = np.load(qual_path, allow_pickle=True)
    order = list(data["order"])
    methods = [m for m in methods if ("baseline_" + m) in data.files and ("cascade_" + m) in data.files]
    if not methods:
        return None

    cascade0 = data["cascade_" + methods[0]]
    n_depths = len(cascade0)
    if depth_indices is None:
        depth_indices = select_depth_indices(n_depths, max_cols=max_depth_cols)
    depth_indices = list(depth_indices)
    arch_family = resolve_arch_family(arch, order)

    col_labels = ["Normal\nmodel"]
    for d in depth_indices:
        layer = order[d] if d < len(order) else ""
        col_labels.append(format_cascade_column_label(layer, arch_family))

    nrows = len(methods)
    ncols = 1 + len(depth_indices)
    label_col_width = 0.48
    fig_w = 1.15 * ncols + label_col_width
    fig_h = max(1.05 * nrows, 4.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid_top = 0.84 if title else 0.90
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[label_col_width] + [1.0] * ncols,
        wspace=0.10,
        hspace=0.42,
        left=0.04,
        right=0.99,
        top=grid_top,
        bottom=0.04,
    )

    for i, method in enumerate(methods):
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.axis("off")
        ax_label.text(
            1.0,
            0.5,
            _method_display(method),
            ha="right",
            va="center",
            fontsize=9,
            linespacing=1.25,
            transform=ax_label.transAxes,
        )
        baseline = prepare_map_for_display(data["baseline_" + method])
        cascade = data["cascade_" + method]
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j + 1])
            if j == 0:
                ax.imshow(baseline, vmin=0.0, vmax=1.0, cmap="gray")
            else:
                d = depth_indices[j - 1]
                ax.imshow(
                    prepare_map_for_display(cascade[d]),
                    vmin=0.0,
                    vmax=1.0,
                    cmap="gray",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(
                    col_labels[j],
                    fontsize=8,
                    pad=12,
                    rotation=35,
                    ha="right",
                )

    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    if out_path:
        fig.savefig(Path(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cascading_grid(
    qual_path: Union[str, Path],
    method: str,
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    arch: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Vertical strip: input, baseline, then each cascade depth for one method."""
    qual_path = Path(qual_path)
    if not qual_path.exists():
        return None
    data = np.load(qual_path, allow_pickle=True)
    key = "cascade_" + method
    if key not in data.files:
        return None
    cascade = data[key]
    order = list(data["order"])
    arch_family = resolve_arch_family(arch, order)
    nrows = len(cascade) + 2
    fig = plt.figure(figsize=(4, 0.4 * nrows))
    gs = gridspec.GridSpec(nrows, 1)
    ax = fig.add_subplot(gs[0])
    ax.imshow(data["image"])
    ax.set_title("Input")
    ax.axis("off")
    ax = fig.add_subplot(gs[1])
    ax.imshow(prepare_map_for_display(data["baseline_" + method]), vmin=0, vmax=1, cmap="gray")
    ax.set_title("Baseline (no randomization)")
    ax.axis("off")
    for i, m in enumerate(cascade):
        ax = fig.add_subplot(gs[i + 2])
        ax.imshow(prepare_map_for_display(m), vmin=0, vmax=1, cmap="gray")
        layer = order[i] if i < len(order) else ""
        ax.set_title(
            "Depth %d: %s" % (i, format_cascade_column_label(layer, arch_family)),
            fontsize=8,
        )
        ax.axis("off")
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=0.96)
    if out_path:
        fig.savefig(Path(out_path), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
