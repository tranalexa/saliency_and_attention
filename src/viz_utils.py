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
    "input_grad": "Input-Grad",
    "ig": "Integrated\nGradients",
    "gradcam": "GradCAM",
    "transformer_gradcam": "Transformer\nGradCAM",
    "gbp": "Guided\nBackProp",
    "raw_attn": "Raw\nAttention",
}


ARCH_FAMILY_CNN = "cnn"
ARCH_FAMILY_TRANSFORMER = "transformer"

# Results folder name -> architecture family for column labels.
ARCH_TO_FAMILY = {
    "resnet50": ARCH_FAMILY_CNN,
    "vit": ARCH_FAMILY_TRANSFORMER,
}

# Optional ResNet-50 labels aligned with Adebayo Inception figure naming.
# Default figures now use actual ResNet module names.
_RESNET_STAGE_MAX_BLOCK = {4: 2, 3: 5, 2: 3, 1: 2}
_RESNET_STAGE_LABELS = {
    4: ["Mixed_7c", "Mixed_7b", "Mixed_7a"],
    3: ["Mixed_6e", "Mixed_6d", "Mixed_6c", "Mixed_6b", "Mixed_6a", "Mixed_6"],
    2: ["Mixed_5d", "Mixed_5c", "Mixed_5b", "Mixed_5a"],
    1: ["Conv2d_4a", "Conv2d_3b", "Conv2d_2b"],
}


def format_resnet_adebayo_label(layer_name: str) -> str:
    """Map ResNet bottleneck names to optional Adebayo-style column labels."""
    name = str(layer_name)
    if name == "fc":
        return "Logits"
    m = re.match(r"layer(\d+)\.(\d+)(?:\.conv1)?$", name)
    if not m:
        return short_layer_label(name)
    stage, block = int(m.group(1)), int(m.group(2))
    labels = _RESNET_STAGE_LABELS.get(stage)
    max_block = _RESNET_STAGE_MAX_BLOCK.get(stage)
    if labels is not None and max_block is not None:
        idx = max_block - block
        if 0 <= idx < len(labels):
            return labels[idx]
    return short_layer_label(name)


def format_resnet_label(layer_name: str, adebayo_style: bool = False) -> str:
    """Human-readable ResNet label; real module names by default."""
    if adebayo_style:
        return format_resnet_adebayo_label(layer_name)
    name = str(layer_name)
    if name == "fc":
        return "fc"
    return short_layer_label(name, max_len=12)


def infer_arch_family(order: Sequence[str]) -> str:
    """Infer CNN (ResNet) vs transformer (ViT) from randomization order."""
    for name in order:
        if re.match(r"blocks\.\d+$", str(name)):
            return ARCH_FAMILY_TRANSFORMER
    for name in order:
        if re.match(r"layer\d+\.\d+", str(name)):
            return ARCH_FAMILY_CNN
    if order and str(order[0]).startswith("layer"):
        return ARCH_FAMILY_CNN
    return "unknown"


def resolve_arch_family(arch: Optional[str], order: Sequence[str]) -> str:
    if arch and arch in ARCH_TO_FAMILY:
        return ARCH_TO_FAMILY[arch]
    return infer_arch_family(order)


def format_cascade_column_label(
    layer_name: str, arch_family: str, resnet_adebayo_labels: bool = False
) -> str:
    """Human-readable column header; CNN vs transformer prefixes differ."""
    name = str(layer_name)
    if arch_family == ARCH_FAMILY_TRANSFORMER:
        m = re.match(r"blocks\.(\d+)$", name)
        if m:
            return "blocks.%s" % m.group(1)
        if name.startswith("head"):
            return "head"
        if name == "fc":
            return "Classifier"
    if arch_family == ARCH_FAMILY_CNN:
        return format_resnet_label(name, adebayo_style=resnet_adebayo_labels)
    return short_layer_label(name)


def first_resnet_layer4_depth(order: Sequence[str]) -> int | None:
    """Depth where the ResNet GradCAM target stage is first randomized."""
    for i, name in enumerate(order):
        if str(name).startswith("layer4."):
            return i
    return None


def add_resnet_gradcam_target_marker(
    ax: plt.Axes,
    order: Sequence[str],
    *,
    x_values: Sequence[float] | None = None,
    label: str = "GradCAM target layer first randomized",
) -> None:
    """Draw a vertical marker where ResNet layer4 is first randomized."""
    depth = first_resnet_layer4_depth(order)
    if depth is None:
        return
    x = x_values[depth] if x_values is not None and depth < len(x_values) else depth
    ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(
        x,
        0.98,
        label,
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        transform=ax.get_xaxis_transform(),
    )


def _load_logit_corr(results_root: Union[str, Path], arch: str) -> np.ndarray | None:
    tag = {"resnet50": "resnet", "vit": "vit"}.get(arch, arch)
    path = Path(results_root) / "mechanistic" / ("logit_corr_%s.npy" % tag)
    if path.exists():
        return np.load(path)
    return None


def add_logit_corr_band(
    ax: plt.Axes,
    results_root: Union[str, Path],
    arch: str,
    *,
    x_values: Sequence[float] | None = None,
    label: str = "Model output preservation (logit corr to pretrained)",
    annotate_resnet_skip: bool = True,
) -> plt.Axes | None:
    """Overlay model-output preservation as a gray twin-axis control band."""
    logit_corr = _load_logit_corr(results_root, arch)
    if logit_corr is None:
        return None
    x = np.asarray(x_values if x_values is not None else np.arange(len(logit_corr)))
    y = np.asarray(logit_corr, dtype=np.float64)
    n = min(len(x), len(y))
    if n == 0:
        return None
    twin = ax.twinx()
    twin.fill_between(x[:n], 0.0, y[:n], color="0.85", alpha=0.55, label=label)
    twin.plot(x[:n], y[:n], color="0.45", linewidth=1.0)
    twin.set_ylim(0, 1)
    twin.set_ylabel("logit corr", color="0.35", fontsize=8)
    twin.tick_params(axis="y", labelsize=7, colors="0.35")
    if arch == "resnet50" and annotate_resnet_skip:
        ax.text(
            0.02,
            0.02,
            "ResNet skip connections preserve logit correlation; sustained saliency similarity here may reflect architectural bypass.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            alpha=0.78,
            wrap=True,
        )
    return twin


def load_raw_attention_entropy(results_dir: Union[str, Path]) -> np.ndarray | None:
    """Load per-depth raw attention entropy means if present."""
    paths = sorted(Path(results_dir).glob("raw_attn_entropy_depth*.npy"))
    if not paths:
        return None
    vals = []
    for path in paths:
        arr = np.load(path)
        vals.append(float(np.nanmean(arr)))
    return np.array(vals, dtype=np.float64)


def add_attention_entropy_overlay(
    ax: plt.Axes,
    results_dir: Union[str, Path],
    *,
    num_patches: int = 196,
    x_values: Sequence[float] | None = None,
) -> plt.Axes | None:
    """Overlay raw-attention entropy and shade near-uniform collapse regions."""
    entropy = load_raw_attention_entropy(results_dir)
    if entropy is None:
        return None
    x = np.asarray(x_values if x_values is not None else np.arange(len(entropy)))
    n = min(len(x), len(entropy))
    if n == 0:
        return None
    entropy = entropy[:n]
    x = x[:n]
    threshold = 0.8 * np.log(num_patches)
    twin = ax.twinx()
    twin.plot(x, entropy, color="tab:red", linewidth=1.0, label="raw attention entropy")
    twin.set_ylabel("attention entropy", color="tab:red", fontsize=8)
    twin.tick_params(axis="y", labelsize=7, colors="tab:red")
    collapsed = entropy > threshold
    if np.any(collapsed):
        ax.fill_between(
            x,
            0,
            1,
            where=collapsed,
            transform=ax.get_xaxis_transform(),
            color="tab:red",
            alpha=0.10,
            label="Attention near-uniform",
        )
    return twin


def load_gradcam_zero_map_rate(results_dir: Union[str, Path]) -> np.ndarray | None:
    """Load transformer GradCAM zero-map rates if present."""
    path = Path(results_dir) / "transformer_gradcam_zero_map_rate.npy"
    if not path.exists():
        return None
    return np.load(path)


def add_gradcam_unreliable_overlay(
    ax: plt.Axes,
    results_dir: Union[str, Path],
    *,
    x_values: Sequence[float] | None = None,
    threshold: float = 0.2,
) -> plt.Axes | None:
    """Shade cascade regions where transformer GradCAM zero maps are common."""
    zero_rate = load_gradcam_zero_map_rate(results_dir)
    if zero_rate is None:
        return None
    x = np.asarray(x_values if x_values is not None else np.arange(len(zero_rate)))
    n = min(len(x), len(zero_rate))
    if n == 0:
        return None
    x = x[:n]
    unreliable = np.asarray(zero_rate[:n], dtype=np.float64) > threshold
    if np.any(unreliable):
        ax.fill_between(
            x,
            0,
            1,
            where=unreliable,
            transform=ax.get_xaxis_transform(),
            color="tab:orange",
            alpha=0.12,
            label="GradCAM unreliable (>20% zero maps)",
        )
    return ax


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
    """Min-max stretch to [0, 1] for imshow.

    Unlike abs_grayscale_norm (divides by max, origin at 0), this subtracts the
    minimum first so the full dynamic range is always visible. This matters for
    diffuse maps like raw attention from a pretrained ViT where all values cluster
    near the maximum, making abs-max norm render everything white.
    """
    m = np.abs(np.asarray(map_2d, dtype=np.float64))
    lo, hi = m.min(), m.max()
    if hi > lo:
        return (m - lo) / (hi - lo)
    return np.zeros_like(m)


def prepare_heatmap(
    map_2d: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    """Normalize saliency for overlay; percentile clip avoids one-pixel max dominating."""
    m = np.abs(np.asarray(map_2d, dtype=np.float64))
    if m.size == 0:
        return m
    if percentile is not None and 0 < percentile < 100:
        hi = float(np.percentile(m, percentile))
        if hi > 0:
            return np.clip(m / hi, 0.0, 1.0)
    return abs_grayscale_norm(m)


def overlay_saliency_on_image(
    rgb: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
    percentile: float = 99.0,
) -> np.ndarray:
    """Blend a warm heatmap on the input image (Adebayo bird-demo style)."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be HxWx3, got %s" % (rgb.shape,))
    heat = prepare_heatmap(saliency, percentile=percentile)
    colored = plt.get_cmap(cmap)(heat)[:, :, :3]
    return np.clip((1.0 - alpha) * rgb + alpha * colored, 0.0, 1.0)


def _show_map_panel(
    ax: plt.Axes,
    rgb: np.ndarray,
    saliency: np.ndarray,
    *,
    overlay: bool,
    show_input_only: bool = False,
    overlay_alpha: float = 0.45,
    heatmap_cmap: str = "jet",
    display_percentile: float = 99.0,
) -> None:
    if show_input_only or (overlay and saliency is None):
        ax.imshow(np.clip(rgb, 0, 1))
    elif overlay:
        ax.imshow(
            overlay_saliency_on_image(
                rgb,
                saliency,
                alpha=overlay_alpha,
                cmap=heatmap_cmap,
                percentile=display_percentile,
            )
        )
    else:
        ax.imshow(
            prepare_map_for_display(saliency),
            vmin=0.0,
            vmax=1.0,
            cmap="gray",
        )


def select_depth_indices(n_depths: int, max_cols: Optional[int] = None) -> List[int]:
    """Subsample cascade depth indices. max_cols=None shows every step (Adebayo-style)."""
    if n_depths <= 0:
        return []
    if max_cols is None:
        return list(range(n_depths))
    # max_cols is total grid columns including the Normal Model column
    max_depth_cols = max_cols - 1
    if n_depths <= max_depth_cols:
        return list(range(n_depths))
    n_show = max_depth_cols
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
    path = Path(results_dir) / ("%s_ssim_abs_rms.npy" % method)
    if not path.exists():
        for alt in ("gradient", "input_grad", "gradcam", "transformer_gradcam"):
            alt_path = Path(results_dir) / ("%s_ssim_abs_rms.npy" % alt)
            if alt_path.exists():
                path = alt_path
                break
        else:
            legacy = Path(results_dir) / ("%s_ssim.npy" % method)
            if legacy.exists():
                path = legacy
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
    overlay: bool = True,
    overlay_alpha: float = 0.45,
    heatmap_cmap: str = "jet",
    display_percentile: float = 99.0,
    max_depth_cols: Optional[int] = None,
    dpi: int = 150,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Adebayo-style grid: rows = saliency methods, cols = baseline + cascade depths.

    max_depth_cols=None shows every randomization step (e.g. all 17 for ResNet-50).
    Set max_depth_cols=8 to subsample for a compact figure.

    overlay=True blends a jet-style heatmap on the stored input image (paper bird figure).
    overlay=False shows grayscale masks on black only.
    """
    qual_path = Path(qual_path)
    if not qual_path.exists():
        return None
    data = np.load(qual_path, allow_pickle=True)
    order = list(data["order"])
    methods = [m for m in methods if ("baseline_" + m) in data.files and ("cascade_" + m) in data.files]
    if not methods:
        return None

    rgb = np.clip(np.asarray(data["image"], dtype=np.float64), 0.0, 1.0)

    cascade0 = data["cascade_" + methods[0]]
    n_depths = len(cascade0)
    if depth_indices is None:
        depth_indices = select_depth_indices(n_depths, max_cols=max_depth_cols)
    depth_indices = list(depth_indices)
    arch_family = resolve_arch_family(arch, order)

    col_labels = ["Normal\nModel"]
    for d in depth_indices:
        if d == 0:
            col_labels.append("Pretrained\n(depth 0)")
        else:
            layer = order[d - 1] if d - 1 < len(order) else ""
            col_labels.append(format_cascade_column_label(layer, arch_family))

    nrows = len(methods)
    ncols = 1 + len(depth_indices)
    label_col_width = 0.48
    col_inches = 0.72 if ncols > 14 else (0.85 if ncols > 10 else 1.15)
    fig_w = col_inches * ncols + label_col_width
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
        cascade = data["cascade_" + method]
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j + 1])
            if j == 0:
                # Normal Model = saliency from pretrained weights (not a plain photo).
                _show_map_panel(
                    ax,
                    rgb,
                    data["baseline_" + method],
                    overlay=overlay,
                    overlay_alpha=overlay_alpha,
                    heatmap_cmap=heatmap_cmap,
                    display_percentile=display_percentile,
                )
            else:
                d = depth_indices[j - 1]
                _show_map_panel(
                    ax,
                    rgb,
                    cascade[d],
                    overlay=overlay,
                    overlay_alpha=overlay_alpha,
                    heatmap_cmap=heatmap_cmap,
                    display_percentile=display_percentile,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                title_fs = 7 if ncols > 14 else 8
                title_rot = 55 if ncols > 10 else 35
                ax.set_title(
                    col_labels[j],
                    fontsize=title_fs,
                    pad=12,
                    rotation=title_rot,
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
    overlay: bool = True,
    overlay_alpha: float = 0.45,
    heatmap_cmap: str = "jet",
    display_percentile: float = 99.0,
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
    rgb = np.clip(np.asarray(data["image"], dtype=np.float64), 0.0, 1.0)
    nrows = len(cascade) + 2
    fig = plt.figure(figsize=(4, 0.4 * nrows))
    gs = gridspec.GridSpec(nrows, 1)
    ax = fig.add_subplot(gs[0])
    ax.imshow(rgb)
    ax.set_title("Input")
    ax.axis("off")
    ax = fig.add_subplot(gs[1])
    _show_map_panel(
        ax,
        rgb,
        data["baseline_" + method],
        overlay=overlay,
        overlay_alpha=overlay_alpha,
        heatmap_cmap=heatmap_cmap,
        display_percentile=display_percentile,
    )
    ax.set_title("Baseline (no randomization)")
    ax.axis("off")
    for i, m in enumerate(cascade):
        ax = fig.add_subplot(gs[i + 2])
        _show_map_panel(
            ax,
            rgb,
            m,
            overlay=overlay,
            overlay_alpha=overlay_alpha,
            heatmap_cmap=heatmap_cmap,
            display_percentile=display_percentile,
        )
        if i == 0:
            label = "Pretrained (no randomization)"
        else:
            layer = order[i - 1] if i - 1 < len(order) else ""
            label = format_cascade_column_label(layer, arch_family)
        ax.set_title("Depth %d: %s" % (i, label), fontsize=8)
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
