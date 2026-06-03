"""Visualization helpers for cascading saliency sanity-check figures."""
from __future__ import annotations

import json
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

# Default: one colormap for every method (fair cross-method comparison).
CASCADE_MASK_CMAP_UNIFIED = "bwr"
CASCADE_MASK_CMAP_PAPER = "gray"

# Named presets for notebooks: suffix -> plot_cascade_paper_grid kwargs.
CASCADE_DISPLAY_PRESETS = {
    "bwr_pct": {
        "overlay": False,
        "mask_cmap": "bwr",
        "display_norm": "percentile",
        "colormap_style": "unified",
    },
    "jet_pct": {
        "overlay": False,
        "mask_cmap": "jet",
        "display_norm": "percentile",
        "colormap_style": "unified",
    },
    "turbo_pct": {
        "overlay": False,
        "mask_cmap": "turbo",
        "display_norm": "percentile",
        "colormap_style": "unified",
    },
    "hot_pct": {
        "overlay": False,
        "mask_cmap": "hot",
        "display_norm": "percentile",
        "colormap_style": "unified",
    },
    "overlay_jet": {
        "overlay": True,
        "heatmap_cmap": "jet",
        "display_percentile": 99.0,
    },
    "bwr_absmax": {
        "overlay": False,
        "mask_cmap": "bwr",
        "display_norm": "absmax",
        "colormap_style": "unified",
    },
}

# Legacy TF notebooks used per-method cmaps (second figure block in cascading ipynb).
METHOD_CASCADE_CMAP_ADEBAYO_LEGACY = {
    "gbp": "bwr",
    "gradient": "bwr",
    "input_grad": "bwr",
    "ig": "bwr",
    "gradcam": "hot",
    "transformer_gradcam": "hot",
    "raw_attn": "gray",
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


_FLAT_MAP_STD_THRESH = 1e-4
_FLAT_MAP_RANGE_THRESH = 0.02


def prepare_map_for_display(
    map_2d: np.ndarray,
    *,
    display_norm: str = "minmax",
    percentile: float = 99.0,
    flat_value: float = 0.5,
) -> np.ndarray:
    """
    Normalize qual maps for mask panels (display only; does not change stored .npz).

    display_norm:
      - ``minmax``: stretch |x| to [0, 1] (default; matches pre-33413b2 commit figures).
      - ``percentile``: clip to p-th percentile then scale to [0, 1] (sparse ViT maps).
      - ``absmax``: |x| / max(|x|) (strict Adebayo abs-max).
    Near-uniform maps render as flat_value (avoids saturated raw-attn panels).
    """
    m = np.asarray(map_2d, dtype=np.float64)
    if m.size == 0:
        return m
    if display_norm != "minmax":
        if m.std() < _FLAT_MAP_STD_THRESH and (m.max() - m.min()) < _FLAT_MAP_RANGE_THRESH:
            return np.full_like(m, flat_value)
    if display_norm == "minmax":
        m = np.abs(m)
        lo, hi = m.min(), m.max()
        if hi > lo:
            return (m - lo) / (hi - lo)
        return np.zeros_like(m)
    if display_norm == "absmax":
        return abs_grayscale_norm(m)
    if display_norm == "percentile":
        return prepare_heatmap(m, percentile=percentile)
    raise ValueError(
        "display_norm must be 'minmax', 'percentile', or 'absmax', got %s" % display_norm
    )


def cascade_colormap_for_method(
    method: str,
    colormap_style: str = "unified",
    mask_cmap: str | None = None,
) -> str:
    """
    Colormap for cascade mask panels (same for all methods unless overridden).

    colormap_style:
      - ``unified`` (default): ``bwr`` on abs-max masks for every method.
      - ``paper``: ``gray`` for every method (notebook first-figure style).
      - ``legacy``: per-method TF demo (bwr/hot/gray mix).
    mask_cmap: if set, overrides colormap_style for all methods.
    """
    if mask_cmap is not None:
        return mask_cmap
    if colormap_style == "unified":
        return CASCADE_MASK_CMAP_UNIFIED
    if colormap_style == "paper":
        return CASCADE_MASK_CMAP_PAPER
    if colormap_style in ("legacy", "adebayo"):
        return METHOD_CASCADE_CMAP_ADEBAYO_LEGACY.get(method, "gray")
    raise ValueError(
        "colormap_style must be 'unified', 'paper', or 'legacy', got %s" % colormap_style
    )


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
    mask_cmap: str = "gray",
    display_percentile: float = 99.0,
    display_norm: str = "minmax",
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
            prepare_map_for_display(
                saliency,
                display_norm=display_norm,
                percentile=display_percentile,
            ),
            vmin=0.0,
            vmax=1.0,
            cmap=mask_cmap,
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


def _ssim_metric_paths(results_dir: Path, method: str, seed: int = 42) -> list[Path]:
    """SSIM curves for Class A may live under ``seed{N}/`` after parallel Modal runs."""
    names = [method, "ig", "gradient", "input_grad", "gradcam", "transformer_gradcam", "gbp"]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    dirs = [Path(results_dir)]
    seed_dir = Path(results_dir) / ("seed%d" % seed)
    if seed_dir not in dirs:
        dirs.append(seed_dir)
    paths: list[Path] = []
    for directory in dirs:
        for name in ordered:
            for suffix in ("_ssim_abs_rms.npy", "_ssim.npy"):
                path = directory / ("%s%s" % (name, suffix))
                if path.exists():
                    paths.append(path)
                    break
    return paths


def pick_qual_image_index(
    results_dir: Path,
    method: str = "ig",
    fallback: int = 0,
    seed: int | None = None,
) -> int:
    """Pick image with largest SSIM drop (baseline vs fully randomized)."""
    if seed is None:
        seed = experiment_seed(results_dir)
    paths = _ssim_metric_paths(results_dir, method, seed=seed)
    best_idx = fallback
    best_drop = -1.0
    for path in paths:
        ssim = np.load(path)
        if ssim.ndim != 2 or ssim.shape[1] == 0:
            continue
        drop = ssim[0] - ssim[-1]
        if np.all(np.isnan(drop)):
            continue
        idx = int(np.nanargmax(drop))
        drop_val = float(drop[idx])
        if drop_val > best_drop:
            best_drop = drop_val
            best_idx = idx
    return best_idx if best_drop >= 0 else fallback


def pick_qual_image_index_shared(
    results_dirs: list[Path] | tuple[Path, ...],
    method: str = "ig",
    fallback: int = 0,
) -> int:
    """Pick one subset index with the largest SSIM drop across archs and methods."""
    per_image: dict[int, float] = {}
    for results_dir in results_dirs:
        seed = experiment_seed(results_dir)
        for path in _ssim_metric_paths(results_dir, method, seed=seed):
            ssim = np.load(path)
            if ssim.ndim != 2 or ssim.shape[1] == 0:
                continue
            drop = ssim[0] - ssim[-1]
            for i, value in enumerate(drop):
                if np.isnan(value):
                    continue
                v = float(value)
                if v > per_image.get(i, -1.0):
                    per_image[i] = v
    if not per_image:
        return fallback
    return max(per_image, key=per_image.get)


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
    colormap_style: str = "unified",
    mask_cmap: str | None = None,
    display_norm: str = "minmax",
    dpi: int = 150,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Adebayo-style grid: rows = saliency methods, cols = baseline + cascade depths.

    max_depth_cols=None shows every randomization step (e.g. all 17 for ResNet-50).
    Set max_depth_cols=8 to subsample for a compact figure.

    Default overlay=True (jet on input image), matching commit 33413b2 primary figures.
    overlay=False uses gray mask panels with display_norm='minmax' unless overridden.
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
        row_cmap = cascade_colormap_for_method(
            method, colormap_style=colormap_style, mask_cmap=mask_cmap
        )
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
                    mask_cmap=row_cmap,
                    display_percentile=display_percentile,
                    display_norm=display_norm,
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
                    mask_cmap=row_cmap,
                    display_percentile=display_percentile,
                    display_norm=display_norm,
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
    colormap_style: str = "unified",
    mask_cmap: str | None = None,
    display_norm: str = "minmax",
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
    row_cmap = cascade_colormap_for_method(
        method, colormap_style=colormap_style, mask_cmap=mask_cmap
    )
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
        mask_cmap=row_cmap,
        display_percentile=display_percentile,
        display_norm=display_norm,
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
            mask_cmap=row_cmap,
            display_percentile=display_percentile,
            display_norm=display_norm,
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


MECH_LOGIT_TAG = {
    "resnet50": "resnet",
    "vit": "vit",
}


def experiment_seed(results_dir: Path | str, default: int = 42) -> int:
    """RNG seed recorded by cascade (used for Class A ``seed{N}/`` subdirs)."""
    config_path = Path(results_dir) / "experiment_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return int(json.load(f).get("seed", default))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return default


def load_arch_baseline_maps(
    results_root: Path | str,
    arch: str,
    method: str,
    norm: str = "raw",
    seed: int | None = None,
) -> np.ndarray:
    """Load ``baseline_{method}.npz`` maps, including parallel-run ``seed{N}/`` paths."""
    from experiment_utils import load_baseline_maps

    arch_dir = Path(results_root) / arch
    if seed is None:
        seed = experiment_seed(arch_dir)
    return load_baseline_maps(arch_dir, method, norm=norm, seed=seed)


def _method_display_label(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " "))


def _load_step0_target_confidence(
    results_root: Path,
    arch: str,
    mech_arch_tags: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Per-image unoccluded softmax for the fixed occlusion target class."""
    arch_dir = results_root / arch
    targets_path = arch_dir / "target_indices.npy"
    tags = mech_arch_tags or MECH_LOGIT_TAG
    tag = tags.get(arch, arch)
    logits_path = results_root / "mechanistic" / ("pretrained_logits_%s.npy" % tag)
    if not targets_path.exists() or not logits_path.exists():
        return None
    import torch

    targets = np.load(targets_path)
    logits = np.load(logits_path)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1).numpy()
    n = min(len(targets), probs.shape[0])
    return probs[np.arange(n), targets[:n]].astype(np.float64)


def _occlusion_mean_std_curve(
    curve_path: Path,
    step0_per_image: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Mean/std over images per occlusion step; prepend step 0 when available."""
    if not curve_path.exists():
        return None, None, False
    curves = np.asarray(np.load(curve_path), dtype=np.float64)
    mean_post = np.nanmean(curves, axis=0)
    std_post = np.nanstd(curves, axis=0)
    if step0_per_image is not None:
        step0_mean = float(np.nanmean(step0_per_image))
        step0_std = float(np.nanstd(step0_per_image))
        mean = np.concatenate([[step0_mean], mean_post])
        std = np.concatenate([[step0_std], std_post])
        return mean, std, True
    return mean_post, std_post, False


def _plot_occlusion_curve_band(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None,
    label: str,
    **kwargs,
) -> None:
    ax.plot(x, mean, marker="o", markersize=3, label=label, **kwargs)
    if std is not None and np.any(std > 0):
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)


def plot_occlusion_faithfulness_curves(
    results_root: Path | str,
    archs: dict[str, dict],
    figures_dir: Path | str,
    mech_arch_tags: dict[str, str] | None = None,
    dpi: int = 150,
    show: bool = True,
) -> Path | None:
    """
    Binder-style blurred-patch deletion curves: mean target softmax vs step.

    Loads ``occlusion_{method}_curve.npy`` (steps 1–30) and prepends step 0 from
    saved mechanistic logits + target_indices when available.
    """
    results_root = Path(results_root)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    arch_items = [(arch, cfg) for arch, cfg in archs.items() if cfg.get("methods")]
    if not arch_items:
        print("No architectures configured for occlusion faithfulness plot.")
        return None

    fig, axes = plt.subplots(1, len(arch_items), figsize=(6 * len(arch_items), 4.5))
    if len(arch_items) == 1:
        axes = [axes]

    any_curve = False
    used_step0 = True
    for ax, (arch, cfg) in zip(axes, arch_items):
        step0 = _load_step0_target_confidence(results_root, arch, mech_arch_tags)
        if step0 is None:
            used_step0 = False
        plotted = 0
        for method in cfg["methods"]:
            curve_path = results_root / arch / ("occlusion_%s_curve.npy" % method)
            mean, std, has_step0 = _occlusion_mean_std_curve(curve_path, step0)
            if mean is None:
                continue
            if step0 is not None and not has_step0:
                used_step0 = False
            x = np.arange(len(mean)) if has_step0 else np.arange(1, len(mean) + 1)
            _plot_occlusion_curve_band(
                ax, x, mean, std, _method_display_label(method)
            )
            plotted += 1
            any_curve = True
        title = cfg.get("title", arch)
        ax.set_title("%s — blurred-patch deletion" % title)
        ax.set_xlabel("Occlusion step")
        ax.set_ylabel("Mean target softmax confidence")
        ax.set_ylim(0.0, 1.0)
        if plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No occlusion curve files found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    if not any_curve:
        plt.close(fig)
        print("No occlusion curve files found for faithfulness plot.")
        return None

    if not used_step0:
        print(
            "Warning: mechanistic logits or target_indices missing; "
            "plotting occlusion steps 1–30 only (no step 0)."
        )

    fig.suptitle(
        "Blurred-occlusion faithfulness (lower curves = more faithful)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out_path = figures_dir / "occlusion_faithfulness_curves.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
