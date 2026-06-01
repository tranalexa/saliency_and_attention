"""Metrics for comparing saliency maps and model outputs."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity


DEFAULT_RATIO_THRESHOLDS = (0.3, 0.5, 0.7)


def abs_grayscale_norm(img: np.ndarray) -> np.ndarray:
    """Absolute value normalize a 2D array to [0, 1]."""
    assert isinstance(img, np.ndarray)
    if img.ndim != 2:
        raise ValueError("Expected 2D array, got shape %s" % (img.shape,))
    img = np.abs(img.astype(np.float64))
    mx = img.max()
    if mx > 0:
        img = img / mx
    return img


def normalize_maxabs(img: np.ndarray) -> np.ndarray:
    """Legacy signed max-absolute normalization for visualization alignment."""
    arr = np.asarray(img, dtype=np.float64)
    mx = np.abs(arr).max()
    if mx > 0:
        return arr / mx
    return arr.copy()


def normalize_rms(img: np.ndarray) -> np.ndarray:
    """Primary second-moment normalization: x / sqrt(mean(x**2))."""
    arr = np.asarray(img, dtype=np.float64)
    rms = np.sqrt(np.mean(arr**2))
    if rms > 0:
        return arr / rms
    return arr.copy()


def diverging_norm(img: np.ndarray) -> np.ndarray:
    """Normalize signed 2D array by max absolute value."""
    assert isinstance(img, np.ndarray)
    if img.ndim != 2:
        raise ValueError("Expected 2D array, got shape %s" % (img.shape,))
    mx = np.abs(img).max()
    if mx > 0:
        img = img.astype(np.float64) / mx
    return img


def compute_spearman(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Spearman rank correlation between two HxW maps in [0, 1]."""
    a = np.asarray(map_a, dtype=np.float64).ravel()
    b = np.asarray(map_b, dtype=np.float64).ravel()
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    result = spearmanr(a, b)
    return float(result.correlation)


def compute_ssim(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Structural similarity between two HxW maps in [0, 1]."""
    return float(
        structural_similarity(
            np.asarray(map_a, dtype=np.float64),
            np.asarray(map_b, dtype=np.float64),
            data_range=1.0,
        )
    )


def compute_logit_correlation(
    logits_orig: np.ndarray, logits_rand: np.ndarray
) -> float:
    """Mean per-image Pearson correlation between logit vectors."""
    orig = np.asarray(logits_orig, dtype=np.float64)
    rand = np.asarray(logits_rand, dtype=np.float64)
    if orig.ndim == 1:
        orig = orig[None, :]
        rand = rand[None, :]
    corrs = []
    for i in range(orig.shape[0]):
        if np.std(orig[i]) == 0 or np.std(rand[i]) == 0:
            corrs.append(np.nan)
        else:
            corrs.append(pearsonr(orig[i], rand[i]).statistic)
    return float(np.nanmean(corrs))


def _finite_curve(
    values: np.ndarray, fractional_depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    depths = np.asarray(fractional_depths, dtype=np.float64).ravel()
    if vals.shape != depths.shape:
        raise ValueError(
            "values and fractional_depths must have the same shape, got %s and %s"
            % (vals.shape, depths.shape)
        )
    mask = np.isfinite(vals) & np.isfinite(depths)
    return vals[mask], depths[mask]


def _first_crossing_below(
    values: np.ndarray, fractional_depths: np.ndarray, threshold: float
) -> float:
    vals, depths = _finite_curve(values, fractional_depths)
    if vals.size == 0:
        return float("nan")
    below = np.where(vals < threshold)[0]
    if below.size == 0:
        return float("inf")
    idx = int(below[0])
    if idx == 0:
        return float(depths[0])

    prev_val, curr_val = vals[idx - 1], vals[idx]
    prev_depth, curr_depth = depths[idx - 1], depths[idx]
    if prev_val == curr_val:
        return float(curr_depth)

    frac = (threshold - prev_val) / (curr_val - prev_val)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(prev_depth + frac * (curr_depth - prev_depth))


def characterize_cascade_curve(
    similarity_mean: np.ndarray,
    fractional_depths: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Summarize a mean cascade similarity curve."""
    vals, depths = _finite_curve(similarity_mean, fractional_depths)
    if vals.size == 0:
        return {
            "d_half": float("nan"),
            "early_slope": float("nan"),
            "final_sim": float("nan"),
        }

    d_half = _first_crossing_below(vals, depths, threshold)
    depth_min, depth_max = float(depths.min()), float(depths.max())
    depth_span = depth_max - depth_min

    early_mask = depths <= depth_min + 0.5 * depth_span
    if np.count_nonzero(early_mask) >= 2:
        early_slope = float(np.polyfit(depths[early_mask], vals[early_mask], deg=1)[0])
    else:
        early_slope = float("nan")

    final_cutoff = depth_max - 0.1 * depth_span
    final_mask = depths >= final_cutoff
    final_sim = float(np.nanmean(vals[final_mask])) if np.any(final_mask) else float("nan")

    return {
        "d_half": float(d_half),
        "early_slope": early_slope,
        "final_sim": final_sim,
    }


def compute_curve_auc(
    similarity_mean: np.ndarray,
    fractional_depths: np.ndarray,
    up_to: float | None = None,
) -> float:
    """Normalized trapezoidal AUC, optionally truncated at a fractional depth."""
    vals, depths = _finite_curve(similarity_mean, fractional_depths)
    if vals.size < 2:
        return float("nan")
    order = np.argsort(depths)
    depths = depths[order]
    vals = vals[order]

    if up_to is not None and np.isfinite(up_to):
        up_to = float(up_to)
        if up_to <= depths[0]:
            return float(vals[0])
        if up_to < depths[-1]:
            interp_val = float(np.interp(up_to, depths, vals))
            keep = depths < up_to
            depths = np.concatenate([depths[keep], np.array([up_to])])
            vals = np.concatenate([vals[keep], np.array([interp_val])])

    width = float(depths[-1] - depths[0])
    if width <= 0:
        return float("nan")
    return float(np.trapezoid(vals, depths) / width)


def compute_sensitivity_ratio(
    d_half: float,
    logit_corr: np.ndarray,
    fractional_depths: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Ratio of attribution response depth to model-output response depth."""
    d_arch = _first_crossing_below(logit_corr, fractional_depths, threshold)
    if not np.isfinite(d_arch):
        warnings.warn(
            "logit correlation never crossed threshold %.3f; sensitivity ratio is undefined"
            % threshold,
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    if d_arch == 0.0:
        warnings.warn(
            "D_arch is 0.0 at threshold %.3f; returning infinity" % threshold,
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")
    return float(d_half / d_arch)


def characterize_sensitivity_thresholds(
    similarity_mean: np.ndarray,
    logit_corr: np.ndarray,
    fractional_depths: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_RATIO_THRESHOLDS,
) -> dict[str, dict[str, float]]:
    """Evaluate D_half, D_arch, and sensitivity ratio across thresholds."""
    out: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        d_half = _first_crossing_below(similarity_mean, fractional_depths, threshold)
        d_arch = _first_crossing_below(logit_corr, fractional_depths, threshold)
        if d_arch == 0.0:
            ratio = float("inf")
        elif np.isfinite(d_arch):
            ratio = float(d_half / d_arch)
        else:
            ratio = float("nan")
        out[str(threshold)] = {
            "d_half": float(d_half),
            "d_arch": float(d_arch),
            "sensitivity_ratio": ratio,
        }
    return out
