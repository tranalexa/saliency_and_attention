"""Metrics for comparing saliency maps and model outputs."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity


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
