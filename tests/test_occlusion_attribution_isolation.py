"""Regression tests: occlusion must not mutate inputs used for attribution."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import (  # noqa: E402
    _apply_patch_replacement,
    _attribution_batch,
    _blurred_deletion_curve,
    _patch_slices,
)


class _ConstantLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1000)


def test_blurred_deletion_curve_does_not_mutate_source_batch() -> None:
    batch = torch.randn(2, 3, 224, 224)
    snapshot = batch.clone()
    blurred = batch * 0.2
    ranked = _patch_slices(224, 224, 15, 15)[:30]
    model = _ConstantLogitModel()

    _blurred_deletion_curve(
        model,
        batch[0],
        blurred[0],
        target_class=0,
        ranked_patches=ranked,
        num_patches=30,
        eval_batch_size=8,
        device="cpu",
    )

    assert torch.equal(batch, snapshot)


def test_blurred_deletion_curve_supports_batched_tensors() -> None:
    image = torch.randn(1, 3, 224, 224)
    blurred = image * 0.1
    ranked = _patch_slices(224, 224, 15, 15)[:5]
    model = _ConstantLogitModel()

    confidences, curve = _blurred_deletion_curve(
        model,
        image,
        blurred,
        target_class=0,
        ranked_patches=ranked,
        num_patches=5,
        eval_batch_size=8,
        device="cpu",
    )

    assert confidences.shape[0] == 6
    assert curve.shape[0] == 5


def test_apply_patch_replacement_uses_spatial_dims_for_batched_input() -> None:
    current = torch.zeros(1, 3, 224, 224)
    blurred = torch.ones(1, 3, 224, 224)
    ys, xs = slice(0, 15), slice(0, 15)
    _apply_patch_replacement(current, blurred, ys, xs)
    assert current[0, 0, 0, 0].item() == 1.0
    assert current[0, 0, 20, 20].item() == 0.0


def test_attribution_batch_clones_before_device_move() -> None:
    source = torch.randn(2, 3, 8, 8)
    snapshot = source.clone()
    batch = _attribution_batch(source, "cpu")
    batch += 1.0
    assert torch.equal(source, snapshot)
    assert not torch.equal(batch, source.to("cpu"))


def test_occluded_input_changes_block_means_relative_to_clean() -> None:
    """Document expected failure mode if attribution accidentally uses occluded pixels."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((224, 224))
    occluded = clean.copy()
    occluded[:15, :15] = 999.0

    def block_means(m: np.ndarray) -> np.ndarray:
        ps = 15
        return np.array(
            [
                m[y : y + ps, x : x + ps].mean()
                for y in range(0, 224, ps)
                for x in range(0, 224, ps)
            ]
        )

    assert not np.allclose(block_means(occluded), block_means(clean))
