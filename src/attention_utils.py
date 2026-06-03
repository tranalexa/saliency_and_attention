"""ViT attention-based explanation maps (Abnar attention rollout, etc.)."""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics_utils import normalize_rms
from randomize_utils import vit_blocks_prefix


def _get_attn_modules_with_indices(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    """Collect attention modules from timm ViT blocks in order."""
    prefix = vit_blocks_prefix(model)
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.attn$")
    modules = []
    for name, module in model.named_modules():
        m = pat.match(name)
        if m:
            modules.append((int(m.group(1)), name, module))
    modules.sort(key=lambda x: x[0])
    return modules


def _get_attn_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Collect attention modules from timm ViT blocks in order."""
    modules = _get_attn_modules_with_indices(model)
    return [(n, m) for _, n, m in modules]


def _attn_head_dim(module: nn.Module) -> int:
    if hasattr(module, "head_dim"):
        return int(module.head_dim)
    return module.qkv.in_features // module.num_heads


def _compute_attention_probs(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute (B, H, N, N) attention probabilities from timm ViT Attention."""
    b, n, _ = x.shape
    head_dim = _attn_head_dim(module)
    qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, _v = qkv.unbind(0)
    if hasattr(module, "q_norm"):
        q, k = module.q_norm(q), module.k_norm(k)
    scale = module.scale if hasattr(module, "scale") else head_dim**-0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    return attn


def _make_capture_hook(storage: list):
    def hook(module, inp, _output):
        storage.append(_compute_attention_probs(module, inp[0]))

    return hook


def validate_attention_weight_tensor(attn_weights: torch.Tensor, name: str = "") -> None:
    """
    Validate that captured per-layer attention weights are post-softmax probabilities.

    Expected shape: (batch, num_heads, num_tokens, num_tokens). Each attention
    row should be non-negative and sum to ~1.0.
    """
    if attn_weights.dim() != 4:
        raise ValueError(
            "Attention weight tensor %s has shape %s, expected (batch, heads, tokens, tokens)."
            % (name, tuple(attn_weights.shape))
        )
    row_sums = attn_weights.sum(dim=-1)
    mean_sum = row_sums.mean().item()
    if abs(mean_sum - 1.0) > 0.05:
        raise ValueError(
            "Attention weight tensor %s row sums to %.4f, expected ~1.0. You are likely "
            "hooking the wrong submodule and capturing pre-softmax logits or value "
            "projections. The hook should be on block.attn at the point after softmax "
            "is applied to the QK product."
            % (name, mean_sum)
        )
    if (attn_weights < -1e-6).any():
        raise ValueError(
            "Attention weight tensor %s contains negative values. Post-softmax attention "
            "weights must be non-negative." % name
        )


def validate_attention_rollout(model: nn.Module, input_tensor: torch.Tensor, name: str = "") -> None:
    """Validate post-softmax attention capture on the last block (hook sanity check).

    Full ``get_attention_rollout`` uses all blocks; this check stays fast at startup.
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        raise ValueError("No attention modules found in model.")
    module_name, last_module = attn_modules[-1]
    captured: List[torch.Tensor] = []
    handle = last_module.register_forward_hook(_make_capture_hook(captured))
    model.eval()
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        handle.remove()
    if not captured:
        raise ValueError(
            "Attention weight tensor %s was not captured." % (name or module_name)
        )
    validate_attention_weight_tensor(captured[0], name=name or module_name)


_ATTENTION_ROLLOUT_SIGNAL_THRESHOLD = 1e-6


def _cls_to_spatial_map(
    cls_weights: np.ndarray, grid_size: int, out_size: int = 224, apply_rms_norm: bool = True
) -> np.ndarray:
    """Map CLS-to-patch weights to a 2D heatmap with primary RMS normalization.

    Normalizing a near-zero map would amplify numerical noise into a spuriously
    structured map, so return a uniform map when signal is absent.
    """
    n_patches = cls_weights.shape[0]
    expected = grid_size * grid_size
    if n_patches != expected:
        grid_size = int(np.sqrt(n_patches))
    heatmap = cls_weights.reshape(grid_size, grid_size)
    if np.abs(heatmap).max() < _ATTENTION_ROLLOUT_SIGNAL_THRESHOLD:
        return np.full((out_size, out_size), 0.5)
    heatmap_t = torch.from_numpy(heatmap).float()[None, None, ...]
    upsampled = F.interpolate(
        heatmap_t, size=(out_size, out_size), mode="bilinear", align_corners=False
    )
    out = upsampled.squeeze().numpy()
    if apply_rms_norm:
        return normalize_rms(out)
    return out.astype(np.float64)


def compute_attention_entropy(attn: torch.Tensor) -> float:
    """Mean entropy of attention distributions."""
    entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
    return float(entropy.detach().cpu().item())


def _cls_per_head_spatial_maps(
    attn: torch.Tensor,
    grid_size: int,
    out_size: int = 224,
    apply_rms_norm: bool = True,
) -> np.ndarray:
    cls_by_head = attn[0, :, 0, 1:].detach().cpu().numpy()
    return np.stack(
        [
            _cls_to_spatial_map(
                cls_weights,
                grid_size,
                out_size=out_size,
                apply_rms_norm=apply_rms_norm,
            )
            for cls_weights in cls_by_head
        ]
    )


def _attention_rollout(layer_attns: List[torch.Tensor]) -> torch.Tensor:
    """
    Abnar & Zuidema (ACL 2020) attention rollout, Equation 1.

    Per layer: head-mean, 0.5*Watt + 0.5*I, row-normalize, then Ã_i = A_i @ Ã_{i-1}.
    """
    attn_bar = None
    for attn in layer_attns:
        validate_attention_weight_tensor(attn, name="attention_rollout")
        b, n = attn.shape[0], attn.shape[-1]
        watt = attn.mean(dim=1)
        eye = torch.eye(n, device=watt.device, dtype=watt.dtype).expand(b, n, n)
        a = 0.5 * watt + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_bar = a if attn_bar is None else a @ attn_bar
    if attn_bar is None:
        raise ValueError("No attention layers captured for rollout.")
    return attn_bar


def get_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    grid_size: int | None = None,
    out_size: int = 224,
    return_entropy: bool = False,
    apply_rms_norm: bool = True,
    return_per_head: bool = False,
    head_aggregation: str = "auto",
) -> np.ndarray | tuple[np.ndarray, float] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, float, np.ndarray]:
    """
    Abnar & Zuidema (ACL 2020) attention rollout over all ViT blocks (Eq. 1).

    Hooks every block, chains layer attention with residual identity and row-normalization,
    then maps CLS-to-patch rollout weights to a spatial heatmap.

    ``head_aggregation`` does not affect the main rollout map. ``return_entropy`` and
    ``return_per_head`` still use the last block only (pipeline compatibility).
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        raise ValueError("No attention modules found in model.")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm ViT with model.blocks.")

    captured: List[torch.Tensor] = []
    handles = [
        module.register_forward_hook(_make_capture_hook(captured))
        for _, module in attn_modules
    ]
    model.eval()
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != len(blocks):
        raise ValueError(
            "Expected %d attention captures for rollout, got %d."
            % (len(blocks), len(captured))
        )

    last_attn = captured[-1]
    attn_bar = _attention_rollout(captured)
    cls_weights = attn_bar[0, 0, 1:].detach().cpu().numpy()
    if grid_size is None:
        grid_size = int(np.sqrt(cls_weights.shape[0]))
    per_head = _cls_per_head_spatial_maps(
        last_attn, grid_size, out_size=out_size, apply_rms_norm=apply_rms_norm
    )
    spatial_map = _cls_to_spatial_map(
        cls_weights, grid_size, out_size, apply_rms_norm=apply_rms_norm
    )
    entropy = compute_attention_entropy(last_attn)
    if return_entropy and return_per_head:
        return spatial_map, entropy, per_head
    if return_entropy:
        return spatial_map, entropy
    if return_per_head:
        return spatial_map, per_head
    return spatial_map
