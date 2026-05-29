"""ViT attention-based explanation maps (raw attention and rollout)."""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics_utils import abs_grayscale_norm
from randomize_utils import vit_blocks_prefix


def _get_attn_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Collect attention modules from timm or DINOv2 ViT blocks in order."""
    prefix = vit_blocks_prefix(model)
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.attn$")
    modules = []
    for name, module in model.named_modules():
        m = pat.match(name)
        if m:
            modules.append((int(m.group(1)), name, module))
    modules.sort(key=lambda x: x[0])
    return [(n, m) for _, n, m in modules]


def _attn_head_dim(module: nn.Module) -> int:
    if hasattr(module, "head_dim"):
        return int(module.head_dim)
    return module.qkv.in_features // module.num_heads


def _compute_attention_probs(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute (B, H, N, N) attention probabilities from timm or DINOv2 Attention."""
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


_ROLLOUT_SIGNAL_THRESHOLD = 1e-6


def _cls_to_spatial_map(
    cls_weights: np.ndarray, grid_size: int, out_size: int = 224
) -> np.ndarray:
    """Map CLS-to-patch weights to a 2D heatmap normalized to [0, 1].

    When the model is fully randomized the CLS→patch weights collapse toward
    zero. Dividing by a near-zero max (abs_grayscale_norm) would amplify
    numerical noise into a spuriously structured map that can anti-correlate
    with the semantic baseline. Instead, return a uniform map when the signal
    is below threshold.
    """
    n_patches = cls_weights.shape[0]
    expected = grid_size * grid_size
    if n_patches != expected:
        grid_size = int(np.sqrt(n_patches))
    heatmap = cls_weights.reshape(grid_size, grid_size)
    if np.abs(heatmap).max() < _ROLLOUT_SIGNAL_THRESHOLD:
        return np.full((out_size, out_size), 0.5)
    heatmap_t = torch.from_numpy(heatmap).float()[None, None, ...]
    upsampled = F.interpolate(
        heatmap_t, size=(out_size, out_size), mode="bilinear", align_corners=False
    )
    return abs_grayscale_norm(upsampled.squeeze().numpy())


def _rollout_matrices(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """Attention rollout per Abnar & Zuidema (2020)."""
    device = attention_maps[0].device
    n_tokens = attention_maps[0].shape[-1]
    result = torch.eye(n_tokens, device=device)
    for attn in attention_maps:
        attn_mean = attn.mean(dim=1)
        attn_aug = attn_mean + torch.eye(n_tokens, device=device)
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        result = attn_aug @ result
    return result


def get_raw_attention(
    model: nn.Module,
    input_tensor: torch.Tensor,
    grid_size: int | None = None,
    out_size: int = 224,
) -> np.ndarray:
    """
    Raw attention from last block: average heads, CLS -> patches, spatial map.

    Returns HxW numpy array in [0, 1].
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        raise ValueError("No attention modules found in model.")
    _, last_module = attn_modules[-1]
    captured: List[torch.Tensor] = []
    handle = last_module.register_forward_hook(_make_capture_hook(captured))
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    handle.remove()

    attn = captured[0]
    cls_weights = attn[0].mean(dim=0)[0, 1:].detach().cpu().numpy()
    if grid_size is None:
        grid_size = int(np.sqrt(cls_weights.shape[0]))
    return _cls_to_spatial_map(cls_weights, grid_size, out_size)


def get_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    grid_size: int | None = None,
    out_size: int = 224,
) -> np.ndarray:
    """
    Attention rollout across all blocks; returns HxW map in [0, 1].
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        raise ValueError("No attention modules found in model.")
    captured: List[torch.Tensor] = []
    handles = [m.register_forward_hook(_make_capture_hook(captured)) for _, m in attn_modules]
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    for h in handles:
        h.remove()

    rollout = _rollout_matrices(captured)
    if rollout.dim() == 3:
        rollout = rollout[0]
    cls_weights = rollout[0, 1:].detach().cpu().numpy()
    if grid_size is None:
        grid_size = int(np.sqrt(cls_weights.shape[0]))
    return _cls_to_spatial_map(cls_weights, grid_size, out_size)
