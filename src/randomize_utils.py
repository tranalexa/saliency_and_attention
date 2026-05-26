"""Utilities for cascading model weight randomization (Adebayo et al. sanity checks)."""
from __future__ import annotations

import copy
import re
from typing import Dict, List

import torch
import torch.nn as nn


def reset_layer(model: nn.Module, layer_name: str) -> None:
    """Reinitialize weights (Kaiming) and zero biases for a layer or block prefix."""
    prefix = layer_name if layer_name.endswith(".") else layer_name + "."
    matched = False
    for name, param in model.named_parameters():
        if name == layer_name or name.startswith(prefix):
            matched = True
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
    if not matched:
        module = model.get_submodule(layer_name)
        for pname, param in module.named_parameters(recurse=False):
            if "weight" in pname and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in pname:
                nn.init.zeros_(param)
            elif "weight" in pname:
                nn.init.normal_(param, mean=0.0, std=0.02)


_VIT_BLOCK_MODULE_RE = re.compile(r"^(?:(.+)\.)?blocks\.(\d+)$")


def _vit_classifier_layer(model: nn.Module) -> str | None:
    """Submodule path for the ImageNet classifier head (timm or hub DINOv2)."""
    for name, _ in model.named_parameters():
        if name.endswith("linear_head.weight"):
            return name[: -len(".weight")]
        if name.endswith("head.weight"):
            return name[: -len(".weight")]
        if name.endswith("fc.weight"):
            return name[: -len(".weight")]
    return None


def vit_blocks_prefix(model: nn.Module) -> str:
    """Canonical blocks path prefix (``blocks`` for timm, ``backbone.blocks`` for hub DINOv2)."""
    prefix_counts: dict[str, int] = {}
    indices_by_prefix: dict[str, set[int]] = {}
    for name, _ in model.named_modules():
        m = _VIT_BLOCK_MODULE_RE.match(name)
        if not m:
            continue
        parent = m.group(1)
        prefix = f"{parent}.blocks" if parent else "blocks"
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        indices_by_prefix.setdefault(prefix, set()).add(int(m.group(2)))
    if not prefix_counts:
        raise ValueError("Could not find ViT blocks in model.")
    return max(
        prefix_counts,
        key=lambda p: (len(indices_by_prefix[p]), prefix_counts[p], p.startswith("backbone")),
    )


def get_vit_block_names(model: nn.Module) -> List[str]:
    """Return cascade order: classifier first, then ViT blocks top-to-bottom (Adebayo-style)."""
    prefix = vit_blocks_prefix(model)
    max_idx = max(
        int(m.group(2))
        for name, _ in model.named_modules()
        if (m := _VIT_BLOCK_MODULE_RE.match(name))
        and (f"{m.group(1)}.blocks" if m.group(1) else "blocks") == prefix
    )
    order: List[str] = []
    head_layer = _vit_classifier_layer(model)
    if head_layer:
        order.append(head_layer)
    order.extend(f"{prefix}.{i}" for i in range(max_idx, -1, -1))
    return order


def _resnet_block_sort_key(block_name: str):
    parts = block_name.split(".")
    layer = parts[0]
    block = int(parts[1]) if len(parts) > 1 else 0
    stage_order = {"layer4": 0, "layer3": 1, "layer2": 2, "layer1": 3}
    return (stage_order.get(layer, 99), -block)


def get_resnet_block_names(model: nn.Module) -> List[str]:
    """Return cascade order: fc first, then full Bottleneck modules top-to-bottom (Adebayo-style)."""
    blocks = set()
    for name, _ in model.named_parameters():
        m = re.match(r"^(layer\d+)\.(\d+)\.", name)
        if m:
            blocks.add("%s.%s" % (m.group(1), m.group(2)))
    block_names = sorted(blocks, key=_resnet_block_sort_key)
    order: List[str] = []
    if any(n == "fc.weight" or n.startswith("fc.") for n, _ in model.named_parameters()):
        order.append("fc")
    order.extend(block_names)
    return order


def get_resnet_conv1_names(model: nn.Module) -> List[str]:
    """Alias for get_resnet_block_names (full Bottleneck modules)."""
    return get_resnet_block_names(model)


def save_checkpoint(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Deep copy of model state_dict for cascading restore."""
    return copy.deepcopy(model.state_dict())


def restore_checkpoint(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Restore model weights from a saved checkpoint."""
    model.load_state_dict(state_dict, strict=False)
