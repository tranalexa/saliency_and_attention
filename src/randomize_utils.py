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


def get_vit_block_names(model: nn.Module) -> List[str]:
    """Return cascade order: classifier first, then ViT blocks top-to-bottom (Adebayo-style)."""
    block_indices = set()
    for name, _ in model.named_modules():
        m = re.match(r"blocks\.(\d+)$", name)
        if m:
            block_indices.add(int(m.group(1)))
    if not block_indices:
        raise ValueError("Could not find ViT blocks in model.")
    max_idx = max(block_indices)
    order: List[str] = []
    if any(n.startswith("head") for n, _ in model.named_parameters()):
        order.append("head")
    elif any(n.startswith("fc") for n, _ in model.named_parameters()):
        order.append("fc")
    order.extend(f"blocks.{i}" for i in range(max_idx, -1, -1))
    return order


def get_resnet_conv1_names(model: nn.Module) -> List[str]:
    """Return cascade order: classifier (fc) first, then Bottleneck conv1 top-to-bottom (Adebayo-style)."""
    conv1_names = []
    for name, module in model.named_modules():
        if name.endswith(".conv1") and isinstance(module, nn.Conv2d):
            if name.split(".")[0].startswith("layer"):
                conv1_names.append(name)
    stage_order = {"layer4": 0, "layer3": 1, "layer2": 2, "layer1": 3}

    def sort_key(n: str):
        parts = n.split(".")
        layer = parts[0]
        block = int(parts[1]) if len(parts) > 1 else 0
        return (stage_order.get(layer, 99), -block)

    conv1_names.sort(key=sort_key)
    order: List[str] = []
    if any(n == "fc.weight" or n.startswith("fc.") for n, _ in model.named_parameters()):
        order.append("fc")
    order.extend(conv1_names)
    return order


def save_checkpoint(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Deep copy of model state_dict for cascading restore."""
    return copy.deepcopy(model.state_dict())


def restore_checkpoint(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Restore model weights from a saved checkpoint."""
    model.load_state_dict(state_dict, strict=False)
