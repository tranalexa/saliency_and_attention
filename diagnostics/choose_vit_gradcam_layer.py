"""Empirically choose a ViT Transformer GradCAM target layer.

This is a one-off diagnostic. It does not import or use the main pipeline's
Transformer GradCAM fallback; each candidate layer is evaluated directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_utils import (  # noqa: E402
    build_transform,
    denormalize,
    get_target_indices,
    load_all_images,
    load_imagenet_subset,
    set_seed,
    validate_imagenet_root,
    vit_reshape_transform,
)
from randomize_utils import (  # noqa: E402
    get_vit_block_names,
    reset_layer,
    restore_checkpoint,
    save_checkpoint,
)


CANDIDATES = [
    "blocks[-1].norm2",
    "blocks[-1]",
    "blocks[-2].norm2",
    "blocks[-2]",
    "blocks[-3]",
]


@dataclass(frozen=True)
class StateSpec:
    label: str
    cascade_depth: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose the latest ViT GradCAM layer with non-degenerate maps."
    )
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degenerate-std", type=float, default=1e-8)
    parser.add_argument("--pass-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=CANDIDATES,
        help="Candidate target-layer names to evaluate.",
    )
    return parser.parse_args()


def candidate_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "blocks[-1].norm2":
        return model.blocks[-1].norm2
    if name == "blocks[-1]":
        return model.blocks[-1]
    if name == "blocks[-2].norm2":
        return model.blocks[-2].norm2
    if name == "blocks[-2]":
        return model.blocks[-2]
    if name == "blocks[-3]":
        return model.blocks[-3]
    raise ValueError(f"Unknown candidate layer: {name}")


def compute_cam_direct(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
    out_size: int = 224,
) -> np.ndarray:
    """Run pytorch-grad-cam for one image and one target layer, with no fallback."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    with GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=image,
            targets=[ClassifierOutputTarget(int(target_class))],
        )
    heatmap_t = torch.tensor(grayscale_cam[0]).float()[None, None]
    up = F.interpolate(
        heatmap_t,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )
    return up.squeeze().numpy().astype(np.float64)


def apply_state(
    model: nn.Module,
    original_sd: dict[str, torch.Tensor],
    order: Sequence[str],
    state: StateSpec,
) -> None:
    restore_checkpoint(model, original_sd)
    if state.cascade_depth is None:
        return
    for layer_name in order[: state.cascade_depth + 1]:
        reset_layer(model, layer_name)


def choose_states(order: Sequence[str]) -> list[StateSpec]:
    if not order:
        raise ValueError("Empty ViT randomization order.")
    mid = max(0, len(order) // 2)
    deep = max(0, len(order) - 1)
    return [
        StateSpec("pretrained", None),
        StateSpec("cascade_depth0", 0),
        StateSpec(f"cascade_depth{mid}", mid),
        StateSpec(f"cascade_depth{deep}", deep),
    ]


def measure_candidate(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: nn.Module,
    device: str,
    degenerate_std: float,
) -> tuple[float, float, list[np.ndarray]]:
    stds: list[float] = []
    maps: list[np.ndarray] = []
    for i in tqdm(range(images.shape[0]), leave=False):
        image = images[i : i + 1].to(device)
        target = int(get_target_indices(model, image)[0].item())
        cam_map = compute_cam_direct(model, image, target, target_layer)
        maps.append(cam_map)
        stds.append(float(np.std(cam_map)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    std_arr = np.asarray(stds, dtype=np.float64)
    return float(np.mean(std_arr < degenerate_std)), float(np.mean(std_arr)), maps


def format_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def is_nonmonotonic(degenerate_by_state: Sequence[float], tolerance: float = 0.05) -> bool:
    """Flag large decreases in degeneracy as randomization gets deeper."""
    for prev, cur in zip(degenerate_by_state, degenerate_by_state[1:]):
        if cur + tolerance < prev:
            return True
    return False


def recommend_candidate(
    candidates: Sequence[str],
    rows: dict[str, dict[str, float]],
    state_labels: Sequence[str],
    pass_threshold: float,
) -> tuple[str | None, str]:
    for candidate in candidates:
        depth0_frac = rows[candidate]["pretrained_degenerate_frac"]
        deg_curve = [rows[candidate][f"{label}_degenerate_frac"] for label in state_labels]
        if depth0_frac >= pass_threshold:
            continue
        if is_nonmonotonic(deg_curve):
            continue
        return candidate, (
            f"Recommended target: `{candidate}`. It is the latest candidate with "
            f"pretrained degeneracy below {pass_threshold:.0%} and no large "
            "non-monotonic degeneracy decrease across tested cascade states."
        )
    return None, (
        "No candidate met the default rule. Inspect the qualitative figure and "
        "consider relaxing the threshold or stepping back to an earlier block."
    )


def write_report(
    output_path: Path,
    candidates: Sequence[str],
    states: Sequence[StateSpec],
    rows: dict[str, dict[str, float]],
    recommendation: str,
    config: dict,
) -> None:
    state_labels = [s.label for s in states]
    header = [
        "Candidate",
        *[f"{label} %degenerate" for label in state_labels],
        *[f"{label} mean std" for label in state_labels],
    ]
    lines = [
        "# ViT GradCAM Target-Layer Diagnostic",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Degeneracy Table",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for candidate in candidates:
        row = rows[candidate]
        values = [candidate]
        values.extend(
            format_float(100.0 * row[f"{label}_degenerate_frac"], precision=2) + "%"
            for label in state_labels
        )
        values.extend(format_float(row[f"{label}_mean_std"]) for label in state_labels)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Recommendation", "", recommendation, ""])
    output_path.write_text("\n".join(lines))


def save_qualitative_figure(
    output_path: Path,
    candidates: Sequence[str],
    example_images: torch.Tensor,
    example_maps: dict[str, list[np.ndarray]],
) -> None:
    n_examples = example_images.shape[0]
    n_cols = len(candidates) + 1
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(2.4 * n_cols, 2.4 * n_examples))
    if n_examples == 1:
        axes = np.expand_dims(axes, axis=0)
    for row in range(n_examples):
        image = denormalize(example_images[row : row + 1]).squeeze(0).permute(1, 2, 0)
        axes[row, 0].imshow(image.detach().cpu().numpy().clip(0, 1))
        axes[row, 0].set_title("input")
        axes[row, 0].axis("off")
        for col, candidate in enumerate(candidates, start=1):
            cam_map = example_maps[candidate][row]
            axes[row, col].imshow(cam_map, cmap="jet")
            axes[row, col].set_title(candidate)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_imagenet_root(args.imagenet_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, image_indices = load_imagenet_subset(
        args.imagenet_root,
        num_images=args.num_images,
        image_size=args.image_size,
        transform=build_transform(args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        img_size=args.image_size,
    ).to(args.device).eval()
    order = get_vit_block_names(model)
    original_sd = save_checkpoint(model)
    images, _targets, _gt_labels = load_all_images(loader, model, args.device)
    states = choose_states(order)
    state_labels = [s.label for s in states]

    rows: dict[str, dict[str, float]] = {candidate: {} for candidate in args.candidates}
    pretrained_maps_for_examples: dict[str, list[np.ndarray]] = {}

    for state in states:
        print(f"\n== {state.label} ==")
        apply_state(model, original_sd, order, state)
        for candidate in args.candidates:
            print(f"Measuring {candidate}")
            frac_deg, mean_std, maps = measure_candidate(
                model,
                images,
                candidate_layer(model, candidate),
                args.device,
                args.degenerate_std,
            )
            rows[candidate][f"{state.label}_degenerate_frac"] = frac_deg
            rows[candidate][f"{state.label}_mean_std"] = mean_std
            if state.cascade_depth is None:
                pretrained_maps_for_examples[candidate] = maps[: args.examples]

    _winner, recommendation = recommend_candidate(
        args.candidates,
        rows,
        state_labels,
        args.pass_threshold,
    )
    config = {
        "seed": args.seed,
        "num_images": args.num_images,
        "image_indices": image_indices,
        "degenerate_std": args.degenerate_std,
        "pass_threshold": args.pass_threshold,
        "candidate_order_latest_to_earliest": args.candidates,
        "states": [
            {"label": state.label, "cascade_depth": state.cascade_depth}
            for state in states
        ],
        "target_policy": "argmax for current model state",
    }
    report_path = args.output_dir / "vit_gradcam_layer_report.md"
    figure_path = args.output_dir / "vit_gradcam_layer_examples.png"
    write_report(report_path, args.candidates, states, rows, recommendation, config)
    save_qualitative_figure(
        figure_path,
        args.candidates,
        images[: args.examples],
        pretrained_maps_for_examples,
    )
    print("\n" + recommendation)
    print(f"Wrote report: {report_path}")
    print(f"Wrote examples: {figure_path}")


if __name__ == "__main__":
    main()
