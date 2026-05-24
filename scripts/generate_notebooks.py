#!/usr/bin/env python3
"""Generate PyTorch sanity-check notebooks (thin wrappers over experiment_utils)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def cell_code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text,
    }


def notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def save(name, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1))
    print("wrote", path)


IMPORTS = '''import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import (
    run_resnet50_pipeline,
    run_vit_pipeline,
    run_dinov2_pipeline,
    run_mechanistic_pipeline,
)
'''


def resnet_notebook():
    return [
        cell_md(
            "# ResNet-50 Cascading Randomization\n\n"
            "PyTorch replication for Adebayo et al. Methods match the original paper: "
            "Gradient, SmoothGrad, Input-Grad, GBP, GradCAM, GBP-GC, IG, IG-SmoothGrad."
        ),
        cell_code(
            IMPORTS
            + """
# ============ CONFIG ============
IMAGENET_ROOT = Path(os.environ.get("IMAGENET_ROOT", "/path/to/imagenet"))
RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50"
NUM_IMAGES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 8
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
IG_STEPS = 50
SKIP_QUAL = False
# ================================

run_resnet50_pipeline(
    imagenet_root=IMAGENET_ROOT,
    results_dir=RESULTS_DIR,
    num_images=NUM_IMAGES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    ig_steps=IG_STEPS,
    skip_qual=SKIP_QUAL,
)
print("Finished:", RESULTS_DIR)
"""
        ),
    ]


def vit_notebook(title, results_subdir, model_name, grid_size, dinov2=False):
    fn = "run_dinov2_pipeline" if dinov2 else "run_vit_pipeline"
    extra = ""
    if not dinov2:
        extra = f"""
run_vit_pipeline(
    imagenet_root=IMAGENET_ROOT,
    results_dir=RESULTS_DIR,
    model_name="{model_name}",
    grid_size={grid_size},
    num_images=NUM_IMAGES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    ig_steps=IG_STEPS,
    skip_qual=SKIP_QUAL,
)
"""
    else:
        extra = """
run_dinov2_pipeline(
    imagenet_root=IMAGENET_ROOT,
    results_dir=RESULTS_DIR,
    num_images=NUM_IMAGES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    ig_steps=IG_STEPS,
    skip_qual=SKIP_QUAL,
)
"""
    return [
        cell_md("# %s Cascading Randomization\n\nViT sanity checks with Captum + attention maps." % title),
        cell_code(
            IMPORTS
            + """
# ============ CONFIG ============
IMAGENET_ROOT = Path(os.environ.get("IMAGENET_ROOT", "/path/to/imagenet"))
RESULTS_DIR = PROJECT_ROOT / "results" / "%s"
NUM_IMAGES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 8
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
IG_STEPS = 50
SKIP_QUAL = False
# ================================
"""
            % results_subdir
            + extra
            + 'print("Finished:", RESULTS_DIR)\n'
        ),
    ]


def mechanistic_notebook():
    return [
        cell_md("# Mechanistic Checks\n\nLogit correlation and activation scale distributions (ResNet-50 vs ViT-B/16)."),
        cell_code(
            IMPORTS
            + """
# ============ CONFIG ============
IMAGENET_ROOT = Path(os.environ.get("IMAGENET_ROOT", "/path/to/imagenet"))
RESULTS_DIR = PROJECT_ROOT / "results" / "mechanistic"
NUM_IMAGES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
# ================================

run_mechanistic_pipeline(
    imagenet_root=IMAGENET_ROOT,
    results_dir=RESULTS_DIR,
    num_images=NUM_IMAGES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)
print("Finished:", RESULTS_DIR)
"""
        ),
    ]


def analysis_notebook():
    return [
        cell_md("# Analysis Notebook\n\nLoad saved numpy results and generate figures (no model loading)."),
        cell_code(
            """
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "results").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = {
    "resnet50": {
        "methods": [
            "gradient", "smoothgrad", "input_grad", "ig", "ig_smoothgrad", "gradcam",
            "gbp", "gbp_gc",
        ],
        "title": "ResNet-50",
    },
    "vit": {
        "methods": [
            "gradient", "smoothgrad", "input_grad", "ig", "ig_smoothgrad", "gradcam",
            "raw_attn", "rollout",
        ],
        "title": "ViT-B/16",
    },
    "dinov2": {
        "methods": [
            "gradient", "smoothgrad", "input_grad", "ig", "ig_smoothgrad", "gradcam",
            "raw_attn", "rollout",
        ],
        "title": "DINOv2-B",
    },
}
SHARED_METHODS = [
    "gradient", "smoothgrad", "input_grad", "ig", "ig_smoothgrad", "gradcam",
]
CROSS_ARCH_LABELS = {
    "resnet50": "ResNet-50",
    "vit": "ViT-B/16",
    "dinov2": "DINOv2-B",
}
"""
        ),
        cell_code(
            """
def plot_metric_curves(metric_suffix, ylabel, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, (arch, cfg) in zip(axes, ARCHS.items()):
        d = RESULTS_ROOT / arch
        if not d.exists():
            ax.set_title(cfg["title"] + " (missing)")
            continue
        for method in cfg["methods"]:
            path = d / ("%s_%s.npy" % (method, metric_suffix))
            if not path.exists():
                continue
            vals = np.load(path)
            ax.plot(range(len(vals)), vals, marker="o", label=method, markersize=3)
        ax.set_xlabel("Randomization depth")
        ax.set_ylabel(ylabel)
        ax.set_title(cfg["title"])
        ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=150)
    plt.show()

plot_metric_curves("spearman_mean", "Spearman correlation", "spearman_curves.png")
plot_metric_curves("ssim_mean", "SSIM", "ssim_curves.png")
"""
        ),
        cell_code(
            """
def plot_cross_arch_curves(metric_suffix, ylabel, fname):
    n_methods = len(SHARED_METHODS)
    ncols = 3
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, method in zip(axes, SHARED_METHODS):
        for arch, label in CROSS_ARCH_LABELS.items():
            path = RESULTS_ROOT / arch / ("%s_%s.npy" % (method, metric_suffix))
            if not path.exists():
                continue
            vals = np.load(path)
            ax.plot(range(len(vals)), vals, marker="o", label=label, markersize=3)
        ax.set_xlabel("Randomization depth")
        ax.set_ylabel(ylabel)
        ax.set_title(method)
        ax.legend(fontsize=7)
    for ax in axes[n_methods:]:
        ax.axis("off")
    fig.suptitle("Cross-architecture comparison (%s)" % ylabel, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()

plot_cross_arch_curves("spearman_mean", "Spearman correlation", "cross_arch_spearman.png")
plot_cross_arch_curves("ssim_mean", "SSIM", "cross_arch_ssim.png")
"""
        ),
        cell_code(
            """
mech = RESULTS_ROOT / "mechanistic"
if (mech / "logit_corr_resnet.npy").exists():
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, label in [("resnet", "ResNet-50"), ("vit", "ViT-B/16")]:
        p = mech / ("logit_corr_%s.npy" % tag)
        if p.exists():
            ax.plot(np.load(p), marker="o", label=label)
    ax.set_xlabel("Randomization depth")
    ax.set_ylabel("Mean logit Pearson r")
    ax.set_title("Logit correlation under cascading randomization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_correlation.png", dpi=150)
    plt.show()
"""
        ),
        cell_code(
            """
for tag in ["resnet", "vit"]:
    files = sorted(mech.glob("activation_scale_%s_depth*.npy" % tag))
    if len(files) < 2:
        continue
    a0, a1 = np.load(files[0]), np.load(files[-1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(a0, bins=50, alpha=0.5, density=True, label="depth 0")
    ax.hist(a1, bins=50, alpha=0.5, density=True, label="depth %d" % (len(files) - 1))
    ax.set_title("Activation |.| scales: %s" % tag)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / ("activation_scales_%s.png" % tag), dpi=150)
    plt.show()
"""
        ),
        cell_code(
            """
def plot_cascading_grid(arch, method, title):
    qual_path = RESULTS_ROOT / arch / "qual_bundle.npz"
    if not qual_path.exists():
        return
    data = np.load(qual_path, allow_pickle=True)
    key = "cascade_" + method
    if key not in data:
        return
    cascade = data[key]
    order = list(data["order"])
    nrows = len(cascade) + 2
    fig = plt.figure(figsize=(4, 0.4 * nrows))
    gs = gridspec.GridSpec(nrows, 1)
    ax = fig.add_subplot(gs[0])
    ax.imshow(data["image"])
    ax.set_title("Input")
    ax.axis("off")
    ax = fig.add_subplot(gs[1])
    ax.imshow(data["baseline_" + method], cmap="gray")
    ax.set_title("Baseline (no randomization)")
    ax.axis("off")
    for i, m in enumerate(cascade):
        ax = fig.add_subplot(gs[i + 2])
        ax.imshow(m, cmap="gray")
        ax.set_title("Depth %d: %s" % (i, order[i] if i < len(order) else ""))
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / ("cascade_%s_%s.png" % (arch, method)), dpi=150)
    plt.show()

plot_cascading_grid("resnet50", "gbp", "ResNet-50 GBP (Adebayo replication check)")
plot_cascading_grid("resnet50", "input_grad", "ResNet-50 Input-Grad")
plot_cascading_grid("resnet50", "ig", "ResNet-50 IG")
plot_cascading_grid("vit", "ig", "ViT Integrated Gradients")
plot_cascading_grid("vit", "input_grad", "ViT Input-Grad")
plot_cascading_grid("vit", "raw_attn", "ViT raw attention")
plot_cascading_grid("dinov2", "rollout", "DINOv2 rollout")
print("Figures saved to", FIGURES_DIR)
"""
        ),
    ]


if __name__ == "__main__":
    save("notebook_resnet50_cascading.ipynb", resnet_notebook())
    save("notebook_vit_cascading.ipynb", vit_notebook("ViT-B/16", "vit", "vit_base_patch16_224", 14))
    save("notebook_dinov2_cascading.ipynb", vit_notebook("DINOv2-B", "dinov2", "vit_base_patch14_dinov2.lvd142m", 16, dinov2=True))
    save("notebook_mechanistic.ipynb", mechanistic_notebook())
    save("notebook_analysis.ipynb", analysis_notebook())
