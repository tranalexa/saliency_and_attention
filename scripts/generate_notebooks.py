"""Generate scoped project notebooks.

This helper intentionally covers only the active ResNet-50 / ViT-B/16 scope.
It does not generate DINOv2 or removed-method notebooks.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def save(name: str, notebook: dict) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_DIR / name, "w") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")


COMMON_SETUP = """import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
"""


def arch_notebook(title: str, runner: str, results_subdir: str, extra_call_args: str = "") -> dict:
    return make_notebook(
        [
            markdown(f"# {title}\n\nCascading model randomization sanity checks."),
            code(
                COMMON_SETUP
                + f"""
from experiment_utils import {runner}

IMAGENET_ROOT = Path(os.environ.get("IMAGENET_ROOT", "/path/to/imagenet"))
RESULTS_DIR = PROJECT_ROOT / "results" / "{results_subdir}"
NUM_IMAGES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 8
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
IG_STEPS = 50
SKIP_QUAL = False

{runner}(
    imagenet_root=IMAGENET_ROOT,
    results_dir=RESULTS_DIR,
    num_images=NUM_IMAGES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    ig_steps=IG_STEPS,
    skip_qual=SKIP_QUAL,{extra_call_args}
)
print("Finished:", RESULTS_DIR)
"""
            ),
        ]
    )


def mechanistic_notebook() -> dict:
    return make_notebook(
        [
            markdown("# Mechanistic Checks\n\nLogit correlation and activation scales."),
            code(
                COMMON_SETUP
                + """
from experiment_utils import run_mechanistic_pipeline

IMAGENET_ROOT = Path(os.environ.get("IMAGENET_ROOT", "/path/to/imagenet"))
RESULTS_DIR = PROJECT_ROOT / "results" / "mechanistic"
NUM_IMAGES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

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
    )


def main() -> None:
    save("notebook_resnet50_cascading.ipynb", arch_notebook("ResNet-50 Cascading Randomization", "run_resnet50_pipeline", "resnet50"))
    save(
        "notebook_vit_cascading.ipynb",
        arch_notebook(
            "ViT-B/16 Cascading Randomization",
            "run_vit_pipeline",
            "vit",
            extra_call_args='\n    model_name="vit_base_patch16_224",\n    grid_size=14,',
        ),
    )
    save("notebook_mechanistic.ipynb", mechanistic_notebook())
    print("Generated scoped notebooks in", NOTEBOOK_DIR)


if __name__ == "__main__":
    main()
