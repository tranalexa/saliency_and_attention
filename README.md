# Sanity Checks for Saliency Maps

Fork of adebayoj/sanity_checks_saliency (Adebayo et al., arXiv:1806.07529). This repo extends the original TensorFlow / Inception / MNIST replication with a PyTorch pipeline for ResNet-50 and ViT-B/16 on ImageNet, plus a blurred-patch occlusion faithfulness axis (Binder et al.). The original TensorFlow / Inception / MNIST replication remains under `notebooks/legacy_tf/` (see [legacy section](#legacy-tensorflow-replication) below).

## What this fork measures

**Cascade axis (Adebayo):** Cumulatively reinitialize weights from the classifier downward. At each depth, compare saliency on the randomized model to maps from the fully pretrained model (Spearman + SSIM). Methods that stay visually similar while logits decorrelate fail the sanity check.

**Occlusion axis (Binder):** Using fixed baseline attribution maps, delete the top 30 blurred 15×15 patches and track target-class softmax (AUC). Lower AUC ⇒ more faithful map.

**Cross-architecture comparison:** Use **sensitivity ratio** \(D_{\text{half}}^{\text{method}} / D_{\text{half}}^{\text{arch}}\) from saved curve stats — not raw overlay of ResNet vs ViT Spearman curves.

## Quick start

```bash
./scripts/setup_venv.sh && source .venv/bin/activate
export IMAGENET_ROOT=/path/to/imagenet   # local only; needs val/
```

**Dependencies:** use `requirements-pytorch.txt` for the active PyTorch pipeline; `requirements.txt` is only for legacy TensorFlow notebooks under `notebooks/legacy_tf/`.

## For reviewers / graders

Two ways to reproduce figures without re-running the full 500-image GPU pipeline:

1. **Precomputed results (recommended):** extract `results_and_figures.zip` at the repo root so outputs land in `results/`, then run `jupyter notebook notebooks/notebook_analysis.ipynb`.
2. **Full recompute:** requires a Modal account, ImageNet val set, and several GPU-hours — see [docs/modal.md](docs/modal.md).

Optional (not required for grading):

- Legacy TensorFlow replication: `notebooks/legacy_tf/` + `requirements.txt`
- Qualitative gallery script: `scripts/run_qual_gallery.py` (see below)
- Unit test: `pytest tests/`

**Submitting this project:** submit a zip of the repo plus `results_and_figures.zip` separately (the results archive is gitignored due to size). A GitHub link works for code review; graders still need the results zip to regenerate figures locally.

| Task | Command / doc |
|------|----------------|
| Cloud GPU runs | [docs/modal.md](docs/modal.md) |
| Methods, data, metrics | [docs/experimental_setup.md](docs/experimental_setup.md) |
| Rerun checklist | [docs/experimental_protocol.md](docs/experimental_protocol.md) |
| Figures (no GPU) | `jupyter notebook notebooks/notebook_analysis.ipynb` |

```bash
# Modal smoke test
modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual
# Full study (500 images, parallel methods)
modal run modal/app.py --experiment all --num-images 500 --parallel-methods --target-mode dynamic
./scripts/download_modal_results.sh
```

## Methods (active scope)

| Class | ResNet-50 | ViT-B/16 | Role |
|-------|-----------|----------|------|
| A | `gradient`, `input_grad`, `ig` | same | Portable gradients (Captum) |
| B | `gradcam` | `transformer_gradcam` | Native spatial attribution (`blocks[-2]` for ViT) |
| C | `gbp` | `attention_rollout` | Arch-specific (Guided Backprop; 12-layer Abnar attention rollout) |

Removed from active runs: DINOv2, `smoothgrad`, `gbp_gc`, legacy `rollout`, `dino_attn`.

## Data

- ImageNet **val**, first **500** images in **sorted JPEG filename order** (`ILSVRC2012_val_00000001.JPEG`, …) — same indices for ResNet and ViT.
- Human-readable index: `results/diagnostics/subset_manifest_first500.json` (build with `python scripts/build_subset_manifest.py --imagenet-root "$IMAGENET_ROOT"`).
- 224×224 center crop, standard ImageNet normalization.

## Repository layout

| Path | Role |
|------|------|
| `src/experiment_utils.py` | Pipelines: cascade, occlusion, qual bundle, mechanistic |
| `src/attention_utils.py` | ViT attention rollout + validation hooks |
| `src/viz_utils.py` | Cascade grids, occlusion curves, analysis plots |
| `scripts/run_qual_gallery.py` | Build/render per-index qual heatmap galleries |
| `notebooks/notebook_analysis.ipynb` | All paper figures from saved `.npy` / `.npz` |
| `modal/app.py` | Modal entrypoint |
| `legacy_figures/` | Original paper demo figures (reference only) |
| `results/` | Local outputs (gitignored) |

## Qualitative figures

Quant runs can use `--skip-qual`. Build `qual_bundle.npz` later:

```bash
modal run modal/app.py --experiment all --qual-only \
  --image-index-mode fixed --image-index 196 --qual-force   # example: hotdog in manifest
./scripts/download_modal_results.sh
# Re-run notebook_analysis.ipynb
```

Outputs under `results/figures/`: `within_arch_*_spearman.png`, `cascade_grid_*.png`, `occlusion_faithfulness_curves.png`, sensitivity-ratio table, etc.

Per-index qual galleries (after downloading qual bundles):

```bash
python scripts/run_qual_gallery.py --print-plan              # sample indices
python scripts/run_qual_gallery.py --capture-index 303 --render
```

## ViT-specific note

Cascade order randomizes **`head` first**, then `blocks.11` … `blocks.0`. **Attention rollout** only uses block self-attention, so **depth 0 and depth 1 maps are identical** (head-only randomization). Expect slow visual drift until lower blocks are randomized. See [docs/experimental_setup.md](docs/experimental_setup.md).

## Legacy TensorFlow replication

Original paper demos (Inception v3, MNIST) live in `notebooks/legacy_tf/`. Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train MNIST models with `src/train_cnn_models.py` / `src/train_mlp_models.py`. Inception weights: [TensorFlow checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) → `models/inceptionv3/inception_v3.ckpt`.

Legacy notebooks: `cnn_mnist_cascading_randomization.ipynb`, `inceptionv3_cascading_randomization.ipynb`, etc.

---

*Paper:* [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1806.07529) — Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim.
