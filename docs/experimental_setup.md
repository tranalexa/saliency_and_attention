# Experimental Setup

Scoped study: **ResNet-50** and **ViT-B/16** on ImageNet val (500 images), **cascade randomization** (Adebayo) and **blurred occlusion** (Binder). Cross-architecture comparison uses **sensitivity ratio** only.

## Conceptual framing

**Model randomization test:** If a saliency map still looks “meaningful” after weights that drive predictions are destroyed, the method may be acting like an edge detector — insensitive to the model.

**Faithfulness test:** If occluding high-attribution regions does not drop target confidence, the map may not reflect what the model uses.

Together: cascade tests *model sensitivity*; occlusion tests *input faithfulness* on a fixed explanation target.

## Models

| Arch | timm id | Cascade order (first → last) | Class B hook |
|------|---------|------------------------------|--------------|
| ResNet-50 | `resnet50` | `fc`, `layer4.2` … `layer1.0` | GradCAM on `layer4[-1]` |
| ViT-B/16 | `vit_base_patch16_224` | `head`, `blocks.11` … `blocks.0` | Transformer GradCAM on `blocks[-2]` |

ViT GradCAM target: `diagnostics/choose_vit_gradcam_layer.py` — final block was degenerate; `blocks[-2]` was the latest non-degenerate layer.

## Method sets

| Class | ResNet | ViT | Implementation |
|-------|--------|-----|----------------|
| A | `gradient`, `input_grad`, `ig` | same | Captum; IG default baseline `zero`, 50 steps |
| B | `gradcam` | `transformer_gradcam` | pytorch-grad-cam |
| C | `gbp` | `attention_rollout` | Guided Backprop; **Abnar & Zuidema (2020) 12-layer attention rollout** |

**Attention rollout:** Hooks post-softmax QK attention in every ViT block; head-mean; \(0.5 A + 0.5 I\); row-normalize; chain \(\bar{A}_i = A_i \bar{A}_{i-1}\); CLS→patch weights → 14×14 grid → upsample 224. Entropy diagnostic uses **last block only**.

**ViT cascade caveat:** `head` is randomized at depth 1 but rollout ignores it → **identical maps at depths 0 and 1**. Early columns in qual grids can look the same until lower blocks are hit.

**Artifact aliases:** Older runs wrote `raw_attn_*`; code accepts `attention_rollout` ↔ `raw_attn` via `METHOD_ARTIFACT_ALIASES`.

## Data

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

- Default subset: **`sorted`** val JPEG order (`SortedValDataset`), indices `0 … N-1` shared across architectures.
- Alternative: `subset_order="imagefolder"` (WNID folder order — not comparable to sorted runs).

**Manifest** (`results/diagnostics/subset_manifest_first500.json`):

```bash
python scripts/build_subset_manifest.py --imagenet-root "$IMAGENET_ROOT"
# Modal: modal run modal/build_subset_manifest.py
```

`dataset_index` in the manifest = cascade / occlusion / qual `image_index`.

## Cascade axis

1. Save **baseline** maps on pretrained model (`baseline_{method}.npz`).
2. For each depth `d ∈ [0, |order|]`: restore pretrained weights, randomize `order[:d]`, compute maps on clean inputs.
3. Compare to baseline: Spearman (RMS primary; signed RMS for IG/input_grad) and SSIM variants.
4. Default target: **`dynamic`** — argmax class at each depth. Ablation: **`frozen_baseline`**.

Depth 0 Spearman must be ≈ 1.0 (sanity check in code).

Curve summaries: `{method}_curve_stats.json`, `{method}_sensitivity_ratio.json` for all scored methods (A, B, C).

## Occlusion axis

Requires cascade baselines first. Does **not** recompute attributions in the deletion loop.

- **Arch-native 196-tile grids:** ResNet-50 uses 14×14 non-overlapping 15×15 tiles (box blur 15); ViT-B/16 uses native 14×14 tokens as 16×16 pixel patches (box blur 16).
- **Patch fraction:** occlude top `round(patch_fraction × 196)` tiles by mean |saliency| (default fractions 0.10, 0.20, 0.30).
- Target class = step-0 argmax, fixed for all steps.
- AUC = mean softmax over occlusion steps 1–N (lower = more faithful).
- Artifacts: `occlusion_{method}_curve_fracXXX.npy`, `_auc_fracXXX.npy`, `occlusion_auc_summary.csv`.

**Isolation:** Occlusion mutates working copies only; cascade/qual always use `_attribution_batch` clones.

## Mechanistic controls

`--experiment mechanistic` → `results/mechanistic/`:

- `logit_corr_{resnet,vit}.npy` — Pearson corr. of pretrained vs cascade logits (per depth).
- `activation_scale_{tag}_depth{NN}.npy` — mean |activation| per channel.

Used for logit-correlation bands in analysis plots and \(D_{\text{arch}}\) in sensitivity ratio.

## Analysis outputs

`notebooks/notebook_analysis.ipynb` (no model load):

| Figure / table | Source |
|----------------|--------|
| `within_arch_{arch}_spearman.png` | `{method}_spearman_*_mean.npy` (alias-aware) |
| `class_b_spatial_attribution_within_arch.png` | GradCAM / transformer_gradcam |
| ViT rollout + entropy panel | `attention_rollout` curve + `*_entropy_depth*.npy` |
| `occlusion_faithfulness_curves.png` | `occlusion_{method}_curve_frac020.npy` (20% default plot) |
| `occlusion_auc_by_fraction.png` | `occlusion_auc_summary.csv` |
| Sensitivity-ratio LaTeX table | `*_curve_stats.json` (Class A, B, C) |

Attention-rollout qual rows use **shared row-wise display scaling** so depth differences are visible (per-panel minmax was misleading).

## Stale artifacts

Do not mix runs from removed methods (`dinov2`, `smoothgrad`, `gbp_gc`, `rollout`, `dino_attn`) or old cascade orders (deep-to-shallow). Delete `results/<arch>/*_spearman*.npy` and re-run after protocol changes.
