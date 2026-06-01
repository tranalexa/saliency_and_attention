# Experimental Protocol

Cascading model randomization and blurred-occlusion faithfulness on ResNet-50 and ViT-B/16.

## Scope

| Dimension | Current Scope |
|-----------|---------------|
| Architectures | `resnet50`, `vit` |
| Class A | `gradient`, `input_grad`, `ig` |
| Class B | ResNet `gradcam`; ViT `transformer_gradcam` |
| Class C | ResNet `gbp`; ViT `raw_attn` |
| Removed | DINOv2, `smoothgrad`, `gbp_gc`, `rollout`, `dino_attn` |

## Cascade Protocol

| Setting | Behavior |
|---------|----------|
| Cascade order | Classifier first (`fc` / `head`), then top-to-bottom blocks |
| ResNet randomization | Full Bottleneck module per step |
| ViT randomization | Full transformer block per step |
| Explanation class | `dynamic` by default; `frozen_baseline` for ablations |
| Metrics | Spearman and SSIM against pretrained baseline maps |
| Primary normalization | RMS |

## Blurred-Occlusion Protocol

1. Compute or load baseline maps for each scoped method.
2. Denormalize each input, apply Gaussian blur in image space, then re-normalize.
3. Rank patches by absolute saliency intensity.
4. Replace patches in descending saliency order with the blurred copy.
5. Keep the step-0 target class fixed throughout deletion.
6. Save normalized deletion curves and AUCs.

## Modal Rerun

Delete or archive stale scoped folders before a clean rerun. The commands below do not remove old `dinov2` results; decide separately whether to archive or delete them.

```bash
modal volume rm saliency-results /resnet50 --recursive
modal volume rm saliency-results /vit --recursive
modal volume rm saliency-results /mechanistic --recursive

modal run modal/app.py --experiment resnet50 --num-images 10 --sequential --force-recompute
modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods --force-recompute
modal run modal/app.py --experiment mechanistic --num-images 500 --force-recompute
modal run modal/app.py --experiment occlusion --num-images 500 --force-recompute
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim --qual-force

./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

## Verification

Confirm:

- `randomization_order.json` starts with `fc` for ResNet and `head` for ViT.
- `experiment_config.json` lists only scoped methods.
- `gradcam_target` is `layer4[-1]` for ResNet and `blocks[-2]` for ViT. The ViT target is selected by `diagnostics/choose_vit_gradcam_layer.py`, which rejected final-block targets as degenerate.
- No active result generation writes DINOv2 or removed-method outputs.
- Occlusion outputs include `{method}_occlusion_curve.npy` and `{method}_occlusion_auc.npy`.
- Cross-architecture summaries use sensitivity-ratio JSON/table outputs only.
# Experimental protocol (PyTorch / Modal)

Cascading model randomization sanity checks (Adebayo et al.) on ResNet-50, ViT-B/16, and DINOv2-B/14.

## Infrastructure

- **ImageNet:** Modal volume `saliency-imagenet` only (no local val required for runs).
- **Results:** Modal volume `saliency-results` → `results/<arch>/` after `scripts/download_modal_results.sh`.
- **Compute:** `modal run modal/app.py` (not the cascading Jupyter notebooks, which expect local `IMAGENET_ROOT`).

## Protocol

| Setting | Behavior |
|---------|----------|
| Cascade order | **Classifier first** (`fc` / `head`), then top→bottom |
| ResNet randomization | **Full Bottleneck module** per step (`layer4.N`, …), not conv1-only |
| Explanation class | **`argmax` after each cascade step** (`target_mode=dynamic`, default) |
| Map metrics | **abs** (default filenames) + **diverging** (`*_spearman_div.npy`, `*_ssim_div.npy`) |

Each run writes `experiment_config.json` with `target_mode`, `resnet_randomization`, `seed`, `num_images`, `methods`, and `randomization_order`.

Use `target_mode=frozen_baseline` only for ablations (explain the original top-1 class at every depth).

## Saliency methods (7 per arch on ResNet / ViT / DINOv2)

Shared Captum: gradient, smoothgrad, input_grad, ig, gradcam.

- ResNet-50: + gbp, gbp_gc  
- ViT / DINOv2: + raw_attn, rollout  

IG–SmoothGrad is not run (too costly at 500 images × cascade depth).

## DINOv2 classifier (ImageNet-pretrained)

Per [Meta DINOv2 README](https://github.com/facebookresearch/dinov2): use **`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc', layers=1)`** (ViT-B/14 distilled, ~84.5% linear ImageNet), not timm `num_classes=1000` (random head).

- **Val layout:** `/imagenet/val/n01440764/ILSVRC2012_val_....JPEG` (from `modal/download_imagenet.py` + devkit). Labels use ILSVRC indices via `meta.bin` or `ILSVRCValDataset` (not alphabetical `ImageFolder` order). If `meta.bin` is missing on the volume, run `modal/download_imagenet.py --backfill-meta-only`.
- **Preprocess:** `build_dinov2_transform` — Resize(224) + CenterCrop(224).

Before `num_images >= 50`, a short probe checks top-1 accuracy on val.

## Out of scope

- Occlusion faithfulness (Binder-style)
- InceptionV3 on Modal (legacy TF notebooks)
- Independent-layer randomization (TF notebooks only)

## Extensions in this work

- Cascading sanity checks on **ViT** and **DINOv2**
- **Attention maps** under weight randomization
- **Mechanistic** probes: logit correlation and activation scales vs depth
- **Modal** pipeline with parallel per-method jobs

## Full rerun (recommended after any protocol code change)

Delete stale artifacts so skip-if-exists does not reuse old numbers:

```bash
modal volume rm saliency-results /resnet50 --recursive
modal volume rm saliency-results /vit --recursive
modal volume rm saliency-results /dinov2 --recursive
modal volume rm saliency-results /mechanistic --recursive

# optional: local copy
rm -rf results/resnet50 results/vit results/dinov2 results/mechanistic results/figures

modal run modal/app.py --experiment resnet50 --num-images 10 --sequential --force-recompute
modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods --force-recompute
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim --qual-force
modal run modal/app.py --experiment mechanistic --num-images 500 --force-recompute

./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

Verify: `randomization_order.json` starts with `"fc"` or `"head"`; `experiment_config.json` lists current `target_mode` and `resnet_randomization: "block"`.
