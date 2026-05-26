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

- **Val layout:** `/imagenet/val/n01440764/ILSVRC2012_val_....JPEG` (from `modal/download_imagenet.py` + devkit). Meta also uses `labels.txt` + `extra/*.npy` for their training `ImageNet` class; **not required** for hub LC inference.
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
modal volume rm saliency-results/resnet50 --recursive
modal volume rm saliency-results/vit --recursive
modal volume rm saliency-results/dinov2 --recursive
modal volume rm saliency-results/mechanistic --recursive

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
