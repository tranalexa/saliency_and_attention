# Experimental Setup

This repository now targets a narrowed saliency sanity-check study:

- Architectures: `resnet50` and `vit` only.
- Cascade axis: cumulative top-down model randomization.
- Faithfulness axis: fixed-target blurred-patch deletion.
- Cross-architecture comparison: sensitivity ratio only.

## Models

| Architecture | Source | Cascade Order | Class B Target |
|--------------|--------|---------------|----------------|
| ResNet-50 | `timm.create_model("resnet50", pretrained=True)` | `fc`, then `layer4.2` ... `layer1.0` | GradCAM on `layer4[-1]` |
| ViT-B/16 | `timm.create_model("vit_base_patch16_224", pretrained=True, img_size=224)` | `head`, then `blocks.11` ... `blocks.0` | Transformer GradCAM on `blocks[-2]` |

DINOv2 is out of scope and is not an active architecture.

## Method Sets

| Class | ResNet-50 | ViT-B/16 | Role |
|-------|-----------|----------|------|
| A | `gradient`, `input_grad`, `ig` | `gradient`, `input_grad`, `ig` | Portable gradient methods |
| B | `gradcam` | `transformer_gradcam` | Architecture-native spatial attribution |
| C | `gbp` | `raw_attn` | Architecture-specific diagnostics |

Removed active methods: `smoothgrad`, `gbp_gc`, `rollout`, and `dino_attn`.

The ViT Transformer GradCAM target was selected empirically with
`diagnostics/choose_vit_gradcam_layer.py`: final-block candidates were
degenerate on all sampled images, while `blocks[-2]` was the latest candidate
with 0% degenerate maps on the pretrained model and tested cascade states.

## Data

All runs use ImageNet validation images with `Resize(256)`, `CenterCrop(224)`, `ToTensor`, and ImageNet normalization:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

Local notebooks read `IMAGENET_ROOT`; Modal reads `/imagenet` from the `saliency-imagenet` volume.

## Cascade Axis

For each depth, the model is restored to pretrained weights and then cumulatively randomized from the classifier downward through `order[:depth + 1]`. Baseline maps are compared to randomized maps with Spearman and SSIM. RMS normalization is primary; max-abs is retained only as a legacy metric variant.

The default cascade target policy is `dynamic`: explain the current model argmax at each depth. `frozen_baseline` remains available for ablations.

## Blurred-Occlusion Axis

The occlusion axis uses baseline maps to rank non-overlapping saliency patches, then progressively replaces those patches with a blurred copy of the same normalized input. The blurred copy is made by denormalizing to image space, applying Gaussian blur, and re-normalizing.

The target class is fixed at step 0 and is never re-argmaxed during deletion. Outputs per method:

- `{method}_occlusion_scores.npy`
- `{method}_occlusion_curve.npy`
- `{method}_occlusion_auc.npy`
- `{method}_occlusion_auc_mean.npy`
- `occlusion_config.json`

## Mechanistic Controls

`--experiment mechanistic` writes logit-correlation and activation-scale controls for ResNet-50 and ViT-B/16 only:

- `logit_corr_resnet.npy`
- `logit_corr_vit.npy`
- `activation_scale_{tag}_depth{NN}.npy`

## Sensitivity Ratio

Raw cross-architecture Spearman or SSIM curves are not used as the comparison unit. Cross-architecture summaries use `sensitivity_ratio = D_half / D_arch`, where `D_arch` comes from logit correlation. Curve stats are saved for Class A and Class B methods as `{method}_curve_stats.json` and `{method}_sensitivity_ratio.json`.

## Stale Results

Do not mix old and current results. Older artifacts may include `results/dinov2/` or outputs for `smoothgrad`, `gbp_gc`, `rollout`, and `dino_attn`. These should be archived or deleted before final reporting; they were intentionally not deleted by this cleanup.
