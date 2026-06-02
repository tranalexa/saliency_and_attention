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

Following Binder et al. (2023) Section A.1, the occlusion axis loads existing baseline maps (does not recompute attributions), ranks non-overlapping 15×15 patches by absolute saliency, and replaces the top 30 highest-scoring patches with a blurred copy of the same normalized input. **Box blur** (uniform 15×15 kernel) is the default, applied in `[0, 1]` image space after denormalization. Use `--blur-type gaussian` for the legacy Gaussian ablation.

The target class is fixed at step 0 (model argmax) and is never re-argmaxed during deletion. AUC is the mean softmax confidence over the 30 post-occlusion steps (step 0 excluded). Lower AUC indicates more faithful attribution.

Outputs per method:

- `occlusion_{method}_curve.npy` — `(N, 30)` softmax confidence after each replacement
- `occlusion_{method}_auc.npy` — `(N,)` mean confidence over 30 steps per image
- `occlusion_{method}_auc_mean.npy` — scalar mean AUC over all images
- `occlusion_config.json`

Shared metadata (cascade and occlusion):

- `ground_truth_indices.npy` — ImageNet validation labels
- `correctly_classified.npy` — boolean mask (`pred_argmax == ground_truth`); optional subset analysis only

For full-grid deletion (opt-in via `--occlusion-patches`), outputs use the `_full` suffix: `occlusion_{method}_curve_full.npy`, `occlusion_{method}_auc_full.npy`.

## Baseline and Target Policies

| Decision | Default | Scope |
|----------|---------|-------|
| Cascade `--target-mode` | `dynamic` | Adebayo: explain current model argmax at each randomization depth |
| Occlusion target | fixed step-0 argmax | Binder: softmax drop for the explained class |
| IG `--ig-baseline` | `zero` | Integrated Gradients reference input only |
| Occlusion blur | `box` (kernel = patch size) | Binder Section A.1 |

`frozen_baseline` cascade mode remains available as an ablation (fixed semantic class across depths). Occlusion is unaffected by cascade `target_mode`.

## Mechanistic Controls

`--experiment mechanistic` writes logit-correlation and activation-scale controls for ResNet-50 and ViT-B/16 only:

- `logit_corr_resnet.npy`
- `logit_corr_vit.npy`
- `activation_scale_{tag}_depth{NN}.npy`

## Sensitivity Ratio

Raw cross-architecture Spearman or SSIM curves are not used as the comparison unit. Cross-architecture summaries use `sensitivity_ratio = D_half / D_arch`, where `D_arch` comes from logit correlation. Curve stats are saved for Class A and Class B methods as `{method}_curve_stats.json` and `{method}_sensitivity_ratio.json`.

## Stale Results

Do not mix old and current results. Older artifacts may include `results/dinov2/` or outputs for `smoothgrad`, `gbp_gc`, `rollout`, and `dino_attn`. These should be archived or deleted before final reporting; they were intentionally not deleted by this cleanup.
