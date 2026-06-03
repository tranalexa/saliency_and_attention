# Experimental Protocol

Operational checklist for cascade + occlusion on ResNet-50 and ViT-B/16.

## Scope

| Item | Value |
|------|--------|
| Architectures | `resnet50`, `vit` |
| Class A | `gradient`, `input_grad`, `ig` |
| Class B | `gradcam` / `transformer_gradcam` |
| Class C | `gbp` / `attention_rollout` |
| Images | 500 val, sorted JPEG order |
| Out of scope | DINOv2, `smoothgrad`, `gbp_gc`, `rollout`, `dino_attn` |

## Cascade

| Setting | Value |
|---------|--------|
| Order | Classifier first (`fc` / `head`), then blocks top → bottom |
| Reset unit | Full ResNet Bottleneck or full ViT block |
| Baseline | Pretrained, depth 0 |
| Metrics | Spearman + SSIM vs baseline maps |
| Target | `dynamic` (default) or `frozen_baseline` |

**Verify after run:**

- `randomization_order.json` starts with `fc` (ResNet) or `head` (ViT).
- `experiment_config.json` lists only scoped methods.
- Depth-0 Spearman ≈ 1.0 per method.
- ViT: depth-0 and depth-1 **attention_rollout** Spearman both 1.0 (expected).

## Occlusion

1. Cascade baselines exist (`baseline_{method}.npz`).
2. Arch-native grids (196 tiles): ResNet 15×15 blur 15; ViT 16×16 blur 16. Top `round(fraction×196)` tiles at fractions 0.10, 0.20, 0.30; fixed step-0 target.
3. Outputs: `occlusion_{method}_curve_fracXXX.npy`, `_auc_fracXXX.npy`, `occlusion_auc_summary.csv`, `occlusion_config.json` with `"blur_type": "box"`.

**Verify:** `ground_truth_indices.npy`, `correctly_classified.npy` present (from cascade).

## Modal clean rerun

```bash
modal volume rm saliency-results /resnet50 --recursive
modal volume rm saliency-results /vit --recursive
modal volume rm saliency-results /mechanistic --recursive

modal run modal/app.py --experiment resnet50 --num-images 10 --sequential --force-recompute
modal run modal/app.py --experiment all --num-images 500 --parallel-methods --target-mode dynamic --force-recompute
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim_shared --qual-force

./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

`--experiment all` = cascade (both archs) + mechanistic + occlusion.

## Manifest

After ImageNet is on the volume:

```bash
modal run modal/build_subset_manifest.py
```

Maps `dataset_index` → filename, WNID, class name. Use for fixed qual indices (e.g. `--image-index 196`).

## Analysis verification

- Within-arch plots include **attention_rollout** (loads `raw_attn_*` if needed).
- Sensitivity table includes **Class C** rows.
- No raw ResNet–ViT Spearman overlay (by design).

## Tensor hygiene

Never run occlusion then cascade/qual in one process without reloading images. Occlusion uses blurred working tensors; cascade uses `_attribution_batch` on clean clones.
