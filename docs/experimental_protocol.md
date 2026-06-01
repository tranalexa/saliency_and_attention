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

1. Load existing `baseline_{method}.npz` maps (run cascade first; occlusion does not recompute attributions).
2. Denormalize each input, apply Gaussian blur (kernel size = patch size = 15) in image space, then re-normalize.
3. Rank 15×15 patches by absolute saliency intensity.
4. Replace the top 30 highest-scoring patches in descending order with the blurred copy.
5. Keep the step-0 target class fixed throughout deletion.
6. AUC = mean softmax confidence over steps 1–30 (step 0 excluded). Lower AUC = more faithful.

Outputs: `occlusion_{method}_curve.npy` `(N, 30)`, `occlusion_{method}_auc.npy`, `occlusion_{method}_auc_mean.npy`.

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
- Occlusion outputs include `occlusion_{method}_curve.npy` and `occlusion_{method}_auc.npy`.
- Cross-architecture summaries use sensitivity-ratio JSON/table outputs only.
