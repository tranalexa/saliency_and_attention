# Modal Runs

Run ResNet / ViT pipelines on **Modal A10G** without local ImageNet.

## Setup

```bash
./scripts/setup_venv.sh && source .venv/bin/activate && modal setup
modal volume create saliency-imagenet
```

ImageNet on volume (pick one):

```bash
# Download in cloud
modal run modal/download_imagenet.py --val-tar-url URL --devkit-tar-url URL

# Upload local val/
modal volume put saliency-imagenet /path/to/imagenet/val /val
```

Results volume: `saliency-results` (default).

## Experiments

| `--experiment` | Runs |
|----------------|------|
| `resnet50` / `vit` | Cascade (+ qual unless `--skip-qual`) |
| `mechanistic` | Logit correlation + activation scales |
| `occlusion` | Blurred deletion (needs baselines) |
| `all` | Cascade both archs + mechanistic + occlusion |

```bash
# Smoke
modal run modal/app.py --experiment vit --num-images 10 --skip-qual --sequential

# Full quant (500 images, parallel one GPU per method)
modal run modal/app.py --experiment all --num-images 500 --parallel-methods --target-mode dynamic --force-recompute

# Occlusion only
modal run modal/app.py --experiment occlusion --num-images 500

# Qual grids (after cascade)
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim_shared --qual-force
```

Download → analyze:

```bash
./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

Downloads `resnet50/`, `vit/`, `mechanistic/` only (not stale `dinov2/`).

## Useful flags

| Flag | Default | Notes |
|------|---------|-------|
| `--skip-qual` | false | Faster; run `--qual-only` later |
| `--parallel-methods` | true | One GPU per method |
| `--force-recompute` | false | Ignore cached metrics |
| `--target-mode` | `dynamic` | Cascade explanation class |
| `--seeds` | `42` | Class A can use `42,1,2` → `seed42/` subdirs |
| `--occlusion-arch` | `all` | `resnet50` \| `vit` \| `all` |
| `--occlusion-patch-fractions` | `0.10,0.20,0.30` | Comma-separated fractions of 196 arch-native tiles |
| `--image-index` | `0` | With `--qual-only --image-index-mode fixed` |
| `--qual-force` | false | Overwrite `qual_bundle.npz` |

## Manifest on Modal

```bash
modal run modal/build_subset_manifest.py
# Writes locally + volume diagnostics/subset_manifest_first500.json
```

Pick `dataset_index` from manifest for qual, e.g. hotdog at index 196:

```bash
modal run modal/app.py --experiment all --qual-only \
  --image-index-mode fixed --image-index 196 --qual-force
```

## Result layout on volume

```
/resnet50/          # Class B/C at top level; Class A under seed42/
/vit/
/mechanistic/
/diagnostics/       # optional manifest
```

Class A parallel runs: `results/<arch>/seed42/{method}_*.npy`.  
Class B/C: `results/<arch>/{method}_*.npy`.

## Stale volume cleanup

```bash
modal volume rm saliency-results /resnet50 --recursive
modal volume rm saliency-results /vit --recursive
modal volume rm saliency-results /mechanistic --recursive
```

Archive local `dinov2/` or old method folders before reporting.

## Cost

Always smoke-test `--num-images 10` first. Full 500×methods cascade + occlusion at three patch fractions is expensive; `--parallel-methods` trades cost for wall time.
