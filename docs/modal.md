# Running Experiments on Modal

This guide runs the scoped PyTorch pipelines on Modal A10G GPUs without keeping ImageNet on your laptop.

## Setup

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
modal setup
```

Upload or download ImageNet validation into the `saliency-imagenet` volume:

```bash
modal volume create saliency-imagenet
modal run modal/download_imagenet.py \
  --val-tar-url "https://image-net.org/data/.../ILSVRC2012_img_val.tar" \
  --devkit-tar-url "https://image-net.org/data/.../ILSVRC2012_devkit_t12.tar"
```

If you already have class-organized validation data:

```bash
modal volume put saliency-imagenet /path/to/imagenet/val /val
```

## Experiments

Accepted `--experiment` values:

- `resnet50`
- `vit`
- `mechanistic`
- `occlusion`
- `all` (cascade for ResNet/ViT, mechanistic, and occlusion)

Common commands:

```bash
# Cheap cascade smoke
modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual --sequential
modal run modal/app.py --experiment vit --num-images 10 --skip-qual --sequential

# Full study: parallel cascade + qual + mechanistic + occlusion (500 images)
modal run modal/app.py --experiment all --num-images 500 --parallel-methods --target-mode dynamic --seeds 42,1,2

# Faster quant-only rerun (skip qual figures):
modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods --target-mode dynamic

# Occlusion only (after cascade baseline maps exist)
modal run modal/app.py --experiment occlusion --num-images 500

# Qualitative cascade grids after cascade outputs exist
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim
```

`--experiment occlusion` runs both scoped architectures by default. Use `--occlusion-arch resnet50` or `--occlusion-arch vit` to run one architecture.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `resnet50` | `resnet50`, `vit`, `mechanistic`, `occlusion`, or `all` |
| `--num-images` | `500` | Number of ImageNet validation images |
| `--batch-size` | `8` | Batch size; mechanistic uses 16 internally |
| `--skip-qual` | false | Skip `qual_bundle.npz` (faster; use `--qual-only` afterward) |
| `--qual-only` | false | Only build `qual_bundle.npz` |
| `--parallel-methods` | true | Spawn one GPU job per saliency method (default on) |
| `--sequential` | false | Run one architecture on one GPU |
| `--force-recompute` | false | Ignore cached outputs |
| `--target-mode` | `dynamic` | Cascade target policy |
| `--seed` | `42` | Primary RNG seed |
| `--seeds` | `42` | Class A multi-seed list for parallel cascade runs |
| `--ig-baseline` | `zero` | Integrated Gradients baseline: `zero` or `mean` |
| `--ig-steps` | `50` | Integrated Gradients interpolation steps |
| `--occlusion-arch` | `all` | `resnet50`, `vit`, or `all` for occlusion |
| `--occlusion-patches` | `30` | Top-K patches to occlude (Binder A.1 default; increase for full-grid) |
| `--occlusion-patch-size` | `15` | Patch and blur kernel size (15 = Binder main; 8 = appendix robustness) |
| `--blur-type` | `box` | Occlusion blur: `box` (Binder default) or `gaussian` (legacy ablation) |
| `--blur-sigma` | `8.0` | Gaussian sigma only; ignored when `--blur-type=box` |

## Results

Results are written to the `saliency-results` volume under `/<arch>/`, e.g. `/resnet50/` and `/vit/`. Download the scoped folders with:

```bash
./scripts/download_modal_results.sh
```

Then run:

```bash
jupyter notebook notebooks/notebook_analysis.ipynb
```

The helper downloads `resnet50`, `vit`, and `mechanistic`. It intentionally does not download stale `dinov2` folders.

## Stale Results

After protocol changes, delete scoped folders or pass `--force-recompute`:

```bash
modal volume rm saliency-results /resnet50 --recursive
modal volume rm saliency-results /vit --recursive
modal volume rm saliency-results /mechanistic --recursive
```

Older volumes or local folders may contain `dinov2` or removed-method artifacts (`smoothgrad`, `gbp_gc`, `rollout`, `dino_attn`). Archive or delete them before final reporting; this cleanup does not delete result artifacts automatically.

## Cost Notes

Start with `--num-images 10`. Full cascade runs are GPU-heavy; `--parallel-methods` reduces wall-clock time but increases peak GPU concurrency. Occlusion is also expensive because each image requires 30 forward passes per method (Binder A.1 default).
