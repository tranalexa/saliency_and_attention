# Running experiments on Modal

This guide runs the PyTorch sanity-check pipelines on [Modal](https://modal.com) with **A10G** GPUs, without keeping ImageNet on your laptop.

## Prerequisites

1. ImageNet ILSVRC2012 **validation** set (obtain from [image-net.org](https://image-net.org) or your institution).
2. A [Modal](https://modal.com) account.

## 0. Virtual environment (recommended)

From the repository root:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
modal setup
```

`setup_venv.sh` creates `.venv/`, installs PyTorch stack + Modal CLI, and upgrades `pip`. Re-run it anytime to refresh dependencies.

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-pytorch.txt -r requirements-modal.txt
modal setup
```

## 1. ImageNet validation on the volume (one time)

You need class-organized val data at `/imagenet/val/` on volume `saliency-imagenet`. Two options:

### Option A — Download inside Modal (no local copy)

1. Register at [image-net.org](https://image-net.org) and open the ILSVRC2012 download page.
2. Copy the **direct download URLs** for:
   - `ILSVRC2012_img_val.tar` (~6.3 GB)
   - `ILSVRC2012_devkit_t12.tar` (required to sort val into 1000 class folders)
3. On the ImageNet site, under **ILSVRC → 2012**, right-click each download link and choose **Copy Link Address** (must start with `https://`, not just the filename).

4. Run (from repo root, venv active):

```bash
modal volume create saliency-imagenet
modal run modal/download_imagenet.py \
  --val-tar-url "https://image-net.org/data/.../ILSVRC2012_img_val.tar" \
  --devkit-tar-url "https://image-net.org/data/.../ILSVRC2012_devkit_t12.tar"
```

Wrong (will error): `--val-tar-url "ILSVRC2012_img_val.tar"` — that is only the filename, not a URL.

This downloads and extracts on Modal, organizes val with the devkit labels, writes to the volume, and deletes staging tarballs. Takes ~30–60+ minutes depending on ImageNet CDN speed.

Verify:

```bash
modal volume ls saliency-imagenet /val
```

Re-run with the same URLs is a no-op if val already exists (`--skip-if-exists`, default on).

### Option B — Upload from your laptop

If you already have val extracted locally:

```
/path/to/imagenet/val/n01440764/ILSVRC2012_val_....JPEG
...
```

```bash
modal volume create saliency-imagenet
modal volume put saliency-imagenet /path/to/imagenet/val /val
```

## 2. Run experiments

From the repository root:

```bash
# Smoke test (cheap)
modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual

# Fast 500-image ResNet (7 GPUs — one per method, default)
modal run modal/app.py --experiment resnet50 --num-images 500 --skip-qual --parallel-methods

# Full ResNet-50 on a single GPU (debug / lower concurrency cost)
modal run modal/app.py --experiment resnet50 --num-images 500 --sequential

# ViT-B/16 (7 methods, parallel by default)
modal run modal/app.py --experiment vit --num-images 500 --skip-qual

# DINOv2-B
modal run modal/app.py --experiment dinov2 --num-images 500 --skip-qual

# Mechanistic checks (ResNet + ViT)
modal run modal/app.py --experiment mechanistic --num-images 500

# All pipelines in parallel (up to ~22 GPUs with parallel methods — expensive)
modal run modal/app.py --experiment all --num-images 500 --skip-qual --parallel-methods

# Paper cascade figures only (after quant run + download, or results on volume):
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim
modal run modal/app.py --experiment resnet50 --qual-only --image-index-mode fixed --image-index 0
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `resnet50` | `resnet50`, `vit`, `dinov2`, `mechanistic`, or `all` |
| `--num-images` | `500` | Number of ImageNet val images (use `10` for testing) |
| `--batch-size` | `8` | Batch size (mechanistic uses `16`) |
| `--skip-qual` | false | Skip qualitative `qual_bundle.npz` during quant runs |
| `--qual-only` | false | Only build `qual_bundle.npz` (no Spearman/SSIM recompute) |
| `--image-index` | `0` | Demo image index when `--image-index-mode fixed` |
| `--image-index-mode` | `fixed` | `fixed` or `auto_ssim` (pick image with largest SSIM drop; needs `*_ssim.npy` on volume) |
| `--qual-force` | false | Overwrite existing `qual_bundle.npz` |
| `--parallel-methods` | true | Spawn one GPU job per saliency method (ResNet/ViT/DINO) |
| `--sequential` | false | Run the full pipeline on a single GPU (disables parallel methods) |

**Parallelism:** `--parallel-methods` (default) launches one A10G per saliency method. Wall-clock drops roughly 6–7× per architecture; total GPU-hours is similar, but you pay for concurrent GPUs. Use `--sequential` for cheap debugging. `--experiment all` with parallel methods can use up to ~22 GPUs at once (7 methods × 3 archs + mechanistic).

Results are written to the **`saliency-results`** volume under `/<arch>/` (e.g. `/resnet50/`). Inside the container these appear at `/results/<arch>/` because the volume is mounted at `/results`.

Jobs skip work when output files already exist (same as local notebooks).

## 3. Download results for analysis

Volume paths omit the mount prefix. Use the helper script (recommended):

```bash
./scripts/download_modal_results.sh
```

Or manually (download to parent `results/`, not `results/resnet50/`):

```bash
mkdir -p results
modal volume get saliency-results /resnet50 results
```

Wrong: `modal volume get saliency-results /results ./results` (no `/results` prefix on the volume).  
Wrong: `modal volume get saliency-results/resnet50 ./results` (volume name must not include paths).

Verify:

```bash
ls results/resnet50/ig_spearman_mean.npy
```

Then run the analysis notebook locally (no GPU):

```bash
jupyter notebook notebooks/notebook_analysis.ipynb
```

Figures are saved to `results/figures/`:

- `cascade_grid_resnet50.png`, `cascade_grid_vit.png`, `cascade_grid_dinov2.png` — Adebayo-style method × depth grids (requires `qual_bundle.npz`)
- `cascade_<arch>_<method>.png` — optional vertical strips per method

**Qual workflow:** Use `--skip-qual` for the 500-image quant job, then `--qual-only` (cheap). With `--parallel-methods` and `skip_qual=false`, qual runs automatically **after** all method jobs finish for that architecture. Do not rely on per-method workers to build qual bundles.

## Volumes summary

| Volume | Mount | Purpose |
|--------|-------|---------|
| `saliency-imagenet` | `/imagenet` | ImageNet val (read-only in practice) |
| `saliency-results` | `/results` | NumPy outputs at volume paths `/<arch>/` (e.g. `/resnet50/`) |

## Cost notes

- Full runs (500 images × all randomization depths × 7 methods per arch) are **GPU-heavy**.
- Always start with `--num-images 10 --skip-qual`.
- **`--parallel-methods`** (default) uses one GPU per method — faster wall-clock, same total GPU-hours, higher peak concurrency cost.
- Use **`--sequential`** for single-GPU runs when debugging or limiting concurrent spend.
- Pretrained weights are downloaded from timm on first run inside the container.

## Troubleshooting

**`ImageNet not found`**

Run `modal/download_imagenet.py` (Option A) or `modal volume put` (Option B).

**Download URLs fail (403 / expired)**

ImageNet links are tied to your account and can expire. Generate fresh URLs from the download page and re-run.

**Stale results**

Delete a subfolder on the volume or remove specific `.npy` files before re-running.

**Local notebooks**

You can still run computation locally with `IMAGENET_ROOT` set; Modal is optional.
