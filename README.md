
Sanity Checks for Saliency Maps
=====================
This repository provides code to replicate the paper
**Sanity Checks for Saliency Maps** by<br/>
*Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, & Been Kim*.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/saliency_methods_and_edge_detector.png" width="700">


### Overview

Saliency methods have emerged as a popular tool to highlight
features in an input deemed relevant for the prediction of a
learned model. Several saliency methods have been proposed, often
guided by visual appeal on image data. In this work, we propose
an actionable methodology to evaluate what kinds of explanations
a given method can and cannot provide. We find that reliance,
solely, on visual assessment can be misleading. Through extensive
experiments we show that some existing saliency methods are
independent both of the model and of the data generating process.
Consequently, methods that fail the proposed tests are
inadequate for tasks that are sensitive to either data or model,
such as, finding outliers in the data, explaining the
relationship between inputs and outputs that the model learned,
or debugging the model. We interpret our findings through an
analogy with edge detection in images, a technique that requires
neither training data nor model. Theory in the case of a
linear model and a single-layer convolutional neural network
supports our experimental findings.

#### Model Randomization Test

For the model randomization test, we randomize the weights of a
model starting from the top layer, successively, all the way to
the bottom layer. This procedure destroys the learned
weights from the top layers to the bottom ones. We compare the resulting explanation from a network with random weights to the one obtained with the model's original weights. Below we show the
evolution of saliency masks from different methods for a demo image from the ImageNet dataset and the Inception v3 model.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/bird_cascading_demo.png" width="700">

##### Independent Layer randomization

Here we show the results of randomizing each 'layer/block' at a time while keeping the other weights set at the pre-trained (original) values.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/bird_independent_demo.png" width="700">

#### Data Randomization Test

In our data randomization test, we permute the training labels
and train a model on the randomized training data. A model
achieving high training accuracy on the randomized training data
is forced to memorize the randomized labels without being able to
exploit the original structure in the data. We now compare
saliency masks for a model trained on random labels and one
trained true labels. We present examples below on MNIST and Fashion MNIST.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/mnist_cnn_random_labels_test.png" width="700">

See the paper and appendix for additional figures and results on the data randomization test.


#### Guided Backprop Errata
A previous version of the paper said that Guided Backprop was completely invariant to model randomization (weight re-initialization); however, this is not the case. Guided Backprop is still invariant to higher layer weights of a DNN, but it is not completely invariant. As we show in the figure below, when the lower layers are randomized, there is indeed a distortion to the mask. However, we still observe that there is high visual similarity between the mask derived from a completely reinitialized model and the input. Overall, the findings in the paper remain unchanged. We have recently updated the arxiv version as well. See the inceptionv3_guidedbackprop_demo.ipynb in the notebook folder for replication.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/guided_backprop_demo.png" width="700">

### Data

See /doc/data/ for the demo images and the ImageNet image ids used in this
work.  

---

## PyTorch extension (ViT, DINOv2, mechanistic checks)

This fork adds a **PyTorch + timm + Captum** pipeline that extends Adebayo's
cascading model randomization test to modern architectures.

### Setup (virtual environment)

From the repo root:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

This installs [`requirements-pytorch.txt`](requirements-pytorch.txt) (notebooks, local GPU) and [`requirements-modal.txt`](requirements-modal.txt) (Modal CLI). To install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pytorch.txt -r requirements-modal.txt
```

For **local** notebook runs (not Modal):

```bash
export IMAGENET_ROOT=/path/to/imagenet   # must contain val/
jupyter notebook notebooks/
```

### Notebooks

| Notebook | Purpose | Outputs |
|----------|---------|---------|
| `notebooks/notebook_resnet50_cascading.ipynb` | **Replication baseline** — ResNet-50, 7 saliency methods | `results/resnet50/` |
| `notebooks/notebook_vit_cascading.ipynb` | ViT-B/16 — 7 methods (5 shared Captum + attention) | `results/vit/` |
| `notebooks/notebook_dinov2_cascading.ipynb` | DINOv2-B/14 @ 224 — same 7-method set as ViT | `results/dinov2/` |
| `notebooks/notebook_mechanistic.ipynb` | Logit correlation & activation scales (ResNet vs ViT vs DINOv2) | `results/mechanistic/` |
| `notebooks/notebook_analysis.ipynb` | Figures only (no model loading) | `results/figures/` |

Legacy TensorFlow replication notebooks live in `notebooks/legacy_tf/`.

### Shared utilities (`src/`)

- `experiment_utils.py` — full pipelines (data loading, Captum, cascading, mechanistic); used by notebooks and Modal
- `randomize_utils.py` — cascading weight reset (`reset_layer`, checkpoints, layer orders)
- `metrics_utils.py` — Spearman, SSIM, logit Pearson correlation, map normalization
- `viz_utils.py` — Adebayo-style cascade figure grids (`qual_bundle.npz` → paper PNGs)
- `attention_utils.py` — raw attention & rollout (Abnar & Zuidema 2020)

### Running on Modal (cloud GPU)

Run computation without local ImageNet storage. See **[docs/modal.md](docs/modal.md)** for full instructions.

```bash
source .venv/bin/activate
modal setup
# ImageNet on Modal (pick one):
#   A) Download in cloud: modal run modal/download_imagenet.py --val-tar-url URL --devkit-tar-url URL
#   B) Upload from laptop: modal volume put saliency-imagenet /path/to/imagenet/val /val
modal run modal/app.py --experiment resnet50 --num-images 10 --skip-qual   # smoke test
# Fast 500-image run (7 GPUs per arch in parallel, one per method):
modal run modal/app.py --experiment resnet50 --num-images 500 --skip-qual --parallel-methods
./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

Uses **NVIDIA A10G** by default. Full 500-image runs are expensive; always smoke-test with `--num-images 10` first.

### Qualitative cascade figures (paper)

Quant runs can use `--skip-qual` (faster). Build `qual_bundle.npz` afterward (one GPU per architecture, all methods):

```bash
modal run modal/app.py --experiment all --qual-only --image-index-mode auto_ssim
# Or Adebayo default (first val image): --image-index-mode fixed --image-index 0
./scripts/download_modal_results.sh
jupyter notebook notebooks/notebook_analysis.ipynb
```

Outputs: `results/figures/cascade_grid_<arch>.png` (rows = methods, cols = baseline + cascade depths). Legacy Adebayo layout is documented in `notebooks/legacy_tf/inceptionv3_cascading_randomization.ipynb`.

### Dataset

- ImageNet **validation**, first **500** images (Binder-style convention)
- 224×224 center crop, standard ImageNet normalization
- Set `IMAGENET_ROOT` in each notebook CONFIG cell (or env var), or upload val to Modal volume `saliency-imagenet`

### Expected qualitative behavior (ResNet replication)

ResNet-50 runs **7 methods** (Gradient, SmoothGrad, Input-Grad, GBP, GradCAM, GBP-GC, IG):

- **Guided Backprop / GBP-GC / Input-Grad**: high SSIM/Spearman when only upper layers are randomized; Input-Grad often stays near 1.0 (strongest sanity-check failure).
- **Integrated Gradients / Gradient / SmoothGrad / GradCAM**: similarity drops quickly as upper blocks are randomized.

**MNIST** (used in the original paper's Figure 20) is a dataset of 70,000 handwritten digit images (0–9), 28×28 pixels — a classic deep-learning benchmark. The legacy TensorFlow MNIST notebooks live in `notebooks/legacy_tf/`.

### Saliency method sets (cross-architecture)

| Method | ResNet-50 | ViT / DINOv2 | Notes |
|--------|-----------|--------------|-------|
| Gradient, SmoothGrad, Input-Grad, IG, GradCAM | yes | yes | **5 shared methods** — directly comparable across architectures |
| GBP, GBP-GC | yes | no | ReLU-specific Guided Backprop (ResNet only) |
| Raw attention, Rollout | no | yes | Transformer attention maps |

ViT and DINOv2 runs are **extensions** of the Adebayo cascading test. The analysis notebook includes cross-architecture overlay plots for the 5 shared methods (ResNet vs ViT vs DINOv2), ViT vs DINOv2 attention methods, and mechanistic checks across all three architectures.

### DINOv2 note

`timm` model `vit_base_patch14_dinov2.lvd142m` may ship without a trained ImageNet classifier.
Before the full 500-image loop, verify `model.num_classes == 1000` and sensible top-1 predictions;
load an ImageNet linear head checkpoint if needed.

---

### Instructions (legacy TensorFlow)

We have added scripts for training simple MLPs and CNNs on MNIST. To run any of the MNIST notebooks, use these scripts to quickly train either an MLP on MNIST (or Fashion MNIST) or a CNN on MNIST (or Fashion MNIST). The scripts are relatively straight forward. To run the inception v3 notebooks, you will also need to grab pre-trained weights and put them models folder as described in the instructions below.

We use the [saliency python package](https://github.com/pair-code/saliency) to obtain saliency masks. Please see that package for a quick overview. Overall, this replication is mostly for illustration purposes. There are now other packages in PyTorch that provide similar capabilities.

You can use the instructions below to setup an environment with the right dependencies.

```
python3.5 -m venv pathtovirtualvenv
source pathtovirtualvenv/bin/activate
pip install -r requirements.txt
```

### Train simple CNNs/MLPs on MNIST/Fashion MNIST
You can train a CNN on MNIST using *src/train_cnn_models.py* as follows:
```
python train_cnn_models.py --data mnist --savemodelpath ../models/ --reg --log
```

You can toggle the data with the --data option. You can also train MLPs with an analogous command:  

```
python train_mlp_models.py --data mnist --savemodelpath ../models/ --reg --log
```

To run the CNN and MLP on MNIST notebooks, you will need to train quick models with the commands above.

### Inception V3 Checkpoint (Important!)
To run any of the incetion_v3 notebooks, you will need inception pretrained weights. These are available from [tensorflow](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz). Alternatively, the weights can be obtained and decompressed as follows:

```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
```

At the end of this, you should have the file *inception_v3.ckpt* in the folder *models/inceptionv3*. With this, you can run the inception notebooks.


#### Notebooks (legacy TensorFlow)

In `notebooks/legacy_tf/`, you will find replication of the key experiments in the paper:

- *cnn_mnist_cascading_randomization.ipynb*: cascading randomization on a CNN trained on MNIST.
- *cnn_mnist_independent_randomization.ipynb*: independent randomization on a CNN trained on MNIST.
- *inceptionv3_cascading_randomization.ipynb*: cascading randomization on Inception v3 (ImageNet).
- *inceptionv3_independent_layer_randomization.ipynb*: independent randomization for Inception v3.
- *inception_v3_guidedbackprop_demo.ipynb*: guided backprop with cascading randomization.
- *mlp_mnist_cascading_randomization.ipynb*: cascading randomization on an MLP trained on MNIST.
