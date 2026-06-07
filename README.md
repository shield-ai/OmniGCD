# OmniGCD

Official code release draft for **OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism**.

OmniGCD performs **zero-shot generalized category discovery (GCD)** by:

1. extracting frozen features with a modality-specific encoder,
2. reducing those features into a low-dimensional GCD latent space,
3. transforming the latent space with a synthetically trained Transformer (**GCDformer**), and
4. clustering the transformed features with k-means.

> **Pretrained weights:** the final pretrained GCDformer checkpoint used in the paper will be released. Until then, the code below can train a small synthetic checkpoint for sanity checks; performance is not expected to match the paper.

## Repository layout

```text
omnigcd/
  models/gcdformer.py        # GCDformer model
  data/synthetic.py          # synthetic GCD latent-space generator
  training/losses.py         # contrastive training loss
  eval/                      # GCD sequence construction, reduction, k-means, metrics
scripts/
  make_toy_features.py       # creates tiny synthetic feature NPZ
  eval_npz.py                # evaluates from precomputed features
  train_gcdformer.py         # synthetic GCDformer training
  extract_vision_features.py # CIFAR vision feature extraction example
  extract_text_features.py   # TSV text feature extraction example
  extract_audio_features.py  # HuggingFace audio feature extraction draft
  extract_remote_sensing_features.py # torchgeo remote-sensing draft
configs/
  train_gcdformer.yaml
checkpoints/
  README.md                  # pretrained checkpoint placeholder
```

## Installation

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For training and feature extraction, install the full dependencies. Choose the PyTorch command appropriate for your CUDA version from <https://pytorch.org/>; for many CUDA 12.1 systems:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[all]'
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Smoke test without pretrained weights

This quick test does **not** require the final pretrained model. It creates toy features and evaluates k-means on the reduced GCD latent space. This verifies that the lightweight repo/evaluation path works.

```bash
python scripts/make_toy_features.py --output examples/toy_features.npz
python scripts/eval_npz.py --features examples/toy_features.npz --reduction pca --samples-per-class 5
```

Expected output ends with something like:

```text
---- OmniGCD evaluation ----
All: ... | Old: ... | New: ...
```

## Train a small synthetic GCDformer checkpoint

This trains GCDformer on synthetic latent spaces. The default config is intentionally short for sanity checking. Increase `training.epochs` and `training.steps_per_epoch` for real training.

```bash
python scripts/train_gcdformer.py --config configs/train_gcdformer.yaml --output checkpoints/gcdformer_synthetic.pt
```

Evaluate the toy features with the trained checkpoint:

```bash
python scripts/eval_npz.py \
  --features examples/toy_features.npz \
  --checkpoint checkpoints/gcdformer_synthetic.pt \
  --reduction pca \
  --samples-per-class 5
```

## Evaluate from precomputed features

All feature extractors and evaluation scripts use a common NPZ schema:

```text
train_features: [N_train, D]
train_labels:   [N_train]
test_features:  [N_test, D]
test_labels:    [N_test]
known_classes:  [N_known_classes]
unknown_classes:[N_unknown_classes]
```

Run evaluation:

```bash
python scripts/eval_npz.py \
  --features features/my_dataset_features.npz \
  --checkpoint checkpoints/gcdformer_paper.pt \
  --reduction tsne \
  --latent-dim 2 \
  --samples-per-class 20
```

If `--checkpoint` is omitted, the script reports the reduced-feature k-means baseline.

## Feature extraction examples

### Vision: CIFAR-10 with DINOv2

```bash
python scripts/extract_vision_features.py \
  --dataset cifar10 \
  --encoder dinov2_vitb14 \
  --data-root data \
  --output features/dinov2_cifar10.npz
```

Then evaluate:

```bash
python scripts/eval_npz.py --features features/dinov2_cifar10.npz --reduction tsne --samples-per-class 20
```

### Text: TSV files

Prepare `train.tsv` and `test.tsv` with columns `text` and `label`, then run:

```bash
python scripts/extract_text_features.py \
  --train-tsv path/to/train.tsv \
  --test-tsv path/to/test.tsv \
  --encoder intfloat/e5-large-v2 \
  --output features/e5_text_dataset.npz
```

### Audio: HuggingFace audio datasets

```bash
python scripts/extract_audio_features.py \
  --hf-dataset danavery/urbansound8K \
  --label-column classID \
  --known-classes 0 1 2 3 4 \
  --unknown-classes 5 6 7 8 9 \
  --output features/mert_urbansound.npz
```

### Remote sensing

```bash
python scripts/extract_remote_sensing_features.py \
  --dataset resisc45 \
  --data-root data \
  --encoder resnet50 \
  --output features/resisc45_resnet50.npz
```

The paper used DOFA for multispectral remote-sensing experiments. This draft remote-sensing script is a clean RGB/torchgeo starting point; DOFA support can be added while preserving the same NPZ schema.

## Notes for reproducing paper numbers

- Use the released paper checkpoint once available.
- Use t-SNE reduction (`--reduction tsne --latent-dim 2`) for the main OmniGCD setting.
- Use the exact dataset splits and samples-per-class values reported in the supplementary material.
- The provided toy training config is a sanity check, not the final training recipe.

## Citation

```bibtex
@article{shipard2026omnigcd,
  title={OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism},
  author={Shipard, Jordan and Wiliem, Arnold and Nguyen Thanh, Kien and Xiang, Wei and Fookes, Clinton},
  journal={arXiv preprint arXiv:2604.14762},
  year={2026}
}
```
