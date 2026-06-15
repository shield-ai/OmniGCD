# Checkpoints

This directory contains released GCDformer checkpoints.

## `omnigcd_gcdformer.pt` (recommended)

`omnigcd_gcdformer.pt` is the public packaged GCDformer checkpoint. It contains explicit metadata, the inferred model config, and the state dict:

```python
{
    "format_version": 1,
    "architecture": "legacy_toymodel",
    "config": {...},
    "state_dict": {...},
    "metadata": {...},
}
```

Load it with:

```python
from omnigcd.models import GCDFormer
model = GCDFormer.load_checkpoint("checkpoints/omnigcd_gcdformer.pt", map_location="cpu")
```

## `tda2mtv3.pt` (legacy source checkpoint)

`tda2mtv3.pt` is the original raw PyTorch state-dict checkpoint from the research code. It is kept for provenance and corresponds to the original model ID used in the development scripts. The public loader supports this raw format directly, but new users should prefer `omnigcd_gcdformer.pt`.

To recreate the packaged checkpoint from the legacy checkpoint:

```bash
python scripts/convert_legacy_checkpoint.py \
  --input checkpoints/tda2mtv3.pt \
  --output checkpoints/omnigcd_gcdformer.pt
```

## Architecture

Both checkpoint files represent the same trained model. The architecture is:

- input/output latent dimension: 2
- data token dimension: 224
- label token dimension: 32
- transformer width: 256
- layers: 6
- attention heads: 4
- max label vocabulary: 1000
