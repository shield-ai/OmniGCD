# Checkpoints

This directory contains released GCDformer checkpoints.

## `tda2mtv3.pt`

`tda2mtv3.pt` is the original GCDformer checkpoint from the research code used for the OmniGCD paper experiments. It is stored in the legacy raw PyTorch state-dict format produced by the original `ToyModel` implementation. The public loader supports this format directly:

```python
from omnigcd.models import GCDFormer
model = GCDFormer.load_checkpoint("checkpoints/tda2mtv3.pt", map_location="cpu")
```

The checkpoint architecture inferred by the loader is:

- input/output latent dimension: 2
- data token dimension: 224
- label token dimension: 32
- transformer width: 256
- layers: 6
- attention heads: 4
- max label vocabulary: 1000
