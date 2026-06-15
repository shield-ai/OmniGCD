#!/usr/bin/env python
"""Convert the original research checkpoint into the public checkpoint format.

The original GCDformer checkpoint was saved as a raw PyTorch state dict from the
research `ToyModel` implementation, often with `_orig_mod.` prefixes introduced
by `torch.compile`. The public repo can load that raw format directly, but this
script packages it with explicit metadata and config so users have a stable
checkpoint format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="checkpoints/tda2mtv3.pt", help="Legacy raw checkpoint path")
    parser.add_argument("--output", default="checkpoints/omnigcd_gcdformer.pt", help="Converted public checkpoint path")
    args = parser.parse_args()

    import torch
    from omnigcd.models import GCDFormer

    model = GCDFormer.load_checkpoint(args.input, map_location="cpu")
    architecture = "legacy_toymodel" if type(model).__name__ == "LegacyGCDFormer" else "gcdformer"
    checkpoint = {
        "format_version": 1,
        "architecture": architecture,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": {
            "source_checkpoint": str(args.input),
            "note": "Converted from the original OmniGCD research checkpoint.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Converted {args.input} -> {output}")
    print(f"architecture: {architecture}")
    print(f"config: {model.config}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
