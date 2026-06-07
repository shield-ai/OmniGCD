#!/usr/bin/env python
"""Train GCDformer on synthetic GCD latent spaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_gcdformer.yaml")
    parser.add_argument("--output", default=None, help="Checkpoint path. Overrides config output.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    import torch

    from omnigcd.data import SyntheticBatchConfig, generate_synthetic_batch
    from omnigcd.models import GCDFormer, GCDFormerConfig
    from omnigcd.training import pairwise_contrastive_loss

    cfg = load_config(args.config)
    model_cfg = GCDFormerConfig.from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    synthetic_cfg = SyntheticBatchConfig(**cfg.get("synthetic", {}))

    device = args.device or train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs or int(train_cfg.get("epochs", 10))
    steps_per_epoch = args.steps_per_epoch or int(train_cfg.get("steps_per_epoch", 100))
    lr = float(train_cfg.get("lr", 1e-4))
    margin = float(train_cfg.get("margin", 1.0))
    output = args.output or cfg.get("output", "checkpoints/gcdformer_synthetic.pt")

    synthetic_cfg.device = device
    synthetic_cfg.data_dim = model_cfg.input_dim
    synthetic_cfg.max_label_id = model_cfg.max_label_id

    model = GCDFormer(model_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-4)))

    print("Training GCDformer")
    print(f"device={device}, epochs={epochs}, steps_per_epoch={steps_per_epoch}, output={output}")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            points, labels, masked_labels, mask = generate_synthetic_batch(synthetic_cfg)
            output_points = model(points, masked_labels, mask)
            loss = pairwise_contrastive_loss(output_points, labels, margin=margin)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.detach().cpu())
        print(f"epoch {epoch + 1:04d}/{epochs:04d} loss={total / steps_per_epoch:.6f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(output)
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
