#!/usr/bin/env python
"""Draft remote-sensing feature extractor.

This script is intentionally conservative: it supports RGB-style torchgeo image
classification datasets with a timm encoder. For DOFA/multispectral reproduction,
replace `load_encoder` and `forward_features` with the DOFA call used in the
research code, then keep the same NPZ output schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["resisc45", "ucmerced"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import timm
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchgeo.datasets import RESISC45, UCMerced
    from tqdm import tqdm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = lambda sample: {"image": T.Compose([T.Resize((224, 224)), T.ConvertImageDtype(torch.float32)])(sample["image"]), "label": sample["label"]}
    if args.dataset == "resisc45":
        train = RESISC45(root=args.data_root, split="train", download=True, transforms=transform)
        test = RESISC45(root=args.data_root, split="test", download=True, transforms=transform)
        known, unknown = np.arange(15, dtype=np.int64), np.arange(15, 45, dtype=np.int64)
    else:
        train = UCMerced(root=args.data_root, split="train", download=True, transforms=transform)
        test = UCMerced(root=args.data_root, split="test", download=True, transforms=transform)
        known, unknown = np.arange(9, dtype=np.int64), np.arange(9, 21, dtype=np.int64)

    model = timm.create_model(args.encoder, pretrained=True, num_classes=0).to(device).eval()

    def extract(ds):
        feats, labs = [], []
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for batch in tqdm(loader, desc="remote features"):
                x = batch["image"].to(device)
                y = batch["label"].numpy()
                feats.append(model(x).cpu().numpy())
                labs.append(y)
        return np.concatenate(feats).astype(np.float32), np.concatenate(labs).astype(np.int64)

    train_features, train_labels = extract(train)
    test_features, test_labels = extract(test)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, train_features=train_features, train_labels=train_labels, test_features=test_features,
             test_labels=test_labels, known_classes=known, unknown_classes=unknown)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
