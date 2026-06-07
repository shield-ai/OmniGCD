#!/usr/bin/env python
"""Extract vision features for OmniGCD.

This intentionally starts with CIFAR-10/100 as a simple public example. Add your
CVPR benchmark datasets by returning the same NPZ schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_encoder(name: str):
    import timm
    import torch

    if name == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif name == "dino_vitb16":
        model = torch.hub.load("facebookresearch/dino", "dino_vitb16")
    else:
        model = timm.create_model(name, pretrained=True, num_classes=0)
    return model.eval()


def get_transform(image_size: int = 224):
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dataset(name: str, root: str):
    from torchvision.datasets import CIFAR10, CIFAR100

    transform = get_transform()
    if name == "cifar10":
        train = CIFAR10(root=root, train=True, download=True, transform=transform)
        test = CIFAR10(root=root, train=False, download=True, transform=transform)
        known = np.arange(5, dtype=np.int64)
        unknown = np.arange(5, 10, dtype=np.int64)
    elif name == "cifar100":
        train = CIFAR100(root=root, train=True, download=True, transform=transform)
        test = CIFAR100(root=root, train=False, download=True, transform=transform)
        known = np.arange(80, dtype=np.int64)
        unknown = np.arange(80, 100, dtype=np.int64)
    else:
        raise ValueError("This draft extractor supports cifar10/cifar100. Add custom datasets with the same output schema.")
    return train, test, known, unknown


def extract(loader, model, device):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    features, labels = [], []
    model = model.to(device)
    with torch.no_grad():
        for images, target in tqdm(loader, desc="extract"):
            images = images.to(device)
            feats = model(images)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--encoder", default="dinov2_vitb14")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or f"features/{args.encoder}_{args.dataset}.npz"
    train, test, known, unknown = load_dataset(args.dataset, args.data_root)
    model = load_encoder(args.encoder)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_features, train_labels = extract(train_loader, model, device)
    test_features, test_labels = extract(test_loader, model, device)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, train_features=train_features, train_labels=train_labels, test_features=test_features,
             test_labels=test_labels, known_classes=known, unknown_classes=unknown)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
