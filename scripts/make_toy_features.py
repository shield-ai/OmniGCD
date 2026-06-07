#!/usr/bin/env python
"""Create a tiny synthetic feature file for README smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(rng, centers, n_per_class, noise):
    features = []
    labels = []
    for cls, center in enumerate(centers):
        features.append(center + rng.normal(scale=noise, size=(n_per_class, centers.shape[1])))
        labels.extend([cls] * n_per_class)
    return np.concatenate(features, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/toy_features.npz")
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--known-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    centers = rng.normal(size=(args.n_classes, args.feature_dim)).astype(np.float32) * 5.0
    train_features, train_labels = make_split(rng, centers, args.train_per_class, noise=0.8)
    test_features, test_labels = make_split(rng, centers, args.test_per_class, noise=1.0)

    known_classes = np.arange(args.known_classes, dtype=np.int64)
    unknown_classes = np.arange(args.known_classes, args.n_classes, dtype=np.int64)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )
    print(f"Wrote {out}")
    print(f"train_features={train_features.shape}, test_features={test_features.shape}")
    print(f"known_classes={known_classes.tolist()}, unknown_classes={unknown_classes.tolist()}")


if __name__ == "__main__":
    main()
