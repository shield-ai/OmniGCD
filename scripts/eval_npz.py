#!/usr/bin/env python
"""Evaluate OmniGCD from precomputed features saved as an NPZ file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnigcd.eval import build_gcd_sequence, kmeans, load_feature_npz, reduce_features, split_cluster_accuracy


def run_gcdformer_if_requested(latent, labels, mask, checkpoint):
    if checkpoint is None:
        return latent
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    import torch
    from omnigcd.models import GCDFormer

    model = GCDFormer.load_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    with torch.no_grad():
        points = torch.from_numpy(latent).float().unsqueeze(0)
        torch_labels = torch.from_numpy(labels).long().unsqueeze(0)
        torch_mask = torch.from_numpy(mask).long().unsqueeze(0)
        output = model(points, torch_labels, torch_mask)[0].cpu().numpy()
    return output.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to feature NPZ file")
    parser.add_argument("--checkpoint", default=None, help="Optional GCDformer checkpoint. If omitted, evaluate reduced features only.")
    parser.add_argument("--reduction", choices=["pca", "tsne", "umap", "none"], default="pca")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--kmeans-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arrays = load_feature_npz(args.features)
    seq = build_gcd_sequence(
        arrays["train_features"],
        arrays["train_labels"],
        arrays["test_features"],
        arrays["test_labels"],
        arrays["known_classes"],
        arrays["unknown_classes"],
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    latent = reduce_features(seq.joined_features, method=args.reduction, n_components=args.latent_dim, seed=args.seed)
    output = run_gcdformer_if_requested(latent, seq.masked_labels, seq.model_mask, args.checkpoint)
    test_output = output[seq.n_known :]

    preds, _, inertia = kmeans(test_output, n_clusters=seq.n_clusters, n_init=args.kmeans_runs, seed=args.seed)
    all_acc, old_acc, new_acc = split_cluster_accuracy(seq.test_labels, preds, seq.old_class_mask)

    print("---- OmniGCD evaluation ----")
    print(f"features: {args.features}")
    print(f"checkpoint: {args.checkpoint or 'none (reduced-feature baseline)'}")
    print(f"reduction: {args.reduction}, latent_dim: {args.latent_dim}")
    print(f"n_known_tokens: {seq.n_known}, n_test: {len(seq.test_labels)}, n_clusters: {seq.n_clusters}")
    print(f"kmeans_inertia: {inertia:.4f}")
    print(f"All: {all_acc * 100:.2f} | Old: {old_acc * 100:.2f} | New: {new_acc * 100:.2f}")


if __name__ == "__main__":
    main()
