#!/usr/bin/env python
"""Extract text features from train/test TSV files.

TSV format: two columns named `text` and `label`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def encode(df, classes, model, batch_size):
    labels = np.asarray([classes.index(x) for x in df["label"].tolist()], dtype=np.int64)
    embs = []
    for start in range(0, len(df), batch_size):
        texts = df["text"].iloc[start:start + batch_size].tolist()
        embs.append(model.encode(texts, normalize_embeddings=True))
    return np.concatenate(embs, axis=0).astype(np.float32), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tsv", required=True)
    parser.add_argument("--test-tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--known-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    train_df = pd.read_csv(args.train_tsv, sep="\t")
    test_df = pd.read_csv(args.test_tsv, sep="\t")
    classes = sorted(train_df["label"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    class_ids = np.arange(len(classes), dtype=np.int64)
    known = np.sort(rng.choice(class_ids, size=int(len(classes) * args.known_ratio), replace=False))
    unknown = np.asarray([c for c in class_ids if c not in set(known)], dtype=np.int64)

    model = SentenceTransformer(args.encoder)
    train_features, train_labels = encode(train_df, classes, model, args.batch_size)
    test_features, test_labels = encode(test_df, classes, model, args.batch_size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, train_features=train_features, train_labels=train_labels, test_features=test_features,
             test_labels=test_labels, known_classes=known, unknown_classes=unknown)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
