#!/usr/bin/env python
"""Draft audio feature extractor for HuggingFace audio datasets.

Example datasets from the paper are VocalSet and UrbanSound8K. Because their HF
schemas can change, this script exposes column names as arguments and writes the
standard OmniGCD NPZ schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", required=True, help="e.g. Bill13579/vocalset-mirror or danavery/urbansound8K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--known-classes", type=int, nargs="+", required=True)
    parser.add_argument("--unknown-classes", type=int, nargs="+", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset(args.hf_dataset, split=args.split)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.encoder, trust_remote_code=True).to(device).eval()

    features, labels = [], []
    with torch.no_grad():
        for item in tqdm(dataset, desc="audio features"):
            audio = item["audio"]["array"]
            inputs = processor(audio, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = torch.stack(outputs.hidden_states).squeeze()
            # Layer 4 was used in project experiments; adjust if needed.
            feat = hidden[4].mean(dim=0).cpu().numpy()
            features.append(feat)
            labels.append(int(item[args.label_column]))

    features = np.stack(features).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(len(labels))
    split = int(len(labels) * (1.0 - args.test_ratio))
    train_idx, test_idx = perm[:split], perm[split:]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, train_features=features[train_idx], train_labels=labels[train_idx],
             test_features=features[test_idx], test_labels=labels[test_idx],
             known_classes=np.asarray(args.known_classes, dtype=np.int64),
             unknown_classes=np.asarray(args.unknown_classes, dtype=np.int64))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
