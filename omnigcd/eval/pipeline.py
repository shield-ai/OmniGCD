"""Reusable zero-shot GCD evaluation pipeline utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GCDSequence:
    joined_features: np.ndarray
    masked_labels: np.ndarray
    model_mask: np.ndarray
    test_labels: np.ndarray
    old_class_mask: np.ndarray
    n_known: int
    n_clusters: int


def load_feature_npz(path: str) -> dict[str, np.ndarray]:
    """Load feature arrays saved by the extraction scripts.

    Expected keys are ``train_features``, ``train_labels``, ``test_features``,
    ``test_labels``, ``known_classes`` and ``unknown_classes``.
    """
    data = np.load(path, allow_pickle=True)
    required = {
        "train_features",
        "train_labels",
        "test_features",
        "test_labels",
        "known_classes",
        "unknown_classes",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
    return {k: data[k] for k in required}


def build_gcd_sequence(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    known_classes: np.ndarray | list[int],
    unknown_classes: np.ndarray | list[int],
    samples_per_class: int,
    seed: int = 0,
) -> GCDSequence:
    """Build the transductive GCD sequence used by OmniGCD.

    The labeled part contains up to ``samples_per_class`` training samples from
    each known class. The unlabeled part is the whole test set, with labels hidden
    from the model.
    """
    rng = np.random.default_rng(seed)
    known_classes = np.asarray(known_classes, dtype=np.int64)
    unknown_classes = np.asarray(unknown_classes, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    known_feats = []
    known_labs = []
    for cls in known_classes:
        idx = np.flatnonzero(train_labels == cls)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        idx = idx[:samples_per_class]
        known_feats.append(train_features[idx])
        known_labs.append(train_labels[idx])

    if not known_feats:
        raise ValueError("No known-class training examples were found")

    known_feats = np.concatenate(known_feats, axis=0)
    known_labs = np.concatenate(known_labs, axis=0).astype(np.int64)
    joined_features = np.concatenate([known_feats, test_features], axis=0).astype(np.float32)

    # Shift labels by one because 0 is reserved for masked/unlabeled labels.
    masked_labels = np.concatenate(
        [known_labs + 1, np.zeros(len(test_labels), dtype=np.int64)],
        axis=0,
    )
    model_mask = np.concatenate(
        [np.zeros(len(known_labs), dtype=np.int64), np.ones(len(test_labels), dtype=np.int64)],
        axis=0,
    )

    old_class_mask = np.isin(test_labels, known_classes)
    n_clusters = len(known_classes) + len(unknown_classes)
    return GCDSequence(
        joined_features=joined_features,
        masked_labels=masked_labels,
        model_mask=model_mask,
        test_labels=test_labels,
        old_class_mask=old_class_mask,
        n_known=len(known_labs),
        n_clusters=n_clusters,
    )


def normalize_nd(x: np.ndarray, out_min: float = -1.0, out_max: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    min_vals = x.min(axis=0, keepdims=True)
    max_vals = x.max(axis=0, keepdims=True)
    x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    return x * (out_max - out_min) + out_min


def _pca(x: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:n_components].T).astype(np.float32)


def reduce_features(
    features: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Reduce features into a low-dimensional GCD latent space.

    ``pca`` is implemented in NumPy for lightweight smoke tests. ``tsne`` and
    ``umap`` require their optional packages and are used for benchmark-style
    runs matching the paper.
    """
    method = method.lower()
    if method == "none":
        reduced = np.asarray(features, dtype=np.float32)
    elif method == "pca":
        reduced = _pca(features, n_components=n_components)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reduced = TSNE(
            n_components=n_components,
            init="random",
            learning_rate=400,
            random_state=seed,
        ).fit_transform(features)
    elif method == "umap":
        import umap

        reduced = umap.UMAP(n_components=n_components, metric="cosine", random_state=seed).fit_transform(features)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return normalize_nd(reduced.astype(np.float32))
