"""Small NumPy k-means implementation used by the smoke-test evaluator.

The original experiments used scikit-learn k-means. This implementation keeps
the public quickstart lightweight and deterministic. For large benchmark runs,
scikit-learn is still recommended.
"""

from __future__ import annotations

import numpy as np


def _init_centers(x: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) < n_clusters:
        raise ValueError("n_clusters cannot exceed number of samples")
    indices = rng.choice(len(x), size=n_clusters, replace=False)
    return x[indices].copy()


def kmeans(
    x: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run k-means and return ``labels, centers, inertia``."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        centers = _init_centers(x, n_clusters, rng)
        labels = np.zeros(len(x), dtype=np.int64)
        for _iter in range(max_iter):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for k in range(n_clusters):
                members = x[new_labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    new_centers[k] = x[rng.integers(0, len(x))]
            if np.array_equal(labels, new_labels):
                centers = new_centers
                labels = new_labels
                break
            centers = new_centers
            labels = new_labels

        inertia = ((x - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = float(inertia)

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia
