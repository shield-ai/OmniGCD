"""GCD evaluation metrics."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def _assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a linear assignment problem.

    Uses SciPy when available. Falls back to exhaustive search for small matrices
    and a greedy approximation for large matrices, which is sufficient for the
    toy smoke test but not recommended for paper numbers.
    """
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ModuleNotFoundError:
        n_rows, n_cols = cost.shape
        n = min(n_rows, n_cols)
        if max(n_rows, n_cols) <= 10:
            best_perm = None
            best_cost = float("inf")
            for perm in permutations(range(n_cols), n_rows):
                value = sum(cost[i, perm[i]] for i in range(n_rows))
                if value < best_cost:
                    best_cost = value
                    best_perm = perm
            rows = np.arange(n_rows)
            cols = np.asarray(best_perm, dtype=np.int64)
            return rows, cols

        # Greedy fallback for environments without SciPy. This keeps the demo
        # runnable but should not be used for official results.
        rows, cols = [], []
        used_rows, used_cols = set(), set()
        for flat_idx in np.argsort(cost, axis=None):
            r, c = np.unravel_index(flat_idx, cost.shape)
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
            if len(rows) == n:
                break
        return np.asarray(rows), np.asarray(cols)


def split_cluster_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    old_class_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute standard GCD All/Old/New clustering accuracies.

    Args:
        y_true: Ground-truth integer labels for evaluated samples.
        y_pred: Cluster IDs for evaluated samples.
        old_class_mask: Boolean mask where ``True`` means the sample belongs to
            an old/known class and ``False`` means it belongs to a novel class.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    old_class_mask = np.asarray(old_class_mask, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    pred_ids = np.unique(y_pred)
    true_ids = np.unique(y_true)
    pred_to_row = {c: i for i, c in enumerate(pred_ids)}
    true_to_col = {c: i for i, c in enumerate(true_ids)}
    contingency = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for pred, true in zip(y_pred, y_true):
        contingency[pred_to_row[pred], true_to_col[true]] += 1

    rows, cols = _assignment(contingency.max() - contingency)
    mapping = {true_ids[c]: pred_ids[r] for r, c in zip(rows, cols)}

    def _subset_acc(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        correct = 0
        for true_label in np.unique(y_true[mask]):
            mapped_pred = mapping.get(true_label, None)
            if mapped_pred is None:
                continue
            correct += np.sum((y_true == true_label) & (y_pred == mapped_pred) & mask)
        return float(correct / mask.sum())

    return _subset_acc(np.ones_like(old_class_mask, dtype=bool)), _subset_acc(old_class_mask), _subset_acc(~old_class_mask)
