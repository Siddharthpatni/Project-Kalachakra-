"""
Domain 25 — Fisher discriminant score.

A fast, model-free feature ranker for classification targets: the ratio of
between-class to within-class variance. High scores mark features whose class
means are well separated relative to their spread — a useful complement to the
mutual-information screen (Fisher score is linear/second-order; MI captures
arbitrary dependence).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def fisher_score(
    X: npt.ArrayLike, y: npt.ArrayLike, eps: float = 1e-12
) -> npt.NDArray[np.float64]:
    """Fisher discriminant score per feature for a classification target.

    Args:
        X: Design matrix ``(n_samples, n_features)``.
        y: Class labels of length ``n_samples``.
        eps: Floor on within-class variance to avoid division by zero.

    Returns:
        Array of length ``n_features`` with a non-negative score per feature;
        larger means better class separation.
    """
    matrix = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).ravel()
    classes = np.unique(labels)
    global_mean = matrix.mean(axis=0)

    between = np.zeros(matrix.shape[1], dtype=np.float64)
    within = np.zeros(matrix.shape[1], dtype=np.float64)
    for c in classes:
        members = matrix[labels == c]
        n_c = members.shape[0]
        class_mean = members.mean(axis=0)
        between += n_c * (class_mean - global_mean) ** 2
        within += n_c * members.var(axis=0)

    return between / np.clip(within, eps, None)
