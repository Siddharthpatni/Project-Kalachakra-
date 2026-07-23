"""
Domain 25 — Information-theoretic feature selection.

Ranks and selects features by mutual information with the target. ``mrmr``
(minimum-redundancy maximum-relevance) greedily builds a feature set that is
individually informative about the target yet not redundant with features
already chosen — the information-theoretic answer to "which features actually
add something new?"
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kalachakra.information.mutual_information import (
    mutual_information,
    mutual_information_continuous,
)
from kalachakra.math.numerics import discretize


def mi_ranking(
    X: npt.ArrayLike, y: npt.ArrayLike, feature_names: list[str], bins: int = 10
) -> list[tuple[str, float]]:
    """Rank features by mutual information with the target.

    Args:
        X: Design matrix ``(n_samples, n_features)``.
        y: Target labels.
        feature_names: One name per column.
        bins: Discretization bins for continuous features.

    Returns:
        ``(name, mutual_information)`` pairs sorted by MI descending.
    """
    matrix = np.asarray(X, dtype=np.float64)
    scores = [
        (name, mutual_information_continuous(matrix[:, j], y, bins=bins))
        for j, name in enumerate(feature_names)
    ]
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def mrmr(
    X: npt.ArrayLike, y: npt.ArrayLike, feature_names: list[str], k: int, bins: int = 10
) -> list[str]:
    """Select ``k`` features by minimum-redundancy maximum-relevance.

    Relevance is a feature's MI with the target; redundancy is its mean MI with
    features already selected. Each step adds the feature maximizing
    ``relevance − redundancy``.

    Args:
        X: Design matrix ``(n_samples, n_features)``.
        y: Target labels.
        feature_names: One name per column.
        k: Number of features to select.
        bins: Discretization bins.

    Returns:
        Selected feature names, in selection order.

    Raises:
        ValueError: If ``k`` exceeds the number of features.
    """
    matrix = np.asarray(X, dtype=np.float64)
    n_features = matrix.shape[1]
    if k > n_features:
        raise ValueError(f"k={k} exceeds n_features={n_features}")

    disc = np.column_stack([discretize(matrix[:, j], bins=bins) for j in range(n_features)])
    relevance = np.array(
        [mutual_information_continuous(matrix[:, j], y, bins=bins) for j in range(n_features)]
    )

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = set(range(n_features)) - set(selected)

    while len(selected) < k and remaining:
        best_idx, best_score = -1, -np.inf
        for j in remaining:
            redundancy = float(
                np.mean([mutual_information(disc[:, j], disc[:, s]) for s in selected])
            )
            score = relevance[j] - redundancy
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [feature_names[i] for i in selected]
