"""
Domain 25 — The signal gate.

This is the module the whole project hinges on. Before any model is trained,
each feature is tested for whether it carries *statistically significant*
information about the target. Observed mutual information is compared against a
null distribution built by repeatedly shuffling the target: if the real MI is
not clearly above what random alignment produces, the feature is noise and is
gated to zero — no matter how accurate a downstream model later appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kalachakra.core.logging import get_logger
from kalachakra.information.mutual_information import mutual_information
from kalachakra.math.numerics import discretize

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of the signal gate for one feature."""

    feature: str
    mutual_information: float
    p_value: float
    passed: bool


def _as_labels(v: npt.NDArray[np.float64], bins: int, strategy: str) -> npt.NDArray[np.int64]:
    """Discretize a continuous column; pass integer-valued columns through."""
    if np.allclose(v, np.round(v)) and np.unique(v).size <= max(bins, 20):
        return v.astype(np.int64)
    return discretize(v, bins=bins, strategy=strategy)


def mi_permutation_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n_permutations: int = 200,
    bins: int = 10,
    strategy: str = "quantile",
    seed: int | None = None,
) -> tuple[float, float]:
    """Permutation test for the mutual information between ``x`` and ``y``.

    Args:
        x: Feature values (continuous or discrete).
        y: Target values (discrete labels).
        n_permutations: Number of target shuffles forming the null distribution.
        bins: Bins used to discretize a continuous feature.
        strategy: Binning strategy for discretization.
        seed: Seed for the permutation generator (reproducibility).

    Returns:
        ``(observed_mi, p_value)``. The p-value is the fraction of shuffles whose
        MI meets or exceeds the observed MI, with the standard ``+1`` correction.
    """
    xa = _as_labels(np.asarray(x, dtype=np.float64).ravel(), bins, strategy)
    ya = np.asarray(y).ravel()
    observed = mutual_information(xa, ya)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(ya)
        if mutual_information(xa, shuffled) >= observed:
            ge += 1
    p_value = (ge + 1) / (n_permutations + 1)
    return observed, p_value


def signal_gate(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    feature_names: list[str],
    alpha: float = 0.05,
    n_permutations: int = 200,
    bins: int = 10,
    strategy: str = "quantile",
    seed: int | None = None,
) -> list[GateResult]:
    """Run the mutual-information signal gate over every feature.

    Args:
        X: Design matrix of shape ``(n_samples, n_features)``.
        y: Target labels of length ``n_samples``.
        feature_names: One name per column of ``X``.
        alpha: Significance threshold; a feature passes if ``p_value <= alpha``.
        n_permutations: Shuffles per feature.
        bins: Discretization bins for continuous features.
        strategy: Binning strategy.
        seed: Base seed; feature ``j`` uses ``seed + j`` for independence.

    Returns:
        One :class:`GateResult` per feature, sorted by mutual information
        (descending).

    Raises:
        ValueError: If ``feature_names`` does not match the number of columns.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of columns in X")

    results: list[GateResult] = []
    for j, name in enumerate(feature_names):
        col_seed = None if seed is None else seed + j
        mi, p = mi_permutation_test(
            matrix[:, j], y, n_permutations, bins, strategy, col_seed
        )
        results.append(GateResult(name, mi, p, p <= alpha))

    results.sort(key=lambda r: r.mutual_information, reverse=True)
    n_pass = sum(r.passed for r in results)
    log.info(f"Signal gate: {n_pass}/{len(results)} features passed at alpha={alpha}")
    return results
