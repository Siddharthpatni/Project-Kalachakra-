"""
Domain 25 — Mutual information.

Mutual information measures how many bits knowing one variable saves about
another; it is zero exactly when the variables are independent, so it is the
right first screen for "does this feature carry any information about the
target?" Discrete estimators here compose the entropy building blocks; a
continuous helper discretizes first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kalachakra.information.entropy import (
    conditional_entropy,
    entropy,
    joint_entropy,
)
from kalachakra.math.numerics import discretize


def mutual_information(x: npt.ArrayLike, y: npt.ArrayLike, base: float = 2.0) -> float:
    """Mutual information ``I(X; Y) = H(X) + H(Y) − H(X, Y)`` for discrete data.

    Args:
        x: First discrete sample.
        y: Second discrete sample (same length).
        base: Logarithm base (2 → bits).

    Returns:
        Non-negative mutual information; 0 iff ``X`` and ``Y`` are independent.
    """
    mi = entropy(x, base) + entropy(y, base) - joint_entropy(x, y, base)
    return max(0.0, mi)  # clamp tiny negative values from floating point


def normalized_mutual_information(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Mutual information normalized to ``[0, 1]``.

    Uses ``I(X;Y) / sqrt(H(X) H(Y))``. Returns 0 when either variable is
    constant (zero entropy).

    Args:
        x: First discrete sample.
        y: Second discrete sample.

    Returns:
        Normalized mutual information in ``[0, 1]``.
    """
    hx, hy = entropy(x), entropy(y)
    denom = np.sqrt(hx * hy)
    if denom <= 0.0:
        return 0.0
    return float(mutual_information(x, y) / denom)


def conditional_mutual_information(
    x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, base: float = 2.0
) -> float:
    """Conditional mutual information ``I(X; Y | Z)``.

    Computed as ``H(X | Z) − H(X | Y, Z)``. Positive values mean ``Y`` adds
    information about ``X`` beyond what ``Z`` already carries.

    Args:
        x: Target sample.
        y: Candidate sample.
        z: Conditioning sample.
        base: Logarithm base.

    Returns:
        Non-negative conditional mutual information.
    """
    ya = np.asarray(y).ravel()
    za = np.asarray(z).ravel()
    yz = np.stack([ya, za], axis=1)
    # Encode the (y, z) pairs as single labels for the second conditional term.
    _, yz_labels = np.unique(yz, axis=0, return_inverse=True)
    cmi = conditional_entropy(x, z, base) - conditional_entropy(x, yz_labels, base)
    return max(0.0, cmi)


def information_gain(x: npt.ArrayLike, y: npt.ArrayLike, base: float = 2.0) -> float:
    """Information gain of ``Y`` about ``X`` — an alias for mutual information.

    Args:
        x: Target sample.
        y: Feature sample.
        base: Logarithm base.

    Returns:
        Mutual information ``I(X; Y)``.
    """
    return mutual_information(x, y, base)


def mutual_information_continuous(
    x: npt.ArrayLike, y: npt.ArrayLike, bins: int = 10, strategy: str = "quantile"
) -> float:
    """Mutual information between a continuous feature and a target.

    The continuous variable is discretized into ``bins`` before the discrete
    estimator is applied; the target is used as-is if already discrete.

    Args:
        x: Continuous feature values.
        y: Target values (discrete labels, or continuous — also discretized).
        bins: Number of bins for discretization.
        strategy: ``"uniform"`` or ``"quantile"`` binning.

    Returns:
        Estimated mutual information in bits.
    """
    xb = discretize(x, bins=bins, strategy=strategy)
    ya = np.asarray(y).ravel()
    if np.issubdtype(ya.dtype, np.floating):
        ya = discretize(ya, bins=bins, strategy=strategy)
    return mutual_information(xb, ya)
