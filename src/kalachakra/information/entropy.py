"""
Domain 25 — Entropy measures.

Discrete Shannon entropy and its joint/conditional forms, in bits by default.
Continuous variables must be discretized first (see
:func:`kalachakra.math.numerics.discretize`). These are the building blocks the
mutual-information estimators and the signal gate are written on top of.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _probabilities(labels: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Empirical probability of each distinct value in a label array."""
    arr = np.asarray(labels).ravel()
    if arr.size == 0:
        raise ValueError("entropy requires a non-empty sample")
    _, counts = np.unique(arr, return_counts=True)
    return counts.astype(np.float64) / arr.size


def _log(x: npt.NDArray[np.float64], base: float) -> npt.NDArray[np.float64]:
    return np.log(x) / np.log(base)


def entropy(labels: npt.ArrayLike, base: float = 2.0) -> float:
    """Shannon entropy of a sample of discrete labels.

    Args:
        labels: Sample of discrete values.
        base: Logarithm base (2 → bits, e → nats).

    Returns:
        Entropy ``H(X) = -Σ p log_base p`` in units set by ``base``.
    """
    p = _probabilities(labels)
    return float(-np.sum(p * _log(p, base)))


def joint_entropy(x: npt.ArrayLike, y: npt.ArrayLike, base: float = 2.0) -> float:
    """Joint entropy ``H(X, Y)`` of two aligned discrete samples.

    Args:
        x: First sample.
        y: Second sample (same length as ``x``).
        base: Logarithm base.

    Returns:
        Joint entropy.

    Raises:
        ValueError: If the samples differ in length.
    """
    xa = np.asarray(x).ravel()
    ya = np.asarray(y).ravel()
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    pairs = np.stack([xa, ya], axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    p = counts.astype(np.float64) / xa.size
    return float(-np.sum(p * _log(p, base)))


def conditional_entropy(x: npt.ArrayLike, y: npt.ArrayLike, base: float = 2.0) -> float:
    """Conditional entropy ``H(X | Y) = H(X, Y) − H(Y)``.

    Args:
        x: Target sample.
        y: Conditioning sample.
        base: Logarithm base.

    Returns:
        Remaining uncertainty in ``X`` once ``Y`` is known.
    """
    return joint_entropy(x, y, base) - entropy(y, base)


def cross_entropy(
    p: npt.ArrayLike, q: npt.ArrayLike, base: float = 2.0, eps: float = 1e-12
) -> float:
    """Cross entropy ``H(p, q) = -Σ p log q`` between two distributions.

    Args:
        p: True distribution (non-negative, sums to 1).
        q: Model distribution (same length as ``p``).
        base: Logarithm base.
        eps: Floor applied to ``q`` to avoid ``log(0)``.

    Returns:
        Cross entropy.

    Raises:
        ValueError: If the distributions differ in length.
    """
    pa = np.asarray(p, dtype=np.float64)
    qa = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    if pa.shape != qa.shape:
        raise ValueError("p and q must have the same shape")
    return float(-np.sum(pa * _log(qa, base)))
