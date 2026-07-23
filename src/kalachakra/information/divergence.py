"""
Domain 25 — Divergences between probability distributions.

Kullback–Leibler (asymmetric), Jensen–Shannon (symmetric, bounded), and Rényi
divergences. Inputs are probability vectors over the same support; they are
renormalized defensively so small numerical drift does not distort the result.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _normalize(p: npt.ArrayLike, eps: float) -> npt.NDArray[np.float64]:
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    return arr / arr.sum()


def kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: float = 2.0, eps: float = 1e-12) -> float:
    """Kullback–Leibler divergence ``D(p ‖ q) = Σ p log(p/q)``.

    Args:
        p: Reference distribution.
        q: Comparison distribution (same length as ``p``).
        base: Logarithm base (2 → bits).
        eps: Floor applied before normalization to avoid division by zero.

    Returns:
        Non-negative divergence; 0 iff ``p == q``. Asymmetric in ``p`` and ``q``.

    Raises:
        ValueError: If the distributions differ in length.
    """
    pa = _normalize(p, eps)
    qa = _normalize(q, eps)
    if pa.shape != qa.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.sum(pa * (np.log(pa / qa) / np.log(base))))


def jensen_shannon_divergence(
    p: npt.ArrayLike, q: npt.ArrayLike, base: float = 2.0, eps: float = 1e-12
) -> float:
    """Jensen–Shannon divergence — symmetric, bounded in ``[0, 1]`` (base 2).

    Args:
        p: First distribution.
        q: Second distribution.
        base: Logarithm base.
        eps: Numerical floor.

    Returns:
        ``0.5 D(p‖m) + 0.5 D(q‖m)`` with ``m = (p+q)/2``.
    """
    pa = _normalize(p, eps)
    qa = _normalize(q, eps)
    m = 0.5 * (pa + qa)
    return 0.5 * kl_divergence(pa, m, base, eps) + 0.5 * kl_divergence(qa, m, base, eps)


def renyi_divergence(
    p: npt.ArrayLike, q: npt.ArrayLike, alpha: float, base: float = 2.0, eps: float = 1e-12
) -> float:
    """Rényi divergence of order ``alpha``.

    Reduces to the KL divergence in the limit ``alpha → 1`` (handled explicitly).

    Args:
        p: Reference distribution.
        q: Comparison distribution.
        alpha: Order (``> 0``, ``!= 1``); ``alpha == 1`` returns the KL divergence.
        base: Logarithm base.
        eps: Numerical floor.

    Returns:
        Rényi divergence of order ``alpha``.

    Raises:
        ValueError: If ``alpha <= 0``.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-9:
        return kl_divergence(p, q, base, eps)
    pa = _normalize(p, eps)
    qa = _normalize(q, eps)
    s = float(np.sum(pa**alpha * qa ** (1.0 - alpha)))
    return (1.0 / (alpha - 1.0)) * (np.log(s) / np.log(base))
