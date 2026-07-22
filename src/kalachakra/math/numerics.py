"""
Domain 1 — General numeric helpers.

Numerically-careful primitives reused across modelling and information theory:
a floored logarithm, a stable softmax, feature scaling, and discretization of
continuous variables into bins (uniform width or equal-frequency quantiles),
which the mutual-information estimators in Domain 25 depend on.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def safe_log(x: npt.ArrayLike, base: float = np.e, eps: float = 1e-12) -> npt.NDArray[np.float64]:
    """Elementwise logarithm with values floored at ``eps`` to avoid ``log(0)``.

    Args:
        x: Input values (non-negative in normal use).
        base: Logarithm base (default natural log). Base 2 gives bits.
        eps: Lower bound applied to inputs before taking the log.

    Returns:
        ``log_base(max(x, eps))``.
    """
    arr = np.clip(np.asarray(x, dtype=np.float64), eps, None)
    if base == np.e:
        return np.log(arr)
    return np.log(arr) / np.log(base)


def softmax(x: npt.ArrayLike, axis: int = -1) -> npt.NDArray[np.float64]:
    """Numerically stable softmax.

    Args:
        x: Input array of logits.
        axis: Axis along which to normalize.

    Returns:
        Array of the same shape whose entries along ``axis`` are non-negative
        and sum to 1.
    """
    arr = np.asarray(x, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar to ``[low, high]``.

    Args:
        value: Value to clamp.
        low: Lower bound.
        high: Upper bound.

    Returns:
        ``value`` restricted to ``[low, high]``.

    Raises:
        ValueError: If ``low > high``.
    """
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    return max(low, min(value, high))


def minmax_scale(
    x: npt.ArrayLike, feature_range: tuple[float, float] = (0.0, 1.0)
) -> npt.NDArray[np.float64]:
    """Scale values to ``feature_range`` by their min and max.

    A constant input maps to the low end of the range.

    Args:
        x: Input values.
        feature_range: Target ``(low, high)`` interval.

    Returns:
        Scaled values.
    """
    arr = np.asarray(x, dtype=np.float64)
    lo, hi = feature_range
    xmin, xmax = float(np.min(arr)), float(np.max(arr))
    span = xmax - xmin
    if span == 0.0:
        return np.full_like(arr, lo)
    return lo + (arr - xmin) * (hi - lo) / span


def zscore(x: npt.ArrayLike, ddof: int = 0) -> npt.NDArray[np.float64]:
    """Standardize values to zero mean and unit variance.

    A constant input maps to all zeros (rather than dividing by zero).

    Args:
        x: Input values.
        ddof: Delta degrees of freedom for the standard deviation.

    Returns:
        Standardized values.
    """
    arr = np.asarray(x, dtype=np.float64)
    std = float(np.std(arr, ddof=ddof))
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def discretize(
    x: npt.ArrayLike, bins: int = 10, strategy: str = "uniform"
) -> npt.NDArray[np.int64]:
    """Bin a continuous variable into integer labels ``0 .. bins-1``.

    Args:
        x: Continuous input values.
        bins: Number of bins (must be >= 1).
        strategy: ``"uniform"`` for equal-width bins over the data range, or
            ``"quantile"`` for equal-frequency bins.

    Returns:
        Integer bin index per input value.

    Raises:
        ValueError: If ``bins < 1`` or ``strategy`` is unknown.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    arr = np.asarray(x, dtype=np.float64)
    if bins == 1:
        return np.zeros(arr.shape, dtype=np.int64)
    if strategy == "uniform":
        edges = np.linspace(float(np.min(arr)), float(np.max(arr)), bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
    # Deduplicate degenerate edges (constant/heavily-tied data) and clip to range.
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(arr.shape, dtype=np.int64)
    labels = np.digitize(arr, edges[1:-1], right=False)
    return np.clip(labels, 0, edges.size - 2).astype(np.int64)
