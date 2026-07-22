"""
Domain 1 — Descriptive statistics and bootstrap resampling.

``describe`` returns a compact summary of a sample. ``bootstrap_ci`` estimates
a confidence interval for an arbitrary statistic by resampling with
replacement — the non-parametric backbone the reproducibility and methodology
domains reuse for reporting effect sizes with uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def describe(x: npt.ArrayLike) -> dict[str, float]:
    """Compute a compact descriptive summary of a 1-D sample.

    Args:
        x: Input values.

    Returns:
        Dictionary with count, mean, std (population), min, 25th/50th/75th
        percentiles, and max.

    Raises:
        ValueError: If the input is empty.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("describe requires a non-empty sample")
    q25, q50, q75 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": q25,
        "median": q50,
        "q75": q75,
        "max": float(np.max(arr)),
    }


def bootstrap_ci(
    data: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64]], float] = lambda a: float(np.mean(a)),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a sample statistic.

    Args:
        data: 1-D sample to resample from.
        statistic: Callable mapping a resample to a scalar (default: mean).
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional seed for a local, reproducible random generator.

    Returns:
        ``(point_estimate, ci_low, ci_high)`` where the point estimate is the
        statistic on the observed sample and the interval is the percentile
        bootstrap interval.

    Raises:
        ValueError: If the sample is empty or ``confidence`` is out of range.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires a non-empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    point = statistic(arr)
    resampled = np.empty(n_resamples, dtype=np.float64)
    n = arr.size
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        resampled[i] = statistic(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(resampled, 100.0 * alpha / 2.0))
    hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi
