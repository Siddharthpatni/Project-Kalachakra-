"""
Domain 1 — Circular (directional) statistics.

Summary statistics for angular data. The arithmetic mean of longitudes is
meaningless (the mean of 350° and 10° is 180°, not 0°); these functions use the
mean-resultant-vector formulation instead. ``rayleigh_test`` tests the null
hypothesis that a set of angles is uniformly distributed around the circle —
directly relevant to asking whether planetary longitudes cluster.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def _radians(angles_deg: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.deg2rad(np.asarray(angles_deg, dtype=np.float64))


def _mean_resultant(angles_deg: npt.ArrayLike) -> tuple[float, float, int]:
    """Return (mean_angle_rad, resultant_length, n) for angular data."""
    rad = _radians(angles_deg)
    n = rad.size
    if n == 0:
        raise ValueError("circular statistics require at least one angle")
    c = float(np.mean(np.cos(rad)))
    s = float(np.mean(np.sin(rad)))
    r = math.hypot(c, s)
    mean_angle = math.atan2(s, c)
    return mean_angle, r, n


def circular_mean(angles_deg: npt.ArrayLike) -> float:
    """Mean direction of a set of angles.

    Args:
        angles_deg: Angles in degrees.

    Returns:
        Mean direction in ``[0, 360)`` degrees.
    """
    mean_angle, _, _ = _mean_resultant(angles_deg)
    return math.degrees(mean_angle) % 360.0


def resultant_length(angles_deg: npt.ArrayLike) -> float:
    """Mean resultant length ``R`` in ``[0, 1]``.

    ``R`` near 0 indicates angles spread uniformly around the circle; ``R`` near
    1 indicates tight concentration around the mean direction.

    Args:
        angles_deg: Angles in degrees.

    Returns:
        Resultant length ``R``.
    """
    _, r, _ = _mean_resultant(angles_deg)
    return r


def circular_variance(angles_deg: npt.ArrayLike) -> float:
    """Circular variance ``1 - R`` in ``[0, 1]``.

    Args:
        angles_deg: Angles in degrees.

    Returns:
        Circular variance.
    """
    return 1.0 - resultant_length(angles_deg)


def circular_std(angles_deg: npt.ArrayLike) -> float:
    """Circular standard deviation in degrees.

    Defined as ``sqrt(-2 * ln R)`` (radians), converted to degrees. Approaches
    infinity as the distribution becomes uniform (``R -> 0``).

    Args:
        angles_deg: Angles in degrees.

    Returns:
        Circular standard deviation in degrees (``inf`` if ``R == 0``).
    """
    r = resultant_length(angles_deg)
    if r <= 0.0:
        return math.inf
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def rayleigh_test(angles_deg: npt.ArrayLike) -> tuple[float, float]:
    """Rayleigh test for uniformity of a circular distribution.

    Null hypothesis: the angles are uniformly distributed around the circle.
    A small p-value is evidence of a preferred direction (clustering).

    Args:
        angles_deg: Angles in degrees.

    Returns:
        ``(z_statistic, p_value)``. Uses the standard first-order small-sample
        correction of Zar for the p-value.
    """
    _, r, n = _mean_resultant(angles_deg)
    z = n * r * r
    # Zar (1999) approximation with O(1/n) correction term.
    p = math.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n) -
                        (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n * n))
    return z, min(max(p, 0.0), 1.0)
