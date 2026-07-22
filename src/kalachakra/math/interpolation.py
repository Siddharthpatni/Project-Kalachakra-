"""
Domain 1 — Interpolation helpers.

Ephemeris values are typically sampled on a grid and interpolated to arbitrary
instants. ``linear_interpolate`` is a thin, validated wrapper over
``numpy.interp``; ``angular_interpolate`` interpolates along the shortest arc so
that interpolating from 350° to 10° passes through 0°, not 180°.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kalachakra.math.angles import normalize_degrees, signed_difference


def linear_interpolate(
    x: npt.ArrayLike, xp: npt.ArrayLike, fp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Piecewise-linear interpolation of ``fp`` sampled at ``xp``, evaluated at ``x``.

    Args:
        x: Query point(s).
        xp: Sample abscissae — must be strictly increasing.
        fp: Sample ordinates, same length as ``xp``.

    Returns:
        Interpolated values with the same shape as ``x``.

    Raises:
        ValueError: If ``xp`` and ``fp`` differ in length or ``xp`` is not
            strictly increasing.
    """
    xp_arr = np.asarray(xp, dtype=np.float64)
    fp_arr = np.asarray(fp, dtype=np.float64)
    if xp_arr.shape != fp_arr.shape:
        raise ValueError("xp and fp must have the same shape")
    if xp_arr.size < 2:
        raise ValueError("need at least two sample points to interpolate")
    if not np.all(np.diff(xp_arr) > 0):
        raise ValueError("xp must be strictly increasing")
    return np.interp(np.asarray(x, dtype=np.float64), xp_arr, fp_arr)


def angular_interpolate(a: float, b: float, t: float) -> float:
    """Interpolate between two angles along the shortest arc.

    Args:
        a: Start angle in degrees (``t == 0``).
        b: End angle in degrees (``t == 1``).
        t: Interpolation fraction, typically in ``[0, 1]``.

    Returns:
        Interpolated angle in ``[0, 360)`` degrees.
    """
    delta = signed_difference(b, a)
    return normalize_degrees(a + t * delta)
