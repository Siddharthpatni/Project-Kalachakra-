"""
Domain 3 — Cyclical encoding of angular features.

An ecliptic longitude of 359° is adjacent to 1°, but a model fed the raw
degrees sees them as maximally far apart. Every angular feature is therefore
encoded as a ``(sin, cos)`` pair on the unit circle, which preserves adjacency
across the 0°/360° seam. Also derived here: the Sun–Moon elongation and the
tithi (lunar day) it defines.
"""

from __future__ import annotations

import math

from kalachakra.math.angles import normalize_degrees


def cyclical_encode(degrees: float) -> tuple[float, float]:
    """Encode an angle as a point ``(sin, cos)`` on the unit circle.

    Args:
        degrees: Angle in degrees.

    Returns:
        ``(sin, cos)`` with each component in ``[-1, 1]``.
    """
    rad = math.radians(normalize_degrees(degrees))
    return math.sin(rad), math.cos(rad)


def elongation(sun_longitude: float, moon_longitude: float) -> float:
    """Moon–Sun elongation in ``[0, 360)`` degrees (0 = new moon, 180 = full).

    Args:
        sun_longitude: Sun's ecliptic longitude in degrees.
        moon_longitude: Moon's ecliptic longitude in degrees.

    Returns:
        Elongation angle in ``[0, 360)``.
    """
    return normalize_degrees(moon_longitude - sun_longitude)


def tithi(sun_longitude: float, moon_longitude: float) -> int:
    """Tithi (lunar day) index in ``[1, 30]``.

    Each tithi spans 12° of elongation. Tithis 1–15 are the waxing (shukla)
    fortnight; 16–30 are the waning (krishna) fortnight.

    Args:
        sun_longitude: Sun's ecliptic longitude in degrees.
        moon_longitude: Moon's ecliptic longitude in degrees.

    Returns:
        Tithi number in ``[1, 30]``.
    """
    return int(elongation(sun_longitude, moon_longitude) // 12.0) + 1


def is_waxing(sun_longitude: float, moon_longitude: float) -> bool:
    """Whether the Moon is waxing (shukla paksha).

    Args:
        sun_longitude: Sun's ecliptic longitude in degrees.
        moon_longitude: Moon's ecliptic longitude in degrees.

    Returns:
        True during the waxing fortnight (elongation < 180°).
    """
    return elongation(sun_longitude, moon_longitude) < 180.0
