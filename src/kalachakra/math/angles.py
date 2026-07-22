"""
Domain 1 — Angular arithmetic on the ecliptic circle.

Planetary longitudes are points on a 360° circle, so ordinary arithmetic
(subtraction, averaging, distance) is wrong near the 0°/360° wrap. These
helpers implement the correct modular operations plus the fixed divisions of
the zodiac (30° signs, 13°20' nakshatras) as pure longitude arithmetic. The
astro/feature layers wrap the integer indices returned here with the ``Rashi``
and ``Nakshatra`` enums from :mod:`kalachakra.core.constants`.
"""

from __future__ import annotations

import math

from kalachakra.core.constants import (
    DEGREES_PER_CIRCLE,
    DEGREES_PER_SIGN,
    NAKSHATRA_SPAN,
    PADA_SPAN,
)

_FULL: float = DEGREES_PER_CIRCLE


def normalize_degrees(angle: float) -> float:
    """Wrap an angle into the half-open interval ``[0, 360)`` degrees.

    Args:
        angle: Angle in degrees (any real value, positive or negative).

    Returns:
        Equivalent angle in ``[0, 360)``.
    """
    return angle % _FULL


def normalize_signed(angle: float) -> float:
    """Wrap an angle into the interval ``(-180, 180]`` degrees.

    Args:
        angle: Angle in degrees.

    Returns:
        Equivalent angle in ``(-180, 180]`` — useful for expressing a signed
        offset from a reference point.
    """
    wrapped = normalize_degrees(angle)
    if wrapped > _FULL / 2.0:
        wrapped -= _FULL
    return wrapped


def signed_difference(a: float, b: float) -> float:
    """Signed shortest angular difference ``a - b`` in ``(-180, 180]`` degrees.

    Positive means ``a`` is counter-clockwise (ahead) of ``b``.

    Args:
        a: First longitude in degrees.
        b: Second longitude in degrees.

    Returns:
        Signed difference in ``(-180, 180]``.
    """
    return normalize_signed(a - b)


def angular_separation(a: float, b: float) -> float:
    """Unsigned shortest angular separation between two longitudes.

    Args:
        a: First longitude in degrees.
        b: Second longitude in degrees.

    Returns:
        Separation in ``[0, 180]`` degrees.
    """
    return abs(signed_difference(a, b))


def dms_to_degrees(degrees: int, minutes: int = 0, seconds: float = 0.0) -> float:
    """Convert degrees/arc-minutes/arc-seconds to decimal degrees.

    The sign of ``degrees`` is applied to the whole quantity, so
    ``dms_to_degrees(-5, 30, 0)`` returns ``-5.5``.

    Args:
        degrees: Whole degrees (may be negative).
        minutes: Arc-minutes in ``[0, 60)``.
        seconds: Arc-seconds in ``[0, 60)``.

    Returns:
        Decimal degrees.
    """
    sign = -1.0 if degrees < 0 else 1.0
    magnitude = abs(degrees) + minutes / 60.0 + seconds / 3600.0
    return sign * magnitude


def degrees_to_dms(angle: float) -> tuple[int, int, float]:
    """Convert decimal degrees to a (degrees, minutes, seconds) triple.

    Args:
        angle: Decimal degrees (may be negative).

    Returns:
        ``(degrees, minutes, seconds)`` where ``degrees`` carries the sign and
        ``minutes``/``seconds`` are non-negative.
    """
    sign = -1 if angle < 0 else 1
    magnitude = abs(angle)
    d = int(magnitude)
    rem_minutes = (magnitude - d) * 60.0
    m = int(rem_minutes)
    s = (rem_minutes - m) * 60.0
    return sign * d, m, s


def harmonic_longitude(longitude: float, harmonic: int) -> float:
    """Map a longitude onto its N-th harmonic (varga / divisional) circle.

    The N-th harmonic multiplies angular position by ``harmonic`` and re-wraps,
    the arithmetic underlying divisional charts (e.g. Navamsha = 9th harmonic).

    Args:
        longitude: Ecliptic longitude in degrees.
        harmonic: Positive integer harmonic number.

    Returns:
        Harmonic longitude in ``[0, 360)``.

    Raises:
        ValueError: If ``harmonic`` is not a positive integer.
    """
    if harmonic < 1:
        raise ValueError(f"harmonic must be >= 1, got {harmonic}")
    return normalize_degrees(longitude * harmonic)


def sign_index(longitude: float) -> int:
    """Zero-based zodiac sign index for a longitude.

    Args:
        longitude: Ecliptic longitude in degrees.

    Returns:
        Integer in ``[0, 11]`` (0 = Mesha/Aries … 11 = Meena/Pisces). Add 1 to
        obtain the value of the corresponding ``Rashi`` enum member.
    """
    return int(normalize_degrees(longitude) // DEGREES_PER_SIGN)


def degrees_in_sign(longitude: float) -> float:
    """Position within the current sign, in ``[0, 30)`` degrees.

    Args:
        longitude: Ecliptic longitude in degrees.

    Returns:
        Offset from the start of the occupied sign.
    """
    return normalize_degrees(longitude) % DEGREES_PER_SIGN


def nakshatra_index(longitude: float) -> int:
    """Zero-based nakshatra (lunar mansion) index for a longitude.

    Args:
        longitude: Ecliptic longitude in degrees.

    Returns:
        Integer in ``[0, 26]``. Add 1 to obtain the value of the corresponding
        ``Nakshatra`` enum member.
    """
    return int(normalize_degrees(longitude) // NAKSHATRA_SPAN)


def nakshatra_pada(longitude: float) -> tuple[int, int]:
    """Nakshatra index and pada (quarter) for a longitude.

    Args:
        longitude: Ecliptic longitude in degrees.

    Returns:
        ``(nakshatra_index, pada)`` where ``nakshatra_index`` is in ``[0, 26]``
        and ``pada`` is in ``[1, 4]``.
    """
    lon = normalize_degrees(longitude)
    nak = int(lon // NAKSHATRA_SPAN)
    within = lon - nak * NAKSHATRA_SPAN
    pada = int(within // PADA_SPAN) + 1
    return nak, min(pada, 4)


def to_radians(degrees: float) -> float:
    """Degrees to radians."""
    return math.radians(degrees)


def to_degrees(radians: float) -> float:
    """Radians to degrees."""
    return math.degrees(radians)
