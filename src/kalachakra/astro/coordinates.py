"""
Domain 2 — Coordinate frames and the ayanamsha.

Vedic (sidereal) longitudes are the tropical longitudes returned by the
ephemeris shifted by the *ayanamsha* — the accumulated precession offset
between the two zodiacs. This module maps our :class:`Ayanamsha` enum onto the
Swiss Ephemeris sidereal modes and provides tropical/sidereal conversion plus
ecliptic<->equatorial rotation by the obliquity of the ecliptic.
"""

from __future__ import annotations

import swisseph as swe

from kalachakra.core.constants import Ayanamsha
from kalachakra.math.angles import normalize_degrees
from kalachakra.math.linalg import (
    cartesian_to_spherical,
    rotation_matrix,
    spherical_to_cartesian,
)

# Map our ayanamsha enum to Swiss Ephemeris sidereal-mode integer constants.
_SWE_SIDEREAL_MODE: dict[Ayanamsha, int] = {
    Ayanamsha.LAHIRI: swe.SIDM_LAHIRI,
    Ayanamsha.RAMAN: swe.SIDM_RAMAN,
    Ayanamsha.KRISHNAMURTI: swe.SIDM_KRISHNAMURTI,
    Ayanamsha.YUKTESHWAR: swe.SIDM_YUKTESHWAR,
    Ayanamsha.FAGAN_BRADLEY: swe.SIDM_FAGAN_BRADLEY,
    Ayanamsha.TRUE_CHITRAPAKSHA: swe.SIDM_TRUE_CITRA,
}


def swe_sidereal_mode(ayanamsha: Ayanamsha) -> int:
    """Return the Swiss Ephemeris sidereal-mode constant for an ayanamsha.

    Args:
        ayanamsha: Ayanamsha system.

    Returns:
        The corresponding ``swe.SIDM_*`` integer.
    """
    return _SWE_SIDEREAL_MODE[ayanamsha]


def ayanamsha_degrees(jd: float, ayanamsha: Ayanamsha = Ayanamsha.LAHIRI) -> float:
    """Ayanamsha value (degrees) at a given instant.

    Args:
        jd: Julian Day (UT).
        ayanamsha: Ayanamsha system.

    Returns:
        Ayanamsha offset in degrees.
    """
    swe.set_sid_mode(_SWE_SIDEREAL_MODE[ayanamsha], 0.0, 0.0)
    return float(swe.get_ayanamsa_ut(jd))


def tropical_to_sidereal(
    longitude: float, jd: float, ayanamsha: Ayanamsha = Ayanamsha.LAHIRI
) -> float:
    """Convert a tropical longitude to sidereal by subtracting the ayanamsha.

    Args:
        longitude: Tropical ecliptic longitude in degrees.
        jd: Julian Day (UT).
        ayanamsha: Ayanamsha system.

    Returns:
        Sidereal longitude in ``[0, 360)`` degrees.
    """
    return normalize_degrees(longitude - ayanamsha_degrees(jd, ayanamsha))


def sidereal_to_tropical(
    longitude: float, jd: float, ayanamsha: Ayanamsha = Ayanamsha.LAHIRI
) -> float:
    """Convert a sidereal longitude to tropical by adding the ayanamsha.

    Args:
        longitude: Sidereal ecliptic longitude in degrees.
        jd: Julian Day (UT).
        ayanamsha: Ayanamsha system.

    Returns:
        Tropical longitude in ``[0, 360)`` degrees.
    """
    return normalize_degrees(longitude + ayanamsha_degrees(jd, ayanamsha))


def obliquity(jd: float) -> float:
    """True obliquity of the ecliptic (degrees) at a given instant.

    Args:
        jd: Julian Day (UT).

    Returns:
        Obliquity in degrees (angle between ecliptic and equatorial planes).
    """
    values, _ = swe.calc_ut(jd, swe.ECL_NUT, swe.FLG_SWIEPH)
    return float(values[0])


def ecliptic_to_equatorial(
    longitude: float, latitude: float, jd: float
) -> tuple[float, float]:
    """Rotate ecliptic (longitude, latitude) into equatorial (RA, Dec).

    Args:
        longitude: Ecliptic longitude in degrees.
        latitude: Ecliptic latitude in degrees.
        jd: Julian Day (UT), used to obtain the obliquity.

    Returns:
        ``(right_ascension, declination)`` in degrees, RA in ``[0, 360)``.
    """
    eps = obliquity(jd)
    vec = spherical_to_cartesian(longitude, latitude, 1.0)
    rotated = rotation_matrix("x", eps) @ vec
    ra, dec, _ = cartesian_to_spherical(rotated)
    return ra, dec
