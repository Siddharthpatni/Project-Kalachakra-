"""
Domain 3 — Per-graha numeric features.

Expands each planet's position into a block of model-ready numbers: its
longitude as a (sin, cos) pair, position within its sign, motion (speed and
retrograde flag), combustion (proximity to the Sun's glare), and the dignity
features from :mod:`kalachakra.features.dignity`.
"""

from __future__ import annotations

from kalachakra.astro.ephemeris import PlanetaryPosition
from kalachakra.core.constants import Graha
from kalachakra.features.cyclical import cyclical_encode
from kalachakra.features.dignity import (
    dignity_score,
    exaltation_proximity,
    is_own_sign,
)
from kalachakra.math.angles import angular_separation, sign_index

# Combustion orbs (degrees from the Sun) at which each graha is "burnt".
_COMBUSTION_ORB: dict[Graha, float] = {
    Graha.CHANDRA: 12.0,
    Graha.MANGALA: 17.0,
    Graha.BUDHA: 14.0,
    Graha.GURU: 11.0,
    Graha.SHUKRA: 10.0,
    Graha.SHANI: 15.0,
}


def is_combust(graha: Graha, longitude: float, sun_longitude: float) -> bool:
    """Whether a graha is combust (too close to the Sun).

    Args:
        graha: The planet (the Sun and nodes are never combust).
        longitude: The planet's sidereal longitude in degrees.
        sun_longitude: The Sun's sidereal longitude in degrees.

    Returns:
        True if within the graha's combustion orb of the Sun.
    """
    orb = _COMBUSTION_ORB.get(graha)
    if orb is None:
        return False
    return angular_separation(longitude, sun_longitude) <= orb


def planet_features(
    graha: Graha, position: PlanetaryPosition, sun_longitude: float, house: int
) -> dict[str, float]:
    """Compute the numeric feature block for one graha.

    Args:
        graha: The planet.
        position: Its :class:`~kalachakra.astro.ephemeris.PlanetaryPosition`.
        sun_longitude: The Sun's sidereal longitude (for combustion).
        house: The whole-sign house (1–12) the graha occupies.

    Returns:
        Dictionary of feature name to value, all prefixed by the graha name.
    """
    lon = position.sidereal_longitude
    lon_sin, lon_cos = cyclical_encode(lon)
    p = graha.name.lower()
    return {
        f"{p}_lon_sin": lon_sin,
        f"{p}_lon_cos": lon_cos,
        f"{p}_deg_in_sign": position.rashi_degree,
        f"{p}_sign": float(sign_index(lon)),
        f"{p}_house": float(house),
        f"{p}_speed": position.speed_deg_per_day,
        f"{p}_retrograde": 1.0 if position.is_retrograde else 0.0,
        f"{p}_combust": 1.0 if is_combust(graha, lon, sun_longitude) else 0.0,
        f"{p}_exalt_prox": exaltation_proximity(graha, lon),
        f"{p}_own_sign": 1.0 if is_own_sign(graha, lon) else 0.0,
        f"{p}_dignity": dignity_score(graha, lon),
    }
