"""
Domain 3 — Per-graha numeric features.

Expands each planet's position into a block of model-ready numbers: its
longitude as a (sin, cos) pair, position within its sign, motion (speed and
retrograde flag), combustion (proximity to the Sun's glare), and the dignity
features from :mod:`kalachakra.features.dignity`.
"""

from __future__ import annotations

from kalachakra.astro.ephemeris import PlanetPosition
from kalachakra.core.constants import Graha
from kalachakra.features.cyclical import cyclical_encode
from kalachakra.features.dignity import (
    dignity_score,
    exaltation_proximity,
    is_own_sign,
)
from kalachakra.math.angles import angular_separation

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
    graha: Graha, position: PlanetPosition, sun_longitude: float, house: int
) -> dict[str, float]:
    """Compute the numeric feature block for one graha.

    Args:
        graha: The planet.
        position: Its :class:`PlanetPosition`.
        sun_longitude: The Sun's sidereal longitude (for combustion).
        house: The whole-sign house (1–12) the graha occupies.

    Returns:
        Dictionary of feature name to value, all prefixed by the graha name.
    """
    lon_sin, lon_cos = cyclical_encode(position.longitude)
    p = graha.name.lower()
    return {
        f"{p}_lon_sin": lon_sin,
        f"{p}_lon_cos": lon_cos,
        f"{p}_deg_in_sign": position.degrees_in_sign,
        f"{p}_sign": float(position.sign_index),
        f"{p}_house": float(house),
        f"{p}_speed": position.speed_longitude,
        f"{p}_retrograde": 1.0 if position.retrograde else 0.0,
        f"{p}_combust": 1.0 if is_combust(graha, position.longitude, sun_longitude) else 0.0,
        f"{p}_exalt_prox": exaltation_proximity(graha, position.longitude),
        f"{p}_own_sign": 1.0 if is_own_sign(graha, position.longitude) else 0.0,
        f"{p}_dignity": dignity_score(graha, position.longitude),
    }
