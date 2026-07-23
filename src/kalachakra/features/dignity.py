"""
Domain 3 — Planetary dignity and strength.

Turns the qualitative dignities of classical astrology into continuous numeric
features so the information-theory gate can measure whether any of them carry
signal. The headline feature is ``exaltation_proximity``: 1.0 at a planet's
exact exaltation degree, 0.0 at its debilitation, varying smoothly between —
a continuous relaxation of the discrete exalted/debilitated categories.
"""

from __future__ import annotations

from kalachakra.core.constants import (
    DEBILITATION_DEGREES,
    EXALTATION_DEGREES,
    NATURAL_ENEMIES,
    NATURAL_FRIENDS,
    RASHI_LORDS,
    Graha,
    Rashi,
)
from kalachakra.math.angles import angular_separation, sign_index


def exaltation_proximity(graha: Graha, longitude: float) -> float:
    """Continuous exaltation strength in ``[0, 1]``.

    Args:
        graha: The planet.
        longitude: Its sidereal longitude in degrees.

    Returns:
        1.0 at the exact exaltation degree, 0.0 at debilitation (180° away),
        linear in the angular separation between.
    """
    exalt = EXALTATION_DEGREES[graha]
    return 1.0 - angular_separation(longitude, exalt) / 180.0


def is_exalted(graha: Graha, longitude: float) -> bool:
    """Whether the planet occupies its sign of exaltation."""
    return sign_index(longitude) == sign_index(EXALTATION_DEGREES[graha])


def is_debilitated(graha: Graha, longitude: float) -> bool:
    """Whether the planet occupies its sign of debilitation."""
    return sign_index(longitude) == sign_index(DEBILITATION_DEGREES[graha])


def is_own_sign(graha: Graha, longitude: float) -> bool:
    """Whether the planet rules the sign it occupies (svakshetra)."""
    rashi = Rashi(sign_index(longitude) + 1)
    return RASHI_LORDS.get(rashi) == graha


def sign_relationship(graha: Graha, longitude: float) -> int:
    """Natural relationship of the planet to the lord of its occupied sign.

    Args:
        graha: The planet.
        longitude: Its sidereal longitude in degrees.

    Returns:
        ``+1`` if the sign lord is a natural friend (or the planet itself),
        ``-1`` if a natural enemy, ``0`` if neutral.
    """
    rashi = Rashi(sign_index(longitude) + 1)
    lord = RASHI_LORDS.get(rashi)
    if lord is None or lord == graha:
        return 1
    if lord in NATURAL_FRIENDS.get(graha, []):
        return 1
    if lord in NATURAL_ENEMIES.get(graha, []):
        return -1
    return 0


def dignity_score(graha: Graha, longitude: float) -> float:
    """Composite dignity score in roughly ``[-1, 1]``.

    Blends exaltation proximity (centered on 0) with own-sign and
    friend/enemy placement into a single strength-like number.

    Args:
        graha: The planet.
        longitude: Its sidereal longitude in degrees.

    Returns:
        A continuous dignity score; higher is stronger.
    """
    base = 2.0 * exaltation_proximity(graha, longitude) - 1.0  # [-1, 1]
    own = 0.5 if is_own_sign(graha, longitude) else 0.0
    rel = 0.25 * sign_relationship(graha, longitude)
    return max(-1.5, min(1.5, base + own + rel))
