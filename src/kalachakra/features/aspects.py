"""
Domain 3 — Aspect features.

Two aspect systems are encoded as counts. The Ptolemaic system (conjunction,
sextile, square, trine, opposition, each within an orb) treats aspects as
symmetric angular relationships. The Vedic drishti system is asymmetric and
sign-distance based: every graha aspects the 7th sign from itself, and Mars,
Jupiter, Saturn and the nodes cast additional special aspects.
"""

from __future__ import annotations

from itertools import combinations

from kalachakra.astro.ephemeris import PlanetaryPosition
from kalachakra.core.constants import SPECIAL_ASPECTS, Graha
from kalachakra.math.angles import angular_separation, sign_index

# Ptolemaic aspect angles and their orbs (degrees).
_PTOLEMAIC: dict[str, tuple[float, float]] = {
    "conjunction": (0.0, 8.0),
    "sextile": (60.0, 4.0),
    "square": (90.0, 6.0),
    "trine": (120.0, 6.0),
    "opposition": (180.0, 8.0),
}


def _casts_drishti(from_graha: Graha, from_sign: int, to_sign: int) -> bool:
    """Whether ``from_graha`` casts a Vedic aspect onto the target sign."""
    house_distance = (to_sign - from_sign) % 12 + 1  # 1..12
    if house_distance == 7:  # every graha aspects the 7th
        return True
    return house_distance in SPECIAL_ASPECTS.get(from_graha, [])


def aspect_features(positions: dict[Graha, PlanetaryPosition]) -> dict[str, float]:
    """Compute aspect-count features for a set of graha positions.

    Args:
        positions: Mapping from graha to its :class:`PlanetaryPosition`.

    Returns:
        Dictionary of feature name to count, covering Ptolemaic aspect counts,
        the total Ptolemaic aspects, and the total Vedic drishti relationships.
    """
    counts: dict[str, float] = {f"aspect_{name}": 0.0 for name in _PTOLEMAIC}
    grahas = list(positions.keys())

    for a, b in combinations(grahas, 2):
        sep = angular_separation(
            positions[a].sidereal_longitude, positions[b].sidereal_longitude
        )
        for name, (angle, orb) in _PTOLEMAIC.items():
            if abs(sep - angle) <= orb:
                counts[f"aspect_{name}"] += 1.0
    counts["aspect_total"] = sum(counts.values())

    drishti = 0
    for a in grahas:
        sa = sign_index(positions[a].sidereal_longitude)
        for b in grahas:
            if a == b:
                continue
            sb = sign_index(positions[b].sidereal_longitude)
            if _casts_drishti(a, sa, sb):
                drishti += 1
    counts["drishti_total"] = float(drishti)
    return counts
