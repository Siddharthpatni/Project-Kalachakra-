"""
Domain 2 — Astronomical engine.

Sidereal (Vedic) planetary computation on top of Swiss Ephemeris: time
conversion, coordinate frames and the ayanamsha, the nine-graha ephemeris, and
full birth-chart assembly including the Vimshottari dasha timeline.
"""

from kalachakra.astro.charts import (
    BirthChart,
    DashaPeriod,
    compute_chart,
    vimshottari_dasha,
)
from kalachakra.astro.coordinates import (
    ayanamsha_degrees,
    ecliptic_to_equatorial,
    obliquity,
    sidereal_to_tropical,
    tropical_to_sidereal,
)
from kalachakra.astro.ephemeris import (
    PlanetPosition,
    all_positions,
    configure_ephemeris,
    planet_position,
)
from kalachakra.astro.time import (
    from_julian_day,
    julian_centuries_since_j2000,
    to_julian_day,
)

__all__ = [
    # time
    "to_julian_day",
    "from_julian_day",
    "julian_centuries_since_j2000",
    # coordinates
    "ayanamsha_degrees",
    "tropical_to_sidereal",
    "sidereal_to_tropical",
    "obliquity",
    "ecliptic_to_equatorial",
    # ephemeris
    "PlanetPosition",
    "planet_position",
    "all_positions",
    "configure_ephemeris",
    # charts
    "BirthChart",
    "DashaPeriod",
    "compute_chart",
    "vimshottari_dasha",
]
