"""
Domain 2 — The ephemeris engine.

Thin, typed wrapper over Swiss Ephemeris that returns rich
:class:`PlanetPosition` records for the nine grahas. Sidereal (Vedic) positions
are the default. The two lunar nodes are handled specially: Rahu is the mean
node, and Ketu is defined as exactly 180° opposite Rahu (Swiss Ephemeris planet
id 11 is the *true* node, which is **not** how classical Vedic Ketu is defined,
so we never call the engine for Ketu directly).
"""

from __future__ import annotations

from dataclasses import dataclass

import swisseph as swe

from kalachakra.astro.coordinates import ayanamsha_degrees
from kalachakra.core.constants import Ayanamsha, Graha, Nakshatra, Rashi
from kalachakra.core.logging import get_logger
from kalachakra.math.angles import degrees_in_sign as _degrees_in_sign
from kalachakra.math.angles import nakshatra_pada as _nakshatra_pada
from kalachakra.math.angles import normalize_degrees
from kalachakra.math.angles import sign_index as _sign_index

log = get_logger(__name__)

# Grahas computed directly from a Swiss Ephemeris body id.
_SWE_BODY: dict[Graha, int] = {
    Graha.SURYA: swe.SUN,
    Graha.CHANDRA: swe.MOON,
    Graha.MANGALA: swe.MARS,
    Graha.BUDHA: swe.MERCURY,
    Graha.GURU: swe.JUPITER,
    Graha.SHUKRA: swe.VENUS,
    Graha.SHANI: swe.SATURN,
    Graha.RAHU: swe.MEAN_NODE,
}


@dataclass(frozen=True, slots=True)
class PlanetPosition:
    """Position of a single graha at one instant.

    Longitude/latitude are in the requested frame (sidereal by default). The
    derived zodiacal properties are computed lazily from ``longitude``.
    """

    graha: Graha
    longitude: float
    latitude: float
    distance_au: float
    speed_longitude: float
    sidereal: bool

    @property
    def retrograde(self) -> bool:
        """True when apparent ecliptic longitude is decreasing."""
        return self.speed_longitude < 0.0

    @property
    def sign_index(self) -> int:
        """Zero-based zodiac sign index (0 = Mesha … 11 = Meena)."""
        return _sign_index(self.longitude)

    @property
    def rashi(self) -> Rashi:
        """Occupied zodiac sign."""
        return Rashi(self.sign_index + 1)

    @property
    def degrees_in_sign(self) -> float:
        """Position within the occupied sign, in ``[0, 30)`` degrees."""
        return _degrees_in_sign(self.longitude)

    @property
    def nakshatra(self) -> Nakshatra:
        """Occupied nakshatra (lunar mansion)."""
        nak, _ = _nakshatra_pada(self.longitude)
        return Nakshatra(nak + 1)

    @property
    def pada(self) -> int:
        """Pada (quarter, 1–4) within the occupied nakshatra."""
        _, pada = _nakshatra_pada(self.longitude)
        return pada


def configure_ephemeris(path: str | None) -> None:
    """Point Swiss Ephemeris at a directory of ``.se1`` data files, if provided.

    Args:
        path: Directory containing ephemeris files, or ``None`` to rely on the
            built-in analytic (Moshier) ephemeris.
    """
    if path:
        swe.set_ephe_path(path)
        log.info(f"Swiss Ephemeris path set to {path}")


def _raw_calc(jd: float, body: int, sidereal: bool, ayanamsha: Ayanamsha) -> tuple[float, ...]:
    """Call ``calc_ut`` with a Swiss-then-Moshier fallback.

    Returns the six-element result vector (lon, lat, dist, and their speeds).
    """
    flags = swe.FLG_SPEED
    if sidereal:
        swe.set_sid_mode(
            {
                Ayanamsha.LAHIRI: swe.SIDM_LAHIRI,
                Ayanamsha.RAMAN: swe.SIDM_RAMAN,
                Ayanamsha.KRISHNAMURTI: swe.SIDM_KRISHNAMURTI,
                Ayanamsha.YUKTESHWAR: swe.SIDM_YUKTESHWAR,
                Ayanamsha.FAGAN_BRADLEY: swe.SIDM_FAGAN_BRADLEY,
                Ayanamsha.TRUE_CHITRAPAKSHA: swe.SIDM_TRUE_CITRA,
            }[ayanamsha],
            0.0,
            0.0,
        )
        flags |= swe.FLG_SIDEREAL
    try:
        values, retflag = swe.calc_ut(jd, body, flags | swe.FLG_SWIEPH)
        if retflag < 0:
            raise RuntimeError("swieph returned error flag")
    except Exception:  # noqa: BLE001 — deliberately fall back to analytic ephemeris
        values, _ = swe.calc_ut(jd, body, flags | swe.FLG_MOSEPH)
    return tuple(float(v) for v in values)


def planet_position(
    graha: Graha,
    jd: float,
    sidereal: bool = True,
    ayanamsha: Ayanamsha = Ayanamsha.LAHIRI,
) -> PlanetPosition:
    """Compute the position of one graha at a Julian Day.

    Args:
        graha: Which of the nine celestial bodies to compute.
        jd: Julian Day (UT).
        sidereal: If True (default) return sidereal (Vedic) longitude.
        ayanamsha: Ayanamsha system used when ``sidereal`` is True.

    Returns:
        A :class:`PlanetPosition`.
    """
    if graha is Graha.KETU:
        rahu = planet_position(Graha.RAHU, jd, sidereal, ayanamsha)
        return PlanetPosition(
            graha=Graha.KETU,
            longitude=normalize_degrees(rahu.longitude + 180.0),
            latitude=-rahu.latitude,
            distance_au=rahu.distance_au,
            speed_longitude=rahu.speed_longitude,
            sidereal=sidereal,
        )

    lon, lat, dist, lon_speed, _lat_speed, _dist_speed = _raw_calc(
        jd, _SWE_BODY[graha], sidereal, ayanamsha
    )
    return PlanetPosition(
        graha=graha,
        longitude=normalize_degrees(lon),
        latitude=lat,
        distance_au=dist,
        speed_longitude=lon_speed,
        sidereal=sidereal,
    )


def all_positions(
    jd: float,
    sidereal: bool = True,
    ayanamsha: Ayanamsha = Ayanamsha.LAHIRI,
) -> dict[Graha, PlanetPosition]:
    """Compute positions of all nine grahas at a Julian Day.

    Args:
        jd: Julian Day (UT).
        sidereal: If True (default) return sidereal longitudes.
        ayanamsha: Ayanamsha system used when ``sidereal`` is True.

    Returns:
        Mapping from each :class:`Graha` to its :class:`PlanetPosition`,
        ordered as in the ``Graha`` enum.
    """
    return {g: planet_position(g, jd, sidereal, ayanamsha) for g in Graha}


def ayanamsha_at(jd: float, ayanamsha: Ayanamsha = Ayanamsha.LAHIRI) -> float:
    """Convenience re-export of the ayanamsha value at an instant."""
    return ayanamsha_degrees(jd, ayanamsha)
