"""
Domain 2 — Birth chart assembly and the Vimshottari dasha.

Research References:
    - Parashara, "Brihat Parashara Hora Shastra" (Vimshottari dasha system,
      Ch. 46–47; nakshatra-lord period lengths summing to 120 years)
    - Meeus, J. "Astronomical Algorithms" (2nd ed., 1998) — house/ascendant math
    - Swiss Ephemeris Technical Documentation — swe.houses_ex()

Combines the :class:`~kalachakra.astro.ephemeris.EphemerisEngine`, the ascendant
(lagna), whole-sign house assignment, and the 120-year Vimshottari mahadasha
timeline into a single :class:`BirthChart`. The dasha is derived deterministically
from the Moon's sidereal longitude: the nakshatra it occupies fixes the starting
planetary period, and the fraction of that nakshatra already traversed fixes how
much of the first period remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import swisseph as swe

from kalachakra.astro.coordinates import tropical_to_sidereal
from kalachakra.astro.ephemeris import (
    AyanamshaSystem,
    EphemerisEngine,
    PlanetaryPosition,
)
from kalachakra.astro.time import to_julian_day
from kalachakra.core.constants import (
    DASHA_SEQUENCE,
    DASHA_YEARS,
    NAKSHATRA_DASHA_LORDS,
    NAKSHATRA_SPAN,
    Ayanamsha,
    Graha,
    Nakshatra,
    Rashi,
)
from kalachakra.math.angles import normalize_degrees, sign_index

# Vimshottari convention: a "year" is 365.25 days (Parashara, BPHS Ch. 46).
_DASHA_YEAR_DAYS: float = 365.25

# Map the project's Ayanamsha enum onto the ephemeris engine's system enum.
_AYANAMSHA_MAP: dict[Ayanamsha, AyanamshaSystem] = {
    Ayanamsha.LAHIRI: AyanamshaSystem.LAHIRI,
    Ayanamsha.RAMAN: AyanamshaSystem.RAMAN,
    Ayanamsha.KRISHNAMURTI: AyanamshaSystem.KRISHNAMURTI,
    Ayanamsha.YUKTESHWAR: AyanamshaSystem.YUKTESHWAR,
    Ayanamsha.FAGAN_BRADLEY: AyanamshaSystem.FAGAN_BRADLEY,
    Ayanamsha.TRUE_CHITRAPAKSHA: AyanamshaSystem.TRUE_CITRA,
}


@dataclass(frozen=True, slots=True)
class DashaPeriod:
    """A single Vimshottari mahadasha (major planetary period)."""

    lord: Graha
    start: datetime
    end: datetime
    years: float

    def contains(self, moment: datetime) -> bool:
        """Whether ``moment`` falls within ``[start, end)``."""
        return self.start <= moment < self.end


@dataclass(frozen=True, slots=True)
class BirthChart:
    """A fully computed sidereal birth chart."""

    moment: datetime
    jd: float
    latitude: float
    longitude: float
    ayanamsha: Ayanamsha
    house_system: str
    ascendant: float
    positions: dict[Graha, PlanetaryPosition]
    houses: dict[Graha, int]
    dashas: list[DashaPeriod] = field(default_factory=list)

    @property
    def lagna_rashi(self) -> Rashi:
        """Ascendant sign."""
        return Rashi(sign_index(self.ascendant) + 1)

    @property
    def moon(self) -> PlanetaryPosition:
        """The Moon's position (drives the dasha and much of Vedic analysis)."""
        return self.positions[Graha.CHANDRA]

    def dasha_at(self, moment: datetime) -> DashaPeriod | None:
        """Return the mahadasha active at ``moment``, or ``None`` if outside range."""
        for period in self.dashas:
            if period.contains(moment):
                return period
        return None


def vimshottari_dasha(
    moon_longitude: float, birth: datetime, span_years: float = 120.0
) -> list[DashaPeriod]:
    """Compute the Vimshottari mahadasha timeline from the Moon's longitude.

    Reference: Parashara, BPHS Ch. 46–47. The 120-year cycle is partitioned
    among the nine grahas in fixed proportions; the birth nakshatra's lord opens
    the sequence, and the balance of that first period is the un-traversed
    fraction of the nakshatra times the lord's full period.

    Args:
        moon_longitude: Moon's **sidereal** ecliptic longitude in degrees.
        birth: Birth instant (timezone-aware recommended).
        span_years: How many years of periods to generate from birth.

    Returns:
        Chronological list of :class:`DashaPeriod` covering at least
        ``span_years`` from ``birth``. The first period is partial.
    """
    lon = normalize_degrees(moon_longitude)
    nak_index = int(lon // NAKSHATRA_SPAN)
    start_lord = NAKSHATRA_DASHA_LORDS[Nakshatra(nak_index + 1)]

    traversed_fraction = (lon - nak_index * NAKSHATRA_SPAN) / NAKSHATRA_SPAN
    remaining_fraction = 1.0 - traversed_fraction

    start_idx = DASHA_SEQUENCE.index(start_lord)
    periods: list[DashaPeriod] = []
    cursor = birth
    total = 0.0
    i = 0
    while total < span_years:
        lord = DASHA_SEQUENCE[(start_idx + i) % len(DASHA_SEQUENCE)]
        full_years = DASHA_YEARS[lord]
        years = full_years * remaining_fraction if i == 0 else full_years
        end = cursor + timedelta(days=years * _DASHA_YEAR_DAYS)
        periods.append(DashaPeriod(lord=lord, start=cursor, end=end, years=years))
        cursor = end
        total += years
        i += 1
    return periods


def _sidereal_ascendant(
    jd: float, latitude: float, longitude: float, ayanamsha: Ayanamsha, house_system: str
) -> tuple[float, list[float]]:
    """Return the sidereal ascendant and 12 sidereal house cusps (degrees)."""
    hsys = house_system.encode("ascii")[:1] or b"W"
    cusps, ascmc = swe.houses_ex(jd, latitude, longitude, hsys)
    asc_sidereal = tropical_to_sidereal(float(ascmc[0]), jd, ayanamsha)
    cusps_sidereal = [tropical_to_sidereal(float(c), jd, ayanamsha) for c in cusps[:12]]
    return asc_sidereal, cusps_sidereal


def _house_of(longitude: float, lagna_sign: int, cusps: list[float], whole_sign: bool) -> int:
    """Assign a longitude to a house number (1–12)."""
    if whole_sign:
        return (sign_index(longitude) - lagna_sign) % 12 + 1
    # Quadrant systems: find the cusp interval containing the longitude.
    for house in range(12):
        start = cusps[house]
        end = cusps[(house + 1) % 12]
        span = normalize_degrees(end - start)
        offset = normalize_degrees(longitude - start)
        if offset < span:
            return house + 1
    return 12


def compute_chart(
    moment: datetime,
    latitude: float,
    longitude: float,
    ayanamsha: Ayanamsha = Ayanamsha.LAHIRI,
    house_system: str = "W",
) -> BirthChart:
    """Compute a complete sidereal birth chart.

    Args:
        moment: Birth instant. Timezone-aware datetimes are strongly preferred;
            a naive datetime is treated as UTC.
        latitude: Geographic latitude in degrees (north positive).
        longitude: Geographic longitude in degrees (east positive).
        ayanamsha: Ayanamsha system (default Lahiri).
        house_system: Single-letter Swiss Ephemeris house code; ``"W"``
            (whole-sign, the Vedic default) is used unless overridden.

    Returns:
        A populated :class:`BirthChart`.
    """
    engine = EphemerisEngine(ayanamsha=_AYANAMSHA_MAP[ayanamsha], use_true_node=True)
    jd = to_julian_day(moment)
    positions = engine.get_celestial_snapshot(jd=jd).positions

    asc, cusps = _sidereal_ascendant(jd, latitude, longitude, ayanamsha, house_system)
    lagna_sign = sign_index(asc)
    whole_sign = house_system.upper() == "W"
    houses = {
        graha: _house_of(pos.sidereal_longitude, lagna_sign, cusps, whole_sign)
        for graha, pos in positions.items()
    }
    dashas = vimshottari_dasha(positions[Graha.CHANDRA].sidereal_longitude, moment)
    return BirthChart(
        moment=moment,
        jd=jd,
        latitude=latitude,
        longitude=longitude,
        ayanamsha=ayanamsha,
        house_system=house_system,
        ascendant=asc,
        positions=positions,
        houses=houses,
        dashas=dashas,
    )
