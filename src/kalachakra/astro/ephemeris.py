"""
Domain 2: Astronomical Engine — Ephemeris Computation

Research References:
    - Meeus, J. "Astronomical Algorithms" (2nd ed., 1998)
    - Swiss Ephemeris Technical Documentation (Astrodienst)
    - Urban & Seidelmann, "Explanatory Supplement to the Astronomical Almanac"
      (3rd ed., 2013)
    - Bretagnon & Francou, "VSOP87: Planetary solutions in elliptic variables"
      Astronomy & Astrophysics 202, 309-315 (1988)

This is the computational astronomy backbone of Project Kalachakra.
All planetary positions are computed via the Swiss Ephemeris (pyswisseph),
the industry standard used by professional astrology software worldwide.

Swiss Ephemeris is based on JPL DE431 (for planets) and provides:
    - Planetary positions to sub-arcsecond accuracy
    - Both Tropical and Sidereal (Vedic) coordinates
    - Multiple Ayanamsha systems (Lahiri, KP, Raman, etc.)
    - Lunar nodes (Rahu/Ketu) — both True and Mean
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kalachakra.core.constants import Graha, Rashi, Nakshatra
from kalachakra.core.logging import get_logger

log = get_logger(__name__)

try:
    import swisseph as swe

    _SWE_AVAILABLE = True
except ImportError:
    _SWE_AVAILABLE = False
    log.warning("pyswisseph not installed — ephemeris functions will use fallback")


# =============================================================================
# Constants
# =============================================================================


class Planet(IntEnum):
    """Swiss Ephemeris planet IDs.

    Reference: Swiss Ephemeris documentation, Section 2.1
    """

    SUN = 0
    MOON = 1
    MERCURY = 2
    VENUS = 3
    MARS = 4
    JUPITER = 5
    SATURN = 6
    URANUS = 7  # Not used in traditional Vedic, but available
    NEPTUNE = 8
    PLUTO = 9
    MEAN_NODE = 10  # Rahu (Mean Lunar Node)
    TRUE_NODE = 11  # Rahu (True/Osculating Lunar Node)
    MEAN_APOGEE = 12  # Mean Lilith


class AyanamshaSystem(IntEnum):
    """Ayanamsha systems for sidereal calculations.

    Ayanamsha = difference between Tropical and Sidereal zodiacs.

    Reference: Swiss Ephemeris documentation, Section 2.7
    """

    LAHIRI = 1           # Official Indian calendar (most common)
    RAMAN = 3            # B.V. Raman system
    KRISHNAMURTI = 5     # KP system
    FAGAN_BRADLEY = 0    # Western sidereal
    TRUE_CITRA = 27      # True Chitrapaksha
    TRUE_PUSHYA = 29     # True Pushyapaksha
    YUKTESHWAR = 7       # Sri Yukteshwar


# Map from our Graha enum to Swiss Ephemeris planet IDs
_GRAHA_TO_SWE: dict[Graha, int] = {
    Graha.SURYA: Planet.SUN,
    Graha.CHANDRA: Planet.MOON,
    Graha.MANGALA: Planet.MARS,
    Graha.BUDHA: Planet.MERCURY,
    Graha.GURU: Planet.JUPITER,
    Graha.SHUKRA: Planet.VENUS,
    Graha.SHANI: Planet.SATURN,
    Graha.RAHU: Planet.TRUE_NODE,
    Graha.KETU: -1,  # Computed as Rahu + 180°
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class PlanetaryPosition:
    """Complete positional data for a celestial body at a given moment.

    All angles in degrees (0-360).

    Attributes:
        graha: The Vedic planet identifier.
        tropical_longitude: Tropical ecliptic longitude (Western zodiac).
        sidereal_longitude: Sidereal ecliptic longitude (Vedic zodiac).
        latitude: Ecliptic latitude.
        distance_au: Distance in AU.
        speed_deg_per_day: Daily motion in degrees/day.
        rashi: Zodiac sign (Vedic).
        rashi_degree: Degree within the rashi (0-30).
        nakshatra: Lunar mansion.
        nakshatra_pada: Pada (quarter) within nakshatra (1-4).
        nakshatra_degree: Degree within nakshatra (0-13.333).
        is_retrograde: Whether the planet appears to move backward.
    """

    graha: Graha
    tropical_longitude: float
    sidereal_longitude: float
    latitude: float
    distance_au: float
    speed_deg_per_day: float
    rashi: Rashi
    rashi_degree: float
    nakshatra: Nakshatra
    nakshatra_pada: int
    nakshatra_degree: float
    is_retrograde: bool


@dataclass(frozen=True, slots=True)
class CelestialSnapshot:
    """Complete planetary state at a single moment in time.

    Contains positions for all 9 Navagrahas plus ayanamsha value.
    """

    julian_day: float
    datetime_utc: datetime
    ayanamsha: float
    ayanamsha_system: str
    positions: dict[Graha, PlanetaryPosition]


# =============================================================================
# Julian Day Conversion
# =============================================================================


def datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Day Number.

    Reference: Meeus, "Astronomical Algorithms", Ch. 7

    Algorithm (Meeus, Eq. 7.1):
        For dates after October 15, 1582 (Gregorian calendar):

        If month ≤ 2:
            Y = year - 1
            M = month + 12
        Else:
            Y = year
            M = month

        A = INT(Y / 100)
        B = 2 - A + INT(A / 4)

        JD = INT(365.25 · (Y + 4716)) + INT(30.6001 · (M + 1)) + D + B - 1524.5

    The Julian Day starts at noon UTC, so:
        JD for 2000-01-01 12:00 UT = 2451545.0 (J2000.0 epoch)

    Args:
        dt: Python datetime (preferably timezone-aware UTC).

    Returns:
        Julian Day Number (float).
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)

    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    return jd


def jd_to_datetime(jd: float) -> datetime:
    """Convert Julian Day Number back to datetime (UTC).

    Reference: Meeus, "Astronomical Algorithms", Ch. 7

    Inverse algorithm (Meeus, Eq. 7.2):
        Z = INT(JD + 0.5)
        F = (JD + 0.5) - Z

        If Z < 2299161:
            A = Z
        Else:
            α = INT((Z - 1867216.25) / 36524.25)
            A = Z + 1 + α - INT(α/4)

        B = A + 1524
        C = INT((B - 122.1) / 365.25)
        D = INT(365.25 · C)
        E = INT((B - D) / 30.6001)

        Day = B - D - INT(30.6001 · E) + F
        Month = E - 1 if E < 14 else E - 13
        Year = C - 4716 if Month > 2 else C - 4715

    Args:
        jd: Julian Day Number.

    Returns:
        datetime (UTC).
    """
    jd_plus = jd + 0.5
    Z = int(jd_plus)
    F = jd_plus - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day_frac = B - D - int(30.6001 * E) + F
    day = int(day_frac)
    frac = day_frac - day

    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    hours_total = frac * 24
    hour = int(hours_total)
    minutes_total = (hours_total - hour) * 60
    minute = int(minutes_total)
    second = int((minutes_total - minute) * 60)

    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def julian_centuries(jd: float) -> float:
    """Convert Julian Day to Julian Centuries from J2000.0.

    Reference: Meeus, Ch. 12

    Formula:
        T = (JD - 2451545.0) / 36525.0

    J2000.0 = 2000 January 1.5 TT = JD 2451545.0

    Args:
        jd: Julian Day Number.

    Returns:
        Julian centuries T from J2000.0.
    """
    return (jd - 2451545.0) / 36525.0


# =============================================================================
# Core Ephemeris Engine
# =============================================================================


class EphemerisEngine:
    """High-precision ephemeris engine using Swiss Ephemeris.

    Reference: Swiss Ephemeris documentation (astro.com/swisseph)
               Based on JPL DE431 ephemeris (for planets)
               Based on JPL DE431 + ELP/MPP02 (for Moon)

    Accuracy:
        - Planets: < 0.001 arcseconds (JPL DE431)
        - Moon: < 0.02 arcseconds
        - Date range: -13200 to +16800 (with ephemeris files)

    This engine provides:
        1. Tropical and sidereal planetary positions
        2. Multiple ayanamsha systems
        3. Retrograde detection
        4. Rashi (sign) and Nakshatra (lunar mansion) computation
    """

    def __init__(
        self,
        ayanamsha: AyanamshaSystem = AyanamshaSystem.LAHIRI,
        use_true_node: bool = True,
        ephe_path: str | None = None,
    ) -> None:
        """Initialize the ephemeris engine.

        Args:
            ayanamsha: Sidereal ayanamsha system.
            use_true_node: Use True Node (oscillating) vs Mean Node for Rahu.
            ephe_path: Path to Swiss Ephemeris data files. None = bundled.
        """
        if not _SWE_AVAILABLE:
            raise ImportError(
                "pyswisseph is required for the ephemeris engine. "
                "Install with: pip install pyswisseph"
            )

        self.ayanamsha_system = ayanamsha
        self.use_true_node = use_true_node

        if ephe_path:
            swe.set_ephe_path(ephe_path)

        swe.set_sid_mode(int(ayanamsha))
        log.info(
            f"EphemerisEngine initialized: ayanamsha={ayanamsha.name}, "
            f"true_node={use_true_node}"
        )

    def get_planetary_position(
        self,
        graha: Graha,
        jd: float,
    ) -> PlanetaryPosition:
        """Compute the position of a single planet at a given Julian Day.

        Uses swe.calc_ut() which returns:
            [0] longitude (degrees, 0-360)
            [1] latitude (degrees)
            [2] distance (AU)
            [3] speed in longitude (degrees/day)
            [4] speed in latitude
            [5] speed in distance

        For sidereal calculations, adds FLG_SIDEREAL flag.

        Args:
            graha: The Vedic planet.
            jd: Julian Day (Universal Time).

        Returns:
            PlanetaryPosition with all computed fields.
        """
        # --- Compute tropical position ---
        if graha == Graha.KETU:
            # Ketu = Rahu + 180° (always opposite the north node)
            rahu_pos = self.get_planetary_position(Graha.RAHU, jd)
            tropical_lon = (rahu_pos.tropical_longitude + 180.0) % 360.0
            sidereal_lon = (rahu_pos.sidereal_longitude + 180.0) % 360.0
            latitude = -rahu_pos.latitude
            distance = rahu_pos.distance_au
            speed = rahu_pos.speed_deg_per_day
        else:
            swe_id = _GRAHA_TO_SWE[graha]

            # Tropical position (no sidereal flag)
            trop_flags = swe.FLG_SWIEPH | swe.FLG_SPEED
            trop_result, _ = swe.calc_ut(jd, swe_id, trop_flags)
            tropical_lon = trop_result[0]
            latitude = trop_result[1]
            distance = trop_result[2]
            speed = trop_result[3]

            # Sidereal position
            sid_flags = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_SIDEREAL
            sid_result, _ = swe.calc_ut(jd, swe_id, sid_flags)
            sidereal_lon = sid_result[0]

        # --- Derive Vedic attributes ---
        rashi, rashi_degree = self._longitude_to_rashi(sidereal_lon)
        nakshatra, pada, nak_degree = self._longitude_to_nakshatra(sidereal_lon)
        is_retrograde = speed < 0

        return PlanetaryPosition(
            graha=graha,
            tropical_longitude=tropical_lon,
            sidereal_longitude=sidereal_lon,
            latitude=latitude,
            distance_au=distance,
            speed_deg_per_day=speed,
            rashi=rashi,
            rashi_degree=rashi_degree,
            nakshatra=nakshatra,
            nakshatra_pada=pada,
            nakshatra_degree=nak_degree,
            is_retrograde=is_retrograde,
        )

    def get_celestial_snapshot(
        self,
        dt: datetime | None = None,
        jd: float | None = None,
    ) -> CelestialSnapshot:
        """Compute positions for all 9 Navagrahas at a given moment.

        Args:
            dt: Python datetime. Mutually exclusive with jd.
            jd: Julian Day. Mutually exclusive with dt.

        Returns:
            CelestialSnapshot with all planetary positions.
        """
        if dt is not None:
            jd_val = datetime_to_jd(dt)
            dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        elif jd is not None:
            jd_val = jd
            dt_utc = jd_to_datetime(jd)
        else:
            raise ValueError("Either dt or jd must be provided")

        # Get ayanamsha value for this moment
        ayanamsha_val = swe.get_ayanamsa_ut(jd_val)

        # Compute all 9 planets
        positions: dict[Graha, PlanetaryPosition] = {}
        for graha in Graha:
            positions[graha] = self.get_planetary_position(graha, jd_val)

        return CelestialSnapshot(
            julian_day=jd_val,
            datetime_utc=dt_utc,
            ayanamsha=ayanamsha_val,
            ayanamsha_system=self.ayanamsha_system.name,
            positions=positions,
        )

    def get_ayanamsha(self, jd: float) -> float:
        """Get the ayanamsha value at a given Julian Day.

        Ayanamsha is the angular difference between the Tropical
        (Western) and Sidereal (Vedic) zodiacs.

        As of 2024, Lahiri ayanamsha ≈ 24.17°
        It increases by approximately 50.3" per year (precession of equinoxes).

        Formula (approximate, for Lahiri):
            Ayanamsha ≈ 24.042° + 50.27"/year × (year - 2000)

        The Swiss Ephemeris computes this precisely using the chosen system.

        Args:
            jd: Julian Day.

        Returns:
            Ayanamsha in degrees.
        """
        return swe.get_ayanamsa_ut(jd)

    def get_positions_over_range(
        self,
        graha: Graha,
        jd_start: float,
        jd_end: float,
        step_days: float = 1.0,
    ) -> list[PlanetaryPosition]:
        """Compute planetary positions over a time range.

        Useful for generating time series data for ML features.

        Args:
            graha: Planet to track.
            jd_start: Start Julian Day.
            jd_end: End Julian Day.
            step_days: Time step in days.

        Returns:
            List of PlanetaryPosition objects.
        """
        positions = []
        jd = jd_start
        while jd <= jd_end:
            positions.append(self.get_planetary_position(graha, jd))
            jd += step_days
        return positions

    # =========================================================================
    # Vedic Attribute Computation
    # =========================================================================

    @staticmethod
    def _longitude_to_rashi(
        sidereal_longitude: float,
    ) -> tuple[Rashi, float]:
        """Convert sidereal longitude to Rashi (zodiac sign).

        Each Rashi spans exactly 30° of the ecliptic:
            Rashi index = INT(longitude / 30)
            Degree within Rashi = longitude mod 30

        The 12 Rashis:
            0°-30°:   Mesha (Aries)
            30°-60°:  Vrishabha (Taurus)
            60°-90°:  Mithuna (Gemini)
            ... (each 30°)
            330°-360°: Meena (Pisces)

        Args:
            sidereal_longitude: Sidereal ecliptic longitude (0-360°).

        Returns:
            Tuple of (Rashi, degree within rashi).
        """
        lon = sidereal_longitude % 360.0
        rashi_index = int(lon / 30.0)
        rashi_degree = lon % 30.0

        rashi_list = list(Rashi)
        rashi = rashi_list[rashi_index] if rashi_index < len(rashi_list) else rashi_list[0]

        return rashi, rashi_degree

    @staticmethod
    def _longitude_to_nakshatra(
        sidereal_longitude: float,
    ) -> tuple[Nakshatra, int, float]:
        """Convert sidereal longitude to Nakshatra and Pada.

        Reference: Brihat Parashara Hora Shastra (BPHS)

        There are 27 Nakshatras, each spanning 13°20' (13.3333°):
            Nakshatra index = INT(longitude / 13.3333)
            Pada = INT((longitude mod 13.3333) / 3.3333) + 1

        Each Nakshatra has 4 Padas (quarters), each spanning 3°20' (3.3333°):
            Nakshatra span: 360° / 27 = 13°20'
            Pada span: 13°20' / 4 = 3°20'

        The 27 Nakshatras start from 0° Aries (Ashwini) and proceed
        through the entire zodiac.

        Args:
            sidereal_longitude: Sidereal ecliptic longitude (0-360°).

        Returns:
            Tuple of (Nakshatra, pada 1-4, degree within nakshatra).
        """
        lon = sidereal_longitude % 360.0
        nak_span = 360.0 / 27.0  # 13.3333°
        pada_span = nak_span / 4.0  # 3.3333°

        nak_index = int(lon / nak_span)
        nak_degree = lon % nak_span
        pada = int(nak_degree / pada_span) + 1
        pada = min(pada, 4)  # Safety clamp

        nak_list = list(Nakshatra)
        nakshatra = nak_list[nak_index] if nak_index < len(nak_list) else nak_list[0]

        return nakshatra, pada, nak_degree

    def close(self) -> None:
        """Release Swiss Ephemeris resources."""
        swe.close()


# =============================================================================
# Standalone Functions (no SwissEph dependency)
# =============================================================================


def obliquity_of_ecliptic(jd: float) -> float:
    """Mean obliquity of the ecliptic.

    Reference: Meeus, "Astronomical Algorithms", Ch. 22, Eq. 22.3

    Formula (Laskar 1986):
        ε = 23°26'21.448" - 4680.93"T - 1.55"T² + 1999.25"T³
            - 51.38"T⁴ - 249.67"T⁵ - 39.05"T⁶ + 7.12"T⁷
            + 27.87"T⁸ + 5.79"T⁹ + 2.45"T¹⁰

    where T is Julian centuries from J2000.0.

    As of 2024: ε ≈ 23.436°

    Args:
        jd: Julian Day.

    Returns:
        Mean obliquity in degrees.
    """
    T = julian_centuries(jd)
    # Coefficients in arcseconds
    eps0 = 23.0 + 26.0 / 60.0 + 21.448 / 3600.0  # 23°26'21.448"
    coeffs = [
        -4680.93, -1.55, 1999.25, -51.38, -249.67,
        -39.05, 7.12, 27.87, 5.79, 2.45,
    ]
    U = T / 100.0
    correction = sum(c * U**i for i, c in enumerate(coeffs, 1)) / 3600.0
    return eps0 + correction


def nutation(jd: float) -> tuple[float, float]:
    """Nutation in longitude (Δψ) and obliquity (Δε).

    Reference: Meeus, Ch. 22 (IAU 1980 nutation theory, simplified)

    Simplified formula using dominant term:
        Ω = 125.04452 - 1934.136261·T (longitude of Moon's ascending node)

        Δψ ≈ -17.20"·sin(Ω) - 1.32"·sin(2L₀) - 0.23"·sin(2L')
        Δε ≈ +9.20"·cos(Ω) + 0.57"·cos(2L₀) + 0.10"·cos(2L')

    where L₀ = Sun's mean longitude, L' = Moon's mean longitude.

    For full accuracy, the IAU 2000A nutation model uses 1365 terms.
    This simplified version is accurate to ~0.5".

    Args:
        jd: Julian Day.

    Returns:
        Tuple of (Δψ in degrees, Δε in degrees).
    """
    T = julian_centuries(jd)

    # Moon's longitude of ascending node
    omega = math.radians(125.04452 - 1934.136261 * T)

    # Sun's mean longitude
    L0 = math.radians(280.4664567 + 360007.6982779 * T)

    # Moon's mean longitude
    Lp = math.radians(218.3164477 + 481267.88123421 * T)

    # Nutation in longitude (arcseconds → degrees)
    delta_psi = (-17.20 * math.sin(omega) - 1.32 * math.sin(2 * L0)
                 - 0.23 * math.sin(2 * Lp)) / 3600.0

    # Nutation in obliquity
    delta_eps = (9.20 * math.cos(omega) + 0.57 * math.cos(2 * L0)
                 + 0.10 * math.cos(2 * Lp)) / 3600.0

    return delta_psi, delta_eps


def precession(jd: float) -> float:
    """General precession in longitude.

    Reference: Meeus, Ch. 21, Eq. 21.3

    Formula:
        p = 5029.0966"·T + 1.1120"·T² - 0.000006"·T³

    where T is Julian centuries from J2000.0.
    This gives the accumulated precession from J2000.0 in arcseconds.

    The precession rate ≈ 50.29" per year (1° per 71.6 years).
    This is the physical basis of the ayanamsha.

    Args:
        jd: Julian Day.

    Returns:
        Accumulated precession in degrees.
    """
    T = julian_centuries(jd)
    p_arcsec = 5029.0966 * T + 1.1120 * T**2 - 0.000006 * T**3
    return p_arcsec / 3600.0
