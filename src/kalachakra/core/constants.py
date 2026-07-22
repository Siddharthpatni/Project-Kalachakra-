"""
Project Kalachakra — Vedic Astronomical Constants

All constants are defined as mathematically precise enumerations.
No hard-coded beliefs — purely representational.
"""

from enum import IntEnum, StrEnum
from typing import Final

# =============================================================================
# Navagrahas (Nine Celestial Bodies)
# =============================================================================


class Graha(IntEnum):
    """The nine Vedic celestial bodies (Navagrahas).

    Swiss Ephemeris planet IDs for the seven physical bodies.
    Rahu/Ketu are computed as lunar node positions.
    """

    SURYA = 0       # Sun
    CHANDRA = 1     # Moon
    MANGALA = 4     # Mars
    BUDHA = 2       # Mercury
    GURU = 5        # Jupiter
    SHUKRA = 3      # Venus
    SHANI = 6       # Saturn
    RAHU = 10       # North Lunar Node (mean)
    KETU = 11       # South Lunar Node (mean, computed as Rahu + 180°)


NAVAGRAHAS: Final[list[str]] = [g.name.capitalize() for g in Graha]

# Sanskrit names mapping
GRAHA_NAMES: Final[dict[Graha, str]] = {
    Graha.SURYA: "सूर्य",
    Graha.CHANDRA: "चन्द्र",
    Graha.MANGALA: "मङ्गल",
    Graha.BUDHA: "बुध",
    Graha.GURU: "गुरु",
    Graha.SHUKRA: "शुक्र",
    Graha.SHANI: "शनि",
    Graha.RAHU: "राहु",
    Graha.KETU: "केतु",
}

# =============================================================================
# Rashis (Zodiac Signs — 12 divisions of 30° each)
# =============================================================================


class Rashi(IntEnum):
    """The 12 Rashis (zodiac signs), each spanning 30° of the ecliptic."""

    MESHA = 1       # Aries
    VRISHABHA = 2   # Taurus
    MITHUNA = 3     # Gemini
    KARKA = 4       # Cancer
    SIMHA = 5       # Leo
    KANYA = 6       # Virgo
    TULA = 7        # Libra
    VRISHCHIKA = 8  # Scorpio
    DHANU = 9       # Sagittarius
    MAKARA = 10     # Capricorn
    KUMBHA = 11     # Aquarius
    MEENA = 12      # Pisces


RASHIS: Final[list[str]] = [r.name.capitalize() for r in Rashi]

# Rashi lords (planetary rulership)
RASHI_LORDS: Final[dict[Rashi, Graha]] = {
    Rashi.MESHA: Graha.MANGALA,
    Rashi.VRISHABHA: Graha.SHUKRA,
    Rashi.MITHUNA: Graha.BUDHA,
    Rashi.KARKA: Graha.CHANDRA,
    Rashi.SIMHA: Graha.SURYA,
    Rashi.KANYA: Graha.BUDHA,
    Rashi.TULA: Graha.SHUKRA,
    Rashi.VRISHCHIKA: Graha.MANGALA,
    Rashi.DHANU: Graha.GURU,
    Rashi.MAKARA: Graha.SHANI,
    Rashi.KUMBHA: Graha.SHANI,
    Rashi.MEENA: Graha.GURU,
}

# Element (Tattva) classification
RASHI_ELEMENTS: Final[dict[str, list[Rashi]]] = {
    "fire": [Rashi.MESHA, Rashi.SIMHA, Rashi.DHANU],
    "earth": [Rashi.VRISHABHA, Rashi.KANYA, Rashi.MAKARA],
    "air": [Rashi.MITHUNA, Rashi.TULA, Rashi.KUMBHA],
    "water": [Rashi.KARKA, Rashi.VRISHCHIKA, Rashi.MEENA],
}

# Modality (Guna)
RASHI_MODALITIES: Final[dict[str, list[Rashi]]] = {
    "cardinal": [Rashi.MESHA, Rashi.KARKA, Rashi.TULA, Rashi.MAKARA],
    "fixed": [Rashi.VRISHABHA, Rashi.SIMHA, Rashi.VRISHCHIKA, Rashi.KUMBHA],
    "mutable": [Rashi.MITHUNA, Rashi.KANYA, Rashi.DHANU, Rashi.MEENA],
}

# =============================================================================
# Nakshatras (27 Lunar Mansions — each spanning 13°20')
# =============================================================================


class Nakshatra(IntEnum):
    """The 27 Nakshatras (lunar mansions), each spanning 13°20' of the ecliptic."""

    ASHWINI = 1
    BHARANI = 2
    KRITTIKA = 3
    ROHINI = 4
    MRIGASHIRA = 5
    ARDRA = 6
    PUNARVASU = 7
    PUSHYA = 8
    ASHLESHA = 9
    MAGHA = 10
    PURVA_PHALGUNI = 11
    UTTARA_PHALGUNI = 12
    HASTA = 13
    CHITRA = 14
    SWATI = 15
    VISHAKHA = 16
    ANURADHA = 17
    JYESHTHA = 18
    MOOLA = 19
    PURVA_ASHADHA = 20
    UTTARA_ASHADHA = 21
    SHRAVANA = 22
    DHANISHTHA = 23
    SHATABHISHA = 24
    PURVA_BHADRAPADA = 25
    UTTARA_BHADRAPADA = 26
    REVATI = 27


NAKSHATRAS: Final[list[str]] = [n.name.replace("_", " ").title() for n in Nakshatra]

# Nakshatra lords for Vimshottari Dasha
NAKSHATRA_DASHA_LORDS: Final[dict[Nakshatra, Graha]] = {
    Nakshatra.ASHWINI: Graha.KETU,
    Nakshatra.BHARANI: Graha.SHUKRA,
    Nakshatra.KRITTIKA: Graha.SURYA,
    Nakshatra.ROHINI: Graha.CHANDRA,
    Nakshatra.MRIGASHIRA: Graha.MANGALA,
    Nakshatra.ARDRA: Graha.RAHU,
    Nakshatra.PUNARVASU: Graha.GURU,
    Nakshatra.PUSHYA: Graha.SHANI,
    Nakshatra.ASHLESHA: Graha.BUDHA,
    Nakshatra.MAGHA: Graha.KETU,
    Nakshatra.PURVA_PHALGUNI: Graha.SHUKRA,
    Nakshatra.UTTARA_PHALGUNI: Graha.SURYA,
    Nakshatra.HASTA: Graha.CHANDRA,
    Nakshatra.CHITRA: Graha.MANGALA,
    Nakshatra.SWATI: Graha.RAHU,
    Nakshatra.VISHAKHA: Graha.GURU,
    Nakshatra.ANURADHA: Graha.SHANI,
    Nakshatra.JYESHTHA: Graha.BUDHA,
    Nakshatra.MOOLA: Graha.KETU,
    Nakshatra.PURVA_ASHADHA: Graha.SHUKRA,
    Nakshatra.UTTARA_ASHADHA: Graha.SURYA,
    Nakshatra.SHRAVANA: Graha.CHANDRA,
    Nakshatra.DHANISHTHA: Graha.MANGALA,
    Nakshatra.SHATABHISHA: Graha.RAHU,
    Nakshatra.PURVA_BHADRAPADA: Graha.GURU,
    Nakshatra.UTTARA_BHADRAPADA: Graha.SHANI,
    Nakshatra.REVATI: Graha.BUDHA,
}

# Nakshatra span in degrees
NAKSHATRA_SPAN: Final[float] = 13.333333  # 13°20' = 360°/27

# Pada span in degrees
PADA_SPAN: Final[float] = 3.333333  # 3°20' = 13°20'/4

# =============================================================================
# Bhavas (Houses — 12 divisions)
# =============================================================================

BHAVAS: Final[list[str]] = [
    "Tanu",         # 1 — Self, body, personality
    "Dhana",        # 2 — Wealth, family, speech
    "Sahaja",       # 3 — Siblings, courage, communication
    "Sukha",        # 4 — Home, mother, comfort
    "Putra",        # 5 — Children, intelligence, creativity
    "Ari",          # 6 — Enemies, disease, service
    "Yuvati",       # 7 — Spouse, partnerships, trade
    "Randhra",      # 8 — Death, transformation, occult
    "Dharma",       # 9 — Fortune, father, higher learning
    "Karma",        # 10 — Career, fame, authority
    "Labha",        # 11 — Gains, income, elder siblings
    "Vyaya",        # 12 — Losses, liberation, foreign lands
]

# =============================================================================
# Vimshottari Dasha Periods (in years)
# =============================================================================

DASHA_YEARS: Final[dict[Graha, float]] = {
    Graha.SURYA: 6.0,
    Graha.CHANDRA: 10.0,
    Graha.MANGALA: 7.0,
    Graha.RAHU: 18.0,
    Graha.GURU: 16.0,
    Graha.SHANI: 19.0,
    Graha.BUDHA: 17.0,
    Graha.KETU: 7.0,
    Graha.SHUKRA: 20.0,
}

# Total Vimshottari cycle
DASHA_TOTAL_YEARS: Final[float] = 120.0

# Dasha sequence (standard order)
DASHA_SEQUENCE: Final[list[Graha]] = [
    Graha.SURYA, Graha.CHANDRA, Graha.MANGALA, Graha.RAHU,
    Graha.GURU, Graha.SHANI, Graha.BUDHA, Graha.KETU, Graha.SHUKRA,
]

# =============================================================================
# Planetary Relationships (Natural)
# =============================================================================

NATURAL_FRIENDS: Final[dict[Graha, list[Graha]]] = {
    Graha.SURYA: [Graha.CHANDRA, Graha.MANGALA, Graha.GURU],
    Graha.CHANDRA: [Graha.SURYA, Graha.BUDHA],
    Graha.MANGALA: [Graha.SURYA, Graha.CHANDRA, Graha.GURU],
    Graha.BUDHA: [Graha.SURYA, Graha.SHUKRA],
    Graha.GURU: [Graha.SURYA, Graha.CHANDRA, Graha.MANGALA],
    Graha.SHUKRA: [Graha.BUDHA, Graha.SHANI],
    Graha.SHANI: [Graha.BUDHA, Graha.SHUKRA],
    Graha.RAHU: [Graha.SHUKRA, Graha.SHANI],
    Graha.KETU: [Graha.MANGALA, Graha.SHUKRA, Graha.SHANI],
}

NATURAL_ENEMIES: Final[dict[Graha, list[Graha]]] = {
    Graha.SURYA: [Graha.SHANI, Graha.SHUKRA],
    Graha.CHANDRA: [],  # No natural enemies
    Graha.MANGALA: [Graha.BUDHA],
    Graha.BUDHA: [Graha.CHANDRA],
    Graha.GURU: [Graha.BUDHA, Graha.SHUKRA],
    Graha.SHUKRA: [Graha.SURYA, Graha.CHANDRA],
    Graha.SHANI: [Graha.SURYA, Graha.CHANDRA, Graha.MANGALA],
    Graha.RAHU: [Graha.SURYA, Graha.CHANDRA, Graha.MANGALA],
    Graha.KETU: [Graha.SURYA, Graha.CHANDRA],
}

# =============================================================================
# Exaltation / Debilitation Degrees
# =============================================================================

EXALTATION_DEGREES: Final[dict[Graha, float]] = {
    Graha.SURYA: 10.0,       # 10° Aries
    Graha.CHANDRA: 33.0,     # 3° Taurus
    Graha.MANGALA: 298.0,    # 28° Capricorn
    Graha.BUDHA: 165.0,      # 15° Virgo
    Graha.GURU: 95.0,        # 5° Cancer
    Graha.SHUKRA: 357.0,     # 27° Pisces
    Graha.SHANI: 200.0,      # 20° Libra
    Graha.RAHU: 50.0,        # 20° Taurus
    Graha.KETU: 230.0,       # 20° Scorpio
}

# Debilitation is exactly 180° from exaltation
DEBILITATION_DEGREES: Final[dict[Graha, float]] = {
    graha: (deg + 180.0) % 360.0
    for graha, deg in EXALTATION_DEGREES.items()
}

# =============================================================================
# Aspects (Drishti — degrees of planetary gaze)
# =============================================================================

# Standard aspects (all planets aspect 7th from self = 180°)
STANDARD_ASPECT: Final[float] = 180.0

# Special aspects
SPECIAL_ASPECTS: Final[dict[Graha, list[int]]] = {
    Graha.MANGALA: [4, 7, 8],   # Mars: 4th, 7th, 8th house aspects
    Graha.GURU: [5, 7, 9],      # Jupiter: 5th, 7th, 9th house aspects
    Graha.SHANI: [3, 7, 10],    # Saturn: 3rd, 7th, 10th house aspects
    Graha.RAHU: [5, 7, 9],      # Rahu: same as Jupiter
    Graha.KETU: [5, 7, 9],      # Ketu: same as Jupiter
}

# =============================================================================
# Ayanamsha Types
# =============================================================================


class Ayanamsha(StrEnum):
    """Ayanamsha systems for sidereal coordinate conversion."""

    LAHIRI = "lahiri"
    RAMAN = "raman"
    KRISHNAMURTI = "krishnamurti"
    YUKTESHWAR = "yukteshwar"
    FAGAN_BRADLEY = "fagan_bradley"
    TRUE_CHITRAPAKSHA = "true_chitrapaksha"


# =============================================================================
# House Systems
# =============================================================================

HOUSE_SYSTEMS: Final[dict[str, str]] = {
    "P": "Placidus",
    "K": "Koch",
    "B": "Alcabitius",
    "E": "Equal",
    "W": "Whole Sign",
    "R": "Regiomontanus",
    "C": "Campanus",
    "O": "Porphyry",
    "M": "Morinus",
}

# =============================================================================
# Time Constants
# =============================================================================

WEEKDAYS: Final[list[str]] = [
    "Ravivara",   # Sunday (Sun)
    "Somavara",   # Monday (Moon)
    "Mangalavara",  # Tuesday (Mars)
    "Budhavara",  # Wednesday (Mercury)
    "Guruvara",   # Thursday (Jupiter)
    "Shukravara", # Friday (Venus)
    "Shanivara",  # Saturday (Saturn)
]

# Julian Day epoch
J2000: Final[float] = 2451545.0  # Jan 1, 2000 12:00 TT

# Degrees per sign
DEGREES_PER_SIGN: Final[float] = 30.0
DEGREES_PER_CIRCLE: Final[float] = 360.0
