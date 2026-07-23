"""
Domain 2 — Time handling: civil time <-> Julian Day (UT).

Swiss Ephemeris works in Universal Time expressed as a Julian Day number.
These helpers convert timezone-aware ``datetime`` objects to and from that
representation. Naive datetimes are treated as UTC with a warning, since a
silent wrong-timezone assumption is a classic source of chart errors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import swisseph as swe

from kalachakra.core.constants import J2000
from kalachakra.core.logging import get_logger

log = get_logger(__name__)


def to_julian_day(moment: datetime) -> float:
    """Convert a ``datetime`` to a Julian Day number in Universal Time.

    Args:
        moment: Instant to convert. If timezone-aware it is converted to UTC;
            if naive it is assumed to already be UTC (a warning is logged).

    Returns:
        Julian Day (UT) suitable for ``swisseph`` ``*_ut`` functions.
    """
    if moment.tzinfo is None:
        log.warning("Naive datetime passed to to_julian_day; assuming UTC")
        utc = moment.replace(tzinfo=timezone.utc)
    else:
        utc = moment.astimezone(timezone.utc)
    decimal_hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0 + utc.microsecond / 3.6e9
    return swe.julday(utc.year, utc.month, utc.day, decimal_hour, swe.GREG_CAL)


def from_julian_day(jd: float) -> datetime:
    """Convert a Julian Day number (UT) back to a UTC ``datetime``.

    Args:
        jd: Julian Day in Universal Time.

    Returns:
        Timezone-aware ``datetime`` in UTC.
    """
    year, month, day, hour_frac = swe.revjul(jd, swe.GREG_CAL)
    whole_hours = int(hour_frac)
    remainder_seconds = (hour_frac - whole_hours) * 3600.0
    minutes = int(remainder_seconds // 60)
    seconds = remainder_seconds - minutes * 60
    whole_seconds = int(seconds)
    microseconds = int(round((seconds - whole_seconds) * 1e6))
    base = datetime(year, month, day, tzinfo=timezone.utc)
    return base + timedelta(
        hours=whole_hours, minutes=minutes, seconds=whole_seconds, microseconds=microseconds
    )


def julian_centuries_since_j2000(jd: float) -> float:
    """Julian centuries elapsed from the J2000.0 epoch.

    Args:
        jd: Julian Day (UT).

    Returns:
        ``(jd - J2000) / 36525`` — the time argument used by many
        astronomical polynomial series.
    """
    return (jd - J2000) / 36525.0
