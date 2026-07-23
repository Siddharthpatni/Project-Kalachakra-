"""
Domain 3 — Dasha-state features.

At any moment the Vimshottari timeline places the native in one mahadasha. This
module turns that state into features: which planet rules the current period
(one-hot), how far through the period we are, and the dignity of the ruling
planet in the natal chart. These are the features that make Kalachakra's
question testable — does the *timing* system carry information about *when*
events occur?
"""

from __future__ import annotations

from datetime import datetime

from kalachakra.astro.charts import BirthChart
from kalachakra.core.constants import DASHA_SEQUENCE
from kalachakra.features.dignity import dignity_score


def dasha_features(chart: BirthChart, as_of: datetime | None = None) -> dict[str, float]:
    """Compute dasha-state features for a chart at a given moment.

    Args:
        chart: The birth chart (carries the computed dasha timeline).
        as_of: Moment to evaluate the dasha at; defaults to the birth moment.
            Must share timezone-awareness with the chart's periods.

    Returns:
        Dictionary with a one-hot indicator per possible ruling planet, the
        fraction of the current period elapsed, and the ruling planet's natal
        dignity and retrograde flag.
    """
    moment = as_of if as_of is not None else chart.moment
    period = chart.dasha_at(moment)

    feats: dict[str, float] = {
        f"dasha_is_{g.name.lower()}": 0.0 for g in DASHA_SEQUENCE
    }
    if period is None:
        feats["dasha_fraction_elapsed"] = 0.0
        feats["dasha_lord_dignity"] = 0.0
        feats["dasha_lord_retrograde"] = 0.0
        return feats

    feats[f"dasha_is_{period.lord.name.lower()}"] = 1.0
    span = (period.end - period.start).total_seconds()
    elapsed = (moment - period.start).total_seconds()
    feats["dasha_fraction_elapsed"] = max(0.0, min(1.0, elapsed / span)) if span > 0 else 0.0

    lord_pos = chart.positions[period.lord]
    feats["dasha_lord_dignity"] = dignity_score(period.lord, lord_pos.longitude)
    feats["dasha_lord_retrograde"] = 1.0 if lord_pos.retrograde else 0.0
    return feats
