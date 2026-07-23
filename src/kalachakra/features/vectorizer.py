"""
Domain 3 — The feature vectorizer.

Assembles every feature block — per-graha, aspects, chart-level cyclical
quantities, and dasha state — into a single named, ordered feature vector, and
stacks vectors into a design matrix for the information-theory and modelling
domains. Feature order is deterministic (sorted names) so vectors from
different charts always align column-for-column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from kalachakra.astro.charts import BirthChart
from kalachakra.core.constants import Graha
from kalachakra.features.aspects import aspect_features
from kalachakra.features.cyclical import (
    cyclical_encode,
    elongation,
    is_waxing,
    tithi,
)
from kalachakra.features.dasha import dasha_features
from kalachakra.features.planetary import planet_features
from kalachakra.math.angles import sign_index


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A named feature vector for a single chart."""

    names: list[str]
    values: npt.NDArray[np.float64]

    def as_dict(self) -> dict[str, float]:
        """Return the vector as an ordered ``name -> value`` mapping."""
        return {n: float(v) for n, v in zip(self.names, self.values, strict=True)}


def chart_to_features(chart: BirthChart, as_of: datetime | None = None) -> FeatureVector:
    """Vectorize a single birth chart into named features.

    Args:
        chart: The birth chart to encode.
        as_of: Moment for the dasha state; defaults to the birth moment.

    Returns:
        A :class:`FeatureVector` with deterministic feature ordering.
    """
    feats: dict[str, float] = {}
    sun_lon = chart.positions[Graha.SURYA].sidereal_longitude
    moon_lon = chart.positions[Graha.CHANDRA].sidereal_longitude

    for graha, pos in chart.positions.items():
        feats.update(planet_features(graha, pos, sun_lon, chart.houses[graha]))

    feats.update(aspect_features(chart.positions))

    phase_sin, phase_cos = cyclical_encode(elongation(sun_lon, moon_lon))
    feats["moon_phase_sin"] = phase_sin
    feats["moon_phase_cos"] = phase_cos
    feats["tithi"] = float(tithi(sun_lon, moon_lon))
    feats["waxing"] = 1.0 if is_waxing(sun_lon, moon_lon) else 0.0

    lagna_sin, lagna_cos = cyclical_encode(chart.ascendant)
    feats["lagna_sin"] = lagna_sin
    feats["lagna_cos"] = lagna_cos
    feats["lagna_sign"] = float(sign_index(chart.ascendant))

    feats.update(dasha_features(chart, as_of))

    names = sorted(feats)
    values = np.array([feats[n] for n in names], dtype=np.float64)
    return FeatureVector(names=names, values=values)


def feature_matrix(
    charts: list[BirthChart], as_of: list[datetime] | None = None
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Build an ``(n_samples, n_features)`` design matrix from many charts.

    Args:
        charts: Birth charts to vectorize.
        as_of: Optional per-chart moments for the dasha state; if given, must
            match ``charts`` in length.

    Returns:
        ``(feature_names, matrix)``. All rows share the same feature order.

    Raises:
        ValueError: If ``charts`` is empty or ``as_of`` length mismatches.
    """
    if not charts:
        raise ValueError("feature_matrix requires at least one chart")
    if as_of is not None and len(as_of) != len(charts):
        raise ValueError("as_of must match charts in length")

    vectors = [
        chart_to_features(chart, None if as_of is None else as_of[i])
        for i, chart in enumerate(charts)
    ]
    names = vectors[0].names
    matrix = np.vstack([v.values for v in vectors])
    return names, matrix
