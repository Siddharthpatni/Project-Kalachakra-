"""
Domain 3 — Feature engineering.

Turns a :class:`~kalachakra.astro.charts.BirthChart` into model-ready features:
angular quantities encoded on the unit circle, planetary dignity and strength,
Ptolemaic and Vedic aspect counts, lunar-phase/tithi state, and the current
dasha. The vectorizer assembles them into a named, deterministically-ordered
feature vector (and design matrix) for the information-theory and modelling
domains.
"""

from kalachakra.features.aspects import aspect_features
from kalachakra.features.cyclical import (
    cyclical_encode,
    elongation,
    is_waxing,
    tithi,
)
from kalachakra.features.dasha import dasha_features
from kalachakra.features.dignity import (
    dignity_score,
    exaltation_proximity,
    is_debilitated,
    is_exalted,
    is_own_sign,
    sign_relationship,
)
from kalachakra.features.planetary import is_combust, planet_features
from kalachakra.features.vectorizer import (
    FeatureVector,
    chart_to_features,
    feature_matrix,
)

__all__ = [
    # cyclical
    "cyclical_encode",
    "elongation",
    "tithi",
    "is_waxing",
    # dignity
    "exaltation_proximity",
    "is_exalted",
    "is_debilitated",
    "is_own_sign",
    "sign_relationship",
    "dignity_score",
    # planetary / aspects
    "is_combust",
    "planet_features",
    "aspect_features",
    # dasha
    "dasha_features",
    # vectorizer
    "FeatureVector",
    "chart_to_features",
    "feature_matrix",
]
