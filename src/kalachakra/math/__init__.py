"""
Domain 1 — Mathematical Foundations.

Pure, dependency-light numerical primitives used across the whole platform:
angular arithmetic (planetary longitudes live on a circle), circular
statistics, small-vector linear algebra for coordinate transforms,
interpolation, and reusable numeric/statistical helpers.

Nothing in this package imports from other Kalachakra domains — it is the
bottom of the dependency graph.
"""

from kalachakra.math.angles import (
    angular_separation,
    degrees_in_sign,
    degrees_to_dms,
    dms_to_degrees,
    harmonic_longitude,
    nakshatra_index,
    nakshatra_pada,
    normalize_degrees,
    normalize_signed,
    sign_index,
    signed_difference,
)
from kalachakra.math.circular import (
    circular_mean,
    circular_std,
    circular_variance,
    rayleigh_test,
    resultant_length,
)
from kalachakra.math.interpolation import angular_interpolate, linear_interpolate
from kalachakra.math.linalg import (
    angle_between_vectors,
    cartesian_to_spherical,
    normalize_vector,
    rotation_matrix,
    spherical_to_cartesian,
)
from kalachakra.math.numerics import (
    clamp,
    discretize,
    minmax_scale,
    safe_log,
    softmax,
    zscore,
)
from kalachakra.math.statistics import bootstrap_ci, describe

__all__ = [
    # angles
    "normalize_degrees",
    "normalize_signed",
    "signed_difference",
    "angular_separation",
    "dms_to_degrees",
    "degrees_to_dms",
    "harmonic_longitude",
    "sign_index",
    "degrees_in_sign",
    "nakshatra_index",
    "nakshatra_pada",
    # circular
    "circular_mean",
    "resultant_length",
    "circular_variance",
    "circular_std",
    "rayleigh_test",
    # linalg
    "rotation_matrix",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "normalize_vector",
    "angle_between_vectors",
    # interpolation
    "linear_interpolate",
    "angular_interpolate",
    # numerics
    "safe_log",
    "softmax",
    "minmax_scale",
    "zscore",
    "clamp",
    "discretize",
    # statistics
    "describe",
    "bootstrap_ci",
]
