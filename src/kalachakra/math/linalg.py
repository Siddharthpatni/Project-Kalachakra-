"""
Domain 1 — Small-vector linear algebra for coordinate transforms.

Rotation matrices and spherical/Cartesian conversions used by the astronomy
engine to move between ecliptic and equatorial frames (a rotation about the
x-axis by the obliquity of the ecliptic) and to compute 3-D angular
separations. Everything operates on plain NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vector3 = npt.NDArray[np.float64]
Matrix3 = npt.NDArray[np.float64]


def rotation_matrix(axis: str, angle_deg: float) -> Matrix3:
    """Right-handed 3-D rotation matrix about a principal axis.

    Args:
        axis: One of ``"x"``, ``"y"``, ``"z"`` (case-insensitive).
        angle_deg: Rotation angle in degrees.

    Returns:
        A ``3x3`` rotation matrix.

    Raises:
        ValueError: If ``axis`` is not one of x/y/z.
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    key = axis.lower()
    if key == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if key == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if key == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def spherical_to_cartesian(
    longitude_deg: float, latitude_deg: float, radius: float = 1.0
) -> Vector3:
    """Convert spherical (longitude, latitude, radius) to Cartesian x/y/z.

    Args:
        longitude_deg: Longitude in degrees (angle in the x-y plane from +x).
        latitude_deg: Latitude in degrees (angle above the x-y plane).
        radius: Radial distance (default unit sphere).

    Returns:
        Length-3 array ``[x, y, z]``.
    """
    lon = np.deg2rad(longitude_deg)
    lat = np.deg2rad(latitude_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def cartesian_to_spherical(vec: npt.ArrayLike) -> tuple[float, float, float]:
    """Convert Cartesian x/y/z to spherical (longitude, latitude, radius).

    Args:
        vec: Length-3 sequence ``[x, y, z]``.

    Returns:
        ``(longitude_deg, latitude_deg, radius)`` with longitude in ``[0, 360)``
        and latitude in ``[-90, 90]``.
    """
    v = np.asarray(vec, dtype=np.float64)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    radius = float(np.linalg.norm(v))
    if radius == 0.0:
        return 0.0, 0.0, 0.0
    longitude = float(np.rad2deg(np.arctan2(y, x))) % 360.0
    latitude = float(np.rad2deg(np.arcsin(z / radius)))
    return longitude, latitude, radius


def normalize_vector(vec: npt.ArrayLike) -> Vector3:
    """Return the unit vector in the direction of ``vec``.

    Args:
        vec: Any 1-D vector.

    Returns:
        Unit vector. A zero vector is returned unchanged.
    """
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def angle_between_vectors(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Angle between two vectors in degrees.

    Uses a numerically stable ``atan2`` formulation rather than ``arccos`` to
    avoid precision loss for nearly-parallel or nearly-antiparallel vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Angle in ``[0, 180]`` degrees.

    Raises:
        ValueError: If either vector has zero magnitude.
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if float(np.linalg.norm(va)) == 0.0 or float(np.linalg.norm(vb)) == 0.0:
        raise ValueError("angle_between_vectors is undefined for a zero vector")
    cross = float(np.linalg.norm(np.cross(va, vb)))
    dot = float(np.dot(va, vb))
    return float(np.rad2deg(np.arctan2(cross, dot)))
