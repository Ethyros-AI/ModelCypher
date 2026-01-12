# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Analytic manifolds for validating geodesic and curvature estimates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def sample_circle_points(
    n: int,
    radius: float = 1.0,
    backend: "Backend | None" = None,
    dtype: str | None = None,
) -> "Array":
    """Sample evenly spaced points on a circle of given radius."""
    b = backend or get_default_backend()
    if n <= 0:
        return b.zeros((0, 2))

    two_pi = 2.0 * math.pi
    if n == 1:
        angles = b.array([0.0], dtype=dtype)
    else:
        stop = two_pi * (1.0 - 1.0 / float(n))
        angles = b.linspace(0.0, stop, n, dtype=dtype)

    x = radius * b.cos(angles)
    y = radius * b.sin(angles)
    points = b.stack([x, y], axis=1)
    b.eval(points)
    return points


def sample_sphere_points(
    n_lat: int,
    n_lon: int,
    radius: float = 1.0,
    backend: "Backend | None" = None,
    dtype: str | None = None,
) -> "Array":
    """Sample points on a sphere using a lat/lon grid."""
    b = backend or get_default_backend()
    if n_lat <= 0 or n_lon <= 0:
        return b.zeros((0, 3))

    pi = math.pi
    two_pi = 2.0 * pi
    lat = b.linspace(-0.5 * pi, 0.5 * pi, n_lat, dtype=dtype)
    if n_lon == 1:
        lon = b.array([0.0], dtype=dtype)
    else:
        stop = two_pi * (1.0 - 1.0 / float(n_lon))
        lon = b.linspace(0.0, stop, n_lon, dtype=dtype)

    lon_grid, lat_grid = b.meshgrid(lon, lat, indexing="xy")
    cos_lat = b.cos(lat_grid)
    x = radius * cos_lat * b.cos(lon_grid)
    y = radius * cos_lat * b.sin(lon_grid)
    z = radius * b.sin(lat_grid)
    points = b.stack([x, y, z], axis=2)
    points = b.reshape(points, (-1, 3))
    b.eval(points)
    return points


def analytic_geodesic_distances(
    points: "Array",
    radius: float = 1.0,
    backend: "Backend | None" = None,
) -> "Array":
    """Compute analytic geodesic distances for points on a sphere/circle."""
    b = backend or get_default_backend()
    points = b.array(points)
    b.eval(points)

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    dot = b.matmul(points, b.transpose(points))
    inv_r2 = 1.0 / (radius * radius)
    dot = dot * inv_r2

    ones = b.full(dot.shape, 1.0, dtype=dot.dtype)
    neg_ones = b.full(dot.shape, -1.0, dtype=dot.dtype)
    dot = b.maximum(neg_ones, b.minimum(dot, ones))
    angles = b.arccos(dot)
    distances = angles * radius
    b.eval(distances)
    return distances


__all__ = [
    "analytic_geodesic_distances",
    "sample_circle_points",
    "sample_sphere_points",
]
