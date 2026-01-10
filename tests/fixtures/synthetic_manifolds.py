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

"""Synthetic manifold generators for testing geometry algorithms.

Provides generators for manifolds with known ground truth properties:
- n-sphere: known intrinsic dimension n
- Swiss roll: 2D manifold in 3D, tests geodesic distance
- Flat torus: product manifold S^1 × S^1
- Linear subspace: exactly k-dimensional
- Hyperbolic paraboloid: negative curvature

All generators use the Backend protocol for GPU acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    pi_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class ManifoldSample:
    """A sample from a synthetic manifold.

    Attributes:
        points: [n_samples, ambient_dim] array of sampled points
        intrinsic_dimension: True intrinsic dimension of the manifold
        ambient_dimension: Dimension of the embedding space
        curvature: Expected curvature type ("positive", "negative", "flat")
        name: Descriptive name of the manifold
    """

    points: "Array"
    intrinsic_dimension: int
    ambient_dimension: int
    curvature: str
    name: str


def sample_sphere(
    n_samples: int = 100,
    dimension: int = 2,
    radius: float = 1.0,
    seed: int = 42,
    backend: "Backend | None" = None,
) -> ManifoldSample:
    """Sample uniformly from an n-sphere.

    The n-sphere S^n embedded in R^{n+1} has intrinsic dimension n.
    Uses Gaussian projection method for uniform sampling.

    Args:
        n_samples: Number of points to sample
        dimension: Intrinsic dimension (n for S^n)
        radius: Radius of the sphere
        seed: Random seed
        backend: Backend to use (defaults to MLX)

    Returns:
        ManifoldSample with points on the sphere

    Example:
        >>> sample = sample_sphere(n_samples=100, dimension=2)  # S^2 in R^3
        >>> assert sample.intrinsic_dimension == 2
        >>> assert sample.ambient_dimension == 3
    """
    b = backend or get_default_backend()
    b.random_seed(seed)

    ambient_dim = dimension + 1

    # Sample from Gaussian and normalize to sphere
    points = b.random_normal((n_samples, ambient_dim))
    norms_flat = geodesic_norms(points, b)
    b.eval(norms_flat)
    norms = b.reshape(norms_flat, (-1, 1))
    eps = division_epsilon(b, points)
    points = (points / (norms + eps)) * radius
    b.eval(points)

    return ManifoldSample(
        points=points,
        intrinsic_dimension=dimension,
        ambient_dimension=ambient_dim,
        curvature="positive",
        name=f"S^{dimension} (radius={radius})",
    )


def sample_swiss_roll(
    n_samples: int = 150,
    t_min: float = 1.5 * math.pi,
    t_max: float = 4.5 * math.pi,
    height: float = 10.0,
    seed: int = 42,
    backend: "Backend | None" = None,
) -> ManifoldSample:
    """Sample from a Swiss roll manifold.

    Swiss roll is a 2D manifold embedded in 3D:
    (t*cos(t), h, t*sin(t)) for t in [t_min, t_max], h in [0, height]

    The manifold is locally flat but globally curved, making it
    a good test for geodesic distance computation.

    Args:
        n_samples: Number of points to sample
        t_min: Minimum t parameter
        t_max: Maximum t parameter
        height: Height range
        seed: Random seed
        backend: Backend to use

    Returns:
        ManifoldSample with points on the Swiss roll
    """
    b = backend or get_default_backend()
    b.random_seed(seed)

    t = b.linspace(t_min, t_max, n_samples)
    b.random_seed(seed)
    noise = b.random_uniform(low=-0.2, high=0.2, shape=(n_samples,))
    t = t + noise
    h = b.random_uniform(low=0.0, high=height, shape=(n_samples,))

    x = t * b.cos(t)
    y = h
    z = t * b.sin(t)
    points = b.stack([x, y, z], axis=1)
    b.eval(points)

    return ManifoldSample(
        points=points,
        intrinsic_dimension=2,
        ambient_dimension=3,
        curvature="flat",  # Locally flat
        name="Swiss roll",
    )


def sample_flat_torus(
    n_samples: int = 150,
    radii: tuple[float, float] = (1.0, 1.0),
    seed: int = 42,
    backend: "Backend | None" = None,
) -> ManifoldSample:
    """Sample from a flat torus T^2 = S^1 × S^1.

    Embedded in R^4 as (cos(θ), sin(θ), cos(φ), sin(φ)).
    Intrinsic dimension is 2 (dim(S^1) + dim(S^1) = 1 + 1 = 2).

    Args:
        n_samples: Number of points
        radii: Radii of the two circles
        seed: Random seed
        backend: Backend to use

    Returns:
        ManifoldSample with points on the torus
    """
    b = backend or get_default_backend()
    b.random_seed(seed)

    r1, r2 = radii
    two_pi = 2.0 * pi_value(b)
    theta = b.linspace(0.0, two_pi, n_samples)
    b.random_seed(seed)
    noise_theta = b.random_uniform(low=-0.1, high=0.1, shape=(n_samples,))
    theta = theta + noise_theta

    b.random_seed(seed + 1000)
    phi = b.random_uniform(low=0.0, high=two_pi, shape=(n_samples,))

    points = b.stack(
        [
            r1 * b.cos(theta),
            r1 * b.sin(theta),
            r2 * b.cos(phi),
            r2 * b.sin(phi),
        ],
        axis=1,
    )
    b.eval(points)

    return ManifoldSample(
        points=points,
        intrinsic_dimension=2,
        ambient_dimension=4,
        curvature="flat",
        name="T^2 (flat torus)",
    )


def sample_linear_subspace(
    n_samples: int = 100,
    intrinsic_dim: int = 3,
    ambient_dim: int = 10,
    seed: int = 42,
    backend: "Backend | None" = None,
) -> ManifoldSample:
    """Sample from a linear subspace.

    Creates a random k-dimensional subspace of R^n by:
    1. Generating a random basis (k × n)
    2. Generating random coefficients (samples × k)
    3. Projecting: points = coeffs @ basis

    Args:
        n_samples: Number of points
        intrinsic_dim: Dimension of subspace (k)
        ambient_dim: Dimension of ambient space (n)
        seed: Random seed
        backend: Backend to use

    Returns:
        ManifoldSample with points in the subspace
    """
    b = backend or get_default_backend()

    # Generate random basis
    b.random_seed(seed + 100)
    basis = b.random_normal((intrinsic_dim, ambient_dim))
    b.eval(basis)

    # Generate random coefficients
    b.random_seed(seed)
    coeffs = b.random_normal((n_samples, intrinsic_dim))
    b.eval(coeffs)

    # Project onto subspace
    points = b.matmul(coeffs, basis)
    b.eval(points)

    return ManifoldSample(
        points=points,
        intrinsic_dimension=intrinsic_dim,
        ambient_dimension=ambient_dim,
        curvature="flat",
        name=f"Linear subspace (dim {intrinsic_dim} in R^{ambient_dim})",
    )


def sample_hyperbolic_paraboloid(
    n_samples: int = 100,
    scale: float = 1.0,
    seed: int = 42,
    backend: "Backend | None" = None,
) -> ManifoldSample:
    """Sample from a hyperbolic paraboloid (saddle surface).

    z = x^2 - y^2 for (x, y) in a square.
    This surface has negative Gaussian curvature everywhere.

    Args:
        n_samples: Number of points
        scale: Scale of the x-y domain
        seed: Random seed
        backend: Backend to use

    Returns:
        ManifoldSample with points on the saddle
    """
    b = backend or get_default_backend()
    b.random_seed(seed)

    b.random_seed(seed)
    x = b.random_uniform(low=-scale, high=scale, shape=(n_samples,))
    y = b.random_uniform(low=-scale, high=scale, shape=(n_samples,))
    z = x * x - y * y
    points = b.stack([x, y, z], axis=1)
    b.eval(points)

    return ManifoldSample(
        points=points,
        intrinsic_dimension=2,
        ambient_dimension=3,
        curvature="negative",
        name="Hyperbolic paraboloid",
    )


def random_orthogonal_matrix(
    n: int,
    seed: int = 42,
    backend: "Backend | None" = None,
) -> "Array":
    """Generate a random orthogonal matrix using QR decomposition.

    Args:
        n: Matrix dimension
        seed: Random seed
        backend: Backend to use

    Returns:
        n × n orthogonal matrix
    """
    b = backend or get_default_backend()
    b.random_seed(seed)

    # Generate random matrix
    A = b.random_normal((n, n))
    b.eval(A)

    # QR decomposition gives orthogonal Q
    Q, _ = b.qr(A)
    b.eval(Q)

    return Q
