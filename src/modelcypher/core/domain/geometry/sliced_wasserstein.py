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

"""Sliced Wasserstein distance for scalable distribution comparison.

The Sliced Wasserstein distance approximates optimal transport by projecting
high-dimensional distributions onto random 1D directions and averaging the
1D Wasserstein distances.

Complexity:
    - Full Wasserstein: O(n³) for n samples
    - Gromov-Wasserstein: O(n² m²) for n×m problems
    - Sliced Wasserstein: O(n_slices × n log n)

Use Cases:
    - Comparing activation distributions between models
    - Fast approximation when full OT is too expensive
    - Large-scale probe set comparisons

References:
    - Bonneel et al. (2015) "Sliced and Radon Wasserstein Barycenters"
    - Kolouri et al. (2019) "Generalized Sliced Wasserstein Distances"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_svd,
    precision_dtype,
    regularization_epsilon,
    svd_auto_rank,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class SlicedWassersteinResult:
    """Result of Sliced Wasserstein distance computation."""

    # Mean Sliced Wasserstein distance (p=2)
    distance: float

    # Standard deviation across slices (uncertainty estimate)
    std: float

    # Number of slices used
    n_slices: int

    # Dimension of the point clouds
    dimension: int

    # Number of points in each distribution
    n_points_x: int
    n_points_y: int

    # Individual slice distances (for analysis)
    slice_distances: Any | None = None


def random_unit_vectors(
    n_slices: int,
    dimension: int,
    backend: "Backend",
    seed: int | None = None,
) -> "Array":
    """Generate n_slices random unit vectors on the d-dimensional sphere.

    Uses normalized Gaussian vectors, which are uniformly distributed
    on the unit sphere.

    Parameters
    ----------
    n_slices : int
        Number of random directions to generate.
    dimension : int
        Dimensionality of the space.
    backend : Backend
        Compute backend.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Array
        Random unit vectors of shape [n_slices, dimension].
    """
    if seed is not None:
        backend.random_seed(seed)

    # Sample from standard normal distribution
    vectors = backend.random_normal((n_slices, dimension))
    backend.eval(vectors)

    # Normalize to unit length
    norms = backend.sqrt(backend.sum(vectors * vectors, axis=1, keepdims=True))
    reg = regularization_epsilon(backend, vectors)
    norms = backend.maximum(norms, backend.full(backend.shape(norms), reg))
    backend.eval(norms)

    unit_vectors = vectors / norms
    backend.eval(unit_vectors)

    return unit_vectors


def _derive_slice_count(
    points: "Array",
    backend: "Backend",
    extras: list["Array"] | None = None,
) -> int:
    """Derive slice count from numeric rank of centered points."""
    arrays = [backend.array(points)]
    if extras:
        arrays.extend(backend.array(extra) for extra in extras)
    pts = backend.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    pts = backend.astype(pts, precision_dtype(backend, reference=pts))
    backend.eval(pts)

    n = int(backend.shape(pts)[0])
    d = int(backend.shape(pts)[1])
    if n <= 1 or d <= 1:
        return max(1, min(n, d))

    try:
        mean = backend.mean(pts, axis=0, keepdims=True)
        centered = pts - mean
        backend.eval(centered)

        _u, singular_values, _v = geodesic_svd(backend, centered)
        backend.eval(singular_values)

        rank = svd_auto_rank(singular_values, backend, max_dim=max(n, d))
        return max(1, rank)
    except Exception:
        return max(1, min(n, d))


def wasserstein_1d(
    x_sorted: "Array",
    y_sorted: "Array",
    backend: "Backend",
    p: int = 2,
) -> float:
    """Compute 1D Wasserstein distance between sorted samples.

    For sorted samples of equal size, the p-Wasserstein distance is:
        W_p = (mean(|x_sorted - y_sorted|^p))^(1/p)

    For unequal sizes, uses quantile interpolation.

    Parameters
    ----------
    x_sorted : Array
        Sorted 1D samples from first distribution, shape [n].
    y_sorted : Array
        Sorted 1D samples from second distribution, shape [m].
    backend : Backend
        Compute backend.
    p : int
        Order of Wasserstein distance (default 2).

    Returns
    -------
    float
        1D Wasserstein-p distance.
    """
    n = int(backend.shape(x_sorted)[0])
    m = int(backend.shape(y_sorted)[0])

    if n == m:
        # Equal sizes: direct comparison
        diff = backend.abs(x_sorted - y_sorted)
        if p == 1:
            w_p = backend.mean(diff)
        elif p == 2:
            w_p = backend.sqrt(backend.mean(diff * diff))
        else:
            w_p = backend.pow(backend.mean(backend.pow(diff, p)), 1.0 / p)
        backend.eval(w_p)
        return float(backend.to_scalar(w_p))

    # Unequal sizes: use quantile matching
    # Interpolate to common grid
    max_n = max(n, m)
    compute_dtype = precision_dtype(backend, reference=x_sorted)

    # Create quantile indices
    quantiles = (backend.arange(max_n) + 0.5) / max_n  # [max_n]
    quantiles = backend.astype(quantiles, compute_dtype)
    backend.eval(quantiles)

    # Interpolate x to quantiles
    x_indices = quantiles * (n - 1)
    x_lower = backend.astype(backend.floor(x_indices), backend.dtype(backend.array([0])))
    x_upper = backend.minimum(x_lower + 1, n - 1)
    x_frac = x_indices - backend.astype(x_lower, compute_dtype)
    backend.eval(x_lower, x_upper, x_frac)

    x_interp = (
        backend.take(x_sorted, x_lower, axis=0) * (1 - x_frac)
        + backend.take(x_sorted, x_upper, axis=0) * x_frac
    )
    backend.eval(x_interp)

    # Interpolate y to quantiles
    y_indices = quantiles * (m - 1)
    y_lower = backend.astype(backend.floor(y_indices), backend.dtype(backend.array([0])))
    y_upper = backend.minimum(y_lower + 1, m - 1)
    y_frac = y_indices - backend.astype(y_lower, compute_dtype)
    backend.eval(y_lower, y_upper, y_frac)

    y_interp = (
        backend.take(y_sorted, y_lower, axis=0) * (1 - y_frac)
        + backend.take(y_sorted, y_upper, axis=0) * y_frac
    )
    backend.eval(y_interp)

    # Compute Wasserstein distance on interpolated samples
    diff = backend.abs(x_interp - y_interp)
    if p == 1:
        w_p = backend.mean(diff)
    elif p == 2:
        w_p = backend.sqrt(backend.mean(diff * diff))
    else:
        w_p = backend.pow(backend.mean(backend.pow(diff, p)), 1.0 / p)
    backend.eval(w_p)

    return float(backend.to_scalar(w_p))


def sliced_wasserstein_distance(
    X: "Array",
    Y: "Array",
    n_slices: int | None = None,
    p: int = 2,
    backend: "Backend | None" = None,
    seed: int | None = None,
    return_slice_distances: bool = False,
) -> SlicedWassersteinResult:
    """Compute Sliced Wasserstein distance between two point clouds.

    Algorithm:
        1. Generate n_slices random unit vectors θ_1, ..., θ_n on S^(d-1)
        2. For each θ_i:
           a. Project X and Y: x_proj = X @ θ_i, y_proj = Y @ θ_i
           b. Sort projections
           c. Compute 1D Wasserstein distance
        3. Return mean (and std) of slice distances

    Parameters
    ----------
    X : Array
        First point cloud, shape [n, d].
    Y : Array
        Second point cloud, shape [m, d].
    n_slices : int | None
        Number of random projections. If None, derived from numeric rank of
        the concatenated point cloud.
    p : int
        Order of Wasserstein distance (default 2).
    backend : Backend, optional
        Compute backend.
    seed : int, optional
        Random seed for reproducibility.
    return_slice_distances : bool
        If True, include individual slice distances in result.

    Returns
    -------
    SlicedWassersteinResult
        Result containing mean distance, std, and diagnostics.

    Notes
    -----
    Complexity: O(n_slices × max(n,m) × log(max(n,m)))

    When n_slices is None, the count is derived from numeric rank.
    """
    b = backend or get_default_backend()

    X = b.array(X)
    Y = b.array(Y)
    compute_dtype = precision_dtype(b, reference=X)
    X = b.astype(X, compute_dtype)
    Y = b.astype(Y, compute_dtype)
    b.eval(X, Y)

    X_shape = b.shape(X)
    Y_shape = b.shape(Y)

    if len(X_shape) != 2 or len(Y_shape) != 2:
        raise ValueError(
            f"X and Y must be 2D point clouds. Got shapes {X_shape} and {Y_shape}"
        )

    n, d_x = int(X_shape[0]), int(X_shape[1])
    m, d_y = int(Y_shape[0]), int(Y_shape[1])

    if d_x != d_y:
        raise ValueError(
            f"X and Y must have same dimension. Got {d_x} and {d_y}"
        )

    d = d_x

    if n_slices is None:
        n_slices = _derive_slice_count(X, b, extras=[Y])

    # Generate random unit vectors
    directions = random_unit_vectors(n_slices, d, b, seed=seed)
    # directions: [n_slices, d]

    # Project all points onto all directions via matmul
    # X @ directions.T: [n, d] @ [d, n_slices] = [n, n_slices]
    # Y @ directions.T: [m, d] @ [d, n_slices] = [m, n_slices]
    X_proj = b.matmul(X, b.transpose(directions))  # [n, n_slices]
    Y_proj = b.matmul(Y, b.transpose(directions))  # [m, n_slices]
    b.eval(X_proj, Y_proj)

    # Compute 1D Wasserstein for each slice
    slice_distances = []
    for i in range(n_slices):
        # Extract column i and sort
        x_slice = X_proj[:, i]
        y_slice = Y_proj[:, i]

        # Sort both
        x_sorted_idx = b.argsort(x_slice)
        y_sorted_idx = b.argsort(y_slice)
        x_sorted = b.take(x_slice, x_sorted_idx, axis=0)
        y_sorted = b.take(y_slice, y_sorted_idx, axis=0)
        b.eval(x_sorted, y_sorted)

        w_1d = wasserstein_1d(x_sorted, y_sorted, b, p=p)
        slice_distances.append(w_1d)

    # Compute mean and std
    distances_arr = b.array(slice_distances)
    b.eval(distances_arr)

    mean_distance = float(b.to_scalar(b.mean(distances_arr)))
    std_distance = float(b.to_scalar(b.std(distances_arr)))

    logger.debug(
        "SW DISTANCE: d=%d, n_slices=%d, mean=%.4f, std=%.4f",
        d, n_slices, mean_distance, std_distance
    )

    return SlicedWassersteinResult(
        distance=mean_distance,
        std=std_distance,
        n_slices=n_slices,
        dimension=d,
        n_points_x=n,
        n_points_y=m,
        slice_distances=distances_arr if return_slice_distances else None,
    )


def sliced_wasserstein_batch(
    X: "Array",
    Y_list: list["Array"],
    n_slices: int | None = None,
    p: int = 2,
    backend: "Backend | None" = None,
    seed: int | None = None,
) -> list[SlicedWassersteinResult]:
    """Compute Sliced Wasserstein distance from X to multiple point clouds.

    More efficient than calling sliced_wasserstein_distance repeatedly
    because random directions are shared.

    Parameters
    ----------
    X : Array
        Reference point cloud, shape [n, d].
    Y_list : list[Array]
        List of point clouds to compare against X.
    n_slices : int | None
        Number of random projections. If None, derived from numeric rank of
        the concatenated point cloud.
    p : int
        Order of Wasserstein distance.
    backend : Backend, optional
        Compute backend.
    seed : int, optional
        Random seed.

    Returns
    -------
    list[SlicedWassersteinResult]
        Results for each Y in Y_list.
    """
    b = backend or get_default_backend()

    X = b.array(X)
    compute_dtype = precision_dtype(b, reference=X)
    X = b.astype(X, compute_dtype)
    b.eval(X)

    d = int(b.shape(X)[1])
    n = int(b.shape(X)[0])

    if n_slices is None:
        n_slices = _derive_slice_count(X, b, extras=Y_list)

    Y_arrays: list["Array"] = []
    for Y in Y_list:
        Y_arr = b.array(Y)
        Y_arr = b.astype(Y_arr, compute_dtype)
        b.eval(Y_arr)
        d_y = int(b.shape(Y_arr)[1])
        if d_y != d:
            raise ValueError(f"Y dimension {d_y} != X dimension {d}")
        Y_arrays.append(Y_arr)

    if n_slices is None:
        n_slices = _derive_slice_count(X, b, extras=Y_arrays)

    # Generate shared random directions
    directions = random_unit_vectors(n_slices, d, b, seed=seed)

    # Project X once
    X_proj = b.matmul(X, b.transpose(directions))  # [n, n_slices]
    b.eval(X_proj)

    # Sort X projections for each slice
    X_sorted_slices = []
    for i in range(n_slices):
        x_slice = X_proj[:, i]
        x_sorted_idx = b.argsort(x_slice)
        x_sorted = b.take(x_slice, x_sorted_idx, axis=0)
        b.eval(x_sorted)
        X_sorted_slices.append(x_sorted)

    results = []
    for Y in Y_arrays:
        m = int(b.shape(Y)[0])
        Y_proj = b.matmul(Y, b.transpose(directions))
        b.eval(Y_proj)

        slice_distances = []
        for i in range(n_slices):
            y_slice = Y_proj[:, i]
            y_sorted_idx = b.argsort(y_slice)
            y_sorted = b.take(y_slice, y_sorted_idx, axis=0)
            b.eval(y_sorted)

            w_1d = wasserstein_1d(X_sorted_slices[i], y_sorted, b, p=p)
            slice_distances.append(w_1d)

        distances_arr = b.array(slice_distances)
        b.eval(distances_arr)

        mean_distance = float(b.to_scalar(b.mean(distances_arr)))
        std_distance = float(b.to_scalar(b.std(distances_arr)))

        results.append(
            SlicedWassersteinResult(
                distance=mean_distance,
                std=std_distance,
                n_slices=n_slices,
                dimension=d,
                n_points_x=n,
                n_points_y=m,
            )
        )

    return results


def sliced_wasserstein_similarity(
    X: "Array",
    Y: "Array",
    n_slices: int | None = None,
    backend: "Backend | None" = None,
    seed: int | None = None,
) -> float:
    """Compute a similarity score (0-1) based on Sliced Wasserstein distance.

    Converts the unbounded distance to a bounded similarity using:
        similarity = exp(-distance / scale)

    where scale is the mean std of the point clouds (data-derived).

    Parameters
    ----------
    X : Array
        First point cloud.
    Y : Array
        Second point cloud.
    n_slices : int | None
        Number of random projections. If None, derived from numeric rank.
    backend : Backend, optional
        Compute backend.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Similarity score in [0, 1]. 1 = identical distributions.
    """
    b = backend or get_default_backend()

    X = b.array(X)
    Y = b.array(Y)
    b.eval(X, Y)

    # Compute SW distance
    result = sliced_wasserstein_distance(X, Y, n_slices=n_slices, backend=b, seed=seed)

    # Compute scale from data (mean std across dimensions)
    X_std = b.std(X, axis=0)
    Y_std = b.std(Y, axis=0)
    b.eval(X_std, Y_std)

    mean_std = (
        float(b.to_scalar(b.mean(X_std))) + float(b.to_scalar(b.mean(Y_std)))
    ) / 2

    reg = regularization_epsilon(b, X)
    scale = max(mean_std, reg)

    # Convert distance to similarity
    import math
    similarity = math.exp(-result.distance / scale)

    return similarity


__all__ = [
    "SlicedWassersteinResult",
    "random_unit_vectors",
    "wasserstein_1d",
    "sliced_wasserstein_distance",
    "sliced_wasserstein_batch",
    "sliced_wasserstein_similarity",
]
