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

"""Statistical utilities for numerical computations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .precision import division_epsilon, precision_dtype
from .scalars import is_finite

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _geodesic_triplet_distances(
    lhs: "Array",
    rhs: "Array",
    backend: "Backend",
) -> tuple[float, float, float]:
    """Return geodesic distances (0,lhs), (0,rhs), (lhs,rhs) for two vectors."""
    from modelcypher.core.domain.geometry.riemannian_core import _get_riemannian_geometry

    lhs_vec = backend.reshape(lhs, (1, -1))
    rhs_vec = backend.reshape(rhs, (1, -1))
    zero = backend.zeros_like(lhs_vec)
    points = backend.concatenate([zero, lhs_vec, rhs_vec], axis=0)
    rg = _get_riemannian_geometry(backend)
    geo_result = rg.geodesic_distances(points, k_neighbors=2)
    backend.eval(geo_result.distances)

    d01 = float(backend.to_scalar(geo_result.distances[0, 1]))
    d02 = float(backend.to_scalar(geo_result.distances[0, 2]))
    d12 = float(backend.to_scalar(geo_result.distances[1, 2]))
    return d01, d02, d12


def _geodesic_correlation(
    lhs: "Array",
    rhs: "Array",
    backend: "Backend",
    *,
    eps: float,
    default: float,
) -> float:
    """Compute geodesic correlation between two arrays."""
    mean_l = backend.mean(lhs)
    mean_r = backend.mean(rhs)
    diff_l = lhs - mean_l
    diff_r = rhs - mean_r
    backend.eval(diff_l, diff_r)

    d0a, d0b, dab = _geodesic_triplet_distances(diff_l, diff_r, backend)
    d0a = max(0.0, d0a)
    d0b = max(0.0, d0b)
    if d0a <= eps or d0b <= eps:
        return default

    denom = 2.0 * d0a * d0b
    if denom <= eps:
        return default
    cos_val = (d0a * d0a + d0b * d0b - dab * dab) / denom
    corr = max(-1.0, min(1.0, cos_val))
    return corr if is_finite(corr, backend) else default


def compute_median(array: "Array", backend: "Backend") -> float:
    """Compute median of array values using O(n) argpartition."""
    flat = backend.reshape(array, (-1,))
    backend.eval(flat)

    n = int(flat.shape[0])
    if n == 0:
        return 0.0
    if n == 1:
        return float(backend.to_scalar(flat))

    mid = n // 2

    if n % 2 == 1:
        partitioned = backend.argpartition(flat, mid)
        backend.eval(partitioned)
        prefix = backend.take(partitioned, backend.arange(mid + 1), axis=0)
        median_arr = backend.max(backend.take(flat, prefix, axis=0))
    else:
        low_part = backend.argpartition(flat, mid - 1)
        backend.eval(low_part)
        low_prefix = backend.take(low_part, backend.arange(mid), axis=0)
        low = backend.max(backend.take(flat, low_prefix, axis=0))

        high_part = backend.argpartition(flat, mid)
        backend.eval(high_part)
        high_prefix = backend.take(high_part, backend.arange(mid + 1), axis=0)
        high = backend.max(backend.take(flat, high_prefix, axis=0))

        median_arr = (low + high) * 0.5

    median_arr = backend.squeeze(median_arr)
    backend.eval(median_arr)
    return float(backend.to_scalar(median_arr))


def compute_median_nonzero(
    array: "Array",
    backend: "Backend",
    zero_threshold: float | None = None,
) -> float:
    """Compute median of non-zero values using O(n) argpartition."""
    flat = backend.reshape(array, (-1,))
    backend.eval(flat)

    n = int(flat.shape[0])
    if n == 0:
        return 0.0

    if zero_threshold is None:
        zero_threshold = division_epsilon(backend, flat)

    zero_mask = flat <= zero_threshold
    zero_count_arr = backend.sum(backend.astype(zero_mask, "int32"))
    backend.eval(zero_count_arr)
    zero_count = int(backend.to_scalar(zero_count_arr))

    non_zero_count = n - zero_count
    if non_zero_count <= 0:
        return 0.0

    median_idx = min(zero_count + (non_zero_count // 2), n - 1)

    partitioned = backend.argpartition(flat, median_idx)
    backend.eval(partitioned)

    prefix_indices = backend.take(partitioned, backend.arange(median_idx + 1), axis=0)
    prefix_vals = backend.take(flat, prefix_indices, axis=0)
    backend.eval(prefix_vals)

    median_arr = backend.max(prefix_vals)
    backend.eval(median_arr)

    return float(backend.to_scalar(median_arr))


def compute_pearson_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute geodesic correlation coefficient between two lists."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    error_value = default if default is not None else float("nan")

    if not lhs or len(lhs) != len(rhs):
        return error_value

    lhs_arr = backend.array(lhs)
    rhs_arr = backend.array(rhs)
    eps = division_epsilon(backend, lhs_arr)
    return _geodesic_correlation(lhs_arr, rhs_arr, backend, eps=eps, default=error_value)


def compute_spearman_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute Spearman rank correlation coefficient between two lists."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    error_value = default if default is not None else float("nan")

    if not lhs or len(lhs) != len(rhs) or len(lhs) < 2:
        return error_value

    lhs_arr = backend.array(lhs)
    rhs_arr = backend.array(rhs)

    lhs_rank = backend.argsort(backend.argsort(lhs_arr, axis=0), axis=0)
    rhs_rank = backend.argsort(backend.argsort(rhs_arr, axis=0), axis=0)
    rank_dtype = precision_dtype(backend, reference=lhs_rank)
    lhs_rank = backend.astype(lhs_rank, rank_dtype)
    rhs_rank = backend.astype(rhs_rank, rank_dtype)

    eps = division_epsilon(backend, lhs_rank)
    return _geodesic_correlation(lhs_rank, rhs_rank, backend, eps=eps, default=error_value)


__all__ = [
    "compute_median",
    "compute_median_nonzero",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
]
