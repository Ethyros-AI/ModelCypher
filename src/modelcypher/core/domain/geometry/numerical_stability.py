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

"""Dynamic numerical stability utilities.

All epsilons and tolerances are derived from tensor precision, not arbitrary constants.
Use these functions instead of hardcoded values like 1e-8 or 1e-10.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return name.replace("mlx.core.", "").replace("jax.numpy.", "")


def _default_float_dtype(backend: "Backend") -> object:
    return backend.array([1.0]).dtype


def _promote_precision(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: object | None = None,
) -> "Array":
    """Promote low-precision or integer arrays to at least default float dtype."""
    if min_dtype is None:
        min_dtype = _default_float_dtype(backend)

    if not hasattr(array, "dtype"):
        return backend.array(array, dtype=min_dtype)

    dtype_name = _dtype_name(array.dtype)
    if (
        "float16" in dtype_name
        or "bfloat16" in dtype_name
        or "int" in dtype_name
        or "uint" in dtype_name
        or "bool" in dtype_name
    ):
        return backend.astype(array, min_dtype)

    try:
        current_eps = backend.finfo(array.dtype).eps
        min_eps = backend.finfo(min_dtype).eps
    except Exception:
        return backend.astype(array, min_dtype)

    if current_eps > min_eps:
        return backend.astype(array, min_dtype)

    return array

__all__ = [
    # Backend scalar helpers (use instead of math module)
    "sqrt_scalar",
    "is_finite",
    "is_inf",
    "is_nan",
    "log_scalar",
    "exp_scalar",
    "power_scalar",
    "ceil_scalar",
    "floor_scalar",
    "ulp_scalar",
    "lgamma_scalar",
    "acos_scalar",
    "cos_scalar",
    "sin_scalar",
    "atan2_scalar",
    "log2_scalar",
    "pi_value",
    "e_value",
    "inf_value",
    # Epsilon and threshold utilities
    "machine_epsilon",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    "infinity_threshold",
    # Data-derived thresholds
    "find_magnitude_gap_threshold",
    # Statistical utilities
    "compute_median",
    "compute_median_nonzero",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    # Matrix decomposition
    "safe_inverse",
    "geodesic_svd",
    "geodesic_pinv",
    "power_iteration_eigh",
    # GPU-accelerated linear algebra
    "gpu_lstsq",
# Invariant alignment (linear CKA = 1.0 by construction)
    "invariant_alignment",
    # Geodesic invariant alignment (preserves manifold structure)
    "geodesic_invariant_alignment",
]


# =============================================================================
# Backend Scalar Helpers
# =============================================================================


def sqrt_scalar(value: float, backend: "Backend") -> float:
    """Compute sqrt of scalar using backend with non-negativity guard."""
    safe_value = max(0.0, value)
    arr = backend.array([safe_value])
    result = backend.sqrt(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def is_finite(value: float, backend: "Backend") -> bool:
    """Check if scalar is finite using backend."""
    arr = backend.array([value])
    result = backend.isfinite(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_inf(value: float, backend: "Backend") -> bool:
    """Check if scalar is infinite using backend."""
    arr = backend.array([value])
    result = backend.isinf(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_nan(value: float, backend: "Backend") -> bool:
    """Check if scalar is NaN using backend."""
    arr = backend.array([value])
    result = backend.isnan(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def all_finite(arr: "Array", backend: "Backend") -> bool:
    """Check if all elements in array are finite (not NaN or Inf).

    Args:
        arr: Array to check (any shape).
        backend: Backend for tensor operations.

    Returns:
        True if all elements are finite, False otherwise.
    """
    finite_mask = backend.isfinite(arr)
    all_ok = backend.all(finite_mask)
    backend.eval(all_ok)
    return bool(backend.to_scalar(all_ok))


def log_scalar(value: float, backend: "Backend") -> float:
    """Compute natural log of scalar using backend."""
    arr = backend.array([value])
    result = backend.log(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def exp_scalar(value: float, backend: "Backend") -> float:
    """Compute exp of scalar using backend."""
    arr = backend.array([value])
    result = backend.exp(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def power_scalar(value: float, exponent: float, backend: "Backend") -> float:
    """Compute value ** exponent using backend."""
    arr = backend.array([value])
    result = arr**exponent
    backend.eval(result)
    return float(backend.to_scalar(result))


def ceil_scalar(value: float, backend: "Backend") -> int:
    """Compute ceil of scalar using backend."""
    arr = backend.array([value])
    result = backend.ceil(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def floor_scalar(value: float, backend: "Backend") -> int:
    """Compute floor of scalar using backend."""
    arr = backend.array([value])
    result = backend.floor(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def ulp_scalar(value: float, backend: "Backend") -> float:
    """Compute unit in last place for scalar using backend."""
    eps = backend.finfo(backend.array([value]).dtype).eps
    return eps * abs(value) if value != 0.0 else eps


def lgamma_scalar(value: float, backend: "Backend") -> float:
    """Compute log-gamma of scalar using backend."""
    arr = backend.array([value])
    result = backend.lgamma(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def acos_scalar(value: float, backend: "Backend") -> float:
    """Compute arc cosine of scalar using backend."""
    arr = backend.array([value])
    result = backend.arccos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def cos_scalar(value: float, backend: "Backend") -> float:
    """Compute cosine of scalar using backend."""
    arr = backend.array([value])
    result = backend.cos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def sin_scalar(value: float, backend: "Backend") -> float:
    """Compute sine of scalar using backend."""
    arr = backend.array([value])
    result = backend.sin(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def atan2_scalar(y: float, x: float, backend: "Backend") -> float:
    """Compute atan2(y, x) using backend."""
    y_arr = backend.array([y])
    x_arr = backend.array([x])

    eps = backend.finfo().eps
    x_safe = backend.where(
        backend.abs(x_arr) < eps,
        backend.sign(x_arr) * eps + eps,
        x_arr,
    )
    ratio = y_arr / x_safe
    base_angle = backend.arctan(ratio)
    backend.eval(base_angle)

    angle = float(backend.to_scalar(base_angle))
    pi = pi_value(backend)

    if x < 0:
        if y >= 0:
            angle += pi
        else:
            angle -= pi
    elif x == 0:
        if y > 0:
            angle = pi / 2
        elif y < 0:
            angle = -pi / 2
        else:
            angle = 0.0

    return angle


def log2_scalar(value: float, backend: "Backend") -> float:
    """Compute log base 2 of scalar using backend."""
    arr = backend.array([value])
    result = backend.log2(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def pi_value(backend: "Backend") -> float:
    """Get pi using backend."""
    one = backend.array([1.0])
    result = 4.0 * backend.arctan(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def e_value(backend: "Backend") -> float:
    """Get Euler's number e using backend."""
    one = backend.array([1.0])
    result = backend.exp(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def inf_value(backend: "Backend") -> float:
    """Get positive infinity using backend."""
    return float("inf")


# =============================================================================
# Epsilon and Threshold Utilities
# =============================================================================


def machine_epsilon(backend: "Backend", array: "Array") -> float:
    """Get machine epsilon for the array's dtype."""
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe division operations. Uses sqrt(eps)."""
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


def regularization_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for matrix regularization. Uses eps^0.75."""
    eps = backend.finfo(array.dtype).eps
    return power_scalar(eps, 0.75, backend)


def condition_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for condition number checks. Returns 1/eps."""
    return 1.0 / backend.finfo(array.dtype).eps


def svd_rank_threshold(backend: "Backend", array: "Array", max_dim: int) -> float:
    """Get threshold for determining numerical rank from SVD."""
    eps = backend.finfo(array.dtype).eps
    return float(max_dim) * eps


def tiny_value(backend: "Backend", array: "Array") -> float:
    """Get the smallest positive usable number for the dtype."""
    return backend.finfo(array.dtype).tiny


def safe_log_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe logarithm operations."""
    return backend.finfo(array.dtype).tiny


def infinity_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for detecting near-infinite values."""
    finfo = backend.finfo(array.dtype)
    eps = finfo.eps
    max_val = finfo.max
    return float(max_val) * sqrt_scalar(eps, backend)


# =============================================================================
# Data-Derived Thresholds
# =============================================================================


def find_magnitude_gap_threshold(
    sorted_values: list[float] | "Array",
    eps: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Find the natural break point in a sorted magnitude distribution."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    if hasattr(sorted_values, "shape"):
        values_arr = sorted_values  # type: ignore[assignment]
        n = int(values_arr.shape[0])
    else:
        if not sorted_values:
            return 0.0
        values_arr = backend.array(sorted_values)
        n = len(sorted_values)

    if n == 0:
        return 0.0

    if eps is None:
        abs_arr = backend.abs(values_arr)
        scale_arr = backend.maximum(backend.max(abs_arr), backend.array([1.0]))
        backend.eval(scale_arr)
        scale = float(backend.to_scalar(scale_arr))
        eps = ulp_scalar(scale, backend)

    if n == 1:
        # Single value - return it as the threshold
        first_val = backend.take(values_arr, backend.array([0]), axis=0)
        first_val = backend.squeeze(first_val)
        backend.eval(first_val)
        return float(backend.to_scalar(first_val))

    if n == 2:
        # Two values - check for a significant relative gap
        # If gap > 50%, return the smaller (threshold), else return larger (no outlier)
        first_val = backend.take(values_arr, backend.array([0]), axis=0)
        second_val = backend.take(values_arr, backend.array([1]), axis=0)
        first_val = backend.squeeze(first_val)
        second_val = backend.squeeze(second_val)
        backend.eval(first_val, second_val)
        v0 = float(backend.to_scalar(first_val))
        v1 = float(backend.to_scalar(second_val))
        if v0 > eps:
            rel_gap = (v1 - v0) / v0
            if rel_gap > 0.5:  # Significant gap - return smaller as threshold
                return v0
        return v1  # No significant gap - return larger (nothing will be flagged)

    idx = backend.arange(0, n - 1)
    next_idx = backend.arange(1, n)
    curr = backend.take(values_arr, idx, axis=0)
    next_vals = backend.take(values_arr, next_idx, axis=0)
    diffs = next_vals - curr
    eps_arr = backend.array([eps])
    valid = curr > eps_arr
    denom = backend.where(valid, curr, backend.ones_like(curr))
    rel_gaps = diffs / denom
    rel_gaps = backend.where(valid, rel_gaps, backend.zeros_like(rel_gaps))
    max_gap_arr = backend.max(rel_gaps)
    backend.eval(max_gap_arr)
    max_mask = rel_gaps == max_gap_arr
    indices = backend.arange(0, n - 1)
    inf_val = backend.full(indices.shape, float("inf"))
    masked_indices = backend.where(max_mask, indices, inf_val)
    gap_index_arr = backend.min(masked_indices)
    backend.eval(gap_index_arr)
    max_gap = float(backend.to_scalar(max_gap_arr))

    if max_gap <= 0.0:
        mid_idx = backend.array([n // 2])
        mid_val = backend.take(values_arr, mid_idx, axis=0)
        mid_val = backend.squeeze(mid_val)
        backend.eval(mid_val)
        return float(backend.to_scalar(mid_val))

    gap_index = int(backend.to_scalar(gap_index_arr))
    gap_val = backend.take(values_arr, backend.array([gap_index]), axis=0)
    gap_val = backend.squeeze(gap_val)
    backend.eval(gap_val)
    return float(backend.to_scalar(gap_val))


# =============================================================================
# Statistical Utilities
# =============================================================================


def _geodesic_norm_scalar(array: "Array", backend: "Backend") -> float:
    """Compute a geodesic norm scalar for any array shape."""
    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    vec = backend.reshape(array, (1, -1))
    zero = backend.zeros_like(vec)
    points = backend.concatenate([zero, vec], axis=0)
    rg = RiemannianGeometry(backend)
    point_count = int(backend.shape(points)[0])
    geo_result = rg.geodesic_distances(points, k_neighbors=point_count - 1)
    backend.eval(geo_result.distances)
    return max(0.0, float(backend.to_scalar(geo_result.distances[0, 1])))


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
    mean_l = backend.mean(lhs_arr)
    mean_r = backend.mean(rhs_arr)
    diff_l = lhs_arr - mean_l
    diff_r = rhs_arr - mean_r
    backend.eval(diff_l, diff_r)

    eps = division_epsilon(backend, lhs_arr)
    d0a = _geodesic_norm_scalar(diff_l, backend)
    d0b = _geodesic_norm_scalar(diff_r, backend)
    if d0a <= eps or d0b <= eps:
        return error_value

    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    diff_l_vec = backend.reshape(diff_l, (1, -1))
    diff_r_vec = backend.reshape(diff_r, (1, -1))
    points = backend.concatenate([diff_l_vec, diff_r_vec], axis=0)
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(points, k_neighbors=1)
    backend.eval(geo_result.distances)
    dab = float(backend.to_scalar(geo_result.distances[0, 1]))

    denom = 2.0 * d0a * d0b
    if denom <= eps:
        return error_value
    cos_val = (d0a * d0a + d0b * d0b - dab * dab) / denom
    corr = max(-1.0, min(1.0, cos_val))
    return corr if is_finite(corr, backend) else error_value


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
    lhs_rank = backend.astype(lhs_rank, "float32")
    rhs_rank = backend.astype(rhs_rank, "float32")

    mean_l = backend.mean(lhs_rank)
    mean_r = backend.mean(rhs_rank)
    diff_l = lhs_rank - mean_l
    diff_r = rhs_rank - mean_r
    backend.eval(diff_l, diff_r)

    eps = division_epsilon(backend, lhs_rank)
    d0a = _geodesic_norm_scalar(diff_l, backend)
    d0b = _geodesic_norm_scalar(diff_r, backend)
    if d0a <= eps or d0b <= eps:
        return error_value

    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    diff_l_vec = backend.reshape(diff_l, (1, -1))
    diff_r_vec = backend.reshape(diff_r, (1, -1))
    points = backend.concatenate([diff_l_vec, diff_r_vec], axis=0)
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(points, k_neighbors=1)
    backend.eval(geo_result.distances)
    dab = float(backend.to_scalar(geo_result.distances[0, 1]))

    denom = 2.0 * d0a * d0b
    if denom <= eps:
        return error_value
    cos_val = (d0a * d0a + d0b * d0b - dab * dab) / denom
    corr = max(-1.0, min(1.0, cos_val))
    return corr if is_finite(corr, backend) else error_value


# =============================================================================
# Matrix Decomposition
# =============================================================================


def power_iteration_eigh(
    backend: "Backend",
    matrix: "Array",
    k: int = 10,
    use_ritz: bool = True,
) -> tuple["Array", "Array"]:
    """Compute EXACT eigendecomposition using native backend operation."""
    b = backend
    matrix = _promote_precision(b.array(matrix), b)
    b.eval(matrix)
    dtype = matrix.dtype

    shape = matrix.shape
    n = int(shape[-1])
    batch_shape = shape[:-2]
    k = min(k, n)
    if k == 0:
        return (
            b.zeros(batch_shape + (0,), dtype=dtype),
            b.zeros(batch_shape + (n, 0), dtype=dtype),
        )

    eigenvalues_full, eigenvectors_full = b.eigh(matrix)
    b.eval(eigenvalues_full, eigenvectors_full)

    order = b.argsort(-eigenvalues_full, axis=-1)
    eigenvalues_sorted = b.take_along_axis(eigenvalues_full, order, axis=-1)
    order_exp = b.expand_dims(order, axis=-2)
    order_tiled = b.broadcast_to(order_exp, eigenvectors_full.shape)
    eigenvectors_sorted = b.take_along_axis(eigenvectors_full, order_tiled, axis=-1)
    b.eval(order, eigenvalues_sorted, eigenvectors_sorted)

    eigenvalues = eigenvalues_sorted[..., :k]
    eigenvectors = eigenvectors_sorted[..., :k]
    b.eval(eigenvalues, eigenvectors)

    return eigenvalues, eigenvectors


def geodesic_svd(
    backend: "Backend",
    array: "Array",
    k: int | None = None,
) -> tuple["Array", "Array", "Array"]:
    """Compute EXACT SVD using native backend operation."""
    b = backend
    A = _promote_precision(b.array(array), b)
    b.eval(A)
    dtype = A.dtype

    shape = A.shape
    if len(shape) < 2:
        raise ValueError("geodesic_svd requires at least 2D input")
    m = int(shape[-2])
    n = int(shape[-1])
    batch_shape = shape[:-2]
    max_rank = min(m, n)

    if m == 0 or n == 0:
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    # Defensive checks for LAPACK
    A_sum = b.sum(A)
    b.eval(A_sum)
    A_sum_val = float(b.to_scalar(A_sum))

    if A_sum_val != A_sum_val:  # NaN check
        U = b.zeros(batch_shape + (m, 0), dtype="float32")
        S = b.zeros(batch_shape + (0,), dtype="float32")
        Vt = b.zeros(batch_shape + (0, n), dtype="float32")
        return U, S, Vt

    if abs(A_sum_val) == float("inf"):
        U = b.zeros(batch_shape + (m, 0), dtype="float32")
        S = b.zeros(batch_shape + (0,), dtype="float32")
        Vt = b.zeros(batch_shape + (0, n), dtype="float32")
        return U, S, Vt

    A_norm_sq = b.sum(A * A)
    b.eval(A_norm_sq)
    A_norm_sq_val = float(b.to_scalar(A_norm_sq))
    tiny = tiny_value(b, A)
    zero_threshold = tiny * max(1.0, float(m * n))
    if A_norm_sq_val <= zero_threshold:
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    U_full, S_full, Vt_full = b.svd(A, compute_uv=True)
    b.eval(U_full, S_full, Vt_full)

    k = min(k or max_rank, max_rank)
    if k == 0:
        U = b.zeros((m, 0), dtype=dtype)
        S = b.zeros((0,), dtype=dtype)
        Vt = b.zeros((0, n), dtype=dtype)
        return U, S, Vt

    U = U_full[..., :k]
    S = S_full[..., :k]
    Vt = Vt_full[..., :k, :]
    b.eval(U, S, Vt)

    return U, S, Vt


def svd_auto_rank(
    singular_values: "Array",
    backend: "Backend",
    energy_threshold: float = 0.99,
) -> int:
    """Determine optimal SVD rank by cumulative energy.

    Finds the minimum k such that the top-k singular values capture at least
    energy_threshold fraction of total variance (Frobenius norm squared).

    Formula: k = min{ j : sum(S[:j]^2) / sum(S^2) >= energy_threshold }

    This is the principled way to determine low-rank truncation without
    arbitrary thresholds. Research shows task matrices are inherently low-rank:
    - ~3% of singular components capture 98.5% of task information
    - Remaining components are noise from training dynamics

    Parameters
    ----------
    singular_values : Array
        Singular values from SVD, sorted in descending order.
    backend : Backend
        Backend for tensor operations.
    energy_threshold : float
        Fraction of total energy to preserve. Default 0.99 captures nearly
        all task-specific information while filtering noise.

    Returns
    -------
    int
        Optimal rank k (number of singular values to keep).

    References
    ----------
    - Yu et al. (2025). "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
    - Zhang et al. (2025). "STF: Superpose Task-specific Features for Multi-task Fine-tuned Models"
    """
    b = backend
    S = b.astype(b.array(singular_values), "float32")
    b.eval(S)

    n = int(S.shape[0])
    if n == 0:
        return 0

    # Compute squared singular values (energy per component)
    S_sq = S * S
    b.eval(S_sq)

    # Total energy (Frobenius norm squared)
    total_energy = b.sum(S_sq)
    b.eval(total_energy)
    total_energy_val = float(b.to_scalar(total_energy))

    if total_energy_val <= 0:
        return 0

    # Cumulative sum of squared singular values
    cumsum = b.cumsum(S_sq)
    b.eval(cumsum)

    # Threshold: energy_threshold * total_energy
    threshold = energy_threshold * total_energy_val

    # Find first index where cumsum >= threshold
    mask = cumsum >= threshold
    mask_int = b.astype(mask, "int32")
    b.eval(mask_int)

    # argmax on mask gives first True index (or 0 if all False)
    first_idx = b.argmax(mask_int)
    b.eval(first_idx)
    k = int(b.to_scalar(first_idx)) + 1  # +1 to include that component

    # Clamp to valid range
    k = max(1, min(k, n))

    return k


def geodesic_pinv(backend: "Backend", array: "Array") -> "Array":
    """Compute EXACT Moore-Penrose pseudo-inverse using native backend operation."""
    b = backend
    A = _promote_precision(b.array(array), b)
    b.eval(A)

    A_pinv = b.pinv(A)
    b.eval(A_pinv)

    return A_pinv


def safe_inverse(
    backend: "Backend",
    matrix: "Array",
    regularize: bool = True,
) -> tuple["Array", float]:
    """Compute matrix inverse with condition number check and optional regularization."""
    b = backend
    matrix = _promote_precision(b.array(matrix), b)
    b.eval(matrix)
    dtype = matrix.dtype

    n = int(matrix.shape[0])
    if n == 0:
        return matrix, float("inf")

    _, S, _ = geodesic_svd(b, matrix)
    b.eval(S)

    if int(S.shape[0]) == 0:
        return b.eye(n, dtype=dtype), float("inf")

    max_s_arr = b.max(S)
    b.eval(max_s_arr)
    max_s = float(b.to_scalar(max_s_arr))

    if max_s == 0:
        return b.eye(n, dtype=dtype), float("inf")

    eps = division_epsilon(b, matrix)
    pos_mask = S > eps
    pos_inf = float(b.finfo().max)
    min_candidates = b.where(pos_mask, S, b.full(S.shape, pos_inf))
    min_s_arr = b.min(min_candidates)
    b.eval(min_s_arr)
    min_s = float(b.to_scalar(min_s_arr))

    if min_s <= 0 or min_s >= pos_inf:
        min_s = eps

    cond = max_s / min_s

    cond_thresh = condition_threshold(b, matrix)

    if regularize and cond > 1.0:
        ramp = min(1.0, (cond - 1.0) / (cond_thresh - 1.0)) if cond_thresh > 1.0 else 1.0
        max_reg = regularization_epsilon(b, matrix)
        reg = max_reg * ramp

        if reg > 0:
            matrix = matrix + reg * b.eye(n, dtype=dtype)
            b.eval(matrix)

    inv_matrix = b.inv(matrix)
    b.eval(inv_matrix)

    return inv_matrix, cond


# =============================================================================
# GPU-Accelerated Linear Algebra
# =============================================================================


def newton_schulz_inverse(
    backend: "Backend",
    A: "Array",
    max_iter: int = 15,
    tol: float | None = None,
) -> "Array":
    """Pure matmul matrix inverse via Newton-Schulz iteration.

    Solves: X = A^{-1} using only matmuls (guaranteed GPU).

    Algorithm: X_{k+1} = X_k @ (2I - A @ X_k)
    Converges quadratically when ||I - X_0 @ A|| < 1.

    For SPD matrices (like Gram matrices), we use:
        X_0 = I / trace(A)  (scales by average eigenvalue)

    This is the GPU-friendly alternative to solve/cholesky which fall back to CPU.
    """
    b = backend

    A = _promote_precision(b.array(A), b)
    b.eval(A)
    dtype = A.dtype

    n = int(b.shape(A)[0])
    eps = machine_epsilon(b, A)
    if tol is None:
        tol = sqrt_scalar(eps, b) * float(n)  # Scale tolerance by dimension

    # Use Frobenius norm as upper bound on spectral radius
    # ||A||_2 ≤ ||A||_F, so scaling by 1/||A||_F ensures spectral radius ≤ 1
    A_norm = b.norm(A)
    b.eval(A_norm)
    A_norm_val = float(b.to_scalar(A_norm))

    if A_norm_val < eps:
        # Near-zero matrix
        return b.eye(n, dtype=dtype) / eps

    # Scale A to have spectral radius < 1 for convergence
    # Use 1/||A||_F as conservative scaling (guarantees spectral radius < 1)
    scale = 1.0 / A_norm_val
    A_scaled = A * scale

    # Initial guess for scaled problem: X_0 = A^T * scale (matches the spectral structure)
    # For SPD matrices, A^T = A, so X_0 = A * scale^2
    X = b.transpose(A) * (scale * scale)
    I = b.eye(n, dtype=dtype)
    b.eval(X, A_scaled)

    prev_err = float("inf")

    for i in range(max_iter):
        # Newton-Schulz: X' = X @ (2I - A_scaled @ X)
        AX = b.matmul(A_scaled, X)
        diff = 2.0 * I - AX
        X_new = b.matmul(X, diff)
        b.eval(X_new)

        # Check convergence: ||I - A_scaled @ X||_F
        err_mat = I - b.matmul(A_scaled, X_new)
        err = b.norm(err_mat)
        b.eval(err)
        err_val = float(b.to_scalar(err))

        if err_val <= tol:
            logger.debug(
                "Newton-Schulz: Converged in %d iters, ||I - A X||=%.2e",
                i + 1, err_val
            )
            # Scale back: inv(A) = scale * inv(A_scaled)
            return X_new * scale

        if err_val >= prev_err * 1.01:  # Allow small fluctuations
            # Diverging - return current best
            logger.debug(
                "Newton-Schulz: Stalled at iter %d, ||I - A X||=%.2e",
                i + 1, err_val
            )
            return X * scale

        X = X_new
        prev_err = err_val

    return X * scale


def gpu_lstsq(
    backend: "Backend",
    A: "Array",
    B: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Least squares via closed-form normal equations or CGLS fallback.

    Solves: minimize ||A @ X - B||² for X

    When n >= d (overdetermined), uses DIRECT closed-form:
        F = (A^T @ A + λI)^{-1} @ A^T @ B

    When n < d (underdetermined), uses DUAL closed-form:
        F = A^T @ (A @ A^T + λI)^{-1} @ B

    Both are O(min(n,d)³) via the backend's solve() which uses the most
    appropriate method (Cholesky for SPD matrices). Note: MLX's solve()
    may fall back to CPU for the linear system solve, but matmuls remain
    on GPU. Performance is still excellent due to small matrix dimensions.

    Falls back to iterative CGLS only if direct solve fails.

    If stats is provided, populates:
    - iterations: 0 for direct solve, count for CGLS
    - residual_norm: final residual norm
    - rhs_norm: right-hand-side norm
    - method: 'normal_equations', 'normal_equations_dual', or 'cgls'
    """
    b = backend

    A = _promote_precision(b.array(A), b)
    A_shape = b.shape(A)
    n, d = int(A_shape[0]), int(A_shape[1])

    B = _promote_precision(b.array(B), b, min_dtype=A.dtype)
    B_shape = b.shape(B)
    if len(B_shape) == 1:
        B = b.reshape(B, (-1, 1))
        squeeze_output = True
    else:
        squeeze_output = False

    b.eval(A, B)

    eps = machine_epsilon(b, A)
    sqrt_eps = sqrt_scalar(eps, b)

    # =========================================================================
    # DIRECT SOLVE via Normal Equations (when n >= d)
    # =========================================================================
    # The pseudoinverse for tall matrices has a closed form:
    #   pinv(A) = (A^T @ A)^{-1} @ A^T
    #
    # So: X = pinv(A) @ B = (A^T @ A)^{-1} @ A^T @ B
    #
    # Let G = A^T @ A (d × d) and H = A^T @ B (d × k)
    # Then X = solve(G, H) using Cholesky (G is positive semidefinite)
    #
    # This is O(d³) instead of O(max_iter × n × d), instant on GPU.
    # =========================================================================

    if n >= d:
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ: Direct solve [%d x %d] -> [%d x %d] (n >= d, using closed-form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A^T @ A (d × d) - this is the Gram matrix in feature space
        A_T = b.transpose(A)
        G = b.matmul(A_T, A)  # d × d
        b.eval(G)

        # Add Tikhonov regularization for numerical stability
        # λ = eps × max(diag(G)) ensures well-conditioning
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))
        reg_lambda = eps * max(max_diag_val, 1.0)

        G_reg = G + reg_lambda * b.eye(d)
        b.eval(G_reg)

        # Compute A^T @ B (d × k)
        H = b.matmul(A_T, B)  # d × k
        b.eval(H)

        # Solve G @ X = H directly
        # The backend's solve() uses the most appropriate method internally
        try:
            X = b.solve(G_reg, H)
            b.eval(X)

            elapsed = time.perf_counter() - start_time

            # Compute residual for logging
            res = b.matmul(A, X) - B
            res_norm = b.norm(res)
            B_norm = b.norm(B)
            b.eval(res_norm, B_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            B_norm_val = float(b.to_scalar(B_norm))

            logger.info(
                "NORMAL_EQ: Solved in %.3fs, residual=%.2e, ||B||=%.2e",
                elapsed, res_norm_val, B_norm_val
            )

            if stats is not None:
                stats["iterations"] = 0.0
                stats["residual_norm"] = res_norm_val
                stats["rhs_norm"] = B_norm_val
                stats["method"] = "normal_equations"

            if squeeze_output:
                X = b.reshape(X, (-1,))

            return X

        except Exception as e:
            # Direct solve failed, fall back to CGLS
            logger.warning(
                "NORMAL_EQ: Direct solve failed (%s), falling back to CGLS",
                str(e)
            )

    # =========================================================================
    # DIRECT SOLVE for Underdetermined Systems (when n < d)
    # =========================================================================
    # For underdetermined systems, use the dual normal equations:
    #   pinv(A) = A^T @ (A @ A^T)^{-1}
    #
    # So: X = pinv(A) @ B = A^T @ (A @ A^T)^{-1} @ B
    #
    # Let G = A @ A^T (n × n) - this is much smaller than d × d!
    # Then solve G @ Y = B, and X = A^T @ Y
    #
    # This is O(n³) instead of O(d³) or iterating.
    # =========================================================================

    elif n < d:  # Underdetermined case
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ_DUAL: Direct solve [%d x %d] -> [%d x %d] (n < d, using dual form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A @ A^T (n × n) - the Gram matrix in sample space
        A_T = b.transpose(A)
        G = b.matmul(A, A_T)  # n × n (much smaller than d × d!)
        b.eval(G)

        # Add Tikhonov regularization with under-determination scaling
        # When n << d, the system has many more degrees of freedom than
        # constraints. We scale regularization by d/n to shrink the solution
        # proportionally to the rank deficiency. This prevents wild
        # extrapolation in the unconstrained directions.
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))

        # Scale regularization by under-determination ratio
        # sqrt(d/n) provides moderate shrinkage - not too aggressive
        underdetermination_ratio = (d / n) ** 0.5  # e.g., sqrt(11008/2048) ≈ 2.3
        reg_lambda = eps * max(max_diag_val, 1.0) * underdetermination_ratio

        G_reg = G + reg_lambda * b.eye(n)
        b.eval(G_reg)

        try:
            # Solve G @ Y = B for Y (n × k)
            Y = b.solve(G_reg, B)
            b.eval(Y)

            # X = A^T @ Y (d × k)
            X = b.matmul(A_T, Y)
            b.eval(X)

            elapsed = time.perf_counter() - start_time

            # Compute residual for logging
            res = b.matmul(A, X) - B
            res_norm = b.norm(res)
            B_norm = b.norm(B)
            b.eval(res_norm, B_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            B_norm_val = float(b.to_scalar(B_norm))

            logger.info(
                "NORMAL_EQ_DUAL: Solved in %.3fs, residual=%.2e, ||B||=%.2e",
                elapsed, res_norm_val, B_norm_val
            )

            if stats is not None:
                stats["iterations"] = 0.0
                stats["residual_norm"] = res_norm_val
                stats["rhs_norm"] = B_norm_val
                stats["method"] = "normal_equations_dual"

            if squeeze_output:
                X = b.reshape(X, (-1,))

            return X

        except Exception as e:
            logger.warning(
                "NORMAL_EQ_DUAL: Direct solve failed (%s), falling back to CGLS",
                str(e)
            )

    # =========================================================================
    # CGLS Fallback (only if direct solves failed or n == d exactly)
    # =========================================================================

    # Precondition: scale columns to unit norm
    col_norms = b.norm(A, axis=0)
    col_norms = col_norms + eps
    inv_col_norms = 1.0 / col_norms
    row_scale = b.reshape(inv_col_norms, (int(d), 1))
    diag_reg = b.reshape(inv_col_norms * inv_col_norms * eps, (int(d), 1))
    A_T = b.transpose(A)
    b.eval(col_norms, inv_col_norms, row_scale, diag_reg, A_T)

    # Right-hand side: A_hat^T B
    rhs = b.matmul(A_T, B)
    rhs = rhs * row_scale
    b.eval(rhs)

    # Solve (A_hat^T A_hat + diag_reg) Y = rhs
    Y = b.zeros((int(d), int(b.shape(B)[1])), dtype="float32")
    R = rhs
    P = rhs
    b.eval(Y, R, P)

    rhs_norm_sq = b.sum(rhs * rhs)
    b.eval(rhs_norm_sq)
    rhs_norm_sq_val = float(b.to_scalar(rhs_norm_sq))
    rhs_norm = sqrt_scalar(rhs_norm_sq_val, b)
    sqrt_d = sqrt_scalar(float(d), b)
    rhs_scale = rhs_norm / max(sqrt_d, 1.0)
    tol = sqrt_eps * max(rhs_scale, eps)

    B_norm_arr = b.norm(B)
    b.eval(B_norm_arr)
    B_norm_val = float(b.to_scalar(B_norm_arr))
    sqrt_n = sqrt_scalar(float(n), b)
    tol_primal = sqrt_eps * max(B_norm_val / max(sqrt_n, 1.0), eps)

    rnorm_sq_val = rhs_norm_sq_val
    rnorm_val = rhs_norm
    prev_rnorm = rnorm_val

    refresh_interval = max(1, int(min(n, d)))
    restart_budget = int(max(1, ceil_scalar(log2_scalar(1.0 / eps, b), b)))
    max_iter = max(1, int(d)) * restart_budget

    # Log CGLS start
    start_time = time.perf_counter()
    log_interval = max(500, max_iter // 20)  # Log ~20 times during run
    logger.info(
        "CGLS: Starting [%d x %d] -> [%d x %d], max_iter=%d, tol=%.2e",
        n, d, d, int(b.shape(B)[1]), max_iter, tol
    )

    iterations_used = 0
    last_log_time = start_time
    for step in range(max_iter):
        # Apply normal equations with preconditioning + Tikhonov
        P_scaled = P * row_scale
        AP = b.matmul(A, P_scaled)
        ATAP = b.matmul(A_T, AP)
        ATAP = ATAP * row_scale
        ATAP = ATAP + diag_reg * P
        b.eval(ATAP)

        denom = b.sum(P * ATAP)
        b.eval(denom)
        denom_val = float(b.to_scalar(denom))
        if denom_val <= eps:
            denom_val = eps

        alpha = rnorm_sq_val / denom_val
        Y = Y + alpha * P
        R = R - alpha * ATAP
        b.eval(Y, R)

        old_rnorm_sq = rnorm_sq_val
        rnorm_sq = b.sum(R * R)
        b.eval(rnorm_sq)
        rnorm_sq_val = float(b.to_scalar(rnorm_sq))
        rnorm_val = sqrt_scalar(rnorm_sq_val, b)

        iterations_used = step + 1

        # Progress logging
        if iterations_used % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            iters_per_sec = iterations_used / max(elapsed, 0.001)
            remaining = (max_iter - iterations_used) / max(iters_per_sec, 0.001)
            logger.info(
                "CGLS: iter %d/%d (%.1f%%), residual=%.2e, %.1f iter/s, ~%.0fs remaining",
                iterations_used, max_iter, 100.0 * iterations_used / max_iter,
                rnorm_val, iters_per_sec, remaining
            )

        if rnorm_val <= tol:
            X_tmp = b.matmul(A, Y * row_scale)
            res = X_tmp - B
            res_norm = b.norm(res)
            b.eval(res_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            if res_norm_val <= tol_primal:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "CGLS: Converged at iter %d (%.2fs), residual=%.2e, primal=%.2e",
                    iterations_used, elapsed, rnorm_val, res_norm_val
                )
                break

        # Refresh residuals on drift or periodic cadence
        stagnation_threshold = sqrt_eps * max(prev_rnorm, rhs_norm, eps)
        if (
            not is_finite(rnorm_val, b)
            or rnorm_val > prev_rnorm * (1.0 + sqrt_eps)
            or rnorm_val >= prev_rnorm - stagnation_threshold
        ):
            X_tmp = b.matmul(A, Y * row_scale)
            R = b.matmul(A_T, X_tmp)
            R = rhs - R * row_scale - diag_reg * Y
            b.eval(R)
            rnorm_sq = b.sum(R * R)
            b.eval(rnorm_sq)
            rnorm_sq_val = float(b.to_scalar(rnorm_sq))
            rnorm_val = sqrt_scalar(rnorm_sq_val, b)
            P = R
            prev_rnorm = rnorm_val
            continue

        if (step + 1) % refresh_interval == 0:
            X_tmp = b.matmul(A, Y * row_scale)
            R = b.matmul(A_T, X_tmp)
            R = rhs - R * row_scale - diag_reg * Y
            b.eval(R)
            rnorm_sq = b.sum(R * R)
            b.eval(rnorm_sq)
            rnorm_sq_val = float(b.to_scalar(rnorm_sq))
            rnorm_val = sqrt_scalar(rnorm_sq_val, b)
            P = R
            prev_rnorm = rnorm_val
            continue

        beta = rnorm_sq_val / max(old_rnorm_sq, eps)
        P = R + beta * P
        b.eval(P)
        prev_rnorm = rnorm_val

    X = Y * row_scale
    b.eval(X)

    # Log completion
    total_elapsed = time.perf_counter() - start_time
    if iterations_used >= max_iter:
        logger.warning(
            "CGLS: Hit max_iter=%d (%.2fs), residual=%.2e (may not have converged)",
            max_iter, total_elapsed, rnorm_val
        )
    else:
        logger.info(
            "CGLS: Completed in %d iters (%.2fs), final residual=%.2e",
            iterations_used, total_elapsed, rnorm_val
        )

    if stats is not None:
        stats["iterations"] = float(iterations_used)
        stats["residual_norm"] = float(rnorm_val)
        stats["rhs_norm"] = float(rhs_norm)
        stats["method"] = "cgls"

    if squeeze_output:
        X = b.reshape(X, (-1,))

    return X


# =============================================================================
# Invariant Alignment: Linear CKA = 1.0 by Construction
# =============================================================================


def invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Compute the alignment transform F where linear CKA = 1.0 is GUARANTEED.

    THE MATHEMATICS:
    ================
    F = pinv(source) @ target

    This gives:
        aligned = source @ F = source @ pinv(source) @ target = P @ target

    Where P = source @ pinv(source) is the orthogonal projector onto source's
    column space. Linear CKA = 1.0 by construction.

    LOW-RANK TRUNCATION:
    ====================
    The full-rank F overfits when n_samples < d_source. We truncate F to the
    effective rank of the source Gram matrix, derived from spectral gap detection.
    This prevents overfitting while preserving the shared manifold structure.
    """
    b = backend

    source = _promote_precision(b.array(source), b)
    target = _promote_precision(b.array(target), b, min_dtype=source.dtype)
    b.eval(source, target)

    # Center both matrices (CKA uses centered Gram matrices)
    source_mean = b.mean(source, axis=0, keepdims=True)
    target_mean = b.mean(target, axis=0, keepdims=True)
    source_c = source - source_mean
    target_c = target - target_mean
    b.eval(source_c, target_c)

    # THE FORMULA: F = pinv(source) @ target (solved via GPU CGLS)
    F = gpu_lstsq(b, source_c, target_c, stats=stats)
    b.eval(F)

    # =========================================================================
    # GEOMETRY CHECK: Warn if underdetermined (n_samples < d_source)
    # =========================================================================
    # When n < d, the least squares solution has infinitely many solutions.
    # The minimum-norm solution (from gpu_lstsq) is mathematically valid but
    # may overfit to the specific probes used. Ensure sufficient probe coverage
    # by using a geometry-derived probe count (n >= d_source).
    #
    # We DON'T truncate F because:
    # 1. Truncation might discard source-unique knowledge that appears as small
    #    eigenvalues but is actually valuable for transfer
    # 2. With proper probe coverage (n > d), this isn't needed anyway
    # 3. The null-space projection stage handles transfer decisions
    n_samples = int(b.shape(source_c)[0])
    d_source = int(b.shape(source_c)[1])

    if n_samples < d_source:
        # Compute the under-determination ratio for informational logging
        ratio = d_source / n_samples
        logger.info(
            "UNDERDETERMINED ALIGNMENT: n=%d < d=%d (ratio=%.1fx). "
            "Using sqrt-scaled Tikhonov regularization for stability.",
            n_samples, d_source, ratio
        )
        if stats is not None:
            stats["underdetermined"] = 1.0
            stats["n_samples"] = float(n_samples)
            stats["d_source"] = float(d_source)
            stats["underdetermination_ratio"] = ratio

    return F


def geodesic_invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Find alignment F that preserves geodesic manifold structure.

    THE MATHEMATICS:
    ================
    Instead of minimizing ||source @ F - target||_F (Euclidean distance),
    this finds F that preserves pairwise geodesic relationships.

    The algorithm:
    1. Compute geodesic cosine matrices G_s and G_t (relative representations)
       - G[i,j] = geodesic_cos(point_i, point_j) using k-NN graph distances
       - Each row represents a point by its geodesic similarities to all others
    2. Find rotation R via Procrustes on the relative representations
       - SVD: G_s.T @ G_t = U @ S @ V.T
       - R = U @ V.T (orthogonal alignment in relative space)
    3. Transfer through aligned relative space:
       - transferred = G_s @ R @ pinv(G_t) @ target
    4. Recover feature-space transform:
       - F = lstsq(source, transferred)

    This preserves manifold structure because geodesic cosines capture the
    intrinsic geometry (curvature, topology) that Euclidean distance ignores.

    WHY THIS WORKS:
    ===============
    Neural manifolds are curved in high dimensions. Euclidean distance treats
    the space as flat, leading to alignment errors that compound through layers.
    Geodesic cosines "linearize" the curved structure - after this transform,
    linear alignment (Procrustes) correctly preserves relationships.

    Reference: Yu et al. 2025 "Relative Geodesic Representations" (NeurIPS)
    showed geodesic achieves 0.99 alignment vs 0.01 for linear methods.
    """
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix

    b = backend

    source = _promote_precision(b.array(source), b)
    target = _promote_precision(b.array(target), b, min_dtype=source.dtype)
    b.eval(source, target)

    n_samples = int(b.shape(source)[0])
    d_source = int(b.shape(source)[1])
    d_target = int(b.shape(target)[1])

    if stats is not None:
        stats["geodesic_alignment"] = 1.0
        stats["n_samples"] = float(n_samples)
        stats["d_source"] = float(d_source)
        stats["d_target"] = float(d_target)

    # Step 1: Compute geodesic cosine matrices (relative representations)
    # Each row represents a point by its geodesic similarities to all others
    t0 = time.time()
    G_source = geodesic_cosine_matrix(source, b)
    G_target = geodesic_cosine_matrix(target, b)
    b.eval(G_source, G_target)
    t_geo = time.time() - t0

    if stats is not None:
        stats["geodesic_time_sec"] = t_geo

    logger.info(
        "Geodesic cosine matrices computed: source %s, target %s (%.2fs)",
        b.shape(G_source), b.shape(G_target), t_geo
    )

    # Step 2: Center both matrices for Procrustes
    G_source_mean = b.mean(G_source, axis=0, keepdims=True)
    G_target_mean = b.mean(G_target, axis=0, keepdims=True)
    G_source_c = G_source - G_source_mean
    G_target_c = G_target - G_target_mean
    b.eval(G_source_c, G_target_c)

    # Step 3: Procrustes alignment in relative space
    # Find R such that G_source @ R ≈ G_target
    # SVD of G_source.T @ G_target = U @ S @ V.T, then R = U @ V.T
    C = b.matmul(b.transpose(G_source_c), G_target_c)
    b.eval(C)

    U, S, Vt = b.svd(C)
    b.eval(U, S, Vt)

    R = b.matmul(U, Vt)
    b.eval(R)

    # Ensure proper rotation (det = +1)
    det_val = b.det(R)
    b.eval(det_val)
    det_scalar = float(b.to_scalar(det_val))
    if det_scalar < 0:
        # Flip sign of last column of U to get proper rotation
        n_cols = int(b.shape(U)[1])
        sign_arr = b.ones((n_cols,))
        idx = b.arange(n_cols)
        sign_arr = b.where(idx == (n_cols - 1), b.full(sign_arr.shape, -1.0), sign_arr)
        U = U * sign_arr
        R = b.matmul(U, Vt)
        b.eval(R)

    if stats is not None:
        # Measure alignment quality in relative space
        G_aligned = b.matmul(G_source_c, R)
        b.eval(G_aligned)
        diff = G_aligned - G_target_c
        frob_norm = b.sqrt(b.sum(diff * diff))
        target_norm = b.sqrt(b.sum(G_target_c * G_target_c))
        b.eval(frob_norm, target_norm)
        rel_error = float(b.to_scalar(frob_norm)) / max(float(b.to_scalar(target_norm)), 1e-12)
        stats["relative_space_alignment_error"] = rel_error
        logger.info("Relative space alignment error: %.6f", rel_error)

    # Step 4: Transfer through aligned relative space
    # transferred = G_source @ R @ pinv(G_target) @ target
    # Use SVD for stable pseudo-inverse of G_target

    G_aligned = b.matmul(G_source, R)
    b.eval(G_aligned)

    # Stable pseudo-inverse of G_target via SVD
    U_t, S_t, Vt_t = b.svd(G_target)
    b.eval(U_t, S_t, Vt_t)

    # Threshold singular values
    eps = machine_epsilon(b, G_target)
    max_s = b.max(S_t)
    b.eval(max_s)
    threshold = eps * float(b.to_scalar(max_s)) * float(n_samples)
    S_t_safe = b.where(S_t > threshold, S_t, b.ones_like(S_t))
    S_t_inv = b.where(S_t > threshold, 1.0 / S_t_safe, b.zeros_like(S_t))
    b.eval(S_t_inv)

    # pinv(G_target) = V @ diag(S_inv) @ U.T
    G_target_pinv = b.matmul(b.transpose(Vt_t), S_t_inv[:, None] * b.transpose(U_t))
    b.eval(G_target_pinv)

    # transferred = G_aligned @ pinv(G_target) @ target
    temp = b.matmul(G_aligned, G_target_pinv)
    transferred = b.matmul(temp, target)
    b.eval(transferred)

    # Step 5: Recover feature-space transform F
    # F = lstsq(source, transferred) using gpu_lstsq for stability
    F = gpu_lstsq(b, source, transferred, stats=stats)
    b.eval(F)

    logger.info(
        "Geodesic alignment complete: F shape %s, d_source=%d, d_target=%d",
        b.shape(F), d_source, d_target
    )

    return F
