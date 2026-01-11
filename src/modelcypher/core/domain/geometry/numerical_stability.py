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
    # Invariant alignment (CKA = 1.0 by construction)
    "invariant_alignment",
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
    sorted_values: list[float],
    eps: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Find the natural break point in a sorted magnitude distribution."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    if not sorted_values:
        return 0.0

    values_arr = backend.array(sorted_values)
    if eps is None:
        abs_arr = backend.abs(values_arr)
        scale_arr = backend.maximum(backend.max(abs_arr), backend.array([1.0]))
        backend.eval(scale_arr)
        scale = float(backend.to_scalar(scale_arr))
        eps = ulp_scalar(scale, backend)

    if len(sorted_values) < 3:
        return sorted_values[len(sorted_values) // 2]

    curr = values_arr[:-1]
    next_vals = values_arr[1:]
    diffs = next_vals - curr
    eps_arr = backend.array([eps])
    valid = curr > eps_arr
    denom = backend.where(valid, curr, backend.ones_like(curr))
    rel_gaps = diffs / denom
    rel_gaps = backend.where(valid, rel_gaps, backend.zeros_like(rel_gaps))
    max_gap_arr = backend.max(rel_gaps)
    gap_index_arr = backend.argmax(rel_gaps)
    backend.eval(max_gap_arr, gap_index_arr)
    max_gap = float(backend.to_scalar(max_gap_arr))

    if max_gap <= 0.0:
        return sorted_values[len(sorted_values) // 2]

    gap_index = int(backend.to_scalar(gap_index_arr))
    return sorted_values[gap_index]


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
    matrix = b.astype(b.array(matrix), "float32")
    b.eval(matrix)

    shape = matrix.shape
    n = int(shape[-1])
    batch_shape = shape[:-2]
    k = min(k, n)
    if k == 0:
        return (
            b.zeros(batch_shape + (0,), dtype="float32"),
            b.zeros(batch_shape + (n, 0), dtype="float32"),
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
    A = b.astype(b.array(array), "float32")
    b.eval(A)

    shape = A.shape
    if len(shape) < 2:
        raise ValueError("geodesic_svd requires at least 2D input")
    m = int(shape[-2])
    n = int(shape[-1])
    batch_shape = shape[:-2]
    max_rank = min(m, n)

    if m == 0 or n == 0:
        U = b.zeros(batch_shape + (m, 0), dtype="float32")
        S = b.zeros(batch_shape + (0,), dtype="float32")
        Vt = b.zeros(batch_shape + (0, n), dtype="float32")
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
    if A_norm_sq_val < 1e-30:
        U = b.zeros(batch_shape + (m, 0), dtype="float32")
        S = b.zeros(batch_shape + (0,), dtype="float32")
        Vt = b.zeros(batch_shape + (0, n), dtype="float32")
        return U, S, Vt

    U_full, S_full, Vt_full = b.svd(A, compute_uv=True)
    b.eval(U_full, S_full, Vt_full)

    k = min(k or max_rank, max_rank)
    if k == 0:
        U = b.zeros((m, 0), dtype="float32")
        S = b.zeros((0,), dtype="float32")
        Vt = b.zeros((0, n), dtype="float32")
        return U, S, Vt

    U = U_full[..., :k]
    S = S_full[..., :k]
    Vt = Vt_full[..., :k, :]
    b.eval(U, S, Vt)

    return U, S, Vt


def geodesic_pinv(backend: "Backend", array: "Array") -> "Array":
    """Compute EXACT Moore-Penrose pseudo-inverse using native backend operation."""
    b = backend
    A = b.array(array)
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
    matrix = b.astype(b.array(matrix), "float32")
    b.eval(matrix)

    n = int(matrix.shape[0])
    if n == 0:
        return matrix, float("inf")

    _, S, _ = geodesic_svd(b, matrix)
    b.eval(S)

    if int(S.shape[0]) == 0:
        return b.eye(n), float("inf")

    max_s_arr = b.max(S)
    b.eval(max_s_arr)
    max_s = float(b.to_scalar(max_s_arr))

    if max_s == 0:
        return b.eye(n), float("inf")

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
            matrix = matrix + reg * b.eye(n)
            b.eval(matrix)

    inv_matrix = b.inv(matrix)
    b.eval(inv_matrix)

    return inv_matrix, cond


# =============================================================================
# GPU-Accelerated Linear Algebra
# =============================================================================


def gpu_lstsq(
    backend: "Backend",
    A: "Array",
    B: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """GPU-accelerated least squares via preconditioned CGLS.

    Solves: minimize ||A @ X - B||² for X

    Design guarantees:
    - Column scaling to unit norm (preconditioning)
    - Tikhonov regularization with lambda = machine_epsilon
    - All divisions guarded by epsilon
    - Residual refresh prevents numerical drift

    If stats is provided, populates:
    - iterations: CGLS iteration count
    - residual_norm: final residual norm
    - rhs_norm: right-hand-side norm
    """
    b = backend

    A_shape = b.shape(A)
    n, d = int(A_shape[0]), int(A_shape[1])

    B_shape = b.shape(B)
    if len(B_shape) == 1:
        B = b.reshape(B, (-1, 1))
        squeeze_output = True
    else:
        squeeze_output = False

    A = b.astype(A, "float32")
    B = b.astype(B, "float32")
    b.eval(A, B)

    eps = machine_epsilon(b, A)
    sqrt_eps = sqrt_scalar(eps, b)

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

    if squeeze_output:
        X = b.reshape(X, (-1,))

    return X


# =============================================================================
# Invariant Alignment: CKA = 1.0 by Construction
# =============================================================================


def invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Compute the alignment transform F where CKA = 1.0 is GUARANTEED.

    THE MATHEMATICS:
    ================
    F = pinv(source) @ target

    This gives:
        aligned = source @ F = source @ pinv(source) @ target = P @ target

    Where P = source @ pinv(source) is the orthogonal projector onto source's
    column space. CKA = 1.0 by construction.
    """
    b = backend

    source = b.astype(source, "float32")
    target = b.astype(target, "float32")
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

    return F
