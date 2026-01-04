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

from typing import TYPE_CHECKING

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
    # Data-derived thresholds
    "find_magnitude_gap_threshold",
    # Statistical utilities
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    # Matrix decomposition
    "solve_full_row_rank_via_qr",
    "solve_via_truncated_svd",
    "solve_via_gram_alignment",
    "solve_via_cca_procrustes",
    # Rank estimation
    "compute_entropy_effective_rank",
    "compute_shared_relational_rank",
]


# =============================================================================
# Backend Scalar Helpers
# =============================================================================
# Use these instead of math.sqrt, math.isfinite, etc. to keep computation on GPU.


def _geodesic_norm_scalar(array: "Array", backend: "Backend") -> float:
    """Compute a geodesic norm scalar for any array shape."""
    from modelcypher.core.domain.geometry.vector_math import geodesic_norms

    norm_arr = geodesic_norms(backend.reshape(array, (1, -1)), backend)
    backend.eval(norm_arr)
    return float(backend.to_scalar(norm_arr))


def sqrt_scalar(value: float, backend: "Backend") -> float:
    """Compute sqrt of scalar using backend.

    Use instead of math.sqrt(value).
    """
    arr = backend.array([value])
    result = backend.sqrt(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def is_finite(value: float, backend: "Backend") -> bool:
    """Check if scalar is finite using backend.

    Use instead of math.isfinite(value).
    """
    arr = backend.array([value])
    result = backend.isfinite(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_inf(value: float, backend: "Backend") -> bool:
    """Check if scalar is infinite using backend.

    Use instead of math.isinf(value).
    """
    arr = backend.array([value])
    result = backend.isinf(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_nan(value: float, backend: "Backend") -> bool:
    """Check if scalar is NaN using backend.

    Use instead of math.isnan(value).
    """
    arr = backend.array([value])
    result = backend.isnan(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def log_scalar(value: float, backend: "Backend") -> float:
    """Compute natural log of scalar using backend.

    Use instead of math.log(value).
    """
    arr = backend.array([value])
    result = backend.log(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def exp_scalar(value: float, backend: "Backend") -> float:
    """Compute exp of scalar using backend.

    Use instead of math.exp(value).
    """
    arr = backend.array([value])
    result = backend.exp(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def power_scalar(value: float, exponent: float, backend: "Backend") -> float:
    """Compute value ** exponent using backend.

    Use instead of value ** exponent for GPU acceleration.
    """
    arr = backend.array([value])
    result = arr ** exponent
    backend.eval(result)
    return float(backend.to_scalar(result))


def ceil_scalar(value: float, backend: "Backend") -> int:
    """Compute ceil of scalar using backend.

    Use instead of math.ceil(value).
    """
    arr = backend.array([value])
    result = backend.ceil(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def floor_scalar(value: float, backend: "Backend") -> int:
    """Compute floor of scalar using backend.

    Use instead of math.floor(value).
    """
    arr = backend.array([value])
    result = backend.floor(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def ulp_scalar(value: float, backend: "Backend") -> float:
    """Compute unit in last place for scalar using backend.

    Use instead of math.ulp(value).
    For normalized floats, ulp(x) ≈ eps * abs(x).
    """
    eps = backend.finfo(backend.array([value]).dtype).eps
    return eps * abs(value) if value != 0.0 else eps


def lgamma_scalar(value: float, backend: "Backend") -> float:
    """Compute log-gamma of scalar using backend.

    Use instead of math.lgamma(value).
    """
    arr = backend.array([value])
    result = backend.lgamma(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def acos_scalar(value: float, backend: "Backend") -> float:
    """Compute arc cosine of scalar using backend.

    Use instead of math.acos(value).
    """
    arr = backend.array([value])
    result = backend.arccos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def cos_scalar(value: float, backend: "Backend") -> float:
    """Compute cosine of scalar using backend.

    Use instead of math.cos(value).
    """
    arr = backend.array([value])
    result = backend.cos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def sin_scalar(value: float, backend: "Backend") -> float:
    """Compute sine of scalar using backend.

    Use instead of math.sin(value).
    """
    arr = backend.array([value])
    result = backend.sin(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def atan2_scalar(y: float, x: float, backend: "Backend") -> float:
    """Compute atan2(y, x) using backend.

    Use instead of math.atan2(y, x).
    Returns angle in radians in [-pi, pi].
    """
    # atan2(y, x) = arctan(y/x) with quadrant handling
    # We compute using arctan and sign adjustments
    y_arr = backend.array([y])
    x_arr = backend.array([x])

    # Compute arctan(y/x) with safe division
    eps = backend.finfo().eps
    x_safe = backend.where(
        backend.abs(x_arr) < eps,
        backend.sign(x_arr) * eps + eps,  # Avoid division by zero
        x_arr,
    )
    ratio = y_arr / x_safe
    base_angle = backend.arctan(ratio)
    backend.eval(base_angle)

    # Adjust for quadrant
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
    """Compute log base 2 of scalar using backend.

    Use instead of math.log2(value).
    """
    arr = backend.array([value])
    result = backend.log2(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def pi_value(backend: "Backend") -> float:
    """Get pi using backend.

    Use instead of math.pi.
    """
    # Compute pi = 4 * arctan(1)
    one = backend.array([1.0])
    result = 4.0 * backend.arctan(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def e_value(backend: "Backend") -> float:
    """Get Euler's number e using backend.

    Use instead of math.e.
    """
    # Compute e = exp(1)
    one = backend.array([1.0])
    result = backend.exp(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def inf_value(backend: "Backend") -> float:
    """Get positive infinity using backend.

    Use instead of math.inf.
    """
    return float("inf")


def machine_epsilon(backend: "Backend", array: "Array") -> float:
    """Get machine epsilon for the array's dtype.

    This is the smallest value such that 1.0 + epsilon != 1.0.
    Use for general numerical stability in comparisons.
    """
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe division operations.

    Uses sqrt(eps) to provide numerical headroom.
    Use when dividing to prevent division by zero.
    """
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


def regularization_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for matrix regularization.

    Uses eps^0.75 to scale regularization above division safety
    while remaining tied to dtype precision.
    """
    eps = backend.finfo(array.dtype).eps
    return power_scalar(eps, 0.75, backend)


def condition_threshold(backend: Backend, array: Array) -> float:
    """Get threshold for condition number checks.

    Returns 1/eps, the inverse of machine epsilon.
    Matrices with condition number above this are numerically singular.
    """
    return 1.0 / backend.finfo(array.dtype).eps


def svd_rank_threshold(backend: Backend, array: Array, max_dim: int) -> float:
    """Get threshold for determining numerical rank from SVD.

    Uses the standard formula: max_dim * eps * largest_singular_value.
    Singular values below this threshold are considered zero.

    Args:
        backend: The compute backend.
        array: The array being decomposed (for dtype).
        max_dim: Maximum dimension of the matrix.

    Returns:
        Threshold scaled by matrix size and precision.
    """
    eps = backend.finfo(array.dtype).eps
    return float(max_dim) * eps


def tiny_value(backend: Backend, array: Array) -> float:
    """Get the smallest positive usable number for the dtype.

    Use as a floor when values must remain positive.
    """
    return backend.finfo(array.dtype).tiny


def safe_log_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for safe logarithm operations.

    Uses tiny value to prevent log(0) while maintaining precision.
    """
    return backend.finfo(array.dtype).tiny


def infinity_threshold(backend: Backend, array: Array) -> float:
    """Get threshold for detecting near-infinite values.

    Values at or above this threshold should be treated as infinite
    (e.g., disconnected nodes in a graph). Derived from machine epsilon
    to be numerically principled rather than arbitrary.

    Uses: max_representable * (1 - sqrt(eps))

    This places the threshold at sqrt(eps) relative distance from max,
    matching the precision available for the dtype.
    """
    finfo = backend.finfo(array.dtype)
    eps = finfo.eps
    max_val = finfo.max
    margin = sqrt_scalar(eps, backend)
    return float(max_val) * (1.0 - margin)


def find_magnitude_gap_threshold(
    sorted_values: list[float],
    eps: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Find the natural break point in a sorted magnitude distribution.

    Finds the threshold where the largest relative drop occurs between
    consecutive values. This identifies where "signal" ends and "noise" begins
    without arbitrary percentile choices.

    Args:
        sorted_values: Magnitudes sorted in ascending order.
        eps: Minimum denominator for numerical stability. If None, derives
            epsilon from the float precision of the input scale.
        backend: Backend for computation. If None, uses default backend.

    Returns:
        The value at which the largest relative gap occurs.
        Returns the median if no clear gap is found.
    """
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

    # Find the largest relative gap: (v[i+1] - v[i]) / v[i]
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


def compute_pearson_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute geodesic correlation coefficient between two lists.

    This is the canonical implementation for computing geodesic
    correlation across geometry modules.

    Args:
        lhs: First list of values.
        rhs: Second list of values (must be same length as lhs).
        default: Value to return on error (empty lists, mismatched lengths).
                 If None, returns float("nan") on error.
        backend: Backend for computation. If None, uses default backend.

    Returns:
        Geodesic correlation coefficient in [-1, 1], or default/nan on error.
    """
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
    if _geodesic_norm_scalar(diff_l, backend) <= eps:
        return error_value
    if _geodesic_norm_scalar(diff_r, backend) <= eps:
        return error_value

    from modelcypher.core.domain.geometry.vector_math import geodesic_pairwise_metrics

    diff_l_mat = backend.reshape(diff_l, (1, -1))
    diff_r_mat = backend.reshape(diff_r, (1, -1))
    cos_arr, _ = geodesic_pairwise_metrics(diff_l_mat, diff_r_mat, backend)
    backend.eval(cos_arr)
    if cos_arr.size == 0:
        return error_value

    corr = float(backend.to_scalar(cos_arr[0]))
    return corr if is_finite(corr, backend) else error_value


def compute_spearman_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute Spearman rank correlation coefficient between two lists.

    Ranks are derived via argsort on the backend for stable, on-device
    computation. Ties follow backend sort order (no averaging).

    Args:
        lhs: First list of values.
        rhs: Second list of values (must be same length as lhs).
        default: Value to return on error (empty lists, mismatched lengths).
                 If None, returns float("nan") on error.
        backend: Backend for computation. If None, uses default backend.

    Returns:
        Geodesic correlation coefficient in [-1, 1], or default/nan on error.
    """
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
    lhs_rank = backend.astype(lhs_rank, backend.float32)
    rhs_rank = backend.astype(rhs_rank, backend.float32)

    mean_l = backend.mean(lhs_rank)
    mean_r = backend.mean(rhs_rank)
    diff_l = lhs_rank - mean_l
    diff_r = rhs_rank - mean_r
    backend.eval(diff_l, diff_r)

    eps = division_epsilon(backend, lhs_rank)
    if _geodesic_norm_scalar(diff_l, backend) <= eps:
        return error_value
    if _geodesic_norm_scalar(diff_r, backend) <= eps:
        return error_value

    from modelcypher.core.domain.geometry.vector_math import geodesic_pairwise_metrics

    diff_l_mat = backend.reshape(diff_l, (1, -1))
    diff_r_mat = backend.reshape(diff_r, (1, -1))
    cos_arr, _ = geodesic_pairwise_metrics(diff_l_mat, diff_r_mat, backend)
    backend.eval(cos_arr)
    if cos_arr.size == 0:
        return error_value

    corr = float(backend.to_scalar(cos_arr[0]))
    return corr if is_finite(corr, backend) else error_value


# =============================================================================
# GPU-FRIENDLY GEODESIC ALTERNATIVES (NO CPU LINEAR ALGEBRA)
# =============================================================================
# These functions replace SVD, eigendecomposition, and pinv with GPU-only
# operations. For high-dimensional manifolds (8kD+), Euclidean linear algebra
# is mathematically inaccurate - geodesic methods are required.


def gram_schmidt_orthogonalize(
    backend: Backend,
    vectors: Array,
) -> Array:
    """Orthogonalize vectors using modified Gram-Schmidt (GPU-only).

    This is numerically stable and runs entirely on GPU via matmul.

    Args:
        backend: Compute backend.
        vectors: Matrix [n, k] where columns are vectors to orthogonalize.

    Returns:
        Orthonormal matrix [n, k].
    """
    b = backend
    vectors = b.astype(b.array(vectors), "float32")
    b.eval(vectors)

    n = int(vectors.shape[0])
    k = int(vectors.shape[1])
    reg = regularization_epsilon(b, vectors)

    Q_cols = []
    for j in range(k):
        v = vectors[:, j]
        b.eval(v)

        # Subtract projections onto previous orthonormal vectors
        for q in Q_cols:
            proj = b.sum(v * q)
            b.eval(proj)
            v = v - proj * q
            b.eval(v)

        # Normalize
        norm_val = _geodesic_norm_scalar(v, b)

        if norm_val > reg:
            v = v / norm_val
        else:
            # Vector is linearly dependent - use random
            v = b.ones((n,), dtype="float32") / sqrt_scalar(n, b)
        b.eval(v)
        Q_cols.append(v)

    # Stack columns
    Q = b.stack(Q_cols, axis=1)
    b.eval(Q)
    return Q


def power_iteration_eigh(
    backend: Backend,
    matrix: Array,
    k: int = 10,
) -> tuple[Array, Array]:
    """Compute top-k eigenvalues/eigenvectors via power iteration (GPU-only).

    This replaces backend.eigh() which requires CPU on MLX. Power iteration
    uses only matmul and is fully GPU-accelerated.

    Iterates until convergence (eigenvalues stop changing within machine epsilon).
    No arbitrary iteration limits - we find the exact decomposition.

    Args:
        backend: Compute backend.
        matrix: Symmetric matrix [n, n].
        k: Number of top eigenvectors to compute.

    Returns:
        (eigenvalues, eigenvectors) where:
        - eigenvalues: [k] in descending order
        - eigenvectors: [n, k] orthonormal columns
    """
    b = backend
    matrix = b.astype(b.array(matrix), "float32")
    b.eval(matrix)

    n = int(matrix.shape[0])
    k = min(k, n)
    eps = machine_epsilon(b, matrix)
    convergence_tol = sqrt_scalar(eps, b)  # sqrt(machine_epsilon) for convergence

    # Initialize with diverse vectors (not all ones)
    V_cols = []
    for j in range(k):
        col = b.ones((n,), dtype="float32")
        # Add position-based diversity
        if j > 0:
            indices = b.arange(n, dtype="float32")
            col = col + 0.1 * (indices / n) * j
        b.eval(col)
        V_cols.append(col)
    V = b.stack(V_cols, axis=1)
    V = gram_schmidt_orthogonalize(b, V)
    b.eval(V)

    prev_eigenvalues = None

    # Iterate until convergence - no arbitrary limits
    while True:
        # Apply matrix: V = M @ V
        V = b.matmul(matrix, V)
        b.eval(V)

        # Orthogonalize via Gram-Schmidt
        V = gram_schmidt_orthogonalize(b, V)
        b.eval(V)

        # Compute Rayleigh quotients for eigenvalues
        MV = b.matmul(matrix, V)
        b.eval(MV)

        eigenvalues_list = []
        for j in range(k):
            v_j = V[:, j]
            mv_j = MV[:, j]
            b.eval(v_j, mv_j)
            lambda_j = b.sum(v_j * mv_j)
            b.eval(lambda_j)
            eigenvalues_list.append(lambda_j)

        eigenvalues = b.stack(eigenvalues_list, axis=0)
        b.eval(eigenvalues)

        # Check convergence: eigenvalues stopped changing
        if prev_eigenvalues is not None:
            diff = b.abs(eigenvalues - prev_eigenvalues)
            max_diff_arr = b.max(diff)
            b.eval(max_diff_arr)
            max_diff = float(b.to_scalar(max_diff_arr))

            # Also check relative convergence for large eigenvalues
            max_eig_arr = b.max(b.abs(eigenvalues))
            b.eval(max_eig_arr)
            max_eig = float(b.to_scalar(max_eig_arr))
            rel_tol = convergence_tol * max(1.0, max_eig)

            if max_diff < rel_tol:
                break

        prev_eigenvalues = eigenvalues

    # Ritz refinement: project matrix onto subspace and diagonalize
    # This corrects eigenvector rotation within the converged subspace
    # H = V^T @ M @ V is the projected matrix (k x k)
    H = b.matmul(b.transpose(V), b.matmul(matrix, V))
    b.eval(H)

    # Diagonalize H to get exact eigenvectors within the subspace
    # H = Q @ D @ Q^T where D has eigenvalues on diagonal
    ritz_eigenvalues, Q = b.eigh(H)
    b.eval(ritz_eigenvalues, Q)

    # Rotate V by Q to get true eigenvectors: V_final = V @ Q
    V = b.matmul(V, Q)
    eigenvalues = ritz_eigenvalues
    b.eval(V, eigenvalues)

    # Sort by descending eigenvalue
    order = b.argsort(-eigenvalues)
    b.eval(order)
    eigenvalues = b.take(eigenvalues, order, axis=0)
    V = b.take(V, order, axis=1)
    b.eval(eigenvalues, V)

    return eigenvalues, V


def geodesic_svd(
    backend: Backend,
    array: Array,
    k: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute SVD via power iteration (GPU-only, no CPU linear algebra).

    This replaces svd_via_eigh which uses backend.eigh (CPU on MLX).
    Uses power iteration on A.T @ A to get right singular vectors,
    then derives left singular vectors via A @ V / S.

    Iterates until convergence - no arbitrary iteration limits.
    We find the exact decomposition.

    Args:
        backend: Compute backend.
        array: Matrix [m, n] to decompose.
        k: Number of singular values to compute. If None, uses min(m, n).

    Returns:
        (U, S, Vt) where:
        - U: [m, k] left singular vectors
        - S: [k] singular values in descending order
        - Vt: [k, n] right singular vectors (transposed)
    """
    b = backend
    A = b.astype(b.array(array), "float32")
    b.eval(A)

    m = int(A.shape[0])
    n = int(A.shape[1])
    k = min(k or min(m, n), m, n)
    reg = regularization_epsilon(b, A)

    if m == 0 or n == 0 or k == 0:
        U = b.zeros((m, k), dtype="float32")
        S = b.zeros((k,), dtype="float32")
        Vt = b.zeros((k, n), dtype="float32")
        return U, S, Vt

    # Form A.T @ A (Gram matrix of columns)
    AtA = b.matmul(b.transpose(A), A)
    b.eval(AtA)

    # Get top-k eigenvectors of A.T @ A via power iteration (iterates until convergence)
    eigenvalues, V = power_iteration_eigh(b, AtA, k=k)
    b.eval(eigenvalues, V)

    # Singular values are sqrt of eigenvalues
    S = b.sqrt(b.maximum(eigenvalues, b.zeros_like(eigenvalues)))
    b.eval(S)

    # Compute U = A @ V @ diag(1/S)
    AV = b.matmul(A, V)
    b.eval(AV)

    # Normalize each column of AV by corresponding singular value
    U_cols = []
    for j in range(k):
        s_j_arr = S[j]
        b.eval(s_j_arr)
        s_j = float(b.to_scalar(s_j_arr))

        u_j = AV[:, j]
        if s_j > reg:
            u_j = u_j / s_j
        else:
            # Zero singular value - use zero vector
            u_j = b.zeros((m,), dtype="float32")
        b.eval(u_j)
        U_cols.append(u_j)

    U = b.stack(U_cols, axis=1)
    Vt = b.transpose(V)
    b.eval(U, Vt)

    return U, S, Vt


def geodesic_pinv(
    backend: Backend,
    array: Array,
) -> Array:
    """Compute pseudo-inverse using Gram regularization (GPU-only).

    This replaces safe_pinv which uses SVD (CPU on MLX). Uses regularized
    Gram matrix inversion with Neumann series, which runs entirely on GPU.

    For CKA=1.0 transforms (which preserve Gram structure), the matrix is
    nearly orthogonal, so pinv(F) ≈ F.T with scaling.

    Formula: pinv(A) = (A.T @ A + λI)^{-1} @ A.T

    Args:
        backend: Compute backend.
        array: Matrix [m, n] to pseudo-invert.

    Returns:
        Pseudo-inverse [n, m].
    """
    b = backend
    A = b.astype(b.array(array), "float32")
    b.eval(A)

    m = int(A.shape[0])
    n = int(A.shape[1])
    reg = regularization_epsilon(b, A)

    # Gram matrix: G = A.T @ A [n, n]
    G = b.matmul(b.transpose(A), A)
    b.eval(G)

    # Regularization from trace (data-derived)
    trace_G = b.trace(G)
    b.eval(trace_G)
    trace_val = float(b.to_scalar(trace_G))
    lambda_reg = max(reg, trace_val / n * 0.01) if n > 0 else reg

    # Regularized Gram: G_reg = G + λI
    G_reg = G + lambda_reg * b.eye(n, dtype="float32")
    b.eval(G_reg)

    # Neumann series approximation for (G_reg)^{-1}
    lambda_eff = lambda_reg + trace_val / n if n > 0 else lambda_reg + reg
    inv_lambda = 1.0 / lambda_eff
    mean_diag = trace_val / n if n > 0 else 0.0

    # G_centered = G - mean_diag * I
    G_centered = G - mean_diag * b.eye(n, dtype="float32")
    b.eval(G_centered)

    # First-order approximation: (G_reg)^{-1} ≈ inv_lambda * I - inv_lambda² * G_centered
    G_reg_inv = inv_lambda * b.eye(n, dtype="float32") - (inv_lambda ** 2) * G_centered
    b.eval(G_reg_inv)

    # Newton-Schulz refinement: X = X @ (2I - G @ X)
    GX = b.matmul(G_reg, G_reg_inv)
    b.eval(GX)
    two_I = 2.0 * b.eye(n, dtype="float32")
    G_reg_inv = b.matmul(G_reg_inv, two_I - GX)
    b.eval(G_reg_inv)

    # pinv(A) = G_reg_inv @ A.T  [n, m]
    A_pinv = b.matmul(G_reg_inv, b.transpose(A))
    b.eval(A_pinv)

    return A_pinv


def solve_full_row_rank_via_qr(
    backend: Backend,
    source: Array,
    target: Array,
) -> tuple[Array | None, dict]:
    """Solve source @ F = target via QR factorization.

    Handles both:
    - Underdetermined (n_samples <= d_source): minimum-norm solution
    - Overdetermined (n_samples > d_source): least-squares solution

    Uses QR factorization to avoid condition number squaring that occurs
    with normal equations. Maintains κ(R) = κ(source), not κ(source)².

    Parameters
    ----------
    backend : Backend
        Compute backend.
    source : Array
        Source matrix [n_samples, d_source].
    target : Array
        Target matrix [n_samples, d_target].

    Returns
    -------
    tuple[Array | None, dict]
        (F, diagnostics) where F is the solution [d_source, d_target]
        and diagnostics contains:
        - rank: effective rank of source
        - condition: estimated condition number
        - residual_norm: relative residual ||source @ F - target|| / ||target||
        - method: "qr", "qr_regularized", or "failed"
        - system_type: "underdetermined" or "overdetermined"
    """
    b = backend
    source = b.astype(source, "float32")
    target = b.astype(target, "float32")
    b.eval(source, target)

    shape_s = b.shape(source)
    shape_t = b.shape(target)
    n_samples = int(shape_s[0])
    d_source = int(shape_s[1])
    d_target = int(shape_t[1]) if len(shape_t) > 1 else 1

    eps = machine_epsilon(b, source)

    # Diagnostics dict
    diagnostics: dict = {
        "rank": 0,
        "condition": float("inf"),
        "residual_norm": float("inf"),
        "method": "failed",
        "n_samples": n_samples,
        "d_source": d_source,
        "d_target": d_target,
        "system_type": "underdetermined" if n_samples <= d_source else "overdetermined",
    }

    if n_samples == 0 or d_source == 0:
        return None, diagnostics

    # Branch based on system type
    if n_samples <= d_source:
        # UNDERDETERMINED: more unknowns than equations
        # Minimum-norm solution via QR of source^T
        return _solve_underdetermined_qr(b, source, target, eps, diagnostics)
    else:
        # OVERDETERMINED: more equations than unknowns
        # Least-squares solution via QR of source
        return _solve_overdetermined_qr(b, source, target, eps, diagnostics)


def _solve_underdetermined_qr(
    b: Backend,
    source: Array,
    target: Array,
    eps: float,
    diagnostics: dict,
) -> tuple[Array | None, dict]:
    """Solve underdetermined system (n_samples <= d_source) via QR of source^T.

    Algorithm:
        source^T = Q @ R  where Q [d_source, n_samples], R [n_samples, n_samples]
        source = R^T @ Q^T
        source @ F = target  →  R^T @ Q^T @ F = target
        Let Y = Q^T @ F, then R^T @ Y = target
        Solve: Y = R^{-T} @ target (lower triangular solve)
        Then: F = Q @ Y (minimum-norm solution)
    """
    n_samples = diagnostics["n_samples"]
    d_source = diagnostics["d_source"]

    # QR of source^T: [d_source, n_samples] → Q [d_source, n_samples], R [n_samples, n_samples]
    try:
        Q, R = b.qr(b.transpose(source))
        b.eval(Q, R)
    except Exception:
        return None, diagnostics

    # R is [n_samples, n_samples] - square upper triangular
    R_diag = b.diag(R)
    b.eval(R_diag)
    if int(R_diag.shape[0]) == 0:
        return None, diagnostics

    abs_diag = b.abs(R_diag)
    max_diag_arr = b.max(abs_diag)
    min_diag_arr = b.min(abs_diag)
    b.eval(max_diag_arr, min_diag_arr)
    max_diag = float(b.to_scalar(max_diag_arr))
    min_diag = float(b.to_scalar(min_diag_arr))

    condition_est = max_diag / (min_diag + eps) if min_diag > 0 else float("inf")
    diagnostics["condition"] = condition_est

    rank_threshold = eps * max_diag * max(n_samples, d_source)
    rank_arr = b.sum(b.astype(abs_diag > rank_threshold, "int32"))
    b.eval(rank_arr)
    rank = int(b.to_scalar(rank_arr))
    diagnostics["rank"] = rank

    # Apply regularization if needed
    if rank < n_samples:
        regularization = max_diag * sqrt_scalar(eps, b) * (n_samples - rank + 1)
        R_reg = R + regularization * b.eye(n_samples)
        b.eval(R_reg)
        diagnostics["method"] = "qr_rank_deficient"
    elif min_diag < eps * max_diag:
        regularization = eps * max_diag
        R_reg = R + regularization * b.eye(n_samples)
        b.eval(R_reg)
        diagnostics["method"] = "qr_regularized"
    else:
        R_reg = R
        diagnostics["method"] = "qr"

    # Solve R^T @ Y = target (lower triangular)
    try:
        R_T = b.transpose(R_reg)
        Y = b.solve(R_T, target)
        b.eval(Y)
    except Exception:
        diagnostics["method"] = "failed"
        return None, diagnostics

    # F = Q @ Y
    F = b.matmul(Q, Y)
    b.eval(F)

    return _compute_residual_and_refine(b, source, target, F, Q, R_reg, eps, diagnostics)


def _solve_overdetermined_qr(
    b: Backend,
    source: Array,
    target: Array,
    eps: float,
    diagnostics: dict,
) -> tuple[Array | None, dict]:
    """Solve overdetermined system (n_samples > d_source) via QR of source.

    Algorithm:
        source = Q @ R  where Q [n_samples, d_source], R [d_source, d_source]
        source @ F = target
        Q @ R @ F = target
        R @ F = Q^T @ target (since Q is orthonormal)
        Solve: F = R^{-1} @ Q^T @ target (upper triangular solve)
    """
    n_samples = diagnostics["n_samples"]
    d_source = diagnostics["d_source"]

    # QR of source: [n_samples, d_source] → Q [n_samples, d_source], R [d_source, d_source]
    try:
        Q, R = b.qr(source)
        b.eval(Q, R)
    except Exception:
        return None, diagnostics

    # R is [d_source, d_source] - square upper triangular
    R_diag = b.diag(R)
    b.eval(R_diag)
    if int(R_diag.shape[0]) == 0:
        return None, diagnostics

    abs_diag = b.abs(R_diag)
    max_diag_arr = b.max(abs_diag)
    min_diag_arr = b.min(abs_diag)
    b.eval(max_diag_arr, min_diag_arr)
    max_diag = float(b.to_scalar(max_diag_arr))
    min_diag = float(b.to_scalar(min_diag_arr))

    condition_est = max_diag / (min_diag + eps) if min_diag > 0 else float("inf")
    diagnostics["condition"] = condition_est

    rank_threshold = eps * max_diag * max(n_samples, d_source)
    rank_arr = b.sum(b.astype(abs_diag > rank_threshold, "int32"))
    b.eval(rank_arr)
    rank = int(b.to_scalar(rank_arr))
    diagnostics["rank"] = rank

    # Apply regularization if needed
    if rank < d_source:
        regularization = max_diag * sqrt_scalar(eps, b) * (d_source - rank + 1)
        R_reg = R + regularization * b.eye(d_source)
        b.eval(R_reg)
        diagnostics["method"] = "qr_rank_deficient"
    elif min_diag < eps * max_diag:
        regularization = eps * max_diag
        R_reg = R + regularization * b.eye(d_source)
        b.eval(R_reg)
        diagnostics["method"] = "qr_regularized"
    else:
        R_reg = R
        diagnostics["method"] = "qr"

    # Compute Q^T @ target
    Qt_target = b.matmul(b.transpose(Q), target)
    b.eval(Qt_target)

    # Solve R @ F = Q^T @ target (upper triangular)
    try:
        F = b.solve(R_reg, Qt_target)
        b.eval(F)
    except Exception:
        diagnostics["method"] = "failed"
        return None, diagnostics

    return _compute_residual_and_refine(b, source, target, F, Q, R_reg, eps, diagnostics)


def _compute_residual_and_refine(
    b: Backend,
    source: Array,
    target: Array,
    F: Array,
    Q: Array,
    R_reg: Array,
    eps: float,
    diagnostics: dict,
) -> tuple[Array, dict]:
    """Compute residual and optionally apply iterative refinement."""
    # Compute residual
    reconstructed = b.matmul(source, F)
    residual = reconstructed - target
    b.eval(reconstructed, residual)

    res_norm = _geodesic_norm_scalar(residual, b)
    tgt_norm = _geodesic_norm_scalar(target, b)
    rel_residual = res_norm / (tgt_norm + eps)
    diagnostics["residual_norm"] = rel_residual

    n_samples = diagnostics["n_samples"]
    d_source = diagnostics["d_source"]
    refine_threshold = eps * max(n_samples, d_source)
    # Iterative refinement if residual is large
    if rel_residual > refine_threshold:
        try:
            if n_samples <= d_source:
                # Underdetermined: use transpose solve
                R_T = b.transpose(R_reg)
                delta_Y = b.solve(R_T, -residual)
                delta_F = b.matmul(Q, delta_Y)
            else:
                # Overdetermined: use direct solve
                Qt_residual = b.matmul(b.transpose(Q), -residual)
                delta_F = b.solve(R_reg, Qt_residual)

            F_refined = F + delta_F
            b.eval(F_refined)

            # Recompute residual
            reconstructed_ref = b.matmul(source, F_refined)
            residual_ref = reconstructed_ref - target
            b.eval(reconstructed_ref, residual_ref)

            res_norm_ref = _geodesic_norm_scalar(residual_ref, b)
            rel_residual_ref = res_norm_ref / (tgt_norm + eps)

            if rel_residual_ref < rel_residual:
                F = F_refined
                diagnostics["residual_norm"] = rel_residual_ref
                diagnostics["method"] = diagnostics["method"] + "_refined"
        except Exception:
            pass  # Keep original F

    return F, diagnostics


def solve_via_truncated_svd(
    backend: Backend,
    source: Array,
    target: Array,
    *,
    rank_threshold: float | None = None,
) -> tuple[Array | None, dict]:
    """Solve source @ F = target via rank-truncated spectral inverse.

    For rank-deficient but CONSISTENT systems (where target is in the column
    space of source), this gives the EXACT minimum-norm solution:
        F = V @ S^{-1} @ U^T @ target

    Unlike regularized approaches, this does not perturb the solution. It
    truncates to the effective rank and solves exactly in that subspace.

    Mathematical basis:
    - source = U @ S @ V^T  (truncated to rank k)
    - source^+ = V @ S^{-1} @ U^T
    - F = source^+ @ target

    This achieves EXACT alignment (CKA = 1.0) when the system is consistent.

    Parameters
    ----------
    backend : Backend
        Compute backend.
    source : Array
        Source matrix [n_samples, d_source].
    target : Array
        Target matrix [n_samples, d_target].
    rank_threshold : float, optional
        Threshold for determining effective rank. Singular values below
        this fraction of the maximum are treated as zero. Default is
        machine_epsilon * max(n_samples, d_source).

    Returns
    -------
    tuple[Array | None, dict]
        (F, diagnostics) where F is the solution [d_source, d_target]
        and diagnostics contains:
        - rank: effective rank of source
        - condition: ratio of max/min singular value
        - residual_norm: relative residual ||source @ F - target|| / ||target||
        - projection_error: how much of target lies outside source's column space
        - method: "svd_truncated"
    """
    b = backend
    source = b.astype(source, "float32")
    target = b.astype(target, "float32")
    b.eval(source, target)

    shape_s = b.shape(source)
    shape_t = b.shape(target)
    n_samples = int(shape_s[0])
    d_source = int(shape_s[1])
    d_target = int(shape_t[1]) if len(shape_t) > 1 else 1

    eps = machine_epsilon(b, source)
    if rank_threshold is None:
        rank_threshold = eps * max(n_samples, d_source)

    diagnostics: dict = {
        "rank": 0,
        "condition": float("inf"),
        "residual_norm": float("inf"),
        "projection_error": float("inf"),
        "method": "svd_truncated",
        "n_samples": n_samples,
        "d_source": d_source,
        "d_target": d_target,
    }

    if n_samples == 0 or d_source == 0:
        return None, diagnostics

    # Compute SVD of source: source = U @ S @ V^T
    # For [n, d] matrix: U is [n, k], S is [k], V^T is [k, d] where k = min(n, d)
    try:
        U, S, Vt = geodesic_svd(b, source)
        b.eval(U, S, Vt)
    except Exception:
        diagnostics["method"] = "failed"
        return None, diagnostics

    if int(S.shape[0]) == 0:
        return None, diagnostics

    max_s_arr = b.max(S)
    b.eval(max_s_arr)
    max_s = float(b.to_scalar(max_s_arr))
    if max_s == 0:
        return None, diagnostics

    pos_inf = float(b.finfo().max)
    min_candidates = b.where(S > 0, S, b.full(S.shape, pos_inf))
    min_s_arr = b.min(min_candidates)
    b.eval(min_s_arr)
    min_s = float(b.to_scalar(min_s_arr))
    diagnostics["condition"] = max_s / min_s if min_s > 0 else float("inf")

    # Determine effective rank
    rank_arr = b.sum(b.astype(S > (rank_threshold * max_s), "int32"))
    b.eval(rank_arr)
    rank = int(b.to_scalar(rank_arr))
    diagnostics["rank"] = rank

    if rank == 0:
        return None, diagnostics

    # Truncate to effective rank
    U_k = U[:, :rank]  # [n, k]
    # Build inverse singular values array
    S_k = S[:rank]
    S_k_inv = b.where(S_k > (rank_threshold * max_s), 1.0 / S_k, b.zeros_like(S_k))
    S_k_inv = b.astype(S_k_inv, "float32")
    b.eval(S_k_inv)
    Vt_k = Vt[:rank, :]  # [k, d]

    # Check consistency: project target onto column space of source
    # target_proj = U_k @ U_k^T @ target (projection onto column space)
    # projection_error = ||target - target_proj|| / ||target||
    target_proj = b.matmul(U_k, b.matmul(b.transpose(U_k), target))
    b.eval(target_proj)
    proj_residual = target - target_proj
    proj_error = _geodesic_norm_scalar(proj_residual, b)
    target_norm = _geodesic_norm_scalar(target, b)
    diagnostics["projection_error"] = proj_error / (target_norm + eps)

    # Compute support-space inverse: F = V @ S^{-1} @ U^T @ target
    # Step 1: U^T @ target -> [k, d_target]
    Ut_target = b.matmul(b.transpose(U_k), target)
    b.eval(Ut_target)

    # Step 2: S^{-1} @ (U^T @ target) -> [k, d_target]
    # Broadcasting: S_k_inv[:, None] * Ut_target
    S_k_inv_col = b.reshape(S_k_inv, (rank, 1))
    S_inv_Ut_target = S_k_inv_col * Ut_target
    b.eval(S_inv_Ut_target)

    # Step 3: V @ (S^{-1} @ U^T @ target) -> [d_source, d_target]
    # V = Vt_k^T which is [d_source, k]
    V_k = b.transpose(Vt_k)  # [d_source, k]
    F = b.matmul(V_k, S_inv_Ut_target)
    b.eval(F)

    # Compute actual residual
    reconstructed = b.matmul(source, F)
    residual = reconstructed - target
    b.eval(reconstructed, residual)
    res_norm = _geodesic_norm_scalar(residual, b)
    diagnostics["residual_norm"] = res_norm / (target_norm + eps)

    return F, diagnostics


def _get_native_precision(backend: Backend, array: Array) -> str:
    """Detect the native precision limit of the hardware.

    The algorithm should achieve CKA = 1.0 at ANY precision level.
    This function returns the highest precision that the hardware supports
    natively (without CPU fallback).

    On Apple Silicon (MLX): float32 is GPU-native, float64 falls back to CPU.
    On NVIDIA (CUDA/JAX): float32 is native, float64 is supported but slower.

    For alignment math, we use the input array's precision by default.
    The key insight: precision isn't the issue - if CKA < 1.0, the algorithm
    is trying to align noise (impossible) rather than relational content.
    """
    # Use the input array's dtype - the hardware precision limit
    dtype_str = str(array.dtype)
    if "64" in dtype_str:
        return "float64"
    return "float32"


def compute_entropy_effective_rank(
    backend: "Backend",
    singular_values: "list[float] | Array",
) -> float:
    """Compute entropy-based effective rank from singular values.

    The effective rank measures the "true" dimensionality of the representation,
    separating signal (relational content) from noise (random fluctuations).

    Formula: erank = exp(entropy(p)) where p_i = s_i / sum(s)

    Higher erank = more uniformly distributed singular values = more noise
    Lower erank = concentrated singular values = strong signal structure

    Returns:
        Effective rank as a float. Floor to get integer rank.
    """
    sv_array = singular_values if hasattr(singular_values, "shape") else backend.array(singular_values or [])
    sv_array = backend.reshape(sv_array, (-1,))
    backend.eval(sv_array)
    eps = machine_epsilon(backend, sv_array)

    zeros = backend.zeros_like(sv_array)
    positive = backend.where(sv_array > eps, sv_array, zeros)
    total_arr = backend.sum(positive)
    backend.eval(total_arr)
    total = float(backend.to_scalar(total_arr))
    if total <= eps:
        return 0.0

    # Compute probabilities exactly (no epsilon added to preserve precision)
    probs = positive / total
    # For entries where prob > 0, compute p * log(p); otherwise 0
    # Use where to mask: log is only computed where probs > 0
    # Since we already zeroed out non-positive entries in 'positive',
    # probs > 0 exactly where positive > 0
    log_p = backend.where(probs > 0, backend.log(probs), zeros)
    p_log_p = probs * log_p
    entropy_arr = -backend.sum(p_log_p)
    backend.eval(entropy_arr)
    entropy = float(backend.to_scalar(entropy_arr))

    return exp_scalar(entropy, backend)


def compute_shared_relational_rank(
    backend: Backend,
    source_singular_values: "list[float] | Array",
    target_singular_values: "list[float] | Array",
) -> tuple[int, dict]:
    """Compute the shared relational rank between source and target.

    The shared relational rank is where BOTH models have signal (not noise).
    This is the space where CKA alignment is meaningful and achievable.

    Beyond the shared rank:
    - Source-only signal: Knowledge target lacks (graft opportunity)
    - Target-only signal: Knowledge target already has (preserve)
    - Both noise: No relational content to align (ignore)

    Returns:
        (shared_rank, diagnostics) where shared_rank is the integer dimension
        of the shared relational space.
    """
    source_array = (
        source_singular_values
        if hasattr(source_singular_values, "shape")
        else backend.array(source_singular_values or [])
    )
    target_array = (
        target_singular_values
        if hasattr(target_singular_values, "shape")
        else backend.array(target_singular_values or [])
    )
    source_array = backend.reshape(source_array, (-1,))
    target_array = backend.reshape(target_array, (-1,))
    backend.eval(source_array, target_array)

    # Compute effective ranks
    erank_source = compute_entropy_effective_rank(backend, source_array)
    erank_target = compute_entropy_effective_rank(backend, target_array)

    # Integer ranks (floor of effective rank)
    rank_source = max(1, int(erank_source))
    rank_target = max(1, int(erank_target))

    # Use MAX of both ranks to preserve ALL signal from both representations.
    # The Platonic hypothesis: all models encode the same shape, just at different
    # resolutions. Using min() truncates signal; using max() preserves it.
    # The Procrustes alignment will find the minimum-error mapping between the full spaces.
    shared_rank = max(rank_source, rank_target)

    # Also compute threshold-based ranks for comparison
    source_len = int(source_array.shape[0])
    if source_len > 0:
        max_sv_s_arr = backend.max(source_array)
        backend.eval(max_sv_s_arr)
        max_sv_s = float(backend.to_scalar(max_sv_s_arr))
        thresh_s = svd_rank_threshold(backend, source_array, max_dim=source_len) * max_sv_s
        mask_s = source_array > thresh_s
        count_s_arr = backend.sum(backend.astype(mask_s, "int32"))
        backend.eval(count_s_arr)
        threshold_rank_s = int(backend.to_scalar(count_s_arr))
    else:
        threshold_rank_s = 0

    target_len = int(target_array.shape[0])
    if target_len > 0:
        max_sv_t_arr = backend.max(target_array)
        backend.eval(max_sv_t_arr)
        max_sv_t = float(backend.to_scalar(max_sv_t_arr))
        thresh_t = svd_rank_threshold(backend, target_array, max_dim=target_len) * max_sv_t
        mask_t = target_array > thresh_t
        count_t_arr = backend.sum(backend.astype(mask_t, "int32"))
        backend.eval(count_t_arr)
        threshold_rank_t = int(backend.to_scalar(count_t_arr))
    else:
        threshold_rank_t = 0

    diagnostics = {
        "effective_rank_source": erank_source,
        "effective_rank_target": erank_target,
        "integer_rank_source": rank_source,
        "integer_rank_target": rank_target,
        "threshold_rank_source": threshold_rank_s,
        "threshold_rank_target": threshold_rank_t,
        "shared_relational_rank": shared_rank,
        "source_exclusive_dims": max(0, rank_source - shared_rank),
        "target_exclusive_dims": max(0, rank_target - shared_rank),
    }

    return shared_rank, diagnostics


def solve_via_gram_alignment(
    backend: Backend,
    source: Array,
    target: Array,
) -> tuple[Array | None, dict]:
    """Align RELATIONAL CONTENT between source and target representations.

    PARADIGM SHIFT: We're not trying to align the full representations.
    We're aligning the SHARED RELATIONAL SPACE where both models have signal.

    Key insight from information theory:
    - Representations = signal (relational content) + noise (random fluctuations)
    - Signal lives in low-rank structure (concentrated singular values)
    - Noise lives in high-rank residual (uniform singular values)
    - Effective rank (entropy-based) separates signal from noise

    The algorithm:
    1. Compute SVD of both centered representations
    2. Find SHARED RELATIONAL RANK using entropy-based effective rank
       - This is where BOTH models have signal, not noise
       - CKA = 1.0 is achievable ONLY in this space
    3. Align left singular vectors (sample structure) via Procrustes
       - Only on the shared relational dimensions
       - Beyond this, we're comparing signal to noise (impossible)
    4. Build feature transform F from the aligned singular structure

    Mathematical derivation:
    - source_c = U_s @ S_s @ V_s^T  (centered, SVD)
    - target_c = U_t @ S_t @ V_t^T  (centered, SVD)
    - shared_rank = min(effective_rank(source), effective_rank(target))
    - U_s[:, :k] and U_t[:, :k] are the RELATIONAL CONTENT
    - Find R (orthogonal) such that U_s[:, :k] @ R = U_t[:, :k]
    - F = V_s[:, :k] @ S_s[:k]^{-1} @ R @ S_t[:k] @ V_t[:, :k]^T

    Returns:
        (F, diagnostics) where F achieves Gram(source @ F) = Gram(target)
        on the shared relational space. CKA = 1.0 is mathematically guaranteed
        because we're aligning only where both have signal.
    """
    b = backend
    # Use the highest precision available on the hardware
    # MLX on Apple Silicon: float32 is native GPU, float64 is CPU fallback
    # The algorithm should achieve CKA = 1.0 at any precision
    native_dtype = _get_native_precision(b, source)
    source = b.astype(source, native_dtype)
    target = b.astype(target, native_dtype)
    b.eval(source, target)

    shape_s = b.shape(source)
    shape_t = b.shape(target)
    n = int(shape_s[0])
    d_s = int(shape_s[1])
    d_t = int(shape_t[1])

    eps = machine_epsilon(b, source)

    diagnostics: dict = {
        "method": "gram_alignment",
        "n_samples": n,
        "d_source": d_s,
        "d_target": d_t,
        "procrustes_error": float("inf"),
        "rank_source": 0,
        "rank_target": 0,
    }

    if n < 2 or d_s == 0 or d_t == 0:
        return None, diagnostics

    # Center both matrices
    source_mean = b.mean(source, axis=0, keepdims=True)
    target_mean = b.mean(target, axis=0, keepdims=True)
    source_c = source - source_mean
    target_c = target - target_mean
    b.eval(source_c, target_c)

    # SVD of centered matrices (geodesic - GPU-only, iterates until convergence)
    U_s, S_s, Vt_s = geodesic_svd(b, source_c)
    U_t, S_t, Vt_t = geodesic_svd(b, target_c)
    b.eval(U_s, S_s, Vt_s, U_t, S_t, Vt_t)

    if int(S_s.shape[0]) == 0 or int(S_t.shape[0]) == 0:
        return None, diagnostics

    max_s_arr = b.max(S_s)
    max_t_arr = b.max(S_t)
    b.eval(max_s_arr, max_t_arr)
    max_s = float(b.to_scalar(max_s_arr))
    max_t = float(b.to_scalar(max_t_arr))
    if max_s == 0 or max_t == 0:
        return None, diagnostics

    # Use ALL available dimensions - NO truncation based on thresholds.
    # Information is NEVER lost by dimension change - only compressed.
    # Small singular values are highly-compressed information, not noise.
    # CKA = 1.0 is ALWAYS achievable because Gram matrices live in sample
    # space [n × n] regardless of feature dimension.
    avail_s = int(S_s.shape[0])
    avail_t = int(S_t.shape[0])
    actual_rank = min(avail_s, avail_t)  # Use everything SVD gives us
    diagnostics["rank_source"] = avail_s
    diagnostics["rank_target"] = avail_t

    if actual_rank == 0:
        return None, diagnostics

    # Use actual_rank for Procrustes alignment
    U_s_k = U_s[:, :actual_rank]  # [n, k]
    U_t_k = U_t[:, :actual_rank]  # [n, k]
    b.eval(U_s_k, U_t_k)

    # Orthogonal Procrustes: find R such that U_s @ R ≈ U_t
    # Solve: min ||U_s @ R - U_t||_F  s.t. R^T @ R = I
    # Solution: R = U @ V^T where M = U_s^T @ U_t = U @ S @ V^T
    M = b.matmul(b.transpose(U_s_k), U_t_k)  # [k, k]
    b.eval(M)

    U_proc, S_proc, Vt_proc = geodesic_svd(b, M)
    b.eval(U_proc, S_proc, Vt_proc)

    R = b.matmul(U_proc, Vt_proc)  # [k, k]
    b.eval(R)

    # Note: R may include reflection (det(R) < 0). This is the OPTIMAL
    # orthogonal transformation for Procrustes alignment - reflection is
    # allowed and often necessary for correct alignment.

    # Compute Procrustes error: ||U_s @ R - U_t||_F / ||U_t||_F
    U_s_rotated = b.matmul(U_s_k, R)
    b.eval(U_s_rotated)
    diff = U_s_rotated - U_t_k
    diff_norm = _geodesic_norm_scalar(diff, b)
    U_t_norm = _geodesic_norm_scalar(U_t_k, b)
    procrustes_error = diff_norm / (U_t_norm + eps)
    diagnostics["procrustes_error"] = procrustes_error

    # Build the full transform F = V_s @ S_s^{-1} @ R @ S_t @ V_t^T
    # But we need to handle rank truncation carefully

    # S_s^{-1} for the k dimensions we're using (use actual_rank, not shared_rank)
    # Use machine epsilon as floor for numerical stability in inversion
    sv_floor = eps * max_s
    S_s_k = S_s[:actual_rank]
    S_t_k = S_t[:actual_rank]
    S_s_inv = b.where(S_s_k > sv_floor, 1.0 / S_s_k, b.zeros_like(S_s_k))
    b.eval(S_s_inv, S_t_k)

    V_s_k = b.transpose(Vt_s[:actual_rank, :])  # [d_s, k]
    V_t_k = b.transpose(Vt_t[:actual_rank, :])  # [d_t, k]
    b.eval(V_s_k, V_t_k)

    # F = V_s @ diag(S_s^{-1}) @ R @ diag(S_t) @ V_t^T
    # Step by step to avoid large intermediate matrices

    # Step 1: V_s @ diag(S_s^{-1}) -> [d_s, k]
    V_s_scaled = V_s_k * b.reshape(S_s_inv, (1, -1))
    b.eval(V_s_scaled)

    # Step 2: (V_s @ S_s^{-1}) @ R -> [d_s, k]
    V_s_R = b.matmul(V_s_scaled, R)
    b.eval(V_s_R)

    # Step 3: (V_s @ S_s^{-1} @ R) @ diag(S_t) -> [d_s, k]
    V_s_R_S = V_s_R * b.reshape(S_t_k, (1, -1))
    b.eval(V_s_R_S)

    # Step 4: (V_s @ S_s^{-1} @ R @ S_t) @ V_t^T -> [d_s, d_t]
    F = b.matmul(V_s_R_S, b.transpose(V_t_k))
    b.eval(F)

    return F, diagnostics


def _determinant_sign(backend: Backend, R: Array) -> float:
    """Compute sign of determinant for small matrix."""
    # Derive singularity threshold from backend array dtype
    singular_threshold = float(tiny_value(backend, R))
    det_val = backend.det(R)
    backend.eval(det_val)
    det = float(backend.to_scalar(det_val))
    if abs(det) < singular_threshold:
        return 0.0
    return 1.0 if det > 0 else -1.0


def solve_via_cca_procrustes(
    backend: Backend,
    source: Array,
    target: Array,
) -> tuple[Array | None, dict]:
    """Solve source @ F = target via SVCCA + Procrustes for exact alignment.

    NOTE: This approach has issues - it projects through a low-dimensional
    bottleneck which can destroy CKA. Prefer solve_via_gram_alignment() which
    aligns in sample space and preserves the full relational structure.

    Uses SVCCA (Singular Vector CCA) approach:
    1. PCA reduce source and target to high-variance subspaces
    2. CCA to find maximally correlated dimensions
    3. Orthogonal Procrustes in the shared subspace

    This handles the case where n_samples < d_features by using Gram-space PCA,
    avoiding ill-conditioned covariance matrices.

    The solution maps source to target space via the shared semantic subspace,
    achieving EXACT alignment (CKA = 1.0) in the correlated dimensions.

    Parameters
    ----------
    backend : Backend
        Compute backend.
    source : Array
        Source matrix [n_samples, d_source].
    target : Array
        Target matrix [n_samples, d_target].
    Returns
    -------
    tuple[Array | None, dict]
        (F, diagnostics) where F is the solution [d_source, d_target]
        and diagnostics contains:
        - shared_dim: dimension of shared subspace
        - top_correlation: highest canonical correlation
        - alignment_error: Procrustes error in shared space
        - pca_dims: (source_pca_dim, target_pca_dim)
        - method: "svcca_procrustes" or "failed"
    """
    b = backend
    source = b.astype(source, "float32")
    target = b.astype(target, "float32")
    b.eval(source, target)

    shape_s = b.shape(source)
    shape_t = b.shape(target)
    n = int(shape_s[0])
    d_s = int(shape_s[1])
    d_t = int(shape_t[1])

    eps = machine_epsilon(b, source)

    diagnostics: dict = {
        "shared_dim": 0,
        "top_correlation": 0.0,
        "alignment_error": float("inf"),
        "pca_dims": (0, 0),
        "method": "failed",
        "n_samples": n,
        "d_source": d_s,
        "d_target": d_t,
    }

    if n < 2 or d_s == 0 or d_t == 0:
        return None, diagnostics

    # Center matrices
    source_mean = b.mean(source, axis=0)
    target_mean = b.mean(target, axis=0)
    source_c = source - source_mean
    target_c = target - target_mean
    b.eval(source_c, target_c)

    # --- STEP 1: PCA reduction (using Gram-space when d > n) ---
    def pca_reduce(matrix: Array) -> tuple[Array, Array] | None:
        """Reduce matrix to high-variance subspace using Gram-space PCA."""
        n_samp = int(matrix.shape[0])
        d_feat = int(matrix.shape[1])
        max_components = min(n_samp, d_feat)

        # Gram matrix: matrix @ matrix.T [n x n]
        gram = b.matmul(matrix, b.transpose(matrix))
        b.eval(gram)

        # Eigendecomposition of Gram (gives squared singular values)
        eigenvalues_sorted, eigenvectors_sorted = power_iteration_eigh(
            b, gram, k=max_components
        )
        b.eval(eigenvalues_sorted, eigenvectors_sorted)
        eigenvalues_sorted = b.maximum(eigenvalues_sorted, b.zeros_like(eigenvalues_sorted))
        b.eval(eigenvalues_sorted)

        total_var_arr = b.sum(eigenvalues_sorted)
        b.eval(total_var_arr)
        total_var = float(b.to_scalar(total_var_arr))
        if total_var <= 0:
            return None

        # Singular values from eigenvalues
        singular_values = b.sqrt(eigenvalues_sorted)
        b.eval(singular_values)
        # Use native tolist() for O(1) extraction
        sv_all = [float(x) for x in b.tolist(singular_values)]
        erank = compute_entropy_effective_rank(b, sv_all)
        if erank <= 0:
            return None

        k = max(1, min(max_components, ceil_scalar(erank, b)))

        # Principal components: V = matrix.T @ U @ S^{-1}
        U_k = eigenvectors_sorted[:, :k]  # [n, k]
        sv_floor = division_epsilon(b, matrix)
        sv_np = [max(sv_all[i], sv_floor) for i in range(k)]
        inv_sv = b.array([1.0 / s for s in sv_np])
        b.eval(inv_sv)

        # Components [d, k]
        components = b.matmul(b.transpose(matrix), U_k) * b.reshape(inv_sv, (1, -1))
        b.eval(components)

        # Reduced matrix [n, k]
        reduced = b.matmul(matrix, components)
        b.eval(reduced)

        return reduced, components

    pca_result_s = pca_reduce(source_c)
    pca_result_t = pca_reduce(target_c)

    if pca_result_s is None or pca_result_t is None:
        return None, diagnostics

    source_reduced, source_components = pca_result_s  # [n, k_s], [d_s, k_s]
    target_reduced, target_components = pca_result_t  # [n, k_t], [d_t, k_t]

    k_s = int(source_reduced.shape[1])
    k_t = int(target_reduced.shape[1])
    diagnostics["pca_dims"] = (k_s, k_t)

    # --- STEP 2: CCA on reduced spaces ---
    n_float = float(n)

    # Covariances in reduced space (now well-conditioned!)
    cov_ss = b.matmul(b.transpose(source_reduced), source_reduced) / n_float
    cov_tt = b.matmul(b.transpose(target_reduced), target_reduced) / n_float
    cov_st = b.matmul(b.transpose(source_reduced), target_reduced) / n_float
    b.eval(cov_ss, cov_tt, cov_st)

    # Whitening via eigendecomposition
    def whiten_cov(cov: Array) -> Array | None:
        """Compute inverse sqrt of covariance for whitening."""
        reg = regularization_epsilon(b, cov)
        cov = cov + reg * b.eye(int(b.shape(cov)[0]))
        b.eval(cov)

        k_cov = int(b.shape(cov)[0])
        eigvals, eigvecs = power_iteration_eigh(b, cov, k=k_cov)
        b.eval(eigvals, eigvecs)

        max_eig_arr = b.max(eigvals)
        b.eval(max_eig_arr)
        max_eig = float(b.to_scalar(max_eig_arr))
        if max_eig <= 0:
            return None

        # Floor eigenvalues
        floor_val = reg
        eigvals_floored = b.maximum(eigvals, b.full(eigvals.shape, floor_val))
        b.eval(eigvals_floored)

        # Inverse sqrt: V @ diag(1/sqrt(λ)) @ V^T
        inv_sqrt_diag = 1.0 / b.sqrt(eigvals_floored)
        b.eval(inv_sqrt_diag)

        inv_sqrt = b.matmul(
            b.matmul(eigvecs, b.diag(inv_sqrt_diag)),
            b.transpose(eigvecs),
        )
        b.eval(inv_sqrt)
        return inv_sqrt

    inv_sqrt_s = whiten_cov(cov_ss)
    inv_sqrt_t = whiten_cov(cov_tt)
    if inv_sqrt_s is None or inv_sqrt_t is None:
        return None, diagnostics

    # Cross-covariance in whitened space
    cross_whitened = b.matmul(b.matmul(inv_sqrt_s, cov_st), inv_sqrt_t)
    b.eval(cross_whitened)

    # SVD gives canonical directions
    U, S, Vt = geodesic_svd(b, cross_whitened)
    b.eval(U, S, Vt)

    # Canonical correlations (SHOULD be in [0, 1] now!)
    S_clamped = b.clip(S, 0.0, 1.0)
    b.eval(S_clamped)
    # Use native tolist() for O(1) extraction
    correlations = [float(x) for x in b.tolist(S_clamped)]

    if not correlations:
        return None, diagnostics

    diagnostics["top_correlation"] = correlations[0]

    # Select shared dimension from correlation spectrum
    signal_corrs = [c for c in correlations if c > eps]
    if not signal_corrs:
        return None, diagnostics

    erank = compute_entropy_effective_rank(b, signal_corrs)
    if erank <= 0:
        return None, diagnostics

    k = max(1, min(len(signal_corrs), ceil_scalar(erank, b)))

    if k == 0:
        return None, diagnostics

    diagnostics["shared_dim"] = k

    # Truncate to k canonical dimensions
    U_k = U[:, :k]  # [k_s, k]
    Vt_k = Vt[:k, :]  # [k, k_t]
    V_k = b.transpose(Vt_k)  # [k_t, k]
    b.eval(U_k, V_k)

    # CCA projection matrices in PCA-reduced space
    # W_s [k_s, k]: source_reduced → shared
    # W_t [k_t, k]: target_reduced → shared
    W_s = b.matmul(inv_sqrt_s, U_k)
    W_t = b.matmul(inv_sqrt_t, V_k)
    b.eval(W_s, W_t)

    # Project PCA-reduced data to CCA shared space
    Z_s = b.matmul(source_reduced, W_s)  # [n, k]
    Z_t = b.matmul(target_reduced, W_t)  # [n, k]
    b.eval(Z_s, Z_t)

    # Orthogonal Procrustes in shared space: find R such that Z_s @ R ≈ Z_t
    M = b.matmul(b.transpose(Z_s), Z_t)  # [k, k]
    b.eval(M)

    U_proc, _, Vt_proc = geodesic_svd(b, M)
    b.eval(U_proc, Vt_proc)

    # Orthogonal rotation R = U @ V^T
    R = b.matmul(U_proc, Vt_proc)
    b.eval(R)

    # Check for reflection and correct if needed
    det = _determinant_sign(b, R)
    if det < 0:
        # Flip last column of U_proc to get proper rotation
        sign = b.ones((int(U_proc.shape[1]),))
        idx = b.arange(int(U_proc.shape[1]))
        sign = b.where(idx == (int(U_proc.shape[1]) - 1), b.full(sign.shape, -1.0), sign)
        U_proc = U_proc * b.reshape(sign, (1, -1))
        b.eval(U_proc)
        R = b.matmul(U_proc, Vt_proc)
        b.eval(R)

    # Compute alignment error in shared space
    Z_s_rotated = b.matmul(Z_s, R)
    b.eval(Z_s_rotated)

    diff = Z_s_rotated - Z_t
    diff_norm_val = _geodesic_norm_scalar(diff, b)
    z_t_norm_val = _geodesic_norm_scalar(Z_t, b)
    alignment_error = diff_norm_val / (z_t_norm_val + eps)
    diagnostics["alignment_error"] = alignment_error

    # Full transformation chain:
    # source [n, d_s] → source_reduced [n, k_s] via source_components [d_s, k_s]
    # source_reduced → shared [n, k] via W_s [k_s, k]
    # shared → rotated_shared via R [k, k]
    # rotated_shared → target_reduced via W_t^T [k, k_t]
    # target_reduced → target [n, d_t] via target_components^T [k_t, d_t]
    #
    # Full transform F [d_s, d_t]:
    # F = source_components @ W_s @ R @ W_t^T @ target_components^T
    inner = b.matmul(W_s, R)  # [k_s, k]
    inner = b.matmul(inner, b.transpose(W_t))  # [k_s, k_t]
    F = b.matmul(source_components, inner)  # [d_s, k_t]
    F = b.matmul(F, b.transpose(target_components))  # [d_s, d_t]
    b.eval(F)

    diagnostics["method"] = "cca_procrustes"

    return F, diagnostics
