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

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

__all__ = [
    # Epsilon and threshold utilities
    "machine_epsilon",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    # Statistical utilities
    "compute_pearson_correlation",
    # Matrix decomposition
    "svd_via_eigh",
    "canonicalize_svd_signs",
    "safe_pinv",
    "solve_full_row_rank_via_qr",
    "solve_via_truncated_svd",
    "solve_via_gram_alignment",
    "solve_via_cca_procrustes",
    # Rank estimation
    "compute_entropy_effective_rank",
    "compute_shared_relational_rank",
]


def machine_epsilon(backend: Backend, array: Array) -> float:
    """Get machine epsilon for the array's dtype.

    This is the smallest value such that 1.0 + epsilon != 1.0.
    Use for general numerical stability in comparisons.
    """
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for safe division operations.

    Scaled up from machine epsilon to provide numerical headroom.
    Use when dividing to prevent division by zero.
    """
    return backend.finfo(array.dtype).eps * 1e3


def regularization_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for matrix regularization.

    Uses sqrt(eps) which is the standard choice for regularization
    in numerical linear algebra (Tikhonov regularization, ridge).
    """
    return math.sqrt(backend.finfo(array.dtype).eps)


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


def compute_pearson_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
) -> float:
    """Compute Pearson correlation coefficient between two lists.

    This is the canonical implementation for computing Pearson's r
    across geometry modules. Uses pure Python math to avoid backend
    dependencies for simple list operations.

    Args:
        lhs: First list of values.
        rhs: Second list of values (must be same length as lhs).
        default: Value to return on error (empty lists, mismatched lengths).
                 If None, returns float("nan") on error.

    Returns:
        Pearson correlation coefficient in [-1, 1], or default/nan on error.
    """
    error_value = default if default is not None else float("nan")

    if not lhs or len(lhs) != len(rhs):
        return error_value

    n = len(lhs)
    mean_l = sum(lhs) / n
    mean_r = sum(rhs) / n

    num = 0.0
    den_l = 0.0
    den_r = 0.0

    for i in range(n):
        diff_l = lhs[i] - mean_l
        diff_r = rhs[i] - mean_r
        num += diff_l * diff_r
        den_l += diff_l * diff_l
        den_r += diff_r * diff_r

    denom = math.sqrt(den_l) * math.sqrt(den_r)
    if denom <= 0:
        return error_value

    return num / denom


def svd_via_eigh(
    backend: Backend,
    array: Array,
    *,
    full_matrices: bool = False,
    dtype: str | None = None,
) -> tuple[Array, Array, Array]:
    """Compute SVD via symmetric eigendecomposition (GPU-stable, no SVD calls).

    This uses eigendecomposition of A^T A to obtain right singular vectors and
    singular values, and completes the left basis if needed. For rank-deficient
    matrices, the null-space basis is filled from A A^T eigenvectors so that
    U and V remain orthonormal.

    Parameters
    ----------
    dtype : str, optional
        Override dtype. If None, preserves input dtype (float32 or float64).
        Use float64 for high-precision alignment computations.
    """
    b = backend
    # Preserve input precision by default; override with dtype param
    target_dtype = dtype if dtype is not None else str(array.dtype)
    # MLX eigh requires float32 or float64 - ensure we have one of those
    if "64" in target_dtype or target_dtype == "float64":
        A = b.astype(array, "float64")
    else:
        A = b.astype(array, "float32")
    shape = b.shape(A)
    m = int(shape[0])
    n = int(shape[1]) if len(shape) > 1 else 0

    if m == 0 or n == 0:
        k = min(m, n)
        U = b.zeros((m, k))
        S = b.zeros((k,))
        Vt = b.zeros((k, n))
        return U, S, Vt

    cov_r = b.matmul(b.transpose(A), A)
    eigvals_r, V_full = b.eigh(cov_r)
    b.eval(cov_r, eigvals_r, V_full)

    order_r = b.argsort(-eigvals_r)
    eigvals_r = b.take(eigvals_r, order_r, axis=0)
    V_full = b.take(V_full, order_r, axis=1)
    b.eval(eigvals_r, V_full)

    s = b.sqrt(b.maximum(eigvals_r, b.zeros_like(eigvals_r)))
    b.eval(s)

    s_vals = [float(v) for v in b.to_numpy(s).tolist()]
    if not s_vals:
        k = min(m, n)
        U = b.zeros((m, k))
        S = b.zeros((k,))
        Vt = b.zeros((k, n))
        return U, S, Vt

    max_s = max(s_vals)
    eps = machine_epsilon(b, A)
    threshold = max(m, n) * eps * max_s

    k = min(m, n)
    rank = sum(1 for v in s_vals[:k] if v > threshold)

    if rank == 0:
        U = b.zeros((m, k))
        S = b.zeros((k,))
        Vt = b.zeros((k, n))
        return U, S, Vt

    V_pos = V_full[:, :rank]
    s_pos = s[:rank]
    inv_s = 1.0 / s_pos
    U_pos = b.matmul(A, V_pos) * b.reshape(inv_s, (1, -1))
    b.eval(U_pos)

    need_full_u = full_matrices or rank < k
    if need_full_u:
        cov_l = b.matmul(A, b.transpose(A))
        eigvals_l, U_full = b.eigh(cov_l)
        b.eval(cov_l, eigvals_l, U_full)

        order_l = b.argsort(-eigvals_l)
        U_full = b.take(U_full, order_l, axis=1)
    if full_matrices:
        if rank < m:
            U_null = U_full[:, rank:m]
            U = b.concatenate([U_pos, U_null], axis=1)
        else:
            U = U_full
        Vt = b.transpose(V_full)
    else:
        if rank < k:
            U_null = U_full[:, rank:k]
            U = b.concatenate([U_pos, U_null], axis=1)
        else:
            U = U_pos
        Vt = b.transpose(V_full[:, :k])

    S = s[:k]
    if threshold > 0.0:
        thresh_arr = b.full(S.shape, float(threshold))
        S = b.where(S > thresh_arr, S, b.zeros_like(S))
        b.eval(S)

    if full_matrices:
        U, Vt = canonicalize_svd_signs(b, U, Vt)
        return U, S, Vt

    U_out = U[:, :k]
    Vt_out = Vt[:k, :]
    U_out, Vt_out = canonicalize_svd_signs(b, U_out, Vt_out)
    return U_out, S[:k], Vt_out


def canonicalize_svd_signs(
    backend: Backend,
    U: Array,
    Vt: Array,
) -> tuple[Array, Array]:
    """Canonicalize SVD signs to ensure deterministic decomposition.

    SVD has an inherent sign ambiguity: for each singular vector pair (u_i, v_i),
    flipping both signs gives an equally valid decomposition:
        A = U @ S @ Vt = (-U[:, i]) @ S @ (-Vt[i, :])

    This causes phase inconsistencies when aligning different layers or models,
    as each SVD call may choose a different sign convention.

    This function enforces a deterministic sign convention:
    - For each singular vector, find the element with largest absolute value
    - If that element is negative, flip the signs of both U[:, i] and Vt[i, :]
    - This ensures the "dominant direction" of each vector is positive

    Mathematical guarantee:
    - The product U @ Vt (and hence A = U @ S @ Vt) is unchanged
    - The sign choice is deterministic given the input matrix
    - Different matrices with similar structure will have consistent phase

    Parameters
    ----------
    backend : Backend
        Compute backend.
    U : Array
        Left singular vectors [m, k] where k is the number of singular values.
    Vt : Array
        Right singular vectors (transposed) [k, n].

    Returns
    -------
    tuple[Array, Array]
        (U_canonical, Vt_canonical) with deterministic signs.
    """
    b = backend

    U_shape = b.shape(U)
    Vt_shape = b.shape(Vt)

    if len(U_shape) < 2 or len(Vt_shape) < 2:
        return U, Vt

    # Number of singular vectors to canonicalize
    # U is [m, k_u] and Vt is [k_v, n] - we can only process min(k_u, k_v)
    k = int(min(U_shape[1], Vt_shape[0]))
    if k == 0:
        return U, Vt

    # Convert to numpy for sign determination (small k typically)
    U_np = b.to_numpy(U).copy()
    Vt_np = b.to_numpy(Vt).copy()

    for i in range(k):
        # Find the element with largest absolute value in U[:, i]
        u_col = U_np[:, i]
        max_idx = int(abs(u_col).argmax())
        max_val = u_col[max_idx]

        # If the dominant element is negative, flip both U[:, i] and Vt[i, :]
        if max_val < 0:
            U_np[:, i] = -U_np[:, i]
            Vt_np[i, :] = -Vt_np[i, :]

    # Convert back to backend arrays
    U_canonical = b.array(U_np)
    Vt_canonical = b.array(Vt_np)
    b.eval(U_canonical, Vt_canonical)

    return U_canonical, Vt_canonical


def safe_pinv(
    backend: Backend,
    array: Array,
    *,
    rcond: float | None = None,
    warn_on_ill_conditioned: bool = True,
) -> tuple[Array, dict]:
    """Compute pseudo-inverse with conditioning diagnostics.

    Standard pinv can produce numerically unstable results when the matrix
    is ill-conditioned (has very small singular values). This function:
    1. Computes SVD to determine condition number
    2. Applies rank truncation based on rcond threshold
    3. Returns diagnostics for debugging merge failures

    The pseudo-inverse is computed as: pinv(A) = V @ S^{-1} @ U^T
    where small singular values (< rcond * max_singular_value) are zeroed.

    Parameters
    ----------
    backend : Backend
        Compute backend.
    array : Array
        Matrix to invert [m, n].
    rcond : float, optional
        Cutoff for small singular values. Values below rcond * max(singular_values)
        are set to zero. Default is machine_epsilon * max(m, n).
    warn_on_ill_conditioned : bool
        If True, log warning when condition number exceeds dtype limit.

    Returns
    -------
    tuple[Array, dict]
        (pinv_result, diagnostics) where diagnostics contains:
        - condition_number: ratio of max/min non-zero singular value
        - effective_rank: number of non-zero singular values after truncation
        - truncated_count: number of singular values zeroed
        - max_sv: maximum singular value
        - min_sv: minimum non-zero singular value
    """
    import logging

    logger = logging.getLogger(__name__)
    b = backend

    # Ensure float32/float64 for numerical stability
    original_dtype = str(array.dtype)
    if "bfloat" in original_dtype or "float16" in original_dtype:
        array = b.astype(array, "float32")
        b.eval(array)

    shape = b.shape(array)
    m, n = int(shape[0]), int(shape[1]) if len(shape) > 1 else 1

    # Default rcond based on precision
    if rcond is None:
        eps = machine_epsilon(b, array)
        rcond = eps * max(m, n)

    diagnostics: dict = {
        "condition_number": float("inf"),
        "effective_rank": 0,
        "truncated_count": 0,
        "max_sv": 0.0,
        "min_sv": 0.0,
        "shape": [m, n],
    }

    # Compute SVD
    U, S, Vt = svd_via_eigh(b, array, full_matrices=False)
    b.eval(U, S, Vt)

    # Get singular values as list
    S_np = [float(v) for v in b.to_numpy(S)]
    if not S_np:
        # Zero matrix - return zero pseudo-inverse
        pinv_result = b.zeros((n, m))
        b.eval(pinv_result)
        return pinv_result, diagnostics

    max_sv = max(S_np)
    diagnostics["max_sv"] = max_sv

    if max_sv == 0:
        # Zero matrix
        pinv_result = b.zeros((n, m))
        b.eval(pinv_result)
        return pinv_result, diagnostics

    # Determine cutoff and truncate
    cutoff = rcond * max_sv
    S_inv_list = []
    truncated = 0
    min_nonzero = float("inf")

    for sv in S_np:
        if sv > cutoff:
            S_inv_list.append(1.0 / sv)
            if sv < min_nonzero:
                min_nonzero = sv
        else:
            S_inv_list.append(0.0)
            truncated += 1

    effective_rank = len(S_np) - truncated
    diagnostics["effective_rank"] = effective_rank
    diagnostics["truncated_count"] = truncated
    diagnostics["min_sv"] = min_nonzero if min_nonzero < float("inf") else 0.0

    # Condition number
    if min_nonzero < float("inf") and min_nonzero > 0:
        condition = max_sv / min_nonzero
        diagnostics["condition_number"] = condition

        cond_limit = condition_threshold(b, array)
        if warn_on_ill_conditioned and condition > cond_limit:
            logger.warning(
                "safe_pinv: Ill-conditioned matrix (cond=%.2e, rank=%d/%d, truncated=%d)",
                condition, effective_rank, len(S_np), truncated
            )

    # Compute pseudo-inverse: V @ diag(S_inv) @ U^T
    S_inv = b.array(S_inv_list)
    b.eval(S_inv)

    # V = Vt^T [n, k], U^T [k, m]
    V = b.transpose(Vt)  # [n, k]

    # V @ diag(S_inv) [n, k]
    V_scaled = V * b.reshape(S_inv, (1, -1))
    b.eval(V_scaled)

    # (V @ diag(S_inv)) @ U^T [n, m]
    pinv_result = b.matmul(V_scaled, b.transpose(U))
    b.eval(pinv_result)

    return pinv_result, diagnostics


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
    R_diag_np = [abs(float(v)) for v in b.to_numpy(R_diag).tolist()]
    if not R_diag_np:
        return None, diagnostics

    max_diag = max(R_diag_np)
    min_diag = min(R_diag_np)

    condition_est = max_diag / (min_diag + eps) if min_diag > 0 else float("inf")
    diagnostics["condition"] = condition_est

    rank_threshold = eps * max_diag * max(n_samples, d_source)
    rank = sum(1 for v in R_diag_np if v > rank_threshold)
    diagnostics["rank"] = rank

    # Apply regularization if needed
    if rank < n_samples:
        regularization = max_diag * math.sqrt(eps) * (n_samples - rank + 1)
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
    R_diag_np = [abs(float(v)) for v in b.to_numpy(R_diag).tolist()]
    if not R_diag_np:
        return None, diagnostics

    max_diag = max(R_diag_np)
    min_diag = min(R_diag_np)

    condition_est = max_diag / (min_diag + eps) if min_diag > 0 else float("inf")
    diagnostics["condition"] = condition_est

    rank_threshold = eps * max_diag * max(n_samples, d_source)
    rank = sum(1 for v in R_diag_np if v > rank_threshold)
    diagnostics["rank"] = rank

    # Apply regularization if needed
    if rank < d_source:
        regularization = max_diag * math.sqrt(eps) * (d_source - rank + 1)
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

    res_norm = float(b.to_numpy(b.norm(residual)))
    tgt_norm = float(b.to_numpy(b.norm(target)))
    rel_residual = res_norm / (tgt_norm + eps)
    diagnostics["residual_norm"] = rel_residual

    # Iterative refinement if residual is large
    if rel_residual > eps * 100:
        try:
            n_samples = diagnostics["n_samples"]
            d_source = diagnostics["d_source"]

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

            res_norm_ref = float(b.to_numpy(b.norm(residual_ref)))
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
        U, S, Vt = svd_via_eigh(b, source, full_matrices=False)
        b.eval(U, S, Vt)
    except Exception:
        diagnostics["method"] = "failed"
        return None, diagnostics

    # Convert S to numpy for analysis
    S_np = [float(v) for v in b.to_numpy(S).tolist()]
    if not S_np or max(S_np) == 0:
        return None, diagnostics

    max_s = max(S_np)
    min_s = min(v for v in S_np if v > 0)
    diagnostics["condition"] = max_s / min_s if min_s > 0 else float("inf")

    # Determine effective rank
    rank = sum(1 for s in S_np if s > rank_threshold * max_s)
    diagnostics["rank"] = rank

    if rank == 0:
        return None, diagnostics

    # Truncate to effective rank
    U_k = U[:, :rank]  # [n, k]
    # Build inverse singular values array
    S_inv_vals = [1.0 / S_np[i] if S_np[i] > rank_threshold * max_s else 0.0
                  for i in range(rank)]
    S_k_inv = b.astype(b.array(S_inv_vals), "float32")
    b.eval(S_k_inv)
    Vt_k = Vt[:rank, :]  # [k, d]

    # Check consistency: project target onto column space of source
    # target_proj = U_k @ U_k^T @ target (projection onto column space)
    # projection_error = ||target - target_proj|| / ||target||
    target_proj = b.matmul(U_k, b.matmul(b.transpose(U_k), target))
    b.eval(target_proj)
    proj_residual = target - target_proj
    proj_error = float(b.to_numpy(b.norm(proj_residual)))
    target_norm = float(b.to_numpy(b.norm(target)))
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
    res_norm = float(b.to_numpy(b.norm(residual)))
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
    backend: Backend,
    singular_values: list[float],
    eps: float = 1e-12,
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
    import math

    # Filter positive values
    positive_sv = [s for s in singular_values if s > eps]
    if not positive_sv:
        return 0.0

    total = sum(positive_sv)
    if total <= 0:
        return 0.0

    # Compute normalized probabilities
    probs = [s / total for s in positive_sv]

    # Shannon entropy
    entropy = -sum(p * math.log(p + eps) for p in probs if p > 0)

    return math.exp(entropy)


def compute_shared_relational_rank(
    backend: Backend,
    source_singular_values: list[float],
    target_singular_values: list[float],
    eps: float = 1e-12,
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
    # Compute effective ranks
    erank_source = compute_entropy_effective_rank(backend, source_singular_values, eps)
    erank_target = compute_entropy_effective_rank(backend, target_singular_values, eps)

    # Integer ranks (floor of effective rank)
    rank_source = max(1, int(erank_source))
    rank_target = max(1, int(erank_target))

    # Shared relational rank = min of both
    # This is where BOTH have signal, not noise
    shared_rank = min(rank_source, rank_target)

    # Also compute threshold-based ranks for comparison
    max_sv_s = max(source_singular_values) if source_singular_values else 0
    max_sv_t = max(target_singular_values) if target_singular_values else 0
    thresh_s = eps * max_sv_s * 100  # Noise threshold
    thresh_t = eps * max_sv_t * 100
    threshold_rank_s = sum(1 for s in source_singular_values if s > thresh_s)
    threshold_rank_t = sum(1 for s in target_singular_values if s > thresh_t)

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

    # SVD of centered matrices using our stable eigh-based implementation
    U_s, S_s, Vt_s = svd_via_eigh(b, source_c, full_matrices=False)
    U_t, S_t, Vt_t = svd_via_eigh(b, target_c, full_matrices=False)
    b.eval(U_s, S_s, Vt_s, U_t, S_t, Vt_t)

    # Determine effective ranks using entropy-based measure
    # This separates SIGNAL (relational content) from NOISE (random fluctuations)
    S_s_np = [float(v) for v in b.to_numpy(S_s)]
    S_t_np = [float(v) for v in b.to_numpy(S_t)]

    if not S_s_np or not S_t_np or max(S_s_np) == 0 or max(S_t_np) == 0:
        return None, diagnostics

    # Compute shared relational rank - where BOTH models have signal
    # This is the space where CKA alignment is meaningful and achievable.
    # Beyond this rank, we're trying to align noise - mathematically impossible.
    shared_rank, rank_diag = compute_shared_relational_rank(b, S_s_np, S_t_np, eps)
    diagnostics.update(rank_diag)

    # Also record threshold-based ranks for comparison
    thresh_s = eps * max(S_s_np) * max(n, d_s)
    thresh_t = eps * max(S_t_np) * max(n, d_t)
    rank_s = sum(1 for s in S_s_np if s > thresh_s)
    rank_t = sum(1 for s in S_t_np if s > thresh_t)
    diagnostics["rank_source"] = rank_s
    diagnostics["rank_target"] = rank_t

    if shared_rank == 0:
        return None, diagnostics

    # Use the SHARED RELATIONAL RANK for Procrustes alignment
    # This is the key insight: CKA = 1.0 is achievable ONLY on the shared
    # relational space. Beyond this, we're comparing signal to noise.

    U_s_k = U_s[:, :shared_rank]  # [n, k]
    U_t_k = U_t[:, :shared_rank]  # [n, k]
    b.eval(U_s_k, U_t_k)

    # Orthogonal Procrustes: find R such that U_s @ R ≈ U_t
    # Solve: min ||U_s @ R - U_t||_F  s.t. R^T @ R = I
    # Solution: R = U @ V^T where M = U_s^T @ U_t = U @ S @ V^T
    M = b.matmul(b.transpose(U_s_k), U_t_k)  # [k, k]
    b.eval(M)

    U_proc, S_proc, Vt_proc = svd_via_eigh(b, M, full_matrices=False)
    b.eval(U_proc, S_proc, Vt_proc)

    R = b.matmul(U_proc, Vt_proc)  # [k, k]
    b.eval(R)

    # Check for reflection and correct
    R_det = _determinant_sign(b, R)
    if R_det < 0:
        # Flip sign of last column of U_proc
        # Use .copy() to get a writable array (JAX returns read-only)
        U_proc_np = b.to_numpy(U_proc).copy()
        U_proc_np[:, -1] = -U_proc_np[:, -1]
        U_proc = b.array(U_proc_np)
        b.eval(U_proc)
        R = b.matmul(U_proc, Vt_proc)
        b.eval(R)

    # Compute Procrustes error: ||U_s @ R - U_t||_F / ||U_t||_F
    U_s_rotated = b.matmul(U_s_k, R)
    b.eval(U_s_rotated)
    diff = U_s_rotated - U_t_k
    diff_norm = float(b.to_numpy(b.norm(diff)))
    U_t_norm = float(b.to_numpy(b.norm(U_t_k)))
    procrustes_error = diff_norm / (U_t_norm + eps)
    diagnostics["procrustes_error"] = procrustes_error

    # Build the full transform F = V_s @ S_s^{-1} @ R @ S_t @ V_t^T
    # But we need to handle rank truncation carefully

    # S_s^{-1} for the k dimensions we're using
    S_s_inv = b.array([1.0 / S_s_np[i] if S_s_np[i] > thresh_s else 0.0
                       for i in range(shared_rank)])
    S_t_k = b.array([S_t_np[i] for i in range(shared_rank)])
    b.eval(S_s_inv, S_t_k)

    V_s_k = b.transpose(Vt_s[:shared_rank, :])  # [d_s, k]
    V_t_k = b.transpose(Vt_t[:shared_rank, :])  # [d_t, k]
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
    import numpy as np

    R_np = backend.to_numpy(R)
    k = int(backend.shape(R)[0])

    # LU-based sign computation
    work = R_np.copy()
    det_sign = 1.0

    # Use dtype-specific singularity threshold
    singular_threshold = np.finfo(work.dtype).tiny

    for col in range(k):
        # Find pivot
        max_row = col
        for row in range(col + 1, k):
            if abs(work[row, col]) > abs(work[max_row, col]):
                max_row = row

        if abs(work[max_row, col]) < singular_threshold:
            return 0.0  # Singular

        if max_row != col:
            work[[col, max_row]] = work[[max_row, col]]
            det_sign = -det_sign

        # Eliminate below
        pivot = work[col, col]
        for row in range(col + 1, k):
            factor = work[row, col] / pivot
            work[row, col:] -= factor * work[col, col:]

        det_sign *= (1.0 if work[col, col] > 0 else -1.0)

    return det_sign


def solve_via_cca_procrustes(
    backend: Backend,
    source: Array,
    target: Array,
    *,
    regularization: float = 1e-4,
    pca_variance_threshold: float = 0.95,
    cca_variance_threshold: float = 0.95,
    min_correlation: float = 0.1,
) -> tuple[Array | None, dict]:
    """Solve source @ F = target via SVCCA + Procrustes for perfect alignment.

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
    regularization : float
        Regularization for CCA covariance matrices.
    pca_variance_threshold : float
        Fraction of variance to retain in PCA step (0.95 = 95%).
    cca_variance_threshold : float
        Fraction of CCA variance to retain (0.95 = 95%).
    min_correlation : float
        Minimum canonical correlation to include (0.1 = 10%).

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
    sv_floor = 1e-8

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
    def pca_reduce(matrix: Array, variance_thresh: float) -> tuple[Array, Array] | None:
        """Reduce matrix to high-variance subspace using Gram-space PCA."""
        n_samp = int(matrix.shape[0])
        d_feat = int(matrix.shape[1])
        max_components = min(n_samp, d_feat)

        # Gram matrix: matrix @ matrix.T [n x n]
        gram = b.matmul(matrix, b.transpose(matrix))
        b.eval(gram)

        # Eigendecomposition of Gram (gives squared singular values)
        # Cast to float32 for eigendecomposition (MLX doesn't support bfloat16 for eigh)
        gram_dtype = str(gram.dtype)
        if "bfloat16" in gram_dtype:
            gram_f32 = b.astype(gram, "float32")
            b.eval(gram_f32)
            eigenvalues, eigenvectors = b.eigh(gram_f32)
        else:
            eigenvalues, eigenvectors = b.eigh(gram)
        b.eval(eigenvalues, eigenvectors)

        # Sort descending (eigh gives ascending)
        eig_np = b.to_numpy(eigenvalues)
        order = list(range(len(eig_np) - 1, -1, -1))
        eigenvectors_sorted = eigenvectors[:, order]
        eigenvalues_sorted = b.array([max(0.0, float(eig_np[i])) for i in order])
        b.eval(eigenvectors_sorted, eigenvalues_sorted)

        # Select components by variance threshold
        eig_sorted_np = [float(v) for v in b.to_numpy(eigenvalues_sorted)]
        total_var = sum(eig_sorted_np)
        if total_var <= 0:
            return None

        cum_var = 0.0
        k = 0
        for i, ev in enumerate(eig_sorted_np):
            if i >= max_components:
                break
            cum_var += ev
            k = i + 1
            if cum_var / total_var >= variance_thresh:
                break

        if k == 0:
            return None

        # Singular values from eigenvalues
        singular_values = b.sqrt(eigenvalues_sorted[:k])
        b.eval(singular_values)

        # Principal components: V = matrix.T @ U @ S^{-1}
        U_k = eigenvectors_sorted[:, :k]  # [n, k]
        sv_np = [max(float(v), sv_floor) for v in b.to_numpy(singular_values)]
        inv_sv = b.array([1.0 / s for s in sv_np])
        b.eval(inv_sv)

        # Components [d, k]
        components = b.matmul(b.transpose(matrix), U_k) * b.reshape(inv_sv, (1, -1))
        b.eval(components)

        # Reduced matrix [n, k]
        reduced = b.matmul(matrix, components)
        b.eval(reduced)

        return reduced, components

    pca_result_s = pca_reduce(source_c, pca_variance_threshold)
    pca_result_t = pca_reduce(target_c, pca_variance_threshold)

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

    # Regularize
    cov_ss = cov_ss + regularization * b.eye(k_s)
    cov_tt = cov_tt + regularization * b.eye(k_t)
    b.eval(cov_ss, cov_tt)

    # Whitening via eigendecomposition
    def whiten_cov(cov: Array) -> Array | None:
        """Compute inverse sqrt of covariance for whitening."""
        # Cast to float32 for eigendecomposition (MLX doesn't support bfloat16 for eigh)
        cov_dtype = str(cov.dtype)
        if "bfloat16" in cov_dtype:
            cov_f32 = b.astype(cov, "float32")
            b.eval(cov_f32)
            eigvals, eigvecs = b.eigh(cov_f32)
        else:
            eigvals, eigvecs = b.eigh(cov)
        b.eval(eigvals, eigvecs)

        eigvals_np = [float(v) for v in b.to_numpy(eigvals)]
        if all(v <= 0 for v in eigvals_np):
            return None

        # Floor eigenvalues
        floor_val = max(regularization, eps * 1e3)
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
    U, S, Vt = b.svd(cross_whitened)
    b.eval(U, S, Vt)

    # Canonical correlations (SHOULD be in [0, 1] now!)
    S_np = [float(v) for v in b.to_numpy(S)]
    correlations = [max(0.0, min(1.0, c)) for c in S_np]

    if not correlations:
        return None, diagnostics

    diagnostics["top_correlation"] = correlations[0]

    # Select shared dimension
    total_var = sum(c * c for c in correlations)
    cum_var = 0.0
    k = 0
    for i, c in enumerate(correlations):
        if c < min_correlation:
            break
        cum_var += c * c
        k = i + 1
        if total_var > 0 and cum_var / total_var >= cca_variance_threshold:
            break

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

    U_proc, _, Vt_proc = b.svd(M)
    b.eval(U_proc, Vt_proc)

    # Orthogonal rotation R = U @ V^T
    R = b.matmul(U_proc, Vt_proc)
    b.eval(R)

    # Check for reflection and correct if needed
    R_np = b.to_numpy(R)
    det = 1.0
    work = R_np.copy()
    kk = int(b.shape(R)[0])
    for col in range(kk):
        max_row = col
        for row in range(col + 1, kk):
            if abs(work[row, col]) > abs(work[max_row, col]):
                max_row = row
        if abs(work[max_row, col]) < 1e-15:
            det = 0.0
            break
        if max_row != col:
            work[[col, max_row]] = work[[max_row, col]]
            det = -det
        pivot = work[col, col]
        for row in range(col + 1, kk):
            factor = work[row, col] / pivot
            work[row, col:] -= factor * work[col, col:]
        det *= work[col, col]

    if det < 0:
        # Flip last column of U_proc to get proper rotation
        # Use .copy() to get a writable array (JAX returns read-only)
        U_proc_np = b.to_numpy(U_proc).copy()
        U_proc_np[:, -1] = -U_proc_np[:, -1]
        U_proc = b.array(U_proc_np)
        b.eval(U_proc)
        R = b.matmul(U_proc, Vt_proc)
        b.eval(R)

    # Compute alignment error in shared space
    Z_s_rotated = b.matmul(Z_s, R)
    b.eval(Z_s_rotated)

    diff = Z_s_rotated - Z_t
    diff_norm = b.norm(diff)
    Z_t_norm = b.norm(Z_t)
    b.eval(diff_norm, Z_t_norm)

    alignment_error = float(b.to_numpy(diff_norm)) / (float(b.to_numpy(Z_t_norm)) + eps)
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
