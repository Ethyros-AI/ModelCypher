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


def svd_via_eigh(
    backend: Backend,
    array: Array,
    *,
    full_matrices: bool = False,
) -> tuple[Array, Array, Array]:
    """Compute SVD via symmetric eigendecomposition (GPU-stable, no SVD calls).

    This uses eigendecomposition of A^T A to obtain right singular vectors and
    singular values, and completes the left basis if needed. For rank-deficient
    matrices, the null-space basis is filled from A A^T eigenvectors so that
    U and V remain orthonormal.
    """
    b = backend
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
        return U, S, Vt

    return U[:, :k], S[:k], Vt[:k, :]


def solve_full_row_rank_via_qr(
    backend: Backend,
    source: Array,
    target: Array,
) -> tuple[Array | None, dict]:
    """Solve source @ F = target for minimum-norm F via QR factorization.

    This uses QR factorization of source^T to avoid condition number squaring.
    For the underdetermined system A @ x = b where A is [n, d] with n <= d
    (full row rank), the minimum-norm solution is:
        x = A^T (A A^T)^-1 b

    The normal equations approach squares the condition number: κ(A A^T) = κ(A)².
    This QR-based approach maintains κ(R) = κ(A), providing better stability.

    Algorithm:
        A^T = Q @ R  where Q [d, n], R [n, n]
        A = R^T @ Q^T
        A @ F = B  →  R^T @ Q^T @ F = B
        Let Y = Q^T @ F, then R^T @ Y = B
        Solve: Y = R^{-T} @ B
        Then: F = Q @ Y

    Parameters
    ----------
    backend : Backend
        Compute backend.
    source : Array
        Source matrix [n_samples, d_source] with n_samples <= d_source.
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
    }

    if n_samples == 0 or d_source == 0:
        return None, diagnostics

    # QR factorization of source^T: source^T = Q @ R
    # source^T is [d_source, n_samples] → Q [d_source, n_samples], R [n_samples, n_samples]
    try:
        Q, R = b.qr(b.transpose(source))
        b.eval(Q, R)
    except Exception:
        return None, diagnostics

    # Check R conditioning via diagonal elements
    R_diag = b.diag(R)
    b.eval(R_diag)
    R_diag_np = [abs(float(v)) for v in b.to_numpy(R_diag).tolist()]
    if not R_diag_np:
        return None, diagnostics

    max_diag = max(R_diag_np)
    min_diag = min(R_diag_np)

    # Condition estimate (for upper triangular, cond ≈ max/min diagonal)
    condition_est = max_diag / (min_diag + eps) if min_diag > 0 else float("inf")
    diagnostics["condition"] = condition_est

    # Compute rank from diagonal
    rank_threshold = eps * max_diag * max(n_samples, d_source)
    rank = sum(1 for v in R_diag_np if v > rank_threshold)
    diagnostics["rank"] = rank

    # Determine regularization based on conditioning
    # For rank-deficient systems, use stronger regularization to get approximate solution
    if rank < n_samples:
        # Rank-deficient: use regularization proportional to the gap
        # This gives a minimum-norm approximate solution
        regularization = max_diag * math.sqrt(eps) * (n_samples - rank + 1)
        R_reg = R + regularization * b.eye(n_samples)
        b.eval(R_reg)
        diagnostics["method"] = "qr_rank_deficient"
    elif min_diag < eps * max_diag:
        # Full rank but ill-conditioned: use light regularization
        regularization = eps * max_diag
        R_reg = R + regularization * b.eye(n_samples)
        b.eval(R_reg)
        diagnostics["method"] = "qr_regularized"
    else:
        # Well-conditioned: no regularization needed
        R_reg = R
        diagnostics["method"] = "qr"

    # Solve R^T @ Y = target for Y
    # R^T is lower triangular [n_samples, n_samples]
    # target is [n_samples, d_target]
    try:
        R_T = b.transpose(R_reg)
        Y = b.solve(R_T, target)
        b.eval(Y)
    except Exception:
        # Solve failed (singular even with regularization)
        diagnostics["method"] = "failed"
        return None, diagnostics

    # F = Q @ Y
    # Q is [d_source, n_samples], Y is [n_samples, d_target]
    # F is [d_source, d_target]
    F = b.matmul(Q, Y)
    b.eval(F)

    # Compute residual to verify solution quality
    reconstructed = b.matmul(source, F)
    residual = reconstructed - target
    b.eval(reconstructed, residual)

    res_norm = float(b.to_numpy(b.norm(residual)))
    tgt_norm = float(b.to_numpy(b.norm(target)))
    rel_residual = res_norm / (tgt_norm + eps)
    diagnostics["residual_norm"] = rel_residual

    # If residual is too large, try iterative refinement
    if rel_residual > eps * 100:
        # One round of iterative refinement
        try:
            # Solve for correction: source @ delta_F = residual
            # Using same QR factorization
            delta_Y = b.solve(R_T, -residual)
            delta_F = b.matmul(Q, delta_Y)
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


def solve_via_gram_alignment(
    backend: Backend,
    source: Array,
    target: Array,
) -> tuple[Array | None, dict]:
    """Solve for F such that Gram(source @ F) = Gram(target) via SVD alignment.

    The key insight: both source and target represent the same n_samples concepts.
    Their relational geometry (Gram matrices) should be identical. We find F by
    aligning the left singular vectors (sample structure) via Procrustes.

    Mathematical derivation:
    - source_c = U_s @ S_s @ V_s^T  (centered, SVD)
    - target_c = U_t @ S_t @ V_t^T  (centered, SVD)
    - U_s and U_t span the SAME sample space (invariant concept geometry)
    - Find R (orthogonal) such that U_s @ R ≈ U_t
    - Then F = V_s @ S_s^{-1} @ R @ S_t @ V_t^T achieves:
      source_c @ F = U_s @ R @ S_t @ V_t^T ≈ target_c
      => Gram(source @ F) ≈ Gram(target)
      => CKA ≈ 1.0

    This works for ANY dimension combination because we're aligning in sample
    space [n, n], not feature space [d_s, d_t].
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

    # Determine effective ranks
    S_s_np = [float(v) for v in b.to_numpy(S_s)]
    S_t_np = [float(v) for v in b.to_numpy(S_t)]

    if not S_s_np or not S_t_np or max(S_s_np) == 0 or max(S_t_np) == 0:
        return None, diagnostics

    thresh_s = eps * max(S_s_np) * max(n, d_s)
    thresh_t = eps * max(S_t_np) * max(n, d_t)

    rank_s = sum(1 for s in S_s_np if s > thresh_s)
    rank_t = sum(1 for s in S_t_np if s > thresh_t)
    diagnostics["rank_source"] = rank_s
    diagnostics["rank_target"] = rank_t

    if rank_s == 0 or rank_t == 0:
        return None, diagnostics

    # Truncate to shared rank for Procrustes alignment
    shared_rank = min(rank_s, rank_t)

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
        U_proc_np = b.to_numpy(U_proc)
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
    R_np = backend.to_numpy(R)
    k = int(backend.shape(R)[0])

    # LU-based sign computation
    work = R_np.copy()
    det_sign = 1.0

    for col in range(k):
        # Find pivot
        max_row = col
        for row in range(col + 1, k):
            if abs(work[row, col]) > abs(work[max_row, col]):
                max_row = row

        if abs(work[max_row, col]) < 1e-15:
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
        U_proc_np = b.to_numpy(U_proc)
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
