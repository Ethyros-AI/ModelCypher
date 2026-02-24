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

"""Matrix decomposition and GPU-accelerated linear algebra utilities."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .precision import (
    _promote_precision,
    condition_threshold,
    division_epsilon,
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
    svd_rank_threshold,
    tiny_value,
)
from .scalars import all_finite, sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Matrix Decomposition
# =============================================================================


def power_iteration_eigh(
    backend: "Backend",
    matrix: "Array",
    k: int = 10,
    use_ritz: bool = True,
) -> tuple["Array", "Array"]:
    """Compute eigendecomposition via backend.eigh."""
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
    """Compute SVD via backend.svd."""
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

    # Validate input before SVD - NaN/Inf can crash backend LAPACK calls
    if not all_finite(A, b):
        raise ValueError(
            f"geodesic_svd: Input contains NaN/Inf values. "
            f"Shape={shape}, dtype={dtype}, batch_shape={batch_shape}"
        )

    # Check matrix Frobenius norm - truly zero matrices should return zeros
    # Use tiny (smallest positive float) as threshold, NOT machine epsilon
    # A matrix with norm < tiny * max_dim is effectively all zeros
    tiny = tiny_value(b, A)
    frob_norm_sq = b.sum(A * A)
    b.eval(frob_norm_sq)
    frob_norm = float(b.to_scalar(b.sqrt(frob_norm_sq)))

    # If Frobenius norm is below tiny threshold, return zeros directly
    # This avoids numerical artifacts from SVD on near-zero matrices
    zero_threshold = tiny * float(m * n)
    if frob_norm < zero_threshold:
        logger.debug(
            "geodesic_svd: Matrix has near-zero Frobenius norm (%.2e < %.2e). "
            "Returning zeros. Shape=%s",
            frob_norm, zero_threshold, shape,
        )
        U = b.zeros(batch_shape + (m, max_rank), dtype=dtype)
        S = b.zeros(batch_shape + (max_rank,), dtype=dtype)
        Vt = b.zeros(batch_shape + (max_rank, n), dtype=dtype)
        return U, S, Vt

    # No pre-emptive regularization - let LAPACK handle the matrix directly
    # The NaN/Inf check above catches the true crash cases
    A_reg = A

    # Compute SVD with error handling for backend LAPACK crashes
    # NOTE: Python's try/except cannot catch C++ exceptions from backend LAPACK.
    # If LAPACK throws std::runtime_error, the process will terminate.
    # The pre-validation above is the only defense.
    try:
        U_full, S_full, Vt_full = b.svd(A_reg, compute_uv=True)
        b.eval(U_full, S_full, Vt_full)
    except Exception as e:
        # Log detailed info before re-raising
        logger.error(
            "SVD FAILED: shape=%s, dtype=%s, batch_shape=%s, m=%d, n=%d, "
            "frob_norm=%.2e, error=%s",
            shape, dtype, batch_shape, m, n, frob_norm, e,
        )
        raise RuntimeError(
            f"SVD failed for matrix shape {shape} (m={m}, n={n}). "
            f"This may be a backend bug or memory issue. Original error: {e}"
        ) from e

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


def orthogonalize_alignment_full(
    alignment: "Array",
    backend: "Backend",
) -> tuple["Array", "Array", "Array", "Array"]:
    """Extract the orthogonal factor and full singular value spectrum from alignment.

    Uses SVD(F)=U S Vt and returns:
    - U_orth: The orthogonal factor (closest rotation to F)
    - S: Full singular value vector [k] where k = min(m, n)
    - U: Left singular vectors [m, k]
    - Vt: Right singular vectors [k, n]

    This enables direction-dependent scale correction by providing the
    full spectral structure, not just the mean singular value.

    For backward compatibility, use orthogonalize_alignment() which
    returns (U_orth, mean_scale) as before.

    Mathematical note:
        Polar decomposition: F = U_orth @ P where P = sqrt(F.T @ F)
        The singular values S are the eigenvalues of P.
        Per-direction scaling: direction i is scaled by S[i].
    """
    b = backend
    F = _promote_precision(b.array(alignment), b)
    b.eval(F)

    m, n = int(F.shape[0]), int(F.shape[1])
    k = min(m, n)

    # Compute SVD
    U, S, Vt = geodesic_svd(b, F, k=k)
    b.eval(U, S, Vt)

    # Extract orthogonal part via polar decomposition
    # U_orth = U @ Vt gives the closest orthogonal matrix in O(n)
    U_orth = b.matmul(U, Vt)
    b.eval(U_orth)

    # Enforce det=+1 to get SO(n) (proper rotation) not O(n) (may be reflection)
    # If det < 0, flip the sign of the last column of U before computing U @ Vt
    # This ensures we get a rotation, not a reflection
    if m == n:  # Only check determinant for square matrices
        det_val = b.det(U_orth)
        b.eval(det_val)
        if float(b.to_scalar(det_val)) < 0:
            # Flip last column of U to get det=+1
            U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            b.eval(U_fixed)
            U_orth = b.matmul(U_fixed, Vt)
            b.eval(U_orth)

    return U_orth, S, U, Vt


def orthogonalize_alignment(
    alignment: "Array",
    backend: "Backend",
) -> tuple["Array", float]:
    """Extract the orthogonal factor from an alignment transform.

    Uses SVD(F)=U S Vt and returns U @ Vt plus the mean singular value as a
    scale summary.

    Note: This function loses spectral information by returning only the mean
    singular value. For direction-dependent scale correction, use
    orthogonalize_alignment_full() which returns the full singular value vector.
    """
    b = backend
    U_orth, S, _, _ = orthogonalize_alignment_full(alignment, b)

    # Compute scale factor as mean of singular values
    # This measures how much F scales on average
    eps = division_epsilon(b, S)
    n_singular = max(1, int(S.shape[0]))
    scale_factor = float(b.to_scalar(b.sum(S))) / n_singular

    return U_orth, max(scale_factor, float(eps))


def svd_auto_rank(
    singular_values: "Array",
    backend: "Backend",
    max_dim: int | None = None,
) -> int:
    """Choose SVD rank by precision-derived numeric threshold."""
    b = backend
    S_arr = b.array(singular_values)
    S = b.astype(S_arr, precision_dtype(b, reference=S_arr))
    b.eval(S)

    n = int(S.shape[0])
    if n == 0:
        return 0

    max_s_arr = b.max(S)
    b.eval(max_s_arr)
    max_s = float(b.to_scalar(max_s_arr))
    if max_s <= 0.0:
        return 0

    max_dim = max_dim or n
    rank_scale = svd_rank_threshold(b, S, max_dim)
    threshold = max_s * rank_scale
    rank_mask = S > threshold
    rank_arr = b.sum(b.astype(rank_mask, "int32"))
    b.eval(rank_arr)
    return int(b.to_scalar(rank_arr))


def geodesic_pinv(backend: "Backend", array: "Array") -> "Array":
    """Compute Moore-Penrose pseudoinverse with session cache reuse."""
    b = backend
    A = _promote_precision(b.array(array), b)
    b.eval(A)

    from modelcypher.core.domain.cache import ComputationCache

    cache = ComputationCache.shared()
    A_pinv = cache.get_or_compute_pinv(A, b)

    return A_pinv


def null_space_projector(backend: "Backend", A: "Array") -> "Array":
    """Compute the orthogonal projector onto the null space of A.

    This computes P = I - V_r @ V_r^T where V_r contains the right singular
    vectors corresponding to non-zero singular values. The result projects
    vectors onto the null space (kernel) of A.

    This is NOT an Ax=b problem - it requires SVD-based computation. Use this
    for null-space projection operations like boundary constraints in transplant.

    Mathematical background:
        For A with SVD: A = U @ diag(S) @ V^T
        Row space of A = span of rows of V corresponding to non-zero S
        Null space of A = span of rows of V corresponding to zero S
        Projector onto null space: P = I - V_r @ V_r^T

    Args:
        backend: Backend for tensor operations.
        A: Matrix [n, d] to compute null space projector for.

    Returns:
        Projector matrix P [d, d] such that P @ x lies in null(A) for any x.
    """
    b = backend
    A = _promote_precision(b.array(A), b)
    b.eval(A)

    _n, d = int(A.shape[0]), int(A.shape[1])
    eps = machine_epsilon(b, A)
    sqrt_eps = sqrt_scalar(eps, b)

    # Compute SVD
    U, S, Vt = geodesic_svd(b, A)
    b.eval(U, S, Vt)

    # Determine numerical rank (number of significant singular values)
    S_max = b.max(S)
    b.eval(S_max)
    S_max_val = float(b.to_scalar(S_max))
    threshold = S_max_val * sqrt_eps

    # Count significant singular values
    rank_mask = S > threshold
    rank_count = b.sum(b.astype(rank_mask, "int32"))
    b.eval(rank_count)
    r = int(b.to_scalar(rank_count))

    if r == 0:
        # All singular values are zero - null space is entire space
        return b.eye(d, dtype=A.dtype)

    if r >= d:
        # Full rank - null space is empty (just zero vector)
        return b.zeros((d, d), dtype=A.dtype)

    # Extract right singular vectors for non-zero singular values
    # Vt is [min(n,d), d], V_r is [r, d]
    V_r = Vt[:r, :]
    b.eval(V_r)

    # Projector onto column space of A^T (row space of A): V_r^T @ V_r
    # Projector onto null space: I - V_r^T @ V_r
    I = b.eye(d, dtype=A.dtype)
    col_space_proj = b.matmul(b.transpose(V_r), V_r)
    null_proj = I - col_space_proj
    b.eval(null_proj)

    return null_proj


def numerical_rank_truncated_lstsq(
    backend: "Backend",
    source: "Array",
    target: "Array",
) -> tuple["Array", int, int, int, float, float]:
    """Solve least squares with numerical-rank truncation.

    Drops singular values below sigma_max * sqrt(eps) and solves
    F = pinv(A_k) @ B in the truncated space.

    Args:
        backend: Backend for tensor operations.
        source: Source activations [n_samples, d_source].
        target: Target activations [n_samples, d_target].

    Returns:
        Tuple (F, source_rank, target_rank, alignment_rank, condition_number, residual).

        residual is ||source @ F - target||_F / ||target||_F, the relative
        alignment error. If residual < sqrt(eps), the alignment is within
        numerical precision. If residual >> sqrt(eps), the relationship
        between source and target is not well-approximated by a linear
        transform.
    """
    b = backend

    # Promote to highest precision available
    A = _promote_precision(b.array(source), b)
    B = _promote_precision(b.array(target), b)
    b.eval(A, B)

    _n_samples, d_source = int(A.shape[0]), int(A.shape[1])
    _, d_target = int(B.shape[0]), int(B.shape[1])

    # Machine precision threshold: sqrt(eps_machine)
    # This is THE threshold below which singular values are noise
    eps = machine_epsilon(b, A)
    precision_thresh = sqrt_scalar(eps, b)

    # SVD of source: A = U @ diag(S) @ Vt
    U_s, S_s, Vt_s = geodesic_svd(b, A)
    b.eval(U_s, S_s, Vt_s)

    # Numerical rank of source: count(sigma > sigma_max * sqrt(eps))
    if int(S_s.shape[0]) > 0:
        max_s_source = float(b.to_scalar(S_s[0]))  # Singular values are sorted desc
        thresh_source = max_s_source * precision_thresh
        source_rank_mask = S_s > thresh_source
        source_rank_arr = b.sum(b.astype(source_rank_mask, "int32"))
        b.eval(source_rank_arr)
        source_rank = int(b.to_scalar(source_rank_arr))
    else:
        source_rank = 0
        max_s_source = 0.0

    # SVD of target: B = U @ diag(S) @ Vt
    U_t, S_t, Vt_t = geodesic_svd(b, B)
    b.eval(U_t, S_t, Vt_t)

    # Numerical rank of target
    if int(S_t.shape[0]) > 0:
        max_s_target = float(b.to_scalar(S_t[0]))
        thresh_target = max_s_target * precision_thresh
        target_rank_mask = S_t > thresh_target
        target_rank_arr = b.sum(b.astype(target_rank_mask, "int32"))
        b.eval(target_rank_arr)
        target_rank = int(b.to_scalar(target_rank_arr))
    else:
        target_rank = 0

    # Alignment rank: use source_rank for the pseudoinverse computation
    # The target rank is informative but doesn't constrain the solution
    # We truncate to source_rank to remove numerical noise, not to match target
    alignment_rank = source_rank

    logger.info(
        "NUMERICAL RANK: source_rank=%d/%d, target_rank=%d/%d, alignment_rank=%d",
        source_rank, d_source, target_rank, d_target, alignment_rank,
    )

    # If source has no numerically significant structure, alignment is impossible.
    # Return zero F with diagnostics. Don't silently produce garbage.
    if alignment_rank == 0:
        logger.warning(
            "SOURCE HAS NO SIGNAL: source_rank=0, all singular values below noise floor. "
            "Returning zero alignment matrix."
        )
        F = b.zeros((d_source, d_target), dtype=A.dtype)
        return F, source_rank, target_rank, 0, float("inf"), 1.0

    # Truncate source to top-k singular components
    # A_k = U_k @ diag(S_k) @ Vt_k where k = source_rank (numerical rank of source)
    k = alignment_rank
    U_k = U_s[:, :k]  # [n, k]
    S_k = S_s[:k]  # [k]
    Vt_k = Vt_s[:k, :]  # [k, d_source]
    b.eval(U_k, S_k, Vt_k)

    # Compute condition number in truncated space
    if k > 0 and int(S_k.shape[0]) > 0:
        max_s_k = float(b.to_scalar(S_k[0]))
        min_s_k = float(b.to_scalar(S_k[k - 1]))
        if min_s_k > 0:
            condition_number = max_s_k / min_s_k
        else:
            condition_number = float("inf")
    else:
        condition_number = float("inf")

    logger.info(
        "TRUNCATED CONDITION: kappa=%.2e (kappa_limit=%.2e, precision_floor=%.2e)",
        condition_number,
        (1.0 / precision_thresh) if precision_thresh > 0 else float("inf"),
        precision_thresh,
    )

    # Solve in truncated space:
    # We want F such that A @ F ~= B
    # In truncated space: A_k = U_k @ diag(S_k) @ Vt_k
    # pinv(A_k) = V_k @ diag(1/S_k) @ U_k^T
    # F = pinv(A_k) @ B

    # Compute S_k_inv with safe division
    div_eps = division_epsilon(b, S_k)
    S_k_safe = b.maximum(S_k, b.full(S_k.shape, div_eps))
    S_k_inv = 1.0 / S_k_safe
    b.eval(S_k_inv)

    # pinv(A_k) @ B = V_k @ diag(1/S_k) @ U_k^T @ B
    # Step 1: U_k^T @ B  -> [k, d_target]
    UtB = b.matmul(b.transpose(U_k), B)
    b.eval(UtB)

    # Step 2: diag(1/S_k) @ (U_k^T @ B)  -> [k, d_target]
    # Reshape S_k_inv to [k, 1] for broadcasting
    S_k_inv_col = b.reshape(S_k_inv, (k, 1))
    scaled = S_k_inv_col * UtB
    b.eval(scaled)

    # Step 3: V_k @ (diag(1/S_k) @ U_k^T @ B)  -> [d_source, d_target]
    # V_k = Vt_k^T  -> [d_source, k]
    V_k = b.transpose(Vt_k)
    F = b.matmul(V_k, scaled)
    b.eval(F)

    # =========================================================================
    # ALIGNMENT RESIDUAL: ||source @ F - target||_F / ||target||_F
    # =========================================================================
    # This is the closed-form measure of whether the alignment succeeded.
    # If residual < sqrt(eps), the linear transform is within precision.
    # If residual >> sqrt(eps), source and target are NOT linearly related.
    reconstructed = b.matmul(A, F)
    diff = reconstructed - B
    diff_norm = b.sqrt(b.sum(diff * diff))
    target_norm = b.sqrt(b.sum(B * B))
    b.eval(diff_norm, target_norm)

    div_eps = division_epsilon(b, target_norm)
    target_norm_safe = max(float(b.to_scalar(target_norm)), div_eps)
    residual = float(b.to_scalar(diff_norm)) / target_norm_safe

    logger.info(
        "ALIGNMENT RESIDUAL: ||A@F - B||/||B|| = %.6f (precision floor = %.2e)",
        residual,
        precision_thresh,
    )

    return F, source_rank, target_rank, alignment_rank, condition_number, residual


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
        # Linear regularization schedule: standard Tikhonov first-order approximation.
        # Regularization scales from 0 (well-conditioned) to max_reg (ill-conditioned).
        # Ref: Hansen (1998) "Rank-Deficient and Discrete Ill-Posed Problems", Ch. 5.
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


def gpu_lstsq(
    backend: "Backend",
    A: "Array",
    B: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Solve least squares via normal equations.

    Uses (A^T A + lambda*I)^{-1} A^T B when n >= d and the dual form when n < d.
    Uses dtype-derived regularization to stabilize the solve.
    If stats is provided, populates iterations, residual_norm, rhs_norm, method.
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

    # =========================================================================
    # DIRECT SOLVE via Normal Equations (when n >= d)
    # =========================================================================
    # The pseudoinverse for tall matrices has a closed form:
    #   pinv(A) = (A^T @ A)^{-1} @ A^T
    #
    # So: X = pinv(A) @ B = (A^T @ A)^{-1} @ A^T @ B
    #
    # Let G = A^T @ A (d x d) and H = A^T @ B (d x k)
    # Then X = solve(G, H) using Cholesky (G is positive semidefinite)
    #
    # This is O(d^3) instead of O(max_iter * n * d), instant on GPU.
    # =========================================================================

    if n >= d:
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ: Direct solve [%d x %d] -> [%d x %d] (n >= d, using closed-form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A^T @ A (d x d) - this is the Gram matrix in feature space
        A_T = b.transpose(A)
        G = b.matmul(A_T, A)  # d x d
        b.eval(G)

        # Add Tikhonov regularization for numerical stability
        # lambda = eps * max(diag(G)) ensures well-conditioning
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))
        reg_lambda = eps * max(max_diag_val, 1.0)

        G_reg = G + reg_lambda * b.eye(d)
        b.eval(G_reg)

        # Compute A^T @ B (d x k)
        H = b.matmul(A_T, B)  # d x k
        b.eval(H)

        # Solve G @ X = H directly
        # The backend's solve() uses the most appropriate method internally
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

    # =========================================================================
    # DIRECT SOLVE for Underdetermined Systems (when n < d)
    # =========================================================================
    # For underdetermined systems, use the dual normal equations:
    #   pinv(A) = A^T @ (A @ A^T)^{-1}
    #
    # So: X = pinv(A) @ B = A^T @ (A @ A^T)^{-1} @ B
    #
    # Let G = A @ A^T (n x n) - this is much smaller than d x d!
    # Then solve G @ Y = B, and X = A^T @ Y
    #
    # This is O(n^3) instead of O(d^3) or iterating.
    # =========================================================================

    elif n < d:  # Underdetermined case
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ_DUAL: Direct solve [%d x %d] -> [%d x %d] (n < d, using dual form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A @ A^T (n x n) - the Gram matrix in sample space
        A_T = b.transpose(A)
        G = b.matmul(A, A_T)  # n x n (much smaller than d x d!)
        b.eval(G)

        # Add minimal Tikhonov regularization for numerical conditioning only.
        #
        # IMPORTANT: Underdetermined (n < d) does NOT mean rank-deficient.
        # If A has full row rank n, then A @ A^T is nxn with full rank and
        # the pseudoinverse X = A^T @ (A @ A^T)^{-1} @ B gives ZERO residual.
        #
        # The regularization (A @ A^T + lambda*I) introduces residual proportional to lambda:
        #   A @ X = B - lambda*(A @ A^T + lambda*I)^{-1} @ B
        #
        # So lambda must be minimal - just enough to handle numerical conditioning,
        # not scaled by underdetermination ratio. That scaling introduces
        # artificial residual where none should exist.
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))

        # Minimal regularization: just eps * scale for numerical conditioning
        reg_lambda = eps * max(max_diag_val, 1.0)

        G_reg = G + reg_lambda * b.eye(n)
        b.eval(G_reg)

        # Solve G @ Y = B for Y (n x k)
        Y = b.solve(G_reg, B)
        b.eval(Y)

        # X = A^T @ Y (d x k)
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

        # Log residual for diagnostics
        # With minimal regularization, residual should be near-zero for full-rank A
        relative_residual = res_norm_val / max(B_norm_val, eps)
        logger.info(
            "NORMAL_EQ_DUAL: Solved in %.3fs, residual=%.2e (%.2f%% of ||B||=%.2e)",
            elapsed, res_norm_val, 100.0 * relative_residual, B_norm_val
        )

        if stats is not None:
            stats["iterations"] = 0.0
            stats["residual_norm"] = res_norm_val
            stats["rhs_norm"] = B_norm_val
            stats["method"] = "normal_equations_dual"

        if squeeze_output:
            X = b.reshape(X, (-1,))

        return X


__all__ = [
    # Matrix decomposition
    "power_iteration_eigh",
    "geodesic_svd",
    "orthogonalize_alignment",
    "orthogonalize_alignment_full",
    "svd_auto_rank",
    "geodesic_pinv",
    "null_space_projector",
    "numerical_rank_truncated_lstsq",
    "safe_inverse",
    # GPU-accelerated linear algebra
    "gpu_lstsq",
]
