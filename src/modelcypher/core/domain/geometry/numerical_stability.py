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
    eigvals_r, V = b.eigh(cov_r)
    b.eval(cov_r, eigvals_r, V)

    order_r = b.argsort(-eigvals_r)
    eigvals_r = b.take(eigvals_r, order_r, axis=0)
    V = b.take(V, order_r, axis=1)
    b.eval(eigvals_r, V)

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

    V_pos = V[:, :rank]
    s_pos = s[:rank]
    inv_s = 1.0 / s_pos
    U_pos = b.matmul(A, V_pos) * b.reshape(inv_s, (1, -1))
    b.eval(U_pos)

    if rank < k:
        cov_l = b.matmul(A, b.transpose(A))
        eigvals_l, U_full = b.eigh(cov_l)
        b.eval(cov_l, eigvals_l, U_full)

        order_l = b.argsort(-eigvals_l)
        U_full = b.take(U_full, order_l, axis=1)
        U_null = U_full[:, rank:k]
        U = b.concatenate([U_pos, U_null], axis=1)
    else:
        U = U_pos

    V_full = V[:, :k]
    Vt = b.transpose(V_full)

    S = s[:k]
    if threshold > 0.0:
        thresh_arr = b.full(S.shape, float(threshold))
        S = b.where(S > thresh_arr, S, b.zeros_like(S))
        b.eval(S)

    if full_matrices:
        return U, S, Vt

    return U[:, :k], S[:k], Vt[:k, :]
