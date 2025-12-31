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

from __future__ import annotations

import logging

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

logger = logging.getLogger(__name__)


def _matrix_rank_for_alignment(
    matrix: "object",
    backend: "object",
    eps: float | None = None,
) -> int:
    """Compute effective rank using dtype-derived threshold."""
    n = int(matrix.shape[0])
    if n == 0:
        return 0

    gram = backend.matmul(matrix, backend.transpose(matrix))
    # Cast to float32 for eigendecomposition (MLX doesn't support bfloat16 for eigh)
    gram_dtype = str(backend.dtype(gram))
    if "bfloat16" in gram_dtype:
        gram_f32 = backend.astype(gram, "float32")
        backend.eval(gram_f32)
        eigvals, _ = backend.eigh(gram_f32)
    else:
        eigvals, _ = backend.eigh(gram)
    backend.eval(eigvals)
    values = list(backend.to_numpy(eigvals).tolist())
    if not values:
        return 0
    max_val = max(values)
    if eps is None:
        eps = machine_epsilon(backend, matrix)
    threshold = max_val * eps
    return sum(1 for val in values if val > threshold)


def _dynamic_condition_threshold(
    matrix: "object",
    backend: "object",
) -> float:
    """Compute condition number threshold from dtype.

    The condition number threshold is 1/sqrt(eps), which represents
    the boundary where numerical operations become unreliable.
    No arbitrary cap - the dtype determines the threshold.
    """
    eps = machine_epsilon(backend, matrix)
    # 1/sqrt(eps) is the natural condition number threshold
    # Beyond this, matrix operations lose significant precision
    return 1.0 / (eps ** 0.5)


def _solve_feature_transform_exact(
    source_matrix: "object",
    target_matrix: "object",
    backend: "object",
    regularization: float = 0.0,
) -> "object | None":
    """Solve source @ F = target for exact alignment.

    Strategy (in order of preference):
    1. QR-based solve for full-rank, well-conditioned cases (fastest)
    2. Rank-truncated spectral inverse (exact on the support space)
    3. Gram alignment in sample space (relational geometry)
    4. Eigendecomposition in sample space (legacy, squares condition number)

    The caller verifies exact kernel alignment (CKA = 1.0). This function only
    proposes candidate transforms; it does not accept approximate alignment.
    """
    from modelcypher.core.domain.geometry.numerical_stability import (
        solve_full_row_rank_via_qr,
        solve_via_gram_alignment,
        solve_via_truncated_svd,
    )

    n = int(source_matrix.shape[0])
    if n == 0:
        return None

    eps = max(machine_epsilon(backend, source_matrix), 1e-12)

    # Try QR-based solve first (fast for full-rank, well-conditioned cases)
    F_qr, diag = solve_full_row_rank_via_qr(backend, source_matrix, target_matrix)

    best_transform: "object | None" = None
    best_residual = float("inf")
    best_label = "none"

    if F_qr is not None:
        logger.debug(
            "QR solve: method=%s, rank=%d/%d, cond=%.2e, residual=%.2e",
            diag.get("method", "unknown"),
            diag.get("rank", 0),
            n,
            diag.get("condition", float("inf")),
            diag.get("residual_norm", float("inf")),
        )
        # Accept if residual is small (system is full-rank and well-conditioned)
        qr_residual = float(diag.get("residual_norm", float("inf")))
        if qr_residual < best_residual:
            best_transform = F_qr
            best_residual = qr_residual
            best_label = "qr"
        if qr_residual < eps * 1000:
            return F_qr

    # QR residual too large - try rank-truncated spectral inverse (handles rank-deficiency)
    logger.debug(
        "QR residual too large (%.2e), trying spectral inverse",
        diag.get("residual_norm", float("inf")) if F_qr is not None else float("inf"),
    )

    F_svd, diag_svd = solve_via_truncated_svd(backend, source_matrix, target_matrix)

    if F_svd is not None:
        logger.debug(
            "Spectral inverse: rank=%d/%d, cond=%.2e, proj_err=%.2e, residual=%.2e",
            diag_svd.get("rank", 0),
            n,
            diag_svd.get("condition", float("inf")),
            diag_svd.get("projection_error", float("inf")),
            diag_svd.get("residual_norm", float("inf")),
        )
        svd_residual = float(diag_svd.get("residual_norm", float("inf")))
        if svd_residual < best_residual:
            best_transform = F_svd
            best_residual = svd_residual
            best_label = "spectral_inverse"
        # Exact alignment only if residual is at precision scale.
        if svd_residual < eps * 100:
            return F_svd

    # Spectral inverse insufficient - try Gram alignment in sample space.
    logger.debug(
        "Spectral inverse residual too large (%.2e), trying Gram alignment",
        diag_svd.get("residual_norm", float("inf")) if F_svd is not None else float("inf"),
    )

    # Gram alignment: align sample structure (relational geometry).
    F_gram, diag_gram = solve_via_gram_alignment(backend, source_matrix, target_matrix)

    if F_gram is not None:
        logger.debug(
            "Gram alignment: rank_s=%d, rank_t=%d, procrustes_err=%.2e",
            diag_gram.get("rank_source", 0),
            diag_gram.get("rank_target", 0),
            diag_gram.get("procrustes_error", float("inf")),
        )
        aligned_gram = backend.matmul(source_matrix, F_gram)
        backend.eval(aligned_gram)
        residual = aligned_gram - target_matrix
        res_norm = backend.norm(residual)
        tgt_norm = backend.norm(target_matrix)
        backend.eval(res_norm, tgt_norm)
        gram_residual = float(backend.to_numpy(res_norm)) / (
            float(backend.to_numpy(tgt_norm)) + eps
        )
        if gram_residual < best_residual:
            best_transform = F_gram
            best_residual = gram_residual
            best_label = "gram_alignment"
        if gram_residual < eps * 100:
            return F_gram

    # Fall back to eigendecomposition (legacy, squares condition number)
    if best_transform is not None:
        logger.debug(
            "Returning best candidate transform (%s) with residual %.2e",
            best_label,
            best_residual,
        )
        return best_transform

    logger.debug("All direct methods failed, trying eigen solve")

    gram = backend.matmul(source_matrix, backend.transpose(source_matrix))
    if regularization > 0.0:
        gram = gram + regularization * backend.eye(n)
    backend.eval(gram)

    # Cast to float32 for eigendecomposition (MLX doesn't support bfloat16 for eigh)
    gram_dtype = str(backend.dtype(gram))
    if "bfloat16" in gram_dtype:
        gram_f32 = backend.astype(gram, "float32")
        backend.eval(gram_f32)
        eigvals, eigvecs = backend.eigh(gram_f32)
    else:
        eigvals, eigvecs = backend.eigh(gram)
    backend.eval(eigvals, eigvecs)
    values = [float(v) for v in backend.to_numpy(eigvals).tolist()]
    if not values:
        return None
    min_eig = min(values)
    max_eig = max(values)
    if max_eig <= eps:
        return None

    eigvals = backend.maximum(eigvals, backend.zeros_like(eigvals))
    inv_vals = backend.where(
        eigvals > 0.0,
        1.0 / eigvals,
        backend.zeros_like(eigvals),
    )
    backend.eval(inv_vals)
    gram_inv = backend.matmul(
        eigvecs * backend.reshape(inv_vals, (1, -1)),
        backend.transpose(eigvecs),
    )
    backend.eval(gram_inv)

    transform = backend.matmul(
        backend.transpose(source_matrix),
        backend.matmul(gram_inv, target_matrix),
    )
    backend.eval(transform)

    logger.debug(
        "Eigen solve: min_eig=%.2e, max_eig=%.2e, cond=%.2e",
        min_eig,
        max_eig,
        max_eig / (min_eig + eps) if min_eig > 0 else float("inf"),
    )

    return transform
