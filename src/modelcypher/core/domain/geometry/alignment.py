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

"""Invariant alignment: Linear CKA = 1.0 by construction.

Provides alignment transforms for representation matching between neural network layers.

References:
    - Penrose, R. (1955). "A generalized inverse for matrices."
      Proceedings of the Cambridge Philosophical Society.
    - Yu et al. 2025 "Relative Geodesic Representations" (NeurIPS).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .decomposition import gpu_lstsq
from .precision import _promote_precision, division_epsilon, machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Compute the linear alignment transform F = pinv(source) @ target.

    Uses the Moore-Penrose pseudoinverse to solve the least-squares alignment and
    optionally truncates to the effective rank from the source Gram spectrum.

    References:
        - Penrose, R. (1955). "A generalized inverse for matrices."
          Proceedings of the Cambridge Philosophical Society.
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

    # THE FORMULA: F = pinv(source) @ target (solved via closed-form normal equations)
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

    Computes geodesic cosine matrices, aligns them via Procrustes, and
    recovers a feature-space transform.

    Reference: Yu et al. 2025 "Relative Geodesic Representations" (NeurIPS).
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
    # Find R such that G_source @ R ~= G_target
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
        denom_eps = division_epsilon(b, target_norm)
        rel_error = float(b.to_scalar(frob_norm)) / max(float(b.to_scalar(target_norm)), denom_eps)
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

    # =========================================================================
    # PRINCIPLED RANK DETERMINATION VIA GEODESIC INTRINSIC DIMENSION
    # =========================================================================
    # OLD HEURISTIC: threshold = eps * max_s * n_samples (arbitrary)
    #
    # NEW PRINCIPLED: Use intrinsic dimension of target manifold.
    # The effective rank of G_target (similarity matrix) equals the intrinsic
    # dimension of the underlying manifold - no magic numbers needed.
    #
    # Reference: Facco et al. (2017) TwoNN with geodesic distances
    # =========================================================================
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    id_estimator = IntrinsicDimension(b)
    id_result = id_estimator.compute(target)
    intrinsic_dim = id_result.intrinsic_dimension

    # Effective rank = intrinsic dimension (clamped to valid range)
    effective_rank = max(1, min(int(round(intrinsic_dim)), n_samples))

    logger.info(
        "Geodesic alignment: intrinsic_dim=%.2f, effective_rank=%d/%d",
        intrinsic_dim, effective_rank, n_samples
    )

    # Keep top effective_rank singular values, zero out the rest
    # Create mask: True for indices < effective_rank
    indices = b.arange(len(S_t))
    rank_mask = indices < effective_rank
    rank_mask_float = b.astype(rank_mask, S_t.dtype)

    # Safe inversion: replace zeros with ones before division, then mask out
    S_t_safe = b.where(S_t > machine_epsilon(b, S_t), S_t, b.ones_like(S_t))
    S_t_inv = rank_mask_float / S_t_safe
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


__all__ = [
    "invariant_alignment",
    "geodesic_invariant_alignment",
]
