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

"""Subspace overlap analysis for manifold splitting.

Computes overlap between source and target activation subspaces using
principal angle analysis.

References:
    - Björck & Golub (1973) "Numerical Methods for Computing Angles Between Linear Subspaces"
    - Golub & Van Loan (1996) "Matrix Computations" - Chapter on Principal Angles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    machine_epsilon,
    precision_dtype,
    sqrt_scalar,
    svd_rank_threshold,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class SubspaceAnalysisResult:
    """Result of subspace overlap analysis between source and target.

    Attributes:
        shared_rank: Effective shared dimensionality (directions both use)
        source_rank: Intrinsic rank of source activations
        target_rank: Intrinsic rank of target activations
        principal_angles: Angles between principal directions [min(src_rank, tgt_rank)]
        cos_principal_angles: Cosines of principal angles (singular values)
        shared_basis: Basis vectors of shared subspace [shared_rank, d]
        novel_basis: Basis vectors of novel subspace (source-only) [novel_rank, d]
        overlap_fraction: Fraction of source subspace overlapping target
        angle_threshold: Data-derived threshold for shared vs novel classification
    """

    shared_rank: int
    source_rank: int
    target_rank: int
    principal_angles: "Array"
    cos_principal_angles: "Array"
    shared_basis: "Array"
    novel_basis: "Array"
    overlap_fraction: float
    angle_threshold: float


def _compute_intrinsic_rank(
    activations: "Array",
    backend: "Backend",
) -> int:
    """Compute intrinsic rank using participation ratio (Rényi effective rank).

    The participation ratio PR = (∑λᵢ)² / ∑λᵢ² gives the "effective number"
    of significant eigenvalues, which is the intrinsic rank.

    Args:
        activations: Activation matrix [n_samples, d]
        backend: Backend for tensor operations

    Returns:
        Intrinsic rank as integer
    """
    b = backend
    shape = b.shape(activations)
    n = int(shape[0])
    d = int(shape[1])

    if n == 0 or d == 0:
        return 0

    # Promote to high precision for numerical stability
    arr = b.astype(activations, precision_dtype(b, reference=activations))

    # Compute covariance (Gram) matrix in smaller dimension
    if n >= d:
        # Sample covariance: X^T X / n [d, d]
        gram = b.matmul(b.transpose(arr), arr)
    else:
        # Gram matrix in sample space: X X^T [n, n]
        gram = b.matmul(arr, b.transpose(arr))
    b.eval(gram)

    # Eigendecomposition (eigvalsh returns sorted ascending)
    eigenvalues = b.eigvalsh(gram)
    eigenvalues = b.maximum(eigenvalues, b.zeros_like(eigenvalues))
    b.eval(eigenvalues)

    # Participation ratio: (∑λ)² / ∑λ²
    sum_vals = b.sum(eigenvalues)
    sum_sq = b.sum(eigenvalues * eigenvalues)
    b.eval(sum_vals, sum_sq)

    sum_val = float(b.to_scalar(sum_vals))
    sum_sq_val = float(b.to_scalar(sum_sq))

    if sum_sq_val <= 0.0 or sum_val <= 0.0:
        return 0

    eff_rank = (sum_val * sum_val) / sum_sq_val

    # Clamp to valid range [1, min(n, d)]
    max_rank = min(n, d)
    return max(1, min(int(round(eff_rank)), max_rank))


def _compute_numeric_rank(
    singular_values: "Array",
    max_dim: int,
    backend: "Backend",
) -> int:
    """Compute numeric rank from singular values using data-derived threshold.

    Args:
        singular_values: Singular values in descending order [k]
        max_dim: Maximum possible dimension
        backend: Backend for tensor operations

    Returns:
        Numeric rank (count of significant singular values)
    """
    b = backend
    shape = b.shape(singular_values)
    k = int(shape[0])

    if k == 0:
        return 0

    # Threshold based on machine epsilon and largest singular value
    max_sv = b.max(singular_values)
    b.eval(max_sv)
    max_sv_val = float(b.to_scalar(max_sv))

    if max_sv_val <= 0.0:
        return 0

    eps = machine_epsilon(b, singular_values)
    threshold = max_sv_val * max(eps * max_dim, eps)

    # Count singular values above threshold
    threshold_arr = b.array([threshold])
    mask = singular_values > threshold_arr
    b.eval(mask)

    count_dtype = precision_dtype(b, reference=singular_values)
    rank = int(b.to_scalar(b.sum(b.astype(mask, count_dtype))))

    return max(1, rank)


def compute_subspace_overlap(
    aligned_source: "Array",
    target: "Array",
    backend: "Backend | None" = None,
) -> SubspaceAnalysisResult:
    """Compute overlap between source and target activation subspaces.

    Uses SVD to find principal subspaces and computes principal angles
    to determine which directions are genuinely shared vs novel.

    Args:
        aligned_source: Source activations after Procrustes alignment [n, d]
        target: Target activations [n, d]
        backend: Backend for tensor operations

    Returns:
        SubspaceAnalysisResult with shared/novel basis vectors and metrics
    """
    b = backend or get_default_backend()

    shape_src = b.shape(aligned_source)
    shape_tgt = b.shape(target)
    n = int(shape_src[0])
    d = int(shape_src[1])

    if int(shape_tgt[0]) != n or int(shape_tgt[1]) != d:
        raise ValueError(
            f"Shape mismatch: aligned_source {shape_src} vs target {shape_tgt}"
        )

    # Promote to high precision
    src_arr = b.astype(aligned_source, precision_dtype(b, reference=aligned_source))
    tgt_arr = b.astype(target, precision_dtype(b, reference=target))

    # Step 1: Compute intrinsic ranks
    src_rank = _compute_intrinsic_rank(src_arr, b)
    tgt_rank = _compute_intrinsic_rank(tgt_arr, b)
    shared_k = min(src_rank, tgt_rank)

    logger.debug(
        "Subspace overlap: src_rank=%d, tgt_rank=%d, shared_k=%d",
        src_rank,
        tgt_rank,
        shared_k,
    )

    if shared_k == 0:
        # Degenerate case: no meaningful subspace
        empty = b.zeros((0, d))
        return SubspaceAnalysisResult(
            shared_rank=0,
            source_rank=src_rank,
            target_rank=tgt_rank,
            principal_angles=b.array([]),
            cos_principal_angles=b.array([]),
            shared_basis=empty,
            novel_basis=empty,
            overlap_fraction=0.0,
            angle_threshold=0.0,
        )

    # Step 2: SVD of activation matrices
    # We need the right singular vectors V^T which span the feature space
    # SVD: X = U @ S @ V^T where V^T has shape [d, d] or [min(n,d), d]
    U_src, S_src, Vt_src = b.svd(src_arr, compute_uv=True)
    U_tgt, S_tgt, Vt_tgt = b.svd(tgt_arr, compute_uv=True)
    b.eval(U_src, S_src, Vt_src, U_tgt, S_tgt, Vt_tgt)

    # Get numeric ranks from singular values
    src_numeric_rank = _compute_numeric_rank(S_src, d, b)
    tgt_numeric_rank = _compute_numeric_rank(S_tgt, d, b)

    # Use minimum of intrinsic and numeric rank for robustness
    src_effective_rank = min(src_rank, src_numeric_rank)
    tgt_effective_rank = min(tgt_rank, tgt_numeric_rank)
    effective_k = min(src_effective_rank, tgt_effective_rank)

    if effective_k == 0:
        effective_k = 1  # Ensure at least one direction

    logger.debug(
        "Effective ranks: src=%d, tgt=%d, k=%d",
        src_effective_rank,
        tgt_effective_rank,
        effective_k,
    )

    # Step 3: Extract top-k right singular vectors (principal directions in feature space)
    # Vt has shape [min(n, d), d], rows are right singular vectors
    vt_src_shape = b.shape(Vt_src)
    vt_tgt_shape = b.shape(Vt_tgt)
    k_src = min(effective_k, int(vt_src_shape[0]))
    k_tgt = min(effective_k, int(vt_tgt_shape[0]))
    k_actual = min(k_src, k_tgt)

    # Get top-k rows of Vt (principal directions)
    idx = b.arange(0, k_actual)
    V_src_k = b.take(Vt_src, idx, axis=0)  # [k, d]
    V_tgt_k = b.take(Vt_tgt, idx, axis=0)  # [k, d]
    b.eval(V_src_k, V_tgt_k)

    # Step 4: Compute principal angles via SVD of V_src_k @ V_tgt_k^T
    # The singular values of this product are the cosines of principal angles
    M = b.matmul(V_src_k, b.transpose(V_tgt_k))  # [k, k]
    b.eval(M)

    # SVD of M gives cosines of principal angles
    cos_angles = b.svd(M, compute_uv=False)  # Just singular values
    b.eval(cos_angles)

    # Clamp cosines to [0, 1] (numerical precision can give slightly > 1)
    cos_angles = b.clip(cos_angles, 0.0, 1.0)
    b.eval(cos_angles)

    # Principal angles in radians
    eps = division_epsilon(b, cos_angles)
    # arccos is stable for cos in [0, 1]
    principal_angles = b.arccos(b.clip(cos_angles, eps, 1.0 - eps))
    b.eval(principal_angles)

    # =========================================================================
    # GEOMETRY-DERIVED CLASSIFICATION (no arbitrary thresholds)
    # =========================================================================
    # Principal angles between subspaces are geometrically meaningful:
    #   - cos_angle = 1.0: directions are parallel (shared)
    #   - cos_angle = 0.0: directions are orthogonal (novel)
    #
    # Instead of using gap detection with arbitrary fallback, we use machine
    # epsilon as the threshold. A direction is "shared" if its cosine is
    # significantly above numerical noise - i.e., there IS alignment.
    #
    # This is geometry-driven: machine_eps defines "effectively orthogonal"
    # in floating point. Anything above that threshold has real alignment.
    # =========================================================================
    eps = machine_epsilon(b, cos_angles)
    sqrt_eps = sqrt_scalar(eps, b)  # More conservative: sqrt(eps) for cosine threshold

    # Shared = cosine significantly above numerical noise
    # Novel = cosine at or below numerical noise (effectively orthogonal)
    cos_threshold = sqrt_eps
    cos_threshold_arr = b.array([cos_threshold])
    shared_mask = cos_angles > cos_threshold_arr
    novel_mask = cos_angles <= cos_threshold_arr
    b.eval(shared_mask, novel_mask)

    # Count shared and novel directions
    count_dtype = precision_dtype(b, reference=cos_angles)
    shared_count = int(b.to_scalar(b.sum(b.astype(shared_mask, count_dtype))))
    novel_count = int(b.to_scalar(b.sum(b.astype(novel_mask, count_dtype))))

    logger.debug(
        "Direction classification: shared=%d, novel=%d, threshold=%.4f",
        shared_count,
        novel_count,
        cos_threshold,
    )

    # Step 6: Build shared and novel basis vectors
    # For shared: directions in source that align well with target
    # For novel: directions in source that are orthogonal to target's principal subspace

    # Get full Vt from source for extracting basis
    vt_src_full_rows = int(b.shape(Vt_src)[0])

    if shared_count > 0:
        # Extract indices where shared_mask is True
        shared_indices = b.nonzero(shared_mask)
        if len(shared_indices) > 0 and shared_indices[0].shape[0] > 0:
            shared_idx = shared_indices[0]
            # Map back to source singular vectors
            shared_basis = b.take(V_src_k, shared_idx, axis=0)
        else:
            shared_basis = b.zeros((0, d))
    else:
        shared_basis = b.zeros((0, d))
    b.eval(shared_basis)

    if novel_count > 0:
        # Extract indices where novel_mask is True
        novel_indices = b.nonzero(novel_mask)
        if len(novel_indices) > 0 and novel_indices[0].shape[0] > 0:
            novel_idx = novel_indices[0]
            novel_basis = b.take(V_src_k, novel_idx, axis=0)
        else:
            novel_basis = b.zeros((0, d))
    else:
        novel_basis = b.zeros((0, d))
    b.eval(novel_basis)

    # Also include any source directions beyond the shared rank
    # These are directions target doesn't have at all (target's low-rank space)
    if src_effective_rank > k_actual:
        extra_novel_idx = b.arange(k_actual, min(src_effective_rank, vt_src_full_rows))
        extra_novel_basis = b.take(Vt_src, extra_novel_idx, axis=0)
        b.eval(extra_novel_basis)
        # Concatenate with novel basis
        if int(b.shape(novel_basis)[0]) > 0:
            novel_basis = b.concatenate([novel_basis, extra_novel_basis], axis=0)
        else:
            novel_basis = extra_novel_basis
        b.eval(novel_basis)

    # Compute overlap fraction: fraction of source directions that are shared
    overlap_fraction = shared_count / max(k_actual, 1)

    # Convert threshold to angle for logging
    angle_threshold = float(b.to_scalar(b.arccos(b.array([cos_threshold]))))

    logger.info(
        "Subspace analysis: shared_rank=%d/%d, overlap=%.2f%%, angle_thresh=%.1f deg",
        shared_count,
        k_actual,
        overlap_fraction * 100,
        angle_threshold * 180 / 3.14159,
    )

    return SubspaceAnalysisResult(
        shared_rank=shared_count,
        source_rank=src_effective_rank,
        target_rank=tgt_effective_rank,
        principal_angles=principal_angles,
        cos_principal_angles=cos_angles,
        shared_basis=shared_basis,
        novel_basis=novel_basis,
        overlap_fraction=overlap_fraction,
        angle_threshold=angle_threshold,
    )


def project_to_subspace(
    activations: "Array",
    basis: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Project activations onto a subspace defined by basis vectors.

    Args:
        activations: Activation matrix [n, d]
        basis: Basis vectors [k, d] where k is the subspace dimension
        backend: Backend for tensor operations

    Returns:
        Projected activations [n, k]
    """
    b = backend or get_default_backend()

    basis_shape = b.shape(basis)
    if int(basis_shape[0]) == 0:
        # Empty basis - return zeros
        n = int(b.shape(activations)[0])
        return b.zeros((n, 0))

    # Project: X_proj = X @ basis^T
    # This gives coordinates in the subspace basis
    projected = b.matmul(activations, b.transpose(basis))
    b.eval(projected)

    return projected


def compute_subspace_projector(
    basis: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Compute the orthogonal projector onto a subspace.

    The projector P = V V^T where V are the orthonormal basis vectors.
    Applying P to a vector projects it onto the subspace.

    Args:
        basis: Orthonormal basis vectors [k, d]
        backend: Backend for tensor operations

    Returns:
        Projector matrix [d, d]
    """
    b = backend or get_default_backend()

    basis_shape = b.shape(basis)
    k = int(basis_shape[0])
    d = int(basis_shape[1])

    if k == 0:
        return b.zeros((d, d))

    # P = V^T @ V (since basis is [k, d], V^T is [d, k], result is [d, d])
    projector = b.matmul(b.transpose(basis), basis)
    b.eval(projector)

    return projector


__all__ = [
    "SubspaceAnalysisResult",
    "compute_subspace_overlap",
    "project_to_subspace",
    "compute_subspace_projector",
]
