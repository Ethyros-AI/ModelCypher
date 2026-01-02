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

"""
Unified Cross-Dimensional Projection.

ONE function to handle ALL dimension mismatches in model merging.

The geometry is invariant:
- Gram matrix K = X @ X^T is [n×n] regardless of feature dimension d
- CKA compares Gram matrices - works across ANY dimensions
- GW transport finds soft correspondence between different-sized spaces
- Procrustes finds rotation when dimensions match
- Fréchet mean merges magnitudes

This module is THE unified API for all cross-dimensional projection.
The transplant pipeline (stage_3_transplant.py) uses this for cross-arch merges.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_shared_relational_rank,
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
    svd_rank_threshold,
    svd_via_eigh,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class ProjectionMethod(str, Enum):
    """Methods for cross-dimensional projection.

    GRAM_TRANSPORT: Uses Gromov-Wasserstein on Gram matrices.
        - Works for ANY dimension mismatch
        - Preserves relational structure (distances between rows)
        - Best for semantic/conceptual alignment

    PROCRUSTES: Uses orthogonal rotation.
        - Only works when ONE dimension matches
        - Finds R minimizing ||source @ R - target||_F
        - Best for same-architecture alignment

    SVD_PROJECT: Uses SVD to project to shared subspace.
        - Works for any dimension mismatch
        - Preserves top singular values (variance)
        - Best when structure differs significantly
    """
    GRAM_TRANSPORT = "gram_transport"
    PROCRUSTES = "procrustes"
    SVD_PROJECT = "svd_project"


@dataclass(frozen=True)
class ProjectionResult:
    """Result of cross-dimensional projection."""
    projected: "Array"  # [m_t, d_t] - matches target shape
    alignment_score: float  # 0-1, higher = better structural preservation
    method_used: ProjectionMethod
    row_coupling: "Array | None"  # GW coupling for rows (if used)
    col_coupling: "Array | None"  # GW coupling for cols (if used)


def project_cross_dimensional(
    source: "Array",
    target: "Array",
    method: ProjectionMethod | str = ProjectionMethod.GRAM_TRANSPORT,
    backend: "Backend | None" = None,
) -> ProjectionResult:
    """
    Project source weights to target shape using geometry-preserving methods.

    THE UNIFIED API for all dimension mismatches.

    PRECISION MATTERS. Being off by 1e-10 in alignment compounds through layers
    and causes hallucinations at inference. No shortcuts. No approximations.

    The key insight: Gram matrices K = X @ X^T capture relational geometry
    independent of feature dimension. For weight matrix [m, d]:
    - Row Gram: W @ W^T is [m, m] - can be huge for embeddings
    - Col Gram: W^T @ W is [d, d] - always tractable (hidden_dim sized)

    GW on column-space Grams is O(d_s² + d_t²) - independent of vocab size.
    The coupling π[d_s, d_t] is then applied EXACTLY: W @ π

    Args:
        source: Source weight matrix [m_s, d_s]
        target: Target weight matrix [m_t, d_t]
        method: Projection method (gram_transport, procrustes, svd_project)
        backend: Backend for GPU-accelerated operations

    Returns:
        ProjectionResult with projected weights [m_t, d_t] and alignment score
    """
    b = backend or get_default_backend()

    # Convert method string to enum if needed
    if isinstance(method, str):
        method = ProjectionMethod(method)

    # Ensure float32 for numerical stability in intermediate computations
    source_f32 = b.astype(b.array(source), "float32")
    target_f32 = b.astype(b.array(target), "float32")
    b.eval(source_f32, target_f32)

    m_s, d_s = source_f32.shape
    m_t, d_t = target_f32.shape

    # Same shape - no projection needed
    if m_s == m_t and d_s == d_t:
        return ProjectionResult(
            projected=source_f32,
            alignment_score=1.0,
            method_used=method,
            row_coupling=None,
            col_coupling=None,
        )

    # Dispatch to method - NO FALLBACKS, NO SHORTCUTS
    if method == ProjectionMethod.GRAM_TRANSPORT:
        return _project_gram_transport(source_f32, target_f32, b)
    elif method == ProjectionMethod.PROCRUSTES:
        return _project_procrustes(source_f32, target_f32, b)
    elif method == ProjectionMethod.SVD_PROJECT:
        return _project_svd(source_f32, target_f32, b)
    else:
        raise ValueError(f"Unknown projection method: {method}")


def _project_gram_transport(
    source: "Array",
    target: "Array",
    backend: "Backend",
) -> ProjectionResult:
    """
    Project using Gromov-Wasserstein on Gram matrices.

    CRITICAL: Column-space first. Always.

    The key insight:
    - Column Gram: G_col = W^T @ W is [d×d] - ALWAYS tractable (hidden_dim sized)
    - Row Gram: G_row = W @ W^T is [m×m] - can be INTRACTABLE (vocab_size)

    For weight matrix [m, d]:
    - d is typically hidden_dim (896, 2048, 4096) - tractable
    - m is typically hidden_dim OR vocab_size (150k) - may be intractable

    The algorithm:
    1. ALWAYS compute column Grams first (O(d²) - tractable)
    2. Get column coupling π_col [d_s, d_t]
    3. Apply EXACTLY: W_col_aligned = W @ π_col -> [m_s, d_t]
    4. For row mismatch:
       - If rows tractable (< 20k): compute row GW
       - If rows huge (embeddings): token identity alignment must be handled outside this projector

    Projection:
    - Column dimension: source @ π_col projects columns
    - Row dimension: π_row^T @ source projects rows (only if tractable)
    """
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        GromovWassersteinDistance,
    )

    b = backend
    m_s, d_s = source.shape
    m_t, d_t = target.shape

    gw = GromovWassersteinDistance(b)
    row_coupling = None
    col_coupling = None
    total_score = 0.0
    score_count = 0

    projected = source

    # =========================================================================
    # STEP 1: Handle column dimension mismatch FIRST (always tractable)
    # =========================================================================
    # Column Gram is [d×d] where d is hidden_dim - always tractable
    if d_s != d_t:
        # Column Gram matrices: capture input feature relationships
        # G_col = W^T @ W is [d, d] - hidden_dim sized, NOT vocab_size
        G_source_col = b.matmul(b.transpose(source), source)  # [d_s, d_s]
        G_target_col = b.matmul(b.transpose(target), target)  # [d_t, d_t]
        b.eval(G_source_col, G_target_col)

        logger.debug(
            "Column GW: source Gram [%d, %d], target Gram [%d, %d]",
            d_s, d_s, d_t, d_t
        )

        # GW on column Grams
        result = gw.compute(G_source_col, G_target_col)
        col_coupling = result.coupling  # [d_s, d_t]
        b.eval(col_coupling)

        # Column projection: W @ π maps [m_s, d_s] -> [m_s, d_t]
        # This is EXACT - no approximation, no shortcuts
        projected = b.matmul(projected, col_coupling)
        b.eval(projected)

        total_score += result.alignment_score
        score_count += 1

        logger.debug(
            "Col projection: %d -> %d, GW distance=%.4f, alignment=%.4f",
            d_s, d_t, result.distance, result.alignment_score
        )

    # =========================================================================
    # STEP 2: Handle row dimension mismatch (if tractable)
    # =========================================================================
    current_rows = projected.shape[0]
    if current_rows != m_t:
        # Row Gram would be [m×m]. Check if this is tractable.
        # For attention/MLP weights: m is hidden_dim or intermediate_size (tractable)
        # For embeddings: m is vocab_size (intractable, but should be pre-aligned)
        #
        # MEMORY CONSTRAINT (not approximation):
        # - 20k × 20k Gram = 400M elements × 4 bytes = 1.6 GB
        # - GW requires multiple copies: ~6-8 GB total
        # - For embedding layers (vocab_size > 100k), align token identity before projection
        max_tractable_dim = 20000

        if current_rows <= max_tractable_dim and m_t <= max_tractable_dim:
            # Row Gram is tractable - compute exact GW
            # Use column-aligned projected for source Gram
            G_source_row = b.matmul(projected, b.transpose(projected))  # [m_s, m_s]
            G_target_row = b.matmul(target, b.transpose(target))  # [m_t, m_t]
            b.eval(G_source_row, G_target_row)

            logger.debug(
                "Row GW: source Gram [%d, %d], target Gram [%d, %d]",
                current_rows, current_rows, m_t, m_t
            )

            # GW on row Grams
            result = gw.compute(G_source_row, G_target_row)
            row_coupling = result.coupling  # [m_s, m_t]
            b.eval(row_coupling)

            # Barycentric projection: π^T @ source maps [m_s, d_t] -> [m_t, d_t]
            projected = b.matmul(b.transpose(row_coupling), projected)
            b.eval(projected)

            total_score += result.alignment_score
            score_count += 1

            logger.debug(
                "Row projection: %d -> %d, GW distance=%.4f, alignment=%.4f",
                current_rows, m_t, result.distance, result.alignment_score
            )
        else:
            # Row dimension exceeds standard GW tractability limit (20k).
            # This is common for cross-architecture MLP projection:
            #   - Llama 70B intermediate_size: 28,672
            #   - Qwen 8B intermediate_size: 12,288
            #
            # Use Low-Rank GW which has O((n+m)r²) complexity instead of O(n²m + nm²).
            # This preserves geometry EXACTLY within the low-rank approximation.
            #
            # For embedding layers (vocab_size mismatch), align token identity
            # before projection to preserve discrete token correspondence.
            from modelcypher.core.domain.geometry.low_rank_gw import (
                LowRankGromovWasserstein,
                LowRankGWConfig,
            )

            logger.info(
                "Row dimension (%d -> %d) exceeds GW limit (%d), using Low-Rank GW",
                current_rows, m_t, max_tractable_dim
            )

            # Compute Gram matrices for large row dimensions
            # For very large matrices (> 50k), we compute row Gram in chunks to avoid OOM
            if current_rows > 50000 or m_t > 50000:
                # Use sampling for extremely large dimensions
                sample_size = 10000
                idx_source = list(range(0, current_rows, max(1, current_rows // sample_size)))[:sample_size]
                idx_target = list(range(0, m_t, max(1, m_t // sample_size)))[:sample_size]

                projected_sample = b.take(projected, b.array(idx_source), axis=0)
                target_sample = b.take(target, b.array(idx_target), axis=0)
                b.eval(projected_sample, target_sample)

                G_source_row = b.matmul(projected_sample, b.transpose(projected_sample))
                G_target_row = b.matmul(target_sample, b.transpose(target_sample))
            else:
                # Compute full Gram matrices
                G_source_row = b.matmul(projected, b.transpose(projected))  # [m_s, m_s]
                G_target_row = b.matmul(target, b.transpose(target))  # [m_t, m_t]
            b.eval(G_source_row, G_target_row)

            logger.debug(
                "Low-rank row GW: source Gram [%d, %d], target Gram [%d, %d]",
                G_source_row.shape[0], G_source_row.shape[1],
                G_target_row.shape[0], G_target_row.shape[1]
            )

            # Configure low-rank GW for cross-architecture projection
            # Rank 100-200 is typically sufficient for preserving structure
            svd_source = svd_via_eigh(b, projected, full_matrices=False)
            svd_target = svd_via_eigh(b, target, full_matrices=False)
            _, S_source, _ = svd_source
            _, S_target, _ = svd_target
            b.eval(S_source, S_target)
            S_source_np = [float(v) for v in b.to_numpy(S_source)]
            S_target_np = [float(v) for v in b.to_numpy(S_target)]
            rank_eps = float(machine_epsilon(b, projected))
            shared_rank, _ = compute_shared_relational_rank(
                b,
                S_source_np,
                S_target_np,
                rank_eps,
            )
            rank = max(1, min(shared_rank, int(G_source_row.shape[0]), int(G_target_row.shape[0])))

            lr_config = LowRankGWConfig(
                rank=rank,
                reg=float(regularization_epsilon(b, G_source_row)),
            )

            lr_solver = LowRankGromovWasserstein(b)
            row_result = lr_solver.compute(G_source_row, G_target_row, lr_config)
            row_coupling = row_result.coupling
            b.eval(row_coupling.Q, row_coupling.g, row_coupling.R)

            # Apply row coupling: P^T @ source = R @ diag(1/g) @ Q^T @ source
            projected = row_coupling.apply_left(projected, b)
            b.eval(projected)

            total_score += math.exp(-row_result.distance) if math.isfinite(row_result.distance) else 0.0
            score_count += 1

            logger.info(
                "Low-rank row projection: %d -> %d, iterations=%d, distance=%.4f",
                current_rows, m_t, row_result.iterations, row_result.distance
            )

    # Compute alignment score
    alignment_score = total_score / score_count if score_count > 0 else 1.0

    return ProjectionResult(
        projected=projected,
        alignment_score=alignment_score,
        method_used=ProjectionMethod.GRAM_TRANSPORT,
        row_coupling=row_coupling,
        col_coupling=col_coupling,
    )


def _project_procrustes(
    source: "Array",
    target: "Array",
    backend: "Backend",
) -> ProjectionResult:
    """
    Project using Orthogonal Procrustes alignment.

    Finds optimal rotation R minimizing ||source @ R - target||_F
    via SVD of correlation matrix M = target^T @ source.

    Only works when ONE dimension matches. For full dimension mismatch,
    falls back to gram_transport.
    """
    b = backend
    m_s, d_s = source.shape
    m_t, d_t = target.shape

    # If both dimensions differ, fall back to gram transport
    if m_s != m_t and d_s != d_t:
        logger.debug("Procrustes requires at least one matching dim, using gram_transport")
        return _project_gram_transport(source, target, b)

    # Case 1: Rows match (m_s == m_t), columns differ
    if m_s == m_t and d_s != d_t:
        # Use SVD on column space
        if d_s > d_t:
            # Truncate: use SVD to project to smaller dimension
            _, S, Vt = svd_via_eigh(b, source, full_matrices=False)
            b.eval(S, Vt)

            # Number of components is limited by SVD rank
            # rank = min(m_s, d_s), but Vt.shape[0] gives actual rank
            rank = int(Vt.shape[0])
            k = min(d_t, rank)  # Can only project to min(d_t, rank) dimensions

            # Project to top k dimensions
            V_k = b.transpose(Vt[:k, :])  # [d_s, k]
            projected = b.matmul(source, V_k)  # [m_s, k]
            b.eval(projected)

            # If we couldn't reach d_t dimensions due to rank, pad with zeros
            if k < d_t:
                padding = b.zeros((m_s, d_t - k))
                projected = b.concatenate([projected, padding], axis=1)
                b.eval(projected)

            # Now projected has shape [m_s, d_t]
            # Align to target via Procrustes
            M = b.matmul(b.transpose(target), projected)  # [d_t, d_t]
            U, _, Vt_proc = svd_via_eigh(b, M, full_matrices=False)
            R = b.matmul(U, Vt_proc)  # [d_t, d_t]
            b.eval(R)

            # Handle reflection - flip last column of U if det(R) < 0
            det_R = b.det(R)
            b.eval(det_R)
            if float(b.to_numpy(det_R)) < 0:
                U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                R = b.matmul(U_fixed, Vt_proc)
                b.eval(R)

            projected = b.matmul(projected, R)  # [m_s, d_t]
            b.eval(projected)

            # Alignment score from energy preserved
            total_energy = float(b.to_numpy(b.sum(S ** 2)))
            kept_energy = float(b.to_numpy(b.sum(S[:k] ** 2)))
            eps = float(division_epsilon(b, S))
            score = kept_energy / (total_energy + eps)
        else:
            # Expand: Procrustes on shared dims, pad with zeros
            # Zeros are geometrically exact - introduce no spurious correlations
            source_shared = source
            target_shared = target[:, :d_s]

            M = b.matmul(b.transpose(target_shared), source_shared)
            U, _, Vt_proc = svd_via_eigh(b, M, full_matrices=False)
            R = b.matmul(U, Vt_proc)
            b.eval(R)

            projected_shared = b.matmul(source, R)

            # Pad with zeros - geometrically exact (no spurious correlations)
            # The new dimensions have no information from source, which is correct
            padding = b.zeros((m_s, d_t - d_s))
            projected = b.concatenate([projected_shared, padding], axis=1)
            b.eval(projected)

            score = float(d_s) / float(d_t)  # Score reflects information content

        return ProjectionResult(
            projected=projected,
            alignment_score=score,
            method_used=ProjectionMethod.PROCRUSTES,
            row_coupling=None,
            col_coupling=None,
        )

    # Case 2: Columns match (d_s == d_t), rows differ
    # Transpose, apply case 1 logic, transpose back
    source_T = b.transpose(source)
    target_T = b.transpose(target)

    result_T = _project_procrustes(source_T, target_T, b)
    projected = b.transpose(result_T.projected)
    b.eval(projected)

    return ProjectionResult(
        projected=projected,
        alignment_score=result_T.alignment_score,
        method_used=ProjectionMethod.PROCRUSTES,
        row_coupling=None,
        col_coupling=None,
    )


def _project_svd(
    source: "Array",
    target: "Array",
    backend: "Backend",
) -> ProjectionResult:
    """
    Project using SVD-based subspace alignment.

    Finds shared subspace via truncated SVD of both matrices,
    then aligns via Procrustes on the subspace.
    """
    b = backend
    m_s, d_s = source.shape
    m_t, d_t = target.shape

    # =========================================================================
    # STEP 1: SVD both matrices
    # =========================================================================
    # Use a single stable path (svd_via_eigh) for all aspect ratios.
    _, S_s, Vt_s = svd_via_eigh(b, source, full_matrices=False)
    _, S_t, Vt_t = svd_via_eigh(b, target, full_matrices=False)

    b.eval(S_s, Vt_s, S_t, Vt_t)

    # =========================================================================
    # STEP 2: Find shared subspace dimension (rank-aware)
    # =========================================================================
    # Compute numerical rank for each matrix to avoid projecting onto null space
    max_dim_s = max(m_s, d_s)
    max_dim_t = max(m_t, d_t)
    rank_thresh_s = svd_rank_threshold(b, source, max_dim_s)  # returns float
    rank_thresh_t = svd_rank_threshold(b, target, max_dim_t)  # returns float

    # Count singular values above threshold (numerical rank)
    S_s_np = b.to_numpy(S_s)
    S_t_np = b.to_numpy(S_t)
    # Multiply threshold by max singular value for proper scaling
    thresh_s = rank_thresh_s * (float(S_s_np[0]) if len(S_s_np) > 0 else 1.0)
    thresh_t = rank_thresh_t * (float(S_t_np[0]) if len(S_t_np) > 0 else 1.0)
    rank_s = int((S_s_np > thresh_s).sum())
    rank_t = int((S_t_np > thresh_t).sum())

    # Use minimum of ranks and dimensions for safe truncation
    k = min(rank_s, rank_t, d_s, d_t, int(S_s.shape[0]), int(S_t.shape[0]))
    k = max(k, 1)  # Ensure at least 1 dimension

    # =========================================================================
    # STEP 3: Project source to shared subspace
    # =========================================================================
    # Project source columns to top-k right singular vectors
    V_s_k = b.transpose(Vt_s[:k, :])  # [d_s, k]
    V_t_k = b.transpose(Vt_t[:k, :])  # [d_t, k] - defined early for row mismatch handling
    source_k = b.matmul(source, V_s_k)  # [m_s, k]
    b.eval(source_k, V_t_k)

    # =========================================================================
    # STEP 4: Handle row dimension mismatch via SVD-based scaling
    # =========================================================================
    # For cross-architecture merges, row mismatches can be large (different
    # intermediate_size, different num_kv_heads, etc.). We handle ALL mismatches
    # via SVD truncation/padding - NEVER fall back to GW on weights.
    #
    # Embedding layers (vocab_size mismatch) require token identity alignment
    # before projection to preserve discrete token correspondence.
    if m_s != m_t:
        mismatch = abs(m_s - m_t)
        logger.debug(
            "Row dimension mismatch: %d -> %d (delta=%d), using SVD scaling",
            m_s, m_t, mismatch
        )

        if m_s > m_t:
            # Truncate: keep top m_t rows by magnitude in SVD subspace
            # This preserves the most important components
            row_norms = b.sqrt(b.sum(source_k ** 2, axis=1))
            b.eval(row_norms)
            # Sort by negative norms (descending order) and keep top m_t
            neg_norms = -row_norms
            indices = b.argsort(neg_norms)[:m_t]
            b.eval(indices)
            source_k = b.take(source_k, indices, axis=0)
            b.eval(source_k)
        else:
            # Expand: initialize new rows from target subspace, scaled
            # to match the source's average row norm in the shared space.
            n_new = m_t - m_s
            # Get target's projection to the shared subspace
            target_k = b.matmul(target, V_t_k)
            b.eval(target_k)
            source_norms = b.norm(source_k, axis=1)
            target_norms = b.norm(target_k, axis=1)
            b.eval(source_norms, target_norms)
            source_mean = (
                float(b.to_numpy(b.mean(source_norms))) if int(source_norms.shape[0]) > 0 else 0.0
            )
            target_mean = (
                float(b.to_numpy(b.mean(target_norms))) if int(target_norms.shape[0]) > 0 else 0.0
            )
            scale_eps = division_epsilon(b, target_k)
            scale = source_mean / (target_mean + scale_eps) if source_mean > 0.0 else 0.0
            # Use the first n_new rows of target (scaled down) for expansion
            if n_new <= m_t:
                expansion = target_k[:n_new, :] * scale
            else:
                # Need more rows than target has - tile and truncate
                repeats = (n_new // m_t) + 1
                tiled = b.concatenate([target_k] * repeats, axis=0)
                expansion = tiled[:n_new, :] * scale
            b.eval(expansion)
            source_k = b.concatenate([source_k, expansion], axis=0)
            b.eval(source_k)

    # =========================================================================
    # STEP 4.5: Procrustes alignment in shared k-dimensional subspace
    # =========================================================================
    # source_k and target_k are both in k-dimensional subspace, but the
    # basis vectors V_s_k and V_t_k may represent different directions.
    # We need to find the optimal rotation R such that source_k @ R ≈ target_k.
    #
    # This is the critical step that was missing - without it, SVD projection
    # produces weights with similar magnitude but wrong direction (cosine sim ≈ 0).
    target_k = b.matmul(target, V_t_k)  # [m_t, k]
    b.eval(target_k)

    # Center both for Procrustes (improves alignment stability)
    source_k_mean = b.mean(source_k, axis=0, keepdims=True)
    target_k_mean = b.mean(target_k, axis=0, keepdims=True)
    source_k_centered = source_k - source_k_mean
    target_k_centered = target_k - target_k_mean
    b.eval(source_k_centered, target_k_centered)

    # Compute cross-covariance matrix: M = source_k.T @ target_k [k, k]
    M = b.matmul(b.transpose(source_k_centered), target_k_centered)
    b.eval(M)

    # SVD of M to find optimal rotation: M = U @ S @ V.T, R = V @ U.T
    U_m, S_m, Vt_m = svd_via_eigh(b, M, full_matrices=False)
    b.eval(U_m, Vt_m)

    # R = V @ U.T is the optimal orthogonal matrix (Procrustes solution)
    R = b.matmul(b.transpose(Vt_m), b.transpose(U_m))
    b.eval(R)

    # Ensure R is orthogonal (correct for numerical errors)
    # det(R) should be +1 for proper rotation. If -1, flip the sign of one column.
    # For weight alignment, we allow reflections (det = -1) as they still preserve
    # distances and can represent valid transformations.

    # Apply rotation to source_k
    source_k_aligned = b.matmul(source_k, R)
    b.eval(source_k_aligned)

    # Log alignment quality in k-space
    residual_k = source_k_aligned - target_k
    residual_norm = float(b.to_numpy(b.sqrt(b.sum(residual_k ** 2))))
    target_k_norm = float(b.to_numpy(b.sqrt(b.sum(target_k ** 2))))
    div_eps = division_epsilon(b, target_k)
    logger.debug(
        "Subspace Procrustes: k=%d, residual_norm=%.4f, target_norm=%.4f, ratio=%.4f",
        k, residual_norm, target_k_norm, residual_norm / (target_k_norm + div_eps)
    )

    # =========================================================================
    # STEP 5: Project to target's column space
    # =========================================================================
    projected = b.matmul(source_k_aligned, b.transpose(V_t_k))  # [m_t, d_t]
    b.eval(projected)

    # =========================================================================
    # STEP 5.5: Scale projected weights to match target's Frobenius norm
    # =========================================================================
    # Without scaling, SVD projection can produce weights with very different
    # magnitude than the target, causing activation explosion during inference.
    target_fro = b.sqrt(b.sum(target ** 2))
    proj_fro = b.sqrt(b.sum(projected ** 2))
    b.eval(target_fro, proj_fro)
    scale_eps = division_epsilon(b, projected)
    scale_factor = target_fro / (proj_fro + scale_eps)
    projected = projected * scale_factor
    b.eval(projected)
    logger.debug(
        "SVD projection scale: target_fro=%.4f, proj_fro=%.4f, scale_factor=%.4f",
        float(b.to_numpy(target_fro)),
        float(b.to_numpy(proj_fro)),
        float(b.to_numpy(scale_factor)),
    )

    # =========================================================================
    # STEP 6: Compute alignment score from variance preserved
    # =========================================================================
    total_var_s = float(b.to_numpy(b.sum(S_s ** 2)))
    kept_var_s = float(b.to_numpy(b.sum(S_s[:k] ** 2)))

    total_var_t = float(b.to_numpy(b.sum(S_t ** 2)))
    kept_var_t = float(b.to_numpy(b.sum(S_t[:k] ** 2)))

    var_eps = float(division_epsilon(b, S_s))
    score = 0.5 * (kept_var_s / (total_var_s + var_eps) + kept_var_t / (total_var_t + var_eps))

    return ProjectionResult(
        projected=projected,
        alignment_score=score,
        method_used=ProjectionMethod.SVD_PROJECT,
        row_coupling=None,
        col_coupling=None,
    )
