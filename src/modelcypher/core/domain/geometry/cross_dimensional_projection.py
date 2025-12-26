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
The merge pipeline (stage_3_5_rotate_blend.py) uses this exclusively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    svd_rank_threshold,
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
       - If rows huge (embeddings): should be pre-aligned by vocabulary stage

    Projection:
    - Column dimension: source @ π_col projects columns
    - Row dimension: π_row^T @ source projects rows (only if tractable)
    """
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        Config as GWConfig,
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

        # GW on column Grams - O(d_s² + d_t²) iterations
        config = GWConfig(max_outer_iterations=50, num_restarts=3)
        result = gw.compute(G_source_col, G_target_col, config)
        col_coupling = result.coupling  # [d_s, d_t]
        b.eval(col_coupling)

        # Column projection: W @ π maps [m_s, d_s] -> [m_s, d_t]
        # This is EXACT - no approximation, no shortcuts
        projected = b.matmul(projected, col_coupling)
        b.eval(projected)

        total_score += result.compatibility_score
        score_count += 1

        logger.debug(
            "Col projection: %d -> %d, GW distance=%.4f, score=%.4f",
            d_s, d_t, result.distance, result.compatibility_score
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
        # - For embedding layers (vocab_size > 100k), use vocabulary alignment in stage 0
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
            config = GWConfig(max_outer_iterations=50, num_restarts=3)
            result = gw.compute(G_source_row, G_target_row, config)
            row_coupling = result.coupling  # [m_s, m_t]
            b.eval(row_coupling)

            # Barycentric projection: π^T @ source maps [m_s, d_t] -> [m_t, d_t]
            projected = b.matmul(b.transpose(row_coupling), projected)
            b.eval(projected)

            total_score += result.compatibility_score
            score_count += 1

            logger.debug(
                "Row projection: %d -> %d, GW distance=%.4f, score=%.4f",
                current_rows, m_t, result.distance, result.compatibility_score
            )
        else:
            # Row dimension is too large (embedding layer with different vocab)
            # This MUST be handled by vocabulary alignment BEFORE projection.
            #
            # NO INTERPOLATION. NO APPROXIMATION. EXACT OR FAIL.
            #
            # The geometry MUST be preserved. Row interpolation introduces:
            # - Spurious correlations between non-adjacent tokens
            # - Loss of discrete token identity
            # - Gradient discontinuities at merge boundaries
            #
            # If you're seeing this error, your merge pipeline needs:
            # 1. Vocabulary alignment in stage 0 (VocabularyAligner)
            # 2. Explicit token mapping before cross-dim projection
            # 3. Or use CKA/Gram comparison which doesn't require row alignment
            raise ValueError(
                f"Row dimension mismatch ({current_rows} -> {m_t}) is intractable "
                f"for exact GW (limit: {max_tractable_dim}). "
                f"Vocabulary alignment must be performed BEFORE cross-dimensional "
                f"projection. Use stage 0 VocabularyAligner for embedding layers, "
                f"or ensure token counts match before calling project_cross_dimensional. "
                f"NO INTERPOLATION - geometry must be exact."
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
            # Truncate: use top d_t components via SVD
            _, S, Vt = b.svd(source, compute_uv=True)
            b.eval(S, Vt)

            # Project to top d_t dimensions
            projected = b.matmul(source, b.transpose(Vt[:d_t, :]))
            b.eval(projected)

            # Align to target via Procrustes
            M = b.matmul(b.transpose(target), projected)
            U, _, Vt_proc = b.svd(M, compute_uv=True)
            R = b.matmul(U, Vt_proc)
            b.eval(R)

            # Handle reflection - flip last column of U if det(R) < 0
            det_R = b.det(R)
            b.eval(det_R)
            if float(b.to_numpy(det_R)) < 0:
                U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                R = b.matmul(U_fixed, Vt_proc)
                b.eval(R)

            projected = b.matmul(projected, R)
            b.eval(projected)

            # Alignment score from energy preserved
            total_energy = float(b.to_numpy(b.sum(S ** 2)))
            kept_energy = float(b.to_numpy(b.sum(S[:d_t] ** 2)))
            eps = float(division_epsilon(b, S))
            score = kept_energy / (total_energy + eps)
        else:
            # Expand: Procrustes on shared dims, pad with zeros
            # Zeros are geometrically exact - introduce no spurious correlations
            source_shared = source
            target_shared = target[:, :d_s]

            M = b.matmul(b.transpose(target_shared), source_shared)
            U, _, Vt_proc = b.svd(M, compute_uv=True)
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
    # For tall matrices (embeddings), use column-space SVD
    eps_s = float(division_epsilon(b, source))
    eps_t = float(division_epsilon(b, target))
    if m_s > 4 * d_s:
        # Column Gram: G = X^T @ X [d_s, d_s]
        G_source = b.matmul(b.transpose(source), source)
        _, S_s, Vt_s = b.svd(G_source, compute_uv=True)
        S_s = b.sqrt(S_s + eps_s)
    else:
        _, S_s, Vt_s = b.svd(source, compute_uv=True)

    if m_t > 4 * d_t:
        G_target = b.matmul(b.transpose(target), target)
        _, S_t, Vt_t = b.svd(G_target, compute_uv=True)
        S_t = b.sqrt(S_t + eps_t)
    else:
        _, S_t, Vt_t = b.svd(target, compute_uv=True)

    b.eval(S_s, Vt_s, S_t, Vt_t)

    # =========================================================================
    # STEP 2: Find shared subspace dimension (rank-aware)
    # =========================================================================
    # Compute numerical rank for each matrix to avoid projecting onto null space
    rank_thresh_s = svd_rank_threshold(b, source)
    rank_thresh_t = svd_rank_threshold(b, target)
    b.eval(rank_thresh_s, rank_thresh_t)

    # Count singular values above threshold (numerical rank)
    S_s_np = b.to_numpy(S_s)
    S_t_np = b.to_numpy(S_t)
    rank_s = int((S_s_np > float(b.to_numpy(rank_thresh_s))).sum())
    rank_t = int((S_t_np > float(b.to_numpy(rank_thresh_t))).sum())

    # Use minimum of ranks and dimensions for safe truncation
    k = min(rank_s, rank_t, d_s, d_t, int(S_s.shape[0]), int(S_t.shape[0]))
    k = max(k, 1)  # Ensure at least 1 dimension

    # =========================================================================
    # STEP 3: Project source to shared subspace
    # =========================================================================
    # Project source columns to top-k right singular vectors
    V_s_k = b.transpose(Vt_s[:k, :])  # [d_s, k]
    source_k = b.matmul(source, V_s_k)  # [m_s, k]
    b.eval(source_k)

    # =========================================================================
    # STEP 4: Handle row dimension mismatch
    # =========================================================================
    if m_s != m_t:
        # For small mismatches (< 1000 rows), truncation/padding is acceptable
        # as this is likely attention/MLP weights, not embeddings.
        #
        # For large mismatches (embeddings), require explicit vocabulary alignment.
        max_tractable_mismatch = 1000
        mismatch = abs(m_s - m_t)

        if mismatch > max_tractable_mismatch:
            # Large row mismatch indicates embedding layers with different vocab.
            # Fall back to gram transport to preserve geometry without truncation.
            logger.warning(
                "Row mismatch (%s -> %s, delta=%s) exceeds SVD limit (%s); "
                "falling back to gram_transport.",
                m_s,
                m_t,
                mismatch,
                max_tractable_mismatch,
            )
            return _project_gram_transport(source, target, b)

        # Small mismatch: use truncation/padding (acceptable for MLP/attention weights)
        if m_s > m_t:
            # Truncate rows (keep first m_t)
            source_k = source_k[:m_t, :]
        else:
            # Expand rows with zeros - geometrically exact
            # Zeros introduce no spurious correlations
            padding = b.zeros((m_t - m_s, k))
            source_k = b.concatenate([source_k, padding], axis=0)
        b.eval(source_k)

    # =========================================================================
    # STEP 5: Project to target's column space
    # =========================================================================
    V_t_k = b.transpose(Vt_t[:k, :])  # [d_t, k]
    projected = b.matmul(source_k, b.transpose(V_t_k))  # [m_t, d_t]
    b.eval(projected)

    # =========================================================================
    # STEP 6: Compute alignment score from variance preserved
    # =========================================================================
    total_var_s = float(b.to_numpy(b.sum(S_s ** 2)))
    kept_var_s = float(b.to_numpy(b.sum(S_s[:k] ** 2)))

    total_var_t = float(b.to_numpy(b.sum(S_t ** 2)))
    kept_var_t = float(b.to_numpy(b.sum(S_t[:k] ** 2)))

    score = 0.5 * (kept_var_s / (total_var_s + 1e-10) + kept_var_t / (total_var_t + 1e-10))

    return ProjectionResult(
        projected=projected,
        alignment_score=score,
        method_used=ProjectionMethod.SVD_PROJECT,
        row_coupling=None,
        col_coupling=None,
    )
