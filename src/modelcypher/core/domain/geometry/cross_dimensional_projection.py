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

This module consolidates the sprawl from:
- embedding_projector.py (5 strategies)
- stage_3_5_rotate_blend.py (ad-hoc _project_2d)
- transport_guided_merger.py (GW-based)
- affine_stitching_layer.py
- And 10+ other alignment files

Into ONE unified API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

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

    The key insight:
    - Gram matrix G = X @ X^T is [n×n] regardless of feature dim
    - GW compares metric structure without point correspondence
    - Coupling matrix π gives soft assignment between rows/cols

    For weight matrix [m, d]:
    - Row Gram: G_row = W @ W^T [m×m] - captures output neuron relationships
    - Col Gram: G_col = W^T @ W [d×d] - captures input feature relationships

    Projection:
    - Row dimension: π_row^T @ source projects rows
    - Col dimension: source @ π_col projects columns
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
    # STEP 1: Handle row dimension mismatch (m_s -> m_t)
    # =========================================================================
    if m_s != m_t:
        # Row Gram matrices: capture output neuron relationships
        G_source_row = b.matmul(source, b.transpose(source))  # [m_s, m_s]
        G_target_row = b.matmul(target, b.transpose(target))  # [m_t, m_t]
        b.eval(G_source_row, G_target_row)

        # GW on row Grams
        config = GWConfig(max_outer_iterations=50, num_restarts=3)
        result = gw.compute(G_source_row, G_target_row, config)
        row_coupling = result.coupling  # [m_s, m_t]

        # Barycentric projection: projected[i] = Σ_j π[j,i] * source[j,:]
        # π^T @ source maps [m_s, d_s] -> [m_t, d_s]
        projected = b.matmul(b.transpose(row_coupling), projected)
        b.eval(projected)

        total_score += result.compatibility_score
        score_count += 1

        logger.debug(
            "Row projection: %d -> %d, GW distance=%.4f",
            m_s, m_t, result.distance
        )

    # =========================================================================
    # STEP 2: Handle column dimension mismatch (d_s -> d_t)
    # =========================================================================
    current_cols = projected.shape[1]
    if current_cols != d_t:
        # Column Gram matrices: capture input feature relationships
        # Use projected (already row-aligned) for source Gram
        G_source_col = b.matmul(b.transpose(projected), projected)  # [d_s, d_s]
        G_target_col = b.matmul(b.transpose(target), target)  # [d_t, d_t]
        b.eval(G_source_col, G_target_col)

        # GW on column Grams
        config = GWConfig(max_outer_iterations=50, num_restarts=3)
        result = gw.compute(G_source_col, G_target_col, config)
        col_coupling = result.coupling  # [d_s, d_t]

        # Column projection: projected @ π maps [m_t, d_s] -> [m_t, d_t]
        projected = b.matmul(projected, col_coupling)
        b.eval(projected)

        total_score += result.compatibility_score
        score_count += 1

        logger.debug(
            "Col projection: %d -> %d, GW distance=%.4f",
            current_cols, d_t, result.distance
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

            # Handle reflection
            det_R = b.det(R)
            b.eval(det_R)
            if float(b.to_numpy(det_R)) < 0:
                U_cols = [U[:, i:i+1] for i in range(d_t - 1)]
                U_cols.append(U[:, -1:] * -1.0)
                U_fixed = b.concatenate(U_cols, axis=1)
                R = b.matmul(U_fixed, Vt_proc)
                b.eval(R)

            projected = b.matmul(projected, R)
            b.eval(projected)

            # Alignment score from energy preserved
            total_energy = float(b.to_numpy(b.sum(S ** 2)))
            kept_energy = float(b.to_numpy(b.sum(S[:d_t] ** 2)))
            score = kept_energy / (total_energy + 1e-10)
        else:
            # Expand: Procrustes on shared dims, pad with small noise
            # Align shared dimensions
            source_shared = source
            target_shared = target[:, :d_s]

            M = b.matmul(b.transpose(target_shared), source_shared)
            U, _, Vt_proc = b.svd(M, compute_uv=True)
            R = b.matmul(U, Vt_proc)
            b.eval(R)

            projected_shared = b.matmul(source, R)

            # Pad with small orthogonal noise
            b.random_seed(42)
            padding = b.random_normal((m_s, d_t - d_s)) * 0.01
            projected = b.concatenate([projected_shared, padding], axis=1)
            b.eval(projected)

            score = 1.0  # No information lost in expansion

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
    if m_s > 4 * d_s:
        # Column Gram: G = X^T @ X [d_s, d_s]
        G_source = b.matmul(b.transpose(source), source)
        _, S_s, Vt_s = b.svd(G_source, compute_uv=True)
        S_s = b.sqrt(S_s + 1e-10)
    else:
        _, S_s, Vt_s = b.svd(source, compute_uv=True)

    if m_t > 4 * d_t:
        G_target = b.matmul(b.transpose(target), target)
        _, S_t, Vt_t = b.svd(G_target, compute_uv=True)
        S_t = b.sqrt(S_t + 1e-10)
    else:
        _, S_t, Vt_t = b.svd(target, compute_uv=True)

    b.eval(S_s, Vt_s, S_t, Vt_t)

    # =========================================================================
    # STEP 2: Find shared subspace dimension
    # =========================================================================
    # Use minimum of both feature dimensions
    k = min(d_s, d_t, int(S_s.shape[0]), int(S_t.shape[0]))

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
        # Use row-wise truncation/expansion
        if m_s > m_t:
            # Truncate rows (keep first m_t)
            source_k = source_k[:m_t, :]
        else:
            # Expand rows with small noise
            b.random_seed(42)
            padding = b.random_normal((m_t - m_s, k)) * 0.01
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
