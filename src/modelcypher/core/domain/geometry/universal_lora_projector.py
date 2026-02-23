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

"""Universal LoRA Projector for cross-model adapter transfer.

This module implements training-free LoRA adapter transfer between different
base models using geometric subspace alignment. The algorithm is inspired by
Cross-LoRA (2025) but built on ModelCypher's existing geometry primitives.

The transfer process has two phases:
    1. LoRA-Align: Compute subspace alignment between source/target SVD bases
    2. LoRA-Shift: Project source ΔW into target's parameter space

Key Insight:
    LoRA adapters learn low-rank updates ΔW = B @ A that live in specific
    subspaces of the weight matrix. If we can align the subspaces of the
    source and target base models, we can project the adapter update into
    the target's coordinate system.

Mathematical Foundation:
    Given source base weight W_s with SVD: W_s = U_s @ S_s @ V_s^T
    And target base weight W_t with SVD: W_t = U_t @ S_t @ V_t^T

    The LoRA update ΔW_s lives in span(U_s) × span(V_s).
    We project it to span(U_t) × span(V_t) via:
        ΔW_t = U_t @ U_t^T @ ΔW_s @ V_s @ V_t^T
    
    With Procrustes refinement to minimize ||ΔW_t - aligned||_F.

Usage:
    from modelcypher.core.domain.geometry.universal_lora_projector import (
        UniversalLoRAProjector,
        LoRATransferResult,
    )

    projector = UniversalLoRAProjector(backend=backend)
    transferred, result = projector.transfer(
        source_adapter_weights=adapter_weights,
        source_base_svd=source_svd,
        target_base_svd=target_svd,
    )

References:
    - Cross-LoRA (2025): Training-free transfer via subspace alignment
    - ModelCypher Procrustes: backend_matrix_utils.py
    - ModelCypher Subspace: subspace.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.backend_matrix_utils import BackendMatrixUtils
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
    svd_rank_threshold,
)
from modelcypher.core.domain.geometry.subspace import (
    compute_grassmann_distance,
    compute_subspace_projector,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GQAConfig:
    """Configuration for Grouped Query Attention handling.

    GQA models (Llama-3, Mistral) use fewer K/V heads than Q heads.
    When transferring from dense attention (all heads separate) to GQA,
    we must group the source projection into H_tgt groups and project
    each group independently.

    Attributes:
        source_heads: Number of attention heads in source (Q heads)
        target_heads: Number of K/V heads in target (may differ for GQA)
        source_head_dim: Dimension per head in source
        target_head_dim: Dimension per head in target
    """

    source_heads: int
    target_heads: int
    source_head_dim: int
    target_head_dim: int

    @property
    def source_kv_heads(self) -> int:
        """K/V heads in source (same as Q for dense, less for GQA)."""
        return self.source_heads

    @property
    def target_kv_heads(self) -> int:
        """K/V heads in target."""
        return self.target_heads

    @property
    def groups_ratio(self) -> float:
        """Ratio of source to target heads."""
        return self.source_heads / self.target_heads if self.target_heads > 0 else 1.0

    @property
    def is_gqa_transfer(self) -> bool:
        """True if this is a cross-GQA transfer requiring grouping."""
        return self.source_heads != self.target_heads


@dataclass
class LayerTransferResult:
    """Result of transferring a single layer's LoRA weights."""

    layer_key: str
    source_rank: int
    target_rank: int
    grassmann_distance: float
    projection_error: float  # ||original - projected||_F / ||original||_F
    was_truncated: bool  # True if rank was reduced during transfer
    gqa_groups_used: int = 1  # Number of GQA groups used in transfer
    warning: str | None = None


@dataclass
class LoRATransferResult:
    """Result of transferring a complete LoRA adapter between models.

    Combines geometric metrics (Grassmann distance, projection error) with
    optional semantic drift verification for comprehensive transfer assessment.
    """

    source_base_model: str
    target_base_model: str
    layers_transferred: int
    layers_skipped: int
    mean_grassmann_distance: float
    mean_projection_error: float
    max_projection_error: float
    layer_results: list[LayerTransferResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Optional semantic drift verification (set after running SemanticProbeVerifier)
    semantic_drift: Any | None = None  # SemanticDriftResult when verified

    @property
    def success(self) -> bool:
        """True when transfer completed without layer-level warnings."""
        if self.layers_transferred == 0:
            return False
        if self.warnings:
            return False
        return self.mean_projection_error >= 0.0 and self.max_projection_error >= 0.0

    @property
    def semantic_verified(self) -> bool:
        """True if semantic verification was run and passed."""
        if self.semantic_drift is None:
            return False
        return getattr(self.semantic_drift, "passed", False)


# =============================================================================
# SVD Cache Entry
# =============================================================================


@dataclass
class SVDComponents:
    """SVD components for a weight matrix: W = U @ S @ Vt.

    U and Vt are always full-dimensional (matching the original weight),
    even when row subsampling is used during SVD computation. This is
    achieved by reconstructing U from the full weight: U = W @ V @ S^{-1}.
    """

    U: "Array"  # Left singular vectors [m, k]
    S: "Array"  # Singular values [k]
    Vt: "Array"  # Right singular vectors [k, n]
    effective_rank: int  # Number of significant singular values


# =============================================================================
# Universal LoRA Projector
# =============================================================================


class UniversalLoRAProjector:
    """Transfer LoRA adapters between different base models.

    Uses subspace alignment to project adapter weights from source model's
    coordinate system to target model's coordinate system without retraining.

    Example:
        projector = UniversalLoRAProjector(backend=backend)
        
        # Prepare SVD caches for both models
        source_svd = projector.compute_layer_svd(source_base_weights)
        target_svd = projector.compute_layer_svd(target_base_weights)
        
        # Transfer the adapter
        transferred, result = projector.transfer(
            source_adapter_weights=adapter_weights,
            source_base_svd=source_svd,
            target_base_svd=target_svd,
        )
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        rank_threshold: float | None = None,
    ) -> None:
        """Initialize the projector.

        Args:
            backend: Compute backend (uses default if None)
            rank_threshold: Override for SVD rank threshold (uses data-derived if None)
        """
        self._backend = backend or get_default_backend()
        self._matrix_utils = BackendMatrixUtils(self._backend)
        self._rank_threshold = rank_threshold

    @property
    def backend(self) -> "Backend":
        """Get compute backend."""
        return self._backend

    def compute_layer_svd(
        self,
        weight: "Array",
        max_rank: int | None = None,
        sample_size: int | None = None,
    ) -> SVDComponents:
        """Compute SVD components for a weight matrix.

        Args:
            weight: Weight matrix [out_features, in_features]
            max_rank: Maximum rank to keep (keeps all significant if None)
            sample_size: Max rows to use for SVD approximation (None = no subsampling).
                         WARNING: Subsampling is only for analysis/profiling. For transfer
                         operations, use None (full SVD) to ensure numerical stability.

        Returns:
            SVDComponents with U, S, Vt and effective rank
        """
        b = self._backend

        # Subsample rows if requested (for analysis/profiling only)
        # WARNING: Subsampling can cause numerical instability in transfer operations.
        # For transfer, use sample_size=None (the default).
        src_shape = b.shape(weight)
        rows = int(src_shape[0])

        if sample_size is not None and rows > sample_size:
            # Stride-based sampling for determinism
            step = max(1, rows // sample_size)
            # MLX/Numpy slicing [::step]
            # We assume backend supports basic slicing. If not, this might be tricky.
            # But standard numpy/mlx supports it.
            # Note: We can't easily slice generically without knowing the backend type
            # but standard syntax `weight[::step]` works for both usually.
            W_subset = weight[::step]
            if int(b.shape(W_subset)[0]) > sample_size:
                W_subset = W_subset[:sample_size]
        else:
            W_subset = weight

        # Promote to high precision
        W = b.astype(W_subset, precision_dtype(b, reference=W_subset))
        b.eval(W)

        # Full SVD
        try:
            U, S, Vt = b.svd(W, compute_uv=True)
        except Exception:
            # Fallback to CPU if possible or re-raise
            # For now, strict failure but the subsampling should prevent it
            raise

        b.eval(U, S, Vt)

        # Compute effective rank using data-derived threshold
        # max_dim is the larger dimension of the *subsampled* matrix
        shape = b.shape(W)
        max_dim = max(int(shape[0]), int(shape[1]))
        threshold = svd_rank_threshold(b, S, max_dim)
        significant_mask = S > threshold
        b.eval(significant_mask)

        count_dtype = precision_dtype(b, reference=S)
        eff_rank = int(b.to_scalar(b.sum(b.astype(significant_mask, count_dtype))))
        eff_rank = max(1, eff_rank)  # At least 1

        # Optionally truncate
        if max_rank is not None and eff_rank > max_rank:
            eff_rank = max_rank

        # Truncate to effective rank
        idx = b.arange(0, eff_rank)

        S_trunc = b.take(S, idx, axis=0)  # [k]
        Vt_trunc = b.take(Vt, idx, axis=0)  # [k, n]
        b.eval(S_trunc, Vt_trunc)

        # If we subsampled rows, reconstruct full U from original weight.
        # Note: This reconstruction can be numerically unstable when S has small values.
        # For transfer operations, use sample_size=None to avoid this issue.
        was_subsampled = sample_size is not None and rows > sample_size

        if was_subsampled:
            # Reconstruct U_full from original weight
            # U = W @ V @ diag(1/S)
            W_full = b.astype(weight, precision_dtype(b, reference=weight))
            b.eval(W_full)

            V_trunc = b.transpose(Vt_trunc)  # [n, k]
            b.eval(V_trunc)

            # Compute S_inv = diag(1/S) with numerical stability
            eps = division_epsilon(b, S_trunc)
            S_inv = 1.0 / (S_trunc + eps)  # [k]
            b.eval(S_inv)

            # U_full = W @ V @ diag(S_inv)
            U_full = b.matmul(W_full, V_trunc)  # [rows, k]
            U_full = U_full * b.reshape(S_inv, (1, -1))  # broadcast multiply
            b.eval(U_full)
        else:
            # No subsampling: use U directly from SVD
            U_full = b.take(U, idx, axis=1)  # [rows, k]
            b.eval(U_full)

        return SVDComponents(
            U=U_full,
            S=S_trunc,
            Vt=Vt_trunc,
            effective_rank=eff_rank,
        )

    def _align_subspaces(
        self,
        source_svd: SVDComponents,
        target_svd: SVDComponents,
    ) -> tuple["Array", "Array"]:
        """Compute alignment matrices between source and target subspaces.

        Uses Procrustes analysis to find optimal rotations that align
        the source singular vectors to target singular vectors.

        Args:
            source_svd: SVD of source base weight
            target_svd: SVD of target base weight

        Returns:
            Tuple (R_left, R_right) where:
                R_left aligns source U to target U
                R_right aligns source V to target V
        """
        b = self._backend

        # Get the minimum shared rank
        k_src = source_svd.effective_rank
        k_tgt = target_svd.effective_rank
        k_shared = min(k_src, k_tgt)

        # Truncate to shared rank for alignment
        idx = b.arange(0, k_shared)

        U_src = b.take(source_svd.U, idx, axis=1)  # [m_src, k]
        U_tgt = b.take(target_svd.U, idx, axis=1)  # [m_tgt, k]
        b.eval(U_src, U_tgt)

        # V needs to come from Vt (transpose to get V)
        Vt_src = b.take(source_svd.Vt, idx, axis=0)  # [k, n_src]
        Vt_tgt = b.take(target_svd.Vt, idx, axis=0)  # [k, n_tgt]
        V_src = b.transpose(Vt_src)  # [n_src, k]
        V_tgt = b.transpose(Vt_tgt)  # [n_tgt, k]
        b.eval(V_src, V_tgt)

        # Dimension checks
        m_src, m_tgt = int(b.shape(U_src)[0]), int(b.shape(U_tgt)[0])
        n_src, n_tgt = int(b.shape(V_src)[0]), int(b.shape(V_tgt)[0])

        # For same-dimension case: use Procrustes for optimal rotation
        # For different-dimension case: use projection + truncation

        if m_src == m_tgt:
            # Left space alignment: find R_left s.t. U_src @ R_left ≈ U_tgt
            # Procrustes: M = U_src^T @ U_tgt, SVD -> R = U @ V^T
            result_left = self._matrix_utils.procrustes_rotation(U_src, U_tgt)
            R_left = result_left.rotation
        else:
            # Different dimensions: create projection matrix
            # R_left projects from m_src -> m_tgt via shared k-dim space
            # R_left = U_tgt @ U_src^T (maps m_src -> m_tgt through k-dim)
            R_left = b.matmul(U_tgt, b.transpose(U_src))
        b.eval(R_left)

        if n_src == n_tgt:
            # Right space alignment: find R_right s.t. V_src @ R_right ≈ V_tgt
            result_right = self._matrix_utils.procrustes_rotation(V_src, V_tgt)
            R_right = result_right.rotation
        else:
            # Different dimensions: create projection matrix
            # R_right projects from n_src -> n_tgt via shared k-dim space
            R_right = b.matmul(V_tgt, b.transpose(V_src))
        b.eval(R_right)

        return R_left, R_right

    def _transfer_gqa_grouped(
        self,
        lora_delta: "Array",
        source_svd: SVDComponents,
        target_svd: SVDComponents,
        gqa_config: GQAConfig,
        layer_key: str = "",
    ) -> tuple["Array", float, int]:
        """Transfer with GQA grouping for mismatched attention head counts.

        When source has dense attention (H_src heads) and target has GQA
        (H_tgt K/V heads), we split the projection into H_tgt groups.

        Each group processes H_src/H_tgt source heads and projects them
        to 1 target head's worth of dimensions.

        Algorithm:
            1. Reshape delta from [out, in] to [H_tgt, head_out, in]
            2. For each group g in H_tgt:
                a. Extract delta_g = delta[g * stride : (g+1) * stride, :]
                b. Project delta_g using truncated source/target SVD
                c. Store in result[g, :, :]
            3. Reshape result back to [out, in]

        Args:
            lora_delta: LoRA weight update [out_features, in_features]
            source_svd: SVD of source base weight
            target_svd: SVD of target base weight
            gqa_config: GQA configuration with head counts
            layer_key: Identifier for logging

        Returns:
            Tuple (transferred_delta, projection_error, num_groups)
        """
        b = self._backend

        # Promote to high precision
        delta = b.astype(lora_delta, precision_dtype(b, reference=lora_delta))
        b.eval(delta)

        src_shape = b.shape(delta)
        m_src, n_src = int(src_shape[0]), int(src_shape[1])

        # Target dimensions
        m_tgt = int(b.shape(target_svd.U)[0])
        n_tgt = int(b.shape(target_svd.Vt)[1])

        # Number of groups = target heads
        n_groups = gqa_config.target_heads

        # Source heads per group
        src_heads_per_group = gqa_config.source_heads // n_groups
        if gqa_config.source_heads % n_groups != 0:
            logger.warning(
                "GQA grouping: source_heads=%d not divisible by n_groups=%d",
                gqa_config.source_heads,
                n_groups,
            )

        # Dimensions per group
        src_head_dim = gqa_config.source_head_dim
        tgt_head_dim = gqa_config.target_head_dim

        # Output stride for source (rows per group)
        src_out_stride = src_heads_per_group * src_head_dim
        # Output stride for target
        tgt_out_stride = tgt_head_dim

        # Accumulate grouped results
        group_results: list["Array"] = []
        total_error = 0.0

        for g in range(n_groups):
            # Extract source group: rows [g * stride : (g+1) * stride]
            src_start = g * src_out_stride
            src_end = min(src_start + src_out_stride, m_src)

            # Slice the delta for this group
            delta_g = delta[src_start:src_end, :]
            b.eval(delta_g)

            # Get shared rank for this group's transfer
            k_src = source_svd.effective_rank
            k_tgt = target_svd.effective_rank
            k_shared = min(k_src, k_tgt)
            idx_shared = b.arange(0, k_shared)

            # Source SVD components (full, will project into)
            U_src = b.take(source_svd.U, idx_shared, axis=1)
            Vt_src = b.take(source_svd.Vt, idx_shared, axis=0)
            V_src = b.transpose(Vt_src)
            b.eval(U_src, V_src)

            # For GQA, we need to handle the row dimension mismatch
            # U_src is [m_src, k], but we're projecting [src_out_stride, n_src]
            # We take the corresponding rows of U_src for this group
            U_src_g = U_src[src_start:src_end, :]
            b.eval(U_src_g)

            # Target SVD components
            U_tgt = b.take(target_svd.U, idx_shared, axis=1)
            Vt_tgt = b.take(target_svd.Vt, idx_shared, axis=0)
            b.eval(U_tgt, Vt_tgt)

            # For target, extract corresponding rows
            tgt_start = g * tgt_out_stride
            tgt_end = min(tgt_start + tgt_out_stride, m_tgt)
            U_tgt_g = U_tgt[tgt_start:tgt_end, :]
            b.eval(U_tgt_g)

            # Step 1: PROJECT into source k-space
            # δ_k = U_s_g^T @ delta_g @ V_s
            delta_k = b.matmul(b.transpose(U_src_g), delta_g)
            b.eval(delta_k)
            delta_k = b.matmul(delta_k, V_src)
            b.eval(delta_k)

            # Step 2: ROTATE (within k-space, using group-local Procrustes)
            # For GQA, we skip rotation within groups since dimensions differ
            rotated_k = delta_k

            # Step 3: LIFT to target group's space
            # ΔW'_g = U_t_g @ rotated_k @ V_t^T
            transferred_g = b.matmul(U_tgt_g, rotated_k)
            b.eval(transferred_g)
            transferred_g = b.matmul(transferred_g, Vt_tgt)
            b.eval(transferred_g)

            group_results.append(transferred_g)

            # Compute error for this group
            delta_g_norm_sq = b.sum(delta_g * delta_g)
            b.eval(delta_g_norm_sq)
            delta_g_norm = float(b.to_scalar(b.sqrt(delta_g_norm_sq)))

            trans_g_norm_sq = b.sum(transferred_g * transferred_g)
            b.eval(trans_g_norm_sq)
            trans_g_norm = float(b.to_scalar(b.sqrt(trans_g_norm_sq)))

            if delta_g_norm > 0:
                group_error = abs(trans_g_norm - delta_g_norm) / delta_g_norm
                total_error += group_error

        # Concatenate all group results
        if group_results:
            transferred = b.concatenate(group_results, axis=0)
            b.eval(transferred)
            mean_error = total_error / n_groups
        else:
            # Fallback: zero delta
            transferred = b.zeros((m_tgt, n_tgt), dtype=precision_dtype(b, reference=lora_delta))
            mean_error = 1.0

        logger.debug(
            "GQA transfer %s: %d groups, mean_error=%.4f",
            layer_key,
            n_groups,
            mean_error,
        )

        return transferred, mean_error, n_groups

    def transfer_layer(
        self,
        lora_delta: "Array",
        source_svd: SVDComponents,
        target_svd: SVDComponents,
        layer_key: str = "",
        gqa_config: GQAConfig | None = None,
    ) -> tuple["Array", LayerTransferResult]:
        """Transfer a single layer's LoRA delta to target model space.

        The algorithm (Project-Rotate-Lift):
        1. Project: δ_k = U_s^T @ ΔW @ V_s  (into source's k-dim subspace)
        2. Rotate: δ_k' = R_left^T @ δ_k @ R_right  (align to target coords)
        3. Lift: ΔW' = U_t @ δ_k' @ V_t^T  (into target's full space)

        The Procrustes rotations R_left, R_right align the singular vector
        bases between source and target models. Without this rotation,
        the transfer assumes principal directions are identically ordered
        between models—which is false for different architectures.

        GQA Handling:
            When gqa_config is provided and source/target have different head
            counts (e.g., Qwen dense -> Llama GQA), we use grouped projection:
            - Split the source projection into H_tgt groups
            - Project each group independently
            - Concatenate results

            This handles dimension mismatches caused by Grouped Query Attention.

        Args:
            lora_delta: LoRA weight update [out_features, in_features]
            source_svd: SVD of source base weight
            target_svd: SVD of target base weight
            layer_key: Identifier for logging
            gqa_config: Optional GQA configuration for cross-attention transfer

        Returns:
            Tuple (transferred_delta, LayerTransferResult)
        """
        b = self._backend

        # Promote to high precision
        delta = b.astype(lora_delta, precision_dtype(b, reference=lora_delta))
        b.eval(delta)

        src_shape = b.shape(delta)
        m_src, n_src = int(src_shape[0]), int(src_shape[1])

        # Target dimensions from SVD
        m_tgt = int(b.shape(target_svd.U)[0])
        n_tgt = int(b.shape(target_svd.Vt)[1])

        # Check if GQA grouping is needed
        if gqa_config is not None and gqa_config.is_gqa_transfer:
            # Use GQA-aware grouped projection
            transferred, projection_error, n_groups = self._transfer_gqa_grouped(
                lora_delta=lora_delta,
                source_svd=source_svd,
                target_svd=target_svd,
                gqa_config=gqa_config,
                layer_key=layer_key,
            )

            # Compute Grassmann distance
            try:
                grass_result = compute_grassmann_distance(
                    source_svd.Vt, target_svd.Vt, backend=b
                )
                grassmann_dist = grass_result.geodesic_distance
            except Exception as e:
                logger.warning("Grassmann distance failed for %s: %s", layer_key, e)
                grassmann_dist = float("inf")

            result = LayerTransferResult(
                layer_key=layer_key,
                source_rank=source_svd.effective_rank,
                target_rank=target_svd.effective_rank,
                grassmann_distance=grassmann_dist,
                projection_error=projection_error,
                was_truncated=source_svd.effective_rank != target_svd.effective_rank,
                gqa_groups_used=n_groups,
                warning=f"GQA transfer: {gqa_config.source_heads}->{gqa_config.target_heads} heads",
            )

            logger.debug(
                "GQA transferred %s: rank %d->%d, groups=%d, error=%.4f",
                layer_key,
                source_svd.effective_rank,
                target_svd.effective_rank,
                n_groups,
                projection_error,
            )

            return transferred, result

        # Standard transfer (no GQA)
        warning = None
        was_truncated = False

        # Get shared rank (minimum of source and target effective ranks)
        k_src = source_svd.effective_rank
        k_tgt = target_svd.effective_rank
        k_shared = min(k_src, k_tgt)

        # Compute delta norm for error metrics
        delta_norm_sq = b.sum(delta * delta)
        b.eval(delta_norm_sq)
        delta_norm = float(b.to_scalar(b.sqrt(delta_norm_sq)))

        # CRITICAL FIX: For same-dimension transfer, use DIRECT ROTATION
        # not project-rotate-lift (which loses info outside k-dim subspace)
        #
        # The old algorithm: δ_k = U_s^T @ delta @ V_s → loses components
        # orthogonal to the weight's principal subspace (even for same-model!)
        #
        # New algorithm for same dimensions:
        #   transferred = R_left @ delta @ R_right^T
        # where R_left, R_right are full-space rotation matrices constructed
        # from the SVD bases.

        if m_src == m_tgt and n_src == n_tgt:
            # SAME DIMENSIONS: Use direct rotation (preserves all energy)
            #
            # The rotation is: transferred = R_full_left @ delta @ R_full_right^T
            # where R_full maps source basis to target basis.
            #
            # R_full = U_tgt @ R_k @ U_src^T + (I - U_tgt @ U_tgt^T)
            # The second term preserves components orthogonal to both subspaces.
            #
            # For efficiency, we compute:
            #   transferred = U_tgt @ R_k @ U_src^T @ delta @ V_src @ R_k^T @ V_tgt^T
            #                 + delta - U_src @ U_src^T @ delta @ V_src @ V_src^T
            #
            # But simpler: express delta in source basis, rotate coefficients, reconstruct
            # For full-rank case, this is equivalent to direct rotation.

            # Truncate SVD components to shared rank
            idx_shared = b.arange(0, k_shared)
            U_src = b.take(source_svd.U, idx_shared, axis=1)  # [m, k]
            Vt_src = b.take(source_svd.Vt, idx_shared, axis=0)  # [k, n]
            V_src = b.transpose(Vt_src)  # [n, k]
            U_tgt = b.take(target_svd.U, idx_shared, axis=1)  # [m, k]
            Vt_tgt = b.take(target_svd.Vt, idx_shared, axis=0)  # [k, n]
            V_tgt = b.transpose(Vt_tgt)  # [n, k]
            b.eval(U_src, V_src, U_tgt, V_tgt, Vt_tgt)

            # Compute Procrustes rotations in k-space
            result_left = self._matrix_utils.procrustes_rotation(U_src, U_tgt)
            R_k_left = result_left.rotation  # [k, k]
            b.eval(R_k_left)

            result_right = self._matrix_utils.procrustes_rotation(V_src, V_tgt)
            R_k_right = result_right.rotation  # [k, k]
            b.eval(R_k_right)

            # Construct full-space rotation matrices:
            # R_full_left = U_tgt @ R_k @ U_src^T + (I - U_tgt @ U_tgt^T)(I - U_src @ U_src^T)
            # For simplicity, we use the formula:
            #   R_full = U_tgt @ R_k @ U_src^T + orthogonal_complement
            # where orthogonal_complement preserves components outside both subspaces.
            #
            # The key insight: if the delta has components in the shared subspace,
            # they get rotated. Components orthogonal to BOTH source and target
            # subspaces are preserved. Components in one but not the other are
            # handled by the projection terms.

            # Project delta's in-subspace component
            delta_in_subspace = b.matmul(U_src, b.matmul(b.transpose(U_src), delta))
            delta_in_subspace = b.matmul(delta_in_subspace, b.matmul(V_src, b.transpose(V_src)))
            b.eval(delta_in_subspace)

            # Compute orthogonal component (what project-rotate-lift loses!)
            delta_orthogonal = delta - delta_in_subspace
            b.eval(delta_orthogonal)

            # Rotate the in-subspace component using project-rotate-lift
            delta_k = b.matmul(b.transpose(U_src), delta_in_subspace)
            delta_k = b.matmul(delta_k, V_src)  # [k, k]
            b.eval(delta_k)

            rotated_k = b.matmul(b.transpose(R_k_left), delta_k)
            rotated_k = b.matmul(rotated_k, R_k_right)
            b.eval(rotated_k)

            transferred_subspace = b.matmul(U_tgt, rotated_k)
            transferred_subspace = b.matmul(transferred_subspace, Vt_tgt)
            b.eval(transferred_subspace)

            # CRITICAL: Add back the orthogonal component!
            # This is what the old algorithm was missing.
            transferred = transferred_subspace + delta_orthogonal
            b.eval(transferred)

            # Compute true reconstruction error (should be near-zero for same-model)
            diff = delta - transferred
            diff_norm_sq = b.sum(diff * diff)
            b.eval(diff_norm_sq)
            diff_norm = float(b.to_scalar(b.sqrt(diff_norm_sq)))
            projection_error = diff_norm / delta_norm if delta_norm > 0 else 0.0

        else:
            # DIFFERENT DIMENSIONS: Must use project-rotate-lift
            # (no way to preserve orthogonal components when dimensions differ)
            idx_shared = b.arange(0, k_shared)
            U_src = b.take(source_svd.U, idx_shared, axis=1)  # [m_src, k]
            Vt_src = b.take(source_svd.Vt, idx_shared, axis=0)  # [k, n_src]
            V_src = b.transpose(Vt_src)
            U_tgt = b.take(target_svd.U, idx_shared, axis=1)  # [m_tgt, k]
            Vt_tgt = b.take(target_svd.Vt, idx_shared, axis=0)  # [k, n_tgt]
            b.eval(U_src, V_src, U_tgt, Vt_tgt)

            # Project-Lift (no rotation needed for cross-dimension)
            delta_k = b.matmul(b.transpose(U_src), delta)
            delta_k = b.matmul(delta_k, V_src)
            b.eval(delta_k)

            transferred = b.matmul(U_tgt, delta_k)
            transferred = b.matmul(transferred, Vt_tgt)
            b.eval(transferred)

            # Measure energy preservation
            transferred_norm_sq = b.sum(transferred * transferred)
            b.eval(transferred_norm_sq)
            transferred_norm = float(b.to_scalar(b.sqrt(transferred_norm_sq)))
            projection_error = (
                abs(transferred_norm - delta_norm) / delta_norm
                if delta_norm > 0
                else 0.0
            )

        # Check if ranks differ (indicates truncation)
        if source_svd.effective_rank != target_svd.effective_rank:
            was_truncated = True
            warning = (
                f"Rank mismatch: source={source_svd.effective_rank}, "
                f"target={target_svd.effective_rank}"
            )

        # Compute Grassmann distance between source and target right singular spaces
        # This measures how different the weight spaces are
        try:
            grass_result = compute_grassmann_distance(
                source_svd.Vt, target_svd.Vt, backend=b
            )
            grassmann_dist = grass_result.geodesic_distance
        except Exception as e:
            logger.warning("Grassmann distance failed for %s: %s", layer_key, e)
            grassmann_dist = float("inf")

        result = LayerTransferResult(
            layer_key=layer_key,
            source_rank=source_svd.effective_rank,
            target_rank=target_svd.effective_rank,
            grassmann_distance=grassmann_dist,
            projection_error=projection_error,
            was_truncated=was_truncated,
            warning=warning,
        )

        logger.debug(
            "Transferred %s: rank %d->%d, grass=%.4f, error=%.4f",
            layer_key,
            source_svd.effective_rank,
            target_svd.effective_rank,
            grassmann_dist,
            projection_error,
        )

        return transferred, result

    def transfer(
        self,
        source_adapter_weights: dict[str, "Array"],
        source_base_svd: dict[str, SVDComponents],
        target_base_svd: dict[str, SVDComponents],
        source_base_model: str = "unknown",
        target_base_model: str = "unknown",
        gqa_configs: dict[str, GQAConfig] | None = None,
    ) -> tuple[dict[str, "Array"], LoRATransferResult]:
        """Transfer a complete LoRA adapter from source to target model.

        Args:
            source_adapter_weights: Dict mapping layer keys to LoRA deltas
            source_base_svd: Dict mapping layer keys to source SVD components
            target_base_svd: Dict mapping layer keys to target SVD components
            source_base_model: Name of source base model (for logging)
            target_base_model: Name of target base model (for logging)
            gqa_configs: Optional dict mapping layer keys to GQA configs for
                cross-attention transfers (e.g., k_proj, v_proj layers)

        Returns:
            Tuple (transferred_weights, LoRATransferResult)
        """
        transferred_weights: dict[str, "Array"] = {}
        layer_results: list[LayerTransferResult] = []
        warnings: list[str] = []
        skipped = 0
        gqa_configs = gqa_configs or {}

        for key, delta in source_adapter_weights.items():
            # Check if we have SVD for this layer in both models
            if key not in source_base_svd:
                warnings.append(f"Missing source SVD for {key}")
                skipped += 1
                continue

            if key not in target_base_svd:
                warnings.append(f"Missing target SVD for {key}")
                skipped += 1
                continue

            # Get GQA config for this layer if available
            gqa_config = gqa_configs.get(key)

            try:
                transferred, result = self.transfer_layer(
                    lora_delta=delta,
                    source_svd=source_base_svd[key],
                    target_svd=target_base_svd[key],
                    layer_key=key,
                    gqa_config=gqa_config,
                )
                transferred_weights[key] = transferred
                layer_results.append(result)

                if result.warning:
                    warnings.append(result.warning)

            except Exception as e:
                warnings.append(f"Failed to transfer {key}: {e}")
                skipped += 1
                logger.warning("Layer transfer failed for %s: %s", key, e)

        # Compute aggregate metrics
        if layer_results:
            mean_grass = sum(r.grassmann_distance for r in layer_results) / len(
                layer_results
            )
            mean_error = sum(r.projection_error for r in layer_results) / len(
                layer_results
            )
            max_error = max(r.projection_error for r in layer_results)
        else:
            mean_grass = float("inf")
            mean_error = float("inf")
            max_error = float("inf")

        result = LoRATransferResult(
            source_base_model=source_base_model,
            target_base_model=target_base_model,
            layers_transferred=len(transferred_weights),
            layers_skipped=skipped,
            mean_grassmann_distance=mean_grass,
            mean_projection_error=mean_error,
            max_projection_error=max_error,
            layer_results=layer_results,
            warnings=warnings,
        )

        logger.info(
            "LoRA transfer complete: %d layers transferred, %d skipped, "
            "mean_error=%.4f, mean_grassmann=%.4f",
            result.layers_transferred,
            result.layers_skipped,
            mean_error,
            mean_grass,
        )

        return transferred_weights, result


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_lora_delta(lora_A: "Array", lora_B: "Array", backend: "Backend") -> "Array":
    """Compute the LoRA weight delta from A and B matrices.

    LoRA parameterizes the weight update as: ΔW = B @ A
    where A ∈ R^{r×in} and B ∈ R^{out×r}.

    Args:
        lora_A: LoRA A matrix [rank, in_features]
        lora_B: LoRA B matrix [out_features, rank]
        backend: Compute backend

    Returns:
        Weight delta [out_features, in_features]
    """
    delta = backend.matmul(lora_B, lora_A)
    backend.eval(delta)
    return delta


def decompose_to_lora(
    delta: "Array",
    rank: int,
    backend: "Backend",
) -> tuple["Array", "Array"]:
    """Decompose a weight delta back into LoRA A and B matrices.

    Uses truncated SVD: ΔW ≈ U_k @ S_k @ V_k^T = (U_k @ sqrt(S_k)) @ (sqrt(S_k) @ V_k^T)
    where B = U_k @ sqrt(S_k) and A = sqrt(S_k) @ V_k^T.

    Args:
        delta: Weight delta [out_features, in_features]
        rank: Target LoRA rank
        backend: Compute backend

    Returns:
        Tuple (lora_A, lora_B) where:
            lora_A: [rank, in_features]
            lora_B: [out_features, rank]
    """
    b = backend

    # Promote to high precision
    delta_hp = b.astype(delta, precision_dtype(b, reference=delta))
    b.eval(delta_hp)

    # SVD
    U, S, Vt = b.svd(delta_hp, compute_uv=True)
    b.eval(U, S, Vt)

    # Truncate to rank
    actual_rank = min(rank, int(b.shape(S)[0]))
    idx = b.arange(0, actual_rank)

    U_k = b.take(U, idx, axis=1)  # [out, rank]
    S_k = b.take(S, idx, axis=0)  # [rank]
    Vt_k = b.take(Vt, idx, axis=0)  # [rank, in]
    b.eval(U_k, S_k, Vt_k)

    # Compute sqrt(S) for balanced split
    S_sqrt = b.sqrt(S_k)
    b.eval(S_sqrt)

    # B = U_k @ diag(sqrt(S_k)) -> [out, rank]
    # A = diag(sqrt(S_k)) @ Vt_k -> [rank, in]
    lora_B = U_k * b.reshape(S_sqrt, (1, -1))  # Broadcast multiply
    lora_A = b.reshape(S_sqrt, (-1, 1)) * Vt_k
    b.eval(lora_A, lora_B)

    return lora_A, lora_B


def create_gqa_configs_from_model_configs(
    source_config: dict,
    target_config: dict,
    n_layers: int,
) -> dict[str, GQAConfig]:
    """Create GQA configs for all layers needing grouped transfer.

    Detects GQA mismatch between source and target models and creates
    appropriate GQAConfig for K/V projection layers.

    Args:
        source_config: Source model config with num_attention_heads,
            num_key_value_heads (optional), hidden_size
        target_config: Target model config with same keys
        n_layers: Number of transformer layers

    Returns:
        Dict mapping layer keys to GQAConfig for layers needing GQA transfer.
        Empty dict if no GQA mismatch.

    Example:
        >>> source = {"num_attention_heads": 32, "hidden_size": 4096}  # Dense
        >>> target = {"num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}  # GQA
        >>> configs = create_gqa_configs_from_model_configs(source, target, 32)
        >>> len(configs)  # k_proj and v_proj for each layer
        64
    """
    # Extract attention head counts
    src_q_heads = source_config.get("num_attention_heads", 32)
    src_kv_heads = source_config.get("num_key_value_heads", src_q_heads)

    tgt_q_heads = target_config.get("num_attention_heads", 32)
    tgt_kv_heads = target_config.get("num_key_value_heads", tgt_q_heads)

    # Extract hidden dimensions
    src_hidden = source_config.get("hidden_size", 4096)
    tgt_hidden = target_config.get("hidden_size", 4096)

    # Compute head dimensions
    src_head_dim = src_hidden // src_q_heads
    tgt_head_dim = tgt_hidden // tgt_q_heads

    # Check if GQA transfer is needed
    if src_kv_heads == tgt_kv_heads and src_head_dim == tgt_head_dim:
        return {}  # No GQA mismatch

    gqa_configs: dict[str, GQAConfig] = {}

    # Create configs for K and V projections (these have different head counts)
    for layer_idx in range(n_layers):
        for proj in ["k_proj", "v_proj"]:
            key = f"layer.{layer_idx}.self_attn.{proj}"
            gqa_configs[key] = GQAConfig(
                source_heads=src_kv_heads,
                target_heads=tgt_kv_heads,
                source_head_dim=src_head_dim,
                target_head_dim=tgt_head_dim,
            )

    return gqa_configs


def detect_gqa_from_weights(
    source_weight_shape: tuple[int, int],
    target_weight_shape: tuple[int, int],
    source_q_heads: int,
    target_q_heads: int,
) -> GQAConfig | None:
    """Detect GQA configuration from weight shapes.

    Infers head dimensions and counts from weight matrix shapes.

    Args:
        source_weight_shape: (out_features, in_features) of source K/V proj
        target_weight_shape: (out_features, in_features) of target K/V proj
        source_q_heads: Number of Q heads in source model
        target_q_heads: Number of Q heads in target model

    Returns:
        GQAConfig if GQA transfer needed, None otherwise.
    """
    src_out, src_in = source_weight_shape
    tgt_out, tgt_in = target_weight_shape

    # Infer head dimensions from weight shapes
    # K/V projection: [kv_heads * head_dim, hidden_dim]
    # We need to figure out head_dim and kv_heads

    # Common head dims: 64, 128, 256
    # Try to find consistent head_dim
    for head_dim in [64, 128, 256, 96, 80]:
        if src_out % head_dim == 0 and tgt_out % head_dim == 0:
            src_kv_heads = src_out // head_dim
            tgt_kv_heads = tgt_out // head_dim

            if src_kv_heads != tgt_kv_heads:
                return GQAConfig(
                    source_heads=src_kv_heads,
                    target_heads=tgt_kv_heads,
                    source_head_dim=head_dim,
                    target_head_dim=head_dim,
                )

    # Could not detect GQA mismatch
    return None


__all__ = [
    "UniversalLoRAProjector",
    "LoRATransferResult",
    "LayerTransferResult",
    "SVDComponents",
    "GQAConfig",
    "compute_lora_delta",
    "decompose_to_lora",
    "create_gqa_configs_from_model_configs",
    "detect_gqa_from_weights",
]
