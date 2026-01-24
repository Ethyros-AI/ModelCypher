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

"""Active subspace blending for knowledge transfer.

Uses activation covariance to split directions into utilized (active) and
available (null) subspaces, then applies different blend ratios per subspace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    compute_variance_null_space,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveSubspaceBlendResult:
    """Result of active-subspace-aware blending.

    Attributes:
        blended_weight: Weight matrix after active subspace blending.
        effective_blend_ratio: Variance-weighted average of blend ratios.
        active_blend: Blend ratio applied in utilized (active) directions.
        null_blend: Blend ratio applied in available (null) directions.
        active_rank: Number of utilized (active) directions.
        null_rank: Number of available (null) directions.
        variance_captured: Fraction of variance in the active subspace.
    """

    blended_weight: "Array"
    effective_blend_ratio: float
    active_blend: float
    null_blend: float
    active_rank: int
    null_rank: int
    variance_captured: float


def compute_active_subspace_blend(
    source_weight: "Array",
    target_weight: "Array",
    input_activations: "Array",
    base_blend: float = 0.1,
    active_boost: float = 2.0,
    null_dampen: float = 0.5,
    backend: "Backend | None" = None,
) -> ActiveSubspaceBlendResult:
    """Blend weights in the activation-defined active subspace.

    Uses activation covariance to define active vs. available directions,
    blends each subspace with its own ratio, and reconstructs the weight.

    Args:
        source_weight: Source weight matrix [out_dim, in_dim].
        target_weight: Target weight matrix [out_dim, in_dim].
        input_activations: Input activations that define the active subspace
            [n_samples, in_dim]. These are typically target activations.
        base_blend: Base blend ratio (0.0 = all target, 1.0 = all source).
            Default 0.1 = 10% source knowledge.
        active_boost: Multiplier for blend ratio in active (utilized) directions.
            Default 2.0 means active directions get 20% source (vs 10% base).
        null_dampen: Multiplier for blend ratio in null (available) directions.
            Default 0.5 means null directions get 5% source (vs 10% base).
        backend: Compute backend.

    Returns:
        ActiveSubspaceBlendResult with blended weight and diagnostics.

    Algorithm:
        1. Compute variance null space from input_activations
           - utilized_basis [in_dim, r]: directions with high variance
           - available_basis [in_dim, d-r]: directions with low variance
        2. Project both weights into these bases
           - W_active = W @ U_utilized @ U_utilized.T
           - W_null = W @ U_available @ U_available.T
        3. Blend with different ratios:
           - W_blend_active = (1 - α_a) * W_tgt_active + α_a * W_src_active
           - W_blend_null = (1 - α_n) * W_tgt_null + α_n * W_src_null
        4. Reconstruct: W_blend = W_blend_active + W_blend_null
    """
    b = backend or get_default_backend()

    source_weight = b.array(source_weight)
    target_weight = b.array(target_weight)
    input_activations = b.array(input_activations)

    # Use high precision for computation
    compute_dtype = precision_dtype(b, reference=target_weight)
    source_weight = b.astype(source_weight, compute_dtype)
    target_weight = b.astype(target_weight, compute_dtype)
    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(source_weight, target_weight, input_activations)

    out_dim = int(target_weight.shape[0])
    in_dim = int(target_weight.shape[1])
    n_samples = int(input_activations.shape[0])

    logger.info(
        "ACTIVE SUBSPACE BLEND: weight=[%d, %d], n_samples=%d, base_blend=%.3f",
        out_dim, in_dim, n_samples, base_blend,
    )

    # Step 1: Compute variance null space from activations
    # This gives us the ACTIVATION-defined basis (not weight SVD)
    variance_result = compute_variance_null_space(
        activations=input_activations,
        backend=b,
    )

    U_utilized = variance_result.utilized_basis  # [in_dim, active_rank]
    U_available = variance_result.available_basis  # [in_dim, null_rank]
    active_rank = variance_result.utilized_rank
    null_rank = variance_result.available_rank
    eigenvalues = variance_result.eigenvalues

    b.eval(U_utilized, U_available, eigenvalues)

    # Compute variance captured in active subspace
    if active_rank > 0:
        total_var_arr = b.sum(eigenvalues)
        active_var_arr = b.sum(eigenvalues[:active_rank])
        b.eval(total_var_arr, active_var_arr)
        total_var = float(b.to_scalar(total_var_arr))
        active_var = float(b.to_scalar(active_var_arr))
        variance_captured = active_var / total_var if total_var > 0 else 0.0
    else:
        variance_captured = 0.0

    logger.info(
        "ACTIVE SUBSPACE: active_rank=%d, null_rank=%d, variance_captured=%.4f",
        active_rank, null_rank, variance_captured,
    )

    # Handle edge cases
    if active_rank == 0:
        logger.warning("ACTIVE SUBSPACE BLEND: No active directions, returning target")
        return ActiveSubspaceBlendResult(
            blended_weight=target_weight,
            effective_blend_ratio=0.0,
            active_blend=0.0,
            null_blend=0.0,
            active_rank=0,
            null_rank=null_rank,
            variance_captured=0.0,
        )

    if null_rank == 0:
        # All directions are active - use uniform blend
        logger.info("ACTIVE SUBSPACE BLEND: All directions active, using uniform blend")
        alpha = min(1.0, base_blend * active_boost)
        blended = (1.0 - alpha) * target_weight + alpha * source_weight
        b.eval(blended)
        return ActiveSubspaceBlendResult(
            blended_weight=blended,
            effective_blend_ratio=alpha,
            active_blend=alpha,
            null_blend=0.0,
            active_rank=active_rank,
            null_rank=0,
            variance_captured=1.0,
        )

    # Step 2: Project weights into active and null subspaces
    # Projector for active: P_active = U_utilized @ U_utilized.T [in_dim, in_dim]
    # Projector for null: P_null = U_available @ U_available.T [in_dim, in_dim]
    #
    # For weight W [out_dim, in_dim], the projection along INPUT dimension is:
    # W_active = W @ P_active = W @ U_utilized @ U_utilized.T
    # W_null = W @ P_null = W @ U_available @ U_available.T

    # Source weight projections
    src_Uu = b.matmul(source_weight, U_utilized)  # [out, active_rank]
    src_active = b.matmul(src_Uu, b.transpose(U_utilized))  # [out, in]
    b.eval(src_active)

    src_Ua = b.matmul(source_weight, U_available)  # [out, null_rank]
    src_null = b.matmul(src_Ua, b.transpose(U_available))  # [out, in]
    b.eval(src_null)

    # Target weight projections
    tgt_Uu = b.matmul(target_weight, U_utilized)  # [out, active_rank]
    tgt_active = b.matmul(tgt_Uu, b.transpose(U_utilized))  # [out, in]
    b.eval(tgt_active)

    tgt_Ua = b.matmul(target_weight, U_available)  # [out, null_rank]
    tgt_null = b.matmul(tgt_Ua, b.transpose(U_available))  # [out, in]
    b.eval(tgt_null)

    # Step 3: Blend with different ratios
    alpha_active = min(1.0, base_blend * active_boost)
    alpha_null = max(0.0, base_blend * null_dampen)

    # Blend active component (higher ratio - models agree here)
    blend_active = (1.0 - alpha_active) * tgt_active + alpha_active * src_active
    b.eval(blend_active)

    # Blend null component (lower ratio - preserve target's expansion factors)
    blend_null = (1.0 - alpha_null) * tgt_null + alpha_null * src_null
    b.eval(blend_null)

    # Step 4: Reconstruct blended weight
    # Since P_active + P_null = I (orthogonal complement), we have:
    # W_blend = W_blend_active + W_blend_null
    blended_weight = blend_active + blend_null
    b.eval(blended_weight)

    # Compute effective blend ratio (variance-weighted average)
    # The active subspace captures variance_captured of total variance
    # So effective = variance_captured * alpha_active + (1 - variance_captured) * alpha_null
    effective_blend = variance_captured * alpha_active + (1.0 - variance_captured) * alpha_null

    logger.info(
        "ACTIVE SUBSPACE BLEND: active_blend=%.3f, null_blend=%.3f, effective=%.3f",
        alpha_active, alpha_null, effective_blend,
    )

    return ActiveSubspaceBlendResult(
        blended_weight=blended_weight,
        effective_blend_ratio=effective_blend,
        active_blend=alpha_active,
        null_blend=alpha_null,
        active_rank=active_rank,
        null_rank=null_rank,
        variance_captured=variance_captured,
    )


def compute_adaptive_active_blend(
    source_weight: "Array",
    target_weight: "Array",
    input_activations: "Array",
    base_blend: float = 0.1,
    backend: "Backend | None" = None,
) -> ActiveSubspaceBlendResult:
    """Adaptive active subspace blend based on activation concentration.

    Adjusts boost/dampen ratios based on how concentrated the activation
    variance is. High concentration (bottleneck) = more conservative.
    Low concentration = can be more aggressive.

    Args:
        source_weight: Source weight matrix [out_dim, in_dim].
        target_weight: Target weight matrix [out_dim, in_dim].
        input_activations: Input activations [n_samples, in_dim].
        base_blend: Base blend ratio.
        backend: Compute backend.

    Returns:
        ActiveSubspaceBlendResult with adaptively blended weight.
    """
    b = backend or get_default_backend()

    input_activations = b.array(input_activations)
    compute_dtype = precision_dtype(b, reference=target_weight)
    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(input_activations)

    # First compute variance null space to get concentration
    variance_result = compute_variance_null_space(
        activations=input_activations,
        backend=b,
    )

    # Compute concentration from eigenvalues
    eigenvalues = variance_result.eigenvalues
    b.eval(eigenvalues)

    n_eig = int(eigenvalues.shape[0])
    if n_eig == 0:
        # Fallback to basic blend
        return compute_active_subspace_blend(
            source_weight, target_weight, input_activations,
            base_blend, active_boost=2.0, null_dampen=0.5, backend=b
        )

    # Variance concentration = top eigenvalue / total
    total_var_arr = b.sum(eigenvalues)
    top_var_arr = eigenvalues[0]
    b.eval(total_var_arr, top_var_arr)
    total_var = float(b.to_scalar(total_var_arr))
    top_var = float(b.to_scalar(top_var_arr))

    eps = float(machine_epsilon(b, eigenvalues))
    if total_var < eps:
        # Fallback
        return compute_active_subspace_blend(
            source_weight, target_weight, input_activations,
            base_blend, active_boost=2.0, null_dampen=0.5, backend=b
        )

    concentration = top_var / total_var

    logger.info(
        "ADAPTIVE ACTIVE BLEND: variance_concentration=%.3f",
        concentration,
    )

    # Adaptive boost/dampen based on concentration
    # High concentration (bottleneck): be conservative
    # Low concentration: can be more aggressive
    if concentration > 0.9:
        # Very strong bottleneck - minimal differential
        active_boost = 1.2
        null_dampen = 0.9
    elif concentration > 0.7:
        # Strong bottleneck - conservative
        active_boost = 1.5
        null_dampen = 0.7
    elif concentration > 0.5:
        # Moderate - standard
        active_boost = 2.0
        null_dampen = 0.5
    else:
        # Weak concentration - can be more aggressive
        active_boost = 2.5
        null_dampen = 0.3

    return compute_active_subspace_blend(
        source_weight, target_weight, input_activations,
        base_blend, active_boost=active_boost, null_dampen=null_dampen,
        backend=b
    )
