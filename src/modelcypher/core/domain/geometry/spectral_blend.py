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

"""Spectral-aware blending for knowledge transfer.

KEY INSIGHT from compression experiments:
    The transformation at bottleneck layers is NOT isotropic.

    At LFM2-350M layer 7:
    - Input: 99.85% variance in σ₁ (nearly 1D)
    - Output: Secondary directions EXPAND by 15-20x

    This means:
    - Dominant direction (σ₁): Both models encode similar primary content
    - Secondary directions (σ₂...): Model-specific expansion factors

    Uniform blending (10% everywhere) doesn't respect this geometry.

SPECTRAL-AWARE BLENDING:
    Instead of: W_merged = 0.9 * W_target + 0.1 * W_source

    We do:
    1. SVD decompose both weights: W = U @ diag(S) @ Vt
    2. For each singular direction i, compute blend ratio α_i
    3. Blend singular values: S_merged[i] = (1 - α_i) * S_target[i] + α_i * S_source[i]
    4. Use target's U, Vt (preserve target's coordinate system)

    The blend ratios α_i are computed from variance concentration:
    - High variance direction: α = base_blend * boost  (transfer more)
    - Low variance direction: α = base_blend * dampen  (transfer less)

    This respects the model's internal geometry while transferring knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
    precision_dtype,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralBlendResult:
    """Result of spectral-aware blending.

    Attributes:
        blended_weight: Weight matrix after spectral blending.
        effective_blend_ratio: Weighted average of per-direction blend ratios.
        dominant_blend: Blend ratio for the dominant singular direction.
        secondary_blend: Mean blend ratio for non-dominant directions.
        variance_concentration: Fraction of variance in top-1 singular value.
        spectral_condition: Ratio of largest to smallest singular value.
    """
    blended_weight: "Array"
    effective_blend_ratio: float
    dominant_blend: float
    secondary_blend: float
    variance_concentration: float
    spectral_condition: float


def compute_spectral_blend(
    source_weight: "Array",
    target_weight: "Array",
    base_blend: float = 0.1,
    dominant_boost: float = 3.0,
    secondary_dampen: float = 0.3,
    backend: "Backend | None" = None,
) -> SpectralBlendResult:
    """Compute spectral-aware blend of source and target weights.

    Instead of uniform blending, this respects the model's internal geometry:
    - Dominant directions: blend more aggressively (models likely agree)
    - Secondary directions: blend conservatively (model-specific expansion)

    The insight: at bottleneck layers, the transformation EXPANDS secondary
    directions by 15-20x. These expansion factors are model-specific and
    shouldn't be replaced - only gently perturbed.

    Args:
        source_weight: Source weight matrix [out_dim, in_dim].
        target_weight: Target weight matrix [out_dim, in_dim].
        base_blend: Base blend ratio (0.0 = all target, 1.0 = all source).
            Default 0.1 = 10% source knowledge.
        dominant_boost: Multiplier for dominant direction blend ratio.
            Default 3.0 means dominant direction gets 30% source (vs 10% base).
        secondary_dampen: Multiplier for secondary direction blend ratios.
            Default 0.3 means secondary directions get 3% source (vs 10% base).
        backend: Compute backend.

    Returns:
        SpectralBlendResult with blended weight and diagnostics.

    Math:
        1. SVD: W_target = U @ diag(S) @ Vt
        2. Project source into target's coordinate system:
           S_source_proj[i] = ||U[:, i].T @ W_source @ Vt[i, :].T||
        3. Compute per-direction blend:
           α_i = base_blend * (dominant_boost if i==0 else secondary_dampen)
        4. Blend singular values:
           S_blend[i] = (1 - α_i) * S_target[i] + α_i * S_source_proj[i]
        5. Reconstruct: W_blend = U @ diag(S_blend) @ Vt
    """
    b = backend or get_default_backend()

    source_weight = b.array(source_weight)
    target_weight = b.array(target_weight)

    # Use high precision for SVD
    compute_dtype = precision_dtype(b, reference=target_weight)
    source_weight = b.astype(source_weight, compute_dtype)
    target_weight = b.astype(target_weight, compute_dtype)
    b.eval(source_weight, target_weight)

    out_dim = int(target_weight.shape[0])
    in_dim = int(target_weight.shape[1])

    # Step 1: SVD of target weight
    # Target defines the coordinate system we'll work in
    U_t, S_t, Vt_t = geodesic_svd(b, target_weight)
    b.eval(U_t, S_t, Vt_t)

    n_sv = int(S_t.shape[0])
    if n_sv == 0:
        logger.warning("SPECTRAL BLEND: No singular values, returning target unchanged")
        return SpectralBlendResult(
            blended_weight=target_weight,
            effective_blend_ratio=0.0,
            dominant_blend=0.0,
            secondary_blend=0.0,
            variance_concentration=0.0,
            spectral_condition=float("inf"),
        )

    # Step 2: Compute variance concentration
    # High concentration = information compressed into few directions
    S_squared = S_t * S_t
    total_variance = b.sum(S_squared)
    b.eval(total_variance)
    total_var_val = float(b.to_scalar(total_variance))

    eps = float(machine_epsilon(b, S_t))
    if total_var_val < eps:
        logger.warning("SPECTRAL BLEND: Zero variance, returning target unchanged")
        return SpectralBlendResult(
            blended_weight=target_weight,
            effective_blend_ratio=0.0,
            dominant_blend=0.0,
            secondary_blend=0.0,
            variance_concentration=0.0,
            spectral_condition=float("inf"),
        )

    # Variance concentration = σ₁² / Σσᵢ²
    top_variance = float(b.to_scalar(S_squared[0]))
    variance_concentration = top_variance / total_var_val

    # Spectral condition = σ₁ / σₙ
    max_sv = float(b.to_scalar(S_t[0]))
    min_sv = float(b.to_scalar(S_t[n_sv - 1]))
    spectral_condition = max_sv / max(min_sv, eps)

    logger.info(
        "SPECTRAL BLEND: var_concentration=%.3f, spectral_condition=%.2e, n_sv=%d",
        variance_concentration, spectral_condition, n_sv
    )

    # Step 3: Project source weight into target's coordinate system
    # S_source_proj[i] = ||U_t[:, i].T @ W_source @ V_t[i, :].T||
    # This measures source's "weight" in each of target's singular directions
    #
    # More efficient: S_source_proj = diag(U_t.T @ W_source @ V_t.T)
    # where V_t.T = Vt_t.T (transpose of Vt_t)
    V_t = b.transpose(Vt_t)  # [in_dim, n_sv]
    Ut_W_source = b.matmul(b.transpose(U_t), source_weight)  # [n_sv, in_dim]
    S_source_proj_matrix = b.matmul(Ut_W_source, V_t)  # [n_sv, n_sv]
    b.eval(S_source_proj_matrix)

    # Extract diagonal - these are source's projections onto target's directions
    S_source_proj = b.diag(S_source_proj_matrix)  # [n_sv]
    b.eval(S_source_proj)

    # Step 4: Compute per-direction blend ratios
    # Dominant direction: blend more (models share primary content)
    # Secondary directions: blend less (model-specific expansion factors)
    dominant_alpha = min(1.0, base_blend * dominant_boost)
    secondary_alpha = max(0.0, base_blend * secondary_dampen)

    # Build blend ratio array: [dominant, secondary, secondary, ...]
    alphas = b.zeros((n_sv,), dtype=compute_dtype)
    # MLX doesn't have in-place updates, so we construct with where
    idx_range = b.arange(n_sv)
    is_dominant = idx_range < 1  # Only index 0 is dominant
    alphas = b.where(is_dominant, dominant_alpha, secondary_alpha)
    b.eval(alphas)

    # Step 5: Blend singular values
    # S_blend[i] = (1 - α_i) * S_target[i] + α_i * S_source_proj[i]
    one_minus_alpha = 1.0 - alphas
    S_blend = one_minus_alpha * S_t + alphas * S_source_proj
    b.eval(S_blend)

    # Step 6: Reconstruct blended weight
    # W_blend = U_t @ diag(S_blend) @ Vt_t
    # Efficient: U_scaled = U_t * S_blend (broadcast)
    U_scaled = U_t * b.reshape(S_blend, (1, -1))  # [out_dim, n_sv]
    b.eval(U_scaled)
    W_blend = b.matmul(U_scaled, Vt_t)  # [out_dim, in_dim]
    b.eval(W_blend)

    # Compute effective blend ratio (variance-weighted average)
    # This tells us what fraction of "energy" came from source
    alpha_weighted = alphas * (S_squared / total_var_val)
    effective_blend = float(b.to_scalar(b.sum(alpha_weighted)))

    logger.info(
        "SPECTRAL BLEND: dominant_blend=%.3f, secondary_blend=%.3f, effective=%.3f",
        dominant_alpha, secondary_alpha, effective_blend
    )

    return SpectralBlendResult(
        blended_weight=W_blend,
        effective_blend_ratio=effective_blend,
        dominant_blend=dominant_alpha,
        secondary_blend=secondary_alpha,
        variance_concentration=variance_concentration,
        spectral_condition=spectral_condition,
    )


def compute_adaptive_spectral_blend(
    source_weight: "Array",
    target_weight: "Array",
    input_activations: "Array",
    base_blend: float = 0.1,
    backend: "Backend | None" = None,
) -> SpectralBlendResult:
    """Adaptive spectral blend using input activation statistics.

    REVISED APPROACH: Instead of blending in the weight's SVD basis (which
    doesn't align with activation flow), we use a more conservative approach:

    1. Compute variance concentration of INPUT activations
    2. Use activation concentration to SCALE the base blend
       - High concentration (>0.9): activations are nearly 1D, model is sensitive
         → REDUCE blend to avoid disrupting the bottleneck
       - Moderate (0.5-0.9): activations have some structure
         → Use close to base blend
       - Low (<0.5): activations are spread out
         → Can blend more safely

    KEY INSIGHT FROM FAILURE: The previous approach (40% in dominant, 2% in
    secondary) produced degenerate output. The weight's SVD basis doesn't
    correspond to the activation's information flow. Conservative uniform
    blending (10%) worked better than aggressive spectral blending.

    NEW STRATEGY: Use activation concentration to adjust TOTAL blend amount,
    but keep the blend uniform across directions.

    Args:
        source_weight: Source weight matrix [out_dim, in_dim].
        target_weight: Target weight matrix [out_dim, in_dim].
        input_activations: Input activations [n_samples, in_dim].
        base_blend: Base blend ratio.
        backend: Compute backend.

    Returns:
        SpectralBlendResult with conservatively blended weight.
    """
    b = backend or get_default_backend()

    input_activations = b.array(input_activations)
    compute_dtype = precision_dtype(b, reference=target_weight)
    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(input_activations)

    # Compute variance concentration of input activations
    _, S_act, _ = geodesic_svd(b, input_activations)
    b.eval(S_act)

    n_sv_act = int(S_act.shape[0])
    if n_sv_act == 0:
        # Fallback to basic spectral blend with conservative ratios
        return compute_spectral_blend(
            source_weight, target_weight, base_blend,
            dominant_boost=1.5, secondary_dampen=0.8, backend=b
        )

    S_act_squared = S_act * S_act
    total_act_var = b.sum(S_act_squared)
    b.eval(total_act_var)
    total_act_var_val = float(b.to_scalar(total_act_var))

    eps = float(machine_epsilon(b, S_act))
    if total_act_var_val < eps:
        # Fallback to basic spectral blend with conservative ratios
        return compute_spectral_blend(
            source_weight, target_weight, base_blend,
            dominant_boost=1.5, secondary_dampen=0.8, backend=b
        )

    top_act_var = float(b.to_scalar(S_act_squared[0]))
    act_concentration = top_act_var / total_act_var_val

    logger.info(
        "ADAPTIVE SPECTRAL: input activation var_concentration=%.3f",
        act_concentration
    )

    # NEW APPROACH: Use conservative boost/dampen ratios
    # The key insight is that the weight's SVD doesn't align with activations.
    # So we use a mild spectral bias while staying mostly uniform.
    #
    # For HIGH concentration (bottleneck):
    #   - The model is sensitive at this layer
    #   - Use conservative blend (1.2x dominant, 0.9x secondary)
    #   - This is close to uniform but slightly prefers dominant
    #
    # For LOW concentration:
    #   - The model has more room for perturbation
    #   - Can use slightly more aggressive spectral blend
    if act_concentration > 0.9:
        # Very strong bottleneck - be VERY conservative
        # Dominant: 12% (1.2 * 10%), Secondary: 9% (0.9 * 10%)
        dominant_boost = 1.2
        secondary_dampen = 0.9
    elif act_concentration > 0.7:
        # Strong bottleneck - still conservative
        # Dominant: 15%, Secondary: 8%
        dominant_boost = 1.5
        secondary_dampen = 0.8
    elif act_concentration > 0.5:
        # Moderate concentration
        # Dominant: 18%, Secondary: 7%
        dominant_boost = 1.8
        secondary_dampen = 0.7
    else:
        # Weak concentration - can be more aggressive
        # Dominant: 20%, Secondary: 6%
        dominant_boost = 2.0
        secondary_dampen = 0.6

    return compute_spectral_blend(
        source_weight, target_weight, base_blend,
        dominant_boost=dominant_boost,
        secondary_dampen=secondary_dampen,
        backend=b
    )
