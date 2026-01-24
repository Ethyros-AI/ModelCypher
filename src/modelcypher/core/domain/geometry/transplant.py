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

"""Functional transplant for knowledge transfer.

Implements null-space projection of weight deltas with optional density
weighting based on k-NN statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    gpu_lstsq,
    machine_epsilon,
    orthogonalize_alignment,
    precision_dtype,
    svd_rank_threshold,
)
from modelcypher.core.domain.geometry.geodesic_null_space import (
    GeodesicNullSpaceFilter,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        TrajectoryResult,
        TrajectoryTangentResult,
    )

logger = logging.getLogger(__name__)


def _weight_frobenius_norm(
    weight: "Array",
    backend: "Backend",
) -> float:
    """Compute Frobenius norm for weight matrices.

    Uses Euclidean (not geodesic) norm because weight space is treated as
    flat/spectral rather than a curved manifold.

    Args:
        weight: Weight matrix [out_dim, in_dim]
        backend: Backend for tensor operations

    Returns:
        Frobenius norm (scalar)
    """
    b = backend
    shape = b.shape(weight)
    if len(shape) != 2:
        weight = b.reshape(weight, (1, -1))
    elif shape[0] < 1:
        return 0.0

    # Standard Frobenius norm: sqrt(sum(w_ij²))
    frob_norm = b.sqrt(b.sum(weight * weight))
    return float(b.to_scalar(frob_norm))


def _behavioral_norm(
    delta_W: "Array",
    input_activations: "Array",
    backend: "Backend",
) -> float:
    """Compute behavioral impact norm for a weight delta.

    The behavioral impact measures how much the layer's OUTPUT would change
    if we applied this weight delta. This is the correct metric for transplant
    because it measures actual behavioral change, not just weight magnitude.

    Mathematically: ||A @ ΔW.T||_F where A is [n, in_dim] and ΔW is [out, in].
    This computes the Frobenius norm of the output change across all samples.

    Why this is correct:
    - Frobenius norm ignores activation structure (misleading)
    - Behavioral norm measures actual output change (correct)
    - After null-space projection, behavioral norm on TARGET activations → 0
    - preserved_fraction = behavioral_after / behavioral_before

    Args:
        delta_W: Weight delta [out_dim, in_dim]
        input_activations: Activations that flow through this weight [n, in_dim]
        backend: Backend for tensor operations

    Returns:
        Behavioral impact norm (scalar)
    """
    b = backend

    # Compute output change: [n, in_dim] @ [in_dim, out_dim] -> [n, out_dim]
    output_change = b.matmul(input_activations, b.transpose(delta_W))

    # Frobenius norm of output change
    behavioral = b.sqrt(b.sum(output_change * output_change))
    return float(b.to_scalar(behavioral))


@dataclass(frozen=True)
class WeightSpaceTransplantResult:
    """Result of weight-space null-space transplant.

    Attributes:
        merged_weight: The transplanted weight [out_dim, in_dim].
        delta_norm: BEHAVIORAL norm of weight delta before projection.
            This is ||A @ ΔW.T||_F - the output change magnitude.
        projected_norm: BEHAVIORAL norm of delta after null-space projection.
        preserved_fraction: Ratio of projected_norm / delta_norm (1.0 = no loss).
            Measures behavioral transfer, not weight magnitude.
        transfer_strength: Mean density-derived transfer weight (0-1).
        null_rank: Approximate rank of the null-space projector.
    """

    merged_weight: "Array"
    delta_norm: float
    projected_norm: float
    preserved_fraction: float
    transfer_strength: float
    null_rank: int


@dataclass(frozen=True)
class NullSpaceProjector:
    """Precomputed null-space projector with density-derived transfer strength."""

    weighted_activations: "Array"
    gram_inv: "Array"
    null_rank: int
    transfer_strength: float
    projector: "Array | None" = None


@dataclass(frozen=True)
class TrajectoryTangentProjector:
    """Trajectory-tangent null-space projector.

    Projects deltas into null-space directions that are TANGENT to the
    model's natural trajectory flow - "building along the road."

    This is more targeted than variance-based null-space projection because:
    - Variance-based: projects into ALL low-variance directions
    - Trajectory-tangent: projects into directions ALIGNED with velocity flow

    The intuition: velocities show WHERE the model is heading. Null-space
    directions aligned with velocities are "shoulders of the road" - unused
    capacity that's still reachable by the model's natural information flow.
    """

    tangent_result: "TrajectoryTangentResult"
    null_rank: int
    tangent_rank: int
    velocity_alignment: float
    use_full_null: bool = False  # If True, use full null space instead of tangent


@dataclass(frozen=True)
class BehavioralReconstructionResult:
    """Result of behavioral weight reconstruction for cross-dimensional transfer.

    Instead of transforming weights directly (P @ W @ Q), we reconstruct the
    weight that produces the same input→output behavior in the target space.

    This preserves the FUNCTION of the layer, not the MATRIX values.

    Attributes:
        reconstructed_weight: Weight that reproduces source behavior in target coords.
        reconstruction_error: Mean absolute error of behavior matching.
        source_behavior_norm: Norm of source outputs (for diagnostics).
        target_behavior_norm: Norm of reconstructed outputs (for diagnostics).
        condition_number: Condition number of the lstsq solve.
    """

    reconstructed_weight: "Array"
    reconstruction_error: float
    source_behavior_norm: float
    target_behavior_norm: float
    condition_number: float


def reconstruct_weight_from_behavior(
    source_weight: "Array",
    input_activations_source: "Array",
    alignment_in: "Array",
    alignment_out: "Array",
    backend: "Backend | None" = None,
) -> BehavioralReconstructionResult:
    """Reconstruct weight to preserve input→output behavior across dimensions.

    Uses orthogonalized alignments (polar decomposition) for coordinate changes,
    solves least squares in target space, then applies scale correction.
    """
    b = backend or get_default_backend()

    source_weight = b.array(source_weight)
    input_activations_source = b.array(input_activations_source)
    alignment_in = b.array(alignment_in)
    alignment_out = b.array(alignment_out)

    # Use high precision for the reconstruction
    compute_dtype = precision_dtype(b, reference=source_weight)
    source_weight = b.astype(source_weight, compute_dtype)
    input_activations_source = b.astype(input_activations_source, compute_dtype)
    alignment_in = b.astype(alignment_in, compute_dtype)
    alignment_out = b.astype(alignment_out, compute_dtype)
    b.eval(source_weight, input_activations_source, alignment_in, alignment_out)

    # Get dimensions
    out_src = int(source_weight.shape[0])
    in_src = int(source_weight.shape[1])
    n_samples = int(input_activations_source.shape[0])
    in_tgt = int(alignment_in.shape[1])
    out_tgt = int(alignment_out.shape[1])

    logger.info(
        "BEHAVIORAL RECONSTRUCTION: [%d, %d] -> [%d, %d], n_samples=%d",
        out_src, in_src, out_tgt, in_tgt, n_samples
    )

    # Step 1: Orthogonalize alignment transforms via polar decomposition
    # This separates rotation (norm-preserving) from scaling
    U_in, scale_in = orthogonalize_alignment(alignment_in, b)
    U_out, scale_out = orthogonalize_alignment(alignment_out, b)
    b.eval(U_in, U_out)

    # Compute scale correction factor: output scaling / input scaling
    # This corrects for the non-orthogonal part of the alignment
    eps = float(division_epsilon(b, alignment_in))
    scale_correction = scale_out / max(scale_in, eps)

    logger.info(
        "ORTHOGONALIZATION: scale_in=%.4f, scale_out=%.4f, correction=%.4f",
        scale_in, scale_out, scale_correction
    )

    # Step 2: Compute source layer behavior
    # output_src = input_src @ W_src.T  [n, out_src]
    output_source = b.matmul(input_activations_source, b.transpose(source_weight))
    b.eval(output_source)

    # Step 3: Project to target coordinates using ORTHOGONALIZED transforms
    # input_tgt = input_src @ U_in  [n, in_tgt]
    input_target = b.matmul(input_activations_source, U_in)
    b.eval(input_target)

    # output_tgt = output_src @ U_out  [n, out_tgt]
    output_target = b.matmul(output_source, U_out)
    b.eval(output_target)

    # Step 3: Solve for weight via least squares
    # Find W such that input_tgt @ W.T = output_tgt
    # Equivalently: W.T = lstsq(input_tgt, output_tgt)
    # So: W = lstsq(input_tgt, output_tgt).T

    # Step 4: Solve for weight via closed-form normal equations
    # Solve input_target @ W.T = output_target for W.T
    W_T = gpu_lstsq(b, input_target, output_target)  # [in_tgt, out_tgt]
    b.eval(W_T)

    reconstructed_weight_raw = b.transpose(W_T)  # [out_tgt, in_tgt]
    b.eval(reconstructed_weight_raw)

    # Step 5: Apply scale correction for non-orthogonal alignment
    # The orthogonalization removes scaling from the coordinate transform,
    # but the weight magnitudes may need adjustment based on the original scaling.
    # scale_correction = scale_out / scale_in compensates for this.
    eps = float(machine_epsilon(b, alignment_in))
    if abs(scale_correction - 1.0) > eps:
        reconstructed_weight = reconstructed_weight_raw * scale_correction
        logger.info(
            "SCALE CORRECTION: applied factor %.4f to reconstructed weight",
            scale_correction
        )
    else:
        reconstructed_weight = reconstructed_weight_raw
    b.eval(reconstructed_weight)

    # Step 6: Compute reconstruction error
    output_reconstructed = b.matmul(input_target, W_T)  # [n, out_tgt]
    b.eval(output_reconstructed)

    error = b.mean(b.abs(output_reconstructed - output_target))
    reconstruction_error = float(b.to_scalar(error))

    # Compute norms for diagnostics
    source_norm = float(b.to_scalar(b.sqrt(b.sum(output_source * output_source))))
    target_norm = float(b.to_scalar(b.sqrt(b.sum(output_target * output_target))))

    # Estimate condition number from Gram matrix
    gram = b.matmul(b.transpose(input_target), input_target)
    b.eval(gram)
    eigvals = b.eigvalsh(gram)
    b.eval(eigvals)
    eps = float(machine_epsilon(b, gram))
    max_eig = float(b.to_scalar(b.max(eigvals)))
    min_eig = float(b.to_scalar(b.min(b.abs(eigvals))))
    condition_number = max_eig / max(min_eig, eps)

    logger.info(
        "BEHAVIORAL RESULT: error=%.6f, src_norm=%.2f, tgt_norm=%.2f, cond=%.2e",
        reconstruction_error, source_norm, target_norm, condition_number
    )

    return BehavioralReconstructionResult(
        reconstructed_weight=reconstructed_weight,
        reconstruction_error=reconstruction_error,
        source_behavior_norm=source_norm,
        target_behavior_norm=target_norm,
        condition_number=condition_number,
    )


def reconstruct_weight_manifold_aware(
    source_weight: "Array",
    input_activations_source: "Array",
    alignment_in: "Array",
    alignment_out: "Array",
    backend: "Backend | None" = None,
) -> BehavioralReconstructionResult:
    """Manifold-aware reconstruction using RMT signal/noise rank detection."""
    b = backend or get_default_backend()

    source_weight = b.array(source_weight)
    input_activations_source = b.array(input_activations_source)
    alignment_in = b.array(alignment_in)
    alignment_out = b.array(alignment_out)

    # Use high precision
    compute_dtype = precision_dtype(b, reference=source_weight)
    source_weight = b.astype(source_weight, compute_dtype)
    input_activations_source = b.astype(input_activations_source, compute_dtype)
    alignment_in = b.astype(alignment_in, compute_dtype)
    alignment_out = b.astype(alignment_out, compute_dtype)
    b.eval(source_weight, input_activations_source, alignment_in, alignment_out)

    # Get dimensions
    out_src = int(source_weight.shape[0])
    in_src = int(source_weight.shape[1])
    n_samples = int(input_activations_source.shape[0])
    in_tgt = int(alignment_in.shape[1])
    out_tgt = int(alignment_out.shape[1])

    # Step 1: Orthogonalize alignment transforms via polar decomposition
    # This separates rotation (norm-preserving) from scaling
    U_in, scale_in = orthogonalize_alignment(alignment_in, b)
    U_out, scale_out = orthogonalize_alignment(alignment_out, b)
    b.eval(U_in, U_out)

    # Compute scale correction factor: output scaling / input scaling
    eps = float(division_epsilon(b, alignment_in))
    scale_correction = scale_out / max(scale_in, eps)

    logger.info(
        "MANIFOLD ORTHOGONALIZATION: scale_in=%.4f, scale_out=%.4f, correction=%.4f",
        scale_in, scale_out, scale_correction
    )

    # Step 2: Compute source layer behavior
    output_source = b.matmul(input_activations_source, b.transpose(source_weight))
    b.eval(output_source)

    # Step 3: Project to target coordinates using ORTHOGONALIZED transforms
    input_target = b.matmul(input_activations_source, U_in)
    output_target = b.matmul(output_source, U_out)
    b.eval(input_target, output_target)

    # Step 4: Detect intrinsic rank using Marchenko-Pastur signal/noise separation
    # OPTIMIZATION: Compute SVD once and reuse for both MP rank detection and pinv
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )

    U_svd, S, Vt = geodesic_svd(b, input_target)
    b.eval(U_svd, S, Vt)

    # Use RMT Marchenko-Pastur distribution to determine true signal rank
    # Eigenvalues above the MP bulk edge are TRUE SIGNAL, within bulk are NOISE
    # This reuses the singular values S instead of recomputing eigenvalues
    mp_result = compute_signal_rank_from_singular_values(
        S, n_samples=n_samples, n_features=in_tgt, backend=b
    )
    n_sv = int(S.shape[0])
    intrinsic_rank = max(0, min(int(mp_result.signal_rank), n_sv))

    logger.info(
        "RMT SIGNAL/NOISE: signal_rank=%d, noise_rank=%d, MP_edge=%.6f, signal_var=%.1f%%",
        mp_result.signal_rank, mp_result.noise_rank,
        mp_result.mp_upper_edge, 100.0 * mp_result.signal_variance_fraction
    )

    logger.info(
        "MANIFOLD RECONSTRUCTION: [%d, %d] -> [%d, %d], n=%d, intrinsic_rank=%d",
        out_src, in_src, out_tgt, in_tgt, n_samples, intrinsic_rank
    )

    # Step 5: Rank-truncated lstsq using pre-computed SVD
    # W.T = pinv_k(input_target) @ output_target
    # pinv_k = V_k @ diag(1/S_k) @ U_k.T
    k = min(intrinsic_rank, int(S.shape[0]))
    U_k = U_svd[:, :k]  # [n, k]
    S_k = S[:k]         # [k]
    Vt_k = Vt[:k, :]    # [k, d]

    eps_pinv = float(division_epsilon(b, S))
    S_inv = 1.0 / (S_k + eps_pinv)  # [k]
    V_k = b.transpose(Vt_k)  # [d, k]
    VS = V_k * S_inv  # [d, k]
    input_tgt_pinv_k = b.matmul(VS, b.transpose(U_k))  # [d, n]
    b.eval(input_tgt_pinv_k)

    W_T = b.matmul(input_tgt_pinv_k, output_target)  # [in_tgt, out_tgt]
    b.eval(W_T)

    reconstructed_weight_raw = b.transpose(W_T)  # [out_tgt, in_tgt]
    b.eval(reconstructed_weight_raw)

    # Step 6: Apply scale correction for non-orthogonal alignment
    # The orthogonalization removes scaling from the coordinate transform,
    # but the weight magnitudes may need adjustment based on the original scaling.
    eps = float(machine_epsilon(b, alignment_in))
    if abs(scale_correction - 1.0) > eps:
        reconstructed_weight = reconstructed_weight_raw * scale_correction
        logger.info(
            "MANIFOLD SCALE CORRECTION: applied factor %.4f to reconstructed weight",
            scale_correction
        )
    else:
        reconstructed_weight = reconstructed_weight_raw
    b.eval(reconstructed_weight)

    # Step 7: Compute reconstruction error
    output_reconstructed = b.matmul(input_target, W_T)  # [n, out_tgt]
    b.eval(output_reconstructed)

    # Full error
    error_full = b.mean(b.abs(output_reconstructed - output_target))
    reconstruction_error = float(b.to_scalar(error_full))

    # Manifold-only error (project both to rank-k)
    if intrinsic_rank > 0:
        U_k = U_svd[:, :intrinsic_rank]
        output_target_manifold = b.matmul(U_k, b.matmul(b.transpose(U_k), output_target))
        output_recon_manifold = b.matmul(U_k, b.matmul(b.transpose(U_k), output_reconstructed))
        error_manifold = b.mean(b.abs(output_recon_manifold - output_target_manifold))
        manifold_error = float(b.to_scalar(error_manifold))
    else:
        manifold_error = reconstruction_error

    # Compute norms
    source_norm = float(b.to_scalar(b.sqrt(b.sum(output_source * output_source))))
    target_norm = float(b.to_scalar(b.sqrt(b.sum(output_target * output_target))))

    # Condition number of truncated system
    if intrinsic_rank > 0:
        S_k = S[:intrinsic_rank]
        eps = float(machine_epsilon(b, S))
        max_s = float(b.to_scalar(b.max(S_k)))
        min_s = float(b.to_scalar(b.min(S_k)))
        condition_number = max_s / max(min_s, eps)
    else:
        condition_number = float("inf")

    logger.info(
        "MANIFOLD RESULT: full_error=%.6f, manifold_error=%.10f, rank=%d, cond=%.2e",
        reconstruction_error, manifold_error, intrinsic_rank, condition_number
    )

    return BehavioralReconstructionResult(
        reconstructed_weight=reconstructed_weight,
        reconstruction_error=manifold_error,  # Report manifold error as the true error
        source_behavior_norm=source_norm,
        target_behavior_norm=target_norm,
        condition_number=condition_number,
    )


def compute_cross_dimensional_transplant(
    source_weight: "Array",
    target_weight: "Array",
    input_activations_source: "Array",
    input_activations_target: "Array",
    alignment_in: "Array",
    alignment_out: "Array",
    source_activations_for_density: "Array | None" = None,
    target_activations_for_density: "Array | None" = None,
    delta_scale: float = 1.0,
    backend: "Backend | None" = None,
    manifold_aware: bool = True,
    use_spectral_blend: bool = True,
) -> "WeightSpaceTransplantResult":
    """Cross-dimensional weight transplant via behavioral reconstruction.

    For different model dimensions, reconstruct a weight that matches source
    input/output behavior in target coordinates.

    Algorithm:
        1. Reconstruct source behavior in target coordinates:
           W_behavior = reconstruct_weight_from_behavior(source_weight, ...)
        2. Compute delta from target: delta = W_behavior - target_weight
        3. Apply null-space projection to preserve target behavior
        4. Merge: merged = target_weight + delta_scale * delta_projected

    SPECTRAL BLENDING MODE (delta_scale < 0.5 and use_spectral_blend=True):
        Instead of uniform blending, uses spectral-aware blending:
        - Dominant singular directions: blend more aggressively (models agree)
        - Secondary directions: blend conservatively (model-specific expansion)

        This respects the geometry discovered in compression experiments:
        bottleneck layers EXPAND secondary directions by 15-20x, so those
        expansion factors should be gently perturbed, not replaced.

    Args:
        source_weight: Source weight matrix [out_src, in_src].
        target_weight: Target weight matrix [out_tgt, in_tgt].
        input_activations_source: Input activations in source space [n, in_src].
        input_activations_target: Input activations in target space [n, in_tgt].
        alignment_in: Procrustes transform for inputs [in_src, in_tgt].
        alignment_out: Procrustes transform for outputs [out_src, out_tgt].
        source_activations_for_density: For density-weighted transfer [n, d_src].
        target_activations_for_density: For density-weighted transfer [n, d_tgt].
        delta_scale: Scaling factor for the delta (default 1.0).
        backend: Compute backend.
        manifold_aware: Use manifold-aware reconstruction (default True).
            When True, truncates to intrinsic dimension for near-zero error.
            When False, uses full-rank reconstruction (legacy behavior).
        use_spectral_blend: Use spectral-aware blending when delta_scale < 0.5.
            Default True. Set False to use uniform blending (legacy behavior).

    Returns:
        WeightSpaceTransplantResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()

    target_weight = b.array(target_weight)
    input_activations_target = b.array(input_activations_target)
    output_dtype = b.dtype(target_weight)

    # Step 1: Reconstruct source weight in target coordinates
    # Use manifold-aware reconstruction for near-zero error
    if manifold_aware:
        reconstruction = reconstruct_weight_manifold_aware(
            source_weight=source_weight,
            input_activations_source=input_activations_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )
    else:
        reconstruction = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_activations_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

    source_behavioral = reconstruction.reconstructed_weight
    b.eval(source_behavioral)

    logger.info(
        "CROSS-DIM TRANSPLANT: reconstruction_error=%.6f, condition=%.2e",
        reconstruction.reconstruction_error,
        reconstruction.condition_number,
    )

    # Step 2: Compute delta between behavioral source and target
    compute_dtype = precision_dtype(b, reference=target_weight)
    source_behavioral = b.astype(source_behavioral, compute_dtype)
    target_weight_compute = b.astype(target_weight, compute_dtype)
    b.eval(source_behavioral, target_weight_compute)

    delta_W = source_behavioral - target_weight_compute
    b.eval(delta_W)

    # Use BEHAVIORAL norm (not Frobenius) to measure actual output change
    delta_norm = _behavioral_norm(delta_W, input_activations_target, b)

    # =========================================================================
    # BLENDING MODE: ACTIVE SUBSPACE, SPECTRAL, OR UNIFORM (delta_scale < 0.5)
    # =========================================================================
    # For small delta_scale (blending mode), we have three options:
    #
    # 1. ACTIVE SUBSPACE: Blend in ACTIVATION-defined subspace (recommended)
    #    Uses compute_variance_null_space() to identify active (~465) and null
    #    directions based on activation covariance, NOT weight SVD.
    #    - Active directions: blend MORE (models agree on computation)
    #    - Null directions: blend LESS (preserve target's expansion factors)
    #
    # 2. SPECTRAL-AWARE: Blend differently per weight singular direction
    #    DEPRECATED: The weight's SVD basis doesn't align with activation flow.
    #    Produced degenerate output - DO NOT USE.
    #
    # 3. UNIFORM (legacy): Same blend ratio across all directions
    #    - Works! Produced coherent output.
    #    - Use as fallback if active subspace fails.
    #
    # CURRENT DEFAULT: Active subspace blending (activation-defined basis).
    # =========================================================================
    use_active_subspace_blend = True   # Enabled - uses activation variance basis
    use_spectral_blend_active = False  # Disabled - weight SVD doesn't work
    if delta_scale < 0.5 and use_active_subspace_blend:
        # ACTIVE SUBSPACE BLENDING (recommended)
        from modelcypher.core.domain.geometry.active_subspace_blend import (
            compute_adaptive_active_blend,
        )

        logger.info(
            "CROSS-DIM ACTIVE SUBSPACE BLEND: delta_scale=%.3f, using activation-defined basis",
            delta_scale,
        )

        # Use adaptive active blend - adapts boost/dampen based on concentration
        active_result = compute_adaptive_active_blend(
            source_weight=source_behavioral,
            target_weight=target_weight_compute,
            input_activations=input_activations_target,
            base_blend=delta_scale,  # Use delta_scale as the base (e.g., 0.1)
            backend=b,
        )

        merged_weight = active_result.blended_weight
        if str(b.dtype(merged_weight)) != str(output_dtype):
            merged_weight = b.astype(merged_weight, output_dtype)
        b.eval(merged_weight)

        # Compute behavioral metrics for compatibility
        delta_W_effective = merged_weight - target_weight_compute
        b.eval(delta_W_effective)
        projected_norm = _behavioral_norm(delta_W_effective, input_activations_target, b)

        eps = float(division_epsilon(b, delta_W))
        if delta_norm > eps:
            preserved_fraction = projected_norm / delta_norm
        else:
            preserved_fraction = 1.0

        logger.info(
            "CROSS-DIM ACTIVE RESULT: effective_blend=%.3f, active=%.3f, "
            "null=%.3f, active_rank=%d, null_rank=%d, var_captured=%.3f",
            active_result.effective_blend_ratio,
            active_result.active_blend,
            active_result.null_blend,
            active_result.active_rank,
            active_result.null_rank,
            active_result.variance_captured,
        )

        return WeightSpaceTransplantResult(
            merged_weight=merged_weight,
            delta_norm=delta_norm,
            projected_norm=projected_norm,
            preserved_fraction=preserved_fraction,
            transfer_strength=active_result.effective_blend_ratio,
            null_rank=active_result.null_rank,  # Report actual null rank
        )

    elif delta_scale < 0.5 and use_spectral_blend_active:
        # SPECTRAL-AWARE BLENDING
        from modelcypher.core.domain.geometry.spectral_blend import (
            compute_adaptive_spectral_blend,
        )

        logger.info(
            "CROSS-DIM SPECTRAL BLEND: delta_scale=%.3f, using direction-aware blending",
            delta_scale,
        )

        # Use adaptive spectral blend - adapts boost/dampen based on input manifold
        spectral_result = compute_adaptive_spectral_blend(
            source_weight=source_behavioral,
            target_weight=target_weight_compute,
            input_activations=input_activations_target,
            base_blend=delta_scale,  # Use delta_scale as the base (e.g., 0.1)
            backend=b,
        )

        merged_weight = spectral_result.blended_weight
        if str(b.dtype(merged_weight)) != str(output_dtype):
            merged_weight = b.astype(merged_weight, output_dtype)
        b.eval(merged_weight)

        # Compute behavioral metrics for compatibility
        delta_W_effective = merged_weight - target_weight_compute
        b.eval(delta_W_effective)
        projected_norm = _behavioral_norm(delta_W_effective, input_activations_target, b)

        eps = float(division_epsilon(b, delta_W))
        if delta_norm > eps:
            preserved_fraction = projected_norm / delta_norm
        else:
            preserved_fraction = 1.0

        logger.info(
            "CROSS-DIM SPECTRAL RESULT: effective_blend=%.3f, dominant=%.3f, "
            "secondary=%.3f, var_concentration=%.3f",
            spectral_result.effective_blend_ratio,
            spectral_result.dominant_blend,
            spectral_result.secondary_blend,
            spectral_result.variance_concentration,
        )

        return WeightSpaceTransplantResult(
            merged_weight=merged_weight,
            delta_norm=delta_norm,
            projected_norm=projected_norm,
            preserved_fraction=preserved_fraction,
            transfer_strength=spectral_result.effective_blend_ratio,
            null_rank=0,  # Spectral blend doesn't use null-space projection
        )

    elif delta_scale < 0.5:
        # UNIFORM BLENDING (legacy mode)
        logger.info(
            "CROSS-DIM UNIFORM BLEND: delta_scale=%.3f < 0.5, skipping k-NN density",
            delta_scale,
        )
        source_density = None
    else:
        # FULL TRANSFER MODE
        # Check if source activations are compatible with target dimension
        # In cross-dimensional transplant, source may have different dim than target
        if source_activations_for_density is not None:
            source_acts = b.array(source_activations_for_density)
            source_dim = int(source_acts.shape[1])
            target_dim = int(input_activations_target.shape[1])
            if source_dim != target_dim:
                logger.info(
                    "CROSS-DIM GEODESIC: Skipping source density (dim %d != target dim %d)",
                    source_dim, target_dim
                )
                source_density = None
            else:
                source_density = source_activations_for_density
        else:
            source_density = None

    # =========================================================================
    # GEODESIC NULL-SPACE FILTERING (RMT-based signal/noise separation)
    # =========================================================================
    # Uses Random Matrix Theory (Marchenko-Pastur distribution) to separate
    # true signal from noise. This is more principled than variance-based
    # heuristics and respects the geodesic geometry of activation space.
    # =========================================================================
    geodesic_filter = GeodesicNullSpaceFilter(b)

    logger.info(
        "CROSS-DIM GEODESIC: Filtering delta with RMT-based null-space projection"
    )

    # Filter delta using geodesic null-space (RMT-based)
    geodesic_result = geodesic_filter.filter_delta(
        weight_delta=delta_W,
        prior_activations=input_activations_target,
        source_activations=source_density,
    )
    b.eval(geodesic_result.filtered_delta)

    delta_W_proj = geodesic_result.filtered_delta
    null_rank = geodesic_result.orthogonal_dim
    transfer_strength = 1.0 - geodesic_result.projection_loss

    # Use BEHAVIORAL norm (not Frobenius) to measure actual output change after projection
    projected_norm = _behavioral_norm(delta_W_proj, input_activations_target, b)

    # Preserved fraction
    eps = float(division_epsilon(b, delta_W))
    if delta_norm > eps:
        preserved_fraction = projected_norm / delta_norm
    else:
        preserved_fraction = 1.0

    # Step 5: Apply to target weight
    merged_weight = target_weight_compute + delta_scale * delta_W_proj
    if str(b.dtype(merged_weight)) != str(output_dtype):
        merged_weight = b.astype(merged_weight, output_dtype)
    b.eval(merged_weight)

    logger.info(
        "CROSS-DIM GEODESIC RESULT: delta_norm=%.4f, proj_norm=%.4f, "
        "preserved=%.1f%%, orthogonal_dim=%d, k=%d",
        delta_norm, projected_norm, 100.0 * preserved_fraction,
        geodesic_result.orthogonal_dim, geodesic_result.k_neighbors,
    )

    return WeightSpaceTransplantResult(
        merged_weight=merged_weight,
        delta_norm=delta_norm,
        projected_norm=projected_norm,
        preserved_fraction=preserved_fraction,
        transfer_strength=transfer_strength,
        null_rank=null_rank,
    )


def compute_null_space_projector(
    input_activations: "Array",
    *,
    source_activations_for_density: "Array | None" = None,
    target_activations_for_density: "Array | None" = None,
    density_weights: "Array | None" = None,
    coupling_weight: float | None = None,
    backend: "Backend | None" = None,
) -> NullSpaceProjector:
    """Compute a reusable null-space projector from input activations.

    If density_weights are provided, they are used directly to weight the
    boundary activations. Otherwise, density weights are computed from
    source/target activations when available.

    Args:
        input_activations: Target activations to build null-space from.
        source_activations_for_density: Optional source activations for density.
        target_activations_for_density: Optional target activations for density.
        density_weights: Optional pre-computed density weights.
        coupling_weight: Optional HOT coupling weight for this layer pair.
            If provided, transfer_strength is scaled by this weight.
            High coupling = strong alignment = transfer more.
        backend: Backend for tensor operations.
    """
    from modelcypher.core.domain.geometry.knowledge_density import (
        compute_density_weights,
        compute_knn_point_cloud_density,
    )

    b = backend or get_default_backend()

    input_activations = b.array(input_activations)
    compute_dtype = precision_dtype(b, reference=input_activations)

    if density_weights is not None:
        density_weights = b.astype(b.array(density_weights), compute_dtype)
    if source_activations_for_density is not None:
        source_activations_for_density = b.astype(
            b.array(source_activations_for_density), compute_dtype
        )
    if target_activations_for_density is not None:
        target_activations_for_density = b.astype(
            b.array(target_activations_for_density), compute_dtype
        )

    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(input_activations)

    n_samples = int(input_activations.shape[0])
    if n_samples == 0:
        raise ValueError(
            "Cannot compute null-space with n_samples=0. "
            "This indicates a bug in the probe stage - no activations were collected."
        )

    transfer_strength = 1.0

    if density_weights is None and source_activations_for_density is not None and target_activations_for_density is not None:
        b.eval(source_activations_for_density, target_activations_for_density)
        density_result = compute_knn_point_cloud_density(
            source_activations=source_activations_for_density,
            target_activations=target_activations_for_density,
            backend=b,
        )
        density_weights = compute_density_weights(
            source_densities=density_result.source_densities,
            target_densities=density_result.target_densities,
            backend=b,
        )
        density_weights = b.astype(density_weights, compute_dtype)
        b.eval(density_weights)

    if density_weights is not None:
        n_density = int(density_weights.shape[0])
        if n_density != n_samples:
            n_compare = min(n_density, n_samples)
            density_weights = density_weights[:n_compare]
            input_activations = input_activations[:n_compare]
            n_samples = n_compare
            b.eval(density_weights, input_activations)

        transfer_strength = float(b.to_scalar(b.mean(density_weights)))

        # Scale by HOT coupling weight if provided
        # High coupling = strong layer alignment = transfer more
        # Low coupling = weak alignment = transfer less (even if density is high)
        if coupling_weight is not None:
            transfer_strength *= coupling_weight

        constraint_weights = 1.0 - density_weights
        eps = division_epsilon(b, constraint_weights)
        sqrt_weights = b.sqrt(constraint_weights + eps)
        A_weighted = input_activations * b.reshape(sqrt_weights, (-1, 1))
        b.eval(A_weighted)
    else:
        A_weighted = input_activations

    # =========================================================================
    # CHOOSE SMALLER GRAM MATRIX: O(min(n,d)³) instead of O(n³)
    # =========================================================================
    # When n > d (more samples than dimensions), work in d-dimensional space:
    # - Build A.T @ A (d×d) instead of A @ A.T (n×n)
    # - Return explicit projector matrix P = I - V_r @ V_r.T
    # This is 27x faster when n=3060, d=1024 (our bottleneck case).
    # =========================================================================
    in_dim = int(input_activations.shape[1])

    # Use smaller dimension for eigendecomposition
    use_feature_space = n_samples > in_dim

    if use_feature_space:
        # Build d×d covariance matrix (feature space)
        AtA = b.matmul(b.transpose(A_weighted), A_weighted)
        b.eval(AtA)
        eps = machine_epsilon(b, AtA)
        gram_size = in_dim
        logger.info(
            "NULL-SPACE: Using feature space (d×d=%d×%d instead of n×n=%d×%d)",
            in_dim, in_dim, n_samples, n_samples,
        )
    else:
        # Build n×n Gram matrix (sample space) - original behavior
        AtA = b.matmul(A_weighted, b.transpose(A_weighted))
        b.eval(AtA)
        eps = machine_epsilon(b, AtA)
        gram_size = n_samples

    # =========================================================================
    # EIGENDECOMPOSITION: COMPUTE ONCE, REUSE FOR RANK AND PROJECTOR
    # =========================================================================
    eigvals, eigvecs = b.eigh(AtA)
    b.eval(eigvals, eigvecs)

    # eigh returns eigenvalues in ascending order; reverse to descending
    n_eigs = int(eigvals.shape[0])
    idx = b.arange(n_eigs - 1, -1, -1)
    eigvals = b.take(eigvals, idx, axis=0)
    eigvecs = b.take(eigvecs, idx, axis=1)  # columns are eigenvectors
    b.eval(eigvals, eigvecs)

    eigvals_pos = b.maximum(eigvals, eps)
    total_var = b.sum(eigvals_pos)
    b.eval(total_var)
    total_var_val = float(b.to_scalar(total_var))

    if total_var_val < eps:
        raise ValueError(
            f"Zero variance in activations (total_var={total_var_val:.2e}). "
            "This indicates a bug in the pipeline - activations should have non-zero variance. "
            "Check that the model is loaded correctly and inference is running properly."
        )

    # =========================================================================
    # COMPUTE NUMERICAL RANK AND NULL-SPACE DIMENSION
    # =========================================================================
    eps_rank = machine_epsilon(b, eigvals_pos)
    max_eig_arr = b.max(eigvals_pos)
    b.eval(max_eig_arr)
    max_eig = float(b.to_scalar(max_eig_arr))
    max_eig_safe = max(max_eig, eps_rank)

    rank_scale = svd_rank_threshold(b, eigvals_pos, in_dim)
    rank_threshold = max_eig_safe * rank_scale
    rank_mask = eigvals_pos > rank_threshold
    rank_mask = b.astype(rank_mask, compute_dtype)

    activation_rank_arr = b.sum(rank_mask)
    b.eval(activation_rank_arr)
    activation_rank = int(round(float(b.to_scalar(activation_rank_arr))))
    activation_rank = max(0, min(activation_rank, gram_size))
    null_rank = max(0, in_dim - activation_rank)

    logger.info(
        "NULL-SPACE: numeric_rank=%d/%d, null_rank=%d, max_eig=%.3e",
        activation_rank, gram_size, null_rank, max_eig,
    )

    if use_feature_space:
        # =====================================================================
        # FEATURE SPACE: Build explicit d×d projector
        # =====================================================================
        # For A.T @ A = V @ diag(λ) @ V.T, the null-space projector is:
        # P = I - V_r @ V_r.T  (where V_r = columns with λ > threshold)
        # This projects into directions NOT spanned by activation covariance.
        # =====================================================================
        eps_pinv = division_epsilon(b, eigvals_pos)
        pinv_threshold = eps_rank * n_eigs * max_eig_safe

        # Build row-space projector V_r @ V_r.T (occupied directions)
        pinv_mask = eigvals_pos > pinv_threshold
        pinv_mask_float = b.astype(pinv_mask, compute_dtype)

        # V_r = eigenvectors with significant eigenvalues
        # Row-space projector: V_r @ V_r.T = V @ diag(mask) @ V.T
        V_masked = eigvecs * b.reshape(pinv_mask_float, (1, -1))
        b.eval(V_masked)
        row_space_proj = b.matmul(V_masked, b.transpose(eigvecs))
        b.eval(row_space_proj)

        # Null-space projector: P = I - row_space_proj
        I_d = b.eye(in_dim, dtype=compute_dtype)
        null_projector = I_d - row_space_proj
        b.eval(null_projector)

        return NullSpaceProjector(
            weighted_activations=A_weighted,
            gram_inv=null_projector,  # Not actually gram_inv, but kept for compat
            null_rank=null_rank,
            transfer_strength=transfer_strength,
            projector=null_projector,  # Explicit projector - will be used directly
        )
    else:
        # =====================================================================
        # SAMPLE SPACE: Original behavior (n×n pseudoinverse)
        # =====================================================================
        eps_pinv = division_epsilon(b, eigvals_pos)
        pinv_threshold = eps_rank * n_eigs * max_eig_safe
        pinv_mask = eigvals_pos > pinv_threshold
        eigvals_inv = b.where(pinv_mask, 1.0 / (eigvals_pos + eps_pinv), b.zeros_like(eigvals_pos))
        b.eval(eigvals_inv)

        # AAt_inv = V @ diag(1/λ) @ V.T
        V_scaled = eigvecs * b.reshape(eigvals_inv, (1, -1))
        b.eval(V_scaled)
        AAt_inv = b.matmul(V_scaled, b.transpose(eigvecs))
        b.eval(AAt_inv)

        return NullSpaceProjector(
            weighted_activations=A_weighted,
            gram_inv=AAt_inv,
            null_rank=null_rank,
            transfer_strength=transfer_strength,
        )


def compute_trajectory_tangent_projector(
    trajectories: "list[TrajectoryResult]",
    backend: "Backend | None" = None,
    use_full_null: bool = False,
) -> "TrajectoryTangentProjector | None":
    """Compute a trajectory-tangent projector for null-space projection.

    This builds a projector that puts deltas "along the road" - in null-space
    directions that are aligned with the model's natural trajectory flow.

    Key insight: Not all null-space directions are equal. Velocities show
    WHERE the trajectory is heading. Null-space directions aligned with
    velocities are "shoulders of the road" - unused capacity that's still
    reachable by the model's natural information flow.

    This is more targeted than variance-based null-space projection:
    - Variance-based: projects into ALL low-variance directions
    - Trajectory-tangent: projects into directions ALIGNED with velocity flow

    Args:
        trajectories: List of TrajectoryResult objects from trajectory collection.
        backend: Compute backend.
        use_full_null: If True, use full null space instead of tangent subspace.
            Useful for comparison or fallback.

    Returns:
        TrajectoryTangentProjector, or None if computation fails.
    """
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        compute_trajectory_tangent_null_space,
    )

    b = backend or get_default_backend()

    if not trajectories:
        logger.warning("TRAJECTORY TANGENT PROJECTOR: No trajectories provided")
        return None

    tangent_result = compute_trajectory_tangent_null_space(
        trajectories=trajectories,
        backend=b,
    )

    if tangent_result is None:
        logger.warning("TRAJECTORY TANGENT PROJECTOR: Computation failed")
        return None

    logger.info(
        "TRAJECTORY TANGENT PROJECTOR: null_rank=%d, tangent_rank=%d, "
        "velocity_alignment=%.4f, use_full_null=%s",
        tangent_result.null_rank,
        tangent_result.tangent_rank,
        tangent_result.velocity_alignment,
        use_full_null,
    )

    return TrajectoryTangentProjector(
        tangent_result=tangent_result,
        null_rank=tangent_result.null_rank,
        tangent_rank=tangent_result.tangent_rank,
        velocity_alignment=tangent_result.velocity_alignment,
        use_full_null=use_full_null,
    )


def apply_trajectory_tangent_projection(
    delta_W: "Array",
    tangent_projector: "TrajectoryTangentProjector",
    backend: "Backend",
) -> "Array":
    """Apply trajectory-tangent projection to a weight delta.

    Projects the delta into null-space directions aligned with trajectory
    velocity flow - "building along the road."

    Args:
        delta_W: Weight delta [out_dim, in_dim].
        tangent_projector: Precomputed trajectory-tangent projector.
        backend: Compute backend.

    Returns:
        Projected delta [out_dim, in_dim].
    """
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        project_delta_to_trajectory_tangent,
    )

    b = backend

    delta_proj = project_delta_to_trajectory_tangent(
        delta=delta_W,
        tangent_result=tangent_projector.tangent_result,
        backend=b,
        use_full_null=tangent_projector.use_full_null,
    )

    return delta_proj


def compute_weight_space_transplant(
    source_aligned: "Array",
    target_weight: "Array",
    input_activations: "Array",
    source_activations_for_density: "Array | None" = None,
    target_activations_for_density: "Array | None" = None,
    null_space_projector: "NullSpaceProjector | None" = None,
    trajectory_tangent_projector: "TrajectoryTangentProjector | None" = None,
    delta_scale: float = 1.0,
    backend: "Backend | None" = None,
) -> WeightSpaceTransplantResult:
    """Weight-space null-space projection with density-weighted transfer.

    This is the SINGULAR PIPELINE for knowledge transfer. Works for ALL weight
    dimensions: hidden→hidden, hidden→intermediate, intermediate→hidden.

    The math:
        delta_W = source_aligned - target_weight  [out_dim, in_dim]
        N = I - pinv(A_input_weighted) @ A_input_weighted  [in_dim, in_dim]
        delta_W_proj = delta_W @ N  [out_dim, in_dim]
        merged = target_weight + delta_W_proj

    Projection modes (in order of preference):
        1. trajectory_tangent_projector: Projects into null-space directions
           aligned with trajectory velocity flow ("building along the road").
           Most targeted - uses velocity alignment to find reachable null space.
        2. null_space_projector: Variance-based null-space projection.
           Projects into ALL low-variance directions.
        3. Computed from input_activations: Default fallback.

    Density weighting:
        - Compares k-NN density between source and target activations
        - High target density → stronger constraint (preserve target)
        - Low target density → weaker constraint (accept source knowledge)
        - Weights are applied to boundary activations via sqrt(w) scaling

    Args:
        source_aligned: Source weight already stitched to target dims [out, in].
        target_weight: Target weight to modify [out, in].
        input_activations: INPUT activations to this weight [n, in_dim].
            For hidden→X weights: use hidden activations.
            For intermediate→X weights: use intermediate activations.
        source_activations_for_density: Source activations for density comparison [n, d_src].
            If None, density weighting is disabled (uniform transfer).
            For cross-arch, this can have different dimension than target.
        target_activations_for_density: Target activations for density comparison [n, d_tgt].
            If None, density weighting is disabled (uniform transfer).
        null_space_projector: Precomputed variance-based null-space projector.
            Ignored if trajectory_tangent_projector is provided.
        trajectory_tangent_projector: Precomputed trajectory-tangent projector.
            Takes precedence over null_space_projector if both provided.
            Projects delta into null-space directions aligned with velocity flow.
        delta_scale: Scaling factor for the delta (default 1.0).
        backend: Compute backend.

    Returns:
        WeightSpaceTransplantResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()

    source_aligned = b.array(source_aligned)
    target_weight = b.array(target_weight)
    input_activations = b.array(input_activations)
    output_dtype = b.dtype(target_weight)
    compute_dtype = precision_dtype(b, reference=target_weight)
    for arr in (source_aligned, input_activations):
        if hasattr(arr, "dtype"):
            try:
                if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = arr.dtype
            except Exception as exc:
                logger.debug("Failed to compare dtype eps for transplant inputs: %s", exc)
    source_aligned = b.astype(source_aligned, compute_dtype)
    target_weight = b.astype(target_weight, compute_dtype)
    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(source_aligned, target_weight, input_activations)

    out_dim = int(target_weight.shape[0])
    in_dim = int(target_weight.shape[1])
    n_samples = int(input_activations.shape[0])

    # =========================================================================
    # MAGNITUDE NORMALIZATION FOR CROSS-ARCHITECTURE MERGING
    # =========================================================================
    # The Procrustes transform F is NOT norm-preserving when dimensions differ.
    # The stitch formula P @ W @ Q can amplify/shrink weights by 50x or more.
    #
    # To ensure meaningful knowledge transfer (not magnitude artifacts):
    # 1. Normalize source_aligned to match target_weight's Frobenius norm
    # 2. Compute delta in this normalized space
    # 3. The delta represents structural differences, not scale differences
    # =========================================================================
    eps = division_epsilon(b, target_weight)

    # Geodesic Frobenius norms (treat rows as points on manifold)
    source_norm_val = _weight_frobenius_norm(source_aligned, b)
    target_norm_val = _weight_frobenius_norm(target_weight, b)

    # Normalize source_aligned to match target magnitude
    if source_norm_val > eps:
        norm_scale = target_norm_val / source_norm_val
        source_normalized = source_aligned * norm_scale
        b.eval(source_normalized)
        logger.debug(
            "MAGNITUDE NORM: source=%.2f, target=%.2f, scale=%.4f",
            source_norm_val, target_norm_val, norm_scale
        )
    else:
        source_normalized = source_aligned
        norm_scale = 1.0
        logger.debug("MAGNITUDE NORM: skipped (source_norm near zero)")

    logger.debug(
        "WEIGHT-SPACE TRANSPLANT: weight=[%d, %d], n_input=%d",
        out_dim, in_dim, n_samples
    )

    # Step 1: Compute weight delta (in normalized space)
    delta_W = source_normalized - target_weight  # [out_dim, in_dim]
    b.eval(delta_W)

    # =========================================================================
    # NO FAST PATH FOR SINGLE-LAYER TRANSFER
    # =========================================================================
    # For single-layer bottleneck transfer, we NEED null-space projection to
    # preserve target behavior - skipping it breaks network continuity because
    # adjacent layers expect specific activation statistics from this layer.
    #
    # The fast path (full transfer without projection) is only appropriate for
    # multi-layer or full-model merges where we're replacing everything coherently.
    # =========================================================================

    # Compute BEHAVIORAL norm before projection
    # This measures actual output change: ||A @ ΔW.T||_F
    # Unlike Frobenius norm, this captures how much the layer's behavior changes.
    delta_norm = _behavioral_norm(delta_W, input_activations, b)
    delta_frob = _weight_frobenius_norm(delta_W, b)  # For logging comparison

    # Compute cosine similarity between normalized source and target weights
    # Uses standard Euclidean cosine - weight space is NOT a curved manifold.
    # The spectral structure (Hessian eigenvalues) matters, not manifold curvature.
    src_flat = b.reshape(source_normalized, (-1,))
    tgt_flat = b.reshape(target_weight, (-1,))
    dot_prod = b.sum(src_flat * tgt_flat)
    eps_div = division_epsilon(b, src_flat)
    src_norm = b.sqrt(b.sum(src_flat * src_flat)) + eps_div
    tgt_norm = b.sqrt(b.sum(tgt_flat * tgt_flat)) + eps_div
    cosine_sim = float(b.to_scalar(dot_prod / (src_norm * tgt_norm)))

    logger.info(
        "STITCH QUALITY: cosine=%.4f, behavioral=%.4f, frobenius=%.4f, target=%.4f",
        cosine_sim,
        delta_norm,
        delta_frob,
        target_norm_val,
    )

    # =========================================================================
    # PROJECTION MODE SELECTION
    # =========================================================================
    # Priority order:
    # 1. trajectory_tangent_projector: Projects into velocity-aligned null space
    #    ("building along the road" - most targeted approach)
    # 2. null_space_projector: Variance-based null-space projection
    # 3. Computed from input_activations: Default fallback
    # =========================================================================

    if trajectory_tangent_projector is not None:
        # Use trajectory-tangent projection - projects into null-space
        # directions aligned with the model's trajectory velocity flow
        logger.info(
            "TRANSPLANT MODE: TRAJECTORY-TANGENT (tangent_rank=%d, null_rank=%d, "
            "velocity_alignment=%.4f)",
            trajectory_tangent_projector.tangent_rank,
            trajectory_tangent_projector.null_rank,
            trajectory_tangent_projector.velocity_alignment,
        )

        delta_W_proj = apply_trajectory_tangent_projection(
            delta_W=delta_W,
            tangent_projector=trajectory_tangent_projector,
            backend=b,
        )
        b.eval(delta_W_proj)

        # Use tangent_rank for reporting (the subspace we're projecting into)
        if trajectory_tangent_projector.use_full_null:
            null_rank = trajectory_tangent_projector.null_rank
        else:
            null_rank = trajectory_tangent_projector.tangent_rank
        transfer_strength = trajectory_tangent_projector.velocity_alignment

    else:
        # Fall back to variance-based null-space projection
        if null_space_projector is None:
            null_space_projector = compute_null_space_projector(
                input_activations=input_activations,
                source_activations_for_density=source_activations_for_density,
                target_activations_for_density=target_activations_for_density,
                backend=b,
            )

        logger.info(
            "TRANSPLANT MODE: VARIANCE-BASED (null_rank=%d, transfer_strength=%.4f)",
            null_space_projector.null_rank,
            null_space_projector.transfer_strength,
        )

        N = null_space_projector.projector
        null_rank = null_space_projector.null_rank
        transfer_strength = null_space_projector.transfer_strength

        # Project delta to null-space
        if N is None:
            A_weighted = null_space_projector.weighted_activations
            gram_inv = null_space_projector.gram_inv
            b.eval(A_weighted, gram_inv)

            # delta_W_proj = delta_W - (delta_W @ A.T) @ (A @ A.T)^+ @ A
            delta_row = b.matmul(delta_W, b.transpose(A_weighted))
            correction = b.matmul(delta_row, gram_inv)
            correction = b.matmul(correction, A_weighted)
            delta_W_proj = delta_W - correction
            b.eval(delta_W_proj)
        else:
            # delta_W_proj = delta_W @ N
            # [out_dim, in_dim] @ [in_dim, in_dim] -> [out_dim, in_dim]
            delta_W_proj = b.matmul(delta_W, N)
            b.eval(delta_W_proj)

    # Compute BEHAVIORAL norm after projection
    # This measures: "How much behavioral change survived the projection?"
    # For pure null-space projection, this should be ~0 on input_activations.
    projected_norm = _behavioral_norm(delta_W_proj, input_activations, b)
    projected_frob = _weight_frobenius_norm(delta_W_proj, b)  # For logging

    # Preserved fraction = behavioral_after / behavioral_before
    # This answers: "What fraction of the behavioral change transferred?"
    if delta_norm > 0:
        preserved_fraction = projected_norm / delta_norm
    else:
        preserved_fraction = 1.0

    # Log at INFO level for key metrics (behavioral norm is the truth, not Frobenius)
    # preserved_fraction ≈ 0 means projection is working (delta projected to null-space)
    # preserved_fraction > 0 means some delta leaked into used directions
    logger.info(
        "BEHAVIORAL TRANSFER: behavioral_before=%.4f, behavioral_after=%.4f, "
        "preserved=%.2f%%, null_rank=%d/%d (capacity=%.1f%%)",
        delta_norm, projected_norm, 100 * preserved_fraction,
        null_rank, in_dim, 100 * null_rank / in_dim if in_dim > 0 else 0,
    )
    logger.debug(
        "BEHAVIORAL TRANSFER (frob comparison): frob_before=%.4f, frob_after=%.4f, "
        "frob_preserved=%.1f%%",
        delta_frob, projected_frob, 100 * projected_frob / delta_frob if delta_frob > 0 else 0,
    )

    # Step 5: Apply to target weight.
    # Null-space projection preserves target behavior along boundary activations.
    merged_weight = target_weight + delta_scale * delta_W_proj
    if str(b.dtype(merged_weight)) != str(output_dtype):
        merged_weight = b.astype(merged_weight, output_dtype)
    b.eval(merged_weight)

    logger.debug(
        "TRANSPLANT RESULT: delta_norm=%.4f, proj_norm=%.4f, preserved=%.1f%%, transfer=%.3f",
        delta_norm, projected_norm, 100.0 * preserved_fraction, transfer_strength
    )

    return WeightSpaceTransplantResult(
        merged_weight=merged_weight,
        delta_norm=delta_norm,
        projected_norm=projected_norm,
        preserved_fraction=preserved_fraction,
        transfer_strength=transfer_strength,
        null_rank=null_rank,
    )


def _compute_transplant_delta_anchor_relative(
    weight_target: "Array",
    activations_core: "Array",
    delta_activations: "Array",
    boundary_activations: "Array | None",
    delta_scale: float,
    backend: "Backend",
) -> "TransplantDeltaResult":
    """Anchor-relative mode: constrained least-squares with boundary preservation.

    Solves:
        min ||A_core @ delta_W - delta_A_core||_F²
        s.t. A_boundary @ delta_W = 0

    Via null-space projection:
        N = I - pinv(A_boundary) @ A_boundary  (boundary null-space)
        delta_W_unc = pinv(A_core) @ delta_A_core  (unconstrained)
        delta_W = N @ delta_W_unc  (projected)
        W' = W_target + delta_W.T
    """
    b = backend

    weight_target = b.array(weight_target)
    activations_core = b.array(activations_core)
    delta_activations = b.array(delta_activations)
    output_dtype = b.dtype(weight_target)
    compute_dtype = precision_dtype(b, reference=weight_target)
    for arr in (activations_core, delta_activations):
        if hasattr(arr, "dtype"):
            try:
                if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = arr.dtype
            except Exception as exc:
                logger.debug("Failed to compare dtype eps for anchor-relative transplant: %s", exc)
    weight_target = b.astype(weight_target, compute_dtype)
    activations_core = b.astype(activations_core, compute_dtype)
    delta_activations = b.astype(delta_activations, compute_dtype)
    b.eval(weight_target, activations_core, delta_activations)

    if len(weight_target.shape) != 2:
        return TransplantDeltaResult(
            merged_weight=weight_target,
            applied=False,
            null_dim=0,
            delta_norm=0.0,
            filtered_norm=0.0,
            projection_loss=0.0,
            preserved_fraction=1.0,
        )

    out_dim = int(weight_target.shape[0])
    in_dim = int(weight_target.shape[1])
    n_core = int(activations_core.shape[0])

    logger.info(
        "ANCHOR-RELATIVE TRANSPLANT: weight=[%d, %d], n_core=%d, delta_A shape=%s",
        out_dim, in_dim, n_core, b.shape(delta_activations)
    )

    # Step 1: Compute boundary null-space projector N
    # N = I - pinv(A_boundary) @ A_boundary
    if boundary_activations is not None:
        boundary_activations = b.astype(b.array(boundary_activations), compute_dtype)
        b.eval(boundary_activations)
        n_boundary = int(boundary_activations.shape[0])

        if n_boundary > 0:
            # A_b is [m, in_dim], pinv(A_b) is [in_dim, m]
            A_b_pinv = b.pinv(boundary_activations)
            b.eval(A_b_pinv)

            # N = I - pinv(A_b) @ A_b
            # [in_dim, m] @ [m, in_dim] -> [in_dim, in_dim]
            proj_b = b.matmul(A_b_pinv, boundary_activations)
            b.eval(proj_b)

            N = b.eye(in_dim) - proj_b
            b.eval(N)

            logger.info(
                "ANCHOR-RELATIVE: Boundary null-space computed from %d samples",
                n_boundary
            )
        else:
            N = b.eye(in_dim)
            n_boundary = 0
    else:
        N = b.eye(in_dim)
        n_boundary = 0

    # Step 2: Compute unconstrained solution via closed-form normal equations
    # Solve A_core @ delta_W = delta_A for delta_W
    # A_core is [n, in_dim], delta_A is [n, out_dim]
    # Result: [in_dim, out_dim]
    delta_W_unc = gpu_lstsq(b, activations_core, delta_activations)
    b.eval(delta_W_unc)

    # Step 3: Project to boundary null-space
    # delta_W = N @ delta_W_unc
    # [in_dim, in_dim] @ [in_dim, out_dim] -> [in_dim, out_dim]
    delta_W = b.matmul(N, delta_W_unc)
    b.eval(delta_W)

    # Apply scale
    if delta_scale != 1.0:
        delta_W = delta_W * delta_scale
        b.eval(delta_W)

    # Step 4: Apply to weights
    # W' = W_target + delta_W.T
    # delta_W is [in_dim, out_dim], W is [out_dim, in_dim]
    merged_weight = weight_target + b.transpose(delta_W)
    if str(b.dtype(merged_weight)) != str(output_dtype):
        merged_weight = b.astype(merged_weight, output_dtype)
    b.eval(merged_weight)

    # Compute metrics using Euclidean norms
    # Weight space is flat (spectral structure, not manifold curvature).
    # Even for activations, geodesic on a single flattened vector is Euclidean.
    delta_W_flat = b.reshape(delta_W, (-1,))
    delta_W_unc_flat = b.reshape(delta_W_unc, (-1,))
    delta_A_flat = b.reshape(delta_activations, (-1,))

    delta_W_norm = float(b.to_scalar(b.sqrt(b.sum(delta_W_flat * delta_W_flat))))
    delta_W_unc_norm = float(b.to_scalar(b.sqrt(b.sum(delta_W_unc_flat * delta_W_unc_flat))))
    delta_A_norm = float(b.to_scalar(b.sqrt(b.sum(delta_A_flat * delta_A_flat))))

    if delta_W_unc_norm > 0:
        preserved_fraction = delta_W_norm / delta_W_unc_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    # Compute null-space dimension from boundary rank
    boundary_rank = 0
    if n_boundary > 0:
        _, S_b, _ = geodesic_svd(b, boundary_activations)
        b.eval(S_b)
        n_sv = int(S_b.shape[0])
        if n_sv > 0:
            max_s = float(b.to_scalar(b.max(S_b)))
            if max_s > 0.0:
                rank_scale = svd_rank_threshold(b, S_b, max(in_dim, n_boundary))
                threshold = max_s * rank_scale
                rank_mask = S_b > threshold
                rank_arr = b.sum(b.astype(rank_mask, "int32"))
                b.eval(rank_arr)
                boundary_rank = int(b.to_scalar(rank_arr))
    null_dim = max(0, in_dim - boundary_rank)

    logger.info(
        "ANCHOR-RELATIVE RESULT: delta_A_norm=%.4f, delta_W_norm=%.4f, "
        "preserved=%.1f%%, n_boundary=%d",
        delta_A_norm, delta_W_norm, 100.0 * preserved_fraction, n_boundary
    )

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=null_dim,
        delta_norm=delta_A_norm,
        filtered_norm=delta_W_norm,
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        delta_occupancy=None,
        birkhoff_applied=False,
        birkhoff_converged=False,
        birkhoff_iterations=0,
        birkhoff_spectral_clipped=False,
    )


@dataclass(frozen=True)
class CoreBoundaryPartition:
    core_indices: list[int]
    boundary_indices: list[int]
    core_probe_ids: list[str]
    boundary_probe_ids: list[str]


@dataclass(frozen=True)
class TransplantDeltaResult:
    merged_weight: Any
    applied: bool
    null_dim: int
    delta_norm: float
    filtered_norm: float
    projection_loss: float
    preserved_fraction: float
    delta_occupancy: Any | None = None
    # Birkhoff projection metrics (optional, populated when birkhoff_config is used)
    birkhoff_applied: bool = False
    birkhoff_converged: bool = False
    birkhoff_iterations: int = 0
    birkhoff_spectral_clipped: bool = False


def partition_core_boundary(
    activations: "Array",
    probe_ids: list[str],
    core_probe_ids: set[str],
    backend: "Backend | None" = None,
) -> CoreBoundaryPartition:
    """Partition probes into core and boundary sets (boundary = complement)."""
    b = backend or get_default_backend()
    points = b.array(activations)
    b.eval(points)

    n = int(points.shape[0])
    if n == 0 or not probe_ids:
        return CoreBoundaryPartition([], [], [], [])

    core_indices = [i for i, pid in enumerate(probe_ids) if pid in core_probe_ids]
    core_set = set(core_indices)
    if not core_indices:
        return CoreBoundaryPartition([], [], [], [])

    boundary_list = [i for i in range(n) if i not in core_set]
    return CoreBoundaryPartition(
        core_indices=core_indices,
        boundary_indices=boundary_list,
        core_probe_ids=[probe_ids[i] for i in core_indices],
        boundary_probe_ids=[probe_ids[i] for i in boundary_list],
    )


def compute_transplant_delta(
    weight_target: "Array",
    activations_core: "Array",
    delta_activations: "Array",
    boundary_activations: "Array | None" = None,
    backend: "Backend | None" = None,
    delta_scale: float = 1.0,
) -> TransplantDeltaResult:
    """Compute weight update via constrained least-squares.

    Solves:
        min ||A_core @ delta_W - delta_A_core||_F²
        s.t. A_boundary @ delta_W = 0

    Solution via null-space projection:
        N = I - pinv(A_boundary) @ A_boundary  (boundary null-space)
        delta_W_unc = pinv(A_core) @ delta_A_core  (unconstrained)
        delta_W = N @ delta_W_unc  (projected to boundary null-space)
        W' = W_target + delta_W.T

    This preserves boundary outputs within numerical precision while core outputs
    move toward the source activations.

    Args:
        weight_target: Target model weights to modify [out_dim, in_dim].
        activations_core: Core activation samples [n_core, in_dim].
        delta_activations: Desired change in activation space [n_core, out_dim].
            Computed upstream via anchor-relative grafting.
        boundary_activations: Boundary samples [n_boundary, in_dim]. If provided,
            their outputs are constrained to match: A_boundary @ W' = A_boundary @ W.
        backend: Optional Backend for GPU operations.
        delta_scale: Scale factor for delta (default 1.0).

    Returns:
        TransplantDeltaResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()

    return _compute_transplant_delta_anchor_relative(
        weight_target=weight_target,
        activations_core=activations_core,
        delta_activations=delta_activations,
        boundary_activations=boundary_activations,
        delta_scale=delta_scale,
        backend=b,
    )
