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
from sys import float_info
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.behavioral_norm import behavioral_norm
from modelcypher.core.domain.geometry.marchenko_pastur import (
    marchenko_pastur_noise_edge,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    gpu_lstsq,
    machine_epsilon,
    orthogonalize_alignment,
    orthogonalize_alignment_full,
    precision_dtype,
    svd_rank_threshold,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        TrajectoryResult,
        TrajectoryTangentResult,
    )
    from modelcypher.ports.backend import Array, Backend

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
    if len(shape) == 2 and shape[0] < 1:
        return 0.0

    # backend.norm() computes Frobenius norm for matrices
    frob_norm = b.norm(weight)
    b.eval(frob_norm)
    return float(b.to_scalar(frob_norm))




def compute_joint_mlp_scale(
    scale_gate: float,
    scale_up: float,
    scale_down: float,
) -> tuple[float, float, float]:
    """Compute joint scale corrections for composed MLP function.

    SwiGLU MLP computes: output = down(SiLU(gate(x)) * up(x))

    When gate/up/down weights are scaled independently, the composed function
    is distorted because:
    - SiLU(s_gate * x) ≈ s_gate * x for small x (linear regime)
    - SiLU(s_gate * x) ≈ s_gate * x / 2 for large x (saturated regime)
    - The gate and up multiply: (s_gate * gate_out) * (s_up * up_out)
    - Then down amplifies/shrinks: s_down * down(...)

    For the composed function to preserve magnitude, we need:
        s_down * s_gate * s_up ≈ 1 (geometric balance)

    The fix: compute a JOINT scale correction that distributes evenly across
    all three weights, preserving the geometric mean while correcting the
    individual extreme values (e.g., 0.148 and 19.64).

    Args:
        scale_gate: Scale correction computed for gate_proj (w1)
        scale_up: Scale correction computed for up_proj (w3)
        scale_down: Scale correction computed for down_proj (w2)

    Returns:
        Tuple of (corrected_gate, corrected_up, corrected_down) scales.
        Each is the ratio: joint_scale / individual_scale, to multiply
        the already-scaled weight by to get the correct joint behavior.

    Example:
        >>> scale_gate = 0.148  # 14.8x smaller
        >>> scale_up = 1.0
        >>> scale_down = 19.64  # 1964x larger
        >>> g, u, d = compute_joint_mlp_scale(scale_gate, scale_up, scale_down)
        >>> # joint = (0.148 * 1.0 * 19.64)^(1/3) ≈ 1.42
        >>> # g ≈ 1.42 / 0.148 ≈ 9.6 (upscale gate)
        >>> # d ≈ 1.42 / 19.64 ≈ 0.07 (downscale down)
    """
    eps = float_info.epsilon

    # Compute geometric mean of the three scales
    # This is the "balanced" scale that would give neutral composed behavior
    product = abs(scale_gate) * abs(scale_up) * abs(scale_down)
    joint_scale = product ** (1.0 / 3.0) if product > eps else 1.0

    # Compute correction factor for each weight
    # The weight has already been scaled by scale_X, so we need to
    # multiply by (joint_scale / scale_X) to get the correct joint behavior
    corrected_gate = joint_scale / max(abs(scale_gate), eps)
    corrected_up = joint_scale / max(abs(scale_up), eps)
    corrected_down = joint_scale / max(abs(scale_down), eps)

    logger.info(
        "JOINT MLP SCALE: gate=%.4f, up=%.4f, down=%.4f → joint=%.4f → "
        "corrections: gate=%.4f, up=%.4f, down=%.4f",
        scale_gate, scale_up, scale_down, joint_scale,
        corrected_gate, corrected_up, corrected_down,
    )

    return corrected_gate, corrected_up, corrected_down


@dataclass(frozen=True)
class GateDistributionResult:
    """Result of gate distribution check.

    Tracks whether alignment preserves the ON/OFF gating pattern.

    Attributes:
        source_on_fraction: Fraction of source gate activations > 0 (per-sample mean).
        aligned_on_fraction: Fraction of aligned source gate activations > 0.
        target_on_fraction: Fraction of target gate activations > 0.
        divergence: Absolute difference between aligned and target ON fractions.
        divergence_threshold: Threshold used for compatibility check.
            If not provided by caller, derived from sampling variance.
        flipped_fraction: Fraction of dimensions where ON/OFF state flipped.
        is_compatible: True if divergence < threshold.
    """

    source_on_fraction: float
    aligned_on_fraction: float
    target_on_fraction: float
    divergence: float
    divergence_threshold: float
    flipped_fraction: float
    is_compatible: bool


def check_gate_distribution(
    source_gate_activations: "Array",
    target_gate_activations: "Array",
    alignment: "Array",
    backend: "Backend | None" = None,
    divergence_threshold: float | None = None,
) -> GateDistributionResult:
    """Check if alignment preserves the gate's ON/OFF distribution.

    The SwiGLU gate uses SiLU(x) which is ~0 for x < 0 and ~x for x > 0.
    This creates a binary decision: each dimension is either ON (x > 0) or OFF.

    When we align source gate activations to target space, we need to verify
    that the ON/OFF pattern is preserved. If alignment flips many dimensions
    from ON to OFF (or vice versa), the merged model will make wrong gating
    decisions.

    Args:
        source_gate_activations: Gate activations from source [n_samples, d_source].
        target_gate_activations: Gate activations from target [n_samples, d_target].
        alignment: Alignment transform F [d_source, d_target].
        backend: Compute backend.
        divergence_threshold: Optional max allowed divergence. If omitted,
            threshold is derived from observed Bernoulli sampling variance:
            ``sqrt((var_aligned + var_target) / (n_samples * n_dims))``.

    Returns:
        GateDistributionResult with distribution metrics.

    Example:
        >>> result = check_gate_distribution(source_gate, target_gate, F)
        >>> if not result.is_compatible:
        ...     logger.warning("Gate distribution divergence: %.1f%%", result.divergence * 100)
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()

    source_acts = b.array(source_gate_activations)
    target_acts = b.array(target_gate_activations)
    F = b.array(alignment)
    b.eval(source_acts, target_acts, F)

    # Compute aligned source activations
    aligned_source = b.matmul(source_acts, F)
    b.eval(aligned_source)

    # Compute ON fraction per dimension, then average
    # ON = activation > 0
    zero = b.zeros_like(source_acts)
    source_on_mask = source_acts > zero
    source_on_per_dim = b.mean(b.astype(source_on_mask, b.dtype(source_acts)), axis=0)
    source_on_fraction = float(b.to_scalar(b.mean(source_on_per_dim)))
    b.eval(source_on_per_dim)

    zero_aligned = b.zeros_like(aligned_source)
    aligned_on_mask = aligned_source > zero_aligned
    aligned_on_per_dim = b.mean(b.astype(aligned_on_mask, b.dtype(aligned_source)), axis=0)
    aligned_on_fraction = float(b.to_scalar(b.mean(aligned_on_per_dim)))
    b.eval(aligned_on_per_dim)

    zero_target = b.zeros_like(target_acts)
    target_on_mask = target_acts > zero_target
    target_on_per_dim = b.mean(b.astype(target_on_mask, b.dtype(target_acts)), axis=0)
    target_on_fraction = float(b.to_scalar(b.mean(target_on_per_dim)))
    b.eval(target_on_per_dim)

    # Divergence: how far aligned distribution is from target
    divergence = abs(aligned_on_fraction - target_on_fraction)

    # Flipped fraction: dimensions where aligned is mostly ON but target is mostly OFF
    # (or vice versa) - this indicates a structural incompatibility
    # Bernoulli ON/OFF states are encoded as {0, 1}; under symmetric loss,
    # the Bayes decision boundary is the midpoint between these states.
    state_midpoint = (0.0 + 1.0) / 2.0
    aligned_mostly_on = aligned_on_per_dim > state_midpoint
    target_mostly_on = target_on_per_dim > state_midpoint

    # XOR to find flipped dimensions
    aligned_bool = b.astype(aligned_mostly_on, b.dtype(aligned_on_per_dim))
    target_bool = b.astype(target_mostly_on, b.dtype(target_on_per_dim))
    flipped = b.abs(aligned_bool - target_bool)
    flipped_fraction = float(b.to_scalar(b.mean(flipped)))

    resolved_divergence_threshold = divergence_threshold
    if resolved_divergence_threshold is None:
        n_samples = int(source_acts.shape[0]) if len(source_acts.shape) > 0 else 0
        n_dims = int(aligned_on_per_dim.shape[0]) if len(aligned_on_per_dim.shape) > 0 else 0
        if n_samples > 0 and n_dims > 0:
            one_aligned = b.ones_like(aligned_on_per_dim)
            one_target = b.ones_like(target_on_per_dim)
            aligned_var_terms = aligned_on_per_dim * (one_aligned - aligned_on_per_dim)
            target_var_terms = target_on_per_dim * (one_target - target_on_per_dim)
            b.eval(aligned_var_terms, target_var_terms)
            aligned_var_mean = float(b.to_scalar(b.mean(aligned_var_terms)))
            target_var_mean = float(b.to_scalar(b.mean(target_var_terms)))
            denom = float(n_samples * n_dims)
            resolved_divergence_threshold = (
                (aligned_var_mean + target_var_mean) / denom
            ) ** 0.5
        else:
            resolved_divergence_threshold = 0.0

    is_compatible = divergence < resolved_divergence_threshold

    logger.info(
        "GATE DISTRIBUTION: source_on=%.2f, aligned_on=%.2f, target_on=%.2f, "
        "divergence=%.3f, threshold=%.6f, flipped=%.1f%%, compatible=%s",
        source_on_fraction, aligned_on_fraction, target_on_fraction,
        divergence, resolved_divergence_threshold, 100 * flipped_fraction, is_compatible,
    )

    if not is_compatible:
        logger.warning(
            "GATE DISTRIBUTION DIVERGENCE: aligned_on=%.2f differs from target_on=%.2f "
            "by %.1f%% (threshold: %.1f%%). Consider reducing delta_scale.",
            aligned_on_fraction, target_on_fraction,
            100 * divergence, 100 * resolved_divergence_threshold,
        )

    return GateDistributionResult(
        source_on_fraction=source_on_fraction,
        aligned_on_fraction=aligned_on_fraction,
        target_on_fraction=target_on_fraction,
        divergence=divergence,
        divergence_threshold=resolved_divergence_threshold,
        flipped_fraction=flipped_fraction,
        is_compatible=is_compatible,
    )


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
        scale_correction: Scale factor from behavioral reconstruction.
            For MLP weights (gate/up/down), use compute_joint_mlp_scale() to
            correct for the composed function instead of individual weights.
    """

    merged_weight: "Array"
    delta_norm: float
    projected_norm: float
    preserved_fraction: float
    transfer_strength: float
    null_rank: int
    scale_correction: float = 1.0


@dataclass(frozen=True)
class NullSpaceProjector:
    """Precomputed null-space projector with density-derived transfer strength."""

    weighted_activations: "Array"
    gram_inv: "Array"
    null_rank: int
    transfer_strength: float
    projector: "Array | None" = None
    vectorized: bool = False


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
        scale_correction: Scale factor applied from orthogonalization (scale_out/scale_in).
            For MLP weights, this should be composed jointly, not applied independently.
    """

    reconstructed_weight: "Array"
    reconstruction_error: float
    source_behavior_norm: float
    target_behavior_norm: float
    condition_number: float
    scale_correction: float = 1.0


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
        "BEHAVIORAL RESULT: error=%.6f, src_norm=%.2f, tgt_norm=%.2f, cond=%.2e, scale=%.4f",
        reconstruction_error, source_norm, target_norm, condition_number, scale_correction
    )

    return BehavioralReconstructionResult(
        reconstructed_weight=reconstructed_weight,
        reconstruction_error=reconstruction_error,
        source_behavior_norm=source_norm,
        target_behavior_norm=target_norm,
        condition_number=condition_number,
        scale_correction=scale_correction,
    )


def reconstruct_weight_spectral_corrected(
    source_weight: "Array",
    input_activations_source: "Array",
    alignment_in: "Array",
    alignment_out: "Array",
    backend: "Backend | None" = None,
) -> BehavioralReconstructionResult:
    """Reconstruct weight with per-direction scale correction.

    Unlike reconstruct_weight_from_behavior() which uses a single scalar
    scale correction (mean singular value ratio), this function applies
    direction-dependent scaling based on the full singular value spectra.

    Mathematical approach:
        1. Compute full SVD of alignments: F_in = U_in @ S_in @ Vt_in
        2. Extract orthogonal parts: R_in = U_in @ Vt_in (rotation)
        3. Compute per-direction scale ratios: ratio[i] = S_out[i] / S_in[i]
        4. Reconstruct in target coords: W.T = lstsq(input_tgt, output_tgt)
        5. Apply per-direction correction in singular basis

    This preserves more spectral information than the scalar approach,
    which is especially important for:
        - Bottleneck layers (high condition number, σ₁ >> σₙ)
        - Cross-architecture merges (different internal structure)
        - Layers with heterogeneous scaling across directions

    Args:
        source_weight: Source weight [out_src, in_src].
        input_activations_source: Input activations [n, in_src].
        alignment_in: Input alignment transform [in_src, in_tgt].
        alignment_out: Output alignment transform [out_src, out_tgt].
        backend: Compute backend.

    Returns:
        BehavioralReconstructionResult with reconstructed weight.
        The scale_correction field contains the geometric mean of ratios
        for compatibility, but the actual correction is per-direction.
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
        "SPECTRAL RECONSTRUCTION: [%d, %d] -> [%d, %d], n_samples=%d",
        out_src, in_src, out_tgt, in_tgt, n_samples
    )

    # Step 1: Get full SVD of alignments (not just orthogonal + scalar)
    U_in, S_in, _, Vt_in = orthogonalize_alignment_full(alignment_in, b)
    U_out, S_out, _, Vt_out = orthogonalize_alignment_full(alignment_out, b)
    b.eval(U_in, S_in, U_out, S_out)

    n_sv_in = int(S_in.shape[0])
    n_sv_out = int(S_out.shape[0])

    # Compute per-direction scale ratios
    eps = float(division_epsilon(b, S_in))

    # For same-dimension case, S_in and S_out have same length
    # For cross-dimension, we need to handle length mismatch
    n_sv_common = min(n_sv_in, n_sv_out)

    if n_sv_common > 0:
        S_in_common = S_in[:n_sv_common]
        S_out_common = S_out[:n_sv_common]
        S_in_safe = b.maximum(S_in_common, b.full(S_in_common.shape, eps))
        S_ratio = S_out_common / S_in_safe
        b.eval(S_ratio)

        # Geometric mean for logging (compatible with old interface)
        log_ratio = b.log(b.maximum(S_ratio, b.full(S_ratio.shape, eps)))
        mean_log = b.mean(log_ratio)
        b.eval(mean_log)
        geom_mean_ratio = float(b.to_scalar(b.exp(mean_log)))

        # Compute statistics for logging
        min_ratio = float(b.to_scalar(b.min(S_ratio)))
        max_ratio = float(b.to_scalar(b.max(S_ratio)))
        ratio_spread = max_ratio / max(min_ratio, eps)
    else:
        geom_mean_ratio = 1.0
        min_ratio = 1.0
        max_ratio = 1.0
        ratio_spread = 1.0

    logger.info(
        "SPECTRAL SCALES: n_sv=%d, ratio_range=[%.4f, %.4f], spread=%.2fx, geom_mean=%.4f",
        n_sv_common, min_ratio, max_ratio, ratio_spread, geom_mean_ratio
    )

    # Step 2: Compute source layer behavior
    output_source = b.matmul(input_activations_source, b.transpose(source_weight))
    b.eval(output_source)

    # Step 3: Project to target coordinates using ORTHOGONALIZED transforms
    input_target = b.matmul(input_activations_source, U_in)
    output_target = b.matmul(output_source, U_out)
    b.eval(input_target, output_target)

    # Step 4: Solve for weight via least squares
    W_T = gpu_lstsq(b, input_target, output_target)  # [in_tgt, out_tgt]
    b.eval(W_T)

    reconstructed_weight_raw = b.transpose(W_T)  # [out_tgt, in_tgt]
    b.eval(reconstructed_weight_raw)

    # Step 5: Apply per-direction scale correction
    # The correction operates in the singular basis of the input alignment.
    # W_corrected = W_raw @ Vt_in.T @ diag(S_ratio) @ Vt_in
    # This scales each direction by its specific ratio, not the mean.

    if n_sv_common > 0 and ratio_spread > 1.0 + eps:
        # Only apply per-direction correction if there's meaningful spread
        # Project weight columns to singular basis, scale, project back
        # W @ V.T -> singular basis -> scale -> V @ ... back to original

        # W_raw is [out_tgt, in_tgt], Vt_in is [k, in_tgt]
        # W_in_sv = W_raw @ Vt_in.T  -> [out_tgt, k]
        Vt_in_common = Vt_in[:n_sv_common, :]  # [k, in_tgt]
        V_in_common = b.transpose(Vt_in_common)  # [in_tgt, k]
        b.eval(V_in_common)

        W_in_sv = b.matmul(reconstructed_weight_raw, V_in_common)  # [out_tgt, k]
        b.eval(W_in_sv)

        # Scale each column by S_ratio[i]
        # W_in_sv_scaled[j, i] = W_in_sv[j, i] * S_ratio[i]
        S_ratio_row = b.reshape(S_ratio, (1, n_sv_common))  # [1, k]
        W_in_sv_scaled = W_in_sv * S_ratio_row  # [out_tgt, k]
        b.eval(W_in_sv_scaled)

        # Project back: W_corrected_partial = W_in_sv_scaled @ Vt_in_common
        W_corrected_partial = b.matmul(W_in_sv_scaled, Vt_in_common)  # [out_tgt, in_tgt]
        b.eval(W_corrected_partial)

        # The residual (directions beyond common singular values) keeps original scale
        # Compute residual = W_raw - W_raw @ V @ V.T (the part not in V's span)
        W_in_V_span = b.matmul(W_in_sv, Vt_in_common)  # [out_tgt, in_tgt]
        W_residual = reconstructed_weight_raw - W_in_V_span
        b.eval(W_residual)

        # Final: corrected part + residual
        reconstructed_weight = W_corrected_partial + W_residual
        b.eval(reconstructed_weight)

        logger.info(
            "PER-DIRECTION SCALE: applied %d directional corrections (spread=%.2fx)",
            n_sv_common, ratio_spread
        )
    else:
        # Fall back to scalar correction if spread is negligible
        if abs(geom_mean_ratio - 1.0) > eps:
            reconstructed_weight = reconstructed_weight_raw * geom_mean_ratio
            logger.info(
                "SCALAR SCALE: spread=%.2fx too small, using geom_mean=%.4f",
                ratio_spread, geom_mean_ratio
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
    eps_cond = float(machine_epsilon(b, gram))
    max_eig = float(b.to_scalar(b.max(eigvals)))
    min_eig = float(b.to_scalar(b.min(b.abs(eigvals))))
    condition_number = max_eig / max(min_eig, eps_cond)

    logger.info(
        "SPECTRAL RESULT: error=%.6f, src_norm=%.2f, tgt_norm=%.2f, cond=%.2e, geom_scale=%.4f",
        reconstruction_error, source_norm, target_norm, condition_number, geom_mean_ratio
    )

    return BehavioralReconstructionResult(
        reconstructed_weight=reconstructed_weight,
        reconstruction_error=reconstruction_error,
        source_behavior_norm=source_norm,
        target_behavior_norm=target_norm,
        condition_number=condition_number,
        scale_correction=geom_mean_ratio,  # Report geom mean for compatibility
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
        "MANIFOLD RESULT: full_error=%.6f, manifold_error=%.10f, rank=%d, cond=%.2e, scale=%.4f",
        reconstruction_error, manifold_error, intrinsic_rank, condition_number, scale_correction
    )

    return BehavioralReconstructionResult(
        reconstructed_weight=reconstructed_weight,
        reconstruction_error=manifold_error,  # Report manifold error as the true error
        source_behavior_norm=source_norm,
        target_behavior_norm=target_norm,
        condition_number=condition_number,
        scale_correction=scale_correction,
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

    Returns:
        WeightSpaceTransplantResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()

    target_weight = b.array(target_weight)
    input_activations_target = b.array(input_activations_target)
    output_dtype = b.dtype(target_weight)

    # Step 1: Reconstruct source weight in target coordinates
    # Use manifold-aware reconstruction for near-zero error
    reconstruction = reconstruct_weight_manifold_aware(
        source_weight=source_weight,
        input_activations_source=input_activations_source,
        alignment_in=alignment_in,
        alignment_out=alignment_out,
        backend=b,
    )

    source_behavioral = reconstruction.reconstructed_weight
    b.eval(source_behavioral)

    # Capture scale correction for potential MLP joint scaling
    scale_correction = reconstruction.scale_correction

    logger.info(
        "CROSS-DIM TRANSPLANT: reconstruction_error=%.6f, condition=%.2e, scale=%.4f",
        reconstruction.reconstruction_error,
        reconstruction.condition_number,
        scale_correction,
    )

    # Step 2: Compute delta between behavioral source and target
    compute_dtype = precision_dtype(b, reference=target_weight)
    source_behavioral = b.astype(source_behavioral, compute_dtype)
    target_weight_compute = b.astype(target_weight, compute_dtype)
    b.eval(source_behavioral, target_weight_compute)

    delta_W = source_behavioral - target_weight_compute
    b.eval(delta_W)

    # Use BEHAVIORAL norm (not Frobenius) to measure actual output change
    delta_norm = behavioral_norm(delta_W, input_activations_target, b)

    # =========================================================================
    # NULL-SPACE PROJECTION (variance-based, activation-defined)
    # =========================================================================
    source_density = None
    if source_activations_for_density is not None:
        source_acts = b.array(source_activations_for_density)
        source_dim = int(source_acts.shape[1])
        target_dim = int(input_activations_target.shape[1])
        if source_dim != target_dim:
            logger.info(
                "CROSS-DIM: Skipping source density (dim %d != target dim %d)",
                source_dim, target_dim
            )
        else:
            source_density = source_activations_for_density

    null_space_projector = compute_null_space_projector(
        input_activations=input_activations_target,
        source_activations_for_density=source_density,
        target_activations_for_density=target_activations_for_density,
        backend=b,
    )

    N = null_space_projector.projector
    null_rank = null_space_projector.null_rank
    transfer_strength = null_space_projector.transfer_strength

    if N is None:
        A_weighted = null_space_projector.weighted_activations
        gram_inv = null_space_projector.gram_inv
        b.eval(A_weighted, gram_inv)

        delta_row = b.matmul(delta_W, b.transpose(A_weighted))
        correction = b.matmul(delta_row, gram_inv)
        correction = b.matmul(correction, A_weighted)
        delta_W_proj = delta_W - correction
        b.eval(delta_W_proj)
    else:
        delta_W_proj = b.matmul(delta_W, N)
        b.eval(delta_W_proj)

    projected_norm = behavioral_norm(delta_W_proj, input_activations_target, b)

    eps = float(division_epsilon(b, delta_W))
    if delta_norm > eps:
        preserved_fraction = projected_norm / delta_norm
    else:
        preserved_fraction = 1.0

    merged_weight = target_weight_compute + delta_scale * delta_W_proj
    if str(b.dtype(merged_weight)) != str(output_dtype):
        merged_weight = b.astype(merged_weight, output_dtype)
    b.eval(merged_weight)

    logger.info(
        "CROSS-DIM RESULT: delta_norm=%.4f, proj_norm=%.4f, preserved=%.1f%%, null_rank=%d",
        delta_norm, projected_norm, 100.0 * preserved_fraction, null_rank
    )

    return WeightSpaceTransplantResult(
        merged_weight=merged_weight,
        delta_norm=delta_norm,
        projected_norm=projected_norm,
        preserved_fraction=preserved_fraction,
        transfer_strength=transfer_strength,
        null_rank=null_rank,
        scale_correction=scale_correction,
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

    max_eig_arr = b.max(eigvals_pos)
    b.eval(max_eig_arr)
    max_eig = float(b.to_scalar(max_eig_arr))

    # =========================================================================
    # TIKHONOV MODE: MP noise edge + continuous weights
    # =========================================================================
    # The MP noise edge separates signal eigenvalues (population structure)
    # from noise eigenvalues (finite-sample artifact). With N << D, this is
    # orders of magnitude above the IEEE √ε threshold.
    #
    # Tikhonov weights w_i = λ_i / (λ_i + α) give continuous [0,1] per
    # direction: λ >> α → w ≈ 1 (occupied), λ << α → w ≈ 0 (noise/available).
    # No integer rank. No cliff.
    #
    # Citation: Marchenko & Pastur (1967), Tikhonov (1963).
    # =========================================================================
    mp_edge = marchenko_pastur_noise_edge(total_var_val, gram_size, n_samples)
    tikhonov_w = eigvals_pos / (eigvals_pos + mp_edge)
    b.eval(tikhonov_w)

    # Effective rank = sum of Tikhonov weights (continuous, not integer)
    eff_rank_arr = b.sum(tikhonov_w)
    b.eval(eff_rank_arr)
    activation_rank = int(round(float(b.to_scalar(eff_rank_arr))))
    activation_rank = max(0, min(activation_rank, gram_size))
    null_rank = max(0, in_dim - activation_rank)

    logger.info(
        "NULL-SPACE [tikhonov]: eff_rank=%.1f/%d, null_rank=%d, max_eig=%.3e, "
        "mp_edge=%.3e, sigma_sq=%.3e",
        float(b.to_scalar(eff_rank_arr)), gram_size, null_rank, max_eig,
        mp_edge, total_var_val / gram_size,
    )

    if use_feature_space:
        # =====================================================================
        # FEATURE SPACE: Build explicit d×d projector (MP-weighted)
        # =====================================================================
        V_weighted = eigvecs * b.reshape(tikhonov_w, (1, -1))
        b.eval(V_weighted)
        row_space_proj = b.matmul(V_weighted, b.transpose(eigvecs))
        b.eval(row_space_proj)

        I_d = b.eye(in_dim, dtype=compute_dtype)
        null_projector = I_d - row_space_proj
        b.eval(null_projector)

        return NullSpaceProjector(
            weighted_activations=A_weighted,
            gram_inv=null_projector,
            null_rank=null_rank,
            transfer_strength=transfer_strength,
            projector=null_projector,
        )
    else:
        # =====================================================================
        # SAMPLE SPACE: Tikhonov-regularized pseudoinverse
        # =====================================================================
        eigvals_inv = 1.0 / (eigvals_pos + mp_edge)
        b.eval(eigvals_inv)

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


def compute_behavior_jacobian_projector(
    gradient_matrix: "Array",
    *,
    backend: "Backend | None" = None,
) -> NullSpaceProjector:
    """Compute null-space projector from per-probe behavior Jacobian.

    Given G ∈ ℝ^{N × D} where each row is the flattened gradient
    ∂CE(probe_i)/∂vec(W) for a single preservation probe, compute
    the projector onto the null space of G:

        P = I - G^T (G G^T)^+ G

    This projects weight deltas into directions that produce zero output
    change on the preservation set (behavior-null directions).

    Always works in probe space (N × N) since N << D (number of probes
    is much smaller than the flattened weight dimension).

    Args:
        gradient_matrix: [N, D] matrix where row i is the flattened
            per-probe CE gradient for probe i w.r.t. a weight matrix.
            D = out_dim * in_dim for the target weight.
        backend: Backend for tensor operations.

    Returns:
        NullSpaceProjector with gram_inv = (G G^T)^+ and
        weighted_activations = G. The implicit projection formula is:
        delta_proj = delta - G^T @ (G G^T)^+ @ G @ delta
    """
    b = backend or get_default_backend()

    G = b.array(gradient_matrix)
    compute_dtype = precision_dtype(b, reference=G)
    G = b.astype(G, compute_dtype)
    b.eval(G)

    n_probes = int(G.shape[0])
    D = int(G.shape[1])

    if n_probes == 0:
        raise ValueError(
            "Cannot compute behavior Jacobian projector with 0 probes.",
        )

    # Always work in probe space: G G^T ∈ ℝ^{N×N}
    GGt = b.matmul(G, b.transpose(G))
    b.eval(GGt)

    eps = machine_epsilon(b, GGt)

    # Eigendecompose G G^T
    eigvals, eigvecs = b.eigh(GGt)
    b.eval(eigvals, eigvecs)

    # Sort descending
    n_eigs = int(eigvals.shape[0])
    idx = b.arange(n_eigs - 1, -1, -1)
    eigvals = b.take(eigvals, idx, axis=0)
    eigvecs = b.take(eigvecs, idx, axis=1)
    b.eval(eigvals, eigvecs)

    eigvals_pos = b.maximum(eigvals, eps)

    # Numerical rank
    max_eig_arr = b.max(eigvals_pos)
    b.eval(max_eig_arr)
    max_eig = float(b.to_scalar(max_eig_arr))
    max_eig_safe = max(max_eig, eps)

    rank_scale = svd_rank_threshold(b, eigvals_pos, n_probes)
    rank_threshold = max_eig_safe * rank_scale
    rank_mask = eigvals_pos > rank_threshold
    rank_mask_float = b.astype(rank_mask, compute_dtype)

    activation_rank_arr = b.sum(rank_mask_float)
    b.eval(activation_rank_arr)
    activation_rank = int(round(float(b.to_scalar(activation_rank_arr))))
    activation_rank = max(0, min(activation_rank, n_probes))
    null_rank = max(0, D - activation_rank)

    logger.info(
        "BEHAVIOR-JACOBIAN: n_probes=%d, weight_dim=%d, "
        "jacobian_rank=%d, null_rank=%d, max_eig=%.3e",
        n_probes, D, activation_rank, null_rank, max_eig,
    )

    # Pseudoinverse of G G^T via eigendecomposition
    eps_pinv = division_epsilon(b, eigvals_pos)
    pinv_threshold = eps * n_eigs * max_eig_safe
    pinv_mask = eigvals_pos > pinv_threshold
    eigvals_inv = b.where(
        pinv_mask,
        1.0 / (eigvals_pos + eps_pinv),
        b.zeros_like(eigvals_pos),
    )
    b.eval(eigvals_inv)

    # (G G^T)^+ = V @ diag(1/λ) @ V^T
    V_scaled = eigvecs * b.reshape(eigvals_inv, (1, -1))
    b.eval(V_scaled)
    GGt_inv = b.matmul(V_scaled, b.transpose(eigvecs))
    b.eval(GGt_inv)

    return NullSpaceProjector(
        weighted_activations=G,
        gram_inv=GGt_inv,
        null_rank=null_rank,
        transfer_strength=1.0,
        vectorized=True,
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
            Takes precedence over all other projectors if provided.
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
    delta_norm = behavioral_norm(delta_W, input_activations, b)
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
            G = null_space_projector.weighted_activations
            gram_inv = null_space_projector.gram_inv
            b.eval(G, gram_inv)

            if null_space_projector.vectorized:
                # Behavior Jacobian: G is [N, out*in], project in flattened weight space
                # delta_flat = vec(delta_W)  [1, D]
                # proj = delta_flat - (delta_flat @ G.T) @ (G G.T)^+ @ G
                delta_flat = b.reshape(delta_W, (1, -1))
                delta_row = b.matmul(delta_flat, b.transpose(G))
                correction = b.matmul(delta_row, gram_inv)
                correction = b.matmul(correction, G)
                delta_W_proj = b.reshape(
                    delta_flat - correction, (out_dim, in_dim),
                )
            else:
                # Activation covariance: G is [N, in_dim], project per output row
                # delta_W_proj = delta_W - (delta_W @ A.T) @ (A A.T)^+ @ A
                delta_row = b.matmul(delta_W, b.transpose(G))
                correction = b.matmul(delta_row, gram_inv)
                correction = b.matmul(correction, G)
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
    projected_norm = behavioral_norm(delta_W_proj, input_activations, b)
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
    # Denominator depends on projector type: activation-covariance projects
    # in in_dim, behavior Jacobian projects in out_dim * in_dim.
    _is_vectorized = (
        null_space_projector is not None and null_space_projector.vectorized
    ) if trajectory_tangent_projector is None else False
    null_rank_denom = out_dim * in_dim if _is_vectorized else in_dim
    logger.info(
        "BEHAVIORAL TRANSFER: behavioral_before=%.4f, behavioral_after=%.4f, "
        "preserved=%.2f%%, null_rank=%d/%d (capacity=%.1f%%)",
        delta_norm, projected_norm, 100 * preserved_fraction,
        null_rank, null_rank_denom,
        100 * null_rank / null_rank_denom if null_rank_denom > 0 else 0,
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
        scale_correction=1.0,  # Same-dim transplant: no reconstruction, no scaling
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
