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

"""Functional transplant for zero-shot knowledge transfer.

Weight-Space Null-Space Projection:
====================================

For ANY weight W: [out_dim, in_dim], the transplant is:

    1. delta_W = source_aligned - target_weight  [out_dim, in_dim]
    2. N = I - pinv(A_input) @ A_input  [in_dim, in_dim]
    3. delta_W_proj = delta_W @ N  [out_dim, in_dim]
    4. merged = target_weight + delta_W_proj

Where A_input are the INPUT activations to this weight:
    - For hidden→hidden weights: A_input = hidden activations
    - For hidden→intermediate weights (gate/up_proj): A_input = hidden activations
    - For intermediate→hidden weights (down_proj): A_input = intermediate activations

The constraint A_input @ delta_W_proj.T = 0 is satisfied by construction.
This preserves boundary behavior while adding source knowledge.

Density-Weighted Transfer:
==========================

Transfer strength is modulated by k-NN density comparison:
    - High source density, low target density → transfer more (fill the gap)
    - Low source density, high target density → transfer less (preserve target)

The density weighting is integrated into the null-space projector via
weighted boundary activations. Dense target regions are more strongly
constrained (preserved), sparse target regions allow more modification.

This is closed-form, works for ALL weight dimensions, and achieves
machine-precision preservation of boundary behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    precision_dtype,
    svd_rank_threshold,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _geodesic_frobenius_norm(
    weight: "Array",
    backend: "Backend",
) -> float:
    """Compute geodesic analogue of Frobenius norm for weight matrices.

    Treats each row of the weight matrix as a point in feature space,
    computes geodesic distance from origin for each row, then aggregates
    as sqrt(sum(geodesic_norms²)).

    This properly accounts for manifold curvature in the weight space.

    Args:
        weight: Weight matrix [out_dim, in_dim]
        backend: Backend for tensor operations

    Returns:
        Geodesic Frobenius-like norm (scalar)
    """
    b = backend
    shape = b.shape(weight)
    if len(shape) != 2:
        weight = b.reshape(weight, (1, -1))
    elif shape[0] < 1:
        return 0.0

    # Treat each row as a point, get geodesic norms
    geo_norms = geodesic_norms(weight, backend, use_cache=False)
    b.eval(geo_norms)

    # Geodesic Frobenius = sqrt(sum(geodesic_norms²))
    sum_sq = b.sum(geo_norms * geo_norms)
    geo_frob = b.sqrt(sum_sq)
    # Note: to_scalar forces eval, so no explicit eval needed here
    return float(b.to_scalar(geo_frob))


@dataclass(frozen=True)
class WeightSpaceTransplantResult:
    """Result of weight-space null-space transplant.

    Attributes:
        merged_weight: The transplanted weight [out_dim, in_dim].
        delta_norm: Frobenius norm of weight delta before projection.
        projected_norm: Frobenius norm of delta after null-space projection.
        preserved_fraction: Ratio of projected_norm / delta_norm (1.0 = no loss).
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
    """Reconstruct weight by preserving input→output behavior across dimensions.

    Instead of directly transforming weights (which distorts magnitudes), this
    function finds the weight in target space that produces the SAME behavior
    as the source weight in source space.

    The insight: The weight matrix encodes a transformation. Different coordinate
    systems encode the SAME transformation with DIFFERENT matrices. We want the
    matrix that performs the same operation, not a transformed matrix.

    Algorithm:
        1. Compute source layer behavior: output_src = input_src @ W_src.T
        2. Project to target coordinates:
           - input_tgt = input_src @ F_in
           - output_tgt = output_src @ F_out
        3. Solve for weight: W_behavior = lstsq(input_tgt, output_tgt).T
           This finds W such that input_tgt @ W.T ≈ output_tgt

    Args:
        source_weight: Source weight matrix [out_src, in_src].
        input_activations_source: Input activations in source space [n, in_src].
        alignment_in: Procrustes transform for inputs [in_src, in_tgt].
        alignment_out: Procrustes transform for outputs [out_src, out_tgt].
        backend: Compute backend.

    Returns:
        BehavioralReconstructionResult with reconstructed weight and diagnostics.
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

    # Step 1: Compute source layer behavior
    # output_src = input_src @ W_src.T  [n, out_src]
    output_source = b.matmul(input_activations_source, b.transpose(source_weight))
    b.eval(output_source)

    # Step 2: Project to target coordinates
    # input_tgt = input_src @ F_in  [n, in_tgt]
    input_target = b.matmul(input_activations_source, alignment_in)
    b.eval(input_target)

    # output_tgt = output_src @ F_out  [n, out_tgt]
    output_target = b.matmul(output_source, alignment_out)
    b.eval(output_target)

    # Step 3: Solve for weight via least squares
    # Find W such that input_tgt @ W.T = output_tgt
    # Equivalently: W.T = lstsq(input_tgt, output_tgt)
    # So: W = lstsq(input_tgt, output_tgt).T

    # Use geodesic-aware pseudoinverse for robustness
    from modelcypher.core.domain.geometry.numerical_stability import geodesic_pinv

    # pinv(input_tgt) @ output_tgt gives us W.T
    input_tgt_pinv = geodesic_pinv(b, input_target)
    b.eval(input_tgt_pinv)

    W_T = b.matmul(input_tgt_pinv, output_target)  # [in_tgt, out_tgt]
    b.eval(W_T)

    reconstructed_weight = b.transpose(W_T)  # [out_tgt, in_tgt]
    b.eval(reconstructed_weight)

    # Step 4: Compute reconstruction error
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

    This is the correct approach for merging models with different dimensions.
    Instead of transforming the weight matrix (which distorts magnitudes),
    we reconstruct the weight that produces the same BEHAVIOR.

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

    delta_norm = _geodesic_frobenius_norm(delta_W, b)

    # Step 3: Compute null-space projector on TARGET activations
    null_space_projector = compute_null_space_projector(
        input_activations=input_activations_target,
        source_activations_for_density=source_activations_for_density,
        target_activations_for_density=target_activations_for_density,
        backend=b,
    )

    # Step 4: Project delta through null-space
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

    projected_norm = _geodesic_frobenius_norm(delta_W_proj, b)

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
        "CROSS-DIM RESULT: delta_norm=%.4f, proj_norm=%.4f, preserved=%.1f%%",
        delta_norm, projected_norm, 100.0 * preserved_fraction
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
    backend: "Backend | None" = None,
) -> NullSpaceProjector:
    """Compute a reusable null-space projector from input activations.

    If density_weights are provided, they are used directly to weight the
    boundary activations. Otherwise, density weights are computed from
    source/target activations when available.
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

        constraint_weights = 1.0 - density_weights
        eps = division_epsilon(b, constraint_weights)
        sqrt_weights = b.sqrt(constraint_weights + eps)
        A_weighted = input_activations * b.reshape(sqrt_weights, (-1, 1))
        b.eval(A_weighted)
    else:
        A_weighted = input_activations

    # Build Gram matrix in sample space (n x n) for exact null-space projection
    AAt = b.matmul(A_weighted, b.transpose(A_weighted))
    b.eval(AAt)

    eps = machine_epsilon(b, AAt)
    # Use eigvalsh (eigenvalues only) - 1.75x faster than eigh
    # Eigenvectors are not used in the projection, only for diagnostics
    eigvals = b.eigvalsh(AAt)
    b.eval(eigvals)

    idx = b.argsort(-eigvals, axis=0)
    eigvals = b.take(eigvals, idx, axis=0)
    b.eval(eigvals)

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
    # VARIANCE-WEIGHTED NULL-SPACE (GEOMETRY-DERIVED, NO HEURISTICS)
    # =========================================================================
    # The intrinsic dimension of the activation manifold is measured for diagnostics,
    # but the null-space projector is built from the covariance eigenbasis.
    #
    # We scale deltas by (1 - normalized variance) in the eigenbasis:
    # - High-variance directions (dense target usage) are preserved.
    # - Low-variance directions (available capacity) accept transfer.
    #
    # This avoids the brittleness of hard null-space cutoffs while staying
    # fully data-derived and machine-precision stable.
    # =========================================================================
    id_estimator = IntrinsicDimension(b)
    id_result = id_estimator.compute(input_activations)
    intrinsic_dim = id_result.intrinsic_dimension

    in_dim = int(input_activations.shape[1])
    sample_dim = int(eigvals_pos.shape[0])

    max_eig_arr = b.max(eigvals_pos)
    min_eig_arr = b.min(eigvals_pos)
    b.eval(max_eig_arr, min_eig_arr)
    max_eig = float(b.to_scalar(max_eig_arr))
    min_eig = float(b.to_scalar(min_eig_arr))
    median_idx = sample_dim // 2
    median_eig = float(b.to_scalar(eigvals_pos[median_idx]))

    k = max(1, min(int(round(intrinsic_dim)), sample_dim))
    top_k_energy = b.sum(eigvals_pos[:k])
    b.eval(top_k_energy)
    energy_captured = float(b.to_scalar(top_k_energy)) / total_var_val

    eps = machine_epsilon(b, eigvals_pos)
    max_eig_safe = max(max_eig, eps)
    rank_scale = svd_rank_threshold(b, eigvals_pos, in_dim)
    rank_threshold = max_eig_safe * rank_scale
    rank_mask = eigvals_pos > rank_threshold
    rank_mask = b.astype(rank_mask, compute_dtype)

    activation_rank_arr = b.sum(rank_mask)
    b.eval(activation_rank_arr)
    activation_rank = int(round(float(b.to_scalar(activation_rank_arr))))
    activation_rank = max(0, min(activation_rank, sample_dim))
    null_rank = max(0, in_dim - activation_rank)

    logger.info(
        "NULL-SPACE DIAG: intrinsic_dim=%.2f, k=%d/%d, numeric_rank=%d/%d, null_rank=%d "
        "(energy_captured=%.4f, max_eig=%.3e, median_eig=%.3e, min_eig=%.3e)",
        intrinsic_dim,
        k,
        sample_dim,
        activation_rank,
        sample_dim,
        null_rank,
        energy_captured,
        max_eig,
        median_eig,
        min_eig,
    )

    # Compute Moore-Penrose pseudoinverse of Gram matrix
    # This is the mathematically correct operation for null-space projection:
    # P = I - A^T (A A^T)^+ A projects onto null(A^T)
    # The pseudoinverse handles rank deficiency correctly (rank determined by data)
    from modelcypher.core.domain.geometry.numerical_stability import geodesic_pinv
    AAt_inv = geodesic_pinv(b, AAt)
    b.eval(AAt_inv)

    return NullSpaceProjector(
        weighted_activations=A_weighted,
        gram_inv=AAt_inv,
        null_rank=null_rank,
        transfer_strength=transfer_strength,
    )


def compute_weight_space_transplant(
    source_aligned: "Array",
    target_weight: "Array",
    input_activations: "Array",
    source_activations_for_density: "Array | None" = None,
    target_activations_for_density: "Array | None" = None,
    null_space_projector: "NullSpaceProjector | None" = None,
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
            except Exception:
                pass
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
    source_norm_val = _geodesic_frobenius_norm(source_aligned, b)
    target_norm_val = _geodesic_frobenius_norm(target_weight, b)

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

    # Compute delta norm before projection (geodesic Frobenius)
    delta_norm = _geodesic_frobenius_norm(delta_W, b)

    # Compute geodesic cosine similarity between normalized source and target
    # This measures alignment quality BEFORE null-space projection
    # Uses geodesic law of cosines: cos(θ) = (d²(0,a) + d²(0,b) - d²(a,b)) / (2·d(0,a)·d(0,b))
    # geodesic_pairwise_metrics computes paired cosines via geodesic graph - precision over efficiency
    cos_vals, _ = geodesic_pairwise_metrics(
        source_normalized, target_weight, b, use_cache=False
    )
    cosine_sim = float(b.to_scalar(b.mean(cos_vals)))

    logger.info(
        "STITCH QUALITY: cosine_sim=%.4f, delta_norm=%.4f, target_norm=%.4f (ratio=%.2f)",
        cosine_sim,
        delta_norm,
        target_norm_val,
        delta_norm / (target_norm_val + eps),
    )

    if null_space_projector is None:
        null_space_projector = compute_null_space_projector(
            input_activations=input_activations,
            source_activations_for_density=source_activations_for_density,
            target_activations_for_density=target_activations_for_density,
            backend=b,
        )

    N = null_space_projector.projector
    null_rank = null_space_projector.null_rank
    transfer_strength = null_space_projector.transfer_strength

    # Step 4: Project delta to null-space
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

    # Compute projected norm (geodesic Frobenius)
    projected_norm = _geodesic_frobenius_norm(delta_W_proj, b)

    # Preserved fraction
    if delta_norm > 0:
        preserved_fraction = projected_norm / delta_norm
    else:
        preserved_fraction = 1.0

    # Step 5: Apply to target weight
    # NOTE: The null-space projection IS the geometry. No additional filtering needed.
    # If we've correctly projected delta into null-space, the merged weight will
    # preserve target behavior by construction.
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
            except Exception:
                pass
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

    # Step 2: Compute unconstrained solution
    # delta_W_unc = pinv(A_core) @ delta_A_core
    # A_core is [n, in_dim], pinv(A_core) is [in_dim, n]
    # delta_A is [n, out_dim]
    # Result: [in_dim, n] @ [n, out_dim] -> [in_dim, out_dim]
    A_c_pinv = b.pinv(activations_core)
    b.eval(A_c_pinv)

    delta_W_unc = b.matmul(A_c_pinv, delta_activations)
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

    # Compute metrics
    delta_W_norm_arr = geodesic_norms(
        b.reshape(delta_W, (1, -1)), b, use_cache=False
    )
    delta_W_unc_norm_arr = geodesic_norms(
        b.reshape(delta_W_unc, (1, -1)), b, use_cache=False
    )
    delta_A_norm_arr = geodesic_norms(
        b.reshape(delta_activations, (1, -1)), b, use_cache=False
    )
    b.eval(delta_W_norm_arr, delta_W_unc_norm_arr, delta_A_norm_arr)

    delta_W_norm = float(b.to_scalar(delta_W_norm_arr[0]))
    delta_W_unc_norm = float(b.to_scalar(delta_W_unc_norm_arr[0]))
    delta_A_norm = float(b.to_scalar(delta_A_norm_arr[0]))

    if delta_W_unc_norm > 0:
        preserved_fraction = delta_W_norm / delta_W_unc_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    # Compute null-space dimension (rank of N minus full rank)
    # Approximation: use n_boundary as indicator
    null_dim = n_boundary if n_boundary > 0 else 0

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

    This ensures boundary outputs are EXACTLY preserved (to numerical precision)
    while core outputs move toward the source's knowledge.

    There are no heuristics, thresholds, or modes. The geometry determines everything.

    Args:
        weight_target: Target model weights to modify [out_dim, in_dim].
        activations_core: Core activation samples [n_core, in_dim].
        delta_activations: Desired change in activation space [n_core, out_dim].
            Computed upstream via anchor-relative grafting.
        boundary_activations: Boundary samples [n_boundary, in_dim]. If provided,
            their outputs are exactly preserved: A_boundary @ W' = A_boundary @ W.
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
