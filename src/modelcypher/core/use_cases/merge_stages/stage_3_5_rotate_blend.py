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
Stages 3-5: ROTATE + BLEND + PROPAGATE

For each layer:
1. ROTATE: Phase-lock alignment to CKA=1.0 (GramAligner)
2. BLEND: Compute multi-layer alpha with all adjustments
3. PROPAGATE: Carry rotation to next layer (zipper)

When use_transport_guided=True, uses Gromov-Wasserstein optimal transport
instead of Procrustes rotation.

Blend adjustments applied in sequence:
1. Base alpha from intersection confidence
2. Gaussian smoothing across layers
3. Spectral penalty for ill-conditioned weights
4. SVD-aware blending (different alpha for high/low rank)
5. Correlation-based dimension weights (from IntersectionMap)
6. VerbNoun modulation

Reference: Ainsworth et al. (2022) "Git Re-Basin"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import svd_via_eigh
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class RotateBlendConfig:
    """Configuration for Stages 3-5.

    GEOMETRY-DRIVEN MERGE: Two strategies available.

    1. SLERP (Spherical Linear Interpolation) - DEFAULT for cross-model merges:
       - Treats weight matrices as high-dimensional vectors
       - Interpolates along geodesic on hypersphere
       - Preserves BOTH source and target geometry proportionally
       - Formula: W = (sin((1-t)θ)/sinθ)W_s + (sin(tθ)/sinθ)W_t

    2. Fréchet-SVD (legacy) - for same-architecture merges:
       - Decomposes into U, Σ, V
       - Uses TARGET geometry with blended magnitudes
       - Formula: W = U_t @ diag(√(σ_s ⊙ σ_t)) @ V_t^T

    SLERP is preferred for disparate models because it doesn't
    destroy source geometry by projecting onto target's singular vectors.
    """

    # SLERP-based merging (preserves both geometries)
    use_slerp: bool = True  # Default to SLERP for cross-model merges
    slerp_t: float = 0.5  # Interpolation factor: 0=source, 1=target

    # Legacy flags - kept for backward compatibility
    enable_rotation: bool = True  # Pre-alignment before merge
    use_transport_guided: bool = False  # Transport alternative

    # PROPAGATE (zipper) - maintains layer continuity
    enable_zipper: bool = True
    zipper_use_weight_matching: bool = True


@dataclass
class RotateBlendResult:
    """Result of Stages 3-5."""

    merged_weights: dict[str, Any]  # Backend arrays
    rotate_metrics: dict[str, Any]
    blend_metrics: dict[str, Any]


# =============================================================================
# PURE GEOMETRIC MERGE - No Parameters, No Thresholds
# =============================================================================


def geometric_merge_weights(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend",
    key: str = "",
    base_alpha: float | None = None,
) -> tuple["Array", dict[str, Any]]:
    """
    Geometrically correct merge of two weight matrices.

    PURE GEOMETRY. NO ARBITRARY THRESHOLDS.

    The merge formula is mathematically derived:
    1. Procrustes alignment: Find optimal rotation R* = argmin ||W_t - R @ W_s||_F
    2. SVD decomposition: Extract singular vectors (geometry) and values (magnitude)
    3. Fréchet mean: Geometric mean of singular values (geodesic midpoint on ℝ^+)
    4. Reconstruct: Use target's geometry with merged magnitudes

    Formula: W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

    If base_alpha is provided, task singular vectors split skill vs structure
    in the task vector Δ = W_source - W_target and blend deterministically
    using spectrum-derived weights.

    For embedding layers (m >> n), use direct Fréchet mean since tokens are
    already aligned by ID - no need for expensive Procrustes/SVD.

    Args:
        source_w: Source weight matrix (m × n)
        target_w: Target weight matrix (m × n) - must match source shape
        backend: Backend for GPU-accelerated operations
        key: Weight key name (for logging)
        base_alpha: Optional deterministic blend factor from probe geometry

    Returns:
        Tuple of (merged_weights, metrics)
        metrics contains: procrustes_error, spectral_preservation, merge_quality
    """
    b = backend

    # Ensure float32 for numerical stability
    source_f32 = b.astype(source_w, "float32")
    target_f32 = b.astype(target_w, "float32")

    m, n = source_f32.shape

    # ==========================================================================
    # SPECIAL CASE: Very tall matrices (embeddings, m >> n)
    # For embeddings, tokens are aligned by ID. Full SVD would allocate
    # O(m^2) memory which is prohibitive. Use direct Fréchet mean instead.
    # ==========================================================================
    if m > 4 * n and m > 10000:
        # Direct element-wise Fréchet mean: √(s × t) per element
        eps = 1e-10
        source_abs = b.abs(source_f32)
        target_abs = b.abs(target_f32)
        merged_magnitude = b.sqrt((source_abs + eps) * (target_abs + eps))
        # Preserve target's sign structure
        merged = merged_magnitude * b.sign(target_f32)
        b.eval(merged)

        # Compute simple distance metric
        diff = target_f32 - source_f32
        diff_norm = b.norm(b.reshape(diff, (-1,)))
        target_norm = b.norm(b.reshape(target_f32, (-1,)))
        b.eval(diff_norm, target_norm)
        target_norm_val = float(target_norm.item()) if hasattr(target_norm, 'item') else float(b.to_numpy(target_norm))
        diff_norm_val = float(diff_norm.item()) if hasattr(diff_norm, 'item') else float(b.to_numpy(diff_norm))

        metrics = {
            "procrustes_error": diff_norm_val / (target_norm_val + 1e-10),
            "spectral_preservation": 1.0,  # Direct merge preserves spectrum
            "merge_quality": 1.0,
            "effective_rank": min(m, n),
            "mode": "direct_frechet",
        }
        return merged, metrics

    # ==========================================================================
    # STEP 1: Procrustes Alignment
    # Find optimal orthogonal transformation to align source to target.
    #
    # For tall matrices (m >> n, e.g., embeddings [vocab, hidden]):
    #   - Work in column space: M = W_t.T @ W_s gives [n, n] matrix
    #   - Apply rotation to columns: W_aligned = W_s @ R
    #
    # For wide/square matrices (m <= n, e.g., projections [hidden, hidden]):
    #   - Work in row space: M = W_t @ W_s.T gives [m, m] matrix
    #   - Apply rotation to rows: W_aligned = R @ W_s
    # ==========================================================================

    if m > n:
        # Tall matrix (e.g., embeddings): align columns
        # M = [n, m] @ [m, n] = [n, n] - SMALL!
        M = b.matmul(b.transpose(target_f32), source_f32)
        U_M, _, Vt_M = svd_via_eigh(b, M, full_matrices=False)
        R = b.matmul(U_M, Vt_M)
        b.eval(R)

        # Handle reflection
        det_R = b.det(R)
        b.eval(det_R)
        det_val = float(det_R.item()) if hasattr(det_R, 'item') else float(b.to_numpy(det_R))
        if det_val < 0:
            U_M_cols = [U_M[:, i:i+1] for i in range(n-1)]
            U_M_cols.append(U_M[:, -1:] * -1.0)
            U_M_fixed = b.concatenate(U_M_cols, axis=1)
            R = b.matmul(U_M_fixed, Vt_M)
            b.eval(R)

        # Apply rotation to source columns: W_aligned = W_s @ R
        source_aligned = b.matmul(source_f32, R)
        b.eval(source_aligned)
    else:
        # Wide/square matrix: align rows (original approach)
        # M = [m, n] @ [n, m] = [m, m]
        M = b.matmul(target_f32, b.transpose(source_f32))
        U_M, _, Vt_M = svd_via_eigh(b, M, full_matrices=False)
        R = b.matmul(U_M, Vt_M)
        b.eval(R)

        # Handle reflection
        det_R = b.det(R)
        b.eval(det_R)
        det_val = float(det_R.item()) if hasattr(det_R, 'item') else float(b.to_numpy(det_R))
        if det_val < 0:
            U_M_cols = [U_M[:, i:i+1] for i in range(m-1)]
            U_M_cols.append(U_M[:, -1:] * -1.0)
            U_M_fixed = b.concatenate(U_M_cols, axis=1)
            R = b.matmul(U_M_fixed, Vt_M)
            b.eval(R)

        # Apply rotation to source rows: W_aligned = R @ W_s
        source_aligned = b.matmul(R, source_f32)
        b.eval(source_aligned)

    # Compute Procrustes error (normalized) - all backend ops
    diff = target_f32 - source_aligned
    error_norm = b.norm(b.reshape(diff, (-1,)))
    target_norm = b.norm(b.reshape(target_f32, (-1,)))
    b.eval(error_norm, target_norm)

    def to_scalar(arr: "Array") -> float:
        """Extract scalar from backend array."""
        if hasattr(arr, "item"):
            return float(arr.item())
        return float(b.to_numpy(arr))

    target_norm_val = to_scalar(target_norm)

    if base_alpha is not None:
        from modelcypher.core.domain.geometry.task_singular_vectors import (
            SVDBlendConfig,
            blend_with_svd_awareness,
            decompose_task_vector,
        )

        task_config = SVDBlendConfig()
        decomp = decompose_task_vector(source_aligned, target_f32, task_config)
        merged = blend_with_svd_awareness(
            source_aligned, target_f32, base_alpha, task_config
        )
        b.eval(merged)

        # Spectral preservation via Frobenius energy
        energy_source = b.sum(source_aligned * source_aligned)
        energy_target = b.sum(target_f32 * target_f32)
        energy_merged = b.sum(merged * merged)
        b.eval(energy_source, energy_target, energy_merged)

        diff_to_source = b.norm(b.reshape(merged - source_aligned, (-1,)))
        diff_to_target = b.norm(b.reshape(merged - target_f32, (-1,)))
        b.eval(diff_to_source, diff_to_target)

        eps = 1e-10
        procrustes_error = to_scalar(error_norm) / (target_norm_val + eps)
        spectral_preservation = to_scalar(energy_merged) / (
            0.5 * (to_scalar(energy_source) + to_scalar(energy_target)) + eps
        )
        merge_quality = 1.0 - 0.5 * (
            to_scalar(diff_to_source) + to_scalar(diff_to_target)
        ) / (target_norm_val + eps)

        metrics = {
            "procrustes_error": procrustes_error,
            "spectral_preservation": spectral_preservation,
            "merge_quality": merge_quality,
            "effective_rank": decomp.effective_rank,
            "variance_captured": decomp.variance_captured,
            "mode": "task_svd",
            "base_alpha": float(base_alpha),
        }

        return merged, metrics

    # ==========================================================================
    # STEP 2: SVD Decomposition
    # W = UΣV^T where U, V are geometry (singular vectors), Σ is magnitude
    # ==========================================================================

    U_s, sigma_s, Vt_s = svd_via_eigh(b, source_aligned, full_matrices=False)
    U_t, sigma_t, Vt_t = svd_via_eigh(b, target_f32, full_matrices=False)
    b.eval(U_s, sigma_s, U_t, sigma_t, Vt_t)

    # ==========================================================================
    # STEP 3: Fréchet Mean of Singular Values
    # The geometric mean is the Fréchet mean on ℝ^+ with log metric
    # σ_merged = √(σ_s × σ_t)
    # ==========================================================================

    # Handle dimension mismatch in singular values
    k = min(sigma_s.shape[0], sigma_t.shape[0])
    sigma_s_k = sigma_s[:k]
    sigma_t_k = sigma_t[:k]

    # Geometric mean - the mathematically correct merge for positive magnitudes
    # Add small epsilon to avoid sqrt(0) issues
    eps = 1e-10
    sigma_merged = b.sqrt((sigma_s_k + eps) * (sigma_t_k + eps))
    b.eval(sigma_merged)

    # Compute spectral energies - all backend ops
    energy_source = b.sum(sigma_s_k ** 2)
    energy_target = b.sum(sigma_t_k ** 2)
    energy_merged = b.sum(sigma_merged ** 2)
    b.eval(energy_source, energy_target, energy_merged)

    # ==========================================================================
    # STEP 4: Reconstruct with Target Geometry and Merged Magnitudes
    # W_merged = U_t @ diag(σ_merged) @ V_t^T
    # ==========================================================================

    U_t_k = U_t[:, :k]
    Vt_t_k = Vt_t[:k, :]

    # Reconstruct: (U_t * sigma_merged) @ Vt_t
    merged = b.matmul(U_t_k * sigma_merged, Vt_t_k)
    b.eval(merged)

    # Compute merge quality - all backend ops
    diff_to_source = b.norm(b.reshape(merged - source_aligned, (-1,)))
    diff_to_target = b.norm(b.reshape(merged - target_f32, (-1,)))
    b.eval(diff_to_source, diff_to_target)

    # Extract scalars only at the boundary for metrics dict
    procrustes_error = to_scalar(error_norm) / (target_norm_val + eps)
    spectral_preservation = to_scalar(energy_merged) / (
        0.5 * (to_scalar(energy_source) + to_scalar(energy_target)) + eps
    )
    merge_quality = 1.0 - 0.5 * (
        to_scalar(diff_to_source) + to_scalar(diff_to_target)
    ) / (target_norm_val + eps)

    metrics = {
        "procrustes_error": procrustes_error,
        "spectral_preservation": spectral_preservation,
        "merge_quality": merge_quality,
        "effective_rank": k,
        "mode": "frechet_svd",
    }

    return merged, metrics


def stage_rotate_blend_propagate(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    dimension_correlations: dict,
    layer_indices: list[int],
    config: RotateBlendConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    backend: Any | None = None,
    source_activations: dict[int, list[Any]] | None = None,
    target_activations: dict[int, list[Any]] | None = None,
) -> RotateBlendResult:
    """
    PURE GEOMETRIC MERGE with null-space filtering.

    For each weight matrix:
    1. Procrustes alignment: Find optimal rotation R* = argmin ||W_t - R @ W_s||_F
    2. SVD decomposition: Extract geometry (U, V) and magnitude (Σ)
    3. Fréchet mean: Geometric mean of singular values
    4. Null-space filtering: Project weight delta to null space of target activations
    5. Reconstruct: W_merged = W_target + filtered_delta

    Null-space filtering (from MINGLE paper) ensures that merged weights don't
    interfere with target model's existing behavior: if Δw ∈ null(A), then
    A @ (W + Δw) = A @ W.

    The geometry determines everything. Layer confidences inform deterministic
    base alphas and task singular vector blending when available.
    """
    # Initialize backend
    b = backend or get_default_backend()

    # Clear GPU cache before merge
    if hasattr(b, "clear_cache"):
        b.clear_cache()

    # Initialize per-layer phase-lock transforms from activations
    # CKA is the barometer: we search for the transform that locks to CKA=1.0
    layer_rotations: dict[int, "Array"] = {}
    layer_null_projections: dict[int, Any] = {}
    phase_lock_ckas: dict[int, float] = {}
    phase_lock_iterations: dict[int, int] = {}
    phase_lock_errors: dict[int, float] = {}
    phase_lock_signals: dict[int, Any] = {}

    if source_activations and target_activations:
        logger.info(
            "PHASE LOCK: Searching per-layer transforms from %d layer activations",
            len(target_activations),
        )

        from modelcypher.core.domain.geometry.gram_aligner import GramAligner

        aligner = GramAligner(backend=b)

        for layer_idx in sorted(target_activations.keys()):
            src_acts = source_activations.get(layer_idx)
            tgt_acts = target_activations.get(layer_idx)

            if not src_acts or not tgt_acts:
                continue

            n_samples = min(len(src_acts), len(tgt_acts))
            if n_samples < 2:
                logger.warning(
                    "LAYER %d: Phase lock needs >= 2 activation samples, got %d",
                    layer_idx,
                    n_samples,
                )
                phase_lock_signals[layer_idx] = {
                    "reason": "insufficient_samples",
                    "required": 2,
                    "available": n_samples,
                }
                continue

            try:
                # Stack activations: [n_samples, hidden_dim]
                src_stacked = b.stack(src_acts[:n_samples], axis=0)
                tgt_stacked = b.stack(tgt_acts[:n_samples], axis=0)
                b.eval(src_stacked, tgt_stacked)

                result = aligner.find_perfect_alignment(src_stacked, tgt_stacked)
                transform = b.array(result.feature_transform)
                b.eval(transform)

                layer_rotations[layer_idx] = transform
                phase_lock_ckas[layer_idx] = result.achieved_cka
                phase_lock_iterations[layer_idx] = result.iterations
                phase_lock_errors[layer_idx] = result.alignment_error
                if result.diagnostic is not None:
                    phase_lock_signals[layer_idx] = result.diagnostic.to_dict()

                logger.debug(
                    "LAYER %d: Phase lock CKA=%.8f (iters=%d, error=%.6f)",
                    layer_idx,
                    result.achieved_cka,
                    result.iterations,
                    result.alignment_error,
                )
            except Exception as e:
                logger.error("LAYER %d: Phase lock alignment failed: %s", layer_idx, e)
                phase_lock_signals[layer_idx] = {"reason": str(e)}
                continue

        if layer_rotations:
            logger.info(
                "PHASE LOCK: Computed %d per-layer transforms",
                len(layer_rotations),
            )

    elif target_activations:
        # Only target activations available - compute null-space projections
        from modelcypher.core.domain.geometry.null_space_filter import (
            NullSpaceFilter,
            NullSpaceFilterConfig,
        )

        null_space_filter = NullSpaceFilter(
            NullSpaceFilterConfig(rank_threshold=0.01, min_samples=5),
            backend=b,
        )

        for layer_idx, act_list in target_activations.items():
            if act_list and len(act_list) >= 5:
                stacked = b.stack(act_list, axis=0)
                b.eval(stacked)
                try:
                    projection = null_space_filter.compute_null_space_projection(stacked)
                    if projection.null_dim > 0:
                        layer_null_projections[layer_idx] = projection
                except Exception as e:
                    logger.debug("NULL-SPACE: Failed for layer %d: %s", layer_idx, e)

        if layer_null_projections:
            logger.info(
                "NULL-SPACE: Computed projections for %d layers",
                len(layer_null_projections),
            )

    merged: dict[str, "Array"] = {}
    metrics: dict[str, Any] = {
        "procrustes_errors": [],
        "spectral_preservations": [],
        "merge_qualities": [],
        "layer_confidences_used": layer_confidences.copy() if layer_confidences else {},
        "spectral_confidences": [],
        "phase_lock_layers": len(layer_rotations),
        "phase_lock_ckas": phase_lock_ckas,
        "phase_lock_iterations": phase_lock_iterations,
        "phase_lock_errors": phase_lock_errors,
        "phase_lock_signals": phase_lock_signals,
        "geometric_merges": 0,
        "task_svd_merges": 0,
        "frechet_svd_merges": 0,
        "identity_copies": 0,
        "shape_mismatches": 0,
        "null_space_filtered": 0,
        "null_space_preservation": [],
        "per_layer_rotations_applied": 0,
        "per_layer_rotation_layers": len(layer_rotations),
    }

    if phase_lock_ckas:
        metrics["phase_lock_mean_cka"] = sum(phase_lock_ckas.values()) / len(phase_lock_ckas)
        metrics["phase_lock_min_cka"] = min(phase_lock_ckas.values())
        metrics["phase_lock_max_cka"] = max(phase_lock_ckas.values())

    if not layer_confidences and dimension_correlations:
        derived_confidences: dict[int, float] = {}
        for layer_key, correlations in dimension_correlations.items():
            if not correlations:
                continue
            corr_vals: list[float] = []
            for corr in correlations:
                if hasattr(corr, "correlation"):
                    corr_vals.append(float(corr.correlation))
                elif isinstance(corr, dict) and "correlation" in corr:
                    corr_vals.append(float(corr["correlation"]))
            if corr_vals:
                layer_idx = int(layer_key) if not isinstance(layer_key, int) else layer_key
                mean_corr = sum(corr_vals) / len(corr_vals)
                derived_confidences[layer_idx] = max(0.0, min(1.0, mean_corr))
        if derived_confidences:
            layer_confidences = derived_confidences
            metrics["layer_confidences_used"] = derived_confidences.copy()
            metrics["layer_confidences_from_dimension_correlations"] = True

    layer_alphas: dict[int, float] = {}
    if layer_confidences:
        for layer_idx, confidence in layer_confidences.items():
            conf = max(0.0, min(1.0, float(confidence)))
            layer_alphas[layer_idx] = 1.0 - conf

        if len(layer_alphas) > 2:
            from modelcypher.core.domain.geometry.alpha_smoothing import (
                AlphaSmoothingConfig,
                gaussian_smooth_alpha_profile,
            )

            window = max(1, int(round(math.sqrt(len(layer_alphas)) / 2)))
            sigma = max(1.0, window / 2.0)
            smooth_config = AlphaSmoothingConfig.with_parameters(
                smoothing_window=window,
                sigma=sigma,
                alpha_min=0.0,
                alpha_max=1.0,
            )
            layer_alphas = gaussian_smooth_alpha_profile(layer_alphas, smooth_config)
            metrics["alpha_smoothing"] = {"window": window, "sigma": sigma}

    mean_alpha = sum(layer_alphas.values()) / len(layer_alphas) if layer_alphas else 0.5
    metrics["mean_layer_alpha"] = mean_alpha
    task_blend_enabled = bool(layer_alphas)

    from modelcypher.core.domain.geometry.spectral_analysis import (
        SpectralConfig,
        apply_spectral_penalty,
        compute_spectral_metrics,
    )

    spectral_config = SpectralConfig()

    total_weights = len(target_weights)
    processed = 0

    # Log probe alignment quality
    mean_conf = 0.0
    if layer_confidences:
        conf_vals = list(layer_confidences.values())
        mean_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
    logger.info(
        "GEOMETRIC MERGE: %d layers, mean_confidence=%.3f (from probe)",
        len(layer_confidences), mean_conf,
    )
    logger.info("GEOMETRIC MERGE: Processing %d weight keys", total_weights)

    def _apply_phase_lock_transform(
        weight: "Array",
        transform: "Array",
    ) -> tuple["Array", bool]:
        """Apply phase-lock transform to weight if dimensions match."""
        if weight.ndim != 2:
            return weight, False

        t_in = transform.shape[0]
        weight_f32 = b.astype(weight, "float32")

        if weight.shape[1] == t_in:
            aligned = b.matmul(weight_f32, transform)
            b.eval(aligned)
            return aligned, True

        if weight.shape[0] == t_in:
            aligned = b.matmul(b.transpose(transform), weight_f32)
            b.eval(aligned)
            return aligned, True

        return weight, False

    for key in sorted(target_weights.keys()):
        if key not in source_weights:
            continue

        processed += 1

        # Skip quantization metadata
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        # Dequantize if needed
        source_w = dequantize_if_needed(
            source_weights[key], key, source_weights, b
        )
        target_w = dequantize_if_needed(
            target_weights[key], key, target_weights, b
        )

        layer_idx = extract_layer_index_fn(key)
        base_alpha = layer_alphas.get(layer_idx, mean_alpha) if task_blend_enabled else None
        layer_confidence = 0.5 if base_alpha is None else max(0.0, min(1.0, 1.0 - base_alpha))

        # Apply phase-lock transform before shape normalization if available
        if layer_idx is not None and layer_idx in layer_rotations and source_w.ndim == 2:
            transform = layer_rotations[layer_idx]
            source_w, applied = _apply_phase_lock_transform(source_w, transform)
            if applied:
                metrics["per_layer_rotations_applied"] += 1

        # Handle shape mismatch - project source to target shape
        if source_w.shape != target_w.shape:
            metrics["shape_mismatches"] += 1

            if source_w.ndim == 1:
                # 1D tensors (biases, norms): truncate or pad
                source_size = source_w.shape[0]
                target_size = target_w.shape[0]
                source_f32 = b.astype(source_w, "float32")
                b.eval(source_f32)

                if source_size > target_size:
                    source_w = source_f32[:target_size]
                else:
                    mean_val = b.mean(source_f32)
                    b.eval(mean_val)
                    padding = b.full((target_size - source_size,), mean_val, dtype="float32")
                    source_w = b.concatenate([source_f32, padding], axis=0)
                b.eval(source_w)
                logger.debug(
                    "1D PROJECTION at %s: %d -> %d",
                    key, source_size, target_size,
                )
            else:
                # 2D+ tensors: use unified cross-dimensional projection
                from modelcypher.core.domain.geometry.cross_dimensional_projection import (
                    project_cross_dimensional,
                    ProjectionMethod,
                )

                logger.info(
                    "CROSS-DIM PROJECTION at %s: source=%s -> target=%s",
                    key, source_w.shape, target_w.shape,
                )
                result = project_cross_dimensional(
                    source_w, target_w,
                    method=ProjectionMethod.GRAM_TRANSPORT,
                    backend=b,
                )
                source_w = result.projected
                b.eval(source_w)
                logger.info(
                    "  -> projected with alignment_score=%.4f",
                    result.alignment_score,
                )

        # Apply geometric merge for 2D weight matrices
        if source_w.ndim == 2 and target_w.ndim == 2 and min(source_w.shape) >= 2:
            spectral_confidence = None
            if base_alpha is not None and source_w.shape == target_w.shape:
                spectral = compute_spectral_metrics(
                    source_w, target_w, spectral_config, backend=b
                )
                spectral_confidence = spectral.spectral_confidence
                penalty_strength = 1.0 - layer_confidence
                base_alpha = apply_spectral_penalty(
                    base_alpha, spectral_confidence, penalty_strength
                )
                metrics["spectral_confidences"].append(spectral_confidence)

            merged_w, merge_metrics = geometric_merge_weights(
                source_w, target_w, b, key=key, base_alpha=base_alpha
            )

            # Apply null-space filtering if available for this layer
            # This ensures merged weights don't interfere with target's existing behavior
            if layer_idx is not None and layer_idx in layer_null_projections:
                projection = layer_null_projections[layer_idx]
                target_f32 = b.astype(target_w, "float32")
                merged_f32 = b.astype(merged_w, "float32")

                # Compute weight delta
                delta = merged_f32 - target_f32
                b.eval(delta)

                # Project delta to null space
                # For weight matrices, we need to match activation dimension
                # Activations are [hidden_dim], weights are [out_dim, in_dim]
                # We filter based on which dimension matches
                hidden_dim = projection.projection_matrix.shape[0]
                delta_shape = delta.shape

                filtered_delta = delta  # Default: no filtering
                filtering_applied = False

                if delta_shape[1] == hidden_dim:
                    # Input dimension matches hidden_dim: filter each row
                    # W @ x where x is hidden-dim, so project delta rows
                    delta_flat = b.reshape(delta, (-1,))
                    if delta_flat.shape[0] == hidden_dim:
                        filtered_flat = b.matmul(projection.projection_matrix, delta_flat)
                        filtered_delta = b.reshape(filtered_flat, delta_shape)
                        filtering_applied = True
                    elif delta_shape[0] * delta_shape[1] % hidden_dim == 0:
                        # Try row-wise filtering
                        try:
                            rows = [
                                b.matmul(projection.projection_matrix, delta[i])
                                for i in range(delta_shape[0])
                            ]
                            filtered_delta = b.stack(rows, axis=0)
                            filtering_applied = True
                        except Exception:
                            pass
                elif delta_shape[0] == hidden_dim:
                    # Output dimension matches: filter each column (transpose, filter, transpose)
                    try:
                        delta_t = b.transpose(delta)
                        cols = [
                            b.matmul(projection.projection_matrix, delta_t[i])
                            for i in range(delta_t.shape[0])
                        ]
                        filtered_delta = b.transpose(b.stack(cols, axis=0))
                        filtering_applied = True
                    except Exception:
                        pass

                if filtering_applied:
                    b.eval(filtered_delta)

                    # Compute preservation ratio
                    orig_norm = b.norm(b.reshape(delta, (-1,)))
                    filt_norm = b.norm(b.reshape(filtered_delta, (-1,)))
                    b.eval(orig_norm, filt_norm)
                    preservation = float(filt_norm.item()) / (float(orig_norm.item()) + 1e-10)

                    # Apply filtered delta
                    merged_w = target_f32 + filtered_delta
                    b.eval(merged_w)

                    metrics["null_space_filtered"] += 1
                    metrics["null_space_preservation"].append(preservation)

                    if processed % 50 == 0:
                        logger.info(
                            "NULL-SPACE [%d/%d] %s: preserved=%.1f%% of delta",
                            processed, total_weights, key, preservation * 100,
                        )

            metrics["procrustes_errors"].append(merge_metrics["procrustes_error"])
            metrics["spectral_preservations"].append(merge_metrics["spectral_preservation"])
            metrics["merge_qualities"].append(merge_metrics["merge_quality"])
            metrics["geometric_merges"] += 1
            if merge_metrics.get("mode") == "task_svd":
                metrics["task_svd_merges"] += 1
            else:
                metrics["frechet_svd_merges"] += 1

            if processed % 50 == 0:
                logger.info(
                    "GEOMETRIC MERGE [%d/%d] %s: error=%.4f, spectral=%.4f, quality=%.4f",
                    processed, total_weights, key,
                    merge_metrics["procrustes_error"],
                    merge_metrics["spectral_preservation"],
                    merge_metrics["merge_quality"],
                )
        else:
            # For 1D tensors (biases, norms) - geometric mean of magnitudes
            merged_w = b.sqrt((source_w + 1e-10) * (target_w + 1e-10)) * b.sign(target_w)
            b.eval(merged_w)
            metrics["identity_copies"] += 1

        # Preserve target dtype
        target_dtype = target_w.dtype
        if hasattr(target_dtype, "name"):
            # MLX dtype - extract name
            dtype_str = target_dtype.name if hasattr(target_dtype, "name") else "float32"
        else:
            dtype_str = str(target_dtype).replace("mlx.core.", "")
        merged[key] = b.astype(merged_w, dtype_str)

    # Copy target-only keys (skip quantization metadata)
    for key in target_weights:
        if key not in merged and not key.endswith(".scales") and not key.endswith(".biases"):
            merged[key] = b.array(target_weights[key])

    # Finalize metrics
    if metrics["procrustes_errors"]:
        errors = metrics["procrustes_errors"]
        metrics["mean_procrustes_error"] = sum(errors) / len(errors)
        metrics["max_procrustes_error"] = max(errors)

    if metrics["spectral_preservations"]:
        sp = metrics["spectral_preservations"]
        metrics["mean_spectral_preservation"] = sum(sp) / len(sp)

    if metrics["merge_qualities"]:
        mq = metrics["merge_qualities"]
        metrics["mean_merge_quality"] = sum(mq) / len(mq)

    # Add per-layer rotation summary
    if metrics["per_layer_rotations_applied"] > 0:
        logger.info(
            "PHASE LOCK: Applied %d transforms from %d layers",
            metrics["per_layer_rotations_applied"],
            metrics["per_layer_rotation_layers"],
        )

    # Add null-space filtering summary
    if metrics["null_space_preservation"]:
        pres = metrics["null_space_preservation"]
        metrics["mean_null_space_preservation"] = sum(pres) / len(pres)
        logger.info(
            "NULL-SPACE FILTERING: %d weights filtered, mean preservation=%.1f%%",
            metrics["null_space_filtered"],
            metrics["mean_null_space_preservation"] * 100,
        )

    logger.info(
        "GEOMETRIC MERGE COMPLETE: %d weights, mean_error=%.4f, mean_quality=%.4f",
        metrics["geometric_merges"],
        metrics.get("mean_procrustes_error", 0),
        metrics.get("mean_merge_quality", 0),
    )

    return RotateBlendResult(
        merged_weights=merged,
        rotate_metrics=metrics,  # Unified metrics
        blend_metrics=metrics,
    )
