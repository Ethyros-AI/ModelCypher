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
1. ROTATE: Compute/apply geometric alignment (Procrustes or GW Transport)
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class RotateBlendConfig:
    """Configuration for Stages 3-5."""

    # ROTATION
    enable_rotation: bool = True
    rotation_confidence_threshold: float = 0.3
    alignment_rank: int = 32
    use_transport_guided: bool = False
    transport_coupling_threshold: float = 0.1

    # BLEND
    base_alpha: float = 0.5
    alpha_min: float = 0.0
    alpha_max: float = 1.0

    # Smoothing
    enable_alpha_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_sigma: float = 1.5

    # Spectral
    enable_spectral_penalty: bool = True
    spectral_penalty_strength: float = 0.3

    # SVD - per-component alphas derived from spectrum, no hardcoded ratios
    enable_svd_blending: bool = False

    # Correlation - weights derived from correlation structure
    enable_correlation_weights: bool = False
    correlation_scale: float = 1.0

    # VerbNoun
    enable_verb_noun: bool = False
    verb_noun_strength: float = 0.5

    # Domain signals
    # Uses gradient SNR and sparsity to adjust per-layer alpha
    enable_domain_signals: bool = True
    domain_signal_strength: float = 0.3

    # Module policy
    enable_module_policy: bool = True
    module_policy_v_alpha: float = 0.7
    module_policy_o_alpha: float = 0.3

    # PROPAGATE (zipper)
    enable_zipper: bool = True
    zipper_use_weight_matching: bool = True

    # Refinement density (3-model merge refinement)
    enable_refinement_density: bool = True
    refinement_density_strength: float = 0.8

    # Intrinsic dimension gating (dimensional hierarchy)
    # Low intrinsic_dim/hidden_dim → simple manifold → blend aggressively
    # High ratio → complex manifold → blend conservatively (trust target)
    enable_intrinsic_dim_gating: bool = True
    intrinsic_dim_strength: float = 0.5
    intrinsic_dim_threshold: float = 0.01  # SVD cutoff for effective rank

    # Thermodynamic phase gating
    # ORDERED: Full alpha (safe for aggressive blend)
    # CRITICAL: Reduce alpha by 30% (near phase boundary)
    # DISORDERED: Reduce alpha by 15% (high entropy)
    entropy_phase: str = "ordered"  # "ordered", "critical", "disordered"

    # Curvature gating (manifold topology risk)
    # POSITIVE curvature → convergent geodesics → interference risk → reduce alpha
    # NEGATIVE curvature → divergent geodesics → safe → slightly increase alpha
    # High anisotropy → directional instability → reduce alpha
    enable_curvature_gating: bool = True
    curvature_strength: float = 0.3
    curvature_anisotropy_threshold: float = 0.7  # Above this, apply extra penalty


@dataclass
class RotateBlendResult:
    """Result of Stages 3-5."""

    merged_weights: dict[str, Any]  # Backend arrays
    rotate_metrics: dict[str, Any]
    blend_metrics: dict[str, Any]


def stage_rotate_blend_propagate(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    dimension_correlations: dict,
    layer_indices: list[int],
    config: RotateBlendConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    refinement_alphas: dict[int, float] | None = None,
    hard_swap_layers: set[int] | None = None,
    backend: Any | None = None,
) -> RotateBlendResult:
    """
    Stages 3-5 merged into single loop for efficiency.

    Args:
        source_weights: Permuted source model weights
        target_weights: Target model weights
        intersection_map_obj: IntersectionMap object
        layer_confidences: Per-layer confidence scores
        dimension_correlations: Per-layer dimension correlation data
        layer_indices: List of layer indices to process
        config: Stage configuration
        extract_layer_index_fn: Function to extract layer index from weight key
        refinement_alphas: Per-layer alphas from refinement density
        hard_swap_layers: Layers to hard-swap due to high refinement

    Returns:
        RotateBlendResult with merged weights and metrics
    """
    from modelcypher.core.domain.geometry.alpha_smoothing import (
        AlphaSmoothingConfig,
        gaussian_smooth_alpha_profile,
    )
    from modelcypher.core.domain.geometry.spectral_analysis import (
        SpectralConfig,
        apply_spectral_penalty,
        compute_spectral_metrics,
    )
    from modelcypher.core.domain.geometry.task_singular_vectors import (
        SVDBlendConfig,
        blend_with_svd_awareness,
    )
    from modelcypher.core.domain.geometry.verb_noun_classifier import (
        VerbNounConfig,
    )

    # Initialize backend for GPU-accelerated operations
    b = backend or get_default_backend()

    intersection_map_obj is not None and bool(dimension_correlations)

    # Pre-compute smoothed alpha profile
    raw_alphas = _compute_raw_alphas(
        layer_indices,
        layer_confidences,
        refinement_alphas,
        config.refinement_density_strength,
    )

    if config.enable_alpha_smoothing:
        smoothing_config = AlphaSmoothingConfig(
            smoothing_window=config.smoothing_window,
            sigma=config.smoothing_sigma,
        )
        smoothed_alphas = gaussian_smooth_alpha_profile(raw_alphas, smoothing_config)
    else:
        smoothed_alphas = raw_alphas

    # Config objects
    # SVDBlendConfig only has numerical stability parameters
    # Per-component alphas are derived from the SVD spectrum
    svd_config = SVDBlendConfig() if config.enable_svd_blending else None

    spectral_config = SpectralConfig(
        penalty_strength=config.spectral_penalty_strength,
    )

    # VerbNounConfig derives alphas from variance ratios (no hardcoded values)
    # alpha_scale controls steepness of sigmoid transition
    verb_noun_config = (
        VerbNounConfig(alpha_scale=config.verb_noun_strength * 2.0)
        if config.enable_verb_noun
        else None
    )

    # Pre-compute per-layer intrinsic dimension (dimensional hierarchy)
    intrinsic_dims_by_layer: dict[int, float] = {}
    hidden_dim = _infer_hidden_dim(target_weights)

    if config.enable_intrinsic_dim_gating:
        intrinsic_dims_by_layer = _compute_layer_intrinsic_dims(
            target_weights,
            layer_indices,
            config.intrinsic_dim_threshold,
            backend=b,
        )
        if intrinsic_dims_by_layer:
            complexity_vals = [d / hidden_dim for d in intrinsic_dims_by_layer.values()]
            mean_complexity = sum(complexity_vals) / len(complexity_vals)
            logger.info(
                "INTRINSIC DIM: hidden_dim=%d, mean_complexity=%.3f",
                hidden_dim,
                mean_complexity,
            )

    # Pre-compute per-layer curvature profile (manifold topology)
    curvature_by_layer: dict[int, tuple[str, float]] = {}  # (sign, anisotropy)
    if config.enable_curvature_gating:
        curvature_by_layer = _compute_layer_curvature_profiles(
            target_weights,
            layer_indices,
            backend=b,
        )
        if curvature_by_layer:
            signs = [s for s, _ in curvature_by_layer.values()]
            positive_frac = signs.count("positive") / len(signs) if signs else 0.0
            logger.info(
                "CURVATURE: %.1f%% positive curvature layers (interference risk)",
                positive_frac * 100,
            )

    # Zipper state
    omega_by_layer: dict[int, "Array"] = {}

    # Clear GPU cache before blend loop to release any lazy computations
    # from intrinsic_dim SVD, curvature estimation, or earlier stages
    if hasattr(b, "clear_cache"):
        b.clear_cache()
        logger.info("GPU cache cleared before blend loop")

    # Start with target weights (all operations via Backend protocol)
    logger.info("BLEND: Using Backend protocol (GPU-accelerated)")
    merged: dict[str, "Array"] = {k: b.array(v) for k, v in target_weights.items()}
    b.eval(*merged.values())

    rotate_metrics: dict[str, Any] = {
        "procrustes_errors": [],
        "rotations_applied": 0,
        "identity_used": 0,
        "transport_guided_applied": 0,
        "gw_distances": [],
        "zipper_propagations": 0,
        "zipper_applications": 0,
    }
    blend_metrics: dict[str, Any] = {
        "effective_alphas": [],
        "spectral_adjustments": 0,
        "svd_blended": 0,
        "correlation_weighted": 0,
        "verb_noun_modulated": 0,
        "hard_swaps": 0,
        "intrinsic_dim_adjustments": 0,
        "complexity_ratios": [],
        "domain_signals_applied": 0,
        "domain_signals_enabled": config.enable_domain_signals,
        "curvature_adjustments": 0,
        "curvature_gating_enabled": config.enable_curvature_gating,
    }

    # Log domain signal status
    if config.enable_domain_signals:
        logger.info(
            "DOMAIN SIGNALS: enabled (strength=%.2f). "
            "Note: Full domain signal gating requires DomainSignalProfile computation.",
            config.domain_signal_strength,
        )

    if hard_swap_layers is None:
        hard_swap_layers = set()

    total_weights = len(target_weights)
    processed = 0

    for key in sorted(target_weights.keys()):
        if key not in source_weights:
            continue

        processed += 1
        if processed % 100 == 0:
            logger.info("BLEND: processed %d/%d weights", processed, total_weights)

        # Dequantize if needed (handles packed quantized weights like 4-bit)
        # then convert to float32 backend arrays
        source_dequant = dequantize_if_needed(
            source_weights[key], key, source_weights, b
        )
        target_dequant = dequantize_if_needed(
            target_weights[key], key, target_weights, b
        )
        source_w = b.astype(b.array(source_dequant), "float32")
        target_w = b.astype(b.array(target_dequant), "float32")
        b.eval(source_w, target_w)

        if source_w.shape != target_w.shape:
            # Project source to target shape using geometry-preserving transformation
            # Different dimensions are compression/expansion levels of same geometry
            logger.warning(
                "SHAPE MISMATCH at %s: source=%s target=%s",
                key,
                source_w.shape,
                target_w.shape,
            )
            source_w, proj_score = _project_weight_to_target_shape(
                source_w, target_w, backend=b
            )
            b.eval(source_w)
            blend_metrics.setdefault("shape_projections", 0)
            blend_metrics["shape_projections"] += 1
            blend_metrics.setdefault("projection_scores", []).append(proj_score)
            logger.debug(
                "BLEND: Projected %s from %s to %s (score=%.3f)",
                key,
                source_weights[key].shape,
                target_w.shape,
                proj_score,
            )

        layer_idx = extract_layer_index_fn(key)
        confidence = layer_confidences.get(layer_idx, 0.0) if layer_idx is not None else 0.0

        # Get base alpha
        if layer_idx is not None and layer_idx in smoothed_alphas:
            effective_alpha = smoothed_alphas[layer_idx]
        else:
            effective_alpha = config.base_alpha

        # Apply thermodynamic phase gating
        # ORDERED: Full alpha (safe for aggressive blend)
        # CRITICAL: Reduce alpha by 30% (near phase boundary, be conservative)
        # DISORDERED: Reduce alpha by 15% (high entropy, slightly conservative)
        phase = (
            config.entropy_phase.lower()
            if isinstance(config.entropy_phase, str)
            else str(config.entropy_phase).split(".")[-1].lower()
        )
        if phase == "critical":
            effective_alpha *= 0.7
            blend_metrics["phase_adjustment"] = "critical_-30%"
        elif phase == "disordered":
            effective_alpha *= 0.85
            blend_metrics["phase_adjustment"] = "disordered_-15%"
        else:
            blend_metrics["phase_adjustment"] = "ordered_no_change"

        # STAGE 3: ROTATE
        transport_blended = None

        can_rotate = (
            config.enable_rotation
            and confidence >= config.rotation_confidence_threshold
            and source_w.ndim == 2
            and target_w.ndim == 2
            and min(source_w.shape) >= config.alignment_rank
        )

        can_transport = config.use_transport_guided and can_rotate and source_w.shape[0] <= 512

        if can_transport:
            # Transport uses scipy internally - convert at boundary
            gw_result = _compute_transport_guided_blend(
                source_w, target_w, effective_alpha, config.transport_coupling_threshold, b
            )
            if gw_result is not None:
                transport_blended, gw_distance = gw_result
                rotate_metrics["transport_guided_applied"] += 1
                rotate_metrics["gw_distances"].append(gw_distance)
            else:
                can_transport = False

        if can_rotate and not can_transport:
            # Procrustes - use backend for GPU-accelerated SVD
            omega_out, procrustes_error = _compute_procrustes_rotation(
                source_w, target_w, rank=config.alignment_rank, backend=b
            )
            rotate_metrics["procrustes_errors"].append(procrustes_error)
            rotate_metrics["rotations_applied"] += 1
        elif not can_transport:
            rotate_metrics["identity_used"] += 1

        # STAGE 5: PROPAGATE (zipper)
        if config.enable_zipper and layer_idx is not None:
            is_residual = _is_residual_output(key)
            is_input_proj = _is_attention_input(key) or _is_mlp_input(key)

            if is_residual and source_w.ndim == 2 and source_w.shape[0] == target_w.shape[0]:
                if config.zipper_use_weight_matching:
                    # Compute permutation via backend
                    P = _compute_weight_matching_permutation(source_w, target_w, b)
                    source_w = b.matmul(P, source_w)
                    omega_by_layer[layer_idx] = P
                    rotate_metrics["zipper_propagations"] += 1
                else:
                    # Compute rotation with backend
                    R, error = _compute_full_rank_rotation(source_w, target_w, backend=b)
                    source_w = b.matmul(R, source_w)
                    omega_by_layer[layer_idx] = R
                    rotate_metrics["zipper_propagations"] += 1
                    rotate_metrics["procrustes_errors"].append(error)
                b.eval(source_w)

            elif is_input_proj and source_w.ndim == 2:
                prev_layer = layer_idx - 1
                if prev_layer in omega_by_layer:
                    omega_in = omega_by_layer[prev_layer]
                    if omega_in.shape[0] == source_w.shape[1]:
                        source_w = b.matmul(source_w, b.transpose(omega_in))
                        b.eval(source_w)
                        rotate_metrics["zipper_applications"] += 1

        # STAGE 4: BLEND

        # 4.0: Hard swap check
        if layer_idx is not None and layer_idx in hard_swap_layers:
            merged[key] = b.astype(source_w, str(target_w.dtype))
            blend_metrics["hard_swaps"] += 1
            blend_metrics["effective_alphas"].append(0.0)
            continue

        # 4.1: Module-specific policy
        if config.enable_module_policy:
            if _is_v_proj(key):
                effective_alpha = config.module_policy_v_alpha
                blend_metrics.setdefault("module_policy_v", 0)
                blend_metrics["module_policy_v"] += 1
            elif _is_o_proj(key):
                effective_alpha = config.module_policy_o_alpha
                blend_metrics.setdefault("module_policy_o", 0)
                blend_metrics["module_policy_o"] += 1

        # 4.2: Spectral penalty (GPU-accelerated via backend)
        if config.enable_spectral_penalty and source_w.ndim >= 1:
            spectral = compute_spectral_metrics(
                source_w, target_w, spectral_config, backend=b
            )
            effective_alpha = apply_spectral_penalty(
                effective_alpha,
                spectral.spectral_confidence,
                config.spectral_penalty_strength,
            )
            blend_metrics["spectral_adjustments"] += 1

        # 4.2.5: Intrinsic dimension gating (dimensional hierarchy)
        # High complexity ratio → trust target more (higher alpha toward 1)
        # Low complexity ratio → simple manifold → can blend aggressively
        if (
            config.enable_intrinsic_dim_gating
            and layer_idx is not None
            and layer_idx in intrinsic_dims_by_layer
            and hidden_dim > 0
        ):
            intrinsic_dim = intrinsic_dims_by_layer[layer_idx]
            complexity_ratio = intrinsic_dim / hidden_dim
            blend_metrics["complexity_ratios"].append(complexity_ratio)

            # Modulate: high complexity → push alpha toward target (1.0)
            # complexity_ratio in [0, 1], typically 0.05-0.5 for LLMs
            # We normalize to [0, 1] assuming max practical ratio is 0.5
            normalized_complexity = min(1.0, complexity_ratio * 2.0)

            # Blend current alpha toward conservative (higher) based on complexity
            complexity_adjustment = normalized_complexity * config.intrinsic_dim_strength
            effective_alpha = effective_alpha + (1.0 - effective_alpha) * complexity_adjustment

            blend_metrics["intrinsic_dim_adjustments"] += 1

        # 4.2.6: Curvature gating (manifold topology risk)
        # POSITIVE curvature → convergent geodesics → interference risk → reduce alpha
        # NEGATIVE curvature → divergent geodesics → safe → slightly increase alpha
        # High anisotropy → directional instability → reduce alpha
        if (
            config.enable_curvature_gating
            and layer_idx is not None
            and layer_idx in curvature_by_layer
        ):
            curv_sign, anisotropy = curvature_by_layer[layer_idx]
            effective_alpha, curv_desc = _apply_curvature_adjustment(
                effective_alpha, curv_sign, anisotropy, config
            )
            blend_metrics["curvature_adjustments"] += 1
            blend_metrics.setdefault("curvature_descriptions", []).append(curv_desc)

        # 4.3: SVD-aware blending
        if transport_blended is not None:
            blended = transport_blended
        elif svd_config is not None and source_w.ndim == 2:
            # SVD blending via backend
            blended = blend_with_svd_awareness(
                source_w, target_w, effective_alpha, svd_config, backend=b
            )
            blend_metrics["svd_blended"] += 1
        else:
            # Core blend via backend (GPU-accelerated)
            blended = (1.0 - effective_alpha) * target_w + effective_alpha * source_w

        # 4.4: Correlation-based dimension weighting
        if config.enable_correlation_weights and source_w.ndim == 2:
            blended = _apply_correlation_weights(
                source_w, target_w, blended, effective_alpha, config, blend_metrics, b
            )

        # 4.5: VerbNoun modulation
        if verb_noun_config is not None and source_w.ndim == 2:
            blended = _apply_verb_noun_modulation(
                source_w,
                target_w,
                blended,
                effective_alpha,
                verb_noun_config,
                blend_metrics,
                b,
            )

        # Clamp alpha and record
        effective_alpha = max(config.alpha_min, min(config.alpha_max, effective_alpha))
        blend_metrics["effective_alphas"].append(effective_alpha)

        merged[key] = b.astype(blended, str(target_w.dtype))

    # Summarize metrics
    _finalize_metrics(rotate_metrics, blend_metrics, config)

    return RotateBlendResult(
        merged_weights=merged,
        rotate_metrics=rotate_metrics,
        blend_metrics=blend_metrics,
    )


def _compute_raw_alphas(
    layer_indices: list[int],
    layer_confidences: dict[int, float],
    refinement_alphas: dict[int, float] | None,
    refinement_strength: float,
) -> dict[int, float]:
    """Compute raw per-layer alphas before smoothing."""
    raw_alphas = {}
    for layer_idx in layer_indices:
        if refinement_alphas is not None and layer_idx in refinement_alphas:
            base_alpha = refinement_alphas[layer_idx]
        else:
            confidence = layer_confidences.get(layer_idx, 0.0)
            base_alpha = 1.0 - (confidence * 0.7)

        if refinement_alphas is not None and layer_idx in refinement_alphas:
            confidence = layer_confidences.get(layer_idx, 0.0)
            conf_alpha = 1.0 - (confidence * 0.7)
            raw_alphas[layer_idx] = (
                refinement_strength * base_alpha + (1.0 - refinement_strength) * conf_alpha
            )
        else:
            raw_alphas[layer_idx] = base_alpha

    return raw_alphas


def _apply_correlation_weights(
    source_w: "Array",
    target_w: "Array",
    blended: "Array",
    effective_alpha: float,
    config: RotateBlendConfig,
    blend_metrics: dict,
    backend: "Backend",
) -> "Array":
    """Apply correlation-based dimension weighting.

    The correlation structure determines per-dimension alpha:
    - High correlation → dimensions agree → use effective_alpha
    - Low correlation → dimensions disagree → use stability bias (trust target)

    The stability bias is derived from the correlation distribution itself:
    - Mean correlation determines confidence in the blend
    - Low mean correlation → higher stability bias
    """
    from modelcypher.core.domain.geometry.dimension_blender import (
        CorrelationWeightConfig,
        compute_correlation_weights,
        compute_dimension_correlations,
    )

    b = backend
    corr_config = CorrelationWeightConfig(
        correlation_scale=config.correlation_scale,
    )

    sample_rows = min(source_w.shape[0], 256)
    source_sample = source_w[:sample_rows, :]
    target_sample = target_w[:sample_rows, :]

    if source_sample.shape == target_sample.shape and source_sample.shape[0] > 1:
        try:
            # dimension_blender uses backend internally
            correlations = compute_dimension_correlations(
                b.transpose(source_sample), b.transpose(target_sample), corr_config, backend=b
            )
            corr_weights = compute_correlation_weights(correlations, corr_config)

            if len(corr_weights) == blended.shape[0]:
                corr_weights_arr = b.array(corr_weights)
                # Derive stability_alpha purely from geometry: coefficient of variation
                # CV = std/mean is dimensionless and captures correlation spread
                # Higher CV (more spread) → need more conservative alpha
                eps = 1e-10
                cv = correlations.std_correlation / (correlations.mean_correlation + eps)
                # Map CV ∈ [0, ∞) to alpha ∈ [0, 1] using hyperbolic transformation
                # cv/(1+cv) is the natural mapping: CV=0→0, CV=1→0.5, CV→∞→1
                # No arbitrary constants - the geometry determines the alpha
                stability_alpha = cv / (1.0 + cv)

                row_alphas = (
                    1.0 - corr_weights_arr
                ) * effective_alpha + corr_weights_arr * stability_alpha
                # Expand dims for broadcasting
                row_alphas_2d = b.expand_dims(row_alphas, axis=1)
                blended = (1.0 - row_alphas_2d) * target_w + row_alphas_2d * source_w
                b.eval(blended)
                blend_metrics["correlation_weighted"] += 1
        except Exception:
            pass

    return blended


def _apply_verb_noun_modulation(
    source_w: "Array",
    target_w: "Array",
    blended: "Array",
    effective_alpha: float,
    verb_noun_config: Any,
    blend_metrics: dict,
    backend: "Backend",
) -> "Array":
    """Apply verb/noun alpha modulation.

    Per-dimension alphas are derived from the variance ratio geometry:
    - High source/target variance ratio → verb-like → high alpha (trust source)
    - Low source/target variance ratio → noun-like → low alpha (trust target)

    Uses sigmoid(log(ratio) * scale) transformation for smooth mapping.
    """
    from modelcypher.core.domain.geometry.verb_noun_classifier import ratio_to_alpha

    b = backend

    # Compute variance along axis 1 using backend
    source_mean = b.mean(source_w, axis=1, keepdims=True)
    target_mean = b.mean(target_w, axis=1, keepdims=True)
    source_var = b.mean((source_w - source_mean) ** 2, axis=1)
    target_var = b.mean((target_w - target_mean) ** 2, axis=1)

    var_ratio = source_var / (target_var + 1e-8)

    # Derive per-row alphas from variance ratio geometry
    # ratio_to_alpha uses sigmoid(log(ratio) * scale)
    b.eval(var_ratio)
    var_ratio_list = [float(b.to_numpy(var_ratio[i])) for i in range(source_w.shape[0])]
    vn_alphas_list = [ratio_to_alpha(r, verb_noun_config.alpha_scale) for r in var_ratio_list]
    vn_alphas = b.array(vn_alphas_list)

    modulated_alphas = (
        1.0 - verb_noun_config.modulation_strength
    ) * effective_alpha + verb_noun_config.modulation_strength * vn_alphas

    # Expand dims for broadcasting
    modulated_alphas_2d = b.expand_dims(modulated_alphas, axis=1)
    blended = (1.0 - modulated_alphas_2d) * target_w + modulated_alphas_2d * source_w
    b.eval(blended)
    blend_metrics["verb_noun_modulated"] += 1

    return blended


def _finalize_metrics(
    rotate_metrics: dict,
    blend_metrics: dict,
    config: RotateBlendConfig,
) -> None:
    """Finalize and log metrics."""
    rotate_metrics["rotations_applied"] = int(rotate_metrics["rotations_applied"])
    rotate_metrics["identity_used"] = int(rotate_metrics["identity_used"])
    rotate_metrics["transport_guided_applied"] = int(rotate_metrics["transport_guided_applied"])

    if rotate_metrics["gw_distances"]:
        gw_dists = rotate_metrics["gw_distances"]
        rotate_metrics["mean_gw_distance"] = sum(gw_dists) / len(gw_dists)

    if blend_metrics["effective_alphas"]:
        alphas = blend_metrics["effective_alphas"]
        blend_metrics["mean_alpha"] = sum(alphas) / len(alphas)
        blend_metrics["min_alpha"] = min(alphas)
        blend_metrics["max_alpha"] = max(alphas)

    logger.info(
        "ROTATE: %d procrustes, %d transport, %d identity",
        rotate_metrics["rotations_applied"],
        rotate_metrics["transport_guided_applied"],
        rotate_metrics["identity_used"],
    )
    if rotate_metrics["transport_guided_applied"] > 0:
        logger.info(
            "GW TRANSPORT: mean_distance=%.4f",
            rotate_metrics.get("mean_gw_distance", 0),
        )
    if rotate_metrics["zipper_propagations"] > 0 or rotate_metrics["zipper_applications"] > 0:
        logger.info(
            "ZIPPER: %d propagations, %d applications",
            rotate_metrics["zipper_propagations"],
            rotate_metrics["zipper_applications"],
        )
    logger.info(
        "BLEND: mean_alpha=%.3f, spectral=%d, svd=%d, corr=%d, vn=%d, hard_swap=%d",
        blend_metrics.get("mean_alpha", 0),
        blend_metrics["spectral_adjustments"],
        blend_metrics["svd_blended"],
        blend_metrics["correlation_weighted"],
        blend_metrics["verb_noun_modulated"],
        blend_metrics["hard_swaps"],
    )
    if blend_metrics.get("module_policy_v", 0) > 0 or blend_metrics.get("module_policy_o", 0) > 0:
        logger.info(
            "MODULE POLICY: v_proj=%d (α=%.1f), o_proj=%d (α=%.1f)",
            blend_metrics.get("module_policy_v", 0),
            config.module_policy_v_alpha,
            blend_metrics.get("module_policy_o", 0),
            config.module_policy_o_alpha,
        )

    # Log intrinsic dimension metrics
    if blend_metrics.get("intrinsic_dim_adjustments", 0) > 0:
        complexity_ratios = blend_metrics.get("complexity_ratios", [])
        if complexity_ratios:
            blend_metrics["mean_complexity_ratio"] = sum(complexity_ratios) / len(complexity_ratios)
            blend_metrics["min_complexity_ratio"] = min(complexity_ratios)
            blend_metrics["max_complexity_ratio"] = max(complexity_ratios)
            logger.info(
                "INTRINSIC DIM: %d adjustments, complexity=%.3f (%.3f-%.3f)",
                blend_metrics["intrinsic_dim_adjustments"],
                blend_metrics["mean_complexity_ratio"],
                blend_metrics["min_complexity_ratio"],
                blend_metrics["max_complexity_ratio"],
            )

    # Log curvature gating metrics
    if blend_metrics.get("curvature_adjustments", 0) > 0:
        curv_descriptions = blend_metrics.get("curvature_descriptions", [])
        if curv_descriptions:
            from collections import Counter

            desc_counts = Counter(curv_descriptions)
            most_common = desc_counts.most_common(3)
            desc_summary = ", ".join(f"{d}:{c}" for d, c in most_common)
            logger.info(
                "CURVATURE: %d adjustments, types=[%s]",
                blend_metrics["curvature_adjustments"],
                desc_summary,
            )

    # Log shape projection metrics (cross-dimension geometry preservation)
    if blend_metrics.get("shape_projections", 0) > 0:
        proj_scores = blend_metrics.get("projection_scores", [])
        if proj_scores:
            blend_metrics["mean_projection_score"] = sum(proj_scores) / len(proj_scores)
            blend_metrics["min_projection_score"] = min(proj_scores)
            blend_metrics["max_projection_score"] = max(proj_scores)
            logger.info(
                "SHAPE PROJECTION: %d weights projected, mean_score=%.3f (%.3f-%.3f)",
                blend_metrics["shape_projections"],
                blend_metrics["mean_projection_score"],
                blend_metrics["min_projection_score"],
                blend_metrics["max_projection_score"],
            )


# =============================================================================
# ROTATION HELPERS
# =============================================================================


def _compute_procrustes_rotation(
    source_w: "Array",
    target_w: "Array",
    rank: int = 32,
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """Compute optimal rotation matrix using Procrustes analysis.

    Args:
        source_w: Source weight matrix (Backend array)
        target_w: Target weight matrix (Backend array)
        rank: Rank for truncated SVD
        backend: Backend for GPU-accelerated SVD (required)

    Returns:
        Tuple of (rotation_matrix, error)
    """
    b = backend or get_default_backend()
    min_dim = min(source_w.shape[0], source_w.shape[1], rank)

    if min_dim < 2:
        return b.eye(rank, dtype="float32"), 0.0

    try:
        # GPU-accelerated path
        source_f32 = b.astype(source_w, "float32")
        target_f32 = b.astype(target_w, "float32")

        # Full SVD to get U, S, Vt
        u_s, s_s, vt_s = b.svd(source_f32, compute_uv=True)
        u_t, s_t, vt_t = b.svd(target_f32, compute_uv=True)
        b.eval(u_s, s_s, u_t, s_t)

        # Truncate to k dimensions
        k = min(min_dim, s_s.shape[0], s_t.shape[0])
        u_s_k = u_s[:, :k]
        u_t_k = u_t[:, :k]

        # Compute optimal rotation: M = U_s^T @ U_t
        m = b.matmul(b.transpose(u_s_k), u_t_k)
        u_m, _, vt_m = b.svd(m, compute_uv=True)
        b.eval(u_m, vt_m)

        # omega = U_m @ V_m^T
        omega = b.matmul(u_m, vt_m)
        b.eval(omega)

        # Check determinant sign (need scalar extraction)
        det_val = float(b.to_numpy(b.det(omega)))
        if det_val < 0:
            # Flip last column of U_m
            u_m_last = u_m[:, -1] * -1.0
            # Reconstruct u_m with flipped last column
            u_m_fixed = b.concatenate([u_m[:, :-1], b.expand_dims(u_m_last, axis=1)], axis=1)
            omega = b.matmul(u_m_fixed, vt_m)
            b.eval(omega)

        # Compute error
        projected = b.matmul(b.matmul(omega, b.transpose(u_s_k)), source_f32)
        target_proj = b.matmul(b.transpose(u_t_k), target_f32)
        diff = projected - target_proj
        b.eval(diff, target_proj)

        error_norm = float(b.to_numpy(b.norm(b.reshape(diff, (-1,)))))
        target_norm = float(b.to_numpy(b.norm(b.reshape(target_proj, (-1,)))))
        error = error_norm / (target_norm + 1e-8)

        # Pad to rank if needed
        if omega.shape[0] < rank:
            padded = b.eye(rank, dtype="float32")
            # Copy omega into padded
            for i in range(omega.shape[0]):
                for j in range(omega.shape[1]):
                    padded = b.array(b.to_numpy(padded))  # Ensure mutable
            # Use scatter-style update - simpler to just build from scratch
            omega_np = b.to_numpy(omega)
            padded_np = b.to_numpy(b.eye(rank, dtype="float32"))
            padded_np[: omega.shape[0], : omega.shape[1]] = omega_np
            omega = b.array(padded_np)

        return omega, float(error)

    except Exception:
        return b.eye(rank, dtype="float32"), 0.0


def _compute_transport_guided_blend(
    source_w: "Array",
    target_w: "Array",
    alpha: float,
    coupling_threshold: float,
    backend: "Backend",
) -> tuple["Array", float] | None:
    """Compute transport-guided blend using Gromov-Wasserstein."""
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        Config as GWConfig,
    )
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        GromovWassersteinDistance,
    )
    from modelcypher.core.domain.geometry.transport_guided_merger import (
        TransportGuidedMerger,
    )

    b = backend

    try:
        # Convert to lists for GW distance (uses Python internally)
        b.eval(source_w, target_w)
        source_points = b.to_numpy(source_w).tolist()
        target_points = b.to_numpy(target_w).tolist()

        source_dist = GromovWassersteinDistance.compute_pairwise_distances(source_points)
        target_dist = GromovWassersteinDistance.compute_pairwise_distances(target_points)

        gw_config = GWConfig(
            epsilon=0.1,
            max_outer_iterations=30,
            convergence_threshold=1e-4,
        )

        gw_result = GromovWassersteinDistance.compute(
            source_distances=source_dist,
            target_distances=target_dist,
            config=gw_config,
        )

        if not gw_result.converged and gw_result.iterations == 0:
            return None

        merge_config = TransportGuidedMerger.Config(
            coupling_threshold=coupling_threshold,
            normalize_rows=True,
            blend_alpha=alpha,
        )

        merged = TransportGuidedMerger.synthesize(
            source_weights=source_points,
            target_weights=target_points,
            transport_plan=gw_result.coupling,
            config=merge_config,
        )

        if merged is None:
            return None

        blended = b.array(merged)
        blended = b.astype(blended, str(source_w.dtype))
        b.eval(blended)
        return blended, gw_result.distance

    except Exception:
        return None


def _compute_weight_matching_permutation(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend",
) -> "Array":
    """Compute optimal permutation matrix using weight matching (LAP)."""
    b = backend
    n = source_w.shape[0]

    # Compute similarity matrix S = source @ target^T
    S = b.matmul(source_w, b.transpose(target_w))
    b.eval(S)

    # Convert to numpy for scipy LAP (specialized algorithm)
    S_np = b.to_numpy(S)

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-S_np)
    except ImportError:
        # Fallback: greedy assignment
        row_ind = list(range(n))
        col_ind = [0] * n
        available = set(range(n))

        for i in range(n):
            best_j = max(available, key=lambda j: S_np[i, j])
            col_ind[i] = best_j
            available.remove(best_j)
        col_ind = list(col_ind)

    # Build permutation matrix using backend
    P = b.zeros((n, n), dtype="float32")
    # Need to set P[col_ind, row_ind] = 1.0
    # Since backends don't support fancy indexing, build via numpy then convert
    P_np = b.to_numpy(P)
    for i, (c, r) in enumerate(zip(col_ind, row_ind)):
        P_np[c, r] = 1.0
    P = b.array(P_np)
    b.eval(P)

    return P


def _compute_full_rank_rotation(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """Compute full-rank orthogonal rotation using Procrustes.

    Args:
        source_w: Source weight matrix (Backend array)
        target_w: Target weight matrix (Backend array)
        backend: Backend for GPU-accelerated SVD (required)

    Returns:
        Tuple of (rotation_matrix, error)
    """
    b = backend or get_default_backend()

    try:
        # GPU-accelerated path
        source_f32 = b.astype(source_w, "float32")
        target_f32 = b.astype(target_w, "float32")

        # M = target @ source.T
        M = b.matmul(target_f32, b.transpose(source_f32))

        # SVD of M
        U, _, Vt = b.svd(M, compute_uv=True)
        b.eval(U, Vt)

        # R = U @ Vt
        R = b.matmul(U, Vt)
        b.eval(R)

        # Check determinant sign
        det_val = float(b.to_numpy(b.det(R)))
        if det_val < 0:
            # Flip last column of U
            U_last = U[:, -1] * -1.0
            U_fixed = b.concatenate([U[:, :-1], b.expand_dims(U_last, axis=1)], axis=1)
            R = b.matmul(U_fixed, Vt)
            b.eval(R)

        # Compute error
        aligned = b.matmul(R, source_f32)
        diff = aligned - target_f32
        b.eval(aligned, diff)

        error_norm = float(b.to_numpy(b.norm(b.reshape(diff, (-1,)))))
        target_norm = float(b.to_numpy(b.norm(b.reshape(target_f32, (-1,)))))
        error = error_norm / (target_norm + 1e-8)

        return R, float(error)

    except Exception:
        n = source_w.shape[0]
        return b.eye(n, dtype="float32"), 1.0


# =============================================================================
# SHAPE PROJECTION HELPERS (Cross-Dimension Geometry Preservation)
# =============================================================================


def _project_weight_to_target_shape(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """
    Project source weight matrix to target shape using geometry-preserving SVD.

    Different dimensions are compression/expansion levels of the same underlying
    geometry. We use the Gram matrix insight: K = X @ X^T captures relational
    geometry independent of feature dimension.

    For weight matrices:
    - Row mismatch (output dim): Project via left singular vectors
    - Col mismatch (input dim): Project via right singular vectors

    The projection preserves the maximum amount of geometric structure by:
    1. Computing SVD of source weights
    2. Truncating/expanding singular value decomposition to target shape
    3. Reconstructing with preserved top singular values

    Args:
        source_w: Source weight matrix (any shape)
        target_w: Target weight matrix (shape we need to match)
        backend: Backend for GPU-accelerated operations

    Returns:
        Tuple of (projected_weights, alignment_score)
        alignment_score: Fraction of variance preserved (1.0 = perfect)
    """
    b = backend or get_default_backend()

    source_shape = source_w.shape
    target_shape = target_w.shape

    # Handle 1D tensors (biases, norms)
    if source_w.ndim == 1:
        return _project_1d(source_w, target_shape[0], b)

    # Handle 2D weight matrices
    if source_w.ndim == 2:
        return _project_2d(source_w, target_shape, b)

    # For higher dimensions, reshape to 2D, project, reshape back
    # Flatten all but last dim, project, then reshape
    original_shape = source_shape
    flat_source = b.reshape(source_w, (-1, source_shape[-1]))
    flat_target_shape = (-1, target_shape[-1]) if len(target_shape) > 1 else target_shape

    projected_flat, score = _project_2d(
        flat_source,
        (flat_source.shape[0], target_shape[-1]),
        b,
    )

    # Compute output shape preserving leading dims from target
    out_shape = target_shape
    projected = b.reshape(projected_flat, out_shape)
    b.eval(projected)

    return projected, score


def _project_1d(
    source: "Array",
    target_size: int,
    backend: "Backend",
) -> tuple["Array", float]:
    """Project 1D tensor (bias, norm) to target size."""
    b = backend
    source_size = source.shape[0]

    if source_size == target_size:
        return source, 1.0

    if source_size > target_size:
        # Truncate: keep first target_size elements
        # For biases, the first dimensions typically correspond to
        # the most important output neurons
        projected = source[:target_size]
        # Score based on energy preserved
        total_energy = float(b.to_numpy(b.sum(source**2)))
        kept_energy = float(b.to_numpy(b.sum(projected**2)))
        score = kept_energy / (total_energy + 1e-10)
    else:
        # Expand: pad with zeros (preserves existing geometry)
        padding = b.zeros((target_size - source_size,), dtype=str(source.dtype))
        projected = b.concatenate([source, padding], axis=0)
        score = 1.0  # No information lost

    b.eval(projected)
    return projected, score


def _project_2d(
    source: "Array",
    target_shape: tuple[int, ...],
    backend: "Backend",
) -> tuple["Array", float]:
    """
    Project 2D weight matrix using SVD-based geometry preservation.

    The key insight: SVD decomposes W = U @ S @ V^T where:
    - U captures output space geometry (row relationships)
    - V captures input space geometry (column relationships)
    - S captures the importance of each geometric mode

    By truncating/expanding U and V while preserving S, we maintain
    the invariant geometric relationships.
    """
    b = backend
    source_rows, source_cols = source.shape
    target_rows, target_cols = target_shape[0], target_shape[1]

    # If shapes match, nothing to do
    if source_rows == target_rows and source_cols == target_cols:
        return source, 1.0

    try:
        # Compute SVD
        U, S, Vt = b.svd(source, compute_uv=True)
        b.eval(U, S, Vt)

        # Number of singular values
        k = min(len(S), min(source_rows, source_cols))

        # Track variance for alignment score
        total_variance = float(b.to_numpy(b.sum(S**2)))

        # Project U (output dimension) to target_rows
        if source_rows != target_rows:
            U = _resize_orthogonal_basis(U, target_rows, k, b)

        # Project V^T (input dimension) to target_cols
        if source_cols != target_cols:
            V = b.transpose(Vt)
            V = _resize_orthogonal_basis(V, target_cols, k, b)
            Vt = b.transpose(V)

        # Determine how many singular values to keep
        # (limited by the smaller dimension after projection)
        k_out = min(k, target_rows, target_cols)

        # Truncate if needed
        U_k = U[:, :k_out]
        S_k = S[:k_out]
        Vt_k = Vt[:k_out, :]

        # Reconstruct: W = U @ diag(S) @ V^T
        projected = b.matmul(U_k * S_k, Vt_k)
        b.eval(projected)

        # Compute alignment score (variance preserved)
        kept_variance = float(b.to_numpy(b.sum(S_k**2)))
        score = kept_variance / (total_variance + 1e-10)

        return projected, score

    except Exception as e:
        logger.warning("SVD projection failed: %s, using truncate/pad fallback", e)
        return _project_2d_simple(source, target_shape, b)


def _resize_orthogonal_basis(
    basis: "Array",
    target_rows: int,
    k: int,
    backend: "Backend",
) -> "Array":
    """
    Resize orthogonal basis matrix (U or V) to target row count.

    For expansion: Extend with orthogonal random vectors
    For reduction: Keep top-k rows that maximize preserved structure

    The geometry is preserved because:
    - Truncation keeps the most important rows (by position, which corresponds
      to output neuron ordering in most architectures)
    - Expansion adds orthogonal directions that don't interfere with existing geometry
    """
    b = backend
    source_rows = basis.shape[0]

    if source_rows == target_rows:
        return basis

    if source_rows > target_rows:
        # Truncate rows - keep first target_rows
        # In neural networks, earlier neurons often capture more general features
        return basis[:target_rows, :]
    else:
        # Expand rows - add orthogonal padding
        # Initialize with small random values to break symmetry
        padding_rows = target_rows - source_rows
        b.random_seed(42)  # Reproducibility
        padding = b.random_normal((padding_rows, k)) * 0.01

        # Orthogonalize padding against existing basis
        # Using Gram-Schmidt-like projection
        existing = basis[:, :k]
        b.eval(existing)

        # For efficiency, just use small random - the SVD reconstruction
        # will naturally weight these low due to small magnitude
        expanded = b.concatenate([existing, padding], axis=0)
        b.eval(expanded)
        return expanded


def _project_2d_simple(
    source: "Array",
    target_shape: tuple[int, ...],
    backend: "Backend",
) -> tuple["Array", float]:
    """Simple truncate/pad fallback when SVD fails."""
    b = backend
    source_rows, source_cols = source.shape
    target_rows, target_cols = target_shape[0], target_shape[1]

    # Handle rows
    if source_rows > target_rows:
        result = source[:target_rows, :]
    elif source_rows < target_rows:
        padding = b.zeros((target_rows - source_rows, source_cols), dtype=str(source.dtype))
        result = b.concatenate([source, padding], axis=0)
    else:
        result = source

    # Handle cols
    if source_cols > target_cols:
        result = result[:, :target_cols]
    elif source_cols < target_cols:
        padding = b.zeros((result.shape[0], target_cols - source_cols), dtype=str(source.dtype))
        result = b.concatenate([result, padding], axis=1)

    b.eval(result)

    # Score based on overlap ratio
    overlap_rows = min(source_rows, target_rows)
    overlap_cols = min(source_cols, target_cols)
    score = (overlap_rows * overlap_cols) / (source_rows * source_cols)

    return result, score


# =============================================================================
# KEY CLASSIFICATION HELPERS
# =============================================================================


def _is_residual_output(key: str) -> bool:
    """Check if weight is a residual stream output (o_proj, down_proj)."""
    lower = key.lower()
    return any(token in lower for token in ("o_proj", "wo", "out_proj", "down_proj", "w2"))


def _is_attention_input(key: str) -> bool:
    """Check if weight is an attention input projection."""
    lower = key.lower()
    return any(
        token in lower
        for token in ("q_proj", "k_proj", "v_proj", "wq", "wk", "wv", "query", "key", "value")
    )


def _is_mlp_input(key: str) -> bool:
    """Check if weight is an MLP input projection."""
    lower = key.lower()
    return any(token in lower for token in ("gate_proj", "up_proj", "w1", "w3", "fc1"))


def _is_v_proj(key: str) -> bool:
    """Check if weight is the value projection."""
    lower = key.lower()
    return any(token in lower for token in ("v_proj", "wv", ".value"))


def _is_o_proj(key: str) -> bool:
    """Check if weight is the output projection."""
    lower = key.lower()
    return any(token in lower for token in ("o_proj", "wo.", "out_proj"))


# =============================================================================
# INTRINSIC DIMENSION HELPERS (Dimensional Hierarchy)
# =============================================================================


def _infer_hidden_dim(weights: dict[str, Any]) -> int:
    """Infer hidden dimension from weight shapes.

    Works with both NumPy and MLX arrays.
    """
    for key, val in weights.items():
        if "embed" in key.lower() and val.ndim == 2:
            return int(val.shape[-1])
        if "layers.0." in key and ("q_proj" in key or "wq" in key) and val.ndim == 2:
            return int(val.shape[-1])
    # Fallback: find most common dimension
    dims = []
    for val in weights.values():
        if val.ndim == 2:
            dims.extend([int(d) for d in val.shape])
    if dims:
        from collections import Counter

        return Counter(dims).most_common(1)[0][0]
    return 4096  # reasonable default


def _compute_layer_intrinsic_dims(
    weights: dict[str, Any],
    layer_indices: list[int],
    threshold: float = 0.01,
    backend: "Backend | None" = None,
) -> dict[int, float]:
    """
    Compute effective rank (intrinsic dimension) per layer.

    Uses SVD to count singular values above threshold * max_singular_value.
    This gives the "true" dimensionality of the weight manifold.

    Args:
        weights: Model weights (Backend arrays)
        layer_indices: Layer indices to analyze
        threshold: Cutoff ratio (default 1% of max singular value)
        backend: Backend for GPU-accelerated SVD (required)

    Returns:
        Dict mapping layer index to median intrinsic dimension
    """
    b = backend or get_default_backend()
    result: dict[int, float] = {}

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        intrinsic_dims: list[int] = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if min(val.shape) < 32:
                continue

            try:
                # Use GPU-accelerated SVD
                val_arr = b.array(val)
                val_f32 = b.astype(val_arr, "float32")
                s = b.svd(val_f32, compute_uv=False)
                b.eval(s)

                # Count singular values above threshold
                s_np = b.to_numpy(s)
                cutoff = float(s_np[0]) * threshold
                effective_rank = int(sum(1 for sv in s_np if sv > cutoff))
                intrinsic_dims.append(effective_rank)
            except Exception:
                pass

        if intrinsic_dims:
            # Compute median using Python (list of ints)
            sorted_dims = sorted(intrinsic_dims)
            n = len(sorted_dims)
            if n % 2 == 0:
                result[layer_idx] = float((sorted_dims[n // 2 - 1] + sorted_dims[n // 2]) / 2)
            else:
                result[layer_idx] = float(sorted_dims[n // 2])

    return result


# =============================================================================
# CURVATURE HELPERS (Manifold Topology Risk)
# =============================================================================


def _compute_layer_curvature_profiles(
    weights: dict[str, Any],
    layer_indices: list[int],
    sample_rows: int = 64,
    k_neighbors: int = 15,
    backend: "Backend | None" = None,
) -> dict[int, tuple[str, float]]:
    """
    Compute curvature profile per layer.

    Uses sectional curvature estimation to assess manifold topology:
    - POSITIVE curvature: convergent geodesics, interference risk
    - NEGATIVE curvature: divergent geodesics, safe for blending
    - FLAT: Euclidean, neutral
    - MIXED: Variable topology

    Args:
        weights: Model weights (Backend arrays)
        layer_indices: Layer indices to analyze
        sample_rows: Number of rows to sample from weight matrices
        k_neighbors: Number of neighbors for curvature estimation
        backend: Backend for array operations

    Returns:
        Dict mapping layer index to (curvature_sign, anisotropy)
    """
    import random

    from modelcypher.core.domain.geometry.manifold_curvature import (
        CurvatureConfig,
        SectionalCurvatureEstimator,
    )

    b = backend or get_default_backend()
    result: dict[int, tuple[str, float]] = {}
    estimator = SectionalCurvatureEstimator(CurvatureConfig(num_directions=5))

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        curvature_profiles: list[tuple[str, float]] = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if val.shape[0] < sample_rows or val.shape[1] < 16:
                continue

            try:
                # Convert to backend array
                val_arr = b.array(val)
                val_f32 = b.astype(val_arr, "float32")
                b.eval(val_f32)

                # Sample rows using Python random (avoid numpy)
                n_rows = val_f32.shape[0]
                sample_size = min(sample_rows, n_rows)
                indices = random.sample(range(n_rows), sample_size)

                # Extract sampled rows
                points = val_f32[indices, :]
                b.eval(points)

                # Estimate curvature profile (manifold_curvature uses backend internally)
                profile = estimator.estimate_manifold_profile(points, k_neighbors=k_neighbors)

                # Extract sign and anisotropy
                sign = profile.dominant_sign.value  # "positive", "negative", "flat", "mixed"
                anisotropy = profile.global_variance  # Higher = more variable curvature

                curvature_profiles.append((sign, anisotropy))
            except Exception:
                pass

        if curvature_profiles:
            # Use most common sign, mean anisotropy
            from collections import Counter

            signs = [s for s, _ in curvature_profiles]
            most_common_sign = Counter(signs).most_common(1)[0][0]
            anisotropies = [a for _, a in curvature_profiles]
            mean_anisotropy = sum(anisotropies) / len(anisotropies)
            result[layer_idx] = (most_common_sign, mean_anisotropy)

    return result


def _apply_curvature_adjustment(
    base_alpha: float,
    curvature_sign: str,
    anisotropy: float,
    config: RotateBlendConfig,
) -> tuple[float, str]:
    """
    Adjust alpha based on curvature topology.

    The adjustment is derived from the curvature geometry itself:
    - Positive curvature → convergent geodesics → interference risk
      Adjustment = exp(-|curvature_strength|) for natural exponential decay
    - Negative curvature → divergent geodesics → safe for blending
      Adjustment = 1 + (1 - exp(-|curvature_strength|)) for bounded increase
    - Anisotropy → directional instability → penalty proportional to excess anisotropy

    No arbitrary multipliers - the curvature magnitude determines the adjustment.

    Args:
        base_alpha: Current alpha value
        curvature_sign: "positive", "negative", "flat", or "mixed"
        anisotropy: Global curvature variance (0=isotropic, higher=variable)
        config: Stage configuration

    Returns:
        (adjusted_alpha, adjustment_description)
    """
    import math

    alpha = base_alpha
    adjustments = []

    # Curvature strength determines the base adjustment magnitude
    # Natural exponential scaling: curvature_strength controls how strongly we respond
    strength = config.curvature_strength

    # Sign-based adjustment derived from curvature geometry
    if curvature_sign == "positive":
        # Convergent geodesics = interference risk = reduce alpha
        # Use exponential decay: adjustment = exp(-strength)
        # At strength=0: adjustment=1.0 (no change)
        # At strength=0.3: adjustment≈0.74 (26% reduction)
        # At strength=1.0: adjustment≈0.37 (63% reduction)
        adjustment = math.exp(-strength)
        alpha *= adjustment
        adjustments.append("positive_curv")
    elif curvature_sign == "negative":
        # Divergent geodesics = safe = bounded increase
        # Use: 1 + (1 - exp(-strength)) * 0.3 for bounded increase
        # At strength=0: adjustment=1.0 (no change)
        # At strength=0.3: adjustment≈1.08 (8% increase)
        # At strength=1.0: adjustment≈1.19 (19% increase, bounded)
        boost_factor = (1.0 - math.exp(-strength)) * 0.3
        adjustment = 1.0 + boost_factor
        alpha = min(alpha * adjustment, 0.95)  # Cap at 0.95
        adjustments.append("negative_curv")
    elif curvature_sign == "mixed":
        # Variable topology = moderate caution
        # Use geometric mean between flat and positive behavior
        adjustment = math.exp(-strength * 0.5)
        alpha *= adjustment
        adjustments.append("mixed_curv")
    # flat = no adjustment (Euclidean-like, no risk)

    # Anisotropy-based adjustment derived from excess anisotropy
    # Only apply when anisotropy exceeds threshold - use the excess as the penalty basis
    if anisotropy > config.curvature_anisotropy_threshold:
        # Excess anisotropy above threshold determines penalty
        excess_anisotropy = anisotropy - config.curvature_anisotropy_threshold
        # Normalize: assuming anisotropy max is ~2.0 above threshold
        normalized_excess = min(1.0, excess_anisotropy / 2.0)
        # Exponential penalty based on excess
        aniso_penalty = 1.0 - math.exp(-strength * normalized_excess)
        alpha *= 1.0 - aniso_penalty
        adjustments.append("high_aniso")

    # Clamp to valid range
    alpha = max(0.0, min(0.95, alpha))

    description = "+".join(adjustments) if adjustments else "flat_no_adj"
    return alpha, description
