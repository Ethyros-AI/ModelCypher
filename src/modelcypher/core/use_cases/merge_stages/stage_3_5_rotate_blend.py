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
from typing import Any, Callable

import numpy as np

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

    # SVD
    enable_svd_blending: bool = False
    svd_rank_ratio: float = 0.5
    high_rank_alpha: float = 0.7
    low_rank_alpha: float = 0.3

    # Correlation
    enable_correlation_weights: bool = False
    correlation_scale: float = 1.0
    stability_alpha: float = 0.3

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

    merged_weights: dict[str, np.ndarray]
    rotate_metrics: dict[str, Any]
    blend_metrics: dict[str, Any]


def stage_rotate_blend_propagate(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    dimension_correlations: dict,
    layer_indices: list[int],
    config: RotateBlendConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    refinement_alphas: dict[int, float] | None = None,
    hard_swap_layers: set[int] | None = None,
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
        compute_spectral_metrics,
        apply_spectral_penalty,
    )
    from modelcypher.core.domain.geometry.task_singular_vectors import (
        SVDBlendConfig,
        blend_with_svd_awareness,
    )
    from modelcypher.core.domain.geometry.verb_noun_classifier import (
        VerbNounConfig,
    )

    has_dimension_correlations = intersection_map_obj is not None and bool(
        dimension_correlations
    )

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
    svd_config = (
        SVDBlendConfig(
            rank_ratio=config.svd_rank_ratio,
            high_rank_alpha=config.high_rank_alpha,
            low_rank_alpha=config.low_rank_alpha,
        )
        if config.enable_svd_blending
        else None
    )

    spectral_config = SpectralConfig(
        penalty_strength=config.spectral_penalty_strength,
    )

    verb_noun_config = (
        VerbNounConfig(
            verb_alpha=0.8,
            noun_alpha=0.2,
            modulation_strength=config.verb_noun_strength,
        )
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
        )
        if intrinsic_dims_by_layer:
            mean_complexity = np.mean([
                d / hidden_dim for d in intrinsic_dims_by_layer.values()
            ])
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
        )
        if curvature_by_layer:
            signs = [s for s, _ in curvature_by_layer.values()]
            positive_frac = signs.count("positive") / len(signs) if signs else 0.0
            logger.info(
                "CURVATURE: %.1f%% positive curvature layers (interference risk)",
                positive_frac * 100,
            )

    # Zipper state
    omega_by_layer: dict[int, np.ndarray] = {}

    # Start with target weights
    merged = {k: np.asarray(v) for k, v in target_weights.items()}

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

        source_w = np.asarray(source_weights[key], dtype=np.float32)
        target_w = np.asarray(target_weights[key], dtype=np.float32)

        if source_w.shape != target_w.shape:
            continue

        layer_idx = extract_layer_index_fn(key)
        confidence = (
            layer_confidences.get(layer_idx, 0.0) if layer_idx is not None else 0.0
        )

        # Get base alpha
        if layer_idx is not None and layer_idx in smoothed_alphas:
            effective_alpha = smoothed_alphas[layer_idx]
        else:
            effective_alpha = config.base_alpha

        # Apply thermodynamic phase gating
        # ORDERED: Full alpha (safe for aggressive blend)
        # CRITICAL: Reduce alpha by 30% (near phase boundary, be conservative)
        # DISORDERED: Reduce alpha by 15% (high entropy, slightly conservative)
        phase = config.entropy_phase.lower() if isinstance(config.entropy_phase, str) else str(config.entropy_phase).split(".")[-1].lower()
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

        can_transport = (
            config.use_transport_guided and can_rotate and source_w.shape[0] <= 512
        )

        if can_transport:
            gw_result = _compute_transport_guided_blend(
                source_w, target_w, effective_alpha, config.transport_coupling_threshold
            )
            if gw_result is not None:
                transport_blended, gw_distance = gw_result
                rotate_metrics["transport_guided_applied"] += 1
                rotate_metrics["gw_distances"].append(gw_distance)
            else:
                can_transport = False

        if can_rotate and not can_transport:
            omega_out, procrustes_error = _compute_procrustes_rotation(
                source_w, target_w, rank=config.alignment_rank
            )
            rotate_metrics["procrustes_errors"].append(procrustes_error)
            rotate_metrics["rotations_applied"] += 1
        elif not can_transport:
            rotate_metrics["identity_used"] += 1

        # STAGE 5: PROPAGATE (zipper)
        if config.enable_zipper and layer_idx is not None:
            is_residual = _is_residual_output(key)
            is_input_proj = _is_attention_input(key) or _is_mlp_input(key)

            if (
                is_residual
                and source_w.ndim == 2
                and source_w.shape[0] == target_w.shape[0]
            ):
                out_dim = source_w.shape[0]

                if config.zipper_use_weight_matching:
                    P = _compute_weight_matching_permutation(source_w, target_w)
                    source_w = P @ source_w
                    omega_by_layer[layer_idx] = P
                    rotate_metrics["zipper_propagations"] += 1
                else:
                    R, error = _compute_full_rank_rotation(source_w, target_w)
                    source_w = R @ source_w
                    omega_by_layer[layer_idx] = R
                    rotate_metrics["zipper_propagations"] += 1
                    rotate_metrics["procrustes_errors"].append(error)

            elif is_input_proj and source_w.ndim == 2:
                prev_layer = layer_idx - 1
                if prev_layer in omega_by_layer:
                    omega_in = omega_by_layer[prev_layer]
                    if omega_in.shape[0] == source_w.shape[1]:
                        source_w = source_w @ omega_in.T
                        rotate_metrics["zipper_applications"] += 1

        # STAGE 4: BLEND

        # 4.0: Hard swap check
        if layer_idx is not None and layer_idx in hard_swap_layers:
            merged[key] = source_w.astype(target_w.dtype)
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

        # 4.2: Spectral penalty
        if config.enable_spectral_penalty and source_w.ndim >= 1:
            spectral = compute_spectral_metrics(source_w, target_w, spectral_config)
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
            blended = blend_with_svd_awareness(
                source_w, target_w, effective_alpha, svd_config
            )
            blend_metrics["svd_blended"] += 1
        else:
            blended = (1.0 - effective_alpha) * target_w + effective_alpha * source_w

        # 4.4: Correlation-based dimension weighting
        if config.enable_correlation_weights and source_w.ndim == 2:
            blended = _apply_correlation_weights(
                source_w, target_w, blended, effective_alpha, config, blend_metrics
            )

        # 4.5: VerbNoun modulation
        if verb_noun_config is not None and source_w.ndim == 2:
            blended = _apply_verb_noun_modulation(
                source_w, target_w, blended, effective_alpha, verb_noun_config, blend_metrics
            )

        # Clamp alpha and record
        effective_alpha = max(config.alpha_min, min(config.alpha_max, effective_alpha))
        blend_metrics["effective_alphas"].append(effective_alpha)

        merged[key] = blended.astype(target_w.dtype)

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
                refinement_strength * base_alpha
                + (1.0 - refinement_strength) * conf_alpha
            )
        else:
            raw_alphas[layer_idx] = base_alpha

    return raw_alphas


def _apply_correlation_weights(
    source_w: np.ndarray,
    target_w: np.ndarray,
    blended: np.ndarray,
    effective_alpha: float,
    config: RotateBlendConfig,
    blend_metrics: dict,
) -> np.ndarray:
    """Apply correlation-based dimension weighting."""
    from modelcypher.core.domain.geometry.dimension_blender import (
        CorrelationWeightConfig,
        compute_dimension_correlations,
        compute_correlation_weights,
    )

    corr_config = CorrelationWeightConfig(
        correlation_scale=config.correlation_scale,
        stability_alpha=config.stability_alpha,
    )

    sample_rows = min(source_w.shape[0], 256)
    source_sample = source_w[:sample_rows, :]
    target_sample = target_w[:sample_rows, :]

    if source_sample.shape == target_sample.shape and source_sample.shape[0] > 1:
        try:
            correlations = compute_dimension_correlations(
                source_sample.T, target_sample.T, corr_config
            )
            corr_weights = compute_correlation_weights(correlations, corr_config)

            if len(corr_weights) == blended.shape[0]:
                row_alphas = (
                    (1.0 - corr_weights) * effective_alpha
                    + corr_weights * config.stability_alpha
                )
                blended = (
                    (1.0 - row_alphas[:, np.newaxis]) * target_w
                    + row_alphas[:, np.newaxis] * source_w
                )
                blend_metrics["correlation_weighted"] += 1
        except Exception:
            pass

    return blended


def _apply_verb_noun_modulation(
    source_w: np.ndarray,
    target_w: np.ndarray,
    blended: np.ndarray,
    effective_alpha: float,
    verb_noun_config: Any,
    blend_metrics: dict,
) -> np.ndarray:
    """Apply verb/noun alpha modulation."""
    source_var = np.var(source_w, axis=1)
    target_var = np.var(target_w, axis=1)

    var_ratio = source_var / (target_var + 1e-8)

    verb_mask = var_ratio > 2.0
    noun_mask = var_ratio < 0.5

    vn_alphas = np.full(source_w.shape[0], effective_alpha, dtype=np.float32)
    vn_alphas[verb_mask] = verb_noun_config.verb_alpha
    vn_alphas[noun_mask] = verb_noun_config.noun_alpha

    modulated_alphas = (
        (1.0 - verb_noun_config.modulation_strength) * effective_alpha
        + verb_noun_config.modulation_strength * vn_alphas
    )

    blended = (
        (1.0 - modulated_alphas[:, np.newaxis]) * target_w
        + modulated_alphas[:, np.newaxis] * source_w
    )
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
    rotate_metrics["transport_guided_applied"] = int(
        rotate_metrics["transport_guided_applied"]
    )

    if rotate_metrics["gw_distances"]:
        rotate_metrics["mean_gw_distance"] = float(
            np.mean(rotate_metrics["gw_distances"])
        )

    if blend_metrics["effective_alphas"]:
        blend_metrics["mean_alpha"] = float(np.mean(blend_metrics["effective_alphas"]))
        blend_metrics["min_alpha"] = float(np.min(blend_metrics["effective_alphas"]))
        blend_metrics["max_alpha"] = float(np.max(blend_metrics["effective_alphas"]))

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
    if (
        rotate_metrics["zipper_propagations"] > 0
        or rotate_metrics["zipper_applications"] > 0
    ):
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
    if (
        blend_metrics.get("module_policy_v", 0) > 0
        or blend_metrics.get("module_policy_o", 0) > 0
    ):
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
            blend_metrics["mean_complexity_ratio"] = float(np.mean(complexity_ratios))
            blend_metrics["min_complexity_ratio"] = float(np.min(complexity_ratios))
            blend_metrics["max_complexity_ratio"] = float(np.max(complexity_ratios))
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


# =============================================================================
# ROTATION HELPERS
# =============================================================================


def _compute_procrustes_rotation(
    source_w: np.ndarray,
    target_w: np.ndarray,
    rank: int = 32,
) -> tuple[np.ndarray, float]:
    """Compute optimal rotation matrix using Procrustes analysis."""
    min_dim = min(source_w.shape[0], source_w.shape[1], rank)
    if min_dim < 2:
        return np.eye(rank, dtype=np.float32), 0.0

    try:
        from scipy.linalg import svd

        u_s, s_s, vt_s = svd(source_w.astype(np.float32), full_matrices=False)
        u_t, s_t, vt_t = svd(target_w.astype(np.float32), full_matrices=False)

        k = min(min_dim, len(s_s), len(s_t))
        u_s = u_s[:, :k]
        u_t = u_t[:, :k]

        m = u_s.T @ u_t
        u_m, _, vt_m = np.linalg.svd(m, full_matrices=False)

        omega = u_m @ vt_m
        if np.linalg.det(omega) < 0:
            u_m[:, -1] *= -1
            omega = u_m @ vt_m

        projected = omega @ u_s.T @ source_w
        error = np.linalg.norm(projected - u_t.T @ target_w) / (
            np.linalg.norm(u_t.T @ target_w) + 1e-8
        )

        if omega.shape[0] < rank:
            padded = np.eye(rank, dtype=np.float32)
            padded[: omega.shape[0], : omega.shape[1]] = omega
            omega = padded

        return omega.astype(np.float32), float(error)

    except Exception:
        return np.eye(rank, dtype=np.float32), 0.0


def _compute_transport_guided_blend(
    source_w: np.ndarray,
    target_w: np.ndarray,
    alpha: float,
    coupling_threshold: float,
) -> tuple[np.ndarray, float] | None:
    """Compute transport-guided blend using Gromov-Wasserstein."""
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        GromovWassersteinDistance,
        Config as GWConfig,
    )
    from modelcypher.core.domain.geometry.transport_guided_merger import (
        TransportGuidedMerger,
    )

    try:
        source_points = source_w.tolist()
        target_points = target_w.tolist()

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

        blended = np.array(merged, dtype=source_w.dtype)
        return blended, gw_result.distance

    except Exception:
        return None


def _compute_weight_matching_permutation(
    source_w: np.ndarray,
    target_w: np.ndarray,
) -> np.ndarray:
    """Compute optimal permutation matrix using weight matching (LAP)."""
    n = source_w.shape[0]
    S = source_w @ target_w.T

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-S)
    except ImportError:
        row_ind = np.arange(n)
        col_ind = np.zeros(n, dtype=np.int64)
        available = set(range(n))

        for i in range(n):
            best_j = max(available, key=lambda j: S[i, j])
            col_ind[i] = best_j
            available.remove(best_j)

    P = np.zeros((n, n), dtype=np.float32)
    P[col_ind, row_ind] = 1.0

    return P


def _compute_full_rank_rotation(
    source_w: np.ndarray,
    target_w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute full-rank orthogonal rotation using Procrustes."""
    M = target_w @ source_w.T

    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=True)

        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        aligned = R @ source_w
        error = np.linalg.norm(aligned - target_w) / (np.linalg.norm(target_w) + 1e-8)

        return R.astype(np.float32), float(error)

    except np.linalg.LinAlgError:
        n = source_w.shape[0]
        return np.eye(n, dtype=np.float32), 1.0


# =============================================================================
# KEY CLASSIFICATION HELPERS
# =============================================================================


def _is_residual_output(key: str) -> bool:
    """Check if weight is a residual stream output (o_proj, down_proj)."""
    lower = key.lower()
    return any(
        token in lower for token in ("o_proj", "wo", "out_proj", "down_proj", "w2")
    )


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
    return any(
        token in lower for token in ("gate_proj", "up_proj", "w1", "w3", "fc1")
    )


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


def _infer_hidden_dim(weights: dict[str, np.ndarray]) -> int:
    """Infer hidden dimension from weight shapes."""
    for key, val in weights.items():
        if "embed" in key.lower() and val.ndim == 2:
            return val.shape[-1]
        if "layers.0." in key and ("q_proj" in key or "wq" in key) and val.ndim == 2:
            return val.shape[-1]
    # Fallback: find most common dimension
    dims = []
    for val in weights.values():
        if val.ndim == 2:
            dims.extend(val.shape)
    if dims:
        from collections import Counter
        return Counter(dims).most_common(1)[0][0]
    return 4096  # reasonable default


def _compute_layer_intrinsic_dims(
    weights: dict[str, np.ndarray],
    layer_indices: list[int],
    threshold: float = 0.01,
) -> dict[int, float]:
    """
    Compute effective rank (intrinsic dimension) per layer.

    Uses SVD to count singular values above threshold * max_singular_value.
    This gives the "true" dimensionality of the weight manifold.

    Args:
        weights: Model weights
        layer_indices: Layer indices to analyze
        threshold: Cutoff ratio (default 1% of max singular value)

    Returns:
        Dict mapping layer index to median intrinsic dimension
    """
    result: dict[int, float] = {}

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        intrinsic_dims = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if min(val.shape) < 32:
                continue

            try:
                _, s, _ = np.linalg.svd(val.astype(np.float32), full_matrices=False)
                cutoff = s[0] * threshold
                effective_rank = int(np.sum(s > cutoff))
                intrinsic_dims.append(effective_rank)
            except Exception:
                pass

        if intrinsic_dims:
            result[layer_idx] = float(np.median(intrinsic_dims))

    return result


# =============================================================================
# CURVATURE HELPERS (Manifold Topology Risk)
# =============================================================================


def _compute_layer_curvature_profiles(
    weights: dict[str, np.ndarray],
    layer_indices: list[int],
    sample_rows: int = 64,
    k_neighbors: int = 15,
) -> dict[int, tuple[str, float]]:
    """
    Compute curvature profile per layer.

    Uses sectional curvature estimation to assess manifold topology:
    - POSITIVE curvature: convergent geodesics, interference risk
    - NEGATIVE curvature: divergent geodesics, safe for blending
    - FLAT: Euclidean, neutral
    - MIXED: Variable topology

    Args:
        weights: Model weights
        layer_indices: Layer indices to analyze
        sample_rows: Number of rows to sample from weight matrices
        k_neighbors: Number of neighbors for curvature estimation

    Returns:
        Dict mapping layer index to (curvature_sign, anisotropy)
    """
    from modelcypher.core.domain.geometry.manifold_curvature import (
        SectionalCurvatureEstimator,
        CurvatureConfig,
    )

    result: dict[int, tuple[str, float]] = {}
    estimator = SectionalCurvatureEstimator(CurvatureConfig(num_directions=5))

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        curvature_profiles = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if val.shape[0] < sample_rows or val.shape[1] < 16:
                continue

            try:
                # Sample rows as points on the manifold
                indices = np.random.choice(val.shape[0], size=min(sample_rows, val.shape[0]), replace=False)
                points = val[indices].astype(np.float32)

                # Estimate curvature profile
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
            mean_anisotropy = float(np.mean([a for _, a in curvature_profiles]))
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

    Args:
        base_alpha: Current alpha value
        curvature_sign: "positive", "negative", "flat", or "mixed"
        anisotropy: Global curvature variance (0=isotropic, higher=variable)
        config: Stage configuration

    Returns:
        (adjusted_alpha, adjustment_description)
    """
    alpha = base_alpha
    adjustments = []

    # Sign-based adjustment
    if curvature_sign == "positive":
        # Convergent geodesics = interference risk = reduce alpha
        adjustment = 1.0 - (0.3 * config.curvature_strength)
        alpha *= adjustment
        adjustments.append("positive_curv")
    elif curvature_sign == "negative":
        # Divergent geodesics = safe = slightly increase alpha
        adjustment = 1.0 + (0.1 * config.curvature_strength)
        alpha *= adjustment
        adjustments.append("negative_curv")
    elif curvature_sign == "mixed":
        # Variable = moderate caution
        adjustment = 1.0 - (0.15 * config.curvature_strength)
        alpha *= adjustment
        adjustments.append("mixed_curv")
    # flat = no adjustment

    # Anisotropy-based adjustment (high variance = instability)
    if anisotropy > config.curvature_anisotropy_threshold:
        aniso_penalty = 0.2 * config.curvature_strength
        alpha *= (1.0 - aniso_penalty)
        adjustments.append("high_aniso")

    # Clamp to valid range
    alpha = max(0.0, min(0.95, alpha))

    description = "+".join(adjustments) if adjustments else "flat_no_adj"
    return alpha, description
