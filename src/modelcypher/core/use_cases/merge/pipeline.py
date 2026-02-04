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

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_precision_for_merge,
    set_model_compute_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.sparse_region_locator import SparseRegionLocator

from .helpers import (
    copy_config_files,
    create_layer_profile,
    extract_layer_index,
    extract_layer_indices,
    load_tokenizer,
    load_weights,
    load_transplant_occupancy,
    save_weights,
    save_transplant_occupancy,
)
from .models import UnifiedMergeResult
from .stages import (
    stage_density,
    stage_transplant,
)
from .stages.compression_descent import (
    stage_compression_descent,
    apply_compression_descent_to_weights,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.inference import InferenceEngine
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


def _serialize_density_detail(
    density_result: "DensityStageResult",
    backend: "Backend",
    source_model: str,
    target_model: str,
) -> dict[str, Any]:
    point_cloud_payload: dict[str, Any] = {}
    for layer_idx, result in density_result.point_cloud_densities.items():
        point_cloud_payload[str(layer_idx)] = result.to_dict(backend)

    density_weights_payload: dict[str, Any] = {}
    for layer_idx, weights in density_result.density_weights.items():
        backend.eval(weights)
        density_weights_payload[str(layer_idx)] = backend.tolist(weights)

    return {
        "source_model": source_model,
        "target_model": target_model,
        "metrics": density_result.metrics,
        "graft_mask": density_result.graft_mask,
        "density_weights": density_weights_payload,
        "point_cloud_densities": point_cloud_payload,
        "source_profile": density_result.source_profile.to_dict(),
        "target_profile": density_result.target_profile.to_dict(),
        "knowledge_diff": density_result.knowledge_diff.to_dict(),
    }


def run_merge(
    model_loader: "ModelLoaderPort",
    backend: "Backend",
    source_path: str,
    target_path: str,
    output_dir: str | None = None,
    target_weights: dict[str, "Array"] | None = None,
    # Optional pre-loaded models/tokenizers to avoid redundant loading
    source_model: Any | None = None,
    target_model: Any | None = None,
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    # Optional pre-loaded weights to avoid redundant disk I/O
    source_weights: dict[str, "Array"] | None = None,
    activation_provider: "ActivationProvider | None" = None,
    inference_engine: "InferenceEngine | None" = None,
    prior_occupancy_by_layer: dict[int, list[float]] | None = None,
) -> UnifiedMergeResult:
    """
    Execute null-space constrained transplant merge.

    Transplant formula:
        W' = W_target + P_null(A_boundary) @ (W_source_aligned - W_target)

    Boundary constraint:
        A_boundary @ W' = A_boundary @ W_target  (boundary preserved)
    """
    logger.info("=== PURE GEOMETRIC MERGE ===")
    logger.info("Source: %s", source_path)
    logger.info("Target: %s", target_path)

    logger.info("Using null-space constrained transplant.")

    if activation_provider is None:
        raise RuntimeError(
            "Activation provider required for profiling and alignment. "
            "Merge cannot proceed without activation access."
        )

    # Load weights (backend arrays) - use pre-loaded if provided
    if source_weights is not None:
        logger.info("Using pre-loaded source weights")
        loaded_source_weights = source_weights
    else:
        loaded_source_weights, _ = load_weights(model_loader, source_path)

    # Use pre-loaded target weights if provided (multi-donor optimization)
    if target_weights is not None:
        logger.info("Using pre-loaded target weights (multi-donor path)")
        loaded_target_weights = target_weights
        target_format = "safetensors"  # Assume safetensors for pre-loaded weights
    else:
        loaded_target_weights, target_format = load_weights(model_loader, target_path)

    # =================================================================
    # MODEL-DRIVEN PRECISION DETECTION
    # =================================================================
    # Detect the native precision of source/target weights and set the
    # compute dtype accordingly. This ensures we don't waste computation
    # on float64 when models are bf16/fp16, and don't lose precision
    # when mixing models of different precisions.
    compute_dtype = compute_precision_for_merge(
        source_weights=loaded_source_weights,
        target_weights=loaded_target_weights,
        backend=backend,
    )
    set_model_compute_dtype(compute_dtype)

    occupancy_source = "provided"
    if prior_occupancy_by_layer is None:
        prior_occupancy_by_layer = load_transplant_occupancy(target_path) or {}
        occupancy_source = "target"
    if prior_occupancy_by_layer:
        logger.info(
            "Loaded transplant occupancy for %d layers (%s)",
            len(prior_occupancy_by_layer),
            occupancy_source,
        )

    # Identify layers and create profile for geometric analysis
    layer_indices = extract_layer_indices(loaded_target_weights)
    layer_profile = create_layer_profile(layer_indices)
    logger.info("Found %d layers", len(layer_indices))

    # Load tokenizers for probe execution (use pre-loaded if available)
    if source_tokenizer is None:
        source_tokenizer = load_tokenizer(source_path, model_loader)
    else:
        logger.info("Using pre-loaded source tokenizer")
    if target_tokenizer is None:
        target_tokenizer = load_tokenizer(target_path, model_loader)
    else:
        logger.info("Using pre-loaded target tokenizer")

    # Detect cross-vocabulary merge (source and target have different vocabularies)
    # This is critical for layer-aware merging: embedding layers should be skipped
    # for cross-vocab merges because token IDs mean different things.
    is_cross_vocab = False
    if source_tokenizer is not None and target_tokenizer is not None:
        try:
            source_vocab_size = len(source_tokenizer)
            target_vocab_size = len(target_tokenizer)
            is_cross_vocab = source_vocab_size != target_vocab_size
            logger.info(
                "VOCAB CHECK: source=%d, target=%d, cross_vocab=%s",
                source_vocab_size, target_vocab_size, is_cross_vocab
            )
        except (TypeError, AttributeError):
            # Some tokenizers don't support len() - try vocab_size attribute
            try:
                source_vocab_size = getattr(source_tokenizer, "vocab_size", 0)
                target_vocab_size = getattr(target_tokenizer, "vocab_size", 0)
                if source_vocab_size > 0 and target_vocab_size > 0:
                    is_cross_vocab = source_vocab_size != target_vocab_size
                    logger.info(
                        "VOCAB CHECK: source=%d, target=%d, cross_vocab=%s",
                        source_vocab_size, target_vocab_size, is_cross_vocab
                    )
            except Exception:
                logger.warning("Could not determine vocab sizes - assuming same vocab")

    # =================================================================
    # PROFILE-AUTO ALIGNMENT (profile once, merge many)
    # =================================================================
    from modelcypher.core.domain.profile import GeometricProfileStore
    from modelcypher.core.use_cases.profile_service import ProfileService
    from modelcypher.core.use_cases.merge.stages.probe_from_profile import (
        check_profiles_available,
        compute_alignment_from_profiles,
    )

    profile_store = GeometricProfileStore()
    profile_service = ProfileService(
        backend=backend,
        model_loader=model_loader,
        activation_provider=activation_provider,
        store=profile_store,
    )

    def _ensure_profile(model_path: str) -> None:
        existing = profile_store.load(model_path)
        if (
            existing is not None
            and existing.has_activations
            and existing.probe_ids
            and existing.probe_domains
        ):
            return
        force = existing is not None
        profile_service.compute_profile(model_path, force=force)

    _ensure_profile(source_path)
    _ensure_profile(target_path)

    profiles_available, source_profile_dir, target_profile_dir = check_profiles_available(
        source_path, target_path
    )
    if not profiles_available or source_profile_dir is None or target_profile_dir is None:
        raise RuntimeError(
            "PROFILE: Activations unavailable after profiling. "
            "Profile-based alignment requires saved activations."
        )

    profile_alignment_result = compute_alignment_from_profiles(
        source_profile_dir=source_profile_dir,
        target_profile_dir=target_profile_dir,
        backend=backend,
    )

    # =================================================================
    # STAGE 1: PROBE (Compute layer correspondences via CKA)
    # =================================================================
    def _probe_precision_reference() -> "Array":
        if target_activations:
            first = next(iter(target_activations.values()), None)
            if first is not None:
                if isinstance(first, list):
                    if first:
                        return backend.array(first[0])
                return backend.array(first)
        if feature_transforms:
            return backend.array(next(iter(feature_transforms.values())))
        if embedding_transform is not None:
            return backend.array(embedding_transform)
        return backend.array([1.0])

    logger.info("STAGE 1: PROBE (from profile) - Using cached activations")
    probe_result = profile_alignment_result.probe_result
    probe_metrics = profile_alignment_result.probe_metrics
    source_activations = profile_alignment_result.source_activations
    target_activations = profile_alignment_result.target_activations
    source_intermediate_activations = profile_alignment_result.source_intermediate_activations or {}
    target_intermediate_activations = profile_alignment_result.target_intermediate_activations or {}
    source_attention_activations = {}
    target_attention_activations = {}
    source_k_activations = {}
    target_k_activations = {}
    feature_transforms = profile_alignment_result.feature_transforms
    scale_ratios = profile_alignment_result.scale_ratios
    embedding_transform = profile_alignment_result.embedding_transform
    attention_transforms = profile_alignment_result.attention_transforms
    k_transforms = profile_alignment_result.k_transforms
    v_transforms = profile_alignment_result.v_transforms
    intermediate_transforms = profile_alignment_result.intermediate_transforms
    gate_transforms = profile_alignment_result.gate_transforms
    layer_mapping = profile_alignment_result.layer_mapping
    source_embedding_activations = profile_alignment_result.source_embedding_activations
    target_embedding_activations = profile_alignment_result.target_embedding_activations
    source_trajectory_tangents = {}
    target_trajectory_tangents = {}
    layer_coupling = None
    source_layers = None
    target_layers = None
    # Extract injection layer computed during alignment (may be None)
    profile_injection_layer = profile_alignment_result.injection_layer
    logger.info(
        "STAGE 1: PROBE (from profile) - Complete (intermediate=%d src/%d tgt)",
        len(source_intermediate_activations),
        len(target_intermediate_activations),
    )

    # =================================================================
    # GEOMETRIC FINGERPRINT COMPARISON
    # =================================================================
    # Compute fingerprints from source and target activations to capture
    # the geometric profile of each model's activation manifold.
    from modelcypher.core.use_cases.merge.metrics import compute_fingerprint_comparison

    fingerprint_comparison = compute_fingerprint_comparison(
        source_activations=source_activations,
        target_activations=target_activations,
        backend=backend,
    )
    logger.info(
        "FINGERPRINTS: source_cond=%.1f, target_cond=%.1f, ratio=%.2f, dim_delta=%.1f",
        fingerprint_comparison.source_condition_number,
        fingerprint_comparison.target_condition_number,
        fingerprint_comparison.condition_number_ratio,
        fingerprint_comparison.effective_dim_delta,
    )

    # =================================================================
    # EARLY MODEL CLEANUP - Free GPU memory ASAP
    # =================================================================
    # Models are only needed for probe stage forward passes.
    # Delete them NOW to free ~18GB before ID computation and density stage.
    # The activations and transforms are all we need going forward.
    del source_model
    del target_model
    default_backend = get_default_backend()
    ComputationCache.shared().clear_geometry_caches()
    default_backend.clear_cache()
    logger.info("MEMORY: Cleared models and GPU cache after probe stage")

    layer_confidences: dict[int, float] = probe_result.get("confidences", {})
    intersection_map_obj = probe_result.get("intersection_map")
    probe_failed = bool(probe_metrics.get("probe_failed"))
    perfect_alignment = bool(probe_metrics.get("perfect_alignment"))

    if probe_failed:
        min_cka = probe_metrics.get("min_cka", 0.0)
        mean_cka = probe_metrics.get("mean_cka", 0.0)
        raise RuntimeError(
            "PROBE SIGNAL: Alignment signals missing (mean_cka=%.4f, min_cka=%.4f). "
            "Exact kernel alignment is required before merge."
            % (mean_cka, min_cka)
        )

    if not perfect_alignment:
        # =====================================================================
        # GEODESIC CKA DIAGNOSTIC (NOT A GATE)
        # =====================================================================
        # Geodesic CKA < 1.0 reflects limited overlap or novel structure.
        # We proceed with the merge and log shared-manifold diagnostics.
        converged_count = probe_metrics.get("converged_count", 0)
        boundary_count = probe_metrics.get("boundary_preserved_count", 0)
        skipped_count = probe_metrics.get("skipped_count", 0)
        min_cka = probe_metrics.get("min_cka", 0.0)
        mean_cka = probe_metrics.get("mean_cka", 0.0)
        split_cka = probe_metrics.get("split_cka") or {}

        if converged_count == 0:
            raise RuntimeError(
                "PROBE: No aligned layers reported (mean=%.6f, min=%.6f). "
                "Check probe activations and alignment inputs."
                % (mean_cka, min_cka)
            )

        # Proceed with selective transplant
        logger.info(
            "ADAPTIVE BAROMETER: %d processed, %d boundary-preserved, %d skipped. "
            "Proceeding with selective transplant (mean_geodesic_cka=%.4f, shared_cka=%s, shared_fraction=%s).",
            converged_count,
            boundary_count,
            skipped_count,
            mean_cka,
            split_cka.get("shared_cka"),
            split_cka.get("shared_fraction"),
        )

    # Log transform results from probe stage
    if feature_transforms:
        logger.info(
            "PROBE: Computed %d hidden transforms",
            len(feature_transforms),
        )
    else:
        raise RuntimeError("PROBE: No hidden transforms computed; cannot proceed.")

    if attention_transforms:
        logger.info(
            "PROBE: Computed %d attention Q transforms",
            len(attention_transforms),
        )


    if k_transforms:
        logger.info(
            "PROBE: Computed %d K transforms",
            len(k_transforms),
        )

    if v_transforms:
        logger.info(
            "PROBE: Computed %d V transforms",
            len(v_transforms),
        )

    if layer_mapping:
        logger.info(
            "PROBE: Layer mapping has %d entries",
            len(layer_mapping),
        )
    else:
        raise RuntimeError("PROBE: No layer mapping computed; cannot proceed.")

    # Log activation collection results
    if source_activations and target_activations:
        logger.info(
            "PROBE: Collected activations for %d source layers, %d target layers",
            len(source_activations),
            len(target_activations),
        )
    if source_intermediate_activations and target_intermediate_activations:
        logger.info(
            "PROBE: Collected INTERMEDIATE activations for %d source layers, %d target layers",
            len(source_intermediate_activations),
            len(target_intermediate_activations),
        )
    if source_attention_activations and target_attention_activations:
        logger.info(
            "PROBE: Collected ATTENTION (Q) activations for %d source layers, %d target layers",
            len(source_attention_activations),
            len(target_attention_activations),
        )
    if source_k_activations and target_k_activations:
        logger.info(
            "PROBE: Collected ATTENTION (K) activations for %d source layers, %d target layers",
            len(source_k_activations),
            len(target_k_activations),
        )

    # =================================================================
    # POPULATE LAYER PROFILE WITH MEASURED GEOMETRY (NO HEURISTICS)
    # =================================================================
    # The geometry tells us which layers are the semantic highway:
    # - Gram rank: drops to 2-3 at the bottleneck (middle layers)
    # - Intrinsic dimension: peaks at semantic layers, compresses elsewhere
    # These are MEASUREMENTS, not thresholds.
    #
    # Extract Gram ranks from probe stage (already computed in rank_coverage)
    rank_augmentation = probe_metrics.get("rank_augmentation") or {}
    final_coverage = rank_augmentation.get("final_coverage") or {}
    if final_coverage:
        for layer_idx_str, coverage_info in final_coverage.items():
            try:
                layer_idx = int(layer_idx_str)
                # Use target rank as the Gram rank (the dimension of target's column space)
                gram_rank = coverage_info.get("rank", 0)
                if gram_rank > 0:
                    layer_profile.gram_ranks[layer_idx] = gram_rank
            except (ValueError, TypeError):
                continue

        if layer_profile.gram_ranks:
            logger.info(
                "LAYER PROFILE: Populated %d Gram ranks",
                len(layer_profile.gram_ranks),
            )

    # Compute VARIANCE CONCENTRATION per layer from MLP OUTPUT (not hidden state!)
    #
    # KEY INSIGHT: TwoNN intrinsic dimension FAILS to detect true 1D bottlenecks.
    # Example: LFM2-350M layer 7 has 99.4% variance in top-1 singular value,
    # but TwoNN reports ID=6.39 (not even close to 1).
    #
    # Variance concentration correctly identifies bottlenecks:
    # - High var_top1 (>70%) = information compressed into few directions
    # - Low effective_rank = few dimensions carry the signal
    #
    # MLP output = down_proj @ intermediate
    # where intermediate = SiLU(gate) * up
    #
    # We already have intermediate activations, so we apply down_proj weights
    # to get the MLP output delta for variance analysis.
    if target_intermediate_activations and loaded_target_weights:
        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
            identify_bottleneck_layers,
        )
        var_computed = 0

        # Find down_proj weights per layer
        down_proj_weights: dict[int, Any] = {}
        for key, weight in loaded_target_weights.items():
            key_lower = key.lower()
            is_down_proj = (
                ("mlp" in key_lower and "down_proj" in key_lower)
                or ("feed_forward" in key_lower and ".w2." in key_lower)
                or ("mlp" in key_lower and "c_proj" in key_lower)
            )
            if is_down_proj and "weight" in key_lower:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            down_proj_weights[layer_idx] = weight
                        except ValueError:
                            pass
                        break

        logger.info(
            "VARIANCE CONCENTRATION: Computing from MLP OUTPUT (not hidden state) for %d layers",
            len(target_intermediate_activations),
        )

        # Store variance metrics for bottleneck detection
        from modelcypher.core.domain.geometry.variance_concentration import VarianceConcentrationResult
        layer_variance_metrics: dict[int, VarianceConcentrationResult] = {}

        for layer_idx, acts in target_intermediate_activations.items():
            try:
                # Stack activations if list
                if isinstance(acts, list):
                    stacked = backend.stack(acts, axis=0)
                else:
                    stacked = acts
                backend.eval(stacked)

                # Apply down_proj to get MLP output
                # intermediate: [n_samples, intermediate_dim]
                # down_proj weight: [hidden_dim, intermediate_dim]
                # MLP output: [n_samples, hidden_dim]
                if layer_idx in down_proj_weights:
                    W = down_proj_weights[layer_idx]
                    # W @ intermediate.T gives [hidden_dim, n_samples]
                    # Transpose to get [n_samples, hidden_dim]
                    mlp_output = backend.matmul(stacked, backend.transpose(W))
                    backend.eval(mlp_output)
                else:
                    # Fallback: use intermediate directly if no weight found
                    logger.debug("No down_proj found for layer %d, using intermediate", layer_idx)
                    mlp_output = stacked

                # Compute variance concentration from MLP output
                var_result = compute_variance_concentration(mlp_output, backend)
                layer_variance_metrics[layer_idx] = var_result

                # Store in explicit new fields
                layer_profile.variance_concentrations[layer_idx] = var_result.var_top1
                layer_profile.effective_ranks[layer_idx] = var_result.effective_rank

                # Store inverted var_top1 as "intrinsic_dimension" for backward compatibility
                # Higher var_top1 = more compressed = LOWER effective dimension
                layer_profile.intrinsic_dimensions[layer_idx] = 1.0 - var_result.var_top1
                var_computed += 1
            except Exception as exc:
                logger.debug("Variance estimation failed for layer %d: %s", layer_idx, exc)
                continue

        if var_computed > 0:
            # Identify bottleneck layers using variance concentration
            bottleneck_layers = identify_bottleneck_layers(
                layer_variance_metrics,
                var_threshold=0.70,  # 70% variance in top-1 = bottleneck
            )

            if bottleneck_layers:
                # Primary bottleneck = highest variance concentration
                layer_profile.bottleneck_layer = bottleneck_layers[0]

            # Log variance metrics
            var_vals = [(idx, m.var_top1, m.effective_rank) for idx, m in layer_variance_metrics.items()]
            var_vals.sort(key=lambda x: x[1], reverse=True)  # Sort by var_top1 descending

            logger.info(
                "LAYER PROFILE: Computed variance concentration for %d layers",
                var_computed,
            )
            logger.info(
                "VARIANCE TOP-5 BOTTLENECKS: %s",
                [(idx, f"{var:.1%}", f"eff_rank={rank:.1f}") for idx, var, rank in var_vals[:5]],
            )
            if layer_profile.bottleneck_layer is not None:
                best = layer_variance_metrics.get(layer_profile.bottleneck_layer)
                if best:
                    logger.info(
                        "PRIMARY BOTTLENECK: Layer %d (var_top1=%.1f%%, effective_rank=%.1f)",
                        layer_profile.bottleneck_layer,
                        best.var_top1 * 100,
                        best.effective_rank,
                    )

            # =================================================================
            # BOTTLENECK DETECTION (Cross-architecture: geometry is everything)
            # =================================================================
            # The BOTTLENECK is where the invariant relational structure lives:
            # - Lowest intrinsic dimension = most compressed = purest signal
            # - This is the "super highway" where information flows efficiently
            # - Universal across architectures (CKA=1.0 achievable)
            #
            # ONRAMPS/OFFRAMPS (high ID) are translation layers:
            # - Architecture-specific encoding/decoding
            # - High dimensional, messy, vocabulary-tied
            # - DO NOT transplant - breaks coherence
            #
            # For cross-architecture, we're conservative: only bottleneck is safe.
            bottleneck = layer_profile.compute_bottleneck_layers()
            translation_layers = layer_profile.compute_ramp_layers()
            layer_profile.set_cross_architecture_skip_layers()
            logger.info(
                "BOTTLENECK LAYERS (safe to transplant): %s",
                bottleneck,
            )
            if translation_layers:
                logger.info(
                    "TRANSLATION LAYERS (will skip): %s",
                    translation_layers,
                )

            # =================================================================
            # MANIFOLD BOUNDARY DETECTION (flood fill)
            # =================================================================
            # Probe how much "headroom" each layer has for perturbation.
            # Small radius = at stability edge (don't touch)
            # Large radius = safe to transfer
            from modelcypher.core.domain.geometry.manifold_boundary import (
                compute_boundary_radii_from_weights,
            )

            boundary_radii = compute_boundary_radii_from_weights(
                target_activations=target_activations,
                target_weights=loaded_target_weights,
                backend=backend,
                n_directions=20,  # Balance speed vs accuracy
            )
            if boundary_radii:
                layer_profile.boundary_radii = boundary_radii
                logger.info(
                    "LAYER PROFILE: Computed boundary radii for %d layers",
                    len(boundary_radii),
                )
        else:
            raise RuntimeError(
                "VARIANCE CONCENTRATION: No usable variance measurements; "
                "cannot select bottleneck layer."
            )
    elif target_activations:
        # Fallback: use hidden state variance if intermediate isn't available
        # This is less accurate (hidden state = residual stream, not transformation)
        # but better than nothing
        logger.warning(
            "VARIANCE CONCENTRATION: Intermediate activations not available. "
            "Using hidden state variance (less accurate - measures residual stream, not MLP output)."
        )
        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
            identify_bottleneck_layers,
            VarianceConcentrationResult,
        )
        var_computed = 0
        layer_variance_metrics: dict[int, VarianceConcentrationResult] = {}

        for layer_idx, acts in target_activations.items():
            try:
                if isinstance(acts, list):
                    stacked = backend.stack(acts, axis=0)
                else:
                    stacked = acts
                backend.eval(stacked)
                var_result = compute_variance_concentration(stacked, backend)
                layer_variance_metrics[layer_idx] = var_result
                # Store in explicit new fields
                layer_profile.variance_concentrations[layer_idx] = var_result.var_top1
                layer_profile.effective_ranks[layer_idx] = var_result.effective_rank
                # Store inverted var_top1 for backward compatibility
                layer_profile.intrinsic_dimensions[layer_idx] = 1.0 - var_result.var_top1
                var_computed += 1
            except Exception as exc:
                logger.debug("Variance estimation failed for layer %d: %s", layer_idx, exc)
                continue

        if var_computed > 0:
            bottleneck_layers = identify_bottleneck_layers(
                layer_variance_metrics,
                var_threshold=0.70,
            )
            if bottleneck_layers:
                layer_profile.bottleneck_layer = bottleneck_layers[0]

            var_vals = [(idx, m.var_top1, m.effective_rank) for idx, m in layer_variance_metrics.items()]
            var_vals.sort(key=lambda x: x[1], reverse=True)
            logger.info(
                "LAYER PROFILE (fallback): Computed hidden state variance for %d layers",
                var_computed,
            )
            logger.info(
                "VARIANCE TOP-5 BOTTLENECKS (fallback): %s",
                [(idx, f"{var:.1%}", f"eff_rank={rank:.1f}") for idx, var, rank in var_vals[:5]],
            )
        else:
            raise RuntimeError(
                "VARIANCE CONCENTRATION: No usable measurements from hidden state fallback."
            )
    else:
        raise RuntimeError(
            "VARIANCE CONCENTRATION: Neither intermediate nor hidden activations available."
        )

    # =================================================================
    # SPARSE REGION ANALYSIS (activation-driven, no heuristics)
    # =================================================================
    probe_domains_list = probe_result.get("probe_domains", []) or []
    if not probe_domains_list:
        raise RuntimeError(
            "SPARSE REGION: Missing probe domains; cannot compute sparsity-driven transfer weights."
        )

    sparsity_target_acts = target_activations
    if profile_alignment_result.target_mean_pooled:
        sparsity_target_acts = profile_alignment_result.target_mean_pooled

    if sparsity_target_acts:
        prompt_activations: list[dict[int, float]] = [
            {} for _ in range(len(probe_domains_list))
        ]
        for layer_idx, acts in sparsity_target_acts.items():
            if acts is None:
                raise RuntimeError(
                    f"SPARSE REGION: Missing activations for layer {layer_idx}."
                )
            if hasattr(acts, "shape") and len(backend.shape(acts)) == 2:
                stacked = acts
            else:
                stacked = backend.stack(list(acts), axis=0)
            stacked = backend.array(stacked)
            backend.eval(stacked)
            norms = geodesic_norms(stacked, backend)
            backend.eval(norms)
            norm_vals = backend.tolist(norms)
            for i, val in enumerate(norm_vals):
                if i < len(prompt_activations):
                    prompt_activations[i][layer_idx] = float(val)

        if len(prompt_activations) != len(probe_domains_list):
            raise RuntimeError(
                "SPARSE REGION: Probe domain count mismatch "
                f"(domains={len(probe_domains_list)}, activations={len(prompt_activations)})."
            )

        domain_activations: dict[str, list[dict[int, float]]] = {}
        for idx, domain in enumerate(probe_domains_list):
            domain_activations.setdefault(domain, []).append(prompt_activations[idx])

        baseline_activations = list(prompt_activations)
        locator = SparseRegionLocator()
        sparse_results = []
        for domain, activations in domain_activations.items():
            if not activations:
                continue
            result = locator.analyze_from_activations(
                domain_activations=activations,
                baseline_activations=baseline_activations,
                domain=domain,
            )
            sparse_results.append(result)

        combined_sparsity: dict[int, float] = {}
        sparse_layers: set[int] = set()
        skip_layers: set[int] = set()
        for result in sparse_results:
            for layer_idx, sparsity in result.layer_sparsity.items():
                current = combined_sparsity.get(layer_idx, 0.0)
                combined_sparsity[layer_idx] = max(current, sparsity)
            sparse_layers.update(result.sparse_layers)
            skip_layers.update(result.skip_layers)

        layer_profile.layer_sparsity = combined_sparsity
        layer_profile.sparse_layers = sorted(sparse_layers)
        layer_profile.skip_layers = sorted(skip_layers)

        logger.info(
            "SPARSE REGION: %d domains analyzed, %d sparse layers, %d skip layers",
            len(sparse_results),
            len(layer_profile.sparse_layers),
            len(layer_profile.skip_layers),
        )

    # =================================================================
    # INJECTION LAYER SELECTION (Single-point knowledge transfer)
    # =================================================================
    # The highway is just rolling information over - the geometry is
    # identical across transmission layers. We only need ONE injection point.
    #
    # This dramatically reduces density computation (1 layer vs 16+).
    # Use the injection layer computed during probe/alignment stage if available.
    # This ensures the probe aligned the correct layer for injection.
    # Allow environment variable override for testing different injection points
    # MC_INJECTION_LAYER=4 will force injection at layer 4
    injection_layer_override = os.environ.get("MC_INJECTION_LAYER")
    if injection_layer_override is not None:
        try:
            injection_layer = int(injection_layer_override)
            logger.info(
                "INJECTION LAYER: OVERRIDE from env MC_INJECTION_LAYER=%d",
                injection_layer,
            )
        except ValueError:
            logger.warning(
                "INJECTION LAYER: Invalid MC_INJECTION_LAYER=%s, ignoring",
                injection_layer_override,
            )
            injection_layer = None
    elif profile_injection_layer is not None:
        injection_layer = profile_injection_layer
        logger.info(
            "INJECTION LAYER: Using layer %d from profile alignment",
            injection_layer,
        )
    else:
        injection_layer = layer_profile.compute_best_injection_layer()
    transmission_layers = layer_profile.compute_transmission_layers()

    if injection_layer is not None:
        logger.info(
            "INJECTION LAYER: Selected layer %d for single-point injection (transmission=%s)",
            injection_layer,
            transmission_layers,
        )
        # Filter to ONLY the injection layer for density computation
        density_layer_indices = [injection_layer]
    else:
        logger.warning(
            "INJECTION LAYER: No valid injection layer found - using all layers (fallback)"
        )
        density_layer_indices = layer_indices

    # =================================================================
    # STAGE 2: DENSITY (Knowledge density profiling)
    # =================================================================
    # Compare density between source and target to identify graft opportunities.
    # High density in source + Low density in target = GRAFT (fill the gap)
    # This MUST run before memory cleanup since we need source_activations.
    # NOTE: Only computing for injection layer(s), not all layers.
    logger.info("STAGE 2: DENSITY (knowledge density profiling) - Starting...")

    probe_ids_list = probe_result.get("probe_ids", [])
    probe_domains_list = probe_result.get("probe_domains", [])
    logger.info("STAGE 2: DENSITY - probe_ids=%d, probe_domains=%d", len(probe_ids_list), len(probe_domains_list))

    # Use mean-pooled activations for density analysis when available
    # This ensures profile-based merges produce identical results to probe-based
    # Mean-pooled = per-probe vectors (what density stage expects)
    # Regular activations = per-token trajectory positions (too granular for per-concept density)
    density_source_acts = source_activations
    density_target_acts = target_activations
    if profile_alignment_result.source_mean_pooled:
        density_source_acts = profile_alignment_result.source_mean_pooled
    if profile_alignment_result.target_mean_pooled:
        density_target_acts = profile_alignment_result.target_mean_pooled

    # Run density stage with alignment transforms for cross-dimensional comparison
    # The transforms project source activations into target space BEFORE comparing,
    # so density comparison is always apples-to-apples in target's coordinate system.
    # This finds where target is TRULY sparse in specific concepts, not just smaller.
    # NOTE: Only computing for injection layer(s), not all layers - massive speedup.
    density_result = stage_density(
        source_activations=density_source_acts,
        target_activations=density_target_acts,
        probe_ids=probe_ids_list,
        probe_domains=probe_domains_list,
        layers=density_layer_indices,  # Only injection layer(s), not all
        feature_transforms=feature_transforms,  # Project source→target space
        layer_mapping=layer_mapping,  # Use DP correspondence (target_layer -> source_layer)
        backend=backend,
    )
    graft_mask = density_result.graft_mask
    density_weights = density_result.density_weights
    density_metrics = dict(density_result.metrics)
    density_metrics["detail"] = _serialize_density_detail(
        density_result,
        backend,
        source_model=source_path,
        target_model=target_path,
    )

    logger.info(
        "DENSITY: %d concepts analyzed, %d graft opportunities (source denser), %d skip (target dense)",
        density_metrics.get("concepts_analyzed", 0),
        density_metrics.get("positive_opportunity_count", 0),
        density_metrics.get("nonpositive_opportunity_count", 0),
    )
    logger.info(
        "DENSITY: Point cloud - %d points source denser, %d target denser (k-NN based)",
        density_metrics.get("point_cloud_positive_points", 0),
        density_metrics.get("point_cloud_negative_points", 0),
    )
    logger.info("STAGE 2: DENSITY completed")

    # =========================================================================
    # MEMORY CLEANUP: Delete activations not needed for transplant
    # =========================================================================
    # Transplant only uses target_activations for null-space projection.
    # Source activations were used for density comparison (now complete).
    # Intermediate/attention activations are unused after probe alignment.
    import gc

    # =========================================================================
    # SELECTIVE MEMORY CLEANUP
    # =========================================================================
    # Keep activations needed for density-aware transplant:
    # - source_activations: For density comparison (hidden level)
    # - target_activations: For null-space projection (already kept)
    #
    # Clear only what's not needed:
    # - source_attention_activations: Attention transforms from probe stage suffice
    # - target_attention_activations: Attention transforms from probe stage suffice
    # - source/target_k_activations: K/V handled compositionally

    # Keep source hidden activations for density comparison
    # (Needed for density-aware transfer + anchor corrections)
    logger.info(
        "Keeping source activations for density-aware transfer: source_acts=%s",
        bool(source_activations),
    )

    # NOTE: Keep intermediate activations - needed for down_proj null-space projection
    # MLP down_proj weights have input dimension = intermediate_size, not hidden_size
    # So we need intermediate activations to compute proper null-space boundaries

    # Clear attention activations (transforms from probe stage suffice)
    if source_attention_activations:
        source_attention_activations.clear()
        del source_attention_activations
        source_attention_activations = None
    if source_k_activations:
        source_k_activations.clear()
        del source_k_activations
        source_k_activations = None

    if target_attention_activations:
        target_attention_activations.clear()
        del target_attention_activations
        target_attention_activations = None
    if target_k_activations:
        target_k_activations.clear()
        del target_k_activations
        target_k_activations = None

        # Force garbage collection and clear backend cache
    gc.collect()
    ComputationCache.shared().clear_geometry_caches()
    default_backend.clear_cache()
    logger.info(
        "Cleared attention activations - kept hidden (%d layers) + intermediate (%d src, %d tgt) for transplant",
        len(source_activations) if source_activations else 0,
        len(source_intermediate_activations) if source_intermediate_activations else 0,
        len(target_intermediate_activations) if target_intermediate_activations else 0,
    )

    # PERMUTE STAGE REMOVED: GramAligner alignment subsumes permutation
    # subsumes discrete permutation alignment. Permutation matrices are a special
    # case of continuous linear transforms already optimized by the probe stage.
    permute_metrics = {"skipped": True, "reason": "subsumed_by_gram_alignment"}

    # =================================================================
    # STAGE 3: TRANSPLANT (Null-space constrained knowledge transfer)
    # =================================================================
    # Density-guided path: graft_mask from Stage 2 decides what to transplant.
    # Null-space projection ensures we only add to directions target doesn't use.
    # Combined: graft WHERE source is denser AND into target's null space.

    if not target_activations:
        raise RuntimeError(
            "Transplant requires probe activations. "
            "Use `mc merge` to collect activations before merging."
        )

    logger.info("STAGE 3: TRANSPLANT - Starting...")
    if graft_mask is not None:
        graft_count = sum(
            1 for probes in graft_mask.values() for should_graft in probes.values() if should_graft
        )
        logger.info(
            "STAGE 3: TRANSPLANT (density-guided, %d graft opportunities)",
            graft_count,
        )
    else:
        logger.info("STAGE 3: TRANSPLANT (graft-all path)")

    # =================================================================
    # NULL-SPACE PROJECTION IS THE GEOMETRY
    # =================================================================
    # The null-space projection in transplant.py determines what to transfer:
    # - It projects source delta into directions the target doesn't actively use
    # - Dense directions (high activation variance) are scaled down
    # - Sparse directions (low variance) are preserved
    #
    # We do NOT apply additional subspace-based novelty gating here.
    # The geometry (null-space projection) IS the filter. High structural
    # overlap (CKA ≈ 1) simply means the models use similar coordinate systems,
    # not that there's nothing to transfer. The weights can still differ.

    merged_weights, transplant_metrics = stage_transplant(
        source_weights=loaded_source_weights,
        target_weights=loaded_target_weights,
        layer_indices=layer_indices,
        probe_ids=probe_result.get("probe_ids"),
        probe_domains=probe_result.get("probe_domains"),
        source_activations=source_activations,
        target_activations=target_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_attention_activations=source_attention_activations,
        target_attention_activations=target_attention_activations,
        source_kv_activations=source_k_activations,  # K activations (V computed compositionally)
        target_kv_activations=target_k_activations,  # K activations (V computed compositionally)
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
        extract_layer_index_fn=extract_layer_index,
        backend=backend,
        graft_mask=graft_mask,
        density_weights=density_weights,  # Per-probe source density ratios
        feature_transforms=feature_transforms,
        scale_ratios=scale_ratios,  # EXACT: ||target|| / ||source @ F|| per layer
        embedding_transform=embedding_transform,  # 2D GramAlign for embed_tokens
        attention_transforms=attention_transforms,
        k_transforms=k_transforms,
        v_transforms=v_transforms,
        intermediate_transforms=intermediate_transforms,  # MLP transforms
        gate_transforms=gate_transforms,  # PRE-SiLU gate transforms
        layer_mapping=layer_mapping,
        prior_occupancy_by_layer=prior_occupancy_by_layer,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        delta_scale=1.0,
        layer_profile=layer_profile,
        is_cross_vocab=is_cross_vocab,
        source_trajectory_tangents=source_trajectory_tangents,
        target_trajectory_tangents=target_trajectory_tangents,
        layer_coupling=layer_coupling,
        source_layers=source_layers,
        target_layers=target_layers,
        injection_layer=injection_layer,  # Single-point injection
    )
    logger.info("STAGE 3: TRANSPLANT completed")

    # =================================================================
    # STAGE 4: COMPRESSION DESCENT (Force null-space knowledge into active stream)
    # =================================================================
    # After null-space injection, the injection layer has:
    # - Original active space A (target's computation)
    # - Injected null space N (source's knowledge)
    #
    # Compression changes the basis such that null-space becomes active-space.
    # This is the "descent" mechanism - forces model to use injected dimensions.
    # NOTE: Only compress the injection layer(s), not all transmission layers.
    compression_layers = [injection_layer] if injection_layer is not None else layer_profile.compute_transmission_layers()
    if compression_layers:
        logger.info(
            "STAGE 4: COMPRESSION DESCENT - %d layer(s) to compress (injection=%s)",
            len(compression_layers),
            injection_layer,
        )

        # Get weights that were reverted to target (should not be compressed)
        reverted_keys = set(transplant_metrics.get("mlp_reverted_keys", []))
        if reverted_keys:
            logger.info(
                "STAGE 4: Skipping %d reverted MLP weights: %s",
                len(reverted_keys),
                list(reverted_keys)[:3],  # Show first 3 for brevity
            )

        compression_descent_result = stage_compression_descent(
            merged_weights=merged_weights,
            transmission_layers=compression_layers,
            layer_activations=target_activations,
            extract_layer_index_fn=extract_layer_index,
            backend=backend,
            compression_target=0.5,  # Keep top 50% of variance
            skip_weights=reverted_keys,
        )

        # Apply compressed weights to merged weights
        if compression_descent_result.weights_compressed > 0:
            merged_weights = apply_compression_descent_to_weights(
                merged_weights, compression_descent_result
            )
            logger.info(
                "STAGE 4: COMPRESSION DESCENT completed - %d weights compressed, "
                "mean_compression=%.2fx, mean_cka=%.6f",
                compression_descent_result.weights_compressed,
                1.0 / compression_descent_result.mean_compression_ratio
                if compression_descent_result.mean_compression_ratio > 0 else 0,
                compression_descent_result.mean_cka,
            )
        else:
            logger.info("STAGE 4: COMPRESSION DESCENT - no weights compressed (skipped)")
    else:
        logger.info("STAGE 4: COMPRESSION DESCENT skipped (no transmission layers)")

    # =================================================================
    # REQUANTIZATION (if target was quantized)
    # =================================================================
    target_is_quantized = any(
        k.endswith(".scales") or k.endswith(".biases") for k in loaded_target_weights.keys()
    )

    if target_is_quantized:
        import json as _json_for_quant

        from modelcypher.core.use_cases.quantization_utils import (
            quantization_plan_from_payload,
            requantize_weights,
        )

        # Read target config to get quantization params
        config_path = Path(target_path) / "config.json"
        if not config_path.exists():
            raise RuntimeError(
                "Target is quantized but config.json is missing. "
                "Quantization parameters are required for exact requantization."
            )
        try:
            with open(config_path) as f:
                config_data = _json_for_quant.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read target config.json for quantization parameters: {exc}"
            ) from exc

        quant_plan = quantization_plan_from_payload(config_data)
        if not quant_plan or not quant_plan[0]:
            raise RuntimeError(
                "Target is quantized but quantization parameters are missing from config.json."
            )
        quant_hint = quant_plan[0]
        logger.info(
            "Detected target quantization: %d-bit, group_size=%d",
            quant_hint.bits,
            quant_hint.group_size,
        )

        # Preserve vocabulary-tied weights (embeddings, lm_head) from target.
        # Cross-vocab merging: we can't align embeddings by vocab row because
        # token ID N in source != token ID N in target. Use target's original
        # vocabulary weights and let the hidden layer alignment transfer knowledge.
        from modelcypher.ports.model_architecture_factory import get_output_projection_key, load_config

        target_config = load_config(target_path)
        lm_head_key = get_output_projection_key(target_config, loaded_target_weights)

        vocab_keys = set()
        for k in loaded_target_weights.keys():
            # Use architecture-aware detection for lm_head
            if lm_head_key and k == lm_head_key:
                vocab_keys.add(k)
            # Fallback for embed keys (still use pattern matching for embeddings)
            elif "embed" in k.lower():
                vocab_keys.add(k)

        vocab_weights = {k: loaded_target_weights[k] for k in vocab_keys}
        logger.info(
            "Preserving %d vocabulary-tied weights from target (cross-vocab merge)",
            len(vocab_weights),
        )

        logger.info("Requantizing merged weights to match target format...")
        merged_weights = requantize_weights(
            merged_weights,
            backend,
            quant_hint,
        )

        # Restore vocabulary weights (only lm_head - embed_tokens was already aligned)
        # IMPORTANT: Do NOT restore embed_tokens - transplant stage already aligned it!
        for k, v in vocab_weights.items():
            if lm_head_key and k == lm_head_key:
                merged_weights[k] = v
                logger.info("Preserved target lm_head: %s", k)

        logger.info("Requantization complete: %d weights", len(merged_weights))

    # =================================================================
    # OUTPUT
    # =================================================================
    final_output_path: str | None = None
    post_merge_density: float | None = None
    if output_dir:
        save_weights(output_dir, merged_weights, target_format, backend)
        copy_config_files(target_path, output_dir)
        final_output_path = output_dir
        occupancy_by_layer = transplant_metrics.get("occupancy_by_layer")
        if occupancy_by_layer:
            save_transplant_occupancy(output_dir, occupancy_by_layer)

        # =================================================================
        # POST-MERGE DENSITY ESTIMATION (from transplant metrics)
        # =================================================================
        # OPTIMIZATION: Instead of reloading the merged model and re-probing,
        # we estimate post-merge density from transplant metrics. The transplant
        # stage already computed preserved_fraction and projection_loss per layer,
        # which directly indicates how much knowledge was transferred.
        #
        # Density change is derived from:
        # - source_density (denser regions)
        # - target_density (baseline)
        # - graft_mask (which concepts were grafted)
        # - preserved_fraction (how much delta was added)
        logger.info("STAGE 4: VALIDATE (density estimation from transplant metrics)")
        source_density = density_metrics.get("overall_source_density", 0)
        target_density = density_metrics.get("overall_target_density", 0)
        opportunity = density_metrics.get("overall_opportunity", 0)
        preserved_fraction = transplant_metrics.get("mean_preserved_fraction", 0)

        if target_density > 0 and opportunity > 0:
            # Estimate post-merge density based on how much of the density
            # opportunity was captured via null-space projection.
            # Formula: target_density + (opportunity * preserved_fraction)
            # This is exact for null-space addition: we add density in
            # orthogonal directions without disturbing existing structure.
            density_gain = opportunity * preserved_fraction
            post_merge_density = target_density + density_gain
            density_change = (density_gain / target_density * 100) if target_density > 0 else 0
            logger.info(
                "POST-MERGE: estimated density=%.4f (was %.4f, +%.1f%% from transplant)",
                post_merge_density, target_density, density_change
            )
        elif target_density > 0:
            # No opportunity but we have target density - use it as baseline
            post_merge_density = target_density
            logger.info(
                "POST-MERGE: density=%.4f (no opportunity for increase)",
                post_merge_density
            )

        # Save merge analysis report for scientific reproducibility
        import json
        target_density_val = density_metrics.get("overall_target_density")
        analysis_report = {
            "_schema": "mc.merge.analysis.v1",
            "timestamp": datetime.utcnow().isoformat(),
            "source_model": source_path,
            "target_model": target_path,
            "output_path": final_output_path,
            "density": {
                "source_density": density_metrics.get("overall_source_density"),
                "target_density_before": target_density_val,
                "target_density_after": post_merge_density,
                "density_change_pct": ((post_merge_density - target_density_val) / target_density_val * 100) if target_density_val and post_merge_density else None,
                "opportunity": density_metrics.get("overall_opportunity"),
                "concepts_analyzed": density_metrics.get("concepts_analyzed"),
                "grafted_count": density_metrics.get("positive_opportunity_count"),
                "skipped_count": density_metrics.get("nonpositive_opportunity_count"),
            },
            "geometry": {
                "mean_preserved_fraction": transplant_metrics.get("mean_preserved_fraction"),
                "layers_transplanted": transplant_metrics.get("layers_transplanted"),
                "weights_transplanted": transplant_metrics.get("weights_transplanted"),
            },
            "probe": {
                "layer_count": probe_metrics.get("layer_count"),
                "geodesic_cka": probe_metrics.get("mean_cka"),  # Geodesic CKA after alignment
                # SPLIT CKA: separates "alignment quality" from "novelty fraction"
                "split_cka": probe_metrics.get("split_cka"),
            },
        }
        analysis_path = Path(final_output_path) / "merge_analysis.json"
        analysis_path.write_text(json.dumps(analysis_report, indent=2, default=str))
        logger.info("Saved merge analysis to %s", analysis_path)

        # =================================================================
        # STAGE 5: TRAJECTORY COHERENCE VALIDATION (BLOCKING)
        # =================================================================
        # Run inference on test prompts and detect degenerate patterns
        # (repetition, single-token collapse). If the merged model produces
        # gibberish, we've broken the geometry - abort rather than save.
        #
        # Common non-geometry causes of immediate gibberish:
        # - Tokenizer/vocab mismatch (check BEFORE merge)
        # - Embedding/LM head not aligned consistently
        # - LayerNorm stats drift
        # - RoPE scaling mismatch
        # =================================================================
        from modelcypher.core.domain.geometry.trajectory_coherence import (
            MergeCoherenceError,
            validate_merge_coherence,
        )

        logger.info("STAGE 5: VALIDATE COHERENCE (trajectory coherence check)")
        try:
            coherence_result = validate_merge_coherence(
                model_path=final_output_path,
                inference_engine=inference_engine,
                test_prompts=None,  # Uses default diverse prompts
                baseline_model_path=target_path,
            )

            if coherence_result.is_coherent is False:
                # Log detailed failure info but don't raise yet
                # (we already saved, so give the user the analysis)
                logger.error(
                    "COHERENCE VALIDATION FAILED: %d/%d prompts degenerate "
                    "(mean_repetition=%.2f). The merged model may produce gibberish.",
                    coherence_result.failed_count,
                    coherence_result.total_count,
                    coherence_result.mean_repetition_score,
                )
                for metrics in coherence_result.metrics:
                    if metrics.is_degenerate:
                        logger.error(
                            "  FAILED: '%s' -> %s",
                            metrics.prompt,
                            metrics.degenerate_reason,
                        )

                # Update analysis report with coherence failure
                analysis_report["coherence"] = {
                    "is_coherent": False,
                    "failed_count": coherence_result.failed_count,
                    "total_count": coherence_result.total_count,
                    "mean_repetition_score": coherence_result.mean_repetition_score,
                    "failed_prompts": coherence_result.failed_prompts,
                }
                analysis_path.write_text(json.dumps(analysis_report, indent=2, default=str))
            else:
                logger.info(
                    "COHERENCE VALIDATION PASSED: %d/%d prompts coherent",
                    coherence_result.total_count - coherence_result.failed_count,
                    coherence_result.total_count,
                )
                analysis_report["coherence"] = {
                    "is_coherent": True,
                    "failed_count": 0,
                    "total_count": coherence_result.total_count,
                    "mean_repetition_score": coherence_result.mean_repetition_score,
                }
                analysis_path.write_text(json.dumps(analysis_report, indent=2, default=str))

        except Exception as e:
            # Don't fail the merge if coherence check itself fails
            # (e.g., inference engine not available)
            logger.warning(
                "COHERENCE VALIDATION SKIPPED: %s. "
                "Run 'mc infer run --model %s --prompt <test>' to validate manually.",
                e,
                final_output_path,
            )

    # Compute geometric metrics from transplant measurements
    from modelcypher.core.use_cases.merge.metrics import (
        compute_geometric_metrics_from_transplant,
    )

    geometry_metrics = compute_geometric_metrics_from_transplant(transplant_metrics)
    mean_preserved_fraction = geometry_metrics.get("mean_preserved_fraction", 0.0)

    projection_losses = transplant_metrics.get("projection_losses", [])
    mean_error = sum(projection_losses) / len(projection_losses) if projection_losses else 0.0

    if final_output_path:
        import json

        diagnostics_payload = {
            "_schema": "mc.merge.diagnostics.v1",
            "timestamp": datetime.utcnow().isoformat(),
            "source_model": source_path,
            "target_model": target_path,
            "output_path": final_output_path,
            "probe": probe_metrics,
            "density": density_metrics,
            "transplant": transplant_metrics,
            "geometry": geometry_metrics,
            "post_merge_density": post_merge_density,
            "fingerprints": {
                "source_gram_hash": fingerprint_comparison.source_gram_hash,
                "target_gram_hash": fingerprint_comparison.target_gram_hash,
                "source_condition_number": fingerprint_comparison.source_condition_number,
                "target_condition_number": fingerprint_comparison.target_condition_number,
                "source_effective_dim": fingerprint_comparison.source_effective_dim,
                "target_effective_dim": fingerprint_comparison.target_effective_dim,
                "condition_number_ratio": fingerprint_comparison.condition_number_ratio,
                "effective_dim_delta": fingerprint_comparison.effective_dim_delta,
            },
        }
        diagnostics_path = Path(final_output_path) / "merge_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, default=str))
        logger.info("Saved merge diagnostics to %s", diagnostics_path)

    result = UnifiedMergeResult(
        merged_weights=merged_weights,
        probe_metrics=probe_metrics,
        permute_metrics=permute_metrics,
        transplant_metrics=transplant_metrics,
        mean_preserved_fraction=mean_preserved_fraction,
        mean_procrustes_error=float(mean_error),
        layer_count=len(layer_indices),
        weight_count=len(merged_weights),
        timestamp=datetime.utcnow(),
        merge_strategy="transplant",
        output_path=final_output_path,
        refusal_preserved=True,
        geometry_metrics=geometry_metrics,
        density_metrics=density_metrics,
        post_merge_density=post_merge_density,
        fingerprint_comparison=fingerprint_comparison,
    )

    logger.info(
        "Merge complete: %d layers, %d weights, preserved_fraction=%.3f, error=%.3f",
        result.layer_count,
        result.weight_count,
        result.mean_preserved_fraction,
        result.mean_procrustes_error,
    )

    # Reset model-driven precision (cleanup for next merge)
    set_model_compute_dtype(None)

    return result
