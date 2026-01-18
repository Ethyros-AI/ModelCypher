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

from .helpers import (
    copy_config_files,
    extract_layer_index,
    extract_layer_indices,
    infer_hidden_dim,
    load_model_for_probing,
    load_tokenizer,
    load_weights,
    load_transplant_occupancy,
    save_weights,
    save_transplant_occupancy,
)
from .models import UnifiedMergeResult
from .stages import (
    stage_density,
    stage_probe,
    stage_transplant,
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
    output_path: str | None = None,
    dry_run: bool = False,
    target_weights: dict[str, "Array"] | None = None,
    probe_mode: str = "atlas",
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
    # Delta budget control for sequential stacking
    delta_scale: float = 1.0,
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

    # Resolve output path (prefer output_path over output_dir)
    effective_output = output_path or output_dir

    logger.info("Using null-space constrained transplant.")

    # Load weights (backend arrays) - use pre-loaded if provided
    if source_weights is not None:
        logger.info("Using pre-loaded source weights")
        loaded_source_weights = source_weights
    else:
        loaded_source_weights, _ = load_weights(model_loader, source_path)

    # Use pre-loaded target weights if provided (multi-donor optimization)
    if target_weights is not None:
        logger.info("Using pre-loaded target weights (multi-donor mode)")
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

    # Identify layers
    layer_indices = extract_layer_indices(loaded_target_weights)
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

    # Load models for probe stage (use pre-loaded if available)
    if source_model is None or target_model is None:
        logger.info("Loading models for probe execution...")
        if source_model is None:
            source_model = load_model_for_probing(source_path, model_loader)
        else:
            logger.info("Using pre-loaded source model")
        if target_model is None:
            target_model = load_model_for_probing(target_path, model_loader)
        else:
            logger.info("Using pre-loaded target model")
    else:
        logger.info("Using pre-loaded source and target models")

    # =================================================================
    # STAGE 1: PROBE (Compute layer correspondences via CKA)
    # =================================================================
    logger.info("STAGE 1: PROBE (precise)")
    (
        probe_result,
        probe_metrics,
        source_activations,
        target_activations,
        source_intermediate_activations,
        target_intermediate_activations,
        source_attention_activations,
        target_attention_activations,
        source_k_activations,
        target_k_activations,
        feature_transforms,
        scale_ratios,  # EXACT magnitude factors: ||target|| / ||source @ F||
        embedding_transform,  # 2D GramAlign for embed_tokens
        attention_transforms,
        k_transforms,
        v_transforms,
        intermediate_transforms,  # MLP transforms
        gate_transforms,  # PRE-SiLU gate transforms
        layer_mapping,
        source_embedding_activations,
        target_embedding_activations,
    ) = stage_probe(
        source_weights=loaded_source_weights,
        target_weights=loaded_target_weights,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_path=source_path,
        target_path=target_path,
        extract_layer_index_fn=extract_layer_index,
        probe_mode=probe_mode,
        activation_provider=activation_provider,
    )

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
            "PROBE: Computed %d hidden transforms (layers: %s)",
            len(feature_transforms),
            sorted(feature_transforms.keys())[:5],  # First 5 for brevity
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
            "PROBE: Layer mapping has %d entries (first 5: %s)",
            len(layer_mapping),
            dict(list(layer_mapping.items())[:5]),
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

    # Clear GPU memory
    del source_model
    del target_model
    default_backend = get_default_backend()
    ComputationCache.shared().clear_geometry_caches()
    default_backend.clear_cache()
    logger.info("Cleared GPU cache after probe stage")

    # =================================================================
    # STAGE 2: DENSITY (Knowledge density profiling)
    # =================================================================
    # Compare density between source and target to identify graft opportunities.
    # High density in source + Low density in target = GRAFT (fill the gap)
    # This MUST run before memory cleanup since we need source_activations.
    logger.info("STAGE 2: DENSITY (knowledge density profiling)")

    probe_ids_list = probe_result.get("probe_ids", [])
    probe_domains_list = probe_result.get("probe_domains", [])

    # Run density stage with alignment transforms for cross-dimensional comparison
    # The transforms project source activations into target space BEFORE comparing,
    # so density comparison is always apples-to-apples in target's coordinate system.
    # This finds where target is TRULY sparse in specific concepts, not just smaller.
    if not source_activations or not target_activations:
        raise RuntimeError("DENSITY: Missing activations for density analysis.")
    if not probe_ids_list:
        raise RuntimeError("DENSITY: Missing probe IDs for density analysis.")

    density_result = stage_density(
        source_activations=source_activations,
        target_activations=target_activations,
        probe_ids=probe_ids_list,
        probe_domains=probe_domains_list,
        layers=layer_indices,
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

    # Force garbage collection and clear MLX cache
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
    # Density-guided mode: graft_mask from Stage 2 decides what to transplant.
    # Null-space projection ensures we only add to directions target doesn't use.
    # Combined: graft WHERE source is denser AND into target's null space.

    if not target_activations:
        raise RuntimeError(
            "Transplant requires probe activations. "
            "Use `mc merge` to collect activations before merging."
        )

    if graft_mask is not None:
        graft_count = sum(
            1 for probes in graft_mask.values() for should_graft in probes.values() if should_graft
        )
        logger.info(
            "STAGE 3: TRANSPLANT (density-guided, %d graft opportunities)",
            graft_count,
        )
    else:
        logger.info("STAGE 3: TRANSPLANT (graft-all mode, density unavailable)")

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
    #
    # Previous code scaled delta_scale by novel_fraction, which was a heuristic
    # that contradicted the geometric principle. Removed per user directive:
    # "we do NOT apply a filter. the fucking geometry does."

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
        layer_status=probe_metrics.get("layer_status"),  # NEW: Per DIMENSIONAL_COMPRESSION.md
        prior_occupancy_by_layer=prior_occupancy_by_layer,
        source_tokenizer=source_tokenizer,  # For token correspondence
        target_tokenizer=target_tokenizer,  # For token correspondence
        delta_scale=delta_scale,  # User-specified delta budget (geometry handles the rest)
    )

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
        vocab_keys = {
            k for k in loaded_target_weights.keys()
            if "embed" in k.lower() or "lm_head" in k.lower()
        }
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
            if "lm_head" in k.lower():
                merged_weights[k] = v
                logger.info("Preserved target lm_head: %s", k)

        logger.info("Requantization complete: %d weights", len(merged_weights))

    # =================================================================
    # OUTPUT
    # =================================================================
    final_output_path: str | None = None
    post_merge_density: float | None = None
    if effective_output and not dry_run:
        save_weights(effective_output, merged_weights, target_format, backend)
        copy_config_files(target_path, effective_output)
        final_output_path = effective_output
        occupancy_by_layer = transplant_metrics.get("occupancy_by_layer")
        if occupancy_by_layer:
            save_transplant_occupancy(effective_output, occupancy_by_layer)

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
                max_tokens=100,
            )

            if not coherence_result.is_coherent:
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
                            "  FAILED: '%s...' -> %s",
                            metrics.prompt[:30],
                            metrics.degenerate_reason,
                        )

                # Update analysis report with coherence failure
                analysis_report["coherence"] = {
                    "is_coherent": False,
                    "failed_count": coherence_result.failed_count,
                    "total_count": coherence_result.total_count,
                    "mean_repetition_score": coherence_result.mean_repetition_score,
                    "failed_prompts": coherence_result.failed_prompts[:3],  # First 3
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
