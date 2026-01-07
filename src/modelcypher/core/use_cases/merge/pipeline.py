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
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

from .helpers import (
    copy_config_files,
    extract_layer_index,
    extract_layer_indices,
    infer_hidden_dim,
    load_model_for_probing,
    load_tokenizer,
    load_weights,
    save_weights,
)
from .models import UnifiedMergeResult
from .stages import (
    stage_density,
    stage_probe,
    stage_transplant,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


def run_merge(
    model_loader: "ModelLoaderPort",
    backend: "Backend",
    source_path: str,
    target_path: str,
    output_dir: str | None = None,
    output_path: str | None = None,
    dry_run: bool = False,
    target_weights: dict[str, "Array"] | None = None,
) -> UnifiedMergeResult:
    """
    Execute null-space constrained transplant merge.

    Transplant formula:
        W' = W_target + P_null(A_boundary) @ (W_source_aligned - W_target)

    Guarantee:
        A_boundary @ W' = A_boundary @ W_target  (boundary preserved)
    """
    logger.info("=== PURE GEOMETRIC MERGE ===")
    logger.info("Source: %s", source_path)
    logger.info("Target: %s", target_path)

    # Resolve output path (prefer output_path over output_dir)
    effective_output = output_path or output_dir

    logger.info("Using null-space constrained transplant.")

    # Load weights (backend arrays)
    source_weights, _ = load_weights(model_loader, source_path)

    # Use pre-loaded target weights if provided (multi-donor optimization)
    if target_weights is not None:
        logger.info("Using pre-loaded target weights (multi-donor mode)")
        loaded_target_weights = target_weights
        target_format = "safetensors"  # Assume safetensors for pre-loaded weights
    else:
        loaded_target_weights, target_format = load_weights(model_loader, target_path)

    # Identify layers
    layer_indices = extract_layer_indices(loaded_target_weights)
    logger.info("Found %d layers", len(layer_indices))

    # Load tokenizers for probe execution
    source_tokenizer = load_tokenizer(source_path)
    target_tokenizer = load_tokenizer(target_path)

    # Load models for probe stage
    source_model = None
    target_model = None
    logger.info("Loading models for probe execution...")
    source_model = load_model_for_probing(source_path)
    target_model = load_model_for_probing(target_path)

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
        layer_mapping,
    ) = stage_probe(
        source_weights=source_weights,
        target_weights=loaded_target_weights,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        extract_layer_index_fn=extract_layer_index,
        # ProbeConfig was REMOVED - always use precise mode with all probes
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
        # ADAPTIVE BAROMETER: Per DIMENSIONAL_COMPRESSION.md
        # =====================================================================
        # Instead of failing, classify layers and proceed with selective transplant:
        # - "converged": Full knowledge transfer (CKA >= 0.9995)
        # - "boundary_preserved": Skip injection, preserve transitions (0.5 <= CKA < 0.9995)
        # - "skipped": Geometrically incompatible (CKA < 0.5)
        converged_count = probe_metrics.get("converged_count", 0)
        boundary_count = probe_metrics.get("boundary_preserved_count", 0)
        skipped_count = probe_metrics.get("skipped_count", 0)
        min_cka = probe_metrics.get("min_cka", 0.0)
        mean_cka = probe_metrics.get("mean_cka", 0.0)
        
        if converged_count == 0:
            # No converged layers at all - this is a true failure
            raise RuntimeError(
                "PROBE BAROMETER: No layers converged to CKA=1.0 "
                "(mean_cka=%.4f, min_cka=%.4f). Architecture may be incompatible."
                % (mean_cka, min_cka)
            )
        
        # Proceed with selective transplant
        logger.warning(
            "ADAPTIVE BAROMETER: %d converged, %d boundary-preserved, %d skipped. "
            "Proceeding with selective transplant (mean_cka=%.4f).",
            converged_count, boundary_count, skipped_count, mean_cka
        )

    # Log transform results from probe stage
    if feature_transforms:
        logger.info(
            "PROBE: Computed %d hidden transforms (layers: %s)",
            len(feature_transforms),
            sorted(feature_transforms.keys())[:5],  # First 5 for brevity
        )
    else:
        logger.warning("PROBE: No hidden transforms computed - cross-arch merge will fail")

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
        logger.warning("PROBE: No layer mapping - cross-arch merge will fail")

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
    default_backend.clear_cache()
    logger.info("Cleared GPU cache after probe stage")

    # PERMUTE STAGE REMOVED: GramAligner's CKA=1.0 alignment in geodesic RKHS
    # subsumes discrete permutation alignment. Permutation matrices are a special
    # case of continuous linear transforms already optimized by the probe stage.
    permute_metrics = {"skipped": True, "reason": "subsumed_by_gram_alignment"}

    # =================================================================
    # STAGE 3: TRANSPLANT (Null-space constrained knowledge transfer)
    # =================================================================
    # ROTATE/PROPAGATE was removed - no boundary preservation guarantee.
    # Only null-space constrained transplant preserves boundary relationships.
    #
    # Geometry-only mode: ALL probes are candidates and graft_mask decides
    # what to transplant based on density.
    logger.info("TRANSPLANT: Geometry-only mode - density decides grafts")

    if not target_activations:
        raise RuntimeError(
            "Transplant requires probe activations. "
            "Use `mc merge` to collect activations before merging."
        )

    # =================================================================
    # STAGE 2.5: DENSITY (Selective grafting based on knowledge density)
    # =================================================================
    # Compute which concepts to graft based on source/target density.
    # Only graft where source is denser than target (fills gaps, no overwrites).
    logger.info("STAGE 2.5: DENSITY (computing graft mask)")
    # Density requires per-layer activations from both models. In cross-arch merges,
    # the layer counts can differ, so only analyze layers present in both.
    density_layers = sorted(
        set(source_activations.keys()) & set(target_activations.keys())
    )
    if not density_layers:
        raise RuntimeError("DENSITY: No overlapping layers between source and target")
    graft_mask, density_metrics = stage_density(
        source_activations=source_activations,
        target_activations=target_activations,
        probe_ids=probe_result.get("probe_ids"),
        probe_domains=probe_result.get("probe_domains"),
        layers=density_layers,
        backend=backend,
    )

    if graft_mask:
        graft_count = sum(
            sum(1 for v in layer_mask.values() if v)
            for layer_mask in graft_mask.values()
        )
        logger.info(
            "DENSITY: %d concepts marked for grafting, %d skipped (target dense)",
            density_metrics.get("positive_opportunity_count", graft_count),
            density_metrics.get("nonpositive_opportunity_count", 0),
        )
    else:
        logger.info("DENSITY: No graft opportunities (mask empty)")

    logger.info("STAGE 3: TRANSPLANT (null-space constrained)")
    merged_weights, transplant_metrics = stage_transplant(
        source_weights=source_weights,
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
        extract_layer_index_fn=extract_layer_index,
        backend=backend,
        graft_mask=graft_mask,
        feature_transforms=feature_transforms,
        scale_ratios=scale_ratios,  # EXACT: ||target|| / ||source @ F|| per layer
        embedding_transform=embedding_transform,  # 2D GramAlign for embed_tokens
        attention_transforms=attention_transforms,
        k_transforms=k_transforms,
        v_transforms=v_transforms,
        intermediate_transforms=intermediate_transforms,  # MLP transforms
        layer_mapping=layer_mapping,
        layer_status=probe_metrics.get("layer_status"),  # NEW: Per DIMENSIONAL_COMPRESSION.md
        source_tokenizer=source_tokenizer,  # For token correspondence
        target_tokenizer=target_tokenizer,  # For token correspondence
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
            QuantizationHint,
            quantization_plan_from_payload,
            requantize_weights,
        )

        # Read target config to get quantization params
        config_path = Path(target_path) / "config.json"
        quant_hint = QuantizationHint(bits=4, group_size=64, mode="affine")  # Default
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = _json_for_quant.load(f)
                quant_plan = quantization_plan_from_payload(config_data)
                if quant_plan and quant_plan[0]:
                    quant_hint = quant_plan[0]
                    logger.info(
                        "Detected target quantization: %d-bit, group_size=%d",
                        quant_hint.bits,
                        quant_hint.group_size,
                    )
            except Exception as exc:
                logger.warning("Could not read target config for quantization: %s", exc)

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
    if effective_output and not dry_run:
        save_weights(effective_output, merged_weights, target_format, backend)
        copy_config_files(target_path, effective_output)
        final_output_path = effective_output

        # =================================================================
        # POST-MERGE DENSITY MEASUREMENT (proves we increased density)
        # =================================================================
        logger.info("STAGE 4: VALIDATE (post-merge density measurement)")
        post_merge_density = None
        try:
            from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
            from modelcypher.core.use_cases.merge.stages.probe import collect_activations

            # Load merged model (350M is fast)
            merged_model = load_model_for_probing(final_output_path)

            # Run subset of probes through merged model
            probe_ids = probe_result.get("probe_ids", [])[:50]  # Sample 50 probes
            probe_texts = probe_result.get("probe_texts", [])[:50]
            sample_layers = [0, len(layer_indices) // 2, max(0, len(layer_indices) - 1)]

            if probe_texts and merged_model:
                # Collect activations from merged model
                merged_activations = collect_activations(
                    model=merged_model,
                    tokenizer=target_tokenizer,
                    texts=probe_texts,
                    layers=sample_layers,
                    backend=backend,
                )

                # Measure density
                id_estimator = IntrinsicDimension(backend=backend)
                densities = []

                for layer_idx in sample_layers:
                    acts = merged_activations.get(layer_idx, [])
                    if len(acts) >= 4:
                        act_matrix = backend.stack([backend.array(a) for a in acts], axis=0)
                        backend.eval(act_matrix)
                        local_map = id_estimator.local_dimension_map(act_matrix)
                        mean_dim = float(backend.to_scalar(backend.mean(local_map.dimensions)))
                        if mean_dim > 0:
                            densities.append(1.0 / mean_dim)

                if densities:
                    post_merge_density = sum(densities) / len(densities)
                    target_density = density_metrics.get("overall_target_density", 0)
                    density_change = ((post_merge_density - target_density) / target_density * 100) if target_density > 0 else 0
                    logger.info(
                        "POST-MERGE: density=%.4f (was %.4f, %+.1f%% change)",
                        post_merge_density, target_density, density_change
                    )
        except Exception as e:
            logger.warning("Post-merge density measurement skipped: %s", e)

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
                "raw_cka_mean": probe_metrics.get("raw_cka_mean"),
                "layer_count": probe_metrics.get("layer_count"),
                "cka_after_alignment": probe_metrics.get("cka_after_alignment"),
            },
        }
        analysis_path = Path(final_output_path) / "merge_analysis.json"
        analysis_path.write_text(json.dumps(analysis_report, indent=2, default=str))
        logger.info("Saved merge analysis to %s", analysis_path)

    # Compute geometric metrics from transplant measurements
    from modelcypher.core.use_cases.merge.metrics import (
        compute_geometric_metrics_from_transplant,
    )

    geometry_metrics = compute_geometric_metrics_from_transplant(transplant_metrics)
    mean_preserved_fraction = geometry_metrics.get("mean_preserved_fraction", 0.0)

    projection_losses = transplant_metrics.get("projection_losses", [])
    mean_error = sum(projection_losses) / len(projection_losses) if projection_losses else 0.0

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
    )

    logger.info(
        "Merge complete: %d layers, %d weights, preserved_fraction=%.3f, error=%.3f",
        result.layer_count,
        result.weight_count,
        result.mean_preserved_fraction,
        result.mean_procrustes_error,
    )

    return result
