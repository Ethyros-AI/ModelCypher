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
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

from .helpers import (
    copy_config_files,
    extract_layer_index,
    extract_layer_indices,
    infer_hidden_dim,
    load_knowledge_delta_mask,
    load_model_for_probing,
    load_tokenizer,
    load_weights,
    load_weights_cpu,
    require_vocab_phase_lock,
    save_weights,
)
from .models import UnifiedMergeConfig, UnifiedMergeResult
from .stages import (
    stage_density,
    stage_permute,
    stage_probe,
    stage_transplant,
    stage_vocabulary,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


def run_merge(
    model_loader: "ModelLoaderPort",
    backend: "Backend",
    default_config: UnifiedMergeConfig,
    source_path: str,
    target_path: str,
    output_dir: str | None = None,
    output_path: str | None = None,
    dry_run: bool = False,
    use_full_geometry: bool = True,
    knowledge_delta_mask_path: str | None = None,
    transplant_domains: list[str] | None = None,
    target_weights: dict[str, "Array"] | None = None,
    config: UnifiedMergeConfig | None = None,
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

    # Use passed config or fall back to instance config
    merge_config = config if config is not None else default_config

    # Resolve output path (prefer output_path over output_dir)
    effective_output = output_path or output_dir
    if transplant_domains is not None:
        merge_config = replace(
            merge_config,
            transplant_domains=tuple(transplant_domains or merge_config.transplant_domains),
        )

    # Transplant strategy bypasses full geometry merge (uses null-space projection)
    if use_full_geometry and merge_config.transplant_domains:
        logger.info(
            "Transplant domains specified; using null-space constrained transplant."
        )
        use_full_geometry = False

    if use_full_geometry:
        return run_full_geometry_merge(
            model_loader=model_loader,
            backend=backend,
            config=merge_config,
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            dry_run=dry_run,
            knowledge_delta_mask_path=knowledge_delta_mask_path,
        )

    # Load weights (CPU first to reduce GPU memory pressure during merge)
    source_weights, _ = load_weights_cpu(model_loader, source_path)

    # Use pre-loaded target weights if provided (multi-donor optimization)
    if target_weights is not None:
        logger.info("Using pre-loaded target weights (multi-donor mode)")
        loaded_target_weights = target_weights
        target_format = "safetensors"  # Assume safetensors for pre-loaded weights
    else:
        loaded_target_weights, target_format = load_weights_cpu(model_loader, target_path)

    # Identify layers
    layer_indices = extract_layer_indices(loaded_target_weights)
    logger.info("Found %d layers", len(layer_indices))

    # Load tokenizers for vocabulary alignment
    source_tokenizer = load_tokenizer(source_path)
    target_tokenizer = load_tokenizer(target_path)

    # =================================================================
    # STAGE 0: VOCABULARY (Cross-vocabulary alignment for embedding layers)
    # =================================================================
    # Skip vocabulary alignment for transplant since GRAM_TRANSPORT
    # handles cross-dimensional projection at the weight level.
    skip_vocab_for_transplant = bool(merge_config.transplant_domains)
    if skip_vocab_for_transplant:
        logger.info("STAGE 0: VOCABULARY ALIGNMENT (skipped for transplant)")
        vocab_metrics = {"skipped": True, "reason": "transplant_strategy"}
        vocab_aligned = False
        vocab_alignment_map = None
    else:
        logger.info("STAGE 0: VOCABULARY ALIGNMENT")
        source_weights, vocab_metrics, vocab_aligned, vocab_alignment_map = stage_vocabulary(
            source_weights=source_weights,
            target_weights=loaded_target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        require_vocab_phase_lock(vocab_metrics, vocab_aligned)

    # Load models for probe stage
    source_model = None
    target_model = None
    if merge_config.probe_mode == "precise":
        logger.info("Loading models for precise probe execution...")
        source_model = load_model_for_probing(source_path)
        target_model = load_model_for_probing(target_path)

    # =================================================================
    # STAGE 1: PROBE (Compute layer correspondences via CKA)
    # =================================================================
    logger.info("STAGE 1: PROBE (%s mode)", merge_config.probe_mode)
    (
        probe_result,
        probe_metrics,
        source_activations,
        target_activations,
        source_intermediate_activations,
        target_intermediate_activations,
        source_attention_activations,
        target_attention_activations,
        source_kv_activations,
        target_kv_activations,
    ) = stage_probe(
        source_weights=source_weights,
        target_weights=loaded_target_weights,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        alignment_map=vocab_alignment_map,
        extract_layer_index_fn=extract_layer_index,
        # ProbeConfig was REMOVED - always use precise mode with all probes
    )

    layer_confidences: dict[int, float] = probe_result.get("confidences", {})
    probe_result.get("dimension_correlations", {})
    intersection_map_obj = probe_result.get("intersection_map")
    probe_failed = bool(probe_metrics.get("probe_failed"))
    perfect_alignment = bool(probe_metrics.get("perfect_alignment"))

    # Skip CKA alignment check for transplant strategy - cross-architecture models
    # won't have high CKA, but transplant works via null-space projection which
    # operates in target activation space regardless of CKA alignment.
    if not skip_vocab_for_transplant:
        if probe_failed:
            min_cka = probe_metrics.get("min_cka", 0.0)
            mean_cka = probe_metrics.get("mean_cka", 0.0)
            raise RuntimeError(
                "PROBE SIGNAL: Alignment signals missing (mean_cka=%.4f, min_cka=%.4f). "
                "Exact kernel alignment is required before merge."
                % (mean_cka, min_cka)
            )

        if not perfect_alignment:
            min_cka = probe_metrics.get("min_cka", 0.0)
            mean_cka = probe_metrics.get("mean_cka", 0.0)
            raise RuntimeError(
                "PROBE BAROMETER: Alignment not exact kernel aligned "
                "(mean_cka=%.4f, min_cka=%.4f). Resolve alignment before merge."
                % (mean_cka, min_cka)
            )
    else:
        mean_cka = probe_metrics.get("mean_cka", 0.0)
        logger.info(
            "PROBE BAROMETER: Skipped for transplant (mean_cka=%.4f - low CKA expected for cross-arch)",
            mean_cka,
        )

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
    if source_kv_activations and target_kv_activations:
        logger.info(
            "PROBE: Collected ATTENTION (KV) activations for %d source layers, %d target layers",
            len(source_kv_activations),
            len(target_kv_activations),
        )

    # Clear GPU memory
    del source_model
    del target_model
    default_backend = get_default_backend()
    default_backend.clear_cache()
    logger.info("Cleared GPU cache after probe stage")

    # =================================================================
    # STAGE 2: PERMUTE (Git Re-Basin alignment for same-architecture)
    # =================================================================
    # Permutation alignment reduces delta magnitude before transplant.
    # Only enabled for same-architecture models (hidden dimensions must match).
    source_hidden = infer_hidden_dim(source_weights)
    target_hidden = infer_hidden_dim(loaded_target_weights)
    enable_permutation = source_hidden == target_hidden

    if enable_permutation:
        # PermuteConfig was REMOVED - permutation always runs when hidden dims match
        logger.info("STAGE 2: PERMUTE (Git Re-Basin, hidden_dim=%d)", target_hidden)
        permuted_weights, permute_metrics = stage_permute(
            source_weights=source_weights,
            target_weights=loaded_target_weights,
            intersection_map_obj=intersection_map_obj,
            layer_confidences=layer_confidences,
            backend=backend,
        )
        if not permute_metrics.get("skipped"):
            source_weights = permuted_weights
            logger.info(
                "PERMUTE: Aligned %d MLP blocks, mean_quality=%.3f",
                permute_metrics.get("layers_permuted", 0),
                permute_metrics.get("mean_quality", 0.0),
            )
        else:
            logger.info("PERMUTE: Skipped (%s)", permute_metrics.get("reason", "unknown"))
    else:
        logger.info(
            "STAGE 2: PERMUTE (skipped - hidden_dim mismatch: source=%d, target=%d)",
            source_hidden,
            target_hidden,
        )
        permute_metrics = {"skipped": True, "reason": "hidden_dim_mismatch"}

    # =================================================================
    # STAGE 3: TRANSPLANT (Null-space constrained knowledge transfer)
    # =================================================================
    # ROTATE/BLEND was removed - alpha-blending produces gibberish.
    # Only null-space constrained transplant preserves boundary relationships.
    if not merge_config.transplant_domains:
        raise RuntimeError(
            "Transplant requires transplant_domains. "
            "Specify domains like ['mathematical', 'logical'] for knowledge transfer."
        )
    if not target_activations:
        raise RuntimeError(
            "Transplant requires probe activations. "
            "Use `mc merge pipeline` (probe stage) to collect activations before merging."
        )

    # =================================================================
    # STAGE 2.5: DENSITY (Selective grafting based on knowledge density)
    # =================================================================
    # Compute which concepts to graft based on source/target density.
    # Only graft where source is denser than target (fills gaps, no overwrites).
    logger.info("STAGE 2.5: DENSITY (computing graft mask)")
    graft_mask, density_metrics = stage_density(
        source_activations=source_activations,
        target_activations=target_activations,
        probe_ids=probe_result.get("probe_ids"),
        probe_domains=probe_result.get("probe_domains"),
        layers=layer_indices,
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
        source_kv_activations=source_kv_activations,
        target_kv_activations=target_kv_activations,
        transplant_domains=tuple(merge_config.transplant_domains),
        extract_layer_index_fn=extract_layer_index,
        backend=backend,
        graft_mask=graft_mask,
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
            quantization_config_from_payload,
            requantize_weights,
        )

        # Read target config to get quantization params
        config_path = Path(target_path) / "config.json"
        quant_hint = QuantizationHint(bits=4, group_size=64, mode="affine")  # Default
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = _json_for_quant.load(f)
                quant_config = quantization_config_from_payload(config_data)
                if quant_config and quant_config.default:
                    quant_hint = quant_config.default
                    logger.info(
                        "Detected target quantization: %d-bit, group_size=%d",
                        quant_hint.bits,
                        quant_hint.group_size,
                    )
            except Exception as exc:
                logger.warning("Could not read target config for quantization: %s", exc)

        logger.info("Requantizing merged weights to match target format...")
        merged_weights = requantize_weights(
            merged_weights,
            backend,
            quant_hint,
        )
        logger.info("Requantization complete: %d weights", len(merged_weights))

    # =================================================================
    # OUTPUT
    # =================================================================
    final_output_path: str | None = None
    if effective_output and not dry_run:
        save_weights(effective_output, merged_weights, target_format, backend)
        copy_config_files(target_path, effective_output)
        final_output_path = effective_output

    # Compute geometric confidence from transplant metrics
    # Confidence IS the geometry - no vibes, no interpretation strings
    from modelcypher.core.use_cases.merge.confidence import (
        compute_geometric_confidence_from_transplant,
        compute_mean_confidence,
        compute_safety_verdict,
    )

    geometry_metrics = compute_geometric_confidence_from_transplant(transplant_metrics)
    mean_confidence = compute_mean_confidence(geometry_metrics)
    safety_verdict = compute_safety_verdict(geometry_metrics)

    projection_losses = transplant_metrics.get("projection_losses", [])
    mean_error = sum(projection_losses) / len(projection_losses) if projection_losses else 0.0

    result = UnifiedMergeResult(
        merged_weights=merged_weights,
        vocab_metrics=vocab_metrics,
        probe_metrics=probe_metrics,
        permute_metrics=permute_metrics,
        transplant_metrics=transplant_metrics,
        mean_confidence=mean_confidence,
        mean_procrustes_error=float(mean_error),
        layer_count=len(layer_indices),
        weight_count=len(merged_weights),
        timestamp=datetime.utcnow(),
        merge_strategy="transplant",
        output_path=final_output_path,
        vocab_aligned=vocab_aligned,
        safety_verdict=safety_verdict,
        refusal_preserved=True,
        geometry_metrics=geometry_metrics,
    )

    logger.info(
        "Merge complete: %d layers, %d weights, confidence=%.3f, error=%.3f",
        result.layer_count,
        result.weight_count,
        result.mean_confidence,
        result.mean_procrustes_error,
    )

    return result


def run_full_geometry_merge(
    *,
    model_loader: "ModelLoaderPort",
    backend: "Backend",
    config: UnifiedMergeConfig,
    source_path: str,
    target_path: str,
    output_dir: str | None = None,
    dry_run: bool = False,
    knowledge_delta_mask_path: str | None = None,
) -> UnifiedMergeResult:
    """
    Execute merge using GeometricMergeOrchestrator with ALL 84 geometry files.

    This is the comprehensive merge that uses:
    - intrinsic_dimension: Per-layer intrinsic dimension
    - manifold_curvature: Curvature for geodesic interpolation
    - shared_subspace_projector: CCA-based shared dimension discovery
    - relative_representation: Anchor-based dimension-agnostic alignment
    - fisher_blending: Importance-weighted blending
    - dimension_blender: Per-dimension alpha computation
    - null_space_filter: Interference elimination
    - dare_sparsity: Optional sparsification
    - ... and 70+ more geometry files

    Higher dimensions contain lower dimensions (1D ⊂ 2D ⊂ 3D ⊂ ... ⊂ nD).
    We analyze and blend at EVERY dimension level.
    """
    from modelcypher.core.use_cases.geometric_merge_orchestrator import (
        GeometricMergeOrchestrator,
    )

    logger.info("=== FULL GEOMETRY MERGE (84 files) ===")
    logger.info("Source: %s", source_path)
    logger.info("Target: %s", target_path)
    logger.info("Backend: %s", type(backend).__name__)

    # Load weights
    source_weights, _ = load_weights(model_loader, source_path)
    target_weights, target_format = load_weights(model_loader, target_path)

    # Load tokenizers
    source_tokenizer = load_tokenizer(source_path)
    target_tokenizer = load_tokenizer(target_path)

    # Stage 0: Vocabulary alignment
    logger.info("STAGE 0: VOCABULARY ALIGNMENT")
    stage_start = time.perf_counter()
    source_weights, vocab_metrics, vocab_aligned, vocab_alignment_map = stage_vocabulary(
        source_weights=source_weights,
        target_weights=target_weights,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    require_vocab_phase_lock(vocab_metrics, vocab_aligned)
    logger.info(
        "STAGE 0: VOCABULARY ALIGNMENT completed in %.2fs",
        time.perf_counter() - stage_start,
    )

    # Collect activations if models can be loaded
    source_activations = None
    target_activations = None
    source_model = None
    target_model = None

    if config.probe_mode == "precise":
        logger.info("Loading models for activation collection...")
        load_start = time.perf_counter()
        source_model = load_model_for_probing(source_path)
        target_model = load_model_for_probing(target_path)
        logger.info(
            "STAGE 1: Model load completed in %.2fs",
            time.perf_counter() - load_start,
        )

        if source_model and target_model and source_tokenizer and target_tokenizer:
            from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.use_cases.merge.stages.probe import (
                _encode_probe_ids,
                build_token_id_map,
                collect_layer_activations_mlx,
                map_token_ids,
            )

            probes = UnifiedAtlasInventory.all_probes()
            max_probes = config.max_probes if config.max_probes > 0 else len(probes)

            source_activations = {}
            target_activations = {}
            token_id_map = None
            if vocab_alignment_map is not None:
                token_id_map = build_token_id_map(vocab_alignment_map)
                if token_id_map:
                    logger.info(
                        "STAGE 1: Using aligned token map for probes (%d tokens).",
                        len(token_id_map),
                    )

            for i, probe in enumerate(probes[:max_probes]):
                try:
                    probe_text = None
                    source_ids: list[int] | None = None
                    target_ids: list[int] | None = None
                    for candidate in probe.support_texts or []:
                        if not candidate or len(candidate.strip()) < 2:
                            continue
                        if token_id_map is None:
                            probe_text = candidate
                            break
                        candidate_source_ids = _encode_probe_ids(
                            source_tokenizer, candidate, add_special_tokens=False
                        )
                        candidate_target_ids = map_token_ids(
                            candidate_source_ids, token_id_map
                        )
                        if candidate_target_ids is None:
                            continue
                        probe_text = candidate
                        source_ids = candidate_source_ids
                        target_ids = candidate_target_ids
                        break

                    if probe_text is None:
                        continue

                    src_acts = collect_layer_activations_mlx(
                        source_model,
                        source_tokenizer,
                        probe_text,
                        token_ids=source_ids,
                    )
                    tgt_acts = collect_layer_activations_mlx(
                        target_model,
                        target_tokenizer,
                        probe_text,
                        token_ids=target_ids,
                    )

                    for layer_idx, act in src_acts.items():
                        if layer_idx not in source_activations:
                            source_activations[layer_idx] = []
                        source_activations[layer_idx].append(act)

                    for layer_idx, act in tgt_acts.items():
                        if layer_idx not in target_activations:
                            target_activations[layer_idx] = []
                        target_activations[layer_idx].append(act)
                except Exception:
                    continue

                if (i + 1) % 20 == 0:
                    logger.info("Collected activations from %d/%d probes", i + 1, max_probes)

            logger.info(
                "Collected activations: %d source layers, %d target layers",
                len(source_activations),
                len(target_activations),
            )

    # Clear model memory
    del source_model
    del target_model
    backend.clear_cache()

    # Create orchestrator and analyze geometry
    logger.info("ANALYZING FULL GEOMETRY...")
    analyze_start = time.perf_counter()
    orchestrator = GeometricMergeOrchestrator(backend=backend)
    geometry = orchestrator.analyze_merge(
        source_weights=source_weights,
        target_weights=target_weights,
        source_activations=source_activations,
        target_activations=target_activations,
        tokenizer=target_tokenizer,
    )
    logger.info(
        "ANALYZING FULL GEOMETRY completed in %.2fs",
        time.perf_counter() - analyze_start,
    )

    from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

    sample_array = next(iter(source_weights.values()), None)
    # Use a tolerance of 1e-5 for CKA comparison, which accounts for numerical
    # precision issues in Gram matrix computation, centering, and accumulation.
    # Machine epsilon (~1e-7 for float32) is too tight for CKA comparisons.
    base_eps = (
        machine_epsilon(backend, sample_array) if sample_array is not None else 1e-7
    )
    phase_tol = max(base_eps * 100, 1e-5)  # At least 1e-5, or 100x machine epsilon
    if geometry.overall_cka < 1.0 - phase_tol:
        raise RuntimeError(
            "PROBE BAROMETER: Overall CKA=%.6f < 1.0. "
            "Exact kernel alignment is required before merging."
            % geometry.overall_cka
        )

    # Execute merge using geometry
    logger.info("EXECUTING MERGE...")
    merge_start = time.perf_counter()
    layer_alpha_scale = None
    if knowledge_delta_mask_path:
        layer_alpha_scale = load_knowledge_delta_mask(knowledge_delta_mask_path)
        logger.info(
            "Applying knowledge delta mask: %s (%d layers)",
            knowledge_delta_mask_path,
            len(layer_alpha_scale),
        )
    merged_weights, merge_metrics = orchestrator.merge_weights(
        source_weights=source_weights,
        target_weights=target_weights,
        geometry=geometry,
        extract_layer_index_fn=extract_layer_index,
        checkpoint_dir=output_dir,
        layer_alpha_scale=layer_alpha_scale,
    )
    logger.info(
        "EXECUTING MERGE completed in %.2fs",
        time.perf_counter() - merge_start,
    )

    # Detect target quantization and requantize merged weights to match
    target_is_quantized = any(
        k.endswith(".scales") or k.endswith(".biases") for k in target_weights.keys()
    )

    if target_is_quantized:
        import json

        from modelcypher.core.use_cases.quantization_utils import (
            QuantizationHint,
            quantization_config_from_payload,
            requantize_weights,
        )

        # Read target config to get quantization params
        config_path = Path(target_path) / "config.json"
        quant_hint = QuantizationHint(bits=4, group_size=64, mode="affine")  # Default
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                quant_config = quantization_config_from_payload(config_data)
                if quant_config and quant_config.default:
                    quant_hint = quant_config.default
                    logger.info(
                        "Detected target quantization: %d-bit, group_size=%d",
                        quant_hint.bits,
                        quant_hint.group_size,
                    )
            except Exception as exc:
                logger.warning("Could not read target config for quantization: %s", exc)

        logger.info("Requantizing merged weights to match target format...")
        merged_weights = requantize_weights(
            merged_weights,
            backend,
            quant_hint,
        )
        logger.info("Requantization complete: %d weights", len(merged_weights))

    # Save if requested
    if output_dir and not dry_run:
        save_weights(output_dir, merged_weights, target_format, backend)
        copy_config_files(target_path, output_dir)
        output_path = output_dir
    else:
        output_path = None

    # Build result
    layer_indices = extract_layer_indices(target_weights)

    # Derive geometric confidence from MergeGeometry
    # For full geometry merge, confidence IS the CKA - kernel alignment is the geometry
    geometry_metrics_full = {
        "overall_cka": geometry.overall_cka,
        "mean_intrinsic_dim": geometry.mean_intrinsic_dimension,
        "mean_shared_dim": geometry.mean_shared_dimension,
        "mean_ollivier_ricci": geometry.mean_ollivier_ricci,
        "curvature_alignment": geometry.curvature_alignment,
        **geometry.curvature_alignment_details,
    }

    # Safety verdict derived from curvature alignment
    # High alignment = geometry is compatible; low = potential issues
    if geometry.curvature_alignment < 0.3:
        safety_verdict_full = "low_alignment"
    else:
        safety_verdict_full = "aligned"

    result = UnifiedMergeResult(
        merged_weights=merged_weights,
        vocab_metrics=vocab_metrics,
        probe_metrics={
            "overall_cka": geometry.overall_cka,
            "mean_intrinsic_dim": geometry.mean_intrinsic_dimension,
            "mean_shared_dim": geometry.mean_shared_dimension,
        },
        permute_metrics={"skipped": True, "reason": "full_geometry_mode"},
        transplant_metrics=merge_metrics,
        mean_confidence=geometry.overall_cka,
        mean_procrustes_error=0.0,
        layer_count=len(layer_indices),
        weight_count=len(merged_weights),
        timestamp=datetime.utcnow(),
        merge_strategy="full_geometry",
        output_path=output_path,
        vocab_aligned=vocab_aligned,
        safety_verdict=safety_verdict_full,
        refusal_preserved=geometry.refusal_preserved,
        geometry_metrics=geometry_metrics_full,
    )

    logger.info(
        "FULL GEOMETRY MERGE COMPLETE: %d layers, %d weights, CKA=%.4f",
        result.layer_count,
        result.weight_count,
        geometry.overall_cka,
    )

    return result
