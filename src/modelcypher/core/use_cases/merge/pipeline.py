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
from .models import UnifiedMergeConfig, UnifiedMergeResult
from .stages import (
    stage_density,
    stage_permute,
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
    default_config: UnifiedMergeConfig,
    source_path: str,
    target_path: str,
    output_dir: str | None = None,
    output_path: str | None = None,
    dry_run: bool = False,
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
        source_kv_activations,
        target_kv_activations,
        feature_transforms,
        attention_transforms,
        kv_transforms,
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
    probe_result.get("dimension_correlations", {})
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
        min_cka = probe_metrics.get("min_cka", 0.0)
        mean_cka = probe_metrics.get("mean_cka", 0.0)
        raise RuntimeError(
            "PROBE BAROMETER: Alignment not exact kernel aligned "
            "(mean_cka=%.4f, min_cka=%.4f). Resolve alignment before merge."
            % (mean_cka, min_cka)
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

    if kv_transforms:
        logger.info(
            "PROBE: Computed %d KV transforms",
            len(kv_transforms),
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
        source_kv_activations=source_kv_activations,
        target_kv_activations=target_kv_activations,
        transplant_domains=(),
        extract_layer_index_fn=extract_layer_index,
        backend=backend,
        graft_mask=graft_mask,
        feature_transforms=feature_transforms,
        attention_transforms=attention_transforms,
        kv_transforms=kv_transforms,
        layer_mapping=layer_mapping,
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

        # Preserve vocabulary-tied weights (embeddings, lm_head) from target.
        # These weren't modified during transplant but requantization would corrupt them
        # because dequantize→requantize is lossy.
        vocab_keys = {
            k for k in loaded_target_weights.keys()
            if "embed" in k.lower() or "lm_head" in k.lower()
        }
        vocab_weights = {k: loaded_target_weights[k] for k in vocab_keys}
        logger.info(
            "Preserving %d vocabulary-tied weights from target (skip requant)",
            len(vocab_weights),
        )

        logger.info("Requantizing merged weights to match target format...")
        merged_weights = requantize_weights(
            merged_weights,
            backend,
            quant_hint,
        )

        # Restore vocabulary weights (original target quantization)
        for k, v in vocab_weights.items():
            merged_weights[k] = v

        logger.info("Requantization complete: %d weights", len(merged_weights))

    # =================================================================
    # OUTPUT
    # =================================================================
    final_output_path: str | None = None
    if effective_output and not dry_run:
        save_weights(effective_output, merged_weights, target_format, backend)
        copy_config_files(target_path, effective_output)
        final_output_path = effective_output

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
    )

    logger.info(
        "Merge complete: %d layers, %d weights, preserved_fraction=%.3f, error=%.3f",
        result.layer_count,
        result.weight_count,
        result.mean_preserved_fraction,
        result.mean_procrustes_error,
    )

    return result
