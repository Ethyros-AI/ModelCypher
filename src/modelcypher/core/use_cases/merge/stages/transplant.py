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

"""Stage 3: TRANSPLANT - Null-space constrained knowledge grafting.

Replaces sparse concept regions while preserving boundary behavior:
    A_core @ W' = A_core @ W_source_aligned
    A_boundary @ W' = A_boundary @ W_target
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
# NOTE: ProjectionMethod import removed - use GRAM_TRANSPORT.
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
)
from modelcypher.core.domain.geometry.transplant import (
    partition_core_boundary,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_paired_distances,
)
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.use_cases.merge.stages.transplant_checkpoint import (
    _load_checkpoint,
    _save_checkpoint,
)
from modelcypher.core.use_cases.merge.stages.transplant_helpers import (
    _promote_precision,
)
from modelcypher.core.use_cases.merge.stages.transplant_stitches import (
    compute_composite_stitches,
)
from modelcypher.core.use_cases.merge.stages.transplant_embeddings import (
    apply_embedding_alignment,
)
from modelcypher.core.use_cases.merge.stages.transplant_weight_processor import (
    process_layer_weights,
)


from modelcypher.core.domain.merging.exceptions import (
    AlignmentFailureError,
    DimensionMismatchError,
    StitchUnavailableError,
)
from modelcypher.core.use_cases.merge.stages.manifest import (
    TransplantManifest,
    WeightStatus,
    WeightTransformRecord,
)
from modelcypher.core.use_cases.merge.stages.density import (
    filter_core_probes_by_graft_mask,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class TransplantStageResult:
    """Result of transplant stage."""

    merged_weights: dict[str, Any]
    metrics: dict[str, Any]
    manifest: TransplantManifest | None = None  # Track every weight's status


def stage_transplant(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    layer_indices: list[int],
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    target_activations: dict[int, list["Array"]] | None,
    extract_layer_index_fn: Callable[[str], int | None],
    source_activations: dict[int, list["Array"]] | None = None,
    source_intermediate_activations: dict[int, list["Array"]] | None = None,
    target_intermediate_activations: dict[int, list["Array"]] | None = None,
    source_attention_activations: dict[int, list["Array"]] | None = None,
    target_attention_activations: dict[int, list["Array"]] | None = None,
    source_kv_activations: dict[int, list["Array"]] | None = None,
    target_kv_activations: dict[int, list["Array"]] | None = None,
    graft_mask: dict[str, dict[int, bool]] | None = None,
    density_weights: dict[int, "Array"] | None = None,  # Per-probe source density ratios
    feature_transforms: dict[int, "Array"] | None = None,  # GPU arrays from GramAligner
    scale_ratios: dict[int, float] | None = None,  # EXACT: ||target|| / ||source @ F||
    embedding_transform: "Array | None" = None,  # 2D GramAlign transform (GPU array)
    attention_transforms: dict[int, "Array"] | None = None,  # GPU arrays
    k_transforms: dict[int, "Array"] | None = None,  # GPU arrays
    v_transforms: dict[int, "Array"] | None = None,  # GPU arrays
    intermediate_transforms: dict[int, "Array"] | None = None,  # MLP intermediate (GPU arrays)
    gate_transforms: dict[int, "Array"] | None = None,  # PRE-SiLU gate transforms (GPU arrays)
    layer_mapping: dict[int, int] | None = None,
    layer_status: dict[int, str] | None = None,  # NEW: Per DIMENSIONAL_COMPRESSION.md
    prior_occupancy_by_layer: dict[int, list[float]] | None = None,
    source_tokenizer: "Any | None" = None,  # For token correspondence
    target_tokenizer: "Any | None" = None,  # For token correspondence
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    backend: "Backend | None" = None,
    delta_scale: float = 1.0,
    # Anchor-relative grafting parameters (canonical pipeline)
    source_anchors: dict[int, "Array"] | None = None,  # Per-layer anchor embeddings
    target_anchors: dict[int, "Array"] | None = None,  # Per-layer anchor embeddings
) -> TransplantStageResult:
    """Stage 3: Null-space constrained transplant using probe activations.

    Linear alignment is closed-form on the shared manifold.
    Layer status is vestigial: all layers are processed.
    "boundary_preserved" and "skipped" are retained for API compatibility
    but should rarely occur; deviations reflect overlap/coverage, not a hard failure.

    Args:
        delta_scale: Scale factor for projected deltas (0.0-1.0). Use < 1.0 for
            sequential stacking to stay within cumulative delta budget. Default
            1.0 = full projection. Threshold is 1% of baseline weight norm -
            exceeding causes generation degradation.
    """
    b = backend or get_default_backend()
    # Release probe/density geodesic caches before heavy transplant work.
    ComputationCache.shared().clear_all()
    if hasattr(b, "clear_cache"):
        b.clear_cache()
    merged: dict[str, "Array"] = dict(target_weights)

    metrics: dict[str, Any] = {
        "merge_strategy": "transplant",
        "layers_considered": 0,
        "layers_transplanted": 0,
        "weights_considered": 0,
        "weights_transplanted": 0,
        "preserved_fractions": [],
        "projection_losses": [],
        "null_dims": [],
        "boundary_relative_diffs": [],
        "core_distance_reductions": [],
        "core_dist_to_source_before": [],
        "core_dist_to_source_after": [],
        "cka_before": [],
        "cka_after": [],
        "core_probes": 0,
    }
    manifest = TransplantManifest()

    def _record_manifest(
        key: str,
        status: WeightStatus,
        source_shape: tuple[int, ...] | None = None,
        target_shape: tuple[int, ...] | None = None,
        stitch_type: str | None = None,
        preserved_fraction: float | None = None,
        cka_achieved: float | None = None,
        error_message: str | None = None,
    ) -> None:
        manifest.record(
            key,
            WeightTransformRecord(
                key=key,
                status=status,
                source_shape=source_shape,
                target_shape=target_shape,
                stitch_type=stitch_type,
                preserved_fraction=preserved_fraction,
                cka_achieved=cka_achieved,
                error_message=error_message,
            ),
        )

    occupancy_by_layer: dict[int, "Array"] = {}
    prior_occupancy_arrays: dict[int, "Array"] = {}
    if prior_occupancy_by_layer:
        for layer_idx, occ in prior_occupancy_by_layer.items():
            if occ is None:
                continue
            occ_arr = b.array(occ)
            occ_arr = _promote_precision(occ_arr, b)
            b.eval(occ_arr)
            prior_occupancy_arrays[int(layer_idx)] = occ_arr
        metrics["occupancy_prior_layers"] = len(prior_occupancy_arrays)

    # REQUIRE real activations collected from probe runs.
    if not target_activations:
        error_msg = (
            "Transplant requires real activations collected from probe runs. "
            "Use `mc merge` to collect activations before merging."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    metrics["activation_source"] = "collected_from_model"

    # Probe-based transplant requires metadata
    if not probe_ids or not probe_domains:
        raise RuntimeError("Transplant requires probe metadata (probe_ids, probe_domains)")

    if len(probe_ids) != len(probe_domains):
        raise RuntimeError("Transplant probe metadata mismatch (probe_ids != probe_domains).")

    if graft_mask is None:
        raise RuntimeError("Transplant requires a graft_mask from density stage.")

    core_probe_ids = set(probe_ids)
    logger.info(
        "TRANSPLANT: Selective mode - %d candidate probes, graft_mask decides",
        len(core_probe_ids)
    )

    metrics["core_probes"] = len(core_probe_ids)
    metrics["density_only_mode"] = True  # Always geometry-driven now

    if not core_probe_ids:
        raise RuntimeError("Transplant requires non-empty probe_ids.")

    weights_by_layer: dict[int, list[str]] = {}
    for key in target_weights:
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is None:
            continue
        weights_by_layer.setdefault(layer_idx, []).append(key)

    # Check for existing checkpoint
    resume_from_layer = -1
    if checkpoint_dir:
        checkpoint_result = _load_checkpoint(checkpoint_dir)
        if checkpoint_result:
            resume_from_layer, checkpoint_meta = checkpoint_result
            # Restore metrics from checkpoint
            metrics["weights_transplanted"] = checkpoint_meta.get("weights_transplanted", 0)
            metrics["layers_transplanted"] = checkpoint_meta.get("layers_transplanted", 0)

    # Count total weights for progress reporting
    total_layers = len(layer_indices)
    total_weights = sum(len(weights_by_layer.get(idx, [])) for idx in layer_indices)
    weights_processed = 0
    stage_start_time = time.time()

    logger.info(
        "TRANSPLANT: Starting stage 3 - %d layers, %d total weights (geometry-driven preserved_fraction)",
        total_layers, total_weights
    )

    # ==========================================================================
    # 2D EMBEDDING ALIGNMENT: Preserve target vocab interface
    # ==========================================================================
    apply_embedding_alignment(
        source_weights=source_weights,
        target_weights=target_weights,
        embedding_transform=embedding_transform,
        merged=merged,
        metrics=metrics,
        backend=b,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    # ==========================================================================
    # PER-LAYER ALIGNMENT: Use transforms from probe stage (linear alignment on shared manifold)
    # ==========================================================================
    # RIGOROUS GEOMETRY: No fallbacks. If probe stage didn't compute a transform,
    # that layer will not have stitches. The per-weight logic handles this by
    # raising AlignmentFailureError when a required stitch is unavailable.
    #
    # Attention weights have a different dimension than hidden:
    #   - q_proj: [num_heads * head_dim, hidden_dim] (e.g., [960, 960] for SmolLM)
    #   - o_proj: [hidden_dim, num_heads * head_dim]
    # When head counts differ (e.g., SmolLM=15 heads → Qwen=14 heads):
    #   - SmolLM attention dim = 15 * 64 = 960
    #   - Qwen attention dim = 14 * 64 = 896
    # ==========================================================================
    # 1. PER-LAYER HIDDEN ALIGNMENT
    # ==========================================================================
    # Log scale_ratios status for debugging
    if scale_ratios:
        logger.info("SCALE RATIOS: %d layers with scale factors", len(scale_ratios))
        sample_layers = list(sorted(scale_ratios.keys()))[:3]
        for l in sample_layers:
            logger.info("  Layer %d scale_ratio=%.4f", l, scale_ratios[l])
    else:
        logger.warning("SCALE RATIOS: Empty or None!")
    
    layer_hidden_stitches = compute_composite_stitches(
        transforms_map=feature_transforms,
        desc="HIDDEN",
        backend=b,
        layer_mapping=layer_mapping,
        layer_scale_ratios=scale_ratios,
    )
    metrics["per_layer_alignment"] = bool(layer_hidden_stitches)
    metrics["layers_with_transforms"] = len(layer_hidden_stitches)

    # ==========================================================================
    # 2. PER-LAYER ATTENTION ALIGNMENT (Q)
    # ==========================================================================
    layer_attention_stitches = compute_composite_stitches(
        transforms_map=attention_transforms,
        desc="ATTENTION Q",
        backend=b,
        layer_mapping=layer_mapping,
    )

    # RIGOROUS GEOMETRY: No fallbacks for attention.
    # If attention transforms are missing, those layers won't have attention stitches.
    # The per-weight logic will raise AlignmentFailureError if a stitch is required but missing.
    metrics["per_layer_attention_alignment"] = bool(layer_attention_stitches)

    # ==========================================================================
    # 3. PER-LAYER K/V ALIGNMENT
    # ==========================================================================
    layer_k_stitches = compute_composite_stitches(
        transforms_map=k_transforms,
        desc="K PROJ",
        backend=b,
        layer_mapping=layer_mapping,
    )
    layer_v_stitches = compute_composite_stitches(
        transforms_map=v_transforms,
        desc="V PROJ",
        backend=b,
        layer_mapping=layer_mapping,
    )
    
    metrics["per_layer_k_alignment"] = bool(layer_k_stitches)
    metrics["per_layer_v_alignment"] = bool(layer_v_stitches)

    # ==========================================================================
    # 4. PER-LAYER INTERMEDIATE (MLP) ALIGNMENT
    # ==========================================================================
    # OPTIMIZATION: Use pre-computed intermediate transforms from probe stage
    # instead of running GramAligner per-layer (~50k steps saved per layer)
    layer_intermediate_stitches = compute_composite_stitches(
        transforms_map=intermediate_transforms,
        desc="INTERMEDIATE",
        backend=b,
        layer_mapping=layer_mapping,
    )
    metrics["per_layer_intermediate_alignment"] = bool(layer_intermediate_stitches)

    layer_gate_stitches = compute_composite_stitches(
        transforms_map=gate_transforms,
        desc="GATE",
        backend=b,
        layer_mapping=layer_mapping,
    )
    metrics["per_layer_gate_alignment"] = bool(layer_gate_stitches)

    # ==========================================================================
    # RIGOROUS GEOMETRY: All transforms handled above via _compute_composite_stitches.
    # No fallbacks - if transforms missing, the per-weight logic handles errors.
    # ==========================================================================

    for layer_num, layer_idx in enumerate(layer_indices):
        # Skip layers already completed (checkpoint resume)
        if layer_idx <= resume_from_layer:
            weights_processed += len(weights_by_layer.get(layer_idx, []))
            logger.debug("TRANSPLANT: Skipping layer %d (already completed)", layer_idx)
            continue

        # =======================================================================
        # LAYER STATUS CHECK (Vestigial - diagnostic only)
        # =======================================================================
        # "skipped" and "boundary_preserved" are retained for API compatibility.
        # If they appear, log diagnostics and proceed with the layer.
        if layer_status:
            status = layer_status.get(layer_idx, "converged")
            if status == "skipped":
                logger.warning(
                    "TRANSPLANT: Layer %d marked 'skipped' by diagnostics; proceeding.",
                    layer_idx
                )
                # Still process the layer - don't give up
            elif status == "boundary_preserved":
                logger.warning(
                    "TRANSPLANT: Layer %d marked 'boundary_preserved' by diagnostics; proceeding.",
                    layer_idx
                )
                # Still process the layer - don't give up
            # All layers fall through to normal transplant

        # =======================================================================
        # LAYER 0 BOUNDARY: Preserve embedding-to-hidden interface
        # =======================================================================
        # Layer 0 directly receives embeddings. Transplanted weights were trained
        # on source's embedding scale, but we're feeding target's embeddings.
        # 
        # This creates a 21x scale mismatch: merged Layer 0 output = 0.05x target.
        # 
        # GEOMETRY: Layer 0 is the 1D→2D→ND transition boundary. Preserve it
        # from target to maintain embedding scale compatibility.
        # =======================================================================
        if layer_idx == 0:
            layer_keys = weights_by_layer.get(layer_idx, [])
            weights_processed += len(layer_keys)
            for key in layer_keys:
                target_w = target_weights.get(key)
                _record_manifest(
                    key,
                    WeightStatus.SKIPPED_BOUNDARY,
                    target_shape=tuple(target_w.shape) if hasattr(target_w, "shape") else None,
                    error_message="layer 0 boundary preserved",
                )
            logger.info(
                "TRANSPLANT: Layer 0 PRESERVED (embedding-to-hidden boundary)"
            )
            metrics.setdefault("boundary_preserved_layers", []).append(layer_idx)
            metrics["layer_0_preserved"] = True
            continue

        layer_keys = weights_by_layer.get(layer_idx, [])
        if not layer_keys:
            continue

        layer_start_time = time.time()
        logger.info(
            "TRANSPLANT: Layer %d/%d (index=%d) - %d weights",
            layer_num + 1, total_layers, layer_idx, len(layer_keys)
        )

        # Get REAL activations from collected probes (required)
        act_list = target_activations.get(layer_idx)
        # Check for missing activations (handle both list and array formats)
        if act_list is None or (hasattr(act_list, '__len__') and len(act_list) == 0):
            raise AlignmentFailureError(
                stage="LAYER_ACTIVATION_VALIDATION",
                weight_key=None,
                message="Missing target activations for layer",
                context={"layer_idx": layer_idx},
            )

        # Get number of activations (works for both list and 2D array)
        n_acts = len(act_list) if hasattr(act_list, '__len__') else int(b.shape(act_list)[0])
        if n_acts != len(probe_ids):
            raise AlignmentFailureError(
                stage="LAYER_ACTIVATION_VALIDATION",
                weight_key=None,
                message="Probe count mismatch for layer activations",
                context={
                    "layer_idx": layer_idx,
                    "activations": n_acts,
                    "probes": len(probe_ids),
                },
            )

        metrics["layers_considered"] += 1

        # Multi-space stitching: compute stitches for hidden, intermediate, AND attention dimensions
        # Hidden stitch: maps layer output activations (source hidden → target hidden)
        # Intermediate stitch: maps MLP internal activations (source intermediate → target intermediate)
        # Attention stitch: maps attention head outputs (source num_heads*head_dim → target num_heads*head_dim)
        #
        # IMPORTANT: For weight folding, we need TWO transforms per space:
        #   - F.T for OUTPUT side (left multiply): maps source output → target output
        #   - pinv(F).T for INPUT side (right multiply): inverse maps source input → target input
        #
        # Weight transform: W_target = F_out.T @ W_source @ pinv(F_in).T
        hidden_stitch_output = None  # F.T for output side [tgt_hidden, src_hidden]
        hidden_stitch_input = None   # pinv(F).T for input side [src_hidden, tgt_hidden]
        intermediate_stitch_output = None  # F.T for output side
        intermediate_stitch_input = None   # pinv(F).T for input side
        gate_stitch_output = None  # F.T for PRE-SiLU gate output
        gate_stitch_input = None   # pinv(F).T for PRE-SiLU gate input
        attention_stitch_output = None  # F.T for Q attention output [tgt_attn, src_attn]
        attention_stitch_input = None   # pinv(F).T for Q attention input [src_attn, tgt_attn]
        k_stitch_output = None  # F.T for K attention output [tgt_k, src_k]
        k_stitch_input = None   # pinv(F).T for K attention input [src_k, tgt_k]
        v_stitch_output = None  # F.T for V attention output [tgt_v, src_v]
        v_stitch_input = None   # pinv(F).T for V attention input [src_v, tgt_v]
        kv_stitch_input = None  # Derived from k_stitch_input for o_proj KV-dim cases

        # Get intermediate activations for this layer (MLP internal states)
        # CRITICAL: For cross-architecture merge, source layer index may differ from target.
        # IMPORTANT: The hidden-space layer_mapping is optimized for hidden activations, but
        # intermediate space has different semantic structure. ALWAYS use proportional mapping
        # for intermediate activations to ensure we compare semantically similar layers.
        # =================================================================
        # GRAM ALIGNMENT: Closed-form linear transforms for hidden AND intermediate
        # =================================================================
        # GramAligner finds the closed-form linear transform on the shared manifold.
        # This is an exact solution for the overlap; geodesic CKA is diagnostic.
        #
        # The feature_transform maps source activations to target space such that
        # their Gram matrices (relational geometry) match on the shared manifold.
        #
        # MLP weights have shape [intermediate, hidden] or [hidden, intermediate].
        # We need transforms for BOTH axes to properly map weights.
        #
        # OPTIMIZATION: All transforms are now pre-computed in probe stage.

        # USE PER-LAYER ALIGNED hidden stitch (from probe stage with linear alignment)
        # Each layer gets its own transform - different layers encode different
        # parts of the geometry at different resolutions.
        hidden_stitch_output = None
        hidden_stitch_input = None

        if layer_idx in layer_hidden_stitches:
            # layer_hidden_stitches[layer_idx] is a dict {src_layer: (P, Q)}
            # For now, use the first/only source's stitch (most layers have 1:1 or composite merged)
            src_stitches_dict = layer_hidden_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                hidden_stitch_output, hidden_stitch_input = src_stitches_dict[first_src]
            # Log only on first layer to avoid spam
            if layer_num == 0:
                stitch_shape = hidden_stitch_output.shape if hidden_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer hidden stitch (shape=%s)",
                    layer_idx, stitch_shape
                )
        elif layer_hidden_stitches:
            # Layer not in mapping - might be unmapped target layer
            # Use identity if we have any transforms (cross-arch case)
            pass  # hidden_stitch_output/input stay None - will skip stitching

        # =================================================================
        # USE PER-LAYER INTERMEDIATE stitch (from probe stage with linear alignment)
        # =================================================================
        # OPTIMIZATION: Use pre-computed transforms from probe stage instead of
        # running GramAligner per-layer. This saves ~50k optimization steps per layer.
        if layer_idx in layer_intermediate_stitches:
            src_stitches_dict = layer_intermediate_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                intermediate_stitch_output, intermediate_stitch_input = src_stitches_dict[first_src]
            if layer_num == 0 or intermediate_stitch_output is not None:
                stitch_shape = intermediate_stitch_output.shape if intermediate_stitch_output is not None else "N/A"
                # Diagnostic: log the intermediate stitch dimensions for debugging dimension swaps
                logger.info(
                    "TRANSPLANT INTER: Layer %d: intermediate_stitch_output shape=%s "
                    "(shape[0]=tgt_inter, shape[1]=src_inter)",
                    layer_idx, stitch_shape
                )
            metrics.setdefault("intermediate_cached_stitches", 0)
            metrics["intermediate_cached_stitches"] += 1

        if layer_idx in layer_gate_stitches:
            src_stitches_dict = layer_gate_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                gate_stitch_output, gate_stitch_input = src_stitches_dict[first_src]
            if layer_num == 0 and gate_stitch_output is not None:
                stitch_shape = gate_stitch_output.shape
                logger.info(
                    "Layer %d: Using per-layer gate stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        # USE PER-LAYER ATTENTION stitch (from probe stage with linear alignment)
        if layer_idx in layer_attention_stitches:
            # layer_attention_stitches[layer_idx] is a dict {src_layer: (P, Q)}
            src_stitches_dict = layer_attention_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                attention_stitch_output, attention_stitch_input = src_stitches_dict[first_src]
            if layer_num == 0:
                stitch_shape = attention_stitch_output.shape if attention_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer Q attention stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        # USE PER-LAYER K stitch for k_proj (from probe stage with linear alignment)
        if layer_idx in layer_k_stitches:
            # layer_k_stitches[layer_idx] is a dict {src_layer: (P, Q)}
            src_stitches_dict = layer_k_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                k_stitch_output, k_stitch_input = src_stitches_dict[first_src]
            if layer_num == 0:
                stitch_shape = k_stitch_output.shape if k_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer K stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        # USE PER-LAYER V stitch for v_proj (from probe stage with linear alignment)
        if layer_idx in layer_v_stitches:
            # layer_v_stitches[layer_idx] is a dict {src_layer: (P, Q)}
            src_stitches_dict = layer_v_stitches[layer_idx]
            if src_stitches_dict:
                first_src = next(iter(src_stitches_dict))
                v_stitch_output, v_stitch_input = src_stitches_dict[first_src]
            if layer_num == 0:
                stitch_shape = v_stitch_output.shape if v_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer V stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        # Derive kv_stitch_input from k_stitch_input for o_proj KV-dimension cases
        # K and V share head structure, so K stitch dimensions work for combined KV
        if k_stitch_input is not None:
            kv_stitch_input = k_stitch_input

        # =====================================================================
        # OPTIMIZATION: Cache stitch dimensions once per layer
        # =====================================================================
        # Each .shape on a lazy MLX array forces GPU→CPU sync. By caching dimensions
        # once after loading stitch outputs, we avoid 50+ redundant syncs per layer.
        stitch_dims: dict[str, int] = {}
        if hidden_stitch_output is not None:
            stitch_dims['src_hidden'] = int(hidden_stitch_output.shape[1])
            stitch_dims['tgt_hidden'] = int(hidden_stitch_output.shape[0])
        if intermediate_stitch_output is not None:
            stitch_dims['src_inter'] = int(intermediate_stitch_output.shape[1])
            stitch_dims['tgt_inter'] = int(intermediate_stitch_output.shape[0])
        if attention_stitch_output is not None:
            stitch_dims['src_attn'] = int(attention_stitch_output.shape[1])
            stitch_dims['tgt_attn'] = int(attention_stitch_output.shape[0])
        if k_stitch_output is not None:
            stitch_dims['src_kv'] = int(k_stitch_output.shape[1])
            stitch_dims['tgt_kv'] = int(k_stitch_output.shape[0])
        if v_stitch_output is not None:
            stitch_dims['src_v'] = int(v_stitch_output.shape[1])
            stitch_dims['tgt_v'] = int(v_stitch_output.shape[0])

        # Handle both formats: list of 1D arrays (legacy) or 2D array (memory-optimized)
        if hasattr(act_list, 'shape') and len(b.shape(act_list)) == 2:
            # Already a 2D array [n_probes, hidden_dim] - use directly
            stacked = act_list
        else:
            # List of 1D arrays - stack them
            stacked = b.stack(act_list, axis=0)
        # Convert to float32 for numerical stability in linalg operations.
        stacked = _promote_precision(stacked, b)
        b.eval(stacked)

        layer_dim = int(stacked.shape[1])

        # =====================================================================
        # ALIGNED SOURCE ACTIVATIONS FOR DENSITY-AWARE TRANSFER
        # =====================================================================
        # For density-aware transfer, we need source activations aligned to target space.
        # - For hidden weights: source_hidden @ F → target hidden space
        # - For MLP weights: source_intermediate @ I → target intermediate space
        # This enables neuron-level density comparison.
        aligned_source_hidden: "Array | None" = None
        aligned_source_inter: "Array | None" = None

        # Align source hidden activations using F (feature transform)
        if source_activations is not None and layer_idx in source_activations:
            src_acts = source_activations[layer_idx]
            if hasattr(src_acts, 'shape') and len(b.shape(src_acts)) == 2:
                src_hidden = src_acts
            else:
                src_hidden = b.stack(src_acts, axis=0)
            src_hidden = _promote_precision(src_hidden, b)
            b.eval(src_hidden)

            # Apply alignment transform F
            if hidden_stitch_output is not None:
                # hidden_stitch_input is pinv(F).T which goes target→source
                # To go source→target, we use hidden_stitch_output.T
                # But stitch_output is F.T [tgt, src], so we need F [src, tgt]
                # source @ F = source @ stitch_output.T
                F = b.transpose(hidden_stitch_output)  # [src_hidden, tgt_hidden]
                aligned_source_hidden = b.matmul(src_hidden, F)
                b.eval(aligned_source_hidden)
                logger.debug(
                    "Layer %d: Aligned source hidden [%d,%d] → [%d,%d]",
                    layer_idx, int(src_hidden.shape[0]), int(src_hidden.shape[1]),
                    int(aligned_source_hidden.shape[0]), int(aligned_source_hidden.shape[1])
                )
            else:
                # No transform available - use unaligned (same-dim case)
                if int(src_hidden.shape[1]) == layer_dim:
                    aligned_source_hidden = src_hidden

        # Align source intermediate activations using I (intermediate transform)
        if source_intermediate_activations is not None and layer_idx in source_intermediate_activations:
            src_inter_acts = source_intermediate_activations[layer_idx]
            if hasattr(src_inter_acts, 'shape') and len(b.shape(src_inter_acts)) == 2:
                src_inter = src_inter_acts
            else:
                src_inter = b.stack(src_inter_acts, axis=0)
            src_inter = _promote_precision(src_inter, b)
            b.eval(src_inter)

            # Apply intermediate alignment transform I
            if intermediate_stitch_output is not None:
                I = b.transpose(intermediate_stitch_output)  # [src_inter, tgt_inter]
                aligned_source_inter = b.matmul(src_inter, I)
                b.eval(aligned_source_inter)
                logger.debug(
                    "Layer %d: Aligned source inter [%d,%d] → [%d,%d]",
                    layer_idx, int(src_inter.shape[0]), int(src_inter.shape[1]),
                    int(aligned_source_inter.shape[0]), int(aligned_source_inter.shape[1])
                )

        prior_occupancy = prior_occupancy_arrays.get(layer_idx)
        if prior_occupancy is not None and int(prior_occupancy.shape[0]) != layer_dim:
            logger.warning(
                "OCCUPANCY: Layer %d dim mismatch (expected %d, got %d) - ignoring",
                layer_idx,
                layer_dim,
                int(prior_occupancy.shape[0]),
            )
            prior_occupancy = None
        if prior_occupancy is None:
            prior_occupancy = b.zeros((layer_dim,), dtype="float32")
        else:
            prior_occupancy = _promote_precision(prior_occupancy, b)
        layer_delta_occupancy = b.zeros((layer_dim,), dtype="float32")
        b.eval(prior_occupancy, layer_delta_occupancy)

        # Filter core probes by graft mask (density-based selection)
        # Only include probes where source is denser than target at this layer
        effective_core_probes = filter_core_probes_by_graft_mask(
            core_probe_ids=core_probe_ids,
            layer_idx=layer_idx,
            graft_mask=graft_mask,
        )

        if not effective_core_probes:
            logger.debug(
                "Layer %d: All core probes filtered by graft mask (target already dense)",
                layer_idx,
            )
            metrics.setdefault("layers_skipped_by_density", 0)
            metrics["layers_skipped_by_density"] += 1
            for key in layer_keys:
                target_w = target_weights.get(key)
                _record_manifest(
                    key,
                    WeightStatus.SKIPPED_DENSITY_FILTER,
                    target_shape=tuple(target_w.shape) if hasattr(target_w, "shape") else None,
                    error_message="density graft mask filtered layer",
                )
            continue

        # boundary_k and geodesic_k_neighbors are derived from geodesic connectivity
        # within partition_core_boundary - no user configuration needed
        partition = partition_core_boundary(
            activations=stacked,
            probe_ids=probe_ids,
            core_probe_ids=effective_core_probes,
            backend=b,
        )

        if not partition.core_indices:
            for key in layer_keys:
                target_w = target_weights.get(key)
                _record_manifest(
                    key,
                    WeightStatus.SKIPPED_DENSITY_FILTER,
                    target_shape=tuple(target_w.shape) if hasattr(target_w, "shape") else None,
                    error_message="no core probes after density partition",
                )
            continue

        core_indices = b.array(partition.core_indices, dtype="int32")
        core_acts = b.take(stacked, core_indices, axis=0)
        b.eval(core_acts)

        boundary_indices = None
        if partition.boundary_indices:
            boundary_indices = b.array(partition.boundary_indices, dtype="int32")
            boundary_acts = b.take(stacked, boundary_indices, axis=0)
            b.eval(boundary_acts)
        else:
            boundary_acts = b.zeros((0, int(stacked.shape[1])))
            b.eval(boundary_acts)

        layer_transplanted = False
        best_alignment: dict[str, float] | None = None
        best_delta_norm = -1.0
        can_measure_alignment = core_acts is not None and int(core_acts.shape[0]) >= 2

        weight_result = process_layer_weights(
            layer_idx=layer_idx,
            layer_keys=layer_keys,
            source_weights=source_weights,
            target_weights=target_weights,
            merged=merged,
            metrics=metrics,
            layer_mapping=layer_mapping,
            extract_layer_index_fn=extract_layer_index_fn,
            backend=b,
            total_weights=total_weights,
            weights_processed=weights_processed,
            progress_callback=progress_callback,
            hidden_stitch_output=hidden_stitch_output,
            hidden_stitch_input=hidden_stitch_input,
            intermediate_stitch_output=intermediate_stitch_output,
            intermediate_stitch_input=intermediate_stitch_input,
            gate_stitch_output=gate_stitch_output,
            gate_stitch_input=gate_stitch_input,
            attention_stitch_output=attention_stitch_output,
            attention_stitch_input=attention_stitch_input,
            k_stitch_output=k_stitch_output,
            k_stitch_input=k_stitch_input,
            v_stitch_output=v_stitch_output,
            v_stitch_input=v_stitch_input,
            kv_stitch_input=kv_stitch_input,
            stitch_dims=stitch_dims,
            source_activations=source_activations,
            target_activations=target_activations,
            source_intermediate_activations=source_intermediate_activations,
            target_intermediate_activations=target_intermediate_activations,
            density_weights_by_layer=density_weights,
            core_acts=core_acts,
            boundary_acts=boundary_acts,
            can_measure_alignment=can_measure_alignment,
            manifest=manifest,
            delta_scale=delta_scale,
        )
        weights_processed = weight_result.weights_processed
        layer_transplanted = weight_result.layer_transplanted
        best_alignment = weight_result.best_alignment
        best_delta_norm = weight_result.best_delta_norm

        if layer_transplanted:
            metrics["layers_transplanted"] += 1

        layer_occupancy = 1.0 - (1.0 - prior_occupancy) * (1.0 - layer_delta_occupancy)
        b.eval(layer_occupancy)
        occupancy_by_layer[layer_idx] = layer_occupancy

        # Layer timing summary
        layer_elapsed = time.time() - layer_start_time
        logger.info(
            "TRANSPLANT: Layer %d complete - %.2fs (%d weights transplanted)",
            layer_idx, layer_elapsed, metrics["weights_transplanted"]
        )

        # Save checkpoint after each layer
        if checkpoint_dir:
            _save_checkpoint(checkpoint_dir, layer_idx, metrics)

        ComputationCache.shared().clear_geometry_caches()
        if hasattr(b, "clear_cache"):
            b.clear_cache()

        if best_alignment is not None:
            metrics["core_distance_reductions"].append(best_alignment["core_distance_reduction"])
            metrics["core_dist_to_source_before"].append(
                best_alignment["core_dist_to_source_before"]
            )
            metrics["core_dist_to_source_after"].append(
                best_alignment["core_dist_to_source_after"]
            )
            metrics["cka_before"].append(best_alignment["cka_before"])
            metrics["cka_after"].append(best_alignment["cka_after"])

    if metrics["preserved_fractions"]:
        pres = metrics["preserved_fractions"]
        metrics["mean_preserved_fraction"] = sum(pres) / len(pres)
    if metrics["projection_losses"]:
        losses = metrics["projection_losses"]
        metrics["mean_projection_loss"] = sum(losses) / len(losses)
    if metrics["null_dims"]:
        null_dims = metrics["null_dims"]
        metrics["mean_null_dim"] = sum(null_dims) / len(null_dims)
    if metrics["boundary_relative_diffs"]:
        diffs = metrics["boundary_relative_diffs"]
        metrics["mean_boundary_relative_diff"] = sum(diffs) / len(diffs)
        metrics["max_boundary_relative_diff"] = max(diffs)
    if metrics["core_distance_reductions"]:
        reductions = metrics["core_distance_reductions"]
        metrics["core_distance_samples"] = len(reductions)
        metrics["mean_core_distance_reduction"] = sum(reductions) / len(reductions)
    if metrics["core_dist_to_source_before"]:
        dists = metrics["core_dist_to_source_before"]
        metrics["mean_core_dist_to_source_before"] = sum(dists) / len(dists)
    if metrics["core_dist_to_source_after"]:
        dists = metrics["core_dist_to_source_after"]
        metrics["mean_core_dist_to_source_after"] = sum(dists) / len(dists)
    if metrics["cka_before"]:
        ckas = metrics["cka_before"]
        metrics["mean_cka_before"] = sum(ckas) / len(ckas)
    if metrics["cka_after"]:
        ckas = metrics["cka_after"]
        metrics["mean_cka_after"] = sum(ckas) / len(ckas)
    if occupancy_by_layer:
        occupancy_payload: dict[str, list[float]] = {}
        for layer_idx, occ in occupancy_by_layer.items():
            b.eval(occ)
            occupancy_payload[str(layer_idx)] = b.tolist(occ)
        metrics["occupancy_by_layer"] = occupancy_payload
        metrics["occupancy_layers"] = len(occupancy_payload)
    # Stage completion summary
    stage_elapsed = time.time() - stage_start_time
    metrics["total_time_seconds"] = stage_elapsed
    metrics["manifest"] = manifest.to_dict()
    logger.info(
        "TRANSPLANT: Stage 3 complete - %.2fs total (%d/%d layers, %d/%d weights)",
        stage_elapsed,
        metrics["layers_transplanted"], metrics["layers_considered"],
        metrics["weights_transplanted"], metrics["weights_considered"],
    )

    # Convert all weights to bfloat16 for consistent output format.
    # This handles the case where original weights are bf16 but transplanted
    # weights are float32 from the numerical computations.
    logger.debug("Converting %d merged weights to bfloat16 for output", len(merged))
    output_weights: dict[str, Any] = {}
    for key, weight in merged.items():
        if hasattr(weight, 'dtype'):
            dtype_str = str(weight.dtype).lower()
            # Skip quantized weights (uint32, int4, etc) - keep as-is
            if 'int' in dtype_str or 'uint' in dtype_str:
                output_weights[key] = weight
            else:
                # Convert to bfloat16 for storage efficiency
                try:
                    output_weights[key] = b.astype(b.array(weight), "bfloat16")
                except Exception as e:
                    logger.debug("Could not convert %s to bfloat16: %s", key, e)
                    output_weights[key] = weight
        else:
            output_weights[key] = weight

    return TransplantStageResult(
        merged_weights=output_weights,
        metrics=metrics,
        manifest=manifest,
    )


__all__ = [
    "TransplantStageResult",
    "stage_transplant",
]
