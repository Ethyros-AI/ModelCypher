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

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from modelcypher.core.domain._backend import get_default_backend
# NOTE: ProjectionMethod import removed - always use GRAM_TRANSPORT (the only correct method)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.transplant import (
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_norms,
    geodesic_paired_distances,
)


def _geodesic_pinv(backend: "Backend", F: "Array") -> "Array":
    """Compute EXACT Moore-Penrose pseudo-inverse using native backend operation.

    Uses native b.pinv() which computes the exact pseudo-inverse via SVD.
    No regularization, no approximation - correctness over efficiency.

    CKA=1.0 requires exact pseudo-inverse. Any regularization or approximation
    introduces error that prevents perfect kernel alignment.
    """
    b = backend
    F = b.astype(b.array(F), "float32")
    b.eval(F)

    # EXACT Moore-Penrose pseudo-inverse - no approximation
    F_pinv = b.pinv(F)
    b.eval(F_pinv)

    return F_pinv


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
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class TransplantStageResult:
    """Result of transplant stage."""

    merged_weights: dict[str, Any]
    metrics: dict[str, Any]
    manifest: TransplantManifest | None = None  # Track every weight's status


def _normalize_domains(domains: Iterable[str]) -> set[str]:
    return {d.strip().lower() for d in domains if d.strip()}


def _save_checkpoint(
    checkpoint_dir: Path,
    layer_idx: int,
    merged_weights: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save transplant progress checkpoint for resume capability."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata (small JSON file with state)
    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    meta = {
        "last_completed_layer": layer_idx,
        "timestamp": time.time(),
        "weights_transplanted": metrics.get("weights_transplanted", 0),
        "layers_transplanted": metrics.get("layers_transplanted", 0),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("CHECKPOINT: Saved progress at layer %d to %s", layer_idx, checkpoint_dir)


def _load_checkpoint(checkpoint_dir: Path) -> tuple[int, dict[str, Any]] | None:
    """Load transplant checkpoint if available.

    Returns:
        Tuple of (last_completed_layer, metadata) or None if no checkpoint exists.
    """
    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        last_layer = meta.get("last_completed_layer", -1)
        logger.info(
            "CHECKPOINT: Resuming from layer %d (weights=%d, layers=%d)",
            last_layer,
            meta.get("weights_transplanted", 0),
            meta.get("layers_transplanted", 0),
        )
        return last_layer, meta
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("CHECKPOINT: Failed to load checkpoint: %s", e)
        return None


def _compute_alignment_metrics(
    core_acts: "Array",
    weight_before: "Array",
    weight_after: "Array",
    weight_source: "Array",
    backend: "Backend",
) -> dict[str, float]:
    """Measure core distance shift toward the source for a single weight.

    Metrics are only defined when activation and weight input dimensions match.
    """
    from modelcypher.core.domain.geometry.cka import compute_cka

    b = backend

    if int(core_acts.shape[1]) != int(weight_before.shape[1]):
        raise DimensionMismatchError(
            f"Alignment metrics require matching input dims; "
            f"acts={int(core_acts.shape[1])}, weight_in={int(weight_before.shape[1])}"
        )

    output_before = b.matmul(core_acts, b.transpose(weight_before))
    output_after = b.matmul(core_acts, b.transpose(weight_after))
    output_source = b.matmul(core_acts, b.transpose(weight_source))
    b.eval(output_before, output_after, output_source)

        # Geodesic distance respects manifold curvature. Chord distance systematically errs.
    # Aggregate per-sample geodesic distances using geodesic norms.
    geo_distances_before = geodesic_paired_distances(output_before, output_source, b)
    geo_distances_after = geodesic_paired_distances(output_after, output_source, b)
    dist_before_arr = geodesic_norms(b.reshape(geo_distances_before, (1, -1)), b)
    dist_after_arr = geodesic_norms(b.reshape(geo_distances_after, (1, -1)), b)
    b.eval(dist_before_arr, dist_after_arr)

    dist_before = float(b.to_scalar(dist_before_arr))
    dist_after = float(b.to_scalar(dist_after_arr))

    eps = float(machine_epsilon(b, weight_before))
    if dist_before > eps:
        core_distance_reduction = (dist_before - dist_after) / dist_before
    else:
        core_distance_reduction = 0.0

    cka_before = compute_cka(output_before, output_source, backend=b)
    cka_after = compute_cka(output_after, output_source, backend=b)

    return {
        "core_dist_to_source_before": dist_before,
        "core_dist_to_source_after": dist_after,
        "core_distance_reduction": core_distance_reduction,
        "cka_before": cka_before.best,
        "cka_after": cka_after.best,
    }


def _set_submatrix(
    backend: "Backend",
    target: "Array",
    source: "Array",
    row_offset: int,
    col_offset: int,
) -> "Array":
    """Set a submatrix of target from source at the given offset.

    Since MLX doesn't support in-place slice assignment, we construct the
    result using concatenation or element-wise operations.
    """
    src_rows, src_cols = int(source.shape[0]), int(source.shape[1])
    tgt_rows, tgt_cols = int(target.shape[0]), int(target.shape[1])

    if src_rows == 0 or src_cols == 0:
        return target

    row_end = row_offset + src_rows
    col_end = col_offset + src_cols

    # Build middle block with column concatenation
    mid_parts = []
    if col_offset > 0:
        mid_parts.append(target[row_offset:row_end, :col_offset])
    mid_parts.append(source)
    if col_end < tgt_cols:
        mid_parts.append(target[row_offset:row_end, col_end:])

    mid = mid_parts[0] if len(mid_parts) == 1 else backend.concatenate(mid_parts, axis=1)

    # Stitch rows via concatenation
    row_parts = []
    if row_offset > 0:
        row_parts.append(target[:row_offset, :])
    row_parts.append(mid)
    if row_end < tgt_rows:
        row_parts.append(target[row_end:, :])

    result = row_parts[0] if len(row_parts) == 1 else backend.concatenate(row_parts, axis=0)
    backend.eval(result)
    return result


def _compute_dimension_projection(
    backend: "Backend",
    src_dim: int,
    tgt_dim: int,
) -> "Array":
    """Compute an orthogonal projection matrix between dimensions.

    For src_dim → tgt_dim:
    - If tgt_dim < src_dim: truncation (keep first tgt_dim dimensions)
    - If tgt_dim > src_dim: padding (embed in larger space)
    - If equal: identity

    Uses identity-based projection to preserve geometric structure.
    """
    if src_dim == tgt_dim:
        return backend.eye(src_dim, dtype="float32")

    min_dim = min(src_dim, tgt_dim)

    # Create identity block of size min_dim x min_dim
    identity_block = backend.eye(min_dim, dtype="float32")

    if tgt_dim < src_dim:
        # Truncation: [src_dim, tgt_dim]
        # Stack: identity_block on top, zeros below
        zeros_below = backend.zeros((src_dim - min_dim, tgt_dim), dtype="float32")
        projection = backend.concatenate([identity_block, zeros_below], axis=0)
    else:
        # Padding: [src_dim, tgt_dim]
        # Stack: identity_block on left, zeros on right
        zeros_right = backend.zeros((src_dim, tgt_dim - min_dim), dtype="float32")
        projection = backend.concatenate([identity_block, zeros_right], axis=1)

    backend.eval(projection)
    return projection


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
    transplant_domains: tuple[str, ...] = (),
    graft_mask: dict[str, dict[int, bool]] | None = None,
    feature_transforms: dict[int, list[list[float]]] | None = None,
    attention_transforms: dict[int, list[list[float]]] | None = None,
    kv_transforms: dict[int, list[list[float]]] | None = None,
    layer_mapping: dict[int, int] | None = None,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    backend: "Backend | None" = None,
) -> TransplantStageResult:
    """Stage 3: Null-space constrained transplant using probe activations."""
    b = backend or get_default_backend()
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

    if graft_mask is None:
        raise RuntimeError("Transplant requires graft_mask from density stage")

    if len(probe_ids) != len(probe_domains):
        metrics["transplant_skipped"] = "probe_metadata_mismatch"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    core_domains = _normalize_domains(transplant_domains)

    if core_domains:
        # Legacy: domain-based filtering
        core_probe_ids = {
            probe_id
            for probe_id, domain in zip(probe_ids, probe_domains)
            if domain and domain.lower() in core_domains
        }
        logger.info(
            "TRANSPLANT: Domain-based mode - %d core probes from domains %s",
            len(core_probe_ids), list(core_domains)
        )
    else:
        # Density-only: ALL probes are candidates, graft_mask decides
        core_probe_ids = set(probe_ids)
        logger.info(
            "TRANSPLANT: Density-only mode - %d candidate probes (geometry decides)",
            len(core_probe_ids)
        )

    metrics["core_probes"] = len(core_probe_ids)
    metrics["density_only_mode"] = not bool(core_domains)

    if not core_probe_ids:
        metrics["transplant_skipped"] = "no_core_probes"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

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
    # PER-LAYER ALIGNMENT: Use transforms from probe stage (CKA=1.0 verified)
    # ==========================================================================
    # The probe stage already found the transforms that achieve CKA=1.0 for each
    # aligned layer pair. We use those directly instead of recomputing.
    # Each layer gets its own transform - different layers encode different parts
    # of the geometry at different resolutions.
    layer_hidden_stitches: dict[int, tuple[Any, Any]] = {}  # layer -> (stitch_out, stitch_in)

    if feature_transforms and layer_mapping:
        # Use pre-computed per-layer transforms from probe stage
        logger.info(
            "PER-LAYER ALIGNMENT: Using %d transforms from probe stage (CKA=1.0 verified)",
            len(feature_transforms),
        )

        # Convert all transforms to backend arrays in one batch for MLX efficiency
        transforms_to_eval = []
        for tgt_layer, transform_list in feature_transforms.items():
            F = b.array(transform_list)
            F = b.astype(F, "float32")
            transforms_to_eval.append((tgt_layer, F))

        # Batch eval for MLX efficiency
        if transforms_to_eval:
            b.eval(*[t[1] for t in transforms_to_eval])

        # Compute stitch matrices for each layer
        for tgt_layer, F in transforms_to_eval:
            stitch_output = b.transpose(F)  # F.T [tgt_dim, src_dim]
            F_pinv = _geodesic_pinv(b, F)
            stitch_input = b.transpose(F_pinv)  # pinv(F).T [src_dim, tgt_dim]
            layer_hidden_stitches[tgt_layer] = (stitch_output, stitch_input)

        # Batch eval all stitch matrices for MLX efficiency
        all_stitches = []
        for stitch_out, stitch_in in layer_hidden_stitches.values():
            all_stitches.extend([stitch_out, stitch_in])
        if all_stitches:
            b.eval(*all_stitches)

        metrics["per_layer_alignment"] = True
        metrics["layers_with_transforms"] = len(layer_hidden_stitches)

    elif source_activations:
        # Fallback: same-architecture case where dims match
        # Get dimensions from first available layer
        first_src_list = next(iter(source_activations.values()), [])
        first_tgt_list = next(iter(target_activations.values()), [])

        if first_src_list and first_tgt_list:
            src_dim = int(b.array(first_src_list[0]).shape[0])
            tgt_dim = int(b.array(first_tgt_list[0]).shape[0])

            if src_dim == tgt_dim:
                # Same dims: identity transform for all layers
                identity = b.eye(src_dim)
                b.eval(identity)
                for layer_idx in layer_indices:
                    layer_hidden_stitches[layer_idx] = (identity, identity)
                logger.info(
                    "SAME-DIM ALIGNMENT: Using identity for %d layers (dim=%d)",
                    len(layer_indices),
                    src_dim,
                )
                metrics["per_layer_alignment"] = True
                metrics["layers_with_transforms"] = len(layer_indices)
            else:
                raise AlignmentFailureError(
                    stage="PER_LAYER_ALIGNMENT",
                    weight_key=None,
                    message=(
                        f"Dimension mismatch ({src_dim} vs {tgt_dim}) but no "
                        "feature_transforms from probe stage. Run probe with "
                        "GramAligner to get CKA=1.0 transforms."
                    ),
                    context={
                        "source_dim": src_dim,
                        "target_dim": tgt_dim,
                    },
                )
        else:
            raise AlignmentFailureError(
                stage="PER_LAYER_ALIGNMENT",
                weight_key=None,
                message="No activations available for alignment",
                context={},
            )
    else:
        raise AlignmentFailureError(
            stage="PER_LAYER_ALIGNMENT",
            weight_key=None,
            message="Source activations are required for alignment",
            context={},
        )

    # ==========================================================================
    # PER-LAYER ATTENTION Q ALIGNMENT: Use transforms from probe stage
    # ==========================================================================
    # Attention weights have a different dimension than hidden:
    #   - q_proj: [num_heads * head_dim, hidden_dim] (e.g., [960, 960] for SmolLM)
    #   - o_proj: [hidden_dim, num_heads * head_dim]
    # When head counts differ (e.g., SmolLM=15 heads → Qwen=14 heads):
    #   - SmolLM attention dim = 15 * 64 = 960
    #   - Qwen attention dim = 14 * 64 = 896
    layer_attention_stitches: dict[int, tuple[Any, Any]] = {}  # layer -> (stitch_out, stitch_in)

    if attention_transforms and layer_mapping:
        logger.info(
            "PER-LAYER ATTENTION Q: Using %d transforms from probe stage (CKA=1.0 verified)",
            len(attention_transforms),
        )

        # Convert all transforms to backend arrays in one batch for MLX efficiency
        attn_transforms_to_eval = []
        for tgt_layer, transform_list in attention_transforms.items():
            F = b.array(transform_list)
            F = b.astype(F, "float32")
            attn_transforms_to_eval.append((tgt_layer, F))

        # Batch eval for MLX efficiency
        if attn_transforms_to_eval:
            b.eval(*[t[1] for t in attn_transforms_to_eval])

        # Compute stitch matrices for each layer
        for tgt_layer, F in attn_transforms_to_eval:
            stitch_output = b.transpose(F)  # F.T [tgt_dim, src_dim]
            F_pinv = _geodesic_pinv(b, F)
            stitch_input = b.transpose(F_pinv)  # pinv(F).T [src_dim, tgt_dim]
            layer_attention_stitches[tgt_layer] = (stitch_output, stitch_input)

        # Batch eval all stitch matrices for MLX efficiency
        all_attn_stitches = []
        for stitch_out, stitch_in in layer_attention_stitches.values():
            all_attn_stitches.extend([stitch_out, stitch_in])
        if all_attn_stitches:
            b.eval(*all_attn_stitches)

        metrics["per_layer_attention_alignment"] = True
        metrics["layers_with_attention_transforms"] = len(layer_attention_stitches)

    elif source_attention_activations and target_attention_activations:
        # Same-architecture case: check if identity transform works
        first_src_attn = next(iter(source_attention_activations.values()), [])
        first_tgt_attn = next(iter(target_attention_activations.values()), [])

        if first_src_attn and first_tgt_attn:
            src_attn_dim = int(b.array(first_src_attn[0]).shape[0])
            tgt_attn_dim = int(b.array(first_tgt_attn[0]).shape[0])

            if src_attn_dim == tgt_attn_dim:
                identity = b.eye(src_attn_dim)
                b.eval(identity)
                for layer_idx in layer_indices:
                    layer_attention_stitches[layer_idx] = (identity, identity)
                logger.info(
                    "SAME-DIM ATTENTION Q: Using identity for %d layers (dim=%d)",
                    len(layer_indices),
                    src_attn_dim,
                )
                metrics["per_layer_attention_alignment"] = True
                metrics["layers_with_attention_transforms"] = len(layer_indices)
            else:
                raise AlignmentFailureError(
                    stage="PER_LAYER_ATTENTION_ALIGNMENT",
                    weight_key=None,
                    message=(
                        f"Attention Q dimension mismatch ({src_attn_dim} vs {tgt_attn_dim}) "
                        "but no attention_transforms from probe stage."
                    ),
                    context={"source_dim": src_attn_dim, "target_dim": tgt_attn_dim},
                )

    # ==========================================================================
    # PER-LAYER KV ALIGNMENT: Use transforms from probe stage for GQA models
    # ==========================================================================
    # GQA (Grouped Query Attention) models have different head counts for Q vs K/V:
    #   - SmolLM: Q = 15 heads × 64 = 960, KV = 5 heads × 64 = 320
    #   - Qwen: Q = 14 heads × 64 = 896, KV = 2 heads × 64 = 128
    #
    # k_proj and v_proj weights have shape [kv_attention_dim, hidden_dim], NOT
    # [q_attention_dim, hidden_dim]. We MUST compute a separate stitch for KV.
    layer_kv_stitches: dict[int, tuple[Any, Any]] = {}  # layer -> (stitch_out, stitch_in)

    if kv_transforms and layer_mapping:
        logger.info(
            "PER-LAYER KV: Using %d transforms from probe stage (CKA=1.0 verified)",
            len(kv_transforms),
        )

        # Convert all transforms to backend arrays in one batch for MLX efficiency
        kv_transforms_to_eval = []
        for tgt_layer, transform_list in kv_transforms.items():
            F = b.array(transform_list)
            F = b.astype(F, "float32")
            kv_transforms_to_eval.append((tgt_layer, F))

        # Batch eval for MLX efficiency
        if kv_transforms_to_eval:
            b.eval(*[t[1] for t in kv_transforms_to_eval])

        # Compute stitch matrices for each layer
        for tgt_layer, F in kv_transforms_to_eval:
            stitch_output = b.transpose(F)  # F.T [tgt_dim, src_dim]
            F_pinv = _geodesic_pinv(b, F)
            stitch_input = b.transpose(F_pinv)  # pinv(F).T [src_dim, tgt_dim]
            layer_kv_stitches[tgt_layer] = (stitch_output, stitch_input)

        # Batch eval all stitch matrices for MLX efficiency
        all_kv_stitches = []
        for stitch_out, stitch_in in layer_kv_stitches.values():
            all_kv_stitches.extend([stitch_out, stitch_in])
        if all_kv_stitches:
            b.eval(*all_kv_stitches)

        metrics["per_layer_kv_alignment"] = True
        metrics["layers_with_kv_transforms"] = len(layer_kv_stitches)

    elif source_kv_activations and target_kv_activations:
        # Same-architecture case: check if identity transform works
        first_src_kv = next(iter(source_kv_activations.values()), [])
        first_tgt_kv = next(iter(target_kv_activations.values()), [])

        if first_src_kv and first_tgt_kv:
            src_kv_dim = int(b.array(first_src_kv[0]).shape[0])
            tgt_kv_dim = int(b.array(first_tgt_kv[0]).shape[0])

            if src_kv_dim == tgt_kv_dim:
                identity = b.eye(src_kv_dim)
                b.eval(identity)
                for layer_idx in layer_indices:
                    layer_kv_stitches[layer_idx] = (identity, identity)
                logger.info(
                    "SAME-DIM KV: Using identity for %d layers (dim=%d)",
                    len(layer_indices),
                    src_kv_dim,
                )
                metrics["per_layer_kv_alignment"] = True
                metrics["layers_with_kv_transforms"] = len(layer_indices)
            else:
                raise AlignmentFailureError(
                    stage="PER_LAYER_KV_ALIGNMENT",
                    weight_key=None,
                    message=(
                        f"KV dimension mismatch ({src_kv_dim} vs {tgt_kv_dim}) "
                        "but no kv_transforms from probe stage."
                    ),
                    context={"source_dim": src_kv_dim, "target_dim": tgt_kv_dim},
                )

    for layer_num, layer_idx in enumerate(layer_indices):
        # Skip layers already completed (checkpoint resume)
        if layer_idx <= resume_from_layer:
            weights_processed += len(weights_by_layer.get(layer_idx, []))
            logger.debug("TRANSPLANT: Skipping layer %d (already completed)", layer_idx)
            continue

        # NOTE: transplant_layers filter was REMOVED. Always transplant all layers.
        # The geometry determines which weights need transplanting, not arbitrary layer selection.

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
        if not act_list:
            raise AlignmentFailureError(
                stage="LAYER_ACTIVATION_VALIDATION",
                weight_key=None,
                message="Missing target activations for layer",
                context={"layer_idx": layer_idx},
            )

        if len(act_list) != len(probe_ids):
            raise AlignmentFailureError(
                stage="LAYER_ACTIVATION_VALIDATION",
                weight_key=None,
                message="Probe count mismatch for layer activations",
                context={
                    "layer_idx": layer_idx,
                    "activations": len(act_list),
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
        attention_stitch_output = None  # F.T for Q attention output [tgt_attn, src_attn]
        attention_stitch_input = None   # pinv(F).T for Q attention input [src_attn, tgt_attn]
        kv_stitch_output = None  # F.T for KV attention output [tgt_kv, src_kv] (GQA)
        kv_stitch_input = None   # pinv(F).T for KV attention input [src_kv, tgt_kv] (GQA)

        # Get intermediate activations for this layer (MLP internal states)
        # CRITICAL: For cross-architecture merge, source layer index may differ from target.
        # IMPORTANT: The hidden-space layer_mapping is optimized for hidden activations, but
        # intermediate space has different semantic structure. ALWAYS use proportional mapping
        # for intermediate activations to ensure we compare semantically similar layers.
        if source_intermediate_activations:
            # Proportional mapping: target_i * (source_layers / target_layers)
            n_source = max(source_intermediate_activations.keys()) + 1
            n_target = len(layer_indices)
            source_layer_idx = int(round(layer_idx * n_source / n_target))
            source_layer_idx = min(source_layer_idx, n_source - 1)  # Clamp to valid range
        else:
            source_layer_idx = layer_idx
        
        # DEBUG: Log which source layer is being used for this target layer
        if layer_num < 3 or layer_idx >= 15:  # Log first 3 and layers 15+
            logger.info(
                "Layer %d: Intermediate alignment using source layer %d (proportional mapping n_src=%d, n_tgt=%d)",
                layer_idx, source_layer_idx, n_source if source_intermediate_activations else 0, len(layer_indices)
            )
        
        src_inter_list = (
            source_intermediate_activations.get(source_layer_idx)
            if source_intermediate_activations else None
        )
        tgt_inter_list = (
            target_intermediate_activations.get(layer_idx)
            if target_intermediate_activations else None
        )

        # =================================================================
        # GRAM ALIGNMENT: Find EXACT CKA = 1.0 transforms for hidden AND intermediate
        # =================================================================
        # GramAligner finds the mathematically guaranteed transformation that achieves
        # CKA = 1.0. This is not an approximation - it's the exact solution.
        #
        # The feature_transform maps source activations to target space such that
        # their Gram matrices (relational geometry) are IDENTICAL.
        #
        # MLP weights have shape [intermediate, hidden] or [hidden, intermediate].
        # We need transforms for BOTH axes to properly map weights.

        from modelcypher.core.domain.geometry.gram_aligner import GramAligner

        # USE PER-LAYER ALIGNED hidden stitch (from probe stage with CKA=1.0)
        # Each layer gets its own transform - different layers encode different
        # parts of the geometry at different resolutions.
        hidden_stitch_output = None
        hidden_stitch_input = None

        if layer_idx in layer_hidden_stitches:
            hidden_stitch_output, hidden_stitch_input = layer_hidden_stitches[layer_idx]
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

        # Compute INTERMEDIATE alignment (source intermediate_dim → target intermediate_dim)
        # Note: Intermediate stitch is per-layer since each MLP has different internal geometry
        if src_inter_list is not None and tgt_inter_list is not None:
            n_inter_samples = min(len(src_inter_list), len(tgt_inter_list))
            if n_inter_samples < 2:
                raise AlignmentFailureError(
                    stage="INTERMEDIATE_ALIGNMENT",
                    weight_key=None,
                    message="Insufficient activations for intermediate alignment",
                    context={
                        "samples_used": n_inter_samples,
                        "required_min_samples": 2,
                        "layer_idx": layer_idx,
                    },
                )

            src_inter = b.stack(src_inter_list[:n_inter_samples], axis=0)
            tgt_inter = b.stack(tgt_inter_list[:n_inter_samples], axis=0)
            src_inter = b.astype(src_inter, "float32")
            tgt_inter = b.astype(tgt_inter, "float32")
            b.eval(src_inter, tgt_inter)

            src_inter_dim = int(src_inter.shape[1])
            tgt_inter_dim = int(tgt_inter.shape[1])

            if src_inter_dim != tgt_inter_dim:
                # Use GramAligner - finds EXACT CKA = 1.0 transform
                aligner = GramAligner(b)
                inter_result = aligner.find_perfect_alignment(src_inter, tgt_inter)

                # With the corrected GramAligner that computes CKA on sample-space
                # Gram-aligned source (T @ source), CKA = 1.0 is mathematically guaranteed
                # for all dimension ratios. The Gram sqrt transform operates in sample space.
                # Use 0.99 threshold for floating-point safety (numerical CKA may be 0.9999999).
                dim_ratio = max(src_inter_dim, tgt_inter_dim) / min(src_inter_dim, tgt_inter_dim)
                cka_threshold = 0.99
                is_acceptable = inter_result.achieved_cka >= cka_threshold

                if is_acceptable:
                    # feature_transform F is [d_source, d_target]
                    # source @ F → target (activation alignment)
                    #
                    # For weight folding W_target = F_out.T @ W_source @ pinv(F_in).T:
                    #   - F.T [d_target, d_source] for OUTPUT side (left multiply)
                    #   - pinv(F).T [d_source, d_target] for INPUT side (right multiply)
                    F = b.array(inter_result.feature_transform)
                    b.eval(F)
                    intermediate_stitch_output = b.transpose(F)  # F.T [tgt, src]
                    F_pinv = _geodesic_pinv(b, F)
                    intermediate_stitch_input = b.transpose(F_pinv)  # pinv(F).T [src, tgt]
                    b.eval(intermediate_stitch_output, intermediate_stitch_input)
                    logger.info(
                        "Layer %d: Intermediate GramAlign CKA=%.4f (%d→%d)",
                        layer_idx, inter_result.achieved_cka,
                        src_inter_dim, tgt_inter_dim,
                    )
                    metrics.setdefault("intermediate_gram_aligned", 0)
                    metrics["intermediate_gram_aligned"] += 1
                else:
                    raise AlignmentFailureError(
                        stage="INTERMEDIATE_ALIGNMENT",
                        weight_key=None,
                        message=f"GramAligner failed to achieve CKA>={cka_threshold:.2f} (got {inter_result.achieved_cka:.4f})",
                        context={
                            "achieved_cka": float(inter_result.achieved_cka),
                            "cka_threshold": cka_threshold,
                            "source_dim": src_inter_dim,
                            "target_dim": tgt_inter_dim,
                            "dim_ratio": dim_ratio,
                        },
                    )
            else:
                intermediate_stitch_output = b.eye(src_inter_dim)
                intermediate_stitch_input = b.eye(src_inter_dim)
                b.eval(intermediate_stitch_output, intermediate_stitch_input)

        # USE PER-LAYER ATTENTION stitch (from probe stage with CKA=1.0)
        if layer_idx in layer_attention_stitches:
            attention_stitch_output, attention_stitch_input = layer_attention_stitches[layer_idx]
            if layer_num == 0:
                stitch_shape = attention_stitch_output.shape if attention_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer Q attention stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        # USE PER-LAYER KV stitch for GQA models (from probe stage with CKA=1.0)
        if layer_idx in layer_kv_stitches:
            kv_stitch_output, kv_stitch_input = layer_kv_stitches[layer_idx]
            if layer_num == 0:
                stitch_shape = kv_stitch_output.shape if kv_stitch_output is not None else "N/A"
                logger.info(
                    "Layer %d: Using per-layer KV stitch (shape=%s)",
                    layer_idx, stitch_shape
                )

        stacked = b.stack(act_list, axis=0)
        # Convert to float32 for numerical stability in linalg operations.
        stacked = b.astype(stacked, "float32")
        b.eval(stacked)

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
            continue

        core_indices = b.array(partition.core_indices, dtype="int32")
        core_acts = b.take(stacked, core_indices, axis=0)
        b.eval(core_acts)

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

        for weight_num, key in enumerate(layer_keys):
            weights_processed += 1
            weight_start_time = time.time()

            # Skip quantization metadata and non-weight tensors (only transplant actual matrices).
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            if not key.endswith(".weight"):
                continue

            # Progress callback for external monitoring
            if progress_callback:
                progress_callback(
                    f"Layer {layer_idx}: {key}",
                    weights_processed,
                    total_weights,
                )

            target_w = target_weights.get(key)
            source_w = source_weights.get(key)
            if target_w is None or source_w is None:
                continue

            metrics["weights_considered"] += 1

            # Skip non-2D weights (bias vectors, etc)
            if not hasattr(target_w, "shape") or not hasattr(source_w, "shape"):
                continue
            if len(target_w.shape) != 2 or len(source_w.shape) != 2:
                continue

            # Dequantize quantized weights (uint32/int dtypes) using scales/biases
            target_dtype = str(getattr(target_w, 'dtype', '')).lower()
            source_dtype = str(getattr(source_w, 'dtype', '')).lower()

            if 'int' in target_dtype or 'uint' in target_dtype:
                logger.debug("Dequantizing target weight: %s", key)
                target_w = dequantize_if_needed(target_w, key, target_weights, b)
                if target_w is None or not hasattr(target_w, 'shape'):
                    logger.debug("Failed to dequantize target weight: %s", key)
                    continue
                # Check if still quantized after dequantize attempt
                target_dtype = str(getattr(target_w, 'dtype', '')).lower()
                if 'int' in target_dtype or 'uint' in target_dtype:
                    logger.debug("Skipping still-quantized target weight: %s", key)
                    continue

            if 'int' in source_dtype or 'uint' in source_dtype:
                logger.debug("Dequantizing source weight: %s", key)
                source_w = dequantize_if_needed(source_w, key, source_weights, b)
                if source_w is None or not hasattr(source_w, 'shape'):
                    logger.debug("Failed to dequantize source weight: %s", key)
                    continue
                # Check if still quantized after dequantize attempt
                source_dtype = str(getattr(source_w, 'dtype', '')).lower()
                if 'int' in source_dtype or 'uint' in source_dtype:
                    logger.debug("Skipping still-quantized source weight: %s", key)
                    continue

            # Skip if shapes became non-2D after dequantization
            if len(target_w.shape) != 2 or len(source_w.shape) != 2:
                continue

            try:
                # Convert to float32 backend arrays
                logger.debug("Converting weight %s to float32", key)
                target_w = b.astype(b.array(target_w), "float32")
                source_w = b.astype(b.array(source_w), "float32")
                b.eval(target_w, source_w)
                logger.debug("Converted %s: target=%s, source=%s", key, target_w.shape, source_w.shape)
            except Exception as e:
                logger.warning("Failed to convert weight %s to float32: %s", key, e)
                continue

            source_candidate = source_w
            if target_w.shape != source_candidate.shape:
                # =================================================================
                # MULTI-SPACE STITCHING: Apply pre-computed stitches to weights
                # =================================================================
                # MLP weights need BOTH hidden AND intermediate stitches:
                #   gate_proj [intermediate, hidden] → [tgt_intermediate, tgt_hidden]
                #   up_proj   [intermediate, hidden] → [tgt_intermediate, tgt_hidden]
                #   down_proj [hidden, intermediate] → [tgt_hidden, tgt_intermediate]
                #
                # Attention weights only need hidden stitch:
                #   q_proj, k_proj, v_proj [hidden, head*num_heads] → [tgt_hidden, ...]
                #   o_proj [head*num_heads, hidden] → [..., tgt_hidden]

                # Use ORIGINAL source shape for dimension matching.
                original_source_shape = source_w.shape
                is_mlp = any(mlp_name in key for mlp_name in [
                    "gate_proj", "up_proj", "down_proj", "mlp.fc1", "mlp.fc2"
                ])

                # =================================================================
                # MIRROR PROBLEM FIX: Correct weight folding transforms
                # =================================================================
                # GramAligner finds F such that: source @ F → target (activation alignment)
                #
                # For WEIGHT folding, we need the inverse on the INPUT side:
                #   W_target = F_out.T @ W_source @ pinv(F_in).T
                #
                # Where:
                #   - F_out.T [tgt_out, src_out] = stitch_output (left multiply for output dim)
                #   - pinv(F_in).T [src_in, tgt_in] = stitch_input (right multiply for input dim)
                #
                # Weight shapes: [output_features, input_features]
                #   - gate_proj/up_proj: [intermediate, hidden] (out=inter, in=hidden)
                #   - down_proj:         [hidden, intermediate] (out=hidden, in=inter)
                #   - attention:         [hidden, hidden] (both dims hidden)

                if is_mlp and hidden_stitch_output is not None and intermediate_stitch_output is not None:
                    # MLP weight: apply BOTH stitches with correct orientations
                    # stitch_output for OUTPUT side (rows), stitch_input for INPUT side (cols)
                    src_hidden_dim = int(hidden_stitch_output.shape[1])  # F.T is [tgt, src]
                    tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                    src_inter_dim = int(intermediate_stitch_output.shape[1])
                    tgt_inter_dim = int(intermediate_stitch_output.shape[0])

                    logger.info(
                        "MLP weight %s: applying dual stitch (hidden %d→%d, inter %d→%d)",
                        key, src_hidden_dim, tgt_hidden_dim, src_inter_dim, tgt_inter_dim
                    )

                    # Determine which dimension is which from ORIGINAL source shape
                    dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                    if dim0 == src_inter_dim and dim1 == src_hidden_dim:
                        # gate_proj/up_proj: [intermediate, hidden] → [tgt_inter, tgt_hidden]
                        # Output=intermediate (rows), Input=hidden (cols)
                        # W_target = inter_stitch_output @ W @ hidden_stitch_input
                        source_aligned = b.matmul(intermediate_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info("Dual stitch (gate/up): [%d,%d] → [%d,%d]",
                                    dim0, dim1, tgt_inter_dim, tgt_hidden_dim)

                    elif dim0 == src_hidden_dim and dim1 == src_inter_dim:
                        # down_proj: [hidden, intermediate] → [tgt_hidden, tgt_inter]
                        # Output=hidden (rows), Input=intermediate (cols)
                        # W_target = hidden_stitch_output @ W @ inter_stitch_input
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, intermediate_stitch_input)
                        b.eval(source_aligned)
                        logger.info("Dual stitch (down): [%d,%d] → [%d,%d]",
                                    dim0, dim1, tgt_hidden_dim, tgt_inter_dim)

                    else:
                        logger.warning(
                            "MLP weight %s shape [%d,%d] doesn't match expected dims "
                            "(hidden=%d, inter=%d) - skipping",
                            key, dim0, dim1, src_hidden_dim, src_inter_dim
                        )
                        continue

                    metrics.setdefault("dual_stitch_applied", 0)
                    metrics["dual_stitch_applied"] += 1

                elif hidden_stitch_output is not None and hidden_stitch_input is not None:
                    # =================================================================
                    # ATTENTION WEIGHT STITCHING (replaces previous skip logic)
                    # =================================================================
                    # A structural incompatibility IS a geometric one. Head count difference
                    # is just another dimension mismatch that GramAligner can solve.
                    #
                    # Example: SmolLM (15 heads) → Qwen (14 heads)
                    #   - q_proj [960, 960] → [896, 896]
                    #   - 960 = 15 * 64 (SmolLM attention dim)
                    #   - 896 = 14 * 64 (Qwen attention dim)
                    #   - GramAligner finds F such that CKA(src_attn @ F, tgt_attn) = 1.0
                    #
                    # Weight stitching for attention:
                    #   - q_proj/k_proj/v_proj: [attn_dim, hidden_dim]
                    #     → attention_stitch_output @ W @ hidden_stitch_input
                    #   - o_proj: [hidden_dim, attn_dim]
                    #     → hidden_stitch_output @ W @ attention_stitch_input
                    is_attention = any(attn_name in key for attn_name in [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "self_attn", "query", "key", "value",
                    ])

                    attention_stitch_applied = False

                    if is_attention and attention_stitch_output is not None:
                        # We have attention stitch - apply it!
                        src_attn_dim = int(attention_stitch_output.shape[1])
                        tgt_attn_dim = int(attention_stitch_output.shape[0])
                        src_hidden_dim = int(hidden_stitch_output.shape[1])
                        tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                        dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                        # Get KV stitch dimensions for GQA detection
                        # GQA models: K/V have fewer heads than Q (e.g., SmolLM: Q=960, KV=320)
                        src_kv_dim = int(kv_stitch_output.shape[1]) if kv_stitch_output is not None else src_attn_dim
                        tgt_kv_dim = int(kv_stitch_output.shape[0]) if kv_stitch_output is not None else tgt_attn_dim

                        # Determine attention weight pattern
                        # GQA: q_proj uses Q-attention dim, k_proj/v_proj use KV-attention dim
                        is_q = any(n in key for n in ["q_proj", "query"])
                        is_kv = any(n in key for n in ["k_proj", "v_proj", "key", "value"])
                        is_o = any(n in key for n in ["o_proj"])

                        if is_q and dim0 == src_attn_dim and dim1 == src_hidden_dim:
                            # q_proj: [Q_attn, hidden] → attention_stitch @ W @ hidden_stitch
                            source_aligned = b.matmul(attention_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info(
                                "Attention stitch (q_proj): %s [%d,%d] → [%d,%d]",
                                key, dim0, dim1, tgt_attn_dim, tgt_hidden_dim
                            )
                            metrics.setdefault("attention_stitched", 0)
                            metrics["attention_stitched"] += 1
                            attention_stitch_applied = True

                        elif is_kv and dim0 == src_kv_dim and dim1 == src_hidden_dim:
                            # k_proj/v_proj (GQA): [KV_attn, hidden] → kv_stitch @ W @ hidden_stitch
                            kv_out = kv_stitch_output if kv_stitch_output is not None else attention_stitch_output
                            source_aligned = b.matmul(kv_out, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            stitch_type = "KV stitch" if kv_stitch_output is not None else "attention stitch"
                            logger.info(
                                "%s (k/v_proj): %s [%d,%d] → [%d,%d]",
                                stitch_type, key, dim0, dim1, tgt_kv_dim, tgt_hidden_dim
                            )
                            metrics.setdefault("kv_stitched", 0)
                            metrics["kv_stitched"] += 1
                            attention_stitch_applied = True

                        elif is_o and dim0 == src_hidden_dim and dim1 == src_attn_dim:
                            # o_proj: [hidden, attn] → hidden_stitch_output @ W @ attention_stitch_input
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, attention_stitch_input)
                            b.eval(source_aligned)
                            logger.info(
                                "Attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                key, dim0, dim1, tgt_hidden_dim, tgt_attn_dim
                            )
                            metrics.setdefault("attention_stitched", 0)
                            metrics["attention_stitched"] += 1
                            attention_stitch_applied = True

                        elif is_o and dim0 == src_hidden_dim and dim1 != src_attn_dim:
                            # Hybrid attention architecture: o_proj input dim differs from Q output dim
                            # This happens in models like Qwen3-Next with mixed regular/linear attention
                            # where o_proj only receives regular attention output
                            #
                            # We compute a stitch based on the actual dimensions:
                            # - hidden_stitch_output: [tgt_hidden, src_hidden] - for rows
                            # - attention dimension needs separate handling
                            target_o_dim1 = int(target_w.shape[1])  # Target's o_proj input dim

                            logger.info(
                                "Hybrid attention detected for %s: o_proj dim1=%d != Q_attn_dim=%d. "
                                "Computing adaptive stitch → target dim=%d",
                                key, dim1, src_attn_dim, target_o_dim1
                            )

                            # Apply hidden stitch to rows first
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            # source_aligned now: [tgt_hidden_dim, dim1]

                            # For columns: compute transformation from dim1 → target_o_dim1
                            # Strategy: If dim1 is a subset of src_attn_dim (e.g., regular heads only),
                            # use the corresponding subset of attention_stitch
                            if dim1 < src_attn_dim and target_o_dim1 <= tgt_attn_dim:
                                # Hybrid attention: o_proj uses subset of attention dimensions
                                # Take first dim1 rows and first target_o_dim1 columns of attention_stitch
                                # attention_stitch_input: [src_attn_dim, tgt_attn_dim]
                                # We need: [dim1, target_o_dim1]
                                partial_stitch = attention_stitch_input[:dim1, :target_o_dim1]
                                source_aligned = b.matmul(source_aligned, partial_stitch)
                                b.eval(source_aligned)
                                logger.info(
                                    "Partial attention stitch (o_proj): %s [%d,%d] → [%d,%d] "
                                    "(using %dx%d submatrix of %dx%d stitch)",
                                    key, dim0, dim1, tgt_hidden_dim, target_o_dim1,
                                    dim1, target_o_dim1, src_attn_dim, tgt_attn_dim
                                )
                            elif dim1 == src_kv_dim and kv_stitch_input is not None:
                                # o_proj uses KV dimension (unusual but handle it)
                                # Resize kv_stitch if needed
                                kv_in_cols = int(kv_stitch_input.shape[1])
                                if kv_in_cols >= target_o_dim1:
                                    o_stitch = kv_stitch_input[:, :target_o_dim1]
                                else:
                                    # Pad with identity-like projection
                                    o_stitch = b.zeros((dim1, target_o_dim1), dtype=kv_stitch_input.dtype)
                                    o_stitch = _set_submatrix(b, o_stitch, kv_stitch_input, 0, 0)
                                source_aligned = b.matmul(source_aligned, o_stitch)
                                b.eval(source_aligned)
                                logger.info(
                                    "KV-based attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                    key, dim0, dim1, tgt_hidden_dim, target_o_dim1
                                )
                            else:
                                # Compute orthogonal projection between dimensions
                                # This preserves as much geometric structure as possible
                                logger.info(
                                    "Computing orthogonal projection for o_proj: %d → %d",
                                    dim1, target_o_dim1
                                )
                                projection = _compute_dimension_projection(b, dim1, target_o_dim1)
                                source_aligned = b.matmul(source_aligned, projection)
                                b.eval(source_aligned)
                                logger.info(
                                    "Projected attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                    key, dim0, dim1, tgt_hidden_dim, target_o_dim1
                                )

                            metrics.setdefault("attention_stitched", 0)
                            metrics["attention_stitched"] += 1
                            metrics.setdefault("hybrid_attention_handled", 0)
                            metrics["hybrid_attention_handled"] += 1
                            attention_stitch_applied = True

                        else:
                            raise DimensionMismatchError(
                                stage="ATTENTION_WEIGHT_STITCH",
                                weight_key=key,
                                message=(
                                    "Attention weight shape does not match expected "
                                    f"(attn={src_attn_dim}, kv={src_kv_dim}, hidden={src_hidden_dim})"
                                ),
                                context={
                                    "weight_shape": [dim0, dim1],
                                    "expected_attn_dim": src_attn_dim,
                                    "expected_kv_dim": src_kv_dim,
                                    "expected_hidden_dim": src_hidden_dim,
                                    "is_q": is_q,
                                    "is_kv": is_kv,
                                    "is_o": is_o,
                                },
                            )

                    elif is_attention and attention_stitch_output is None:
                        # No attention stitch available - this is a critical failure. No fallbacks.
                        raise StitchUnavailableError(
                            stage="ATTENTION_WEIGHT_STITCH",
                            weight_key=key,
                            message="No attention stitch available for cross-architecture merge",
                            context={
                                "source_shape": list(source_w.shape),
                                "target_shape": list(target_w.shape),
                                "stitch_type": "attention",
                                "reason": "Global attention alignment failed or had insufficient samples",
                            },
                        )

                    # Non-attention weight with hidden dimensions (e.g., layer norm)
                    # Skip hidden stitch if attention stitch was already applied
                    if not attention_stitch_applied:
                        # W_target = hidden_stitch_output @ W @ hidden_stitch_input
                        src_hidden_dim = int(hidden_stitch_output.shape[1])
                        tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                        dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                        if dim0 == src_hidden_dim and dim1 == src_hidden_dim:
                            # BOTH dimensions are hidden_dim
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (both dims): [%d,%d] → [%d,%d]",
                                        dim0, dim1, tgt_hidden_dim, tgt_hidden_dim)

                        elif dim0 == src_hidden_dim:
                            # Hidden dim is OUTPUT only (rows): hidden_stitch_output @ W
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (output only): [%d,%d] → [%d,%d]",
                                        dim0, dim1, tgt_hidden_dim, dim1)

                        elif dim1 == src_hidden_dim:
                            # Hidden dim is INPUT only (cols): W @ hidden_stitch_input
                            source_aligned = b.matmul(source_w, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (input only): [%d,%d] → [%d,%d]",
                                        dim0, dim1, dim0, tgt_hidden_dim)

                        else:
                            # No fallbacks. If dimensions don't match, the stitch is wrong.
                            raise DimensionMismatchError(
                                stage="HIDDEN_WEIGHT_STITCH",
                                weight_key=key,
                                message=f"Weight shape [{dim0},{dim1}] doesn't match hidden_dim {src_hidden_dim}",
                                context={
                                    "weight_shape": [dim0, dim1],
                                    "expected_hidden_dim": src_hidden_dim,
                                    "stitch_type": "hidden",
                                },
                            )

                        metrics.setdefault("hidden_stitch_applied", 0)
                        metrics["hidden_stitch_applied"] += 1

                else:
                    # No fallbacks. The stitch MUST exist.
                    raise StitchUnavailableError(
                        stage="CROSS_ARCHITECTURE_STITCH",
                        weight_key=key,
                        message="Cross-architecture weight but no stitch transformation available",
                        context={
                            "source_shape": list(source_w.shape),
                            "target_shape": list(target_w.shape),
                            "reason": "No hidden or attention stitch computed",
                        },
                    )

                # Verify final shape matches target. No fallbacks.
                if source_aligned.shape != target_w.shape:
                    raise DimensionMismatchError(
                        stage="POST_STITCH_VALIDATION",
                        weight_key=key,
                        message=f"Shape mismatch after stitch: {source_aligned.shape} vs {target_w.shape}",
                        context={
                            "aligned_shape": list(source_aligned.shape),
                            "target_shape": list(target_w.shape),
                        },
                    )

            else:
                source_aligned = source_candidate

            try:
                logger.debug("Computing transplant delta for %s", key)
                result = compute_transplant_delta(
                    weight_target=target_w,
                    weight_source_aligned=source_aligned,
                    activations_core=core_acts,
                    activations_boundary=boundary_acts,
                    backend=b,
                )
                logger.debug("Transplant delta computed for %s: applied=%s", key, result.applied)
            except Exception as e:
                logger.warning("Failed to compute transplant delta for %s: %s", key, e)
                continue

            if result.applied:
                # Use geometry-determined transplant result directly.
                # The null-space projection already computed preserved_fraction
                # based on the spectral structure of boundary activations.
                merged[key] = result.merged_weight
                metrics["weights_transplanted"] += 1
                metrics["preserved_fractions"].append(result.preserved_fraction)
                metrics["projection_losses"].append(result.projection_loss)
                metrics["null_dims"].append(result.null_dim)

                weight_elapsed = time.time() - weight_start_time
                logger.debug(
                    "TRANSPLANT: Weight %d/%d %s - %.2fs (preserved=%.3f, loss=%.6f)",
                    weight_num + 1, len(layer_keys), key,
                    weight_elapsed, result.preserved_fraction, result.projection_loss
                )
                # Use the actual stored weight (already geometry-scaled if applicable)
                actual_merged_weight = merged[key]
                if can_measure_alignment and result.delta_norm > best_delta_norm:
                    try:
                        best_alignment = _compute_alignment_metrics(
                            core_acts=core_acts,
                            weight_before=target_w,
                            weight_after=actual_merged_weight,
                            weight_source=source_aligned,
                            backend=b,
                        )
                        best_delta_norm = result.delta_norm
                    except Exception as e:
                        logger.debug("Alignment metrics failed for %s: %s", key, e)
                if int(boundary_acts.shape[0]) > 0:
                    if int(boundary_acts.shape[1]) != int(target_w.shape[1]):
                        logger.debug(
                            "Boundary metrics skipped for %s (acts=%d, weight_in=%d)",
                            key,
                            int(boundary_acts.shape[1]),
                            int(target_w.shape[1]),
                        )
                        continue

                    target_output = b.matmul(boundary_acts, b.transpose(target_w))
                    merged_output = b.matmul(
                        boundary_acts, b.transpose(actual_merged_weight)
                    )
                    # Geodesic distance: works in all dimensions (reduces to chord
                    # in flat spaces). Chord distance fails in high dimensions (4D+).
                    geo_diffs = geodesic_paired_distances(merged_output, target_output, b)
                    origin = b.zeros_like(target_output)
                    geo_target_norms = geodesic_paired_distances(origin, target_output, b)
                    diff_norm_arr = geodesic_norms(b.reshape(geo_diffs, (1, -1)), b)
                    target_norm_arr = geodesic_norms(b.reshape(geo_target_norms, (1, -1)), b)
                    b.eval(diff_norm_arr, target_norm_arr)

                    diff_norm = float(b.to_scalar(diff_norm_arr))
                    target_norm = float(b.to_scalar(target_norm_arr))
                    eps = float(machine_epsilon(b, target_w))

                    if target_norm > eps:
                        relative_diff = diff_norm / target_norm
                    else:
                        relative_diff = 0.0 if diff_norm <= eps else float("inf")

                    metrics["boundary_relative_diffs"].append(relative_diff)
                layer_transplanted = True

        if layer_transplanted:
            metrics["layers_transplanted"] += 1

        # Layer timing summary
        layer_elapsed = time.time() - layer_start_time
        logger.info(
            "TRANSPLANT: Layer %d complete - %.2fs (%d weights transplanted)",
            layer_idx, layer_elapsed, metrics["weights_transplanted"]
        )

        # Save checkpoint after each layer
        if checkpoint_dir:
            _save_checkpoint(checkpoint_dir, layer_idx, merged, metrics)

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
    # Stage completion summary
    stage_elapsed = time.time() - stage_start_time
    metrics["total_time_seconds"] = stage_elapsed
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

    return TransplantStageResult(merged_weights=output_weights, metrics=metrics)


__all__ = [
    "TransplantStageResult",
    "stage_transplant",
]
