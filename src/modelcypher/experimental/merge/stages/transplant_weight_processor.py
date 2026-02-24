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

"""Per-layer weight processing for transplant stage."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    TrajectoryTangentResult,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_paired_distances,
)
from modelcypher.core.domain.geometry.transplant import (
    compute_behavior_jacobian_projector,
    compute_cross_dimensional_transplant,
    compute_joint_mlp_scale,
    compute_null_space_projector,
    compute_weight_space_transplant,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed
from modelcypher.experimental.merge.exceptions import (
    DimensionMismatchError,
    StitchUnavailableError,
)

from .manifest import (
    TransplantManifest,
    WeightStatus,
    WeightTransformRecord,
)
from .transplant_helpers import (
    _compute_dimension_projection,
    _promote_precision,
    _set_submatrix,
)
from .transplant_mapping import (
    _map_weight_key_cross_arch,
)
from .transplant_metrics import (
    _compute_alignment_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _stable_sigmoid(backend: "Backend", values: "Array") -> "Array":
    """Compute sigmoid with overflow-safe branching."""
    b = backend
    one = b.ones_like(values)
    zero = b.zeros_like(values)
    exp_neg = b.exp(-values)
    exp_pos = b.exp(values)
    sigmoid_pos = one / (one + exp_neg)
    sigmoid_neg = exp_pos / (one + exp_pos)
    result = b.where(values >= zero, sigmoid_pos, sigmoid_neg)
    b.eval(result)
    return result


def _silu(backend: "Backend", values: "Array") -> "Array":
    """Compute SiLU activation using backend ops."""
    b = backend
    sigmoid = _stable_sigmoid(b, values)
    silu = values * sigmoid
    b.eval(silu)
    return silu


def _stack_activations(
    backend: "Backend",
    activations: "Array | list[Array] | None",
    *,
    promote_2d: bool = True,
    promote_list: bool = True,
) -> "Array | None":
    """Normalize activations to a [n, d] batch array."""
    if activations is None:
        return None
    if hasattr(activations, "shape") and len(backend.shape(activations)) == 2:
        batch = backend.array(activations)
        return _promote_precision(batch, backend) if promote_2d else batch
    elif hasattr(activations, "__len__") and len(activations) > 0:
        batch = backend.stack([backend.array(a) for a in activations], axis=0)
        return _promote_precision(batch, backend) if promote_list else batch
    return None


def _block_diag_repeat(
    backend: "Backend",
    block: "Array",
    repeat: int,
) -> "Array":
    """Build a block-diagonal matrix by repeating block along the diagonal."""
    if repeat <= 1:
        return block
    zero = backend.zeros_like(block)
    rows = []
    for i in range(repeat):
        rows.append(
            backend.concatenate(
                [block if i == j else zero for j in range(repeat)],
                axis=1,
            )
        )
    return backend.concatenate(rows, axis=0)


def _is_cross_dimensional(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    """Check if this is a cross-dimensional merge (different hidden/intermediate dims).

    Returns True if source and target have different shapes, indicating
    cross-architecture merging that requires behavioral reconstruction.
    """
    return source_shape != target_shape


def _ensure_dequantized_weight(
    weight: "Array",
    *,
    weight_key: str,
    lookup_key: str | None,
    weight_map: dict[str, "Array"],
    backend: "Backend",
    role: str,
) -> tuple["Array | None", str | None]:
    dtype = str(getattr(weight, "dtype", "")).lower()
    if "int" not in dtype and "uint" not in dtype:
        return weight, None

    key = lookup_key or weight_key
    if key != weight_key:
        logger.debug("Dequantizing %s weight: %s (lookup: %s)", role, weight_key, key)
    else:
        logger.debug("Dequantizing %s weight: %s", role, weight_key)
    weight = dequantize_if_needed(weight, key, weight_map, backend)
    if weight is None or not hasattr(weight, "shape"):
        logger.debug("Failed to dequantize %s weight: %s", role, weight_key)
        return None, f"{role} dequantization failed"

    dtype = str(getattr(weight, "dtype", "")).lower()
    if "int" in dtype or "uint" in dtype:
        logger.debug("Skipping still-quantized %s weight: %s", role, weight_key)
        return None, f"{role} still quantized after dequantization"

    return weight, None


def _apply_behavioral_reconstruction(
    source_weight: "Array",
    target_weight: "Array",
    source_activations: "Array",
    target_activations: "Array",
    alignment_in: "Array",
    alignment_out: "Array",
    source_density_acts: "Array | None",
    target_density_acts: "Array | None",
    delta_scale: float,
    backend: "Backend",
    weight_key: str,
) -> "tuple[Array, dict[str, float]] | None":
    """Reconstruct cross-dimensional weights by matching input/output behavior."""
    b = backend

    try:
        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=source_activations,
            input_activations_target=target_activations,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            source_activations_for_density=source_density_acts,
            target_activations_for_density=target_density_acts,
            delta_scale=delta_scale,
            backend=b,
        )

        metrics = {
            "delta_norm": result.delta_norm,
            "projected_norm": result.projected_norm,
            "preserved_fraction": result.preserved_fraction,
            "transfer_strength": result.transfer_strength,
            "null_rank": result.null_rank,
            "scale_correction": result.scale_correction,
        }

        logger.info(
            "BEHAVIORAL RECONSTRUCTION: %s preserved=%.1f%%, delta_norm=%.4f, scale=%.4f",
            weight_key,
            100.0 * result.preserved_fraction,
            result.delta_norm,
            result.scale_correction,
        )

        return result.merged_weight, metrics

    except Exception as e:
        # Behavioral reconstruction is the CORRECT approach for cross-dimensional merging.
        # If it fails, we should NOT silently fall back to P@W@Q transforms which distort
        # weight magnitudes. Instead, raise with context so we can diagnose and fix.
        raise RuntimeError(
            f"Behavioral reconstruction failed for {weight_key}: {e}. "
            f"This indicates numerical instability in the alignment. "
            f"Check Gram condition number and probe coverage."
        ) from e


@dataclass
class LayerWeightResult:
    weights_processed: int
    layer_transplanted: bool
    best_alignment: dict[str, float] | None
    best_delta_norm: float


@dataclass
class StitchContext:
    hidden_output: "Array | None"
    hidden_input: "Array | None"
    intermediate_output: "Array | None"
    intermediate_input: "Array | None"
    gate_output: "Array | None" = None
    gate_input: "Array | None" = None
    attention_output: "Array | None" = None
    attention_input: "Array | None" = None
    k_output: "Array | None" = None
    k_input: "Array | None" = None
    v_output: "Array | None" = None
    v_input: "Array | None" = None
    kv_input: "Array | None" = None
    dims: dict[str, int] | None = None


@dataclass
class ActivationContext:
    source_hidden: dict[int, list["Array"]] | None
    target_hidden: dict[int, list["Array"]] | None
    source_intermediate: dict[int, list["Array"]] | None
    target_intermediate: dict[int, list["Array"]] | None


@dataclass
class BehaviorJacobianContext:
    """Context for behavior Jacobian null-space projection during merge.

    Carries the target model, tokenizer, and probe texts needed to compute
    per-probe CE gradients on-the-fly for each weight matrix.
    """

    model: Any
    tokenizer: Any
    probe_texts: list[str]
    backend: Any  # Backend with compute_per_probe_gradients()


def process_layer_weights(
    *,
    layer_idx: int,
    layer_keys: list[str],
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    merged: dict[str, "Array"],
    metrics: dict[str, Any],
    layer_mapping: dict[int, int] | None,
    extract_layer_index_fn: Callable[[str], int | None],
    backend: "Backend",
    total_weights: int,
    weights_processed: int,
    progress_callback: Callable[[str, int, int], None] | None,
    stitches: StitchContext,
    activations: ActivationContext,
    density_weights_by_layer: dict[int, "Array"] | None = None,
    density_output_stitch: "Array | None" = None,
    core_acts: "Array",
    boundary_acts: "Array",
    can_measure_alignment: bool,
    manifest: TransplantManifest | None = None,
    delta_scale: float = 1.0,
    layer_scale_ratios: dict[int, float] | None = None,
    source_trajectory_tangents: dict[int, "TrajectoryTangentResult"] | None = None,
    target_trajectory_tangents: dict[int, "TrajectoryTangentResult"] | None = None,
    layer_coupling: list[list[float]] | None = None,
    source_layers: list[int] | None = None,
    target_layers: list[int] | None = None,
    behavior_jacobian_ctx: BehaviorJacobianContext | None = None,
) -> LayerWeightResult:
    b = backend
    hidden_stitch_output = stitches.hidden_output
    hidden_stitch_input = stitches.hidden_input
    intermediate_stitch_output = stitches.intermediate_output
    gate_stitch_output = stitches.gate_output
    attention_stitch_output = stitches.attention_output
    attention_stitch_input = stitches.attention_input
    k_stitch_output = stitches.k_output
    v_stitch_output = stitches.v_output
    kv_stitch_input = stitches.kv_input
    stitch_dims = stitches.dims
    source_activations = activations.source_hidden
    target_activations = activations.target_hidden
    source_intermediate_activations = activations.source_intermediate
    target_intermediate_activations = activations.target_intermediate
    layer_transplanted = False
    best_alignment: dict[str, float] | None = None
    best_delta_norm = -1.0
    mlp_weights: dict[str, "Array"] = {}
    mlp_scales: dict[str, float] = {}  # Store scale corrections for gate/up/down
    mlp_keys: dict[str, str] = {}  # Store weight keys for gate/up/down
    merged_intermediate: "Array | None" = None
    # Track ALL transplanted weight keys for this layer (for full-layer revert)
    transplanted_layer_keys: list[str] = []

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
        if manifest is None:
            return
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

    def _mlp_role(weight_key: str) -> str | None:
        key_lower = weight_key.lower()
        if any(pat in key_lower for pat in ["gate_proj", "mlp.gate", "feed_forward.w1", ".w1"]):
            return "gate"
        if any(pat in key_lower for pat in ["up_proj", "mlp.up", "feed_forward.w3", ".w3", "mlp.fc1"]):
            return "up"
        if any(pat in key_lower for pat in ["down_proj", "mlp.down", "feed_forward.w2", ".w2", "mlp.fc2"]):
            return "down"
        return None

    def _mlp_priority(weight_key: str) -> int:
        role = _mlp_role(weight_key)
        if role == "gate":
            return 0
        if role == "up":
            return 1
        if role == "down":
            return 2
        return 3

    def _align_density_acts(
        acts: "Array | None",
        output_stitch: "Array | None",
    ) -> "Array | None":
        if acts is None or output_stitch is None:
            return acts
        acts = _promote_precision(b.array(acts), b)
        stitch = _promote_precision(output_stitch, b)
        aligned = b.matmul(acts, b.transpose(stitch))
        b.eval(aligned)
        return aligned

    def _get_merged_intermediate() -> "Array | None":
        nonlocal merged_intermediate
        if merged_intermediate is not None:
            return merged_intermediate

        gate_w = mlp_weights.get("gate")
        up_w = mlp_weights.get("up")
        if gate_w is None or up_w is None:
            return None

        if target_activations is None or layer_idx not in target_activations:
            return None

        # Use cached stacking to avoid redundant computation
        hidden = _get_stacked(target_activations, layer_idx, "tgt_hidden")
        if hidden is None:
            return None

        gate_w = _promote_precision(b.array(gate_w), b)
        up_w = _promote_precision(b.array(up_w), b)
        b.eval(hidden, gate_w, up_w)

        gate_out = b.matmul(hidden, b.transpose(gate_w))
        up_out = b.matmul(hidden, b.transpose(up_w))
        b.eval(gate_out, up_out)

        merged_intermediate = _silu(b, gate_out) * up_out
        b.eval(merged_intermediate)
        return merged_intermediate

    ordered_layer_keys = sorted(layer_keys, key=_mlp_priority)

    # OPTIMIZATION: Pre-stack activations before per-weight loop to avoid redundant stacking.
    # Each activation dict entry is stacked once and cached for reuse across all weights.
    _stacked_cache: dict[str, "Array"] = {}

    def _get_stacked(
        acts_dict: "dict | None",
        key: int,
        cache_prefix: str,
        promote_2d: bool = True,
        promote_list: bool = True,
    ) -> "Array | None":
        """Get stacked activations with caching."""
        if acts_dict is None or key not in acts_dict:
            return None
        cache_key = f"{cache_prefix}_{key}"
        if cache_key in _stacked_cache:
            return _stacked_cache[cache_key]
        stacked = _stack_activations(b, acts_dict[key], promote_2d=promote_2d, promote_list=promote_list)
        if stacked is not None:
            _stacked_cache[cache_key] = stacked
        return stacked

    for weight_num, key in enumerate(ordered_layer_keys):
        logger.info(
            "WEIGHT PROCESSOR: Starting weight %d/%d: %s",
            weight_num + 1, len(ordered_layer_keys), key
        )
        weights_processed += 1
        weight_start_time = time.time()
        mlp_role = _mlp_role(key)

        if key.endswith(".scales") or key.endswith(".biases"):
            _record_manifest(
                key,
                WeightStatus.SKIPPED_QUANTIZED,
                error_message="quantization parameter",
            )
            continue
        if not key.endswith(".weight"):
            _record_manifest(
                key,
                WeightStatus.SKIPPED_NON_WEIGHT,
                error_message="non-weight tensor",
            )
            continue

        if progress_callback:
            progress_callback(
                f"Layer {layer_idx}: {key}",
                weights_processed,
                total_weights,
            )

        target_w = target_weights.get(key)

        source_key = _map_weight_key_cross_arch(
            target_key=key,
            source_keys=set(source_weights.keys()),
            layer_mapping=layer_mapping,
            extract_layer_fn=extract_layer_index_fn,
        )
        source_w = source_weights.get(source_key) if source_key else None

        if target_w is None or source_w is None:
            _record_manifest(
                key,
                WeightStatus.SKIPPED_UNMAPPED,
                source_shape=tuple(source_w.shape) if hasattr(source_w, "shape") else None,
                target_shape=tuple(target_w.shape) if hasattr(target_w, "shape") else None,
                error_message="missing source weight" if source_w is None else "missing target weight",
            )
            if source_w is None and "conv.conv" not in key:
                metrics.setdefault("unmapped_weights", [])
                metrics["unmapped_weights"].append(key)
            continue

        metrics["weights_considered"] += 1

        if not hasattr(target_w, "shape") or not hasattr(source_w, "shape"):
            _record_manifest(
                key,
                WeightStatus.SKIPPED_NON_2D,
                error_message="missing shape metadata",
            )
            continue
        ndim_t = len(target_w.shape)
        ndim_s = len(source_w.shape)
        if ndim_t != 2 or ndim_s != 2:
            metrics.setdefault("weights_skipped_non_2d", 0)
            metrics["weights_skipped_non_2d"] += 1
            _record_manifest(
                key,
                WeightStatus.SKIPPED_NON_2D,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message="non-2d weight",
            )
            continue

        target_shape = tuple(target_w.shape)
        source_shape = tuple(source_w.shape)

        target_w, target_err = _ensure_dequantized_weight(
            target_w,
            weight_key=key,
            lookup_key=key,
            weight_map=target_weights,
            backend=b,
            role="target",
        )
        if target_err:
            _record_manifest(
                key,
                WeightStatus.SKIPPED_QUANTIZED,
                source_shape=source_shape,
                target_shape=target_shape,
                error_message=target_err,
            )
            continue

        source_w, source_err = _ensure_dequantized_weight(
            source_w,
            weight_key=key,
            lookup_key=source_key,
            weight_map=source_weights,
            backend=b,
            role="source",
        )
        if source_err:
            _record_manifest(
                key,
                WeightStatus.SKIPPED_QUANTIZED,
                source_shape=source_shape,
                target_shape=target_shape,
                error_message=source_err,
            )
            continue

        if len(source_w.shape) == 1 and len(target_w.shape) == 1:
            src_dim = int(source_w.shape[0])
            tgt_dim = int(target_w.shape[0])

            if "layernorm" in key.lower() or "norm" in key.lower():
                metrics.setdefault("norm_weights_preserved", 0)
                metrics["norm_weights_preserved"] += 1
                logger.debug("PRESERVING target LayerNorm: %s (scale adapter)", key)
                _record_manifest(
                    key,
                    WeightStatus.IDENTITY,
                    source_shape=tuple(source_w.shape),
                    target_shape=tuple(target_w.shape),
                    stitch_type="layernorm",
                    error_message="preserved target layernorm",
                )
                continue

            if src_dim != tgt_dim and hidden_stitch_output is not None:
                stitch_success = False
                try:
                    source_w_2d = b.reshape(source_w, (src_dim, 1))
                    b.eval(source_w_2d)

                    projected = b.matmul(hidden_stitch_output, source_w_2d)
                    b.eval(projected)

                    source_aligned = b.reshape(projected, (tgt_dim,))
                    b.eval(source_aligned)

                    merged[key] = source_aligned
                    transplanted_layer_keys.append(key)
                    stitch_success = True
                    logger.info(
                        "1D stitch (norm/bias): %s [%d] → [%d]",
                        key,
                        src_dim,
                        tgt_dim,
                    )
                except Exception as e:
                    logger.warning("Failed to stitch 1D weight %s: %s", key, e)

                if stitch_success:
                    metrics.setdefault("norm_weights_stitched", 0)
                    metrics["norm_weights_stitched"] += 1
                    metrics["weights_transplanted"] += 1
                    _record_manifest(
                        key,
                        WeightStatus.TRANSFORMED,
                        source_shape=tuple(source_w.shape),
                        target_shape=tuple(target_w.shape),
                        stitch_type="hidden_1d",
                    )
                continue
            if src_dim == tgt_dim:
                merged[key] = source_w
                transplanted_layer_keys.append(key)
                metrics["weights_transplanted"] += 1
                _record_manifest(
                    key,
                    WeightStatus.TRANSFORMED,
                    source_shape=tuple(source_w.shape),
                    target_shape=tuple(target_w.shape),
                    stitch_type="direct_1d",
                )
                continue
            _record_manifest(
                key,
                WeightStatus.FAILED_DIMENSION,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message="1d dimension mismatch",
            )
            continue

        if len(target_w.shape) != 2 or len(source_w.shape) != 2:
            _record_manifest(
                key,
                WeightStatus.SKIPPED_NON_2D,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message="non-2d weight after dequantization",
            )
            continue

        try:
            logger.debug("Converting weight %s for linalg", key)
            target_w = _promote_precision(b.array(target_w), b)
            source_w = _promote_precision(b.array(source_w), b)
            b.eval(target_w, source_w)
            logger.debug("Converted %s: target=%s, source=%s", key, target_w.shape, source_w.shape)
        except Exception as e:
            logger.warning("Failed to convert weight %s: %s", key, e)
            _record_manifest(
                key,
                WeightStatus.FAILED_NUMERICAL,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message=str(e),
            )
            continue

        source_candidate = source_w
        if target_w.shape != source_candidate.shape:
            original_source_shape = source_w.shape
            is_mlp = any(mlp_name in key for mlp_name in [
                "gate_proj", "up_proj", "down_proj",
                "mlp.fc1", "mlp.fc2",
                "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                "mlp.gate", "mlp.up", "mlp.down",
            ])

            # Behavioral reconstruction for cross-dimensional merging.
            # Instead of direct matrix transforms (P @ W @ Q), reconstruct a weight
            # that matches source input/output behavior in target coordinates.
            behavioral_reconstruction_applied = False

            if (
                hidden_stitch_output is not None
                and source_activations is not None
                and target_activations is not None
            ):
                # Get source layer index for activation lookup
                mapped_src_layer = layer_mapping.get(layer_idx, layer_idx) if layer_mapping else layer_idx

                # Get source activations (in source coordinates)
                src_hidden_acts = None
                if mapped_src_layer in source_activations:
                    src_hidden_acts = _stack_activations(
                        b,
                        source_activations[mapped_src_layer],
                    )

                # Get target activations (in target coordinates)
                tgt_hidden_acts = None
                if layer_idx in target_activations:
                    tgt_hidden_acts = _stack_activations(
                        b,
                        target_activations[layer_idx],
                    )

                # Determine alignment transforms based on weight type
                dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])
                src_hidden_dim = stitch_dims.get("src_hidden", 0) if stitch_dims else 0
                src_inter_dim = stitch_dims.get("src_inter", 0) if stitch_dims else 0

                alignment_in = None
                alignment_out = None
                source_acts_for_behavior = None
                target_acts_for_behavior = None

                # Hidden transform: transpose stitch_output to get [src, tgt]
                alignment_hidden = b.transpose(hidden_stitch_output) if hidden_stitch_output is not None else None
                alignment_inter = b.transpose(intermediate_stitch_output) if intermediate_stitch_output is not None else None

                if is_mlp and src_hidden_dim > 0 and src_inter_dim > 0:
                    is_gate_or_up = any(
                        n in key for n in ["gate_proj", "up_proj", "mlp.gate", "mlp.up",
                                           "feed_forward.w1", "feed_forward.w3", ".w1", ".w3"]
                    )
                    is_down = any(n in key for n in ["down_proj", "mlp.down", "feed_forward.w2", ".w2", "mlp.fc2"])

                    if is_gate_or_up and dim0 == src_inter_dim and dim1 == src_hidden_dim:
                        # gate/up: input=hidden, output=intermediate
                        if alignment_hidden is not None and alignment_inter is not None:
                            # Use gate-specific alignment if available
                            if gate_stitch_output is not None:
                                alignment_out = b.transpose(gate_stitch_output)
                            else:
                                alignment_out = alignment_inter
                            alignment_in = alignment_hidden
                            source_acts_for_behavior = src_hidden_acts
                            target_acts_for_behavior = tgt_hidden_acts

                    elif is_down and dim0 == src_hidden_dim and dim1 == src_inter_dim:
                        # down: input=intermediate, output=hidden
                        if alignment_hidden is not None and alignment_inter is not None:
                            alignment_in = alignment_inter
                            alignment_out = alignment_hidden
                            # For down_proj, we need intermediate activations
                            if source_intermediate_activations is not None and mapped_src_layer in source_intermediate_activations:
                                source_acts_for_behavior = _stack_activations(
                                    b,
                                    source_intermediate_activations[mapped_src_layer],
                                )
                            if target_intermediate_activations is not None and layer_idx in target_intermediate_activations:
                                target_acts_for_behavior = _stack_activations(
                                    b,
                                    target_intermediate_activations[layer_idx],
                                )

                elif dim0 == src_hidden_dim and dim1 == src_hidden_dim:
                    # Hidden→Hidden weight (e.g., some attention projections)
                    if alignment_hidden is not None:
                        alignment_in = alignment_hidden
                        alignment_out = alignment_hidden
                        source_acts_for_behavior = src_hidden_acts
                        target_acts_for_behavior = tgt_hidden_acts

                # Attempt behavioral reconstruction if we have all required data
                if (
                    alignment_in is not None
                    and alignment_out is not None
                    and source_acts_for_behavior is not None
                    and target_acts_for_behavior is not None
                ):
                    b.eval(alignment_in, alignment_out, source_acts_for_behavior, target_acts_for_behavior)

                    effective_delta_scale = delta_scale

                    result = _apply_behavioral_reconstruction(
                        source_weight=source_w,
                        target_weight=target_w,
                        source_activations=source_acts_for_behavior,
                        target_activations=target_acts_for_behavior,
                        alignment_in=alignment_in,
                        alignment_out=alignment_out,
                        source_density_acts=source_acts_for_behavior,
                        target_density_acts=target_acts_for_behavior,
                        delta_scale=effective_delta_scale,
                        backend=b,
                        weight_key=key,
                    )

                    if result is not None:
                        merged_weight, behavior_metrics = result
                        merged[key] = merged_weight
                        transplanted_layer_keys.append(key)
                        # Store MLP weights and scales for joint correction
                        if mlp_role in ("gate", "up", "down"):
                            mlp_weights[mlp_role] = merged_weight
                            mlp_scales[mlp_role] = behavior_metrics.get("scale_correction", 1.0)
                            mlp_keys[mlp_role] = key
                            if mlp_role in ("gate", "up"):
                                merged_intermediate = None
                        metrics["weights_transplanted"] += 1
                        metrics["preserved_fractions"].append(behavior_metrics["preserved_fraction"])
                        metrics["projection_losses"].append(1.0 - behavior_metrics["preserved_fraction"])
                        metrics["null_dims"].append(behavior_metrics["null_rank"])
                        metrics.setdefault("behavioral_reconstructed", 0)
                        metrics["behavioral_reconstructed"] += 1
                        layer_transplanted = True
                        behavioral_reconstruction_applied = True
                        _record_manifest(
                            key,
                            WeightStatus.TRANSFORMED,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="behavioral",
                            preserved_fraction=behavior_metrics.get("preserved_fraction"),
                        )

                        logger.info(
                            "CROSS-DIM BEHAVIORAL: %s [%s] → [%s] preserved=%.1f%%",
                            key,
                            list(original_source_shape),
                            list(target_w.shape),
                            100.0 * behavior_metrics["preserved_fraction"],
                        )

            # Skip direct stitch if behavioral reconstruction succeeded
            if behavioral_reconstruction_applied:
                continue

            # =========================================================================
            # FALLBACK: Direct stitch transforms (P @ W @ Q)
            # Used when behavioral reconstruction is not available or fails.
            # WARNING: This can distort weight magnitudes in cross-dimensional cases.
            # =========================================================================

            # =========================================================================
            # CONV LAYER HANDLING (LFM2 hybrid architecture)
            # =========================================================================
            # LFM2 models have conv layers with specific dimension patterns:
            # - conv.in_proj.weight: (3×hidden, hidden) - projects to 3x for conv
            # - conv.out_proj.weight: (hidden, hidden) - projects back
            # - conv.conv.weight: (hidden, 3, 1) - the actual convolution kernel
            #
            # These need special stitching since dim0 of in_proj is 3×hidden, not
            # the MLP intermediate dimension.
            # =========================================================================
            is_conv = ".conv." in key and any(
                conv_name in key for conv_name in ["in_proj", "out_proj", "conv.conv"]
            )
            if is_conv and hidden_stitch_output is not None and hidden_stitch_input is not None:
                src_hidden_dim = stitch_dims["src_hidden"]
                tgt_hidden_dim = stitch_dims["tgt_hidden"]
                dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                if "conv.in_proj" in key:
                    # in_proj: (3×src_hidden, src_hidden) → (3×tgt_hidden, tgt_hidden)
                    # The 3× factor is intrinsic to the conv architecture
                    conv_factor = dim0 // src_hidden_dim  # Should be 3
                    if dim0 == conv_factor * src_hidden_dim and dim1 == src_hidden_dim:
                        # Block-diagonal output stitch: repeat hidden_stitch along the diagonal.
                        conv_output_stitch = _block_diag_repeat(
                            b,
                            hidden_stitch_output,
                            conv_factor,
                        )
                        b.eval(conv_output_stitch)

                        source_aligned = b.matmul(conv_output_stitch, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Conv stitch (in_proj): [%d,%d] → [%d,%d] (factor=%d×hidden)",
                            dim0,
                            dim1,
                            conv_factor * tgt_hidden_dim,
                            tgt_hidden_dim,
                            conv_factor,
                        )
                        metrics.setdefault("conv_stitched", 0)
                        metrics["conv_stitched"] += 1
                    else:
                        logger.warning(
                            "Conv in_proj %s shape [%d,%d] unexpected for hidden=%d - skipping",
                            key, dim0, dim1, src_hidden_dim,
                        )
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_DIMENSION,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="conv_in_proj",
                            error_message="conv in_proj shape mismatch",
                        )
                        continue

                elif "conv.out_proj" in key:
                    # out_proj: (src_hidden, src_hidden) → (tgt_hidden, tgt_hidden)
                    if dim0 == src_hidden_dim and dim1 == src_hidden_dim:
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Conv stitch (out_proj): [%d,%d] → [%d,%d]",
                            dim0,
                            dim1,
                            tgt_hidden_dim,
                            tgt_hidden_dim,
                        )
                        metrics.setdefault("conv_stitched", 0)
                        metrics["conv_stitched"] += 1
                    else:
                        logger.warning(
                            "Conv out_proj %s shape [%d,%d] unexpected for hidden=%d - skipping",
                            key, dim0, dim1, src_hidden_dim,
                        )
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_DIMENSION,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="conv_out_proj",
                            error_message="conv out_proj shape mismatch",
                        )
                        continue

                elif "conv.conv" in key:
                    # conv.weight: (src_hidden, kernel_size, 1) → (tgt_hidden, kernel_size, 1)
                    # Only need to stitch dim0
                    if len(original_source_shape) == 3 and dim0 == src_hidden_dim:
                        # 3D weight: (hidden, kernel, 1)
                        kernel_size = int(original_source_shape[1])
                        # Reshape to 2D, stitch, reshape back
                        source_2d = b.reshape(source_w, (dim0, kernel_size))
                        source_aligned_2d = b.matmul(hidden_stitch_output, source_2d)
                        source_aligned = b.reshape(source_aligned_2d, (tgt_hidden_dim, kernel_size, 1))
                        b.eval(source_aligned)
                        logger.info(
                            "Conv stitch (conv.conv): [%d,%d,1] → [%d,%d,1]",
                            dim0,
                            kernel_size,
                            tgt_hidden_dim,
                            kernel_size,
                        )
                        metrics.setdefault("conv_stitched", 0)
                        metrics["conv_stitched"] += 1
                    else:
                        logger.warning(
                            "Conv conv.weight %s shape %s unexpected for hidden=%d - skipping",
                            key, original_source_shape, src_hidden_dim,
                        )
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_DIMENSION,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="conv_conv",
                            error_message="conv weight shape mismatch",
                        )
                        continue

            elif is_mlp and hidden_stitch_output is not None and hidden_stitch_input is not None:
                src_hidden_dim = stitch_dims["src_hidden"]
                tgt_hidden_dim = stitch_dims["tgt_hidden"]
                if "src_inter" in stitch_dims:
                    src_inter_dim = stitch_dims["src_inter"]
                else:
                    src_inter_dim = dim0 if dim0 != src_hidden_dim else dim1

                if "tgt_inter" in stitch_dims:
                    tgt_inter_dim = stitch_dims["tgt_inter"]
                elif target_w is not None:
                    tgt_dim0 = int(target_w.shape[0])
                    tgt_dim1 = int(target_w.shape[1])
                    tgt_inter_dim = tgt_dim0 if tgt_dim0 != tgt_hidden_dim else tgt_dim1
                else:
                    tgt_inter_dim = 0

                logger.info(
                    "MLP weight %s: applying dual stitch (hidden %d→%d, inter %d→%d)",
                    key,
                    src_hidden_dim,
                    tgt_hidden_dim,
                    src_inter_dim,
                    tgt_inter_dim,
                )

                dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                is_gate_or_up = any(
                    mlp_name in key
                    for mlp_name in [
                        "gate_proj",
                        "up_proj",
                        "mlp.gate",
                        "mlp.up",
                        "feed_forward.w1",
                        "feed_forward.w3",
                        ".w1",
                        ".w3",
                    ]
                )

                if dim0 == src_inter_dim and dim1 == src_hidden_dim:
                    # CRITICAL FOR CROSS-ARCHITECTURE:
                    # Use gate_stitch_output (PRE-SiLU alignment) for gate/up weights
                    # instead of intermediate_stitch_output (POST-SiLU alignment).
                    # Compressions don't commute with SiLU, so POST-SiLU alignment
                    # does not match PRE-SiLU weight outputs.
                    if is_gate_or_up and gate_stitch_output is not None:
                        output_stitch = gate_stitch_output
                        logger.info(
                            "Using PRE-SiLU gate_stitch for %s (cross-arch fix)",
                            key,
                        )
                    else:
                        output_stitch = intermediate_stitch_output
                        if is_gate_or_up and src_inter_dim != tgt_inter_dim:
                            # Cross-architecture without gate stitch - log warning
                            logger.warning(
                                "CROSS-ARCH: %s uses POST-SiLU alignment (gate_stitch unavailable). "
                                "This may cause output quality degradation.",
                                key,
                            )
                    if is_gate_or_up and target_w is not None and gate_stitch_output is None:
                        # Compositional stitch: S @ W_src @ H = W_tgt
                        # This is closed-form and correct. If it fails, that's a numerical
                        # stability issue to diagnose, not hide with POST-SiLU fallback.
                        aligner = GramAligner(backend=b)
                        H = b.transpose(hidden_stitch_output)
                        target_w_float = _promote_precision(b.array(target_w), b)
                        b.eval(H, target_w_float)
                        try:
                            output_stitch = aligner.compositional_stitch(
                                hidden_transform=H,
                                source_weight=source_w,
                                target_weight=target_w_float,
                            )
                            b.eval(output_stitch)
                            metrics.setdefault("mlp_gate_up_compositional", 0)
                            metrics["mlp_gate_up_compositional"] += 1
                            logger.info(
                                "MLP compositional stitch for %s: [%d,%d] → [%d,%d]",
                                key,
                                dim0,
                                dim1,
                                tgt_inter_dim,
                                tgt_hidden_dim,
                            )
                        except Exception as e:
                            # Compositional stitch IS correct mathematically.
                            # Failure means numerical instability - raise to diagnose.
                            raise RuntimeError(
                                f"Compositional stitch failed for {key}: {e}. "
                                f"The equation S @ W_src @ H = W_tgt is closed-form. "
                                f"Check condition numbers and probe coverage."
                            ) from e

                    if output_stitch is None:
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_STITCH,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="mlp_gate_up",
                            error_message="missing intermediate stitch for gate/up projection",
                        )
                        raise StitchUnavailableError(
                            stage="MLP_GATE_UP_STITCH",
                            weight_key=key,
                            message="Missing intermediate stitch for gate/up projection",
                            context={
                                "weight_shape": [dim0, dim1],
                                "src_hidden_dim": src_hidden_dim,
                                "src_inter_dim": src_inter_dim,
                            },
                        )

                    source_aligned = b.matmul(output_stitch, source_w)
                    source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                    b.eval(source_aligned)
                    logger.info(
                        "Dual stitch (gate/up): [%d,%d] → [%d,%d]",
                        dim0,
                        dim1,
                        int(output_stitch.shape[0]),
                        tgt_hidden_dim,
                    )

                elif dim0 == src_hidden_dim and dim1 == src_inter_dim:
                    if hidden_stitch_output is None:
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_STITCH,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="mlp_down",
                            error_message="missing hidden stitch for down projection",
                        )
                        raise StitchUnavailableError(
                            stage="MLP_DOWN_STITCH",
                            weight_key=key,
                            message="Missing hidden stitch for down projection",
                            context={
                                "weight_shape": [dim0, dim1],
                                "src_hidden_dim": src_hidden_dim,
                                "src_inter_dim": src_inter_dim,
                            },
                        )
                    # For down_proj INPUT stitch:
                    # Use compositional_stitch_input as PRIMARY approach.
                    # This derives input stitch from hidden alignment + weights,
                    # ensuring consistency with how gate/up stitches are computed.
                    #
                    # intermediate_stitch_input (POST-SiLU) can diverge at inference
                    # because merged gate/up intermediates differ after nonlinearity.
                    # Compositional stitch derives input from hidden+weights for
                    # consistency with merged gate/up outputs.
                    input_stitch = None
                    if target_w is not None:
                        try:
                            aligner = GramAligner(backend=b)
                            H = b.transpose(hidden_stitch_output)
                            target_w_float = _promote_precision(b.array(target_w), b)
                            b.eval(H, target_w_float)
                            input_stitch = aligner.compositional_stitch_input(
                                hidden_transform=H,
                                source_weight=source_w,
                                target_weight=target_w_float,
                            )
                            b.eval(input_stitch)
                            metrics.setdefault("mlp_down_compositional", 0)
                            metrics["mlp_down_compositional"] += 1
                            logger.info(
                                "MLP compositional input stitch for %s: W=[%d,%d], S_in=[%d,%d] → [%d,%d]",
                                key,
                                dim0,
                                dim1,
                                int(input_stitch.shape[0]),
                                int(input_stitch.shape[1]),
                                tgt_hidden_dim,
                                tgt_inter_dim,
                            )
                        except Exception as e:
                            logger.warning(
                                "MLP compositional input stitch failed for %s: %s",
                                key,
                                e,
                            )
                    if input_stitch is None:
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_STITCH,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="mlp_down",
                            error_message="missing compositional input stitch for down projection",
                        )
                        raise StitchUnavailableError(
                            stage="MLP_DOWN_STITCH",
                            weight_key=key,
                            message="Missing compositional input stitch for down projection",
                            context={
                                "weight_shape": [dim0, dim1],
                                "src_hidden_dim": src_hidden_dim,
                                "src_inter_dim": src_inter_dim,
                            },
                        )

                    source_aligned = b.matmul(hidden_stitch_output, source_w)
                    source_aligned = b.matmul(source_aligned, input_stitch)
                    b.eval(source_aligned)
                    logger.info(
                        "Dual stitch (down): [%d,%d] → [%d,%d]",
                        dim0,
                        dim1,
                        tgt_hidden_dim,
                        int(input_stitch.shape[0]),
                    )
                else:
                    logger.warning(
                        "MLP weight %s shape [%d,%d] doesn't match expected dims "
                        "(hidden=%d, inter=%d) - skipping",
                        key,
                        dim0,
                        dim1,
                        src_hidden_dim,
                        src_inter_dim,
                    )
                    _record_manifest(
                        key,
                        WeightStatus.FAILED_DIMENSION,
                        source_shape=tuple(source_w.shape),
                        target_shape=tuple(target_w.shape),
                        stitch_type="mlp",
                        error_message="mlp weight shape mismatch",
                    )
                    continue

                metrics.setdefault("dual_stitch_applied", 0)
                metrics["dual_stitch_applied"] += 1

            elif hidden_stitch_output is not None and hidden_stitch_input is not None:
                is_attention = any(attn_name in key for attn_name in [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "self_attn", "query", "key", "value",
                ])

                attention_stitch_applied = False

                if is_attention and attention_stitch_output is None and k_stitch_output is None and v_stitch_output is None:
                    # ATTENTION STITCH UNAVAILABLE
                    # Check for experimental MLP-only mode (useful for cross-arch experiments)
                    skip_attention = os.environ.get("MC_SKIP_ATTENTION_STITCH", "").lower() in ("1", "true", "yes")
                    if skip_attention:
                        # MLP-ONLY INJECTION MODE: Skip attention weights, preserve target
                        logger.warning(
                            "ATTENTION STITCH UNAVAILABLE: Preserving target for %s (MC_SKIP_ATTENTION_STITCH=1)",
                            key,
                        )
                        _record_manifest(
                            key,
                            WeightStatus.SKIPPED_ATTENTION_NO_STITCH,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="attention",
                            error_message="attention stitch skipped (MLP-only mode)",
                        )
                        metrics.setdefault("attention_skipped", 0)
                        metrics["attention_skipped"] += 1
                        continue  # Skip to next weight, preserving target

                    # Default behavior: FAILURE case
                    # This is a FAILURE case, not a "preserve target" case.
                    # If we're here, the probe stage didn't compute attention stitches.
                    # Silently preserving target hides the fact that we're not transplanting.
                    _record_manifest(
                        key,
                        WeightStatus.FAILED_STITCH,
                        source_shape=tuple(source_w.shape),
                        target_shape=tuple(target_w.shape),
                        stitch_type="attention",
                        error_message="attention stitch not computed by probe stage",
                    )
                    raise StitchUnavailableError(
                        stage="ATTENTION_WEIGHT_STITCH",
                        weight_key=key,
                        message=(
                            "Attention stitch unavailable. This means probe stage did not "
                            "compute attention stitches for this layer. Either: (1) probe "
                            "coverage was insufficient, or (2) architecture has attention "
                            "patterns not yet supported. Cannot silently preserve target. "
                            "Set MC_SKIP_ATTENTION_STITCH=1 for MLP-only injection experiments."
                        ),
                        context={
                            "source_shape": list(source_w.shape),
                            "target_shape": list(target_w.shape),
                            "layer_idx": layer_idx,
                        },
                    )

                if is_attention and attention_stitch_output is not None:
                    src_attn_dim = stitch_dims["src_attn"]
                    tgt_attn_dim = stitch_dims["tgt_attn"]
                    src_hidden_dim = stitch_dims["src_hidden"]
                    tgt_hidden_dim = stitch_dims["tgt_hidden"]
                    dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                    src_kv_dim = stitch_dims.get("src_kv", src_attn_dim)
                    stitch_dims.get("tgt_kv", tgt_attn_dim)

                    is_q = any(n in key for n in ["q_proj", "query"])
                    is_kv = any(n in key for n in ["k_proj", "v_proj", "key", "value"])
                    is_o = any(n in key for n in ["o_proj", "out_proj"])

                    if is_q and dim0 == src_attn_dim and dim1 == src_hidden_dim:
                        source_aligned = b.matmul(attention_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Attention stitch (q_proj): %s [%d,%d] → [%d,%d]",
                            key,
                            dim0,
                            dim1,
                            tgt_attn_dim,
                            tgt_hidden_dim,
                        )
                        metrics.setdefault("attention_stitched", 0)
                        metrics["attention_stitched"] += 1
                        attention_stitch_applied = True

                    elif is_q and dim0 == src_attn_dim and dim1 == src_attn_dim:
                        source_aligned = b.matmul(attention_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, attention_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Attention stitch (q_proj square): %s [%d,%d] → [%d,%d]",
                            key,
                            dim0,
                            dim1,
                            tgt_attn_dim,
                            tgt_attn_dim,
                        )
                        metrics.setdefault("attention_stitched", 0)
                        metrics["attention_stitched"] += 1
                        attention_stitch_applied = True

                    elif is_kv and dim1 == src_hidden_dim:
                        is_k = any(n in key for n in ["k_proj", "key"])
                        is_v_proj = any(n in key for n in ["v_proj", "value"])

                        kv_out = None
                        stitch_type = "probe"

                        if is_k and k_stitch_output is not None:
                            kv_out = k_stitch_output
                            stitch_type = "K stitch (probe)"
                        elif is_v_proj and v_stitch_output is not None:
                            kv_out = v_stitch_output
                            stitch_type = "V stitch (probe)"

                        if kv_out is not None:
                            source_aligned = b.matmul(kv_out, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info(
                                "%s (k/v_proj): %s [%d,%d] → aligned",
                                stitch_type,
                                key,
                                dim0,
                                dim1,
                            )
                            metrics.setdefault("kv_stitched", 0)
                            metrics["kv_stitched"] += 1
                            attention_stitch_applied = True
                        else:
                            metrics.setdefault("attention_preserved", 0)
                            metrics["attention_preserved"] += 1
                            _record_manifest(
                                key,
                                WeightStatus.IDENTITY,
                                source_shape=tuple(source_w.shape),
                                target_shape=tuple(target_w.shape),
                                stitch_type="attention",
                                error_message="k/v stitch unavailable",
                            )
                            continue

                    elif is_o and dim0 == src_hidden_dim and dim1 == src_attn_dim:
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, attention_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                            key,
                            dim0,
                            dim1,
                            tgt_hidden_dim,
                            tgt_attn_dim,
                        )
                        metrics.setdefault("attention_stitched", 0)
                        metrics["attention_stitched"] += 1
                        attention_stitch_applied = True

                    elif is_o and dim0 == src_hidden_dim and dim1 != src_attn_dim:
                        target_o_dim1 = int(target_w.shape[1])

                        logger.info(
                            "Hybrid attention detected for %s: o_proj dim1=%d != Q_attn_dim=%d. "
                            "Computing adaptive stitch → target dim=%d",
                            key,
                            dim1,
                            src_attn_dim,
                            target_o_dim1,
                        )

                        source_aligned = b.matmul(hidden_stitch_output, source_w)

                        if dim1 < src_attn_dim and target_o_dim1 <= tgt_attn_dim:
                            partial_stitch = attention_stitch_input[:dim1, :target_o_dim1]
                            source_aligned = b.matmul(source_aligned, partial_stitch)
                            b.eval(source_aligned)
                            logger.info(
                                "Partial attention stitch (o_proj): %s [%d,%d] → [%d,%d] "
                                "(using %dx%d submatrix of %dx%d stitch)",
                                key,
                                dim0,
                                dim1,
                                tgt_hidden_dim,
                                target_o_dim1,
                                dim1,
                                target_o_dim1,
                                src_attn_dim,
                                tgt_attn_dim,
                            )
                        elif dim1 == src_kv_dim and kv_stitch_input is not None:
                            kv_in_cols = stitch_dims.get("tgt_kv", int(kv_stitch_input.shape[1]))
                            if kv_in_cols >= target_o_dim1:
                                o_stitch = kv_stitch_input[:, :target_o_dim1]
                            else:
                                o_stitch = b.zeros((dim1, target_o_dim1), dtype=kv_stitch_input.dtype)
                                o_stitch = _set_submatrix(b, o_stitch, kv_stitch_input, 0, 0)
                            source_aligned = b.matmul(source_aligned, o_stitch)
                            b.eval(source_aligned)
                            logger.info(
                                "KV-based attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                key,
                                dim0,
                                dim1,
                                tgt_hidden_dim,
                                target_o_dim1,
                            )
                        else:
                            logger.info(
                                "Computing orthogonal projection for o_proj: %d → %d",
                                dim1,
                                target_o_dim1,
                            )
                            projection = _compute_dimension_projection(b, dim1, target_o_dim1)
                            source_aligned = b.matmul(source_aligned, projection)
                            b.eval(source_aligned)
                            logger.info(
                                "Projected attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                key,
                                dim0,
                                dim1,
                                tgt_hidden_dim,
                                target_o_dim1,
                            )

                        metrics.setdefault("attention_stitched", 0)
                        metrics["attention_stitched"] += 1
                        metrics.setdefault("hybrid_attention_handled", 0)
                        metrics["hybrid_attention_handled"] += 1
                        attention_stitch_applied = True

                    else:
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_DIMENSION,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="attention",
                            error_message="attention weight shape mismatch",
                        )
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
                    metrics.setdefault("attention_preserved", 0)
                    metrics["attention_preserved"] += 1
                    _record_manifest(
                        key,
                        WeightStatus.IDENTITY,
                        source_shape=tuple(source_w.shape),
                        target_shape=tuple(target_w.shape),
                        stitch_type="attention",
                        error_message="attention stitch unavailable",
                    )
                    continue

                if not attention_stitch_applied:
                    src_hidden_dim = stitch_dims["src_hidden"]
                    tgt_hidden_dim = stitch_dims["tgt_hidden"]
                    dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                    if dim0 == src_hidden_dim and dim1 == src_hidden_dim:
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Hidden stitch (both dims): [%d,%d] → [%d,%d]",
                            dim0,
                            dim1,
                            tgt_hidden_dim,
                            tgt_hidden_dim,
                        )

                    elif dim0 == src_hidden_dim:
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        b.eval(source_aligned)
                        logger.info(
                            "Hidden stitch (output only): [%d,%d] → [%d,%d]",
                            dim0,
                            dim1,
                            tgt_hidden_dim,
                            dim1,
                        )

                    elif dim1 == src_hidden_dim:
                        source_aligned = b.matmul(source_w, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info(
                            "Hidden stitch (input only): [%d,%d] → [%d,%d]",
                            dim0,
                            dim1,
                            dim0,
                            tgt_hidden_dim,
                        )

                    else:
                        _record_manifest(
                            key,
                            WeightStatus.FAILED_DIMENSION,
                            source_shape=tuple(source_w.shape),
                            target_shape=tuple(target_w.shape),
                            stitch_type="hidden",
                            error_message="hidden weight shape mismatch",
                        )
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
                _record_manifest(
                    key,
                    WeightStatus.FAILED_STITCH,
                    source_shape=tuple(source_w.shape),
                    target_shape=tuple(target_w.shape),
                    error_message="no stitch transformation available",
                )
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

            # Apply scale_ratio for cross-dimensional merges
            # F = pinv(A_s) @ A_t is NOT norm-preserving when d_s != d_t
            # scale_ratio = ||target|| / ||source @ F|| corrects for this
            if layer_scale_ratios and layer_idx in layer_scale_ratios:
                sr = layer_scale_ratios[layer_idx]
                eps = float(machine_epsilon(b, source_aligned))
                if abs(sr - 1.0) > eps:
                    logger.info(
                        "SCALE_RATIO: Applying %.4f to cross-dim %s",
                        sr, key,
                    )
                    source_aligned = source_aligned * sr
                    b.eval(source_aligned)

            if source_aligned.shape != target_w.shape:
                _record_manifest(
                    key,
                    WeightStatus.FAILED_DIMENSION,
                    source_shape=tuple(source_w.shape),
                    target_shape=tuple(target_w.shape),
                    error_message="shape mismatch after stitch",
                )
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

        tgt_hidden_dim = stitch_dims.get("tgt_hidden", 0) if stitch_dims else 0
        tgt_inter_dim = stitch_dims.get("tgt_inter", 0) if stitch_dims else 0
        weight_in_dim = int(target_w.shape[1])

        input_activations = None
        src_density_acts = None
        tgt_density_acts = None
        activation_space = "hidden"

        # For cross-architecture, use layer_mapping to find the mapped source layer.
        mapped_src_layer = layer_mapping.get(layer_idx, layer_idx) if layer_mapping else layer_idx

        if weight_in_dim == tgt_inter_dim and tgt_inter_dim > 0:
            activation_space = "intermediate"

            if mlp_role == "down":
                merged_input = _get_merged_intermediate()
                if merged_input is not None:
                    input_activations = merged_input
                    tgt_density_acts = merged_input

            # First, get target intermediate activations (needed for null-space projection)
            if target_intermediate_activations is None:
                logger.warning(
                    "INTERMEDIATE MISS: target_intermediate_activations is None for %s layer=%d",
                    key, layer_idx
                )
            elif layer_idx not in target_intermediate_activations:
                logger.warning(
                    "INTERMEDIATE MISS: Layer %d not in target_intermediate_activations for %s (key_count=%d)",
                    layer_idx,
                    key,
                    len(target_intermediate_activations),
                )

            if (
                target_intermediate_activations is not None
                and layer_idx in target_intermediate_activations
                and input_activations is None
            ):
                # Use cached stacking to avoid redundant computation
                input_activations = _get_stacked(
                    target_intermediate_activations, layer_idx, "tgt_inter",
                    promote_2d=True, promote_list=False,
                )
                tgt_density_acts = input_activations

            # Then, get source intermediate activations for density comparison (optional)
            if (
                input_activations is not None
                and source_intermediate_activations is not None
                and mapped_src_layer in source_intermediate_activations
            ):
                # Use cached stacking to avoid redundant computation
                src_density_acts = _get_stacked(
                    source_intermediate_activations, mapped_src_layer, "src_inter",
                    promote_2d=True, promote_list=False,
                )
                if src_density_acts is not None:
                    logger.debug(
                        "INTERMEDIATE: Layer %d using mapped source layer %d for density",
                        layer_idx, mapped_src_layer
                    )

        if input_activations is None:
            activation_space = "hidden"
            if target_activations is not None and layer_idx in target_activations:
                # Use cached stacking to avoid redundant computation
                input_activations = _get_stacked(
                    target_activations, layer_idx, "tgt_hidden",
                    promote_2d=True, promote_list=False,
                )
                tgt_density_acts = input_activations

                if (
                    input_activations is not None
                    and (density_weights_by_layer is None or layer_idx not in density_weights_by_layer)
                    and source_activations is not None
                    and mapped_src_layer in source_activations
                ):
                    # Use mapped source layer for density comparison (cached).
                    src_density_acts = _get_stacked(
                        source_activations, mapped_src_layer, "src_hidden",
                        promote_2d=True, promote_list=False,
                    )
                    if src_density_acts is not None:
                        logger.debug(
                            "HIDDEN: Layer %d using mapped source layer %d for density",
                            layer_idx, mapped_src_layer
                        )

        if input_activations is None:
            logger.debug(
                "TRANSPLANT: No %s activations for layer %d, preserving target for %s",
                activation_space,
                layer_idx,
                key,
            )
            metrics.setdefault("no_activation_preserved", 0)
            metrics["no_activation_preserved"] += 1
            _record_manifest(
                key,
                WeightStatus.SKIPPED_MISSING_ACTIVATIONS,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message=f"missing {activation_space} activations",
            )
            continue

        # Align density activations to target space
        if activation_space == "hidden":
            density_stitch = density_output_stitch or hidden_stitch_output
            src_density_acts = _align_density_acts(src_density_acts, density_stitch)
        elif activation_space == "intermediate":
            src_density_acts = _align_density_acts(src_density_acts, intermediate_stitch_output)

        b.eval(input_activations)
        if src_density_acts is not None:
            b.eval(src_density_acts)
        if tgt_density_acts is not None:
            b.eval(tgt_density_acts)

        # Verify activation dimension matches weight input dimension.
        input_act_dim = int(b.shape(input_activations)[1])
        if input_act_dim != weight_in_dim:
            logger.debug(
                "TRANSPLANT: Activation dimension mismatch for %s: got %d, expected %d "
                "(space=%s, layer=%d, mapped_src=%d). Preserving target.",
                key, input_act_dim, weight_in_dim, activation_space, layer_idx, mapped_src_layer
            )
            metrics.setdefault("activation_dim_mismatch", 0)
            metrics["activation_dim_mismatch"] += 1
            _record_manifest(
                key,
                WeightStatus.SKIPPED_ACTIVATION_DIM_MISMATCH,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                error_message="activation dimension mismatch",
            )
            continue

        # Debug logging for shape verification
        input_shape = b.shape(input_activations)
        src_density_shape = b.shape(src_density_acts) if src_density_acts is not None else None
        tgt_density_shape = b.shape(tgt_density_acts) if tgt_density_acts is not None else None
        logger.debug(
            "TRANSPLANT SHAPES: %s - input=%s, src_density=%s, tgt_density=%s (space=%s)",
            key, input_shape, src_density_shape, tgt_density_shape, activation_space
        )

        logger.debug(
            "TRANSPLANT: %s using %s activations [%d samples] for null-space",
            key,
            activation_space,
            int(b.shape(input_activations)[0]),
        )

        density_weights_override = None
        if activation_space == "hidden":
            if density_weights_by_layer is not None:
                density_weights_override = density_weights_by_layer.get(layer_idx)
        elif activation_space == "intermediate":
            pass

        # OPTIMIZATION: For full transfer (delta_scale≈1.0), skip expensive k-NN
        # density computation by providing pre-computed weights of 1.0 (full transfer).
        # This is critical for bottleneck layers where we want full transfer anyway.
        sqrt_eps = sqrt_scalar(machine_epsilon(b, input_activations), b)
        if delta_scale >= 1.0 - sqrt_eps and density_weights_override is None:
            n_acts = int(b.shape(input_activations)[0])
            density_weights_override = b.ones((n_acts,), dtype=precision_dtype(b, input_activations))
            logger.info(
                "FAST PATH: delta_scale=%.3f ≈ 1.0 (within sqrt_eps), using full-transfer density (skipping k-NN)",
                delta_scale,
            )

        # Compute coupling weight for this layer pair (from HOT soft coupling)
        coupling_weight_for_layer: float | None = None
        if (
            layer_coupling is not None
            and source_layers is not None
            and target_layers is not None
            and layer_mapping is not None
        ):
            src_layer = layer_mapping.get(layer_idx)
            if src_layer is not None and src_layer in source_layers and layer_idx in target_layers:
                src_idx = source_layers.index(src_layer)
                tgt_idx = target_layers.index(layer_idx)
                if src_idx < len(layer_coupling) and tgt_idx < len(layer_coupling[src_idx]):
                    coupling_weight_for_layer = layer_coupling[src_idx][tgt_idx]
                    logger.debug(
                        "TRANSPLANT: Layer %d coupling weight = %.4f (src=%d, tgt=%d)",
                        layer_idx,
                        coupling_weight_for_layer,
                        src_layer,
                        layer_idx,
                    )

        if behavior_jacobian_ctx is not None:
            G = behavior_jacobian_ctx.backend.compute_per_probe_gradients(
                model=behavior_jacobian_ctx.model,
                tokenizer=behavior_jacobian_ctx.tokenizer,
                probe_texts=behavior_jacobian_ctx.probe_texts,
                weight_name=key,
            )
            null_space_projector = compute_behavior_jacobian_projector(
                gradient_matrix=G,
                backend=b,
            )
            del G
        else:
            null_space_projector = compute_null_space_projector(
                input_activations=input_activations,
                source_activations_for_density=src_density_acts,
                target_activations_for_density=tgt_density_acts,
                density_weights=density_weights_override,
                coupling_weight=coupling_weight_for_layer,
                backend=b,
            )

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_w,
            input_activations=input_activations,
            source_activations_for_density=src_density_acts,
            target_activations_for_density=tgt_density_acts,
            null_space_projector=null_space_projector,
            delta_scale=delta_scale,
            backend=b,
        )

        logger.info(
            "TRANSPLANT: %s delta_norm=%.4f, preserved=%.1f%%, transfer=%.3f",
            key,
            result.delta_norm,
            100.0 * result.preserved_fraction,
            result.transfer_strength,
        )

        if result.preserved_fraction > 0:
            _record_manifest(
                key,
                WeightStatus.TRANSFORMED,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                stitch_type=activation_space,
                preserved_fraction=result.preserved_fraction,
            )
            final_merged_weight = result.merged_weight

            merged[key] = final_merged_weight
            transplanted_layer_keys.append(key)
            if mlp_role in ("gate", "up"):
                mlp_weights[mlp_role] = final_merged_weight
                merged_intermediate = None
            metrics["weights_transplanted"] += 1
            metrics["preserved_fractions"].append(result.preserved_fraction)
            projection_loss = max(0.0, 1.0 - result.preserved_fraction)
            metrics["projection_losses"].append(projection_loss)
            metrics["null_dims"].append(result.null_rank)

            weight_elapsed = time.time() - weight_start_time
            logger.debug(
                "TRANSPLANT: Weight %d/%d %s - %.2fs (preserved=%.3f, loss=%.6f)",
                weight_num + 1,
                len(ordered_layer_keys),
                key,
                weight_elapsed,
                result.preserved_fraction,
                projection_loss,
            )
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
            # OPTIMIZATION: Only compute boundary metrics for significant deltas.
            # Geodesic boundary computation is expensive (4 ops per weight).
            # Threshold: if (1 - preserved_fraction) < sqrt(eps), the delta is
            # indistinguishable from numerical noise and boundary metrics are meaningless.
            # For float32, sqrt(eps) ≈ 3e-4, so threshold ≈ 0.9997.
            sqrt_eps = sqrt_scalar(machine_epsilon(b, target_w), b)
            significance_threshold = 1.0 - sqrt_eps
            should_compute_boundary = (
                int(boundary_acts.shape[0]) > 0
                and result.preserved_fraction < significance_threshold
            )
            if should_compute_boundary:
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
                geo_diffs = geodesic_paired_distances(
                    merged_output, target_output, b, use_cache=False
                )
                origin = b.zeros_like(target_output)
                geo_target_norms = geodesic_paired_distances(
                    origin, target_output, b, use_cache=False
                )
                diff_norm_arr = geodesic_norms(
                    b.reshape(geo_diffs, (1, -1)), b, use_cache=False
                )
                target_norm_arr = geodesic_norms(
                    b.reshape(geo_target_norms, (1, -1)), b, use_cache=False
                )
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
        else:
            _record_manifest(
                key,
                WeightStatus.IDENTITY,
                source_shape=tuple(source_w.shape),
                target_shape=tuple(target_w.shape),
                stitch_type=activation_space,
                preserved_fraction=result.preserved_fraction,
            )

    # =========================================================================
    # JOINT MLP SCALE CORRECTION
    # =========================================================================
    # The SwiGLU MLP computes: output = down(SiLU(gate(x)) * up(x))
    #
    # When gate/up/down weights are reconstructed independently, each gets
    # its own scale_correction factor. But these factors COMPOSE at inference:
    # - gate scaled by s_g: SiLU(s_g * gate_out) is roughly (s_g * gate_out) / 2
    # - up scaled by s_u: multiplies directly
    # - down scaled by s_d: multiplies the composed result
    #
    # If s_g = 0.148 (gate 14.8x smaller) and s_d = 19.64 (down 1964x larger),
    # the composed function is distorted: tiny gate output gets amplified by down,
    # but since SiLU(small) ≈ small, you're amplifying essentially zero.
    #
    # The fix: compute a JOINT scale correction that distributes evenly across
    # all three weights, preserving the geometric mean while correcting the
    # individual extreme values.
    # =========================================================================
    if len(mlp_scales) >= 2:
        # Get scale factors (default to 1.0 if missing)
        scale_gate = mlp_scales.get("gate", 1.0)
        scale_up = mlp_scales.get("up", 1.0)
        scale_down = mlp_scales.get("down", 1.0)

        # Check if scales are divergent enough to need correction
        scales = [scale_gate, scale_up, scale_down]
        max_scale = max(scales)
        min_scale = min(scales)
        scale_divergence = max_scale / max(min_scale, 1e-10)

        if scale_divergence > 2.0:
            # Scales are divergent - the behavioral reconstruction has extreme scale factors
            # Check if we should try joint MLP scale correction (experimental)
            use_joint_scale = os.environ.get("MC_USE_JOINT_MLP_SCALE", "").lower() in ("1", "true", "yes")

            if use_joint_scale:
                # EXPERIMENTAL: Apply joint MLP scale correction
                # This computes a geometric mean scale and distributes it evenly
                logger.info(
                    "SCALE DIVERGENCE DETECTED (%.2f > 2.0): gate=%.4f, up=%.4f, down=%.4f. "
                    "Applying JOINT MLP SCALE correction (MC_USE_JOINT_MLP_SCALE=1).",
                    scale_divergence, scale_gate, scale_up, scale_down
                )

                # Compute joint scale corrections
                corr_gate, corr_up, corr_down = compute_joint_mlp_scale(
                    scale_gate, scale_up, scale_down
                )

                # Apply corrections to the already-reconstructed weights
                for weight_key in transplanted_layer_keys:
                    if weight_key not in merged:
                        continue
                    if "feed_forward.w1" in weight_key or "gate_proj" in weight_key:
                        merged[weight_key] = merged[weight_key] * corr_gate
                        logger.info("JOINT SCALE: %s *= %.4f (gate correction)", weight_key, corr_gate)
                    elif "feed_forward.w3" in weight_key or "up_proj" in weight_key:
                        merged[weight_key] = merged[weight_key] * corr_up
                        logger.info("JOINT SCALE: %s *= %.4f (up correction)", weight_key, corr_up)
                    elif "feed_forward.w2" in weight_key or "down_proj" in weight_key:
                        merged[weight_key] = merged[weight_key] * corr_down
                        logger.info("JOINT SCALE: %s *= %.4f (down correction)", weight_key, corr_down)

                metrics.setdefault("joint_mlp_correction_applied", True)
                metrics.setdefault("mlp_reverted_to_target", False)
                metrics.setdefault("layer_reverted_to_target", False)
                metrics.setdefault("joint_mlp_scale_divergence", scale_divergence)
                metrics.setdefault("joint_scale_corrections", {
                    "gate": corr_gate, "up": corr_up, "down": corr_down
                })
            else:
                # DEFAULT: Revert to target (conservative, prevents garbage output)
                logger.warning(
                    "SCALE DIVERGENCE DETECTED (%.2f > 2.0): gate=%.4f, up=%.4f, down=%.4f. "
                    "Reverting ENTIRE LAYER %d weights to target (reconstruction unstable). "
                    "Set MC_USE_JOINT_MLP_SCALE=1 to try joint scaling instead.",
                    scale_divergence, scale_gate, scale_up, scale_down, layer_idx
                )

                reverted_keys: list[str] = []
                # Revert ALL transplanted weights in this layer, not just MLP
                for weight_key in transplanted_layer_keys:
                    if weight_key in target_weights:
                        merged[weight_key] = target_weights[weight_key]
                        reverted_keys.append(weight_key)
                        logger.info("REVERTED: %s to target weight", weight_key)

                metrics.setdefault("joint_mlp_correction_applied", False)
                metrics.setdefault("mlp_reverted_to_target", True)
                metrics.setdefault("layer_reverted_to_target", True)
                metrics.setdefault("joint_mlp_scale_divergence", scale_divergence)
                # Track which keys were reverted so compression descent can skip them
                existing_reverted = metrics.get("mlp_reverted_keys", [])
                metrics["mlp_reverted_keys"] = existing_reverted + reverted_keys
        else:
            logger.debug(
                "JOINT MLP CORRECTION: skipped (divergence=%.2f < 2.0)",
                scale_divergence
            )

    return LayerWeightResult(
        weights_processed=weights_processed,
        layer_transplanted=layer_transplanted,
        best_alignment=best_alignment,
        best_delta_norm=best_delta_norm,
    )
