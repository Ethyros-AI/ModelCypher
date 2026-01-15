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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_paired_distances,
)
from modelcypher.core.domain.geometry.transplant import (
    NullSpaceProjector,
    compute_null_space_projector,
    compute_weight_space_transplant,
)
from modelcypher.core.domain.merging.exceptions import (
    DimensionMismatchError,
    StitchUnavailableError,
)
from modelcypher.core.use_cases.merge.stages.transplant_helpers import (
    _compute_dimension_projection,
    _promote_precision,
    _set_submatrix,
)
from modelcypher.core.use_cases.merge.stages.transplant_mapping import (
    _map_weight_key_cross_arch,
)
from modelcypher.core.use_cases.merge.stages.transplant_metrics import (
    _compute_alignment_metrics,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

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


@dataclass
class LayerWeightResult:
    weights_processed: int
    layer_transplanted: bool
    best_alignment: dict[str, float] | None
    best_delta_norm: float


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
    hidden_stitch_output: "Array | None",
    hidden_stitch_input: "Array | None",
    intermediate_stitch_output: "Array | None",
    intermediate_stitch_input: "Array | None",
    gate_stitch_output: "Array | None" = None,  # PRE-SiLU for gate/up cross-arch
    gate_stitch_input: "Array | None" = None,   # PRE-SiLU for gate/up cross-arch
    attention_stitch_output: "Array | None" = None,
    attention_stitch_input: "Array | None" = None,
    k_stitch_output: "Array | None" = None,
    k_stitch_input: "Array | None" = None,
    v_stitch_output: "Array | None" = None,
    v_stitch_input: "Array | None" = None,
    kv_stitch_input: "Array | None" = None,
    stitch_dims: dict[str, int] | None = None,
    source_activations: dict[int, list["Array"]] | None = None,
    target_activations: dict[int, list["Array"]] | None = None,
    source_intermediate_activations: dict[int, list["Array"]] | None = None,
    target_intermediate_activations: dict[int, list["Array"]] | None = None,
    density_weights_by_layer: dict[int, "Array"] | None = None,
    core_acts: "Array",
    boundary_acts: "Array",
    can_measure_alignment: bool,
    delta_scale: float = 1.0,
) -> LayerWeightResult:
    b = backend
    layer_transplanted = False
    best_alignment: dict[str, float] | None = None
    best_delta_norm = -1.0
    mlp_weights: dict[str, "Array"] = {}
    merged_intermediate: "Array | None" = None
    null_space_cache: dict[str, NullSpaceProjector] = {}

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

        tgt_hidden = target_activations[layer_idx]
        if tgt_hidden is None:
            return None

        if hasattr(tgt_hidden, "shape") and len(b.shape(tgt_hidden)) == 2:
            hidden = _promote_precision(b.array(tgt_hidden), b)
        elif hasattr(tgt_hidden, "__len__") and len(tgt_hidden) > 0:
            hidden = b.stack([b.array(a) for a in tgt_hidden], axis=0)
            hidden = _promote_precision(hidden, b)
        else:
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

    for weight_num, key in enumerate(ordered_layer_keys):
        weights_processed += 1
        weight_start_time = time.time()
        mlp_role = _mlp_role(key)

        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if not key.endswith(".weight"):
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
            if source_w is None and "conv.conv" not in key:
                metrics.setdefault("unmapped_weights", [])
                if len(metrics["unmapped_weights"]) < 20:
                    metrics["unmapped_weights"].append(key)
            continue

        metrics["weights_considered"] += 1

        if not hasattr(target_w, "shape") or not hasattr(source_w, "shape"):
            continue
        ndim_t = len(target_w.shape)
        ndim_s = len(source_w.shape)
        if ndim_t != 2 or ndim_s != 2:
            metrics.setdefault("weights_skipped_non_2d", 0)
            metrics["weights_skipped_non_2d"] += 1
            continue

        target_dtype = str(getattr(target_w, "dtype", "")).lower()
        source_dtype = str(getattr(source_w, "dtype", "")).lower()

        if "int" in target_dtype or "uint" in target_dtype:
            logger.debug("Dequantizing target weight: %s", key)
            target_w = dequantize_if_needed(target_w, key, target_weights, b)
            if target_w is None or not hasattr(target_w, "shape"):
                logger.debug("Failed to dequantize target weight: %s", key)
                continue
            target_dtype = str(getattr(target_w, "dtype", "")).lower()
            if "int" in target_dtype or "uint" in target_dtype:
                logger.debug("Skipping still-quantized target weight: %s", key)
                continue

        if "int" in source_dtype or "uint" in source_dtype:
            logger.debug("Dequantizing source weight: %s (source_key: %s)", key, source_key)
            source_w = dequantize_if_needed(source_w, source_key, source_weights, b)
            if source_w is None or not hasattr(source_w, "shape"):
                logger.debug("Failed to dequantize source weight: %s", key)
                continue
            source_dtype = str(getattr(source_w, "dtype", "")).lower()
            if "int" in source_dtype or "uint" in source_dtype:
                logger.debug("Skipping still-quantized source weight: %s", key)
                continue

        if len(source_w.shape) == 1 and len(target_w.shape) == 1:
            src_dim = int(source_w.shape[0])
            tgt_dim = int(target_w.shape[0])

            if "layernorm" in key.lower() or "norm" in key.lower():
                metrics.setdefault("norm_weights_preserved", 0)
                metrics["norm_weights_preserved"] += 1
                logger.debug("PRESERVING target LayerNorm: %s (scale adapter)", key)
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
                continue
            if src_dim == tgt_dim:
                merged[key] = source_w
                metrics["weights_transplanted"] += 1
                continue
            continue

        if len(target_w.shape) != 2 or len(source_w.shape) != 2:
            continue

        try:
            logger.debug("Converting weight %s for linalg", key)
            target_w = _promote_precision(b.array(target_w), b)
            source_w = _promote_precision(b.array(source_w), b)
            b.eval(target_w, source_w)
            logger.debug("Converted %s: target=%s, source=%s", key, target_w.shape, source_w.shape)
        except Exception as e:
            logger.warning("Failed to convert weight %s: %s", key, e)
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
                        # Build a block-diagonal output stitch: replicate hidden_stitch for each factor
                        # For 3× expansion: [[H, 0, 0], [0, H, 0], [0, 0, H]] where H is hidden_stitch_output
                        conv_output_stitch = b.zeros((conv_factor * tgt_hidden_dim, conv_factor * src_hidden_dim))
                        b.eval(conv_output_stitch)
                        for i in range(conv_factor):
                            row_start = i * tgt_hidden_dim
                            row_end = (i + 1) * tgt_hidden_dim
                            col_start = i * src_hidden_dim
                            col_end = (i + 1) * src_hidden_dim
                            # Build full matrix with block diagonal
                            block_rows = []
                            for bi in range(conv_factor):
                                if bi == i:
                                    block_rows.append(hidden_stitch_output)
                                else:
                                    block_rows.append(b.zeros((tgt_hidden_dim, src_hidden_dim)))
                            conv_output_stitch = b.concatenate(
                                [b.concatenate([block_rows[bi] for bi in range(conv_factor)], axis=1)
                                 if j == 0 else b.zeros((tgt_hidden_dim, conv_factor * src_hidden_dim))
                                 for j in range(1)],
                                axis=0
                            ) if conv_factor == 1 else b.concatenate([
                                b.concatenate([
                                    hidden_stitch_output if bi == bj else b.zeros((tgt_hidden_dim, src_hidden_dim))
                                    for bj in range(conv_factor)
                                ], axis=1)
                                for bi in range(conv_factor)
                            ], axis=0)
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
                    # gives wrong results for PRE-SiLU weight outputs.
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
                        # Fallback: try compositional stitch (legacy approach)
                        try:
                            aligner = GramAligner(backend=b)
                            H = b.transpose(hidden_stitch_output)
                            target_w_float = _promote_precision(b.array(target_w), b)
                            b.eval(H, target_w_float)
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
                            logger.warning(
                                "MLP compositional stitch failed for %s: %s",
                                key,
                                e,
                            )
                            if gate_stitch_output is not None:
                                output_stitch = gate_stitch_output
                            else:
                                output_stitch = intermediate_stitch_output

                    if output_stitch is None:
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
                    # While intermediate_stitch_input (Procrustes on POST-SiLU) is
                    # theoretically correct, it causes consistency issues:
                    # - down_proj aligned using TARGET intermediates
                    # - but at inference, receives MERGED intermediates (from gate/up)
                    # - the 2% difference in gate/up + SiLU nonlinearity = divergence
                    #
                    # Compositional stitch derives input from hidden+weights,
                    # which is more consistent with the merged gate/up.
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
                    # Fallback: intermediate stitch from probe
                    if input_stitch is None:
                        input_stitch = intermediate_stitch_input
                        if input_stitch is not None:
                            logger.info(
                                "MLP down_proj using intermediate_stitch fallback for %s",
                                key,
                            )

                    if input_stitch is None:
                        raise StitchUnavailableError(
                            stage="MLP_DOWN_STITCH",
                            weight_key=key,
                            message="Missing intermediate stitch for down projection",
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
                    metrics.setdefault("attention_preserved", 0)
                    metrics["attention_preserved"] += 1
                    continue

                if is_attention and attention_stitch_output is not None:
                    src_attn_dim = stitch_dims["src_attn"]
                    tgt_attn_dim = stitch_dims["tgt_attn"]
                    src_hidden_dim = stitch_dims["src_hidden"]
                    tgt_hidden_dim = stitch_dims["tgt_hidden"]
                    dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                    src_kv_dim = stitch_dims.get("src_kv", src_attn_dim)
                    tgt_kv_dim = stitch_dims.get("tgt_kv", tgt_attn_dim)

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

        tgt_hidden_dim = stitch_dims.get("tgt_hidden", 0) if stitch_dims else 0
        tgt_inter_dim = stitch_dims.get("tgt_inter", 0) if stitch_dims else 0
        weight_in_dim = int(target_w.shape[1])

        input_activations = None
        src_density_acts = None
        tgt_density_acts = None
        activation_space = "hidden"
        merged_intermediate_used = False

        # For cross-architecture, use layer_mapping to find the correct source layer
        mapped_src_layer = layer_mapping.get(layer_idx, layer_idx) if layer_mapping else layer_idx

        if weight_in_dim == tgt_inter_dim and tgt_inter_dim > 0:
            activation_space = "intermediate"

            if mlp_role == "down":
                merged_input = _get_merged_intermediate()
                if merged_input is not None:
                    input_activations = merged_input
                    tgt_density_acts = merged_input
                    merged_intermediate_used = True

            # First, get target intermediate activations (needed for null-space projection)
            if target_intermediate_activations is None:
                logger.warning(
                    "INTERMEDIATE MISS: target_intermediate_activations is None for %s layer=%d",
                    key, layer_idx
                )
            elif layer_idx not in target_intermediate_activations:
                logger.warning(
                    "INTERMEDIATE MISS: Layer %d not in target_intermediate_activations for %s (keys=%s)",
                    layer_idx, key, list(target_intermediate_activations.keys())[:10]
                )

            if (
                target_intermediate_activations is not None
                and layer_idx in target_intermediate_activations
                and input_activations is None
            ):
                tgt_inter = target_intermediate_activations[layer_idx]
                if tgt_inter is not None:
                    if hasattr(tgt_inter, "shape") and len(b.shape(tgt_inter)) == 2:
                        input_activations = _promote_precision(b.array(tgt_inter), b)
                    elif hasattr(tgt_inter, "__len__") and len(tgt_inter) > 0:
                        input_activations = b.stack([b.array(a) for a in tgt_inter], axis=0)
                    tgt_density_acts = input_activations

            # Then, get source intermediate activations for density comparison (optional)
            if (
                input_activations is not None
                and source_intermediate_activations is not None
                and mapped_src_layer in source_intermediate_activations
            ):
                src_inter = source_intermediate_activations[mapped_src_layer]
                if src_inter is not None:
                    if hasattr(src_inter, "shape") and len(b.shape(src_inter)) == 2:
                        src_density_acts = _promote_precision(b.array(src_inter), b)
                    elif hasattr(src_inter, "__len__") and len(src_inter) > 0:
                        src_density_acts = b.stack([b.array(a) for a in src_inter], axis=0)
                    logger.debug(
                        "INTERMEDIATE: Layer %d using mapped source layer %d for density",
                        layer_idx, mapped_src_layer
                    )

        if input_activations is None:
            activation_space = "hidden"
            if target_activations is not None and layer_idx in target_activations:
                tgt_hidden = target_activations[layer_idx]
                if tgt_hidden is not None:
                    if hasattr(tgt_hidden, "shape") and len(b.shape(tgt_hidden)) == 2:
                        input_activations = _promote_precision(b.array(tgt_hidden), b)
                    elif hasattr(tgt_hidden, "__len__") and len(tgt_hidden) > 0:
                        input_activations = b.stack([b.array(a) for a in tgt_hidden], axis=0)
                    tgt_density_acts = input_activations

                    if density_weights_by_layer is None or layer_idx not in density_weights_by_layer:
                        # Use mapped source layer for density comparison
                        if source_activations is not None and mapped_src_layer in source_activations:
                            src_hidden = source_activations[mapped_src_layer]
                            if src_hidden is not None:
                                if hasattr(src_hidden, "shape") and len(b.shape(src_hidden)) == 2:
                                    src_density_acts = _promote_precision(b.array(src_hidden), b)
                                elif hasattr(src_hidden, "__len__") and len(src_hidden) > 0:
                                    src_density_acts = b.stack([b.array(a) for a in src_hidden], axis=0)
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
            continue

        # Align density activations to target space
        if activation_space == "hidden":
            src_density_acts = _align_density_acts(src_density_acts, hidden_stitch_output)
        elif activation_space == "intermediate":
            src_density_acts = _align_density_acts(src_density_acts, intermediate_stitch_output)

        b.eval(input_activations)
        if src_density_acts is not None:
            b.eval(src_density_acts)
        if tgt_density_acts is not None:
            b.eval(tgt_density_acts)

        # Verify activation dimension matches weight input dimension
        # This should NOT trigger after the fix - if it does, something else is wrong
        input_act_dim = int(b.shape(input_activations)[1])
        if input_act_dim != weight_in_dim:
            logger.debug(
                "TRANSPLANT: Activation dimension mismatch for %s: got %d, expected %d "
                "(space=%s, layer=%d, mapped_src=%d). Preserving target.",
                key, input_act_dim, weight_in_dim, activation_space, layer_idx, mapped_src_layer
            )
            metrics.setdefault("activation_dim_mismatch", 0)
            metrics["activation_dim_mismatch"] += 1
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
        cache_key = None
        use_cache = False
        if activation_space == "hidden":
            use_cache = True
            cache_key = "hidden"
            if density_weights_by_layer is not None:
                density_weights_override = density_weights_by_layer.get(layer_idx)
        elif activation_space == "intermediate":
            use_cache = not merged_intermediate_used
            cache_key = "intermediate" if use_cache else None

        null_space_projector = None
        if use_cache and cache_key in null_space_cache:
            null_space_projector = null_space_cache[cache_key]
        else:
            null_space_projector = compute_null_space_projector(
                input_activations=input_activations,
                density_weights=density_weights_override,
                source_activations_for_density=None
                if density_weights_override is not None
                else src_density_acts,
                target_activations_for_density=None
                if density_weights_override is not None
                else tgt_density_acts,
                backend=b,
            )
            if use_cache and cache_key:
                null_space_cache[cache_key] = null_space_projector

        # Null-space projection IS the geometry. No additional filtering.
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
            final_merged_weight = result.merged_weight

            merged[key] = final_merged_weight
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

    return LayerWeightResult(
        weights_processed=weights_processed,
        layer_transplanted=layer_transplanted,
        best_alignment=best_alignment,
        best_delta_norm=best_delta_norm,
    )
