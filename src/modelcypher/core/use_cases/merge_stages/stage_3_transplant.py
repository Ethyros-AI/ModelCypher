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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_dimensional_projection import (
    ProjectionMethod,
    project_cross_dimensional,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.transplant import (
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransplantStageConfig:
    """Configuration for transplant stage."""

    core_domains: tuple[str, ...]
    boundary_k: int | None = None
    geodesic_k_neighbors: int | None = None
    projection_method: ProjectionMethod = ProjectionMethod.GRAM_TRANSPORT
    transplant_layers: tuple[int, ...] | None = None  # None = all layers


@dataclass
class TransplantStageResult:
    """Result of transplant stage."""

    merged_weights: dict[str, Any]
    metrics: dict[str, Any]


def _normalize_domains(domains: Iterable[str]) -> set[str]:
    return {d.strip().lower() for d in domains if d.strip()}


def stage_transplant(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    layer_indices: list[int],
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    target_activations: dict[int, list["Array"]] | None,
    config: TransplantStageConfig,
    extract_layer_index_fn: Callable[[str], int | None],
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
        "core_probes": 0,
        "boundary_k": config.boundary_k,
        "geodesic_k_neighbors": config.geodesic_k_neighbors,
    }

    if not probe_ids or not probe_domains:
        metrics["transplant_skipped"] = "missing_probe_metadata"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    if len(probe_ids) != len(probe_domains):
        metrics["transplant_skipped"] = "probe_metadata_mismatch"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    if not target_activations:
        metrics["transplant_skipped"] = "missing_activations"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    core_domains = _normalize_domains(config.core_domains)
    core_probe_ids = {
        probe_id
        for probe_id, domain in zip(probe_ids, probe_domains)
        if domain and domain.lower() in core_domains
    }

    metrics["core_probes"] = len(core_probe_ids)
    if not core_probe_ids:
        metrics["transplant_skipped"] = "no_core_probes"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    weights_by_layer: dict[int, list[str]] = {}
    for key in target_weights:
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is None:
            continue
        weights_by_layer.setdefault(layer_idx, []).append(key)

    for layer_idx in layer_indices:
        # Filter to specific layers if configured
        if config.transplant_layers is not None:
            if layer_idx not in config.transplant_layers:
                continue

        act_list = target_activations.get(layer_idx)
        if not act_list:
            continue

        if len(act_list) != len(probe_ids):
            logger.debug(
                "LAYER %d: probe count mismatch (acts=%d, probes=%d); skipping",
                layer_idx,
                len(act_list),
                len(probe_ids),
            )
            continue

        metrics["layers_considered"] += 1

        stacked = b.stack(act_list, axis=0)
        b.eval(stacked)

        partition = partition_core_boundary(
            activations=stacked,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            boundary_k=config.boundary_k,
            geodesic_k_neighbors=config.geodesic_k_neighbors,
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

        layer_keys = weights_by_layer.get(layer_idx, [])
        if not layer_keys:
            continue

        layer_transplanted = False

        for key in layer_keys:
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

            if target_w.shape != source_w.shape:
                try:
                    logger.debug("Projecting %s: source=%s -> target=%s", key, source_w.shape, target_w.shape)
                    projection = project_cross_dimensional(
                        source=source_w,
                        target=target_w,
                        method=config.projection_method,
                        backend=b,
                    )
                    source_aligned = projection.projected
                    logger.debug("Projected %s successfully", key)
                except Exception as e:
                    logger.warning("Failed to project weight %s: %s", key, e)
                    continue
            else:
                source_aligned = source_w

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
                merged[key] = result.merged_weight
                metrics["weights_transplanted"] += 1
                metrics["preserved_fractions"].append(result.preserved_fraction)
                metrics["projection_losses"].append(result.projection_loss)
                metrics["null_dims"].append(result.null_dim)
                if int(boundary_acts.shape[0]) > 0:
                    target_output = b.matmul(boundary_acts, b.transpose(target_w))
                    merged_output = b.matmul(
                        boundary_acts, b.transpose(result.merged_weight)
                    )
                    diff = merged_output - target_output
                    diff_norm_arr = b.norm(b.reshape(diff, (-1,)))
                    target_norm_arr = b.norm(b.reshape(target_output, (-1,)))
                    b.eval(diff_norm_arr, target_norm_arr)

                    diff_norm = float(b.to_numpy(diff_norm_arr))
                    target_norm = float(b.to_numpy(target_norm_arr))
                    eps = float(machine_epsilon(b, target_w))

                    if target_norm > eps:
                        relative_diff = diff_norm / target_norm
                    else:
                        relative_diff = 0.0 if diff_norm <= eps else float("inf")

                    metrics["boundary_relative_diffs"].append(relative_diff)
                layer_transplanted = True

        if layer_transplanted:
            metrics["layers_transplanted"] += 1

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
    "TransplantStageConfig",
    "TransplantStageResult",
    "stage_transplant",
]
