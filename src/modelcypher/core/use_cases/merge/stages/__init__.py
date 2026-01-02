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

"""
Merge pipeline stages.

Each stage is a standalone module that can be imported and tested independently.
The UnifiedGeometricMerger orchestrates these stages in sequence.

Pipeline: PROBE → DENSITY → PERMUTE → TRANSPLANT

Stage 1: PROBE - Build intersection map from probe responses
Stage 2a: DENSITY - Knowledge density profiling for graft mask
Stage 2b: PERMUTE - Git Re-Basin permutation alignment for MLP neurons (same-arch)
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting

REMOVED (proven broken):
- ROTATE/PROPAGATE: No mathematical guarantee of boundary preservation.

References:
- Git Re-Basin: Ainsworth et al. (2023) arXiv:2209.04836
- AlphaEdit (null-space transplant): Fang et al. (2025) ICLR Outstanding Paper
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .probe import (
    ProbeResult,
    collect_layer_activations_mlx,
    stage_probe as stage_probe_impl,
)
from .density import (
    DensityStageResult,
    stage_density as stage_density_impl,
)
from .permute import (
    PermuteResult,
    infer_hidden_dim,
    stage_permute as stage_permute_impl,
)
# NOTE: ProbeConfig and PermuteConfig were REMOVED.
# Probe always uses precise mode with all probes.
# Permute always runs (no enable_permutation toggle).
from .transplant import (
    TransplantStageConfig,
    TransplantStageResult,
    stage_transplant as stage_transplant_impl,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

def stage_probe(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    source_model: Any | None,
    target_model: Any | None,
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
    extract_layer_index_fn: Callable[[str], int | None],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict[int, list[list[float]]] | None,  # feature_transforms (hidden)
    dict[int, list[list[float]]] | None,  # attention_transforms (Q)
    dict[int, list[list[float]]] | None,  # kv_transforms (KV)
    dict[int, int] | None,  # layer_mapping
]:
    """Stage 1: Compute layer correspondences via CKA."""
    collect_fn = (
        collect_layer_activations_mlx
        if source_model is not None and source_tokenizer and target_tokenizer
        else None
    )

    result = stage_probe_impl(
        source_weights=source_weights,
        target_weights=target_weights,
        extract_layer_index_fn=extract_layer_index_fn,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        collect_activations_fn=collect_fn,
    )

    return (
        {
            "correlations": result.correlations,
            "confidences": result.confidences,
            "dimension_correlations": result.dimension_correlations,
            "intersection_map": result.intersection_map,
            "probe_ids": result.probe_ids,
            "probe_domains": result.probe_domains,
        },
        result.metrics,
        result.source_activations,
        result.target_activations,
        result.source_intermediate_activations,
        result.target_intermediate_activations,
        result.source_attention_activations,
        result.target_attention_activations,
        result.source_kv_activations,
        result.target_kv_activations,
        result.feature_transforms,
        result.attention_transforms,
        result.kv_transforms,
        result.layer_mapping,
    )


def stage_density(
    *,
    source_activations: dict | None,
    target_activations: dict | None,
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    layers: list[int],
    backend: "Backend",
) -> tuple[dict[str, dict[int, bool]], dict[str, Any]]:
    """Stage 2: Density analysis for selective grafting."""
    result = stage_density_impl(
        source_activations=source_activations or {},
        target_activations=target_activations or {},
        probe_ids=probe_ids or [],
        probe_domains=probe_domains or [],
        layers=layers,
        backend=backend,
    )

    return result.graft_mask, result.metrics


def stage_permute(
    *,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    backend: "Backend",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage 2: Git Re-Basin permutation alignment."""
    result = stage_permute_impl(
        source_weights=source_weights,
        target_weights=target_weights,
        intersection_map_obj=intersection_map_obj,
        layer_confidences=layer_confidences,
        infer_hidden_dim_fn=infer_hidden_dim,
        backend=backend,
    )

    return result.weights, result.metrics


def stage_transplant(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    layer_indices: list[int],
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    source_activations: dict | None,
    target_activations: dict | None,
    source_intermediate_activations: dict | None,
    target_intermediate_activations: dict | None,
    source_attention_activations: dict | None,
    target_attention_activations: dict | None,
    source_kv_activations: dict | None = None,
    target_kv_activations: dict | None = None,
    transplant_domains: tuple[str, ...] = (),
    extract_layer_index_fn: Callable[[str], int | None] = lambda x: None,
    backend: "Backend | None" = None,
    graft_mask: dict[str, dict[int, bool]] | None = None,
    feature_transforms: dict[int, list[list[float]]] | None = None,
    attention_transforms: dict[int, list[list[float]]] | None = None,
    kv_transforms: dict[int, list[list[float]]] | None = None,
    layer_mapping: dict[int, int] | None = None,
) -> tuple[dict[str, "Array"], dict[str, Any]]:
    """Stage 3: Null-space constrained transplant."""
    stage_config = TransplantStageConfig(
        core_domains=tuple(transplant_domains),
        graft_mask=graft_mask,
        feature_transforms=feature_transforms,
        attention_transforms=attention_transforms,
        kv_transforms=kv_transforms,
        layer_mapping=layer_mapping,
    )

    result = stage_transplant_impl(
        source_weights=source_weights,
        target_weights=target_weights,
        layer_indices=layer_indices,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        source_activations=source_activations,
        target_activations=target_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_attention_activations=source_attention_activations,
        target_attention_activations=target_attention_activations,
        source_kv_activations=source_kv_activations,
        target_kv_activations=target_kv_activations,
        config=stage_config,
        extract_layer_index_fn=extract_layer_index_fn,
        backend=backend,
    )

    return result.merged_weights, result.metrics

__all__ = [
    # Stage 1: Probe (ProbeConfig REMOVED - always precise mode, all probes)
    "stage_probe",
    "ProbeResult",
    "collect_layer_activations_mlx",
    # Stage 2a: Density
    "stage_density",
    "DensityStageResult",
    # Stage 2b: Permute (PermuteConfig REMOVED - always runs)
    "stage_permute",
    "PermuteResult",
    "infer_hidden_dim",
    # Stage 3: Transplant (simplified - only core_domains and graft_mask)
    "stage_transplant",
    "TransplantStageConfig",
    "TransplantStageResult",
]
