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

Pipeline: PROBE → DENSITY → TRANSPLANT

Stage 1: PROBE - Compute GramAlign transforms from probe responses
Stage 2: DENSITY - Knowledge density profiling for graft mask
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting

References:
- AlphaEdit (null-space transplant): Fang et al. (2025) ICLR Outstanding Paper
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .probe import (
    ProbeResult,
    stage_probe as stage_probe_impl,
)
from .density import (
    DensityStageResult,
    stage_density as stage_density_impl,
)
# NOTE: ProbeConfig was REMOVED - Probe always uses the precise path with all probes.
# PERMUTE STAGE REMOVED: GramAligner alignment subsumes permutation.
from .transplant import (
    TransplantStageResult,
    stage_transplant as stage_transplant_impl,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

def stage_probe(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    source_model: Any | None,
    target_model: Any | None,
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
    source_path: str = "",
    target_path: str = "",
    extract_layer_index_fn: Callable[[str], int | None],
    activation_provider: "ActivationProvider | None" = None,
    backend: "Backend | None" = None,
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
    dict[int, float] | None,  # scale_ratios (EXACT magnitude factors)
    list[list[float]] | None,  # embedding_transform (2D GramAlign)
    dict[int, list[list[float]]] | None,  # attention_transforms (Q)
    dict[int, list[list[float]]] | None,  # k_transforms (K)
    dict[int, list[list[float]]] | None,  # v_transforms (V)
    dict[int, list[list[float]]] | None,  # intermediate_transforms (MLP)
    dict[int, list[list[float]]] | None,  # gate_transforms (PRE-SiLU)
    dict[int, int] | None,  # layer_mapping
    list["Array"] | "Array" | None,  # source_embedding_activations
    list["Array"] | "Array" | None,  # target_embedding_activations
    dict[int, Any] | None,  # source_trajectory_tangents
    dict[int, Any] | None,  # target_trajectory_tangents
    list[list[float]] | None,  # layer_coupling (HOT soft coupling)
    list[int] | None,  # source_layers
    list[int] | None,  # target_layers
]:
    """Stage 1: Compute layer correspondences via CKA."""
    result = stage_probe_impl(
        source_weights=source_weights,
        target_weights=target_weights,
        extract_layer_index_fn=extract_layer_index_fn,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_path=source_path,
        target_path=target_path,
        activation_provider=activation_provider,
        backend=backend,
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
        result.source_k_activations,
        result.target_k_activations,
        result.feature_transforms,
        result.scale_ratios,  # EXACT: ||target|| / ||source @ F||
        result.embedding_transform,
        result.attention_transforms,
        result.k_transforms,
        result.v_transforms,
        result.intermediate_transforms,  # MLP transforms
        result.gate_transforms,  # PRE-SiLU gate transforms
        result.layer_mapping,
        result.source_embedding_activations,
        result.target_embedding_activations,
        result.source_trajectory_tangents,  # Trajectory-tangent results for transplant
        result.target_trajectory_tangents,  # Trajectory-tangent results for transplant
        result.layer_coupling,  # HOT soft coupling for transfer strength weighting
        result.source_layers,  # Sorted source layer indices for coupling lookup
        result.target_layers,  # Sorted target layer indices for coupling lookup
    )


def stage_density(
    *,
    source_activations: dict | None,
    target_activations: dict | None,
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    layers: list[int],
    feature_transforms: dict[int, Any] | None = None,
    layer_mapping: dict[int, int] | None = None,
    backend: "Backend",
) -> "DensityStageResult":
    """Stage 2: Density analysis for selective grafting.

    Returns DensityStageResult with:
    - graft_mask: Dict mapping domains/layers to bool
    - density_weights: Dict[int, Array] with per-layer source density ratios per probe
    - metrics: Performance metrics
    """
    return stage_density_impl(
        source_activations=source_activations or {},
        target_activations=target_activations or {},
        probe_ids=probe_ids or [],
        probe_domains=probe_domains or [],
        layers=layers,
        feature_transforms=feature_transforms,
        layer_mapping=layer_mapping,
        backend=backend,
    )




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
    source_embedding_activations: list["Array"] | "Array" | None = None,
    target_embedding_activations: list["Array"] | "Array" | None = None,
    extract_layer_index_fn: Callable[[str], int | None] = lambda x: None,
    backend: "Backend | None" = None,
    graft_mask: dict[str, dict[int, bool]] | None = None,
    density_weights: dict[int, "Array"] | None = None,  # Per-probe source density ratios
    feature_transforms: dict[int, list[list[float]]] | None = None,
    scale_ratios: dict[int, float] | None = None,  # EXACT: ||target|| / ||source @ F||
    embedding_transform: list[list[float]] | None = None,  # 2D GramAlign
    attention_transforms: dict[int, list[list[float]]] | None = None,
    k_transforms: dict[int, list[list[float]]] | None = None,
    v_transforms: dict[int, list[list[float]]] | None = None,
    intermediate_transforms: dict[int, list[list[float]]] | None = None,  # MLP transforms
    gate_transforms: dict[int, list[list[float]]] | None = None,  # PRE-SiLU gate transforms
    layer_mapping: dict[int, int] | None = None,
    layer_status: dict[int, str] | None = None,  # NEW: Per DIMENSIONAL_COMPRESSION.md
    prior_occupancy_by_layer: dict[int, list[float]] | None = None,
    source_tokenizer: Any | None = None,  # For token correspondence
    target_tokenizer: Any | None = None,  # For token correspondence
    delta_scale: float = 1.0,  # Delta budget control for sequential stacking
    layer_profile: Any | None = None,  # LayerSemanticProfile from helpers
    is_cross_vocab: bool = False,  # True if source/target have different vocabularies
    source_trajectory_tangents: dict[int, Any] | None = None,  # Trajectory-tangent results
    target_trajectory_tangents: dict[int, Any] | None = None,  # Trajectory-tangent results
    layer_coupling: list[list[float]] | None = None,  # HOT soft coupling
    source_layers: list[int] | None = None,  # Sorted source layer indices
    target_layers: list[int] | None = None,  # Sorted target layer indices
) -> tuple[dict[str, "Array"], dict[str, Any]]:
    """Stage 3: Null-space constrained transplant."""
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
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
        extract_layer_index_fn=extract_layer_index_fn,
        backend=backend,
        graft_mask=graft_mask,
        density_weights=density_weights,  # Per-probe transfer weights from k-NN density
        feature_transforms=feature_transforms,
        scale_ratios=scale_ratios,  # EXACT magnitude factors
        embedding_transform=embedding_transform,
        attention_transforms=attention_transforms,
        k_transforms=k_transforms,
        v_transforms=v_transforms,
        intermediate_transforms=intermediate_transforms,  # MLP transforms
        gate_transforms=gate_transforms,  # PRE-SiLU gate transforms
        layer_mapping=layer_mapping,
        layer_status=layer_status,  # NEW: Per DIMENSIONAL_COMPRESSION.md
        prior_occupancy_by_layer=prior_occupancy_by_layer,
        source_tokenizer=source_tokenizer,  # For token correspondence
        target_tokenizer=target_tokenizer,  # For token correspondence
        delta_scale=delta_scale,
        layer_profile=layer_profile,  # LayerSemanticProfile
        is_cross_vocab=is_cross_vocab,  # Cross-vocab merge flag
        source_trajectory_tangents=source_trajectory_tangents,
        target_trajectory_tangents=target_trajectory_tangents,
        layer_coupling=layer_coupling,
        source_layers=source_layers,
        target_layers=target_layers,
    )

    return result.merged_weights, result.metrics


__all__ = [
    # Stage 1: Probe (ProbeConfig REMOVED - always precise path, all probes)
    "stage_probe",
    "ProbeResult",
    # Stage 2: Density
    "stage_density",
    "DensityStageResult",
    # Stage 3: Transplant (geometry-driven, graft_mask only)
    "stage_transplant",
    "TransplantStageResult",
]
