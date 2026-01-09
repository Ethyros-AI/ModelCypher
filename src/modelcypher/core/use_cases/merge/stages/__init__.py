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

Pipeline: PROBE → DENSITY → TRANSPLANT → VALIDATE

Stage 1: PROBE - Build intersection map from probe responses, compute GramAlign transforms
Stage 2: DENSITY - Knowledge density profiling for graft mask
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting
Stage 4: VALIDATE - Safety checks (numerical stability, refusal preservation, behavioral probes)

REMOVED (proven redundant):
- PERMUTE: GramAligner's CKA=1.0 in geodesic RKHS subsumes discrete permutation alignment.
  Permutation is a special case of continuous linear transforms already optimized by probe.
- ROTATE/PROPAGATE: No mathematical guarantee of boundary preservation.

References:
- AlphaEdit (null-space transplant): Fang et al. (2025) ICLR Outstanding Paper
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .probe import (
    ProbeResult,
    stage_probe as stage_probe_impl,
)
from .density import (
    DensityStageResult,
    stage_density as stage_density_impl,
)
# NOTE: ProbeConfig was REMOVED - Probe always uses precise mode with all probes.
# PERMUTE STAGE REMOVED: GramAligner's CKA=1.0 alignment subsumes permutation.
from .transplant import (
    TransplantStageResult,
    stage_transplant as stage_transplant_impl,
)
from .validate import (
    ValidateResult,
    stage_validate as stage_validate_impl,
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
    source_path: str = "",
    target_path: str = "",
    extract_layer_index_fn: Callable[[str], int | None],
    probe_mode: str = "atlas",
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
    dict[int, int] | None,  # layer_mapping
]:
    """Stage 1: Compute layer correspondences via CKA."""
    # collect_activations_fn=None lets probe.py auto-detect ActivationProvider
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
        collect_activations_fn=None,
        probe_mode=probe_mode,
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
        result.layer_mapping,
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
    - density_weights: Dict[int, Array] with per-layer density-based transfer weights
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
    extract_layer_index_fn: Callable[[str], int | None] = lambda x: None,
    backend: "Backend | None" = None,
    graft_mask: dict[str, dict[int, bool]] | None = None,
    density_weights: dict[int, "Array"] | None = None,  # Per-probe transfer weights from k-NN density
    feature_transforms: dict[int, list[list[float]]] | None = None,
    scale_ratios: dict[int, float] | None = None,  # EXACT: ||target|| / ||source @ F||
    embedding_transform: list[list[float]] | None = None,  # 2D GramAlign
    attention_transforms: dict[int, list[list[float]]] | None = None,
    k_transforms: dict[int, list[list[float]]] | None = None,
    v_transforms: dict[int, list[list[float]]] | None = None,
    intermediate_transforms: dict[int, list[list[float]]] | None = None,  # MLP transforms
    layer_mapping: dict[int, int] | None = None,
    layer_status: dict[int, str] | None = None,  # NEW: Per DIMENSIONAL_COMPRESSION.md
    source_tokenizer: Any | None = None,  # For token correspondence
    target_tokenizer: Any | None = None,  # For token correspondence
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
        layer_mapping=layer_mapping,
        layer_status=layer_status,  # NEW: Per DIMENSIONAL_COMPRESSION.md
        source_tokenizer=source_tokenizer,  # For token correspondence
        target_tokenizer=target_tokenizer,  # For token correspondence
    )

    return result.merged_weights, result.metrics


def stage_validate(
    *,
    merged_weights: dict[str, "Array"],
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    layer_confidences: dict[int, float],
    layer_indices: list[int],
    hidden_dim: int,
    target_model: Any | None = None,
    target_model_path: str | None = None,
    tokenizer: Any | None = None,
    collect_activations_fn: Callable | None = None,
    merged_model_path: str | None = None,
    backend: "Backend | None" = None,
) -> tuple[dict[str, Any], "ValidateResult"]:
    """Stage 4: Validation of merged weights.

    Returns raw measurements only. No verdicts - the geometry IS what it is.
    Callers interpret measurements relative to their own baselines.

    Returns:
        Tuple of (metrics dict, ValidateResult)
    """
    result = stage_validate_impl(
        merged_weights=merged_weights,
        source_weights=source_weights,
        target_weights=target_weights,
        layer_confidences=layer_confidences,
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        target_model=target_model,
        target_model_path=target_model_path,
        tokenizer=tokenizer,
        collect_activations_fn=collect_activations_fn,
        merged_model_path=merged_model_path,
        backend=backend,
    )

    return result.metrics, result


__all__ = [
    # Stage 1: Probe (ProbeConfig REMOVED - always precise mode, all probes)
    "stage_probe",
    "ProbeResult",
    # Stage 2: Density
    "stage_density",
    "DensityStageResult",
    # Stage 3: Transplant (geometry-driven, graft_mask only)
    "stage_transplant",
    "TransplantStageResult",
    # Stage 4: Validate (safety checks for merged weights)
    "stage_validate",
    "ValidateResult",
]
