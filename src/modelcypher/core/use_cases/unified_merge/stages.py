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
from typing import TYPE_CHECKING, Any, Callable

from .models import UnifiedMergeConfig

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_vocabulary(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
) -> tuple[dict[str, "Array"], dict[str, Any], bool, Any | None]:
    """Stage 0: Align source vocabulary to target vocabulary."""
    from modelcypher.core.use_cases.merge_stages.stage_0_vocabulary import (
        VocabularyConfig,
        stage_vocabulary_align,
    )

    config = VocabularyConfig()

    result = stage_vocabulary_align(
        source_weights=source_weights,
        target_weights=target_weights,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        config=config,
    )

    if result.was_aligned:
        logger.info("Vocabulary alignment applied")
    else:
        reason = result.metrics.get("reason", "unknown")
        logger.info("Vocabulary alignment skipped: %s", reason)

    return (
        result.modified_weights,
        result.metrics,
        result.was_aligned,
        result.alignment_map,
    )


def stage_probe(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    source_model: Any | None,
    target_model: Any | None,
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
    alignment_map: Any | None,
    extract_layer_index_fn: Callable[[str], int | None],
) -> tuple[dict[str, Any], dict[str, Any], dict | None, dict | None]:
    """Stage 1: Compute layer correspondences via CKA.

    No configuration - always uses precise mode with all probes.
    """
    from modelcypher.core.use_cases.merge_stages.stage_1_probe import (
        collect_layer_activations_mlx,
    )
    from modelcypher.core.use_cases.merge_stages.stage_1_probe import (
        stage_probe as stage_probe_impl,
    )

    # ProbeConfig was REMOVED. Always use precise mode with all probes.
    # No configuration needed - the probe corpus is designed for complete coverage.

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
        alignment_map=alignment_map,
    )

    return {
        "correlations": result.correlations,
        "confidences": result.confidences,
        "dimension_correlations": result.dimension_correlations,
        "intersection_map": result.intersection_map,
        "probe_ids": result.probe_ids,
        "probe_domains": result.probe_domains,
    }, result.metrics, result.source_activations, result.target_activations, result.source_intermediate_activations, result.target_intermediate_activations, result.source_attention_activations, result.target_attention_activations, result.source_kv_activations, result.target_kv_activations


def stage_permute(
    *,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    backend: "Backend",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Stage 2: Git Re-Basin permutation alignment.

    Solves the permutation symmetry problem for MLP neurons.
    Neural networks have N! permutation symmetries per MLP layer.
    This stage finds the optimal permutation P that minimizes:
        ||W_target - P @ W_source||_F

    This runs BEFORE transplant to reduce the delta magnitude between
    source and target weights. By aligning neuron orderings first, the
    null-space projection in transplant has less work to do.

    No configuration - permutation alignment is always run.

    Reference: Ainsworth et al. (2023) arXiv:2209.04836 "Git Re-Basin"
    """
    from modelcypher.core.use_cases.merge_stages.stage_2_permute import (
        infer_hidden_dim,
    )
    from modelcypher.core.use_cases.merge_stages.stage_2_permute import (
        stage_permute as stage_permute_impl,
    )

    # PermuteConfig was REMOVED. Permutation alignment always runs.
    # Skipping permutation alignment produces worse merges with no benefit.

    result = stage_permute_impl(
        source_weights=source_weights,
        target_weights=target_weights,
        intersection_map_obj=intersection_map_obj,
        layer_confidences=layer_confidences,
        infer_hidden_dim_fn=infer_hidden_dim,
        backend=backend,
    )

    return result.weights, result.metrics


def stage_density(
    *,
    source_activations: dict | None,
    target_activations: dict | None,
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    layers: list[int],
    backend: "Backend",
) -> tuple[dict[str, dict[int, bool]] | None, dict[str, Any]]:
    """Stage 2.5: Density analysis for selective grafting.

    Args:
        source_activations: Activations from source model.
        target_activations: Activations from target model.
        probe_ids: List of probe IDs.
        probe_domains: List of domains for each probe.
        layers: Layer indices to analyze.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (graft_mask, density_metrics).
    """
    from modelcypher.core.use_cases.merge_stages.stage_2_density import (
        stage_density as stage_density_impl,
    )

    result = stage_density_impl(
        source_activations=source_activations or {},
        target_activations=target_activations or {},
        probe_ids=probe_ids or [],
        probe_domains=probe_domains or [],
        layers=layers,
        backend=backend,
    )

    return result.graft_mask, result.metrics


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
) -> tuple[dict[str, "Array"], dict[str, Any]]:
    """Stage 3: Null-space constrained transplant.

    No configuration - all geometric parameters derived from data.
    Only transplant_domains is user input.
    """
    from modelcypher.core.use_cases.merge_stages.stage_3_transplant import (
        TransplantStageConfig,
    )
    from modelcypher.core.use_cases.merge_stages.stage_3_transplant import (
        stage_transplant as stage_transplant_impl,
    )

    # TransplantStageConfig simplified: only core_domains and internal graft_mask
    # boundary_k, geodesic_k_neighbors, transplant_layers all REMOVED
    stage_config = TransplantStageConfig(
        core_domains=tuple(transplant_domains),
        graft_mask=graft_mask,
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
