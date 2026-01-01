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

"""Stage 2: DENSITY - Knowledge density profiling and graft mask computation.

Identifies which concepts (probe_id, layer) should be grafted based on density:
- High density in source + Low density in target = GRAFT (fill the gap)
- Low density in source OR High density in target = SKIP (nothing to add)

This stage MUST run between probe (stage 1) and transplant (stage 3) to enable
selective grafting that improves the merged model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    KnowledgeDensityAnalyzer,
    KnowledgeDensityConfig,
    LayerDensityProfile,
    ModelDensityProfile,
)
from modelcypher.core.domain.geometry.knowledge_diff import (
    KnowledgeDiff,
    KnowledgeDiffer,
    compute_graft_mask,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DensityStageConfig:
    """Configuration for density analysis stage."""

    # Knowledge density analysis config
    density_config: KnowledgeDensityConfig | None = None

    # Include low-opportunity concepts in graft mask (default: only high)
    include_low_opportunity: bool = False

    # Skip density analysis (for backward compatibility)
    # When True, returns an empty graft_mask (all concepts grafted)
    skip_density_analysis: bool = False


@dataclass
class DensityStageResult:
    """Result of density analysis stage."""

    # Density profiles for both models
    source_profile: ModelDensityProfile | None
    target_profile: ModelDensityProfile | None

    # Knowledge diff showing graft opportunities
    knowledge_diff: KnowledgeDiff | None

    # The graft mask: probe_id -> layer -> should_graft
    # If None, transplant ALL concepts (backward compatible)
    graft_mask: dict[str, dict[int, bool]] | None

    # Metrics
    metrics: dict[str, float | int]


class ActivationProviderFromDict:
    """Adapts pre-collected activations dict to ActivationProvider protocol."""

    def __init__(
        self,
        activations: dict[int, list["Array"]],
        probe_ids: list[str],
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with pre-collected activations.

        Args:
            activations: Dict mapping layer_idx -> list of activations.
                Each activation corresponds to a probe_id in order.
            probe_ids: List of probe IDs in same order as activations.
            backend: Backend for tensor operations.
        """
        self._activations = activations
        self._probe_ids = probe_ids
        self._backend = backend or get_default_backend()

        # Build lookup: (probe_id, layer) -> activation
        self._lookup: dict[tuple[str, int], "Array"] = {}
        for layer_idx, act_list in activations.items():
            for i, act in enumerate(act_list):
                if i < len(probe_ids):
                    self._lookup[(probe_ids[i], layer_idx)] = act

    def get_activations(
        self,
        texts: list[str],
        layer: int,
    ) -> list["Array"]:
        """Get activations for texts at a specific layer.

        For pre-collected activations, we look up by position in probe_ids.
        This is a simplified implementation that works with our use case.
        """
        # In practice, the density analyzer calls this with support_texts
        # but we already have the activations indexed by probe_id
        # Just return all activations for this layer
        if layer in self._activations:
            return list(self._activations[layer])
        return []


def stage_density(
    source_activations: dict[int, list["Array"]],
    target_activations: dict[int, list["Array"]],
    probe_ids: list[str],
    probe_domains: list[str],
    layers: list[int],
    config: DensityStageConfig | None = None,
    backend: "Backend | None" = None,
) -> DensityStageResult:
    """Stage 2: Compute knowledge density profiles and graft mask.

    Args:
        source_activations: Activations from source model per layer.
        target_activations: Activations from target model per layer.
        probe_ids: List of probe IDs corresponding to activations.
        probe_domains: List of domains for each probe.
        layers: Layer indices to analyze.
        config: Stage configuration.
        backend: Backend for tensor operations.

    Returns:
        DensityStageResult with profiles, diff, and graft mask.
    """
    b = backend or get_default_backend()
    cfg = config or DensityStageConfig()

    # Metrics to track
    metrics: dict[str, float | int] = {
        "layers_analyzed": 0,
        "concepts_analyzed": 0,
        "high_opportunity_count": 0,
        "no_graft_count": 0,
    }

    # Skip density analysis if configured (backward compatible)
    if cfg.skip_density_analysis:
        logger.info("DENSITY: Skipping density analysis (all concepts will be grafted)")
        return DensityStageResult(
            source_profile=None,
            target_profile=None,
            knowledge_diff=None,
            graft_mask=None,  # None = graft all
            metrics=metrics,
        )

    # Validate inputs
    if not source_activations or not target_activations:
        logger.warning("DENSITY: Missing activations, skipping density analysis")
        return DensityStageResult(
            source_profile=None,
            target_profile=None,
            knowledge_diff=None,
            graft_mask=None,
            metrics=metrics,
        )

    if not probe_ids or len(probe_ids) != len(probe_domains):
        logger.warning("DENSITY: Probe metadata mismatch, skipping density analysis")
        return DensityStageResult(
            source_profile=None,
            target_profile=None,
            knowledge_diff=None,
            graft_mask=None,
            metrics=metrics,
        )

    logger.info(
        "DENSITY: Analyzing %d layers, %d probes for graft opportunities",
        len(layers),
        len(probe_ids),
    )

    # Build simple concept density profiles from activations
    # This is a streamlined version that works with pre-collected activations
    source_profile = _build_density_profile_from_activations(
        activations=source_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        layers=layers,
        backend=b,
    )

    target_profile = _build_density_profile_from_activations(
        activations=target_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        layers=layers,
        backend=b,
    )

    # Compute knowledge diff
    differ = KnowledgeDiffer()
    knowledge_diff = differ.diff(source_profile, target_profile)

    # Compute graft mask
    graft_mask = compute_graft_mask(
        knowledge_diff,
        include_low_opportunity=cfg.include_low_opportunity,
    )

    # Update metrics
    metrics["layers_analyzed"] = len(layers)
    metrics["concepts_analyzed"] = knowledge_diff.total_concepts
    metrics["high_opportunity_count"] = knowledge_diff.high_opportunity_count
    metrics["no_graft_count"] = knowledge_diff.no_graft_count
    metrics["overall_source_density"] = knowledge_diff.overall_source_density
    metrics["overall_target_density"] = knowledge_diff.overall_target_density
    metrics["overall_opportunity"] = knowledge_diff.overall_opportunity

    logger.info(
        "DENSITY: %d concepts analyzed, %d high opportunity (will graft), %d no graft (target dense)",
        knowledge_diff.total_concepts,
        knowledge_diff.high_opportunity_count,
        knowledge_diff.no_graft_count,
    )

    return DensityStageResult(
        source_profile=source_profile,
        target_profile=target_profile,
        knowledge_diff=knowledge_diff,
        graft_mask=graft_mask,
        metrics=metrics,
    )


def _build_density_profile_from_activations(
    activations: dict[int, list["Array"]],
    probe_ids: list[str],
    probe_domains: list[str],
    layers: list[int],
    backend: "Backend",
) -> ModelDensityProfile:
    """Build a density profile from pre-collected activations.

    Uses intrinsic dimension as the density signal:
    - Low intrinsic dimension = dense representation (well-learned)
    - High intrinsic dimension = sparse representation (gap in knowledge)
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    b = backend
    id_estimator = IntrinsicDimension(backend=b)

    layer_profiles: dict[int, LayerDensityProfile] = {}
    all_concepts: list[ConceptDensity] = []

    for layer_idx in layers:
        act_list = activations.get(layer_idx, [])
        if not act_list:
            continue

        concept_densities: list[ConceptDensity] = []

        for i, act in enumerate(act_list):
            if i >= len(probe_ids):
                break

            probe_id = probe_ids[i]
            domain = probe_domains[i] if i < len(probe_domains) else "unknown"

            # Compute intrinsic dimension for this activation
            # Single activation vector - estimate from its variance structure
            act_arr = b.array(act)
            b.eval(act_arr)

            if len(act_arr.shape) == 1:
                act_arr = b.reshape(act_arr, (1, -1))
                b.eval(act_arr)

            # Use variance as a proxy for intrinsic dimension
            # Higher variance = less compressed = lower density
            var_arr = b.var(act_arr)
            b.eval(var_arr)
            variance = float(b.to_numpy(var_arr).item())

            # Convert variance to density score (0-1)
            # Lower variance = higher density
            # Use sigmoid-like transform: density = 1 / (1 + var)
            import math

            intrinsic_dim = max(1.0, math.log(1.0 + variance * 1000))
            density_score = 1.0 / (1.0 + math.log(intrinsic_dim))
            density_score = max(0.0, min(1.0, density_score))

            concept_densities.append(
                ConceptDensity(
                    probe_id=probe_id,
                    name=probe_id,  # Use probe_id as name
                    domain=domain,
                    layer=layer_idx,
                    intrinsic_dimension=intrinsic_dim,
                    density_score=density_score,
                    activation_variance=variance,
                    cluster_tightness=None,
                    dimension_class="estimated",
                )
            )

        # Compute layer threshold from data
        if concept_densities:
            scores = sorted(c.density_score for c in concept_densities)
            n = len(scores)
            median = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2
            threshold = median

            sparse = [c for c in concept_densities if c.density_score < threshold]
            dense = [c for c in concept_densities if c.density_score >= threshold]
            mean_density = sum(c.density_score for c in concept_densities) / len(concept_densities)
        else:
            threshold = 0.5
            sparse = []
            dense = []
            mean_density = 0.0

        layer_profiles[layer_idx] = LayerDensityProfile(
            layer=layer_idx,
            concept_densities=concept_densities,
            mean_density=mean_density,
            median_density=threshold,
            sparse_concept_count=len(sparse),
            dense_concept_count=len(dense),
            density_threshold=threshold,
        )

        all_concepts.extend(concept_densities)

    # Compute per-domain aggregates
    domain_densities: dict[str, list[float]] = {}
    for c in all_concepts:
        if c.domain not in domain_densities:
            domain_densities[c.domain] = []
        domain_densities[c.domain].append(c.density_score)

    domain_means = {
        domain: sum(scores) / len(scores)
        for domain, scores in domain_densities.items()
        if scores
    }

    # Overall density
    overall = sum(c.density_score for c in all_concepts) / len(all_concepts) if all_concepts else 0.0

    # Global threshold
    global_threshold = 0.5
    if all_concepts:
        scores = sorted(c.density_score for c in all_concepts)
        n = len(scores)
        global_threshold = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2

    sparse_all = [c for c in all_concepts if c.density_score < global_threshold]
    dense_all = [c for c in all_concepts if c.density_score >= global_threshold]

    return ModelDensityProfile(
        model_path="",  # Set by caller
        layers=layers,
        layer_profiles=layer_profiles,
        domain_densities=domain_means,
        overall_density=overall,
        sparse_concepts=sparse_all,
        dense_concepts=dense_all,
    )


def filter_core_probes_by_graft_mask(
    core_probe_ids: set[str],
    probe_ids: list[str],
    layer_idx: int,
    graft_mask: dict[str, dict[int, bool]] | None,
) -> set[str]:
    """Filter core probe IDs to only include those that should be grafted.

    Args:
        core_probe_ids: Original set of core probe IDs (from domain selection).
        probe_ids: Full list of probe IDs.
        layer_idx: Current layer being processed.
        graft_mask: Graft mask from density analysis (None = graft all).

    Returns:
        Filtered set of probe IDs that should be grafted at this layer.
    """
    if graft_mask is None:
        # No mask = graft all (backward compatible)
        return core_probe_ids

    filtered = set()
    for probe_id in core_probe_ids:
        # Check if this (probe_id, layer) should be grafted
        should_graft = graft_mask.get(probe_id, {}).get(layer_idx, False)
        if should_graft:
            filtered.add(probe_id)

    if len(filtered) < len(core_probe_ids):
        skipped = len(core_probe_ids) - len(filtered)
        logger.debug(
            "Layer %d: Filtered %d/%d core probes (target already dense)",
            layer_idx,
            skipped,
            len(core_probe_ids),
        )

    return filtered


__all__ = [
    "DensityStageConfig",
    "DensityStageResult",
    "stage_density",
    "filter_core_probes_by_graft_mask",
]
