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

"""Knowledge state diffing for model merging.

Compares knowledge density profiles between source and target models to
identify graft opportunities - concepts where source has dense representation
but target is sparse.

Key insight: Don't merge everything. Only graft into gaps.

Graft Opportunity = source_density - target_density

High positive values indicate concepts where:
- Source knows the concept well (dense)
- Target has gaps (sparse)
- Grafting would add value

Negative values indicate concepts where:
- Target already knows well
- Grafting would waste computation or cause interference
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    ModelDensityProfile,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _aggregate_opportunities(
    items: "Sequence[GraftOpportunity]",
) -> tuple[float, float, float, int]:
    """Compute aggregate statistics for graft opportunities.

    Args:
        items: Sequence of GraftOpportunity objects to aggregate.

    Returns:
        Tuple of (mean_source, mean_target, mean_opportunity, positive_count).
    """
    if not items:
        return 0.0, 0.0, 0.0, 0
    sum_source = 0.0
    sum_target = 0.0
    sum_opp = 0.0
    positive = 0
    for item in items:
        sum_source += item.source_density
        sum_target += item.target_density
        sum_opp += item.opportunity_score
        if item.opportunity_score > 0.0:
            positive += 1
    n = len(items)
    return sum_source / n, sum_target / n, sum_opp / n, positive


@dataclass(frozen=True)
class GraftOpportunity:
    """A single graft opportunity - concept where source can help target."""

    probe_id: str
    name: str
    domain: str
    layer: int

    # Density scores
    source_density: float
    target_density: float

    # Graft opportunity score (source - target)
    # Positive = source has knowledge target lacks
    # Negative = target already knows, don't touch
    opportunity_score: float


@dataclass(frozen=True)
class LayerDiff:
    """Knowledge diff for a single layer."""

    layer: int

    # All graft opportunities at this layer
    opportunities: list[GraftOpportunity]

    # Aggregates
    mean_opportunity: float
    positive_opportunity_count: int
    nonpositive_opportunity_count: int


@dataclass(frozen=True)
class DomainDiff:
    """Knowledge diff aggregated by domain."""

    domain: str
    mean_source_density: float
    mean_target_density: float
    mean_opportunity: float
    concept_count: int
    positive_opportunity_count: int


@dataclass(frozen=True)
class KnowledgeDiff:
    """Complete knowledge diff between two models."""

    source_path: str
    target_path: str

    # Per-layer diffs
    layer_diffs: dict[int, LayerDiff]

    # Per-domain aggregates
    domain_diffs: dict[str, DomainDiff]

    # Global statistics
    overall_source_density: float
    overall_target_density: float
    overall_opportunity: float

    # Ranked list of graft opportunities (highest first)
    ranked_opportunities: list[GraftOpportunity]

    # Summary
    total_concepts: int
    positive_opportunity_count: int
    nonpositive_opportunity_count: int


class KnowledgeDiffer:
    """Compare knowledge density profiles between models.

    Identifies where grafting from source to target would add value
    (source dense, target sparse) vs where it would be wasteful or
    harmful (target already dense).
    """

    def diff(
        self,
        source_profile: ModelDensityProfile,
        target_profile: ModelDensityProfile,
    ) -> KnowledgeDiff:
        """Compute knowledge diff between source and target profiles.

        Args:
            source_profile: Density profile of source model.
            target_profile: Density profile of target model.

        Returns:
            KnowledgeDiff with graft opportunities ranked by value.
        """
        # Build concept lookup by (probe_id, layer)
        source_lookup: dict[tuple[str, int], ConceptDensity] = {}
        for profile in source_profile.layer_profiles.values():
            for concept in profile.concept_densities:
                source_lookup[(concept.probe_id, concept.layer)] = concept

        target_lookup: dict[tuple[str, int], ConceptDensity] = {}
        for profile in target_profile.layer_profiles.values():
            for concept in profile.concept_densities:
                target_lookup[(concept.probe_id, concept.layer)] = concept

        # Find common concepts
        common_keys = set(source_lookup.keys()) & set(target_lookup.keys())

        # Compute opportunities for each common concept
        all_opportunities: list[GraftOpportunity] = []

        for key in common_keys:
            source_concept = source_lookup[key]
            target_concept = target_lookup[key]

            opportunity_score = source_concept.density_score - target_concept.density_score

            all_opportunities.append(
                GraftOpportunity(
                    probe_id=source_concept.probe_id,
                    name=source_concept.name,
                    domain=source_concept.domain,
                    layer=source_concept.layer,
                    source_density=source_concept.density_score,
                    target_density=target_concept.density_score,
                    opportunity_score=opportunity_score,
                )
            )

        # Group by layer
        layer_diffs = self._group_by_layer(all_opportunities)

        # Group by domain
        domain_diffs = self._group_by_domain(all_opportunities)

        # Rank opportunities (highest first)
        ranked = sorted(all_opportunities, key=lambda x: x.opportunity_score, reverse=True)

        # Global statistics using helper
        overall_source, overall_target, overall_opp, positive_count = (
            _aggregate_opportunities(all_opportunities)
        )
        total = len(all_opportunities)
        nonpositive_count = total - positive_count

        return KnowledgeDiff(
            source_path=source_profile.model_path,
            target_path=target_profile.model_path,
            layer_diffs=layer_diffs,
            domain_diffs=domain_diffs,
            overall_source_density=overall_source,
            overall_target_density=overall_target,
            overall_opportunity=overall_opp,
            ranked_opportunities=ranked,
            total_concepts=len(all_opportunities),
            positive_opportunity_count=positive_count,
            nonpositive_opportunity_count=nonpositive_count,
        )

    def _group_by_layer(
        self,
        opportunities: list[GraftOpportunity],
    ) -> dict[int, LayerDiff]:
        """Group opportunities by layer.

        Args:
            opportunities: List of graft opportunities to group.

        Returns:
            Dictionary mapping layer index to LayerDiff.
        """
        by_layer: dict[int, list[GraftOpportunity]] = defaultdict(list)
        for opp in opportunities:
            by_layer[opp.layer].append(opp)

        result: dict[int, LayerDiff] = {}
        for layer, opps in by_layer.items():
            _, _, mean_opp, positive_count = _aggregate_opportunities(opps)
            result[layer] = LayerDiff(
                layer=layer,
                opportunities=opps,
                mean_opportunity=mean_opp,
                positive_opportunity_count=positive_count,
                nonpositive_opportunity_count=len(opps) - positive_count,
            )
        return result

    def _group_by_domain(
        self,
        opportunities: list[GraftOpportunity],
    ) -> dict[str, DomainDiff]:
        """Group opportunities by domain.

        Args:
            opportunities: List of graft opportunities to group.

        Returns:
            Dictionary mapping domain name to DomainDiff.
        """
        by_domain: dict[str, list[GraftOpportunity]] = defaultdict(list)
        for opp in opportunities:
            by_domain[opp.domain].append(opp)

        result: dict[str, DomainDiff] = {}
        for domain, opps in by_domain.items():
            mean_source, mean_target, mean_opp, positive_count = (
                _aggregate_opportunities(opps)
            )
            result[domain] = DomainDiff(
                domain=domain,
                mean_source_density=mean_source,
                mean_target_density=mean_target,
                mean_opportunity=mean_opp,
                concept_count=len(opps),
                positive_opportunity_count=positive_count,
            )
        return result


def compute_graft_mask(diff: KnowledgeDiff) -> dict[str, dict[int, bool]]:
    """Compute a graft mask indicating which concepts to graft.

    Args:
        diff: KnowledgeDiff containing ranked graft opportunities.

    Returns:
        Nested dict mapping probe_id -> layer -> should_graft.
        True indicates positive opportunity (source has knowledge target lacks).
        False indicates non-positive opportunity (target already knows).
    """
    mask: dict[str, dict[int, bool]] = defaultdict(dict)
    for opp in diff.ranked_opportunities:
        mask[opp.probe_id][opp.layer] = opp.opportunity_score > 0.0
    return dict(mask)  # Convert back to regular dict for serialization


__all__ = [
    "GraftOpportunity",
    "LayerDiff",
    "DomainDiff",
    "KnowledgeDiff",
    "KnowledgeDiffer",
    "compute_graft_mask",
]
