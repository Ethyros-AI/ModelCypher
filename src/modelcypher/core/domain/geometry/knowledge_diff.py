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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    ModelDensityProfile,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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

    # Classification
    classification: str  # "high_opportunity", "low_opportunity", "no_graft"


@dataclass(frozen=True)
class LayerDiff:
    """Knowledge diff for a single layer."""

    layer: int

    # All graft opportunities at this layer
    opportunities: list[GraftOpportunity]

    # Aggregates
    mean_opportunity: float
    high_opportunity_count: int
    no_graft_count: int

    # Threshold used for classification (derived from data)
    opportunity_threshold: float


@dataclass(frozen=True)
class DomainDiff:
    """Knowledge diff aggregated by domain."""

    domain: str
    mean_source_density: float
    mean_target_density: float
    mean_opportunity: float
    concept_count: int
    high_opportunity_count: int


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

    # Concepts that should NOT be grafted (target already dense)
    no_graft_concepts: list[GraftOpportunity]

    # Summary
    total_concepts: int
    high_opportunity_count: int
    no_graft_count: int


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
                    classification="pending",  # Will be set after threshold
                )
            )

        # Compute threshold from data
        opportunity_threshold = self._compute_threshold(all_opportunities)

        # Classify opportunities
        classified: list[GraftOpportunity] = []
        for opp in all_opportunities:
            if opp.opportunity_score > opportunity_threshold:
                classification = "high_opportunity"
            elif opp.opportunity_score < -opportunity_threshold:
                classification = "no_graft"
            else:
                classification = "low_opportunity"

            classified.append(
                GraftOpportunity(
                    probe_id=opp.probe_id,
                    name=opp.name,
                    domain=opp.domain,
                    layer=opp.layer,
                    source_density=opp.source_density,
                    target_density=opp.target_density,
                    opportunity_score=opp.opportunity_score,
                    classification=classification,
                )
            )

        # Group by layer
        layer_diffs = self._group_by_layer(classified, opportunity_threshold)

        # Group by domain
        domain_diffs = self._group_by_domain(classified)

        # Rank opportunities (highest first)
        ranked = sorted(classified, key=lambda x: x.opportunity_score, reverse=True)

        # Separate no-graft concepts
        no_graft = [c for c in classified if c.classification == "no_graft"]
        high_opportunity = [c for c in classified if c.classification == "high_opportunity"]

        # Global statistics
        if classified:
            overall_source = sum(c.source_density for c in classified) / len(classified)
            overall_target = sum(c.target_density for c in classified) / len(classified)
            overall_opp = sum(c.opportunity_score for c in classified) / len(classified)
        else:
            overall_source = 0.0
            overall_target = 0.0
            overall_opp = 0.0

        return KnowledgeDiff(
            source_path=source_profile.model_path,
            target_path=target_profile.model_path,
            layer_diffs=layer_diffs,
            domain_diffs=domain_diffs,
            overall_source_density=overall_source,
            overall_target_density=overall_target,
            overall_opportunity=overall_opp,
            ranked_opportunities=ranked,
            no_graft_concepts=no_graft,
            total_concepts=len(classified),
            high_opportunity_count=len(high_opportunity),
            no_graft_count=len(no_graft),
        )

    def _compute_threshold(self, opportunities: list[GraftOpportunity]) -> float:
        """Compute opportunity threshold from data.

        Uses standard deviation as threshold - concepts more than 1 stdev
        from mean opportunity are classified as high/no-graft.
        """
        if not opportunities:
            return 0.1

        scores = [o.opportunity_score for o in opportunities]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        stdev = variance ** 0.5

        # Threshold is 0.5 stdev from zero (not from mean)
        # This ensures small differences don't trigger grafting
        return max(0.05, stdev * 0.5)

    def _group_by_layer(
        self,
        opportunities: list[GraftOpportunity],
        threshold: float,
    ) -> dict[int, LayerDiff]:
        """Group opportunities by layer."""
        by_layer: dict[int, list[GraftOpportunity]] = {}

        for opp in opportunities:
            if opp.layer not in by_layer:
                by_layer[opp.layer] = []
            by_layer[opp.layer].append(opp)

        result: dict[int, LayerDiff] = {}
        for layer, opps in by_layer.items():
            mean_opp = sum(o.opportunity_score for o in opps) / len(opps) if opps else 0.0
            high_count = sum(1 for o in opps if o.classification == "high_opportunity")
            no_graft_count = sum(1 for o in opps if o.classification == "no_graft")

            result[layer] = LayerDiff(
                layer=layer,
                opportunities=opps,
                mean_opportunity=mean_opp,
                high_opportunity_count=high_count,
                no_graft_count=no_graft_count,
                opportunity_threshold=threshold,
            )

        return result

    def _group_by_domain(
        self,
        opportunities: list[GraftOpportunity],
    ) -> dict[str, DomainDiff]:
        """Group opportunities by domain."""
        by_domain: dict[str, list[GraftOpportunity]] = {}

        for opp in opportunities:
            if opp.domain not in by_domain:
                by_domain[opp.domain] = []
            by_domain[opp.domain].append(opp)

        result: dict[str, DomainDiff] = {}
        for domain, opps in by_domain.items():
            n = len(opps)
            mean_source = sum(o.source_density for o in opps) / n if n else 0.0
            mean_target = sum(o.target_density for o in opps) / n if n else 0.0
            mean_opp = sum(o.opportunity_score for o in opps) / n if n else 0.0
            high_count = sum(1 for o in opps if o.classification == "high_opportunity")

            result[domain] = DomainDiff(
                domain=domain,
                mean_source_density=mean_source,
                mean_target_density=mean_target,
                mean_opportunity=mean_opp,
                concept_count=n,
                high_opportunity_count=high_count,
            )

        return result


def compute_graft_mask(
    diff: KnowledgeDiff,
    include_low_opportunity: bool = False,
) -> dict[str, dict[int, bool]]:
    """Compute a graft mask indicating which concepts to graft.

    Args:
        diff: Knowledge diff between source and target.
        include_low_opportunity: If True, include low-opportunity concepts.

    Returns:
        Dict mapping probe_id -> layer -> should_graft.
    """
    mask: dict[str, dict[int, bool]] = {}

    for opp in diff.ranked_opportunities:
        if opp.probe_id not in mask:
            mask[opp.probe_id] = {}

        should_graft = opp.classification == "high_opportunity"
        if include_low_opportunity:
            should_graft = should_graft or opp.classification == "low_opportunity"

        mask[opp.probe_id][opp.layer] = should_graft

    return mask


__all__ = [
    "GraftOpportunity",
    "LayerDiff",
    "DomainDiff",
    "KnowledgeDiff",
    "KnowledgeDiffer",
    "compute_graft_mask",
]
