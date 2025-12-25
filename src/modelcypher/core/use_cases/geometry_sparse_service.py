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
Geometry Sparse Region Service.

Exposes sparse region analysis as CLI/MCP-consumable operations.
Identifies sparse regions suitable for LoRA injection and domain-specific training.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.geometry.refusal_direction_detector import (
    STANDARD_CONTRASTIVE_PAIRS,
    ContrastivePair,
    RefusalDirection,
    RefusalDirectionDetector,
)
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    Configuration as RefusalConfig,
)
from modelcypher.core.domain.geometry.sparse_region_domains import (
    DomainCategory,
    DomainDefinition,
    SparseRegionDomains,
)
from modelcypher.core.domain.geometry.sparse_region_locator import (
    AnalysisResult,
    LayerActivationStats,
    SparseRegionLocator,
)
from modelcypher.core.domain.geometry.sparse_region_locator import (
    Configuration as LocatorConfig,
)


@dataclass(frozen=True)
class DomainInfo:
    """Summary info for a domain."""

    name: str
    description: str
    category: str
    probe_count: int
    expected_layer_range: tuple[float, float] | None


class GeometrySparseService:
    """
    Service for sparse region and refusal direction operations.
    """

    def list_domains(self) -> list[DomainInfo]:
        """List all built-in sparse region domains."""
        return [
            DomainInfo(
                name=domain.name,
                description=domain.description,
                category=domain.category.value,
                probe_count=len(domain.probe_prompts),
                expected_layer_range=domain.expected_active_layer_range,
            )
            for domain in SparseRegionDomains.all_built_in
        ]

    def get_domain(self, name: str) -> DomainDefinition | None:
        """Get a domain by name."""
        return SparseRegionDomains.domain_named(name)

    def get_domains_by_category(self, category: str) -> list[DomainInfo]:
        """Get domains in a specific category."""
        try:
            cat = DomainCategory(category)
        except ValueError:
            return []
        domains = SparseRegionDomains.domains_in_category(cat)
        return [
            DomainInfo(
                name=d.name,
                description=d.description,
                category=d.category.value,
                probe_count=len(d.probe_prompts),
                expected_layer_range=d.expected_active_layer_range,
            )
            for d in domains
        ]

    def locate_sparse_regions(
        self,
        domain_stats: list[dict],
        baseline_stats: list[dict],
        domain_name: str = "unknown",
        base_rank: int = 16,
        sparsity_threshold: float = 0.3,
    ) -> AnalysisResult:
        """
        Locate sparse regions suitable for LoRA injection.

        Args:
            domain_stats: List of layer activation stats for domain prompts
            baseline_stats: List of layer activation stats for baseline prompts
            domain_name: Name of the domain being analyzed
            base_rank: Base LoRA rank to use
            sparsity_threshold: Threshold for considering a layer sparse

        Returns:
            AnalysisResult with sparse layers and LoRA recommendation
        """
        config = LocatorConfig(
            base_rank=base_rank,
            sparsity_threshold=sparsity_threshold,
        )
        locator = SparseRegionLocator(config)

        domain_layer_stats = [
            LayerActivationStats(
                layer_index=s["layer_index"],
                mean_activation=s["mean_activation"],
                max_activation=s.get("max_activation", s["mean_activation"]),
                activation_variance=s.get("activation_variance", 0.0),
                prompt_count=s.get("prompt_count", 1),
            )
            for s in domain_stats
        ]

        baseline_layer_stats = [
            LayerActivationStats(
                layer_index=s["layer_index"],
                mean_activation=s["mean_activation"],
                max_activation=s.get("max_activation", s["mean_activation"]),
                activation_variance=s.get("activation_variance", 0.0),
                prompt_count=s.get("prompt_count", 1),
            )
            for s in baseline_stats
        ]

        return locator.analyze(
            domain_stats=domain_layer_stats,
            baseline_stats=baseline_layer_stats,
            domain=domain_name,
        )

    def detect_refusal_direction(
        self,
        harmful_activations: list[list[float]],
        harmless_activations: list[list[float]],
        layer_index: int,
        model_id: str,
        normalize: bool = True,
    ) -> RefusalDirection | None:
        """
        Compute refusal direction from contrastive activations.

        Args:
            harmful_activations: Activations from harmful prompts
            harmless_activations: Activations from harmless prompts
            layer_index: Layer these activations come from
            model_id: Model identifier
            normalize: Whether to normalize the direction vector

        Returns:
            RefusalDirection if computation succeeds, None otherwise
        """
        config = RefusalConfig(normalize_direction=normalize)
        return RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_activations,
            harmless_activations=harmless_activations,
            configuration=config,
            layer_index=layer_index,
            model_id=model_id,
        )

    def get_contrastive_pairs(self) -> list[ContrastivePair]:
        """Get standard contrastive prompt pairs for refusal direction."""
        return list(STANDARD_CONTRASTIVE_PAIRS)

    @staticmethod
    def domains_payload(domains: list[DomainInfo]) -> dict:
        """Convert domain list to CLI/MCP payload."""
        return {
            "domains": [
                {
                    "name": d.name,
                    "description": d.description,
                    "category": d.category,
                    "probeCount": d.probe_count,
                    "expectedLayerRange": list(d.expected_layer_range)
                    if d.expected_layer_range
                    else None,
                }
                for d in domains
            ],
            "count": len(domains),
        }

    @staticmethod
    def analysis_payload(result: AnalysisResult) -> dict:
        """Convert sparse region analysis to CLI/MCP payload."""
        return {
            "domain": result.domain,
            "sparseLayers": result.sparse_layers,
            "skipLayers": result.skip_layers,
            "layerSparsity": {str(k): v for k, v in result.layer_sparsity.items()},
            "recommendation": {
                "overallRank": result.recommendation.overall_rank,
                "alpha": result.recommendation.alpha,
                "targetModules": result.recommendation.target_modules,
                "quality": result.recommendation.quality.value,
                "estimatedPreservation": result.recommendation.estimated_preservation,
                "rationale": result.recommendation.rationale,
            },
            "dareAlignment": {
                "highDroppabilityLayers": result.dare_alignment.high_droppability_layers,
                "overlapWithSparse": result.dare_alignment.overlap_with_sparse,
                "confidence": result.dare_alignment.confidence,
            }
            if result.dare_alignment
            else None,
        }

    @staticmethod
    def refusal_direction_payload(direction: RefusalDirection) -> dict:
        """Convert refusal direction to CLI/MCP payload."""
        return {
            "layerIndex": direction.layer_index,
            "hiddenSize": direction.hidden_size,
            "strength": direction.strength,
            "explainedVariance": direction.explained_variance,
            "modelId": direction.model_id,
            "computedAt": direction.computed_at.isoformat(),
            "directionNorm": sum(x * x for x in direction.direction) ** 0.5
            if direction.direction
            else 0.0,
        }

    @staticmethod
    def contrastive_pairs_payload(pairs: list[ContrastivePair]) -> dict:
        """Convert contrastive pairs to CLI/MCP payload."""
        return {
            "pairs": [{"harmful": p.harmful, "harmless": p.harmless} for p in pairs],
            "count": len(pairs),
        }
