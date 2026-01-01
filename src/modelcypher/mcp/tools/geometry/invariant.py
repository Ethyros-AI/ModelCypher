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

"""Geometry invariant MCP tools.

Contains tools for:
- Invariant layer mapping between models
- Collapse risk analysis
- Atlas inventory
"""

from __future__ import annotations

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_invariant_tools(ctx: ServiceContext) -> None:
    """Register geometry invariant/atlas tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_invariant_map_layers" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_invariant_map_layers(
            sourcePath: str,
            targetPath: str,
            families: list[str] | None = None,
            scope: str = "sequenceInvariants",
            atlasSources: list[str] | None = None,
            atlasDomains: list[str] | None = None,
            triangulation: bool = True,
        ) -> dict:
            """Map layers between models using multi-atlas triangulation.

            Collapse threshold is derived from the activation variance distribution.
            No user parameters for thresholds.
            """
            from modelcypher.core.use_cases.invariant_layer_mapping_service import (
                InvariantLayerMappingService,
                LayerMappingConfig,
            )

            source_path = require_existing_directory(sourcePath)
            target_path = require_existing_directory(targetPath)
            config = LayerMappingConfig(
                source_model_path=str(source_path),
                target_model_path=str(target_path),
                invariant_scope=scope,
                families=families,
                atlas_sources=atlasSources,
                atlas_domains=atlasDomains,
                use_triangulation=triangulation,
                # collapse_threshold=None - derived from activation variance
            )
            result = ctx.invariant_mapping_service.map_layers(config)
            payload = InvariantLayerMappingService.result_payload(result)
            return payload

    if "mc_geometry_invariant_collapse_risk" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_invariant_collapse_risk(
            modelPath: str,
            families: list[str] | None = None,
        ) -> dict:
            """Analyze layer collapse risk for a model.

            Collapse threshold is derived from the activation variance distribution.
            No user parameters for thresholds.
            """
            from modelcypher.core.use_cases.invariant_layer_mapping_service import (
                CollapseRiskConfig,
                InvariantLayerMappingService,
            )

            model_path = require_existing_directory(modelPath)
            config = CollapseRiskConfig(
                model_path=str(model_path),
                families=families,
                # collapse_threshold=None - derived from variance distribution
            )
            result = ctx.invariant_mapping_service.analyze_collapse_risk(config)
            payload = InvariantLayerMappingService.collapse_risk_payload(result)
            return payload

    if "mc_geometry_atlas_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_atlas_inventory(
            source: str | None = None,
            domain: str | None = None,
        ) -> dict:
            """Get inventory of available probes across all atlases."""
            from modelcypher.core.domain.agents.unified_atlas import (
                AtlasDomain,
                AtlasSource,
                UnifiedAtlasInventory,
            )

            counts = UnifiedAtlasInventory.probe_count()
            total = UnifiedAtlasInventory.total_probe_count()
            filtered_count = total
            if source or domain:
                sources_filter = None
                domains_filter = None
                if source:
                    source_map = {
                        "sequence": AtlasSource.SEQUENCE_INVARIANT,
                        "semantic": AtlasSource.SEMANTIC_PRIME,
                        "gate": AtlasSource.COMPUTATIONAL_GATE,
                        "emotion": AtlasSource.EMOTION_CONCEPT,
                        "temporal": AtlasSource.TEMPORAL_CONCEPT,
                        "social": AtlasSource.SOCIAL_CONCEPT,
                        "moral": AtlasSource.MORAL_CONCEPT,
                        "compositional": AtlasSource.COMPOSITIONAL,
                        "philosophical": AtlasSource.PHILOSOPHICAL_CONCEPT,
                        "genealogy": AtlasSource.CONCEPTUAL_GENEALOGY,
                    }
                    if source.lower() in source_map:
                        sources_filter = {source_map[source.lower()]}
                if domain:
                    domain_map = {
                        "mathematical": AtlasDomain.MATHEMATICAL,
                        "logical": AtlasDomain.LOGICAL,
                        "linguistic": AtlasDomain.LINGUISTIC,
                        "mental": AtlasDomain.MENTAL,
                        "computational": AtlasDomain.COMPUTATIONAL,
                        "structural": AtlasDomain.STRUCTURAL,
                        "affective": AtlasDomain.AFFECTIVE,
                        "relational": AtlasDomain.RELATIONAL,
                        "temporal": AtlasDomain.TEMPORAL,
                        "spatial": AtlasDomain.SPATIAL,
                        "moral": AtlasDomain.MORAL,
                        "philosophical": AtlasDomain.PHILOSOPHICAL,
                    }
                    if domain.lower() in domain_map:
                        domains_filter = {domain_map[domain.lower()]}
                if sources_filter:
                    filtered = UnifiedAtlasInventory.probes_by_source(sources_filter)
                    if domains_filter:
                        filtered = [p for p in filtered if p.domain in domains_filter]
                    filtered_count = len(filtered)
                elif domains_filter:
                    filtered = UnifiedAtlasInventory.probes_by_domain(domains_filter)
                    filtered_count = len(filtered)
            return {
                "_schema": "mc.geometry.atlas.inventory.v1",
                "totalProbes": total,
                "filteredCount": filtered_count,
                "sources": {
                    "sequenceInvariant": {
                        "count": counts.get(AtlasSource.SEQUENCE_INVARIANT, 0),
                        "description": "Mathematical sequences and logical invariants",
                    },
                    "semanticPrime": {
                        "count": counts.get(AtlasSource.SEMANTIC_PRIME, 0),
                        "description": "NSM semantic primitives",
                    },
                    "computationalGate": {
                        "count": counts.get(AtlasSource.COMPUTATIONAL_GATE, 0),
                        "description": "Programming primitives",
                    },
                    "emotionConcept": {
                        "count": counts.get(AtlasSource.EMOTION_CONCEPT, 0),
                        "description": "Plutchik emotion wheel",
                    },
                    "temporalConcept": {
                        "count": counts.get(AtlasSource.TEMPORAL_CONCEPT, 0),
                        "description": "Temporal anchors (direction, duration, causality)",
                    },
                    "socialConcept": {
                        "count": counts.get(AtlasSource.SOCIAL_CONCEPT, 0),
                        "description": "Social structure probes (power, kinship, formality)",
                    },
                    "moralConcept": {
                        "count": counts.get(AtlasSource.MORAL_CONCEPT, 0),
                        "description": "Moral foundations and ethical valence",
                    },
                    "compositional": {
                        "count": counts.get(AtlasSource.COMPOSITIONAL, 0),
                        "description": "Semantic prime compositions",
                    },
                    "philosophicalConcept": {
                        "count": counts.get(AtlasSource.PHILOSOPHICAL_CONCEPT, 0),
                        "description": "Fundamental categories of thought",
                    },
                    "conceptualGenealogy": {
                        "count": counts.get(AtlasSource.CONCEPTUAL_GENEALOGY, 0),
                        "description": "Etymology and lineage probes",
                    },
                },
            }
