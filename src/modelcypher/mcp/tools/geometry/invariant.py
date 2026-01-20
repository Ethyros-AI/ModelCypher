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
        ) -> dict:
            """Map layers between models using multi-atlas triangulation.

            Collapse threshold is derived from the activation variance distribution.
            No user parameters for thresholds.
            """
            source_path = require_existing_directory(sourcePath)
            target_path = require_existing_directory(targetPath)
            result = ctx.invariant_mapping_service.map_layers(
                str(source_path),
                str(target_path),
            )
            payload = ctx.invariant_mapping_service.result_payload(result)
            return payload

    if "mc_geometry_invariant_collapse_risk" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_invariant_collapse_risk(
            modelPath: str,
        ) -> dict:
            """Analyze layer collapse risk for a model.

            Collapse threshold is derived from the activation variance distribution.
            No user parameters for thresholds.
            """
            model_path = require_existing_directory(modelPath)
            result = ctx.invariant_mapping_service.analyze_collapse_risk(str(model_path))
            payload = ctx.invariant_mapping_service.collapse_risk_payload(result)
            return payload

    if "mc_geometry_atlas_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_atlas_inventory(
        ) -> dict:
            """Get inventory of available probes across all atlases."""
            from modelcypher.core.domain.agents.unified_atlas import (
                AtlasSource,
                UnifiedAtlasInventory,
            )

            counts = UnifiedAtlasInventory.probe_count()
            total = UnifiedAtlasInventory.total_probe_count()
            return {
                "_schema": "mc.geometry.atlas.inventory.v1",
                "totalProbes": total,
                "filteredCount": total,
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

    if "mc_geometry_anchor_invariance" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_anchor_invariance(
            sourceModelPath: str,
            targetModelPath: str,
            anchorPrefix: str = "invariant:",
        ) -> dict:
            """Analyze semantic anchor stability across model pairs.

            Measures how consistently semantic anchors maintain their representation
            geometry across different models. Uses geodesic cosine similarity on
            the representation manifold.

            Args:
                sourceModelPath: Path to the source model.
                targetModelPath: Path to the target model.
                anchorPrefix: Prefix for anchor probes (default: "invariant:").
            """
            from modelcypher.core.domain.geometry.anchor_invariance_analyzer import (
                AnchorInvarianceAnalyzer,
                RunInput,
            )
            from modelcypher.core.domain.geometry.manifold_stitcher import (
                ManifoldStitcher,
                ProbeSpace,
            )
            from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
                MetaphorConvergenceAnalyzer,
            )

            source_path = require_existing_directory(sourceModelPath)
            target_path = require_existing_directory(targetModelPath)

            mode = MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED

            # Get fingerprints using ManifoldStitcher
            stitcher = ManifoldStitcher()
            source_fingerprints = stitcher.fingerprint_model(
                str(source_path),
                probe_space=ProbeSpace.prelogits_hidden,
            )
            target_fingerprints = stitcher.fingerprint_model(
                str(target_path),
                probe_space=ProbeSpace.prelogits_hidden,
            )

            # Create run input
            run_input = RunInput(
                id="run-1",
                source=source_fingerprints,
                target=target_fingerprints,
            )

            # Run analysis
            analyzer = AnchorInvarianceAnalyzer()
            report = analyzer.analyze(
                runs=[run_input],
                align_mode=mode,
                anchor_prefix=anchorPrefix,
            )

            # Format response
            return {
                "_schema": "mc.geometry.anchor_invariance.v1",
                "anchorPrefix": report.anchor_prefix,
                "alignMode": report.align_mode.value,
                "runCount": len(report.runs),
                "summary": {
                    "anchorCount": report.summary.anchor_count,
                    "overallMeanCosine": report.summary.overall_mean_cosine,
                    "topAnchors": [
                        {
                            "anchorId": a.anchor_id,
                            "meanCosine": a.mean_cosine,
                            "stabilityScore": a.stability_score,
                        }
                        for a in report.summary.top_anchors
                    ],
                },
                "anchors": [
                    {
                        "anchorId": a.anchor_id,
                        "prompt": a.prompt,
                        "category": a.category,
                        "family": a.family,
                        "meanCosine": a.mean_cosine,
                        "stdCosine": a.std_cosine,
                        "stabilityScore": a.stability_score,
                    }
                    for a in report.anchors[:20]  # Limit to top 20
                ],
            }
