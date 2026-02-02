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

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

from .common import get_context, load_model_and_provider

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("knowledge-diff")
    def knowledge_diff(
        ctx: typer.Context,
        source_path: str = typer.Argument(..., help="Path to source model directory"),
        target_path: str = typer.Argument(..., help="Path to target model directory"),
        output_path: str | None = typer.Option(
            None, "--output-path", "-o", help="Save results to JSON file"
        ),
    ) -> None:
        """Diff knowledge states between source and target models.

        Identifies graft opportunities where source has dense representation
        but target is sparse. Positive opportunity scores indicate concepts
        where grafting would add value.

        Output includes ranked list of graft opportunities and the most
        negative opportunities (target already denser).
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
            ModelDensityProfile,
        )
        from modelcypher.core.domain.geometry.knowledge_diff import KnowledgeDiffer

        # Load source model
        logger.info("Loading source model: %s", source_path)
        _, _, source_backend, source_provider, source_num_layers = load_model_and_provider(
            source_path
        )

        # Load target model
        logger.info("Loading target model: %s", target_path)
        _, _, target_backend, target_provider, target_num_layers = load_model_and_provider(
            target_path
        )

        num_layers = min(source_num_layers, target_num_layers)
        resolved_layers = list(range(num_layers))

        probes = UnifiedAtlasInventory.all_probes()

        # Analyze source model
        logger.info("Analyzing source model density...")
        source_analyzer = KnowledgeDensityAnalyzer(backend=source_backend)
        source_profile = source_analyzer.analyze_model(
            probes, source_provider, resolved_layers
        )
        source_profile = ModelDensityProfile(
            model_path=source_path,
            layers=source_profile.layers,
            layer_profiles=source_profile.layer_profiles,
            domain_densities=source_profile.domain_densities,
            overall_density=source_profile.overall_density,
        )

        # Analyze target model
        logger.info("Analyzing target model density...")
        target_analyzer = KnowledgeDensityAnalyzer(backend=target_backend)
        target_profile = target_analyzer.analyze_model(
            probes, target_provider, resolved_layers
        )
        target_profile = ModelDensityProfile(
            model_path=target_path,
            layers=target_profile.layers,
            layer_profiles=target_profile.layer_profiles,
            domain_densities=target_profile.domain_densities,
            overall_density=target_profile.overall_density,
        )

        # Compute knowledge diff
        logger.info("Computing knowledge diff...")
        differ = KnowledgeDiffer()
        diff = differ.diff(source_profile, target_profile)

        # Build output payload
        payload = {
            "_schema": "mc.geometry.research.knowledge_diff.v1",
            "sourcePath": diff.source_path,
            "targetPath": diff.target_path,
            "layers": resolved_layers,
            "totalConcepts": diff.total_concepts,
            "overallSourceDensity": diff.overall_source_density,
            "overallTargetDensity": diff.overall_target_density,
            "overallOpportunity": diff.overall_opportunity,
            "positiveOpportunityCount": diff.positive_opportunity_count,
            "nonpositiveOpportunityCount": diff.nonpositive_opportunity_count,
            "domainDiffs": [
                {
                    "domain": dd.domain,
                    "meanSourceDensity": dd.mean_source_density,
                    "meanTargetDensity": dd.mean_target_density,
                    "meanOpportunity": dd.mean_opportunity,
                    "conceptCount": dd.concept_count,
                    "positiveOpportunityCount": dd.positive_opportunity_count,
                }
                for dd in diff.domain_diffs.values()
            ],
            "topGraftOpportunities": [
                {
                    "probeID": o.probe_id,
                    "name": o.name,
                    "domain": o.domain,
                    "layer": o.layer,
                    "sourceDensity": o.source_density,
                    "targetDensity": o.target_density,
                    "opportunityScore": o.opportunity_score,
                }
                for o in diff.ranked_opportunities[:50]
            ],
            "mostNegativeOpportunities": [
                {
                    "probeID": o.probe_id,
                    "name": o.name,
                    "domain": o.domain,
                    "layer": o.layer,
                    "opportunityScore": o.opportunity_score,
                }
                for o in sorted(
                    (opp for opp in diff.ranked_opportunities if opp.opportunity_score <= 0.0),
                    key=lambda x: x.opportunity_score,
                )[:20]
            ],
        }

        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Results saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "KNOWLEDGE STATE DIFF",
                f"Source: {source_path}",
                f"Target: {target_path}",
                f"Layers: {', '.join(str(layer) for layer in resolved_layers)}",
                "",
                f"Total concepts compared: {diff.total_concepts}",
                f"Overall source density: {diff.overall_source_density:.3f}",
                f"Overall target density: {diff.overall_target_density:.3f}",
                f"Overall opportunity: {diff.overall_opportunity:.3f}",
                "",
                f"Positive opportunities: {diff.positive_opportunity_count}",
                f"Non-positive opportunities: {diff.nonpositive_opportunity_count}",
                "",
                "Domain Summary:",
            ]
            for dd in sorted(
                diff.domain_diffs.values(), key=lambda x: x.mean_opportunity, reverse=True
            ):
                lines.append(
                    f"  {dd.domain}: opportunity={dd.mean_opportunity:.3f}, "
                    f"source={dd.mean_source_density:.3f}, target={dd.mean_target_density:.3f}"
                )

            lines.append("")
            lines.append("Top 10 Graft Opportunities (source > target density):")
            for o in diff.ranked_opportunities[:10]:
                lines.append(
                    f"  [{o.domain}] {o.name} L{o.layer}: "
                    f"score={o.opportunity_score:.3f} (source={o.source_density:.2f}, target={o.target_density:.2f})"
                )

            lines.append("")
            lines.append("Top 5 Most Negative Opportunities (target already denser):")
            for o in sorted(
                (opp for opp in diff.ranked_opportunities if opp.opportunity_score <= 0.0),
                key=lambda x: x.opportunity_score,
            )[:5]:
                lines.append(
                    f"  [{o.domain}] {o.name} L{o.layer}: score={o.opportunity_score:.3f}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
