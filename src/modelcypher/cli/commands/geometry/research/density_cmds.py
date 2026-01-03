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
    @app.command("concept-density")
    def concept_density(
        ctx: typer.Context,
        model_path: str = typer.Argument(..., help="Path to the model directory"),
        output_path: str | None = typer.Option(
            None, "--output-path", "-o", help="Save results to JSON file"
        ),
    ) -> None:
        """Measure knowledge density per concept at a model layer.

        Knowledge density indicates how efficiently a concept is represented.
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
        )

        model, tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

        target_layer = num_layers - 1

        probes = UnifiedAtlasInventory.all_probes()

        analyzer = KnowledgeDensityAnalyzer(backend=backend)

        logger.info("Analyzing %d probes at layer %d...", len(probes), target_layer)
        profile = analyzer.analyze_layer(probes, provider, target_layer)

        # Build output payload
        densities = [c.density_score for c in profile.concept_densities]
        min_density = min(densities) if densities else 0.0
        max_density = max(densities) if densities else 0.0
        payload = {
            "_schema": "mc.geometry.research.concept_density.v1",
            "modelPath": model_path,
            "layer": profile.layer,
            "totalConcepts": len(profile.concept_densities),
            "meanDensity": profile.mean_density,
            "medianDensity": profile.median_density,
            "minDensity": min_density,
            "maxDensity": max_density,
            "concepts": [
                {
                    "probeID": c.probe_id,
                    "name": c.name,
                    "domain": c.domain,
                    "intrinsicDimension": c.intrinsic_dimension,
                    "densityScore": c.density_score,
                    "activationVariance": c.activation_variance,
                    "clusterTightness": c.cluster_tightness,
                }
                for c in sorted(profile.concept_densities, key=lambda x: x.density_score)
            ],
        }

        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Results saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "CONCEPT DENSITY ANALYSIS",
                f"Model: {model_path}",
                f"Layer: {target_layer}",
                f"Concepts: {len(profile.concept_densities)}",
                f"Mean Density: {profile.mean_density:.3f}",
                f"Median Density: {profile.median_density:.3f}",
                f"Min/Max Density: {min_density:.3f} / {max_density:.3f}",
                "",
                "Top 10 Lowest Density Concepts:",
            ]
            sparse = sorted(profile.concept_densities, key=lambda x: x.density_score)[:10]
            for c in sparse:
                lines.append(
                    f"  {c.name}: density={c.density_score:.3f}, dim={c.intrinsic_dimension:.1f}"
                )

            lines.append("")
            lines.append("Top 10 Highest Density Concepts:")
            dense = sorted(
                profile.concept_densities, key=lambda x: x.density_score, reverse=True
            )[:10]
            for c in dense:
                lines.append(
                    f"  {c.name}: density={c.density_score:.3f}, dim={c.intrinsic_dimension:.1f}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    @app.command("sparse-regions")
    def sparse_regions(
        ctx: typer.Context,
        model_path: str = typer.Argument(..., help="Path to the model directory"),
        output_path: str | None = typer.Option(
            None, "--output-path", "-o", help="Save results to JSON file"
        ),
    ) -> None:
        """Find sparse regions in a model where grafting would add value.

        Sparse regions are concepts where the model has incomplete/gap
        representations. These are opportunities for knowledge transfer
        from another model.

        Reports per-layer and per-domain density summaries, plus
        a ranked list of the lowest-density concepts.
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
            ModelDensityProfile,
        )

        model, tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

        resolved_layers = list(range(num_layers))
        probes = UnifiedAtlasInventory.all_probes()

        analyzer = KnowledgeDensityAnalyzer(backend=backend)

        logger.info("Analyzing %d probes across %d layers...", len(probes), len(resolved_layers))
        profile = analyzer.analyze_model(probes, provider, resolved_layers)
        profile = ModelDensityProfile(
            model_path=model_path,
            layers=profile.layers,
            layer_profiles=profile.layer_profiles,
            domain_densities=profile.domain_densities,
            overall_density=profile.overall_density,
        )

        all_concepts = [
            concept
            for layer_profile in profile.layer_profiles.values()
            for concept in layer_profile.concept_densities
        ]

        # Build output payload
        payload = {
            "_schema": "mc.geometry.research.sparse_regions.v1",
            "modelPath": model_path,
            "layers": resolved_layers,
            "overallDensity": profile.overall_density,
            "totalConcepts": len(all_concepts),
            "domainDensities": profile.domain_densities,
            "layerSummaries": [
                {
                    "layer": lp.layer,
                    "meanDensity": lp.mean_density,
                    "medianDensity": lp.median_density,
                    "conceptCount": len(lp.concept_densities),
                }
                for lp in profile.layer_profiles.values()
            ],
            "lowestDensityConcepts": [
                {
                    "probeID": c.probe_id,
                    "name": c.name,
                    "domain": c.domain,
                    "layer": c.layer,
                    "densityScore": c.density_score,
                    "intrinsicDimension": c.intrinsic_dimension,
                }
                for c in sorted(all_concepts, key=lambda x: x.density_score)[:100]
            ],
        }

        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Results saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "SPARSE REGION ANALYSIS",
                f"Model: {model_path}",
                f"Layers: {', '.join(str(layer) for layer in resolved_layers)}",
                f"Overall Density: {profile.overall_density:.3f}",
                f"Total Concepts: {len(all_concepts)}",
                "",
                "Layer Summary:",
            ]
            for lp in sorted(profile.layer_profiles.values(), key=lambda x: x.layer):
                lines.append(
                    f"  L{lp.layer}: concepts={len(lp.concept_densities)}, "
                    f"mean_density={lp.mean_density:.3f}"
                )

            lines.append("")
            lines.append("Domain Densities:")
            for domain, density in sorted(profile.domain_densities.items(), key=lambda x: x[1]):
                lines.append(f"  {domain}: {density:.3f}")

            lines.append("")
            lines.append("Top 15 Lowest Density Concepts:")
            for c in sorted(all_concepts, key=lambda x: x.density_score)[:15]:
                lines.append(
                    f"  [{c.domain}] {c.name} L{c.layer}: density={c.density_score:.3f}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
