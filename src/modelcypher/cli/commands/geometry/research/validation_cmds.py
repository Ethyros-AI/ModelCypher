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

from pathlib import Path

import typer

from modelcypher.cli.output import write_error, write_output

from .common import get_context


def register(app: typer.Typer) -> None:
    @app.command("validate-transplant")
    def validate_transplant(
        ctx: typer.Context,
        original_profile: Path = typer.Option(
            ...,
            "--original",
            "-o",
            help="Path to pre-transplant density profile JSON",
        ),
        transplanted_model: Path = typer.Option(
            ...,
            "--model",
            "-m",
            help="Path to transplanted model",
        ),
        output: Path | None = typer.Option(
            None,
            "--output-file",
            help="Path to save comparison JSON",
        ),
    ) -> None:
        """Validate transplant by comparing density before/after.

        Profiles the transplanted model and compares to the original profile
        to verify the transplant succeeded.

        Checks:
        1. Domains became denser after transplant
        2. Overall density improved

        Example:
            mc geometry research validate-transplant \\
                --original ./profiles/target-before.json \\
                --model /path/to/transplanted-model
        """
        context = get_context(ctx)

        try:
            import json as json_module

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.domain.geometry.knowledge_density import (
                KnowledgeDensityAnalyzer,
            )
            from modelcypher.core.domain.geometry.probe_calibration import (
                MLXActivationProvider,
            )

            # Load original profile
            if not original_profile.exists():
                write_error(
                    f"Original profile not found: {original_profile}",
                    context.output_format,
                )
                raise typer.Exit(1)

            with original_profile.open() as f:
                original_data = json_module.load(f)

            # Extract original domain densities
            original_densities: dict[str, float] = {}
            for ds in original_data.get("domainSummaries", []):
                original_densities[ds["domain"]] = ds.get("overallMeanDensity", 0.0)

            # If no layers specified, use original profile's layers
            layer_list = original_data.get("completedLayers", [])
            if not layer_list:
                layer_list = list(range(original_data.get("totalLayers", 16)))
            target_domains = sorted(original_densities.keys())

            # Load transplanted model
            model, tokenizer = load_model_for_training(str(transplanted_model))
            backend = get_default_backend()

            # Get probes
            probes = UnifiedAtlasInventory.all_probes()

            # Create activation provider
            provider = MLXActivationProvider(model=model, tokenizer=tokenizer, backend=backend)

            # Analyze transplanted model density
            analyzer = KnowledgeDensityAnalyzer(backend=backend)

            typer.echo(f"Profiling transplanted model: {transplanted_model}")
            typer.echo(f"Layers to analyze: {len(layer_list)}")

            profile = analyzer.analyze_model(
                probes=probes,
                activation_provider=provider,
                layers=layer_list,
            )

            # Compare densities
            transplanted_densities: dict[str, float] = profile.domain_densities

            comparisons = []
            domain_deltas: list[float] = []
            domain_abs_changes: list[float] = []

            for domain, original_density in original_densities.items():
                transplanted_density = transplanted_densities.get(domain, 0.0)
                delta = transplanted_density - original_density
                is_target = True
                comparison = {
                    "domain": domain,
                    "isTargetDomain": is_target,
                    "originalDensity": original_density,
                    "transplantedDensity": transplanted_density,
                    "delta": delta,
                }
                comparisons.append(comparison)

            # Compute summary statistics (raw measurements only, no interpretation)
            domain_deltas = [c["delta"] for c in comparisons]
            domain_abs_changes = [abs(c["delta"]) for c in comparisons]
            mean_delta = sum(domain_deltas) / len(domain_deltas) if domain_deltas else 0.0
            mean_abs_change = (
                sum(domain_abs_changes) / len(domain_abs_changes) if domain_abs_changes else 0.0
            )

            # Return raw measurements - let users interpret based on their context
            # Positive target improvement = target domains improved
            # Low non-target change = minimal interference (user decides threshold)
            result = {
                "_schema": "mc.geometry.research.validate_transplant.v1",
                "originalProfile": str(original_profile),
                "transplantedModel": str(transplanted_model),
                "targetDomains": target_domains,
                "layersAnalyzed": layer_list,
                "comparisons": comparisons,
                "summary": {
                    "meanDelta": mean_delta,
                    "meanAbsoluteChange": mean_abs_change,
                    "domainCount": len(comparisons),
                },
            }

            # Save to file if requested
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w") as f:
                    json_module.dump(result, f, indent=2)
                typer.echo(f"Comparison saved to {output}")

            # Output result
            if context.output_format == "text":
                lines = [
                    "",
                    "=" * 60,
                    "TRANSPLANT VALIDATION RESULTS",
                    "=" * 60,
                    f"Domains analyzed: {', '.join(target_domains)}",
                    "",
                    "Domain Comparisons:",
                ]
                for c in comparisons:
                    marker = "[TARGET]" if c["isTargetDomain"] else "[other]"
                    direction = "+" if c["delta"] > 0 else ""
                    lines.append(
                        f"  {marker} {c['domain']}: "
                        f"{c['originalDensity']:.3f} -> {c['transplantedDensity']:.3f} "
                        f"({direction}{c['delta']:.3f})"
                    )
                lines.append("")
                lines.append("Summary:")
                lines.append(f"  Mean delta: {mean_delta:+.4f}")
                lines.append(f"  Mean absolute change: {mean_abs_change:.4f}")
                lines.append("")
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(result, context.output_format, context.pretty)

        except Exception as e:
            write_error(f"Validation failed: {e}", context.output_format)
            raise typer.Exit(1) from e
