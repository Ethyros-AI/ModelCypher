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

"""Merge entropy validation CLI commands.

Provides commands for entropy-guided model merging analysis and validation.

Commands:
    mc geometry merge-entropy profile <model> [--layers N]
    mc geometry merge-entropy guide --source <path> --target <path>
    mc geometry merge-entropy validate --source-ent <json> --target-ent <json> --merged-ent <json>
"""

from __future__ import annotations

import json
from pathlib import Path


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeValidator,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("profile")
def entropy_profile(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model name or path"),
    layers: int = typer.Option(32, "--layers", "-n", help="Number of layers to profile"),
) -> None:
    """Profile model entropy characteristics for merge planning.

    Creates a simulated entropy profile showing phase distribution
    and merge risk assessment.

    Example:
        mc geometry merge-entropy profile llama3-8b
        mc geometry merge-entropy profile ./my-model --layers 48
    """
    context = _context(ctx)

    try:
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile(model, num_layers=layers)

        # Build compact response
        critical_layers = [
            name for name, lp in profile.layer_profiles.items()
            if lp.is_critical
        ]

        payload = {
            "_schema": "mc.merge.entropy.profile.v1",
            "modelName": profile.model_name,
            "meanEntropy": round(profile.mean_entropy, 3),
            "dominantPhase": profile.dominant_phase.value,
            "criticalLayerCount": profile.critical_layer_count,
            "mergeRisk": profile.merge_risk_level,
            "criticalLayers": critical_layers[:5],  # Top 5 only
            "interpretation": (
                f"Model has {profile.critical_layer_count} critical layers "
                f"({profile.merge_risk_level} risk) in {profile.dominant_phase.value} dominant phase"
            ),
            "nextActions": [
                f"mc geometry merge-entropy guide --source {model} --target <other>",
                "mc model merge with recommended alpha adjustments",
            ],
        }

        if context.output_format == "text":
            lines = [
                "ENTROPY PROFILE",
                f"Model: {model}",
                f"Layers: {layers}",
                "",
                f"Mean Entropy: {profile.mean_entropy:.3f}",
                f"Dominant Phase: {profile.dominant_phase.value}",
                f"Critical Layers: {profile.critical_layer_count}",
                f"Merge Risk: {profile.merge_risk_level}",
            ]

            if critical_layers:
                lines.append(f"\nCritical Layer Names: {', '.join(critical_layers[:5])}")

            lines.append(f"\nInterpretation: {payload['interpretation']}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("guide")
def entropy_guide(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", "-s", help="Source model name or path"),
    target: str = typer.Option(..., "--target", "-t", help="Target model name or path"),
    layers: int = typer.Option(32, "--layers", "-n", help="Number of layers"),
) -> None:
    """Generate entropy-aware merge recommendations.

    Analyzes both models and provides per-layer alpha adjustments
    and smoothing recommendations for stable merging.

    Example:
        mc geometry merge-entropy guide --source ./model-a --target ./model-b
        mc geometry merge-entropy guide -s llama3-8b -t mistral-7b --layers 32
    """
    context = _context(ctx)

    try:
        validator = EntropyMergeValidator()
        source_profile = validator.create_simulated_profile(source, num_layers=layers)
        target_profile = validator.create_simulated_profile(target, num_layers=layers)

        alpha_adjustments = validator.compute_alpha_adjustments(source_profile, target_profile)
        smoothing_sigmas = validator.compute_smoothing_sigmas(source_profile, target_profile)

        # Find critical layers needing adjustment
        critical_recommendations = {}
        for name in alpha_adjustments:
            alpha = alpha_adjustments[name]
            sigma = smoothing_sigmas.get(name, 1.0)
            if alpha < 1.0 or sigma > 1.0:
                critical_recommendations[name] = {
                    "alphaAdjust": round(alpha, 2),
                    "smoothingSigma": round(sigma, 2),
                }

        # Limit to top 5 critical
        top_critical = dict(list(critical_recommendations.items())[:5])

        # Global recommendation
        mean_alpha = sum(alpha_adjustments.values()) / len(alpha_adjustments) if alpha_adjustments else 1.0

        payload = {
            "_schema": "mc.merge.entropy.guide.v1",
            "sourceModel": source,
            "targetModel": target,
            "sourceRisk": source_profile.merge_risk_level,
            "targetRisk": target_profile.merge_risk_level,
            "criticalLayerCount": len(critical_recommendations),
            "globalAlphaAdjust": round(mean_alpha, 2),
            "recommendations": top_critical,
            "interpretation": (
                f"{len(critical_recommendations)} layers need conservative blending. "
                f"Recommended global alpha: {mean_alpha:.2f}"
            ),
            "nextActions": [
                f"mc model merge --alpha {mean_alpha:.2f}",
                "mc geometry merge-entropy validate after merge",
            ],
        }

        if context.output_format == "text":
            lines = [
                "MERGE ENTROPY GUIDE",
                f"Source: {source} (risk: {source_profile.merge_risk_level})",
                f"Target: {target} (risk: {target_profile.merge_risk_level})",
                "",
                f"Critical Layers: {len(critical_recommendations)}",
                f"Recommended Global Alpha: {mean_alpha:.2f}",
            ]

            if top_critical:
                lines.append("\nPer-Layer Recommendations:")
                for name, rec in top_critical.items():
                    lines.append(
                        f"  {name}: alpha={rec['alphaAdjust']}, sigma={rec['smoothingSigma']}"
                    )

            lines.append(f"\nInterpretation: {payload['interpretation']}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("validate")
def entropy_validate(
    ctx: typer.Context,
    source_ent: str = typer.Option(..., "--source-ent", help="Source entropies JSON file or inline"),
    target_ent: str = typer.Option(..., "--target-ent", help="Target entropies JSON file or inline"),
    merged_ent: str = typer.Option(..., "--merged-ent", help="Merged entropies JSON file or inline"),
    source_model: str = typer.Option("source", "--source-model", help="Source model name"),
    target_model: str = typer.Option("target", "--target-model", help="Target model name"),
) -> None:
    """Validate merge stability via entropy comparison.

    Compares entropy before and after merge to detect knowledge loss
    or instability. Entropy values should be dict[layer_name, entropy_value].

    Example:
        mc geometry merge-entropy validate \\
            --source-ent source_entropy.json \\
            --target-ent target_entropy.json \\
            --merged-ent merged_entropy.json

        # Or inline JSON:
        mc geometry merge-entropy validate \\
            --source-ent '{"layers.0": 2.0}' \\
            --target-ent '{"layers.0": 2.1}' \\
            --merged-ent '{"layers.0": 2.05}'
    """
    context = _context(ctx)

    def parse_entropy(value: str) -> dict[str, float]:
        """Parse entropy from file path or inline JSON."""
        if Path(value).exists():
            return json.loads(Path(value).read_text())
        return json.loads(value)

    try:
        source_entropies = parse_entropy(source_ent)
        target_entropies = parse_entropy(target_ent)
        merged_entropies = parse_entropy(merged_ent)

        validator = EntropyMergeValidator()
        validation = validator.validate_merge(
            source_entropies=source_entropies,
            target_entropies=target_entropies,
            merged_entropies=merged_entropies,
            source_model=source_model,
            target_model=target_model,
        )

        payload = {
            "_schema": "mc.merge.entropy.validate.v1",
            "sourceModel": validation.source_model,
            "targetModel": validation.target_model,
            "overallStability": validation.overall_stability.value,
            "knowledgeRetention": round(validation.mean_knowledge_retention, 3),
            "criticalLayers": validation.critical_layer_names[:5],
            "unstableLayers": validation.unstable_layer_names[:5],
            "isSafe": validation.is_safe,
            "interpretation": (
                f"Merge {'is safe' if validation.is_safe else 'has issues'}. "
                f"Knowledge retention: {validation.mean_knowledge_retention:.1%}. "
                f"Stability: {validation.overall_stability.value}"
            ),
            "nextActions": (
                ["mc merge perplexity to verify quality"]
                if validation.is_safe
                else ["Review critical layers", "Consider reducing alpha for unstable layers"]
            ),
        }

        if context.output_format == "text":
            status = "SAFE" if validation.is_safe else "UNSAFE"
            lines = [
                f"MERGE VALIDATION: {status}",
                f"Source: {validation.source_model}",
                f"Target: {validation.target_model}",
                "",
                f"Overall Stability: {validation.overall_stability.value}",
                f"Knowledge Retention: {validation.mean_knowledge_retention:.1%}",
                f"Layers Validated: {len(validation.layer_validations)}",
            ]

            if validation.critical_layer_names:
                lines.append(f"\nCritical Layers: {', '.join(validation.critical_layer_names[:5])}")

            if validation.unstable_layer_names:
                lines.append(f"Unstable Layers: {', '.join(validation.unstable_layer_names[:5])}")

            lines.append(f"\nInterpretation: {payload['interpretation']}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except json.JSONDecodeError as exc:
        write_error(f"Invalid JSON in entropy argument: {exc}", context.output_format)
        raise typer.Exit(1) from exc
    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc
