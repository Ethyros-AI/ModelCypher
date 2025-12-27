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


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[index]


def _compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    sorted_vals = sorted(values)
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p50": _percentile(sorted_vals, 0.5),
        "p90": _percentile(sorted_vals, 0.9),
    }


@app.command("profile")
def entropy_profile(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model directory"),
    layers: int = typer.Option(
        None, "--layers", "-n", help="Number of layers to profile (auto-detected)"
    ),
) -> None:
    """Profile model entropy characteristics for merge planning.

    Measures actual layer entropy using the LinguisticCalorimeter
    and produces phase distribution and raw entropy statistics.

    Example:
        mc geometry merge-entropy profile ./my-model
        mc geometry merge-entropy profile /Volumes/CodeCypher/models/qwen2.5-3b --layers 48
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    context = _context(ctx)

    try:
        validator = EntropyMergeValidator()
        model_loader = MLXModelLoader()
        profile = validator.create_profile(model, model_loader=model_loader, num_layers=layers)

        # Build compact response
        critical_layers = [name for name, lp in profile.layer_profiles.items() if lp.is_critical]

        payload = {
            "_schema": "mc.merge.entropy.profile.v1",
            "modelName": profile.model_name,
            "meanEntropy": round(profile.mean_entropy, 3),
            "entropyVariance": round(profile.entropy_variance, 4),
            "dominantPhase": profile.dominant_phase.value,
            "criticalLayerCount": profile.critical_layer_count,
            "topCriticalLayers": critical_layers[:5],  # Top 5 only
        }

        if context.output_format == "text":
            lines = [
                "ENTROPY PROFILE",
                f"Model: {model}",
                f"Layers: {layers}",
                "",
                f"Mean Entropy: {profile.mean_entropy:.3f}",
                f"Entropy Variance: {profile.entropy_variance:.4f}",
                f"Dominant Phase: {profile.dominant_phase.value}",
                f"Critical Layers: {profile.critical_layer_count}",
            ]

            if critical_layers:
                lines.append(f"\nCritical Layer Names: {', '.join(critical_layers[:5])}")

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
    source: str = typer.Option(..., "--source", "-s", help="Path to source model directory"),
    target: str = typer.Option(..., "--target", "-t", help="Path to target model directory"),
    layers: int = typer.Option(None, "--layers", "-n", help="Number of layers (auto-detected)"),
) -> None:
    """Generate entropy-aware merge guidance.

    Analyzes both models by measuring actual layer entropy and provides
    per-layer alpha adjustments and smoothing sigmas for merging.

    Example:
        mc geometry merge-entropy guide --source ./model-a --target ./model-b
        mc geometry merge-entropy guide -s /path/to/source -t /path/to/target
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    context = _context(ctx)

    try:
        validator = EntropyMergeValidator()
        model_loader = MLXModelLoader()
        source_profile = validator.create_profile(
            source, model_loader=model_loader, num_layers=layers
        )
        target_profile = validator.create_profile(
            target, model_loader=model_loader, num_layers=layers
        )

        alpha_adjustments = validator.compute_alpha_adjustments(source_profile, target_profile)
        smoothing_sigmas = validator.compute_smoothing_sigmas(source_profile, target_profile)

        alpha_values = list(alpha_adjustments.values())
        sigma_values = list(smoothing_sigmas.values())
        alpha_stats = _compute_stats(alpha_values)
        sigma_stats = _compute_stats(sigma_values)

        sorted_alpha = sorted(alpha_adjustments.items(), key=lambda item: item[0])
        sorted_sigma = sorted(smoothing_sigmas.items(), key=lambda item: item[0])

        payload = {
            "_schema": "mc.merge.entropy.guide.v1",
            "sourceModel": source,
            "targetModel": target,
            "layerCount": len(alpha_adjustments),
            "alphaAdjustments": {name: round(value, 4) for name, value in sorted_alpha},
            "smoothingSigmas": {name: round(value, 4) for name, value in sorted_sigma},
            "alphaStats": {k: round(v, 4) for k, v in alpha_stats.items()},
            "sigmaStats": {k: round(v, 4) for k, v in sigma_stats.items()},
        }

        if context.output_format == "text":
            lowest_alpha = sorted(alpha_adjustments.items(), key=lambda item: item[1])[:5]
            highest_sigma = sorted(smoothing_sigmas.items(), key=lambda item: item[1], reverse=True)[
                :5
            ]

            lines = [
                "MERGE ENTROPY GUIDE",
                f"Source: {source}",
                f"Target: {target}",
                "",
                f"Layers Compared: {len(alpha_adjustments)}",
                f"Alpha Mean: {alpha_stats['mean']:.4f}",
                f"Alpha Min/Max: {alpha_stats['min']:.4f}/{alpha_stats['max']:.4f}",
                f"Alpha P50/P90: {alpha_stats['p50']:.4f}/{alpha_stats['p90']:.4f}",
                f"Sigma Mean: {sigma_stats['mean']:.4f}",
                f"Sigma Min/Max: {sigma_stats['min']:.4f}/{sigma_stats['max']:.4f}",
                f"Sigma P50/P90: {sigma_stats['p50']:.4f}/{sigma_stats['p90']:.4f}",
            ]

            if lowest_alpha:
                lines.append("\nLowest Alpha Layers:")
                for name, value in lowest_alpha:
                    lines.append(f"  {name}: alpha={value:.4f}")

            if highest_sigma:
                lines.append("\nHighest Sigma Layers:")
                for name, value in highest_sigma:
                    lines.append(f"  {name}: sigma={value:.4f}")

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
    source_ent: str = typer.Option(
        ..., "--source-ent", help="Source entropies JSON file or inline"
    ),
    target_ent: str = typer.Option(
        ..., "--target-ent", help="Target entropies JSON file or inline"
    ),
    merged_ent: str = typer.Option(
        ..., "--merged-ent", help="Merged entropies JSON file or inline"
    ),
    source_model: str = typer.Option("source", "--source-model", help="Source model name"),
    target_model: str = typer.Option("target", "--target-model", help="Target model name"),
) -> None:
    """Validate merge stability via entropy comparison.

    Compares entropy before and after merge to report raw delta statistics.
    Entropy values should be dict[layer_name, entropy_value].

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

        sorted_layers = sorted(
            validation.layer_validations.values(),
            key=lambda v: v.entropy_ratio,
            reverse=True,
        )
        top_layers = [
            {
                "layerName": v.layer_name,
                "entropyRatio": round(v.entropy_ratio, 4),
                "entropyDelta": round(v.entropy_delta, 4),
                "knowledgeRetentionScore": round(v.knowledge_retention_score, 4),
            }
            for v in sorted_layers[:5]
        ]

        payload = {
            "_schema": "mc.merge.entropy.validate.v1",
            "sourceModel": validation.source_model,
            "targetModel": validation.target_model,
            "knowledgeRetention": round(validation.mean_knowledge_retention, 3),
            "meanEntropyRatio": round(validation.mean_entropy_ratio, 3),
            "maxEntropyRatio": round(validation.max_entropy_ratio, 3),
            "entropyRatioStd": round(validation.entropy_ratio_std, 3),
            "totalLayersValidated": len(validation.layer_validations),
            "topEntropyRatioLayers": top_layers,
        }

        if context.output_format == "text":
            lines = [
                "MERGE VALIDATION",
                f"Source: {validation.source_model}",
                f"Target: {validation.target_model}",
                "",
                f"Knowledge Retention: {validation.mean_knowledge_retention:.1%}",
                f"Mean Entropy Ratio: {validation.mean_entropy_ratio:.2f}",
                f"Max Entropy Ratio: {validation.max_entropy_ratio:.2f}",
                f"Entropy Ratio Std: {validation.entropy_ratio_std:.2f}",
                f"Layers Validated: {len(validation.layer_validations)}",
            ]

            if top_layers:
                lines.append("\nTop Entropy Ratio Layers:")
                for layer in top_layers:
                    lines.append(
                        f"  {layer['layerName']}: ratio={layer['entropyRatio']:.2f} "
                        f"delta={layer['entropyDelta']:.2f} "
                        f"retention={layer['knowledgeRetentionScore']:.2f}"
                    )

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
