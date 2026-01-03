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

Provides commands for entropy-based model analysis and merge validation.

Commands:
    mc geometry merge-entropy profile <model>
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
    model: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Profile model entropy characteristics for merge planning.

    Measures actual layer entropy using Entropy-Lens projection and produces
    raw entropy statistics per layer.

    Example:
        mc geometry merge-entropy profile ./my-model
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    context = _context(ctx)

    try:
        validator = EntropyMergeValidator()
        model_loader = MLXModelLoader()
        profile = validator.create_profile(model, model_loader=model_loader)

        # Sort layers by entropy for reporting
        sorted_layers = sorted(
            profile.layer_profiles.values(),
            key=lambda lp: lp.mean_entropy,
            reverse=True,
        )
        top_entropy_layers = [lp.layer_name for lp in sorted_layers[:5]]

        payload = {
            "_schema": "mc.merge.entropy.profile.v1",
            "modelName": profile.model_name,
            "meanEntropy": round(profile.mean_entropy, 3),
            "entropyVariance": round(profile.entropy_variance, 4),
            "layerCount": len(profile.layer_profiles),
            "topEntropyLayers": top_entropy_layers,
        }

        if context.output_format == "text":
            lines = [
                "ENTROPY PROFILE",
                f"Model: {model}",
                f"Layers: {len(profile.layer_profiles)}",
                "",
                f"Mean Entropy: {profile.mean_entropy:.3f}",
                f"Entropy Variance: {profile.entropy_variance:.4f}",
            ]

            if top_entropy_layers:
                lines.append(f"\nHighest Entropy Layers: {', '.join(top_entropy_layers)}")

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
            source_model="source",
            target_model="target",
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
