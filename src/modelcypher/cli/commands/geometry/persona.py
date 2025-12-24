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

"""Geometry persona vector CLI commands.

Provides commands for extracting and monitoring persona vectors
(helpful, harmless, honest) during model training.

Commands:
    mc geometry persona traits
    mc geometry persona extract --positive <file> --negative <file>
    mc geometry persona drift --positions <file> --step <int>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_persona_service import GeometryPersonaService
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("traits")
def geometry_persona_traits(ctx: typer.Context):
    """List all standard persona traits for vector extraction."""
    context = _context(ctx)
    service = GeometryPersonaService()

    traits = service.list_traits()
    payload = service.traits_payload(traits)
    payload["nextActions"] = [
        "mc geometry persona extract to extract a persona vector",
        "mc geometry persona drift to measure drift during training",
    ]

    if context.output_format == "text":
        lines = ["PERSONA TRAITS", ""]
        for trait in traits:
            lines.append(f"  {trait.id}: {trait.name}")
            lines.append(f"    {trait.description}")
            lines.append(f"    Prompts: +{trait.positive_prompt_count} / -{trait.negative_prompt_count}")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("extract")
def geometry_persona_extract(
    ctx: typer.Context,
    positive_file: Path = typer.Option(..., "--positive", "-p", help="JSON file with positive activations"),
    negative_file: Path = typer.Option(..., "--negative", "-n", help="JSON file with negative activations"),
    trait_id: str = typer.Option(..., "--trait", "-t", help="Trait ID (helpful, harmless, honest)"),
    layer_index: int = typer.Option(..., "--layer", "-l", help="Layer index"),
    model_id: str = typer.Option("unknown", "--model", "-m", help="Model identifier"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", is_flag=True, flag_value=True, help="Normalize direction vector"),
):
    """
    Extract a persona vector from contrastive activations.

    Positive activations from trait-positive prompts, negative from trait-negative.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    positive = json.loads(Path(positive_file).read_text())
    negative = json.loads(Path(negative_file).read_text())

    vector = service.extract_persona_vector(
        positive_activations=positive,
        negative_activations=negative,
        trait_id=trait_id,
        layer_index=layer_index,
        model_id=model_id,
        normalize=normalize,
    )

    if vector is None:
        write_error(
            ErrorDetail(
                code="MC-4011",
                message="Failed to extract persona vector",
                detail="Insufficient data or correlation below threshold",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    payload = service.persona_vector_payload(vector)
    payload["nextActions"] = [
        "mc geometry persona drift to measure training drift",
        "mc safety persona-drift for safety monitoring",
    ]

    if context.output_format == "text":
        lines = [
            "PERSONA VECTOR",
            f"Trait: {vector.name} ({vector.id})",
            f"Model: {vector.model_id}",
            f"Layer: {vector.layer_index}",
            f"Hidden Size: {vector.hidden_size}",
            f"Strength: {vector.strength:.4f}",
            f"Correlation: {vector.correlation_coefficient:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("drift")
def geometry_persona_drift(
    ctx: typer.Context,
    positions_file: Path = typer.Option(..., "--positions", "-p", help="JSON file with position measurements"),
    step: int = typer.Option(..., "--step", "-s", help="Training step number"),
    threshold: float = typer.Option(0.2, "--threshold", "-t", help="Drift threshold"),
):
    """
    Compute drift metrics from position measurements during training.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    positions = json.loads(Path(positions_file).read_text())
    metrics = service.compute_drift(
        positions=positions,
        step=step,
        drift_threshold=threshold,
    )

    payload = service.drift_metrics_payload(metrics)
    payload["nextActions"] = [
        "mc safety circuit-breaker if drift is significant",
        "mc train pause to halt training if needed",
    ]

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT METRICS",
            f"Step: {metrics.step}",
            f"Overall Drift: {metrics.overall_drift_magnitude:.4f}",
            f"Significant Drift: {'Yes' if metrics.has_significant_drift else 'No'}",
            "",
            metrics.interpretation,
        ]
        if metrics.drifting_traits:
            lines.append(f"Drifting Traits: {', '.join(metrics.drifting_traits)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
