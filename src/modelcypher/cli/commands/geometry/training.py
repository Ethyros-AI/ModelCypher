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

"""Geometry training CLI commands.

Provides commands for:
- Training status with geometric metrics
- Training history
- Instrumentation levels

Commands:
    mc geometry training status --job <id>
    mc geometry training history --job <id>
    mc geometry training levels
"""

from __future__ import annotations

import typer

from modelcypher.cli.composition import get_geometry_training_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
)

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def geometry_training_status(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
    format: str = typer.Option("full", "--format"),
    ai: bool = typer.Option(False, "--ai"),
) -> None:
    """Get geometric training status for a job.

    Examples:
        mc geometry training status --job abc123
        mc geometry training status --job abc123 --format summary
    """
    context = _context(ctx)
    if format not in {"full", "summary"}:
        raise typer.BadParameter("Format must be 'full' or 'summary'.")

    service = get_geometry_training_service()
    payload = service.training_status_payload(job_id, output_format=format, require_metrics=False)
    output = {
        "jobId": payload["jobId"],
        "step": payload["step"],
        "flatnessScore": payload["flatnessScore"],
        "flatnessAssessment": payload["flatnessAssessment"] if format == "full" else None,
        "gradientSNR": payload["gradientSNR"],
        "snrAssessment": payload["snrAssessment"] if format == "full" else None,
        "circuitBreakerSeverity": payload["circuitBreakerSeverity"],
        "circuitBreakerTripped": payload["circuitBreakerTripped"],
        "activeLayers": payload["activeLayers"],
        "perLayerGradientNorms": payload["perLayerGradientNorms"] if format == "full" else None,
        "nextActions": (
            [
                f"mc geometry training history --job {job_id}",
                f"mc geometry safety circuit-breaker --job {job_id}",
            ]
            if ai
            else None
        ),
    }

    if context.output_format == "text":
        lines = [
            "GEOMETRIC TRAINING STATUS",
            f"Job: {output['jobId']}",
            f"Step: {output['step']}",
        ]
        if output["flatnessScore"] is not None:
            assessment = output.get("flatnessAssessment") or ""
            lines.append(f"Flatness: {output['flatnessScore']:.3f} {f'({assessment})' if assessment else ''}".strip())
        if output["gradientSNR"] is not None:
            assessment = output.get("snrAssessment") or ""
            lines.append(
                f"Gradient SNR: {output['gradientSNR']:.2f} {f'({assessment})' if assessment else ''}".strip()
            )
        if output["circuitBreakerSeverity"] is not None:
            tripped = "TRIPPED" if output.get("circuitBreakerTripped") else "OK"
            lines.append(f"Circuit Breaker: {output['circuitBreakerSeverity']:.3f} ({tripped})")
        if output["activeLayers"]:
            lines.append(f"Active Layers: {', '.join(output['activeLayers'])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("history")
def geometry_training_history(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Get geometric training history for a job.

    Examples:
        mc geometry training history --job abc123
    """
    context = _context(ctx)
    service = get_geometry_training_service()
    payload = service.training_history_payload(job_id)

    if context.output_format == "text":
        lines = ["GEOMETRIC TRAINING HISTORY", payload["interpretation"]]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("levels")
def geometry_training_levels(ctx: typer.Context) -> None:
    """List geometric instrumentation levels.

    Examples:
        mc geometry training levels
    """
    context = _context(ctx)
    levels = [
        {"name": level.value, "description": level.description, "metricsCollected": level.metrics_collected}
        for level in GeometricInstrumentationLevel
    ]
    payload = {"levels": levels}
    if context.output_format == "text":
        lines = ["GEOMETRIC INSTRUMENTATION LEVELS"]
        for level in levels:
            lines.append(f"\n{level['name']}: {level['description']}")
            lines.append(f"  Metrics: {', '.join(level['metricsCollected'])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    write_output(payload, context.output_format, context.pretty)
