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

"""Geometry stitch CLI commands.

Provides commands for:
- Manifold stitch analysis
- Apply stitching between checkpoints

Commands:
    mc geometry stitch analyze --checkpoint <path1> --checkpoint <path2>
    mc geometry stitch apply --source <path> --target <path> --output <path>
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_stitch_service import GeometryStitchService
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def geometry_stitch_analyze(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(
        ..., "--checkpoint", help="Checkpoint paths (specify multiple times)"
    ),
) -> None:
    """Analyze manifold stitching between checkpoints.

    Examples:
        mc geometry stitch analyze --checkpoint ./ckpt1 --checkpoint ./ckpt2
    """
    context = _context(ctx)
    service = GeometryStitchService()

    try:
        result = service.analyze(checkpoints)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1005",
            title="Stitch analysis failed",
            detail=str(exc),
            hint="Ensure checkpoint paths exist and contain valid model files",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "checkpoints": checkpoints,
        "manifoldDistance": result.manifold_distance,
        "stitchingPoints": [
            {
                "layerName": sp.layer_name,
                "sourceDim": sp.source_dim,
                "targetDim": sp.target_dim,
                "qualityScore": sp.quality_score,
            }
            for sp in result.stitching_points
        ],
        "recommendedConfig": result.recommended_config,
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "STITCH ANALYSIS",
            f"Checkpoints: {len(checkpoints)}",
            f"Manifold Distance: {result.manifold_distance:.3f}",
            f"Stitching Points: {len(result.stitching_points)}",
            "",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("apply")
def geometry_stitch_apply(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source checkpoint path"),
    target: str = typer.Option(..., "--target", help="Target checkpoint path"),
    output: str = typer.Option(..., "--destination", "-d", help="Output path for stitched model"),
    learning_rate: float = typer.Option(0.01, "--learning-rate"),
    max_iterations: int = typer.Option(500, "--max-iterations"),
) -> None:
    """Apply stitching operation between checkpoints.

    Examples:
        mc geometry stitch apply --source ./ckpt1 --target ./ckpt2 --output ./stitched
    """
    context = _context(ctx)
    service = GeometryStitchService()

    config = {
        "learning_rate": learning_rate,
        "max_iterations": max_iterations,
        "use_procrustes_warm_start": True,
    }

    try:
        result = service.apply(source, target, output, config)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1006",
            title="Stitch apply failed",
            detail=str(exc),
            hint="Ensure source and target paths exist",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "outputPath": result.output_path,
        "stitchedLayers": result.stitched_layers,
        "qualityScore": result.quality_score,
    }

    if context.output_format == "text":
        lines = [
            "STITCH APPLIED",
            f"Output: {result.output_path}",
            f"Stitched Layers: {result.stitched_layers}",
            f"Quality Score: {result.quality_score:.3f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
