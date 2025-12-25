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

"""Geometry adapter analysis CLI commands.

Provides commands for:
- DARE sparsity analysis
- DoRA decomposition analysis

Commands:
    mc geometry adapter sparsity --checkpoint <path>
    mc geometry adapter decomposition --checkpoint <path>
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_adapter_service import GeometryAdapterService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("sparsity")
def geometry_adapter_sparsity(
    ctx: typer.Context,
    checkpoint_path: str = typer.Option(..., "--checkpoint"),
    base_path: str | None = typer.Option(None, "--base"),
) -> None:
    """Analyze DARE sparsity of a checkpoint.

    Examples:
        mc geometry adapter sparsity --checkpoint ./checkpoint
        mc geometry adapter sparsity --checkpoint ./checkpoint --base ./base-model
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    context = _context(ctx)
    model_loader = MLXModelLoader()
    service = GeometryAdapterService(model_loader=model_loader)
    analysis = service.analyze_dare(checkpoint_path, base_path)

    interpretation = (
        f"Effective sparsity {analysis.effective_sparsity:.2%}. "
        f"Recommended drop rate {analysis.recommended_drop_rate:.2f}."
    )
    output = {
        "checkpointPath": checkpoint_path,
        "baseModelPath": base_path,
        "effectiveSparsity": analysis.effective_sparsity,
        "interpretation": interpretation,
        "nextActions": [
            f"mc geometry adapter decomposition --checkpoint '{checkpoint_path}'",
            f"mc checkpoint export --path '{checkpoint_path}'",
        ],
    }

    if context.output_format == "text":
        lines = [
            "DARE SPARSITY ANALYSIS",
            f"Checkpoint: {output['checkpointPath']}",
        ]
        if base_path:
            lines.append(f"Base Model: {base_path}")
        lines.append(f"Effective Sparsity: {analysis.effective_sparsity:.3f}")
        lines.append(f"Recommended Drop Rate: {analysis.recommended_drop_rate:.2f}")
        lines.append("")
        lines.append(interpretation)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("decomposition")
def geometry_adapter_decomposition(
    ctx: typer.Context,
    checkpoint_path: str = typer.Option(..., "--checkpoint"),
    base_path: str | None = typer.Option(None, "--base"),
) -> None:
    """Analyze DoRA decomposition of a checkpoint.

    Examples:
        mc geometry adapter decomposition --checkpoint ./checkpoint
        mc geometry adapter decomposition --checkpoint ./checkpoint --base ./base-model
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    context = _context(ctx)
    model_loader = MLXModelLoader()
    service = GeometryAdapterService(model_loader=model_loader)
    result = service.analyze_dora(checkpoint_path, base_path)
    learning_type = service.dora_learning_type(result)
    interpretation = service.dora_interpretation(result)

    output = {
        "checkpointPath": checkpoint_path,
        "baseModelPath": base_path,
        "magnitudeChangeRatio": result.overall_magnitude_change,
        "directionalDrift": result.overall_directional_drift,
        "learningType": learning_type,
        "interpretation": interpretation,
        "nextActions": [
            f"mc geometry adapter sparsity --checkpoint '{checkpoint_path}'",
            f"mc checkpoint export --path '{checkpoint_path}'",
        ],
    }

    if context.output_format == "text":
        lines = [
            "DORA DECOMPOSITION ANALYSIS",
            f"Checkpoint: {output['checkpointPath']}",
        ]
        if base_path:
            lines.append(f"Base Model: {base_path}")
        lines.append(f"Magnitude Change Ratio: {result.overall_magnitude_change:.3f}")
        lines.append(f"Directional Drift: {result.overall_directional_drift:.3f}")
        lines.append(f"Learning Type: {learning_type}")
        lines.append("")
        lines.append(interpretation)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)
