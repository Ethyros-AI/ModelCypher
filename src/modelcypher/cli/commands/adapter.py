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

"""Adapter CLI commands.

Provides commands for:
- Adapter inspection, projection, wrapping

Commands:
    mc adapter inspect <path>
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

adapter_app = typer.Typer(no_args_is_help=True)
calibration_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@adapter_app.command("inspect")
def adapter_inspect(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to adapter directory"),
) -> None:
    """Inspect adapter for detailed analysis.

    Examples:
        mc adapter inspect ./adapter
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.adapter_service import AdapterService

    service = AdapterService()
    try:
        result = service.inspect(adapter_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1007",
            title="Adapter inspect failed",
            detail=str(exc),
            hint="Ensure the path points to a valid adapter directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "rank": result.rank,
        "alpha": result.alpha,
        "targetModules": result.target_modules,
        "sparsity": result.sparsity,
        "parameterCount": result.parameter_count,
        "layerCount": len(result.layer_analysis),
    }

    if context.output_format == "text":
        lines = [
            "ADAPTER INSPECTION",
            f"Rank: {result.rank}",
            f"Alpha: {result.alpha}",
            f"Sparsity: {result.sparsity:.2%}",
            f"Parameters: {result.parameter_count:,}",
            f"Layers: {len(result.layer_analysis)}",
        ]
        if result.target_modules:
            lines.append(f"Target Modules: {', '.join(result.target_modules)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@adapter_app.command("project")
def adapter_project(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to adapter"),
    target_space: str = typer.Option("default", "--target-space"),
    output: str = typer.Option(..., "--output-path", "-o", help="Output path"),
) -> None:
    """Project adapter to target space.

    Examples:
        mc adapter project ./adapter --target-space llama --output-path ./projected
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.adapter_service import AdapterService

    service = AdapterService()
    result = service.project(adapter_path, target_space, output)

    payload = {
        "outputPath": result.output_path,
        "projectedLayers": result.projected_layers,
    }

    write_output(payload, context.output_format, context.pretty)


@adapter_app.command("wrap-mlx")
def adapter_wrap_mlx(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to adapter"),
    output: str = typer.Option(..., "--output-path", "-o", help="Output path"),
) -> None:
    """Wrap adapter for MLX compatibility.

    Examples:
        mc adapter wrap-mlx ./adapter --output-path ./wrapped
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.adapter_service import AdapterService

    service = AdapterService()
    result = service.wrap_mlx(adapter_path, output)

    payload = {
        "outputPath": result.output_path,
        "wrappedLayers": result.wrapped_layers,
    }

    write_output(payload, context.output_format, context.pretty)


# Calibration commands


@calibration_app.command("run")
def calibration_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to calibration dataset"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Max samples"),
    method: str = typer.Option("minmax", "--method", help="Calibration method"),
) -> None:
    """Execute calibration on a model with a dataset.

    Examples:
        mc calibration run --model ./model --dataset ./data.jsonl
        mc calibration run --model ./model --dataset ./data.jsonl --method percentile
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.calibration_service import CalibrationService
    service = CalibrationService()

    try:
        result = service.run(
            model,
            dataset,
            batch_size=batch_size,
            max_samples=max_samples,
            calibration_method=method,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1009",
            title="Calibration failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "calibrationId": result.calibration_id,
        "modelPath": result.model_path,
        "datasetPath": result.dataset_path,
        "status": result.status,
        "startedAt": result.started_at,
        "config": result.config,
        "metrics": result.metrics,
    }

    write_output(payload, context.output_format, context.pretty)


@calibration_app.command("status")
def calibration_status(
    ctx: typer.Context,
    calibration_id: str = typer.Argument(..., help="Calibration ID"),
) -> None:
    """Get status of a calibration operation.

    Examples:
        mc calibration status abc123
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.calibration_service import CalibrationService

    service = CalibrationService()

    try:
        result = service.status(calibration_id)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2009",
            title="Calibration not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "calibrationId": result.calibration_id,
        "status": result.status,
        "progress": result.progress,
        "currentStep": result.current_step,
        "totalSteps": result.total_steps,
        "metrics": result.metrics,
        "error": result.error,
    }

    write_output(payload, context.output_format, context.pretty)


@calibration_app.command("apply")
def calibration_apply(
    ctx: typer.Context,
    calibration_id: str = typer.Argument(..., help="Calibration ID"),
    model: str = typer.Option(..., "--model", help="Path to model"),
    output_path: str | None = typer.Option(None, "--output-path", help="Output path"),
) -> None:
    """Apply calibration results to a model.

    Examples:
        mc calibration apply abc123 --model ./model
        mc calibration apply abc123 --model ./model --output-path ./calibrated
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.calibration_service import CalibrationService

    service = CalibrationService()

    try:
        result = service.apply(calibration_id, model, output_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-3009",
            title="Calibration apply failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "calibrationId": result.calibration_id,
        "modelPath": result.model_path,
        "outputPath": result.output_path,
        "appliedAt": result.applied_at,
        "metrics": result.metrics,
    }

    write_output(payload, context.output_format, context.pretty)
