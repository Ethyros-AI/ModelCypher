"""Adapter and calibration CLI commands.

Provides commands for:
- Adapter inspection, projection, wrapping, smoothing, merging
- Calibration execution, status, application

Commands:
    mc adapter inspect <path>
    mc adapter merge <path1> <path2> --output-dir <dir>
    mc calibration run --model <path> --dataset <path>
"""

from __future__ import annotations

from typing import Optional

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
    output: str = typer.Option(..., "--output", help="Output path"),
) -> None:
    """Project adapter to target space.

    Examples:
        mc adapter project ./adapter --target-space llama --output ./projected
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
    output: str = typer.Option(..., "--output", help="Output path"),
) -> None:
    """Wrap adapter for MLX compatibility.

    Examples:
        mc adapter wrap-mlx ./adapter --output ./wrapped
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


@adapter_app.command("smooth")
def adapter_smooth(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to adapter"),
    output: str = typer.Option(..., "--output", help="Output path"),
    strength: float = typer.Option(0.1, "--strength"),
) -> None:
    """Apply smoothing to adapter weights.

    Examples:
        mc adapter smooth ./adapter --output ./smoothed --strength 0.2
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.adapter_service import AdapterService

    service = AdapterService()
    result = service.smooth(adapter_path, output, strength)

    payload = {
        "outputPath": result.output_path,
        "smoothedLayers": result.smoothed_layers,
        "varianceReduction": result.variance_reduction,
    }

    if context.output_format == "text":
        lines = [
            "ADAPTER SMOOTHED",
            f"Output: {result.output_path}",
            f"Layers: {result.smoothed_layers}",
            f"Variance Reduction: {result.variance_reduction:.2%}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@adapter_app.command("merge")
def adapter_merge(
    ctx: typer.Context,
    adapter_paths: list[str] = typer.Argument(..., help="Paths to adapters to merge (at least 2)"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output directory for merged adapter"),
    strategy: str = typer.Option("ties", "--strategy", help="Merge strategy: ties or dare-ties"),
    ties_topk: float = typer.Option(0.2, "--ties-topk", help="Top-k fraction for TIES (0.0 to 1.0)"),
    drop_rate: Optional[float] = typer.Option(None, "--drop-rate", help="Drop rate for DARE-TIES (0.0 to 1.0)"),
    recommend_ensemble: bool = typer.Option(False, "--recommend-ensemble", help="Compute ensemble routing recommendation"),
) -> None:
    """Merge multiple LoRA adapters using TIES/DARE strategies.

    Examples:
        mc adapter merge ./adapter1 ./adapter2 --output-dir ./merged
        mc adapter merge ./a1 ./a2 ./a3 --strategy dare-ties --drop-rate 0.5
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.adapter_service import AdapterService

    service = AdapterService()
    try:
        result = service.merge(
            adapter_paths=adapter_paths,
            output_dir=output_dir,
            strategy=strategy,
            ties_topk=ties_topk,
            drop_rate=drop_rate,
            recommend_ensemble=recommend_ensemble,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1008",
            title="Adapter merge failed",
            detail=str(exc),
            hint="Ensure all adapter paths exist and contain valid weights",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "outputPath": result.output_path,
        "strategy": result.strategy,
        "mergedModules": result.merged_modules,
        "ensembleRecommendation": result.ensemble_recommendation,
    }

    if context.output_format == "text":
        lines = [
            "ADAPTER MERGED",
            f"Output: {result.output_path}",
            f"Strategy: {result.strategy}",
            f"Merged Modules: {result.merged_modules}",
        ]
        if result.ensemble_recommendation:
            lines.append(f"Ensemble Weights: {result.ensemble_recommendation.get('weights', [])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Calibration commands


@calibration_app.command("run")
def calibration_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to calibration dataset"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Max samples"),
    method: str = typer.Option("minmax", "--method", help="Calibration method"),
) -> None:
    """Execute calibration on a model with a dataset.

    Examples:
        mc calibration run --model ./model --dataset ./data.jsonl
        mc calibration run --model ./model --dataset ./data.jsonl --method percentile
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.calibration_service import (
        CalibrationConfig,
        CalibrationService,
    )

    config = CalibrationConfig(
        batch_size=batch_size,
        max_samples=max_samples,
        calibration_method=method,
    )
    service = CalibrationService()

    try:
        result = service.run(model, dataset, config)
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
    output_path: Optional[str] = typer.Option(None, "--output-path", help="Output path"),
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
