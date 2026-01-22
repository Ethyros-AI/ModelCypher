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

"""Merge models via null-space knowledge transplant."""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.commands.model import prevent_sleep
from modelcypher.cli.composition import get_merge_pipeline_service, get_registry
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.validation import validate_model_path
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.logging import add_file_logger, remove_file_loggers

# Single entrypoint: sources + target + output directory.
# No modes, no fallbacks, no optional merge flags.
app = typer.Typer(invoke_without_command=True, no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _confirm_output_dir(
    context: CLIContext,
    output_dir: str,
    overwrite: bool,
) -> None:
    output_path = Path(output_dir).expanduser()
    if output_path.exists() and not output_path.is_dir():
        error = ErrorDetail(
            code="MC-1101",
            title="Invalid output path",
            detail=f"Output path exists and is not a directory: {output_path}",
            hint="Provide a directory path or remove the conflicting file.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    if output_path.exists() and any(output_path.iterdir()) and not (overwrite or context.yes):
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Output directory '{output_path}' is not empty. Continue?"):
            raise typer.Exit(code=1)


def _emit_pipeline_result(
    ctx: typer.Context,
    result,
    log_path: str | None,
) -> None:
    context = _context(ctx)
    payload = {
        "_schema": "mc.merge.pipeline.v1",
        "pipelineId": result.pipeline_id,
        "timestamp": result.timestamp,
        "sourceModel": result.source_model,
        "targetModel": result.target_model,
        "outputDir": result.output_dir,
        "logFile": log_path,
        "preMerge": {
            "domainsAnalyzed": result.pre_merge.domains_analyzed,
            "meanOverlap": result.pre_merge.mean_overlap,
            "meanSubspaceAlignment": result.pre_merge.mean_subspace_alignment,
            "meanCurvatureDivergence": result.pre_merge.mean_curvature_divergence,
            "meanDistance": result.pre_merge.mean_distance,
            "alignedPairs": result.pre_merge.aligned_pairs,
        },
        "mergeResult": {
            "layerCount": result.merge_result.get("layer_count"),
            "weightCount": result.merge_result.get("weight_count"),
            "meanPreservedFraction": result.merge_result.get("mean_preserved_fraction"),
        },
        "postMerge": {
            "layersTransplanted": result.post_merge.layers_transplanted,
            "weightsTransplanted": result.post_merge.weights_transplanted,
            "meanPreservedFraction": result.post_merge.mean_preserved_fraction,
            "meanCkaAfter": result.post_merge.mean_cka_after,
        },
        "timing": {
            "preMergeDurationS": round(result.pre_merge_duration_s, 2),
            "mergeDurationS": round(result.merge_duration_s, 2),
            "validationDurationS": round(result.validation_duration_s, 2),
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "MERGE PIPELINE RESULT",
            "=" * 70,
            f"Pipeline ID: {result.pipeline_id}",
            f"Source: {result.source_model}",
            f"Target: {result.target_model}",
            f"Output: {result.output_dir}",
            "",
            "PRE-MERGE ANALYSIS",
            f"  Domains: {', '.join(result.pre_merge.domains_analyzed)}",
            f"  Mean Overlap: {result.pre_merge.mean_overlap:.4f}",
            f"  Mean Subspace Alignment: {result.pre_merge.mean_subspace_alignment:.4f}",
            f"  Mean Curvature Divergence: {result.pre_merge.mean_curvature_divergence:.4f}",
            f"  Mean Distance: {result.pre_merge.mean_distance:.4f}",
            f"  Aligned Pairs: {result.pre_merge.aligned_pairs}",
            "",
            "MERGE RESULT",
            f"  Layers: {result.merge_result.get('layer_count')}",
            f"  Weights: {result.merge_result.get('weight_count')}",
            f"  Mean Preserved Fraction: {result.merge_result.get('mean_preserved_fraction', 0):.4f}",
            "",
            "POST-MERGE VALIDATION",
            f"  Mean Preserved Fraction: {result.post_merge.mean_preserved_fraction:.4f}",
            f"  Mean CKA After: {result.post_merge.mean_cka_after:.4f}",
            f"  Layers Transplanted: {result.post_merge.layers_transplanted}",
            f"  Weights Transplanted: {result.post_merge.weights_transplanted}",
            "",
            "TIMING",
            f"  Pre-merge: {result.pre_merge_duration_s:.2f}s",
            f"  Merge: {result.merge_duration_s:.2f}s",
            f"  Validation: {result.validation_duration_s:.2f}s",
        ]
        if log_path:
            lines.extend(["", f"LOG FILE: {log_path}"])
        lines.append("=" * 70)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _emit_batch_result(
    ctx: typer.Context,
    sources: list[str],
    target: str,
    output_dir: str,
    result,
    log_path: str | None,
) -> None:
    context = _context(ctx)
    payload = {
        "_schema": "mc.merge.batch.v1",
        "sources": sources,
        "targetModel": target,
        "outputDir": output_dir,
        "logFile": log_path,
        "mergeResult": {
            "layerCount": result.layer_count,
            "weightCount": result.weight_count,
            "meanPreservedFraction": result.mean_preserved_fraction,
        },
        "probe": result.probe_metrics,
        "density": result.density_metrics,
        "transplant": result.transplant_metrics,
        "geometry": result.geometry_metrics,
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "MERGE RESULT",
            "=" * 70,
            f"Sources: {len(sources)}",
            f"Target: {target}",
            f"Output: {output_dir}",
            "",
            "MERGE RESULT",
            f"  Layers: {result.layer_count}",
            f"  Weights: {result.weight_count}",
            f"  Mean Preserved Fraction: {result.mean_preserved_fraction:.4f}",
        ]
        if log_path:
            lines.extend(["", f"LOG FILE: {log_path}"])
        lines.append("=" * 70)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _run_single_merge(
    ctx: typer.Context,
    source: str,
    target: str,
    output_dir: str,
) -> None:
    context = _context(ctx)
    log_path = add_file_logger()
    if log_path:
        typer.echo(f"LOG FILE: {log_path}")
        typer.echo("")

    service = get_merge_pipeline_service()
    try:
        with prevent_sleep():
            result = service.run(
                source_path=source,
                target_path=target,
                output_dir=output_dir,
            )
        _emit_pipeline_result(ctx, result, log_path)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-1100",
            title="Pipeline failed",
            detail=str(exc),
            hint=f"Check model paths. Log file: {log_path}" if log_path else "Check model paths.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict())
        raise typer.Exit(code=1)
    finally:
        remove_file_loggers()


def _run_batch_merge(
    ctx: typer.Context,
    sources: list[str],
    target: str,
    output_dir: str,
) -> None:
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger

    context = _context(ctx)
    log_path = add_file_logger()
    if log_path:
        typer.echo(f"LOG FILE: {log_path}")
        typer.echo("")

    try:
        with prevent_sleep():
            backend = get_default_backend()
            registry = get_registry()
            merger = UnifiedGeometricMerger(
                model_loader=MLXModelLoader(),
                backend=backend,
                activation_provider=registry.activation_provider,
            )
            result = merger.merge_batch(
                source_paths=sources,
                target_path=target,
                output_dir=output_dir,
            )
        _emit_batch_result(ctx, sources, target, output_dir, result, log_path)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-1100",
            title="Merge failed",
            detail=str(exc),
            hint=f"Check model paths. Log file: {log_path}" if log_path else "Check model paths.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict())
        raise typer.Exit(code=1)
    finally:
        remove_file_loggers()


@app.command("run")
def merge_run(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", "-s", help="Path to source model"),
    target: str = typer.Option(
        ..., "--target", "-t", help="Path to target model (receives knowledge)"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output directory for merged model"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow writing into a non-empty output directory"
    ),
) -> None:
    """Merge one source model into a target."""
    context = _context(ctx)
    validate_model_path(source, context=context)
    validate_model_path(target, context=context)
    _confirm_output_dir(context, output_dir, overwrite)
    _run_single_merge(ctx, source, target, output_dir)


@app.command("batch")
def merge_batch(
    ctx: typer.Context,
    sources: list[str] = typer.Option(
        ..., "--source", "-s", help="Path to source model (repeat for multiple sources)"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="Path to target model (receives knowledge)"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output directory for merged model"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow writing into a non-empty output directory"
    ),
) -> None:
    """Merge multiple sources into one target."""
    context = _context(ctx)
    for source in sources:
        validate_model_path(source, context=context)
    validate_model_path(target, context=context)
    _confirm_output_dir(context, output_dir, overwrite)
    _run_batch_merge(ctx, sources, target, output_dir)


@app.command("deviation")
def merge_deviation(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", "-b"),
    current: str = typer.Option(..., "--current", "-c"),
) -> None:
    """Measure deviation from a baseline model."""
    context = _context(ctx)
    validate_model_path(baseline, context=context)
    validate_model_path(current, context=context)
    payload = {
        "_schema": "mc.merge.deviation.v1",
        "baseline": baseline,
        "current": current,
    }
    write_output(payload, context.output_format, context.pretty)


def _parse_channels(channels: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in channels:
        if ":" not in entry:
            raise ValueError("Invalid channel format. Expected name:path")
        name, path = entry.split(":", 1)
        if not name or not path:
            raise ValueError("Invalid channel format. Expected name:path")
        parsed.append((name, path))
    return parsed


@app.command("multi-channel")
def merge_multi_channel(
    ctx: typer.Context,
    channels: list[str] = typer.Option(
        ..., "--channel", "-c", help="Channel spec in name:path format"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="Path to target model (receives knowledge)"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output directory for merged model"
    ),
    routing: str = typer.Option("density", "--routing", help="Routing strategy"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow writing into a non-empty output directory"
    ),
) -> None:
    """Merge multiple modality channels into a target."""
    context = _context(ctx)
    try:
        parsed = _parse_channels(channels)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1110",
            title="Invalid channel specification",
            detail=str(exc),
            hint="Use --channel name:path (repeat for multiple channels).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    validate_model_path(target, context=context)
    _confirm_output_dir(context, output_dir, overwrite)
    payload = {
        "_schema": "mc.merge.multi_channel.v1",
        "channels": [{"name": name, "path": path} for name, path in parsed],
        "target": target,
        "outputDir": output_dir,
        "routing": routing,
        "status": "not_implemented",
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("bridge")
def merge_bridge(
    ctx: typer.Context,
    source: str = typer.Argument(...),
    target: str = typer.Argument(...),
    output: str = typer.Option(..., "--output", "-o"),
    samples: int = typer.Option(200, "--samples"),
) -> None:
    """Generate a cross-modal bridge between two models."""
    context = _context(ctx)
    validate_model_path(source, context=context)
    validate_model_path(target, context=context)
    payload = {
        "_schema": "mc.merge.bridge.v1",
        "source": source,
        "target": target,
        "output": output,
        "samples": samples,
        "status": "not_implemented",
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("apply-bridge")
def merge_apply_bridge(
    ctx: typer.Context,
    bridge_path: str = typer.Argument(...),
    input_path: str = typer.Argument(...),
    output: str = typer.Option(..., "--output", "-o"),
    inverse: bool = typer.Option(False, "--inverse", "-i"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
) -> None:
    """Apply a bridge transform to embeddings."""
    context = _context(ctx)
    payload = {
        "_schema": "mc.merge.apply_bridge.v1",
        "bridgePath": bridge_path,
        "inputPath": input_path,
        "output": output,
        "inverse": inverse,
        "normalize": normalize,
        "status": "not_implemented",
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("validate")
def merge_validate(
    ctx: typer.Context,
    model: str = typer.Argument(...),
    baseline: str | None = typer.Option(None, "--baseline", "-b"),
    output: str | None = typer.Option(None, "--output", "-o"),
    num_prompts: int = typer.Option(100, "--num-prompts"),
) -> None:
    """Validate merge quality against a baseline."""
    context = _context(ctx)
    validate_model_path(model, context=context)
    if baseline:
        validate_model_path(baseline, context=context)
    payload = {
        "_schema": "mc.merge.validate.v1",
        "model": model,
        "baseline": baseline,
        "output": output,
        "numPrompts": num_prompts,
        "status": "not_implemented",
    }
    write_output(payload, context.output_format, context.pretty)


@app.callback()
def merge_callback(
    ctx: typer.Context,
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Path to source model (repeat for multiple sources)"
    ),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Path to target model (receives knowledge)"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for merged model"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow writing into a non-empty output directory"
    ),
) -> None:
    """Merge models via null-space knowledge transplant."""
    context = _context(ctx)
    if ctx.invoked_subcommand is not None:
        return
    if not sources or not target or not output_dir:
        error = ErrorDetail(
            code="MC-1099",
            title="Missing merge options",
            detail="Missing required options: --source, --target, --output-dir",
            hint="Run `mc merge run --help` for usage.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    for source in sources:
        validate_model_path(source, context=context)
    validate_model_path(target, context=context)
    _confirm_output_dir(context, output_dir, overwrite)

    if len(sources) == 1:
        _run_single_merge(ctx, sources[0], target, output_dir)
    else:
        _run_batch_merge(ctx, sources, target, output_dir)
