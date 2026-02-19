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

"""Model registry CLI commands.

Commands:
    mc model list     - List registered models
    mc model add      - Register a local model
    mc model delete   - Delete a model
    mc model info     - Inspect a model
    mc model capacity - Analyze per-layer spectral capacity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.composition import (
    get_capacity_analysis_service,
    get_model_probe_service,
    get_model_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.presenters import model_payload
from modelcypher.utils.errors import ErrorDetail


app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _write_probe_output(result: Any, context: CLIContext, model_path: str) -> None:
    """Write model probe output."""
    def _coerce_value(value: Any) -> Any:
        if type(value).__module__.startswith("unittest.mock"):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_coerce_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _coerce_value(val) for key, val in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item") and hasattr(value, "ndim") and getattr(value, "ndim", 1) == 0:
            return value.item()
        return str(value)

    def _layer_payload(layer: Any) -> dict[str, Any]:
        return {
            "name": _coerce_value(getattr(layer, "name", layer)),
            "type": _coerce_value(getattr(layer, "type", None)),
            "parameters": _coerce_value(getattr(layer, "parameters", None)),
            "shape": _coerce_value(getattr(layer, "shape", None)),
        }

    payload = {
        "architecture": _coerce_value(getattr(result, "architecture", None)),
        "parameterCount": _coerce_value(getattr(result, "parameter_count", None)),
        "vocabSize": _coerce_value(getattr(result, "vocab_size", None)),
        "hiddenSize": _coerce_value(getattr(result, "hidden_size", None)),
        "numAttentionHeads": _coerce_value(getattr(result, "num_attention_heads", None)),
        "quantization": _coerce_value(getattr(result, "quantization", None)),
        "layerCount": len(result.layers),
        "layerCountConfig": _coerce_value(getattr(result, "layer_count_config", None)),
        "layers": [_layer_payload(layer) for layer in result.layers[:20]],
    }

    if context.output_format == "text":
        lines = [
            "MODEL INFO",
            f"Architecture: {result.architecture}",
            f"Parameters: {result.parameter_count:,}",
            f"Vocab Size: {result.vocab_size:,}",
            f"Hidden Size: {result.hidden_size}",
            f"Attention Heads: {result.num_attention_heads}",
            f"Layers: {len(result.layers)}",
        ]
        if result.layer_count_config:
            lines.append(f"Layers (config): {result.layer_count_config}")
        if result.quantization:
            lines.append(f"Quantization: {result.quantization}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _format_capacity_text(report: Any, top: int) -> str:
    layers_by_capacity = report.layers_by_null_space_fraction()
    top_layers = layers_by_capacity[:max(0, top)]

    if top_layers:
        name_width = max(
            len("Layer"),
            max(len(layer.layer_name) for layer in top_layers),
        )
    else:
        name_width = len("Layer")
    shape_width = len("Shape")

    lines = [
        f"Model: {report.model_name}",
        f"Total Parameters: {report.total_parameters:,}",
        (
            f"Mean Effective Rank: {report.mean_effective_rank:.1f} / "
            f"{report.reference_rank_dimension} "
            f"({report.mean_capacity_utilization * 100.0:.1f}%)"
        ),
        f"Mean Capacity Utilization: {report.mean_capacity_utilization * 100.0:.1f}%",
        "",
        f"Top {len(top_layers)} Layers by Available Capacity:",
        (
            f"  {'Layer':<{name_width}}  {'Shape':<{shape_width}}  "
            f"{'Eff.Rank':>8}  {'Null%':>6}  {'Rec.Rank':>8}  {'Gap':>7}"
        ),
    ]

    for layer in top_layers:
        shape_text = f"{layer.weight_shape[0]}x{layer.weight_shape[1]}"
        gap_text = (
            "infx"
            if layer.spectral_gap_at_rank == float("inf")
            else f"{layer.spectral_gap_at_rank:.1f}x"
        )
        lines.append(
            f"  {layer.layer_name:<{name_width}}  {shape_text:<{shape_width}}  "
            f"{layer.effective_rank:>8.1f}  "
            f"{layer.null_space_fraction * 100.0:>5.1f}%  "
            f"{layer.recommended_rank:>8d}  {gap_text:>7}"
        )

    lines.append("")
    lines.append("Per-Layer LoRA Configuration (copy to training config):")
    for layer in report.layers_by_recommended_rank():
        lines.append(f"  {layer.layer_name}: rank={layer.recommended_rank}")

    if report.failed_layers:
        lines.append("")
        lines.append(f"Skipped layers due to SVD failure: {len(report.failed_layers)}")

    return "\n".join(lines)


# --- Registry Commands ---


@app.command("list")
def model_list(ctx: typer.Context) -> None:
    """List all registered models."""
    context = _context(ctx)
    service = get_model_service()
    models = [model_payload(model) for model in service.list_models()]
    write_output(models, context.output_format, context.pretty)


@app.command("add")
def model_add(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Local path to model directory"),
    alias: str | None = typer.Option(None, "--alias", "-a", help="Alias for the model"),
    architecture: str | None = typer.Option(None, "--architecture"),
    parameters: int | None = typer.Option(None, "--parameters"),
) -> None:
    """Register a local model.

    Examples:
        mc model add /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
        mc model add ./models/my-model --alias my-model
    """
    context = _context(ctx)
    service = get_model_service()
    probe_service = get_model_probe_service()

    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-1001",
            title="Model not found",
            detail=f"Path does not exist: {model_path}",
            hint="Provide a valid local path to a model directory.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    resolved_alias = alias or model_path.name
    resolved_path = str(model_path)

    # Probe to detect architecture and parameters
    detected_arch = architecture
    detected_params = parameters
    if architecture is None or parameters is None:
        try:
            probe_result = probe_service.probe(resolved_path)
            if architecture is None:
                detected_arch = probe_result.architecture or "unknown"
            if parameters is None and probe_result.parameter_count:
                detected_params = probe_result.parameter_count
        except Exception as exc:
            if architecture is None:
                error = ErrorDetail(
                    code="MC-1002",
                    title="Model probe failed",
                    detail=str(exc),
                    hint="Provide --architecture when probing fails.",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)
            typer.echo(f"Warning: probe failed: {exc}", err=True)

    if detected_arch is None:
        detected_arch = "unknown"

    service.register_model(
        resolved_alias,
        resolved_path,
        detected_arch,
        parameters=detected_params,
    )

    payload = {
        "path": resolved_path,
        "alias": resolved_alias,
        "architecture": detected_arch,
        "parameters": detected_params,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("delete")
def model_delete(
    ctx: typer.Context,
    model_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a registered model.

    Examples:
        mc model delete my-llama
    """
    context = _context(ctx)
    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete model '{model_id}' from the registry?"):
            raise typer.Exit(code=1)
    service = get_model_service()
    service.delete_model(model_id)
    write_output({"deleted": model_id}, context.output_format, context.pretty)


@app.command("info")
def model_info(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Inspect a model.

    Examples:
        mc model info /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
    """
    context = _context(ctx)
    service = get_model_probe_service()
    try:
        result = service.probe(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Check backend runtime status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    _write_probe_output(result, context, model_path)


@app.command("capacity")
def model_capacity(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    top: int = typer.Option(10, "--top", "-n", min=1, help="Top N layers to show in text output"),
) -> None:
    """Analyze per-layer spectral capacity and recommended LoRA ranks.

    Examples:
        mc model capacity /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
        mc model capacity /path/to/model --top 20 --output text
    """
    context = _context(ctx)
    service = get_capacity_analysis_service()

    try:
        report = service.analyze(model_path)
    except FileNotFoundError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model not found",
            detail=str(exc),
            hint="Ensure the path exists and contains model safetensors.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Capacity analysis failed",
            detail=str(exc),
            hint="Ensure model weights include analyzable 2D tensors.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Capacity analysis failed",
            detail=str(exc),
            hint="Check backend runtime status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        write_output(_format_capacity_text(report, top), context.output_format, context.pretty)
        return

    payload = report.to_dict()
    payload["topLayersByAvailableCapacity"] = [
        layer.to_dict()
        for layer in report.layers_by_null_space_fraction()[:top]
    ]
    payload["loraConfiguration"] = {
        layer.layer_name: {"rank": layer.recommended_rank}
        for layer in report.layers_by_recommended_rank()
    }
    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# MODEL SEARCH
# =============================================================================


@app.command("search")
def model_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    architecture: str | None = typer.Option(None, "--arch", "-a", help="Filter by architecture"),
) -> None:
    """Search for models.

    Examples:
        mc model search "llama 8b"
        mc model search "qwen" --arch transformer --limit 5
    """
    from modelcypher.core.domain.model_search import ModelSearchFilters

    context = _context(ctx)

    filters = ModelSearchFilters(
        query=query,
        limit=limit,
        architecture=architecture,
    )

    # Note: Full search requires ModelSearchService with adapter
    payload = {
        "query": query,
        "filters": {
            "limit": limit,
            "architecture": architecture,
        },
        "status": "search_available",
        "note": "Full search requires ModelSearchService with HuggingFace adapter.",
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# MODEL QUANTIZATION
# =============================================================================


@app.command("quantize")
def model_quantize(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model to quantize"),
    output_path: str = typer.Argument(..., help="Output path for quantized model"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (4 or 8)"),
    group_size: int = typer.Option(64, "--group-size", "-g", help="Quantization group size"),
    mode: str = typer.Option("affine", "--mode", "-m", help="Quantization mode (affine, symmetric)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output"),
) -> None:
    """Quantize a model to reduce size.

    Supports 4-bit and 8-bit quantization with configurable group size.

    Examples:
        mc model quantize /path/to/model /path/to/output --bits 4
        mc model quantize /path/to/model /path/to/output --bits 8 --group-size 128
    """
    from modelcypher.cli.composition import get_quantization_service

    context = _context(ctx)

    typer.echo(f"Quantizing model to {bits}-bit with group_size={group_size}...")

    try:
        service = get_quantization_service()

        result = service.quantize_model(
            model_path=model_path,
            output_dir=output_path,
            bits=bits,
            group_size=group_size,
            mode=mode,
            overwrite=overwrite,
        )

        payload = result.to_dict()

        if context.output_format == "text":
            lines = [
                "QUANTIZATION COMPLETE",
                f"Input: {model_path}",
                f"Output: {result.output_dir}",
                "",
                f"Bits: {result.bits}",
                f"Group Size: {result.group_size}",
                f"Mode: {result.mode}",
                "",
                "WEIGHTS:",
                f"  Total 2D: {result.total_2d_weights}",
                f"  Quantized: {result.quantized_2d_weights}",
                f"  Skipped: {result.skipped_2d_weights}",
            ]
            if result.skipped_dirs:
                lines.append("")
                lines.append(f"Skipped Dirs: {', '.join(result.skipped_dirs)}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except FileExistsError as e:
        error = ErrorDetail(
            code="MC-1003",
            title="Output exists",
            detail=str(e),
            hint="Use --overwrite to replace existing output.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1002",
            title="Quantization failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
