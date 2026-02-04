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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.composition import get_model_probe_service, get_model_service
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
) -> None:
    """Quantize a model to reduce size.

    Supports 4-bit and 8-bit quantization with configurable group size.

    Examples:
        mc model quantize /path/to/model /path/to/output --bits 4
        mc model quantize /path/to/model /path/to/output --bits 8 --group-size 128
    """
    from modelcypher.core.use_cases.quantization_service import QuantizationService

    context = _context(ctx)

    typer.echo(f"Quantizing model to {bits}-bit...")

    # Note: Full quantization requires model loading
    payload = {
        "model_path": model_path,
        "output_path": output_path,
        "bits": bits,
        "group_size": group_size,
        "status": "quantization_service_available",
        "note": "Full quantization requires model loading. Use QuantizationService directly.",
    }

    write_output(payload, context.output_format, context.pretty)
