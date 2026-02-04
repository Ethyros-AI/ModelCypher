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
    mc model add      - Add/fetch a model
    mc model delete   - Delete a model
    mc model search   - Search HuggingFace Hub
    mc model info     - Inspect a model
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import typer

from modelcypher.cli.composition import (
    get_model_probe_service,
    get_model_search_service,
    get_model_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.presenters import model_payload, model_search_payload
from modelcypher.cli.warnings import warn_network
from modelcypher.core.domain.model_search import (
    MemoryFitStatus,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.utils.errors import ErrorDetail


app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@contextmanager
def _prevent_sleep() -> Generator[None, None, None]:
    """Prevent macOS from sleeping during long operations."""
    caffeinate_proc = None
    if sys.platform == "darwin":
        try:
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-isd"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            caffeinate_proc = None

    try:
        yield
    finally:
        if caffeinate_proc is not None:
            caffeinate_proc.terminate()
            try:
                caffeinate_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                caffeinate_proc.kill()


def _default_alias(source: str, is_repo: bool) -> str:
    if is_repo:
        return source.replace("/", "--")
    return Path(source).expanduser().resolve().name


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

    # Include profile metadata if available
    try:
        from modelcypher.core.domain.geometry.model_profile import ModelProfileStore

        store = ModelProfileStore()
        profile, identity = store.load(model_path)
        if profile:
            payload["profile"] = {
                "modelId": profile.model_id,
                "configHash": profile.config_hash,
                "weightsHash": profile.weights_hash,
                "computedSections": profile.computed_sections,
            }
        elif identity:
            payload["profile"] = {
                "modelId": identity.model_id,
                "configHash": identity.config_hash,
                "weightsHash": identity.weights_hash,
                "computedSections": [],
            }
    except Exception:
        payload["profile"] = None

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
    source: str = typer.Argument(
        ..., help="Hugging Face repo ID or local model path"
    ),
    alias: str | None = typer.Option(None, "--alias"),
    revision: str = typer.Option("main", "--revision"),
    architecture: str | None = typer.Option(None, "--architecture"),
    parameters: int | None = typer.Option(None, "--parameters"),
    default_chat: bool = typer.Option(False, "--default-chat"),
) -> None:
    """Add a model (fetch or register) and persist its identity profile.

    Examples:
        mc model add LiquidAI/LFM2.5-1.2B-Instruct
        mc model add ./models/my-model --alias my-model
    """
    from modelcypher.utils.security import trust_remote_code_enabled

    context = _context(ctx)
    service = get_model_service()
    probe_service = get_model_probe_service()

    source_path = Path(source).expanduser()
    is_local = source_path.exists()
    resolved_alias = alias or _default_alias(source, is_repo=not is_local)

    if is_local:
        resolved_path = str(source_path.resolve())
        probe_result = None
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
                        title="Model add failed",
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
            default_chat=default_chat,
        )

        payload = {
            "source": source,
            "localPath": resolved_path,
            "registeredID": resolved_alias,
            "detectedArchitecture": detected_arch,
            "parameterCount": detected_params,
        }
        write_output(payload, context.output_format, context.pretty)
        return

    warn_network(context, "Fetching model artifacts from Hugging Face Hub.")
    with _prevent_sleep():
        fetch_result = service.fetch_model(source, revision=revision, auto_register=False)

    local_path = fetch_result["localPath"]
    probe_result = None
    detected_arch = architecture or fetch_result.get("detectedArchitecture") or "unknown"
    detected_params = parameters
    if architecture is None or parameters is None:
        try:
            probe_result = probe_service.probe(local_path)
            if architecture is None:
                detected_arch = probe_result.architecture or detected_arch
            if parameters is None and probe_result.parameter_count:
                detected_params = probe_result.parameter_count
        except Exception as exc:
            typer.echo(f"Warning: probe failed: {exc}", err=True)

    service.register_model(
        resolved_alias,
        local_path,
        detected_arch,
        parameters=detected_params,
        default_chat=default_chat,
    )

    payload = {
        "source": source,
        "repoID": source,
        "localPath": local_path,
        "registeredID": resolved_alias,
        "detectedArchitecture": detected_arch,
        "parameterCount": detected_params,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("delete")
def model_delete(
    ctx: typer.Context,
    model_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
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


@app.command("search")
def model_search(
    ctx: typer.Context,
    query: str | None = typer.Argument(None),
    author: str | None = typer.Option(None, "--author"),
    library: str = typer.Option("backend", "--library"),
    quant: str | None = typer.Option(None, "--quant"),
    sort: str = typer.Option("downloads", "--sort"),
    limit: int = typer.Option(20, "--limit"),
    cursor: str | None = typer.Option(None, "--cursor"),
) -> None:
    """Search for models on HuggingFace Hub.

    Examples:
        mc model search llama
        mc model search llama --library backend --quant 4bit
        mc model search --author community --sort downloads
    """
    context = _context(ctx)
    warn_network(context, "Querying Hugging Face Hub for model metadata.")
    library_filter = _parse_model_search_library(library)
    quant_filter = _parse_model_search_quant(quant)
    sort_option = _parse_model_search_sort(sort)

    filters = ModelSearchFilters(
        query=query,
        architecture=None,
        max_size_gb=None,
        author=author,
        library=library_filter,
        quantization=quant_filter,
        sort_by=sort_option,
        limit=limit,
    )

    service = get_model_search_service()
    try:
        page = service.search(filters, cursor)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-5002",
            title="Model search failed",
            detail=str(exc),
            hint="Check your network connection. For private models, set HF_TOKEN.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        _print_model_search_text(page)
        return

    write_output(model_search_payload(page), context.output_format, context.pretty)


@app.command("info")
def model_info(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Inspect a model and surface its stored identity profile.

    Examples:
        mc model info ./models/llama-7b
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


# --- Helper Functions ---


def _parse_model_search_library(value: str) -> ModelSearchLibraryFilter:
    normalized = value.lower()
    for member in ModelSearchLibraryFilter:
        if normalized == member.value.lower():
            return member
    valid = ", ".join(m.value for m in ModelSearchLibraryFilter)
    raise typer.BadParameter(f"Invalid library filter '{value}'. Valid options: {valid}")


def _parse_model_search_quant(value: str | None) -> ModelSearchQuantization | None:
    if value is None:
        return None
    normalized = value.lower()
    for member in ModelSearchQuantization:
        if normalized == member.value.lower():
            return member
    valid = ", ".join(m.value for m in ModelSearchQuantization)
    raise typer.BadParameter(f"Invalid quantization filter '{value}'. Valid options: {valid}")


def _parse_model_search_sort(value: str) -> ModelSearchSortOption:
    normalized = value.lower().replace("_", "")
    for member in ModelSearchSortOption:
        if normalized == member.value.lower().replace("_", ""):
            return member
    valid = ", ".join(m.value for m in ModelSearchSortOption)
    raise typer.BadParameter(f"Invalid sort option '{value}'. Valid options: {valid}")


def _print_model_search_text(page: ModelSearchPage) -> None:
    if not page.models:
        write_output("No models found matching your query.", "text", False)
        return

    lines: list[str] = [f"Found {len(page.models)} models:\n"]
    for model in page.models:
        fit_indicator = ""
        if model.memory_fit_status == MemoryFitStatus.fits:
            fit_indicator = "[fits]"
        elif model.memory_fit_status == MemoryFitStatus.tight:
            fit_indicator = "[tight]"
        elif model.memory_fit_status == MemoryFitStatus.too_big:
            fit_indicator = "[too big]"

        header = f"{model.id} {fit_indicator}".rstrip()
        lines.append(header)
        downloads = _format_number(model.downloads)
        likes = _format_number(model.likes)
        lines.append(f"  Downloads: {downloads} | Likes: {likes}")
        if model.is_gated:
            lines.append("  [Gated - requires access request]")
        lines.append("")

    if page.has_more and page.next_cursor:
        lines.append(f"More results available. Use --cursor '{page.next_cursor}' for next page.")

    write_output("\n".join(lines).rstrip(), "text", False)


def _format_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
