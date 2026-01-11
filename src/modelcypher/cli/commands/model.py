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

"""Model management CLI commands.

Provides commands for:
- Model listing, addition, deletion, fetching
- Model search via HuggingFace Hub
- Model probing for architecture details
- Model merge validation
- Alignment analysis between models

Commands:
    mc model list
    mc model add <repo_id|path> [--alias]
    mc model register <alias> --path <path>
    mc model search <query>
    mc model info <path>
    mc model probe <path>
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import typer


@contextmanager
def prevent_sleep() -> Generator[None, None, None]:
    """Context manager to prevent macOS from sleeping during long operations.

    Uses caffeinate on macOS to prevent idle sleep (-i) and system sleep (-s).
    On other platforms, this is a no-op.
    """
    caffeinate_proc = None
    if sys.platform == "darwin":
        try:
            # -i: prevent idle sleep, -s: prevent system sleep, -d: prevent display sleep
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-isd"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            # caffeinate not available, continue without it
            pass

    try:
        yield
    finally:
        if caffeinate_proc is not None:
            caffeinate_proc.terminate()
            try:
                caffeinate_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                caffeinate_proc.kill()

# Late imports to keep prevent_sleep() context manager lightweight
from modelcypher.cli.composition import (  # noqa: E402
    get_model_probe_service,
    get_model_search_service,
    get_model_service,
)
from modelcypher.cli.context import CLIContext  # noqa: E402
from modelcypher.cli.output import write_error, write_output  # noqa: E402
from modelcypher.cli.presenters import model_payload, model_search_payload  # noqa: E402
from modelcypher.core.domain.model_search import (  # noqa: E402
    MemoryFitStatus,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.utils.errors import ErrorDetail  # noqa: E402

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _default_alias(source: str, is_repo: bool) -> str:
    if is_repo:
        return source.replace("/", "--")
    return Path(source).expanduser().resolve().name


def _write_probe_output(
    result: Any,
    context: CLIContext,
    include_profile: bool,
    model_path: str,
) -> None:
    payload = {
        "architecture": result.architecture,
        "parameterCount": result.parameter_count,
        "vocabSize": result.vocab_size,
        "hiddenSize": result.hidden_size,
        "numAttentionHeads": result.num_attention_heads,
        "quantization": result.quantization,
        "layerCount": len(result.layers),
        "layerCountTensors": len(result.layers),
        "layerCountConfig": result.layer_count_config,
        "layers": [
            {
                "name": layer.name,
                "type": layer.type,
                "parameters": layer.parameters,
                "shape": layer.shape,
            }
            for layer in result.layers[:20]
        ],
    }

    if include_profile:
        try:
            from modelcypher.core.domain.geometry.model_profile import ModelProfileStore

            store = ModelProfileStore()
            profile, identity = store.load(model_path)
            if profile:
                payload["profile"] = {
                    "modelId": profile.model_id,
                    "configHash": profile.config_hash,
                    "weightsHash": profile.weights_hash,
                    "profilePath": str(store.profile_path(profile.model_id)),
                    "sidecarPath": str(store.sidecar_path(profile.model_path)),
                    "computedSections": profile.computed_sections,
                }
            else:
                payload["profile"] = {
                    "modelId": identity.model_id,
                    "configHash": identity.config_hash,
                    "weightsHash": identity.weights_hash,
                    "profilePath": str(store.profile_path(identity.model_id)),
                    "sidecarPath": str(store.sidecar_path(identity.model_path)),
                    "computedSections": [],
                }
        except Exception:
            payload["profile"] = None

    if context.output_format == "text":
        lines = [
            "MODEL PROBE",
            f"Architecture: {result.architecture}",
            f"Parameters: {result.parameter_count:,}",
            f"Vocab Size: {result.vocab_size:,}",
            f"Hidden Size: {result.hidden_size}",
            f"Attention Heads: {result.num_attention_heads}",
            f"Layers (tensors): {len(result.layers)}",
        ]
        if result.layer_count_config:
            lines.append(f"Layers (config): {result.layer_count_config}")
        if result.quantization:
            lines.append(f"Quantization: {result.quantization}")
        if include_profile and payload.get("profile"):
            profile = payload["profile"]
            lines.append(f"Model ID: {profile.get('modelId', '')}")
            lines.append(f"Config Hash: {profile.get('configHash', '')}")
            lines.append(f"Weights Hash: {profile.get('weightsHash', '')}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


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
        if probe_result is None:
            try:
                probe_service.probe(resolved_path)
            except Exception as exc:
                typer.echo(f"Warning: profile probe failed: {exc}", err=True)

        payload = {
            "source": source,
            "localPath": resolved_path,
            "registeredID": resolved_alias,
            "detectedArchitecture": detected_arch,
            "parameterCount": detected_params,
        }
        write_output(payload, context.output_format, context.pretty)
        return

    with prevent_sleep():
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
    if probe_result is None:
        try:
            probe_service.probe(local_path)
        except Exception as exc:
            typer.echo(f"Warning: profile probe failed: {exc}", err=True)

    payload = {
        "source": source,
        "repoID": source,
        "localPath": local_path,
        "registeredID": resolved_alias,
        "detectedArchitecture": detected_arch,
        "parameterCount": detected_params,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("register")
def model_register(
    ctx: typer.Context,
    alias: str = typer.Argument(...),
    path: str = typer.Option(..., "--path"),
    architecture: str = typer.Option(..., "--architecture"),
    parameters: int | None = typer.Option(None, "--parameters"),
    default_chat: bool = typer.Option(False, "--default-chat"),
) -> None:
    """Register a local model.

    Examples:
        mc model register my-llama --path ./models/llama --architecture llama
    """
    typer.echo(
        "DEPRECATED: use `mc model add <path> --alias <alias>` instead.",
        err=True,
    )
    context = _context(ctx)
    service = get_model_service()
    service.register_model(
        alias, path, architecture, parameters=parameters, default_chat=default_chat
    )
    try:
        probe_service = get_model_probe_service()
        probe_service.probe(path)
    except Exception as exc:
        typer.echo(f"Warning: profile probe failed: {exc}", err=True)
    write_output({"registered": alias}, context.output_format, context.pretty)


def _run_smoke_test(model_path: str, context: Any) -> dict:
    """Run a quick inference smoke test on a merged model.

    Returns dict with:
        passed: bool
        response: str (first 100 chars)
        tokens_per_second: float
        error: str | None
    """
    import logging

    from modelcypher.cli.composition import get_inference_engine

    logger = logging.getLogger(__name__)
    smoke_prompts = [
        "Hello, how are you today?",
        "What is 2 + 2?",
        "Complete this sentence: The quick brown fox",
    ]

    try:
        engine = get_inference_engine()
        # Run 3 prompts, check for coherent output
        results = []
        for prompt in smoke_prompts:
            result = engine.run(
                model=model_path,
                prompt=prompt,
            )
            response = result.text if hasattr(result, "text") else str(result)
            tps = result.tokens_per_second if hasattr(result, "tokens_per_second") else 0
            results.append(
                {
                    "prompt": prompt,
                    "response": response[:100],
                    "tokens_per_second": tps,
                }
            )

        # Check for obvious failures
        all_empty = all(len(r["response"].strip()) == 0 for r in results)
        all_garbage = all(len(set(r["response"])) < 5 or "�" in r["response"] for r in results)
        mean_tps = sum(r["tokens_per_second"] for r in results) / len(results)

        if all_empty:
            return {"passed": False, "error": "All responses empty", "results": results}
        if all_garbage:
            return {"passed": False, "error": "Responses appear garbled", "results": results}
        if mean_tps < 1.0:
            return {
                "passed": False,
                "error": f"Very slow inference: {mean_tps:.1f} tok/s",
                "results": results,
            }

        logger.info("SMOKE TEST: PASSED (%.1f tok/s)", mean_tps)
        return {
            "passed": True,
            "tokens_per_second": mean_tps,
            "results": results,
            "error": None,
        }

    except Exception as e:
        logger.warning("SMOKE TEST: FAILED - %s", str(e))
        return {"passed": False, "error": str(e), "results": []}


@app.command("delete")
def model_delete(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    """Delete a registered model.

    Examples:
        mc model delete my-llama
    """
    context = _context(ctx)
    service = get_model_service()
    service.delete_model(model_id)
    write_output({"deleted": model_id}, context.output_format, context.pretty)


@app.command("fetch")
def model_fetch(
    ctx: typer.Context,
    repo_id: str = typer.Argument(...),
    revision: str = typer.Option("main", "--revision"),
    auto_register: bool = typer.Option(False, "--auto-register"),
    alias: str | None = typer.Option(None, "--alias"),
    architecture: str | None = typer.Option(None, "--architecture"),
) -> None:
    """Fetch a model from HuggingFace Hub.

    Examples:
        mc model fetch mlx-community/Llama-2-7b-mlx
        mc model fetch mlx-community/Llama-2-7b-mlx --auto-register --alias my-llama
    """
    typer.echo(
        "DEPRECATED: use `mc model add <repo_id> [--alias <alias>]` instead.",
        err=True,
    )
    context = _context(ctx)
    service = get_model_service()
    result = service.fetch_model(repo_id, revision, auto_register, alias, architecture)
    try:
        probe_service = get_model_probe_service()
        probe_service.probe(result["localPath"])
    except Exception as exc:
        typer.echo(f"Warning: profile probe failed: {exc}", err=True)
    write_output(result, context.output_format, context.pretty)


@app.command("search")
def model_search(
    ctx: typer.Context,
    query: str | None = typer.Argument(None),
    author: str | None = typer.Option(None, "--author"),
    library: str = typer.Option("mlx", "--library"),
    quant: str | None = typer.Option(None, "--quant"),
    sort: str = typer.Option("downloads", "--sort"),
    limit: int = typer.Option(20, "--limit"),
    cursor: str | None = typer.Option(None, "--cursor"),
) -> None:
    """Search for models on HuggingFace Hub.

    Examples:
        mc model search llama
        mc model search llama --library mlx --quant 4bit
        mc model search --author mlx-community --sort downloads
    """
    context = _context(ctx)
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
            hint="Check your network connection. For private models, set HF_TOKEN environment variable.",
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
            hint="Check MLX runtime status (mc system status) and ensure MLX loads on this machine.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    _write_probe_output(result, context, include_profile=True, model_path=model_path)


@app.command("probe")
def model_probe(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Probe a model for architecture details.

    Examples:
        mc model probe ./models/llama-7b
    """
    typer.echo(
        "DEPRECATED: use `mc model info <path>` instead.",
        err=True,
    )
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
            hint="Check MLX runtime status (mc system status) and ensure MLX loads on this machine.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    _write_probe_output(result, context, include_profile=False, model_path=model_path)


@app.command("validate-merge")
def model_validate_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Path to source model"),
    target: str = typer.Option(..., "--target", help="Path to target model"),
) -> None:
    """Validate merge alignment between two models.

    Examples:
        mc model validate-merge --source ./model-a --target ./model-b
    """
    context = _context(ctx)
    service = get_model_probe_service()
    try:
        result = service.validate_merge(source, target)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Merge validation failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Merge validation failed",
            detail=str(exc),
            hint="Check MLX runtime status (mc system status) and ensure MLX loads on this machine.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "lowEffort": result.low_effort,
        "architectureMatch": result.architecture_match,
        "vocabMatch": result.vocab_match,
        "dimensionMatch": result.dimension_match,
        "warnings": result.warnings,
    }

    if context.output_format == "text":
        status = "LOW_EFFORT" if result.low_effort else "NEEDS_ALIGNMENT"
        lines = [
            "MERGE VALIDATION",
            f"Status: {status}",
            f"Architecture Match: {'Yes' if result.architecture_match else 'No'}",
            f"Vocab Match: {'Yes' if result.vocab_match else 'No'}",
            f"Dimension Match: {'Yes' if result.dimension_match else 'No'}",
        ]
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("validate-knowledge")
def model_validate_knowledge(
    ctx: typer.Context,
    merged: str = typer.Option(..., "--merged", help="Path to merged model"),
    source: str | None = typer.Option(None, "--source", help="Path to source model (for baseline)"),
    report_path: str | None = typer.Option(
        None, "--report-path", help="Path to save validation report"
    ),
) -> None:
    """Validate knowledge transfer in merged model.

    Tests whether the merged model retains knowledge from the source model
    across multiple domains using targeted probes.

    Examples:
        mc model validate-knowledge --merged ./merged-model
        mc model validate-knowledge --merged ./merged-model --source ./source-model
    """
    from modelcypher.cli.composition import get_registry
    from modelcypher.core.domain.merging.knowledge_transfer_validator import (
        KnowledgeDomain,
    )
    from modelcypher.core.use_cases.knowledge_transfer_service import (
        KnowledgeTransferService,
    )

    context = _context(ctx)

    typer.echo("Running knowledge transfer validation...", err=True)
    typer.echo(f"  Merged model: {merged}", err=True)
    if source:
        typer.echo(f"  Source model: {source}", err=True)
    typer.echo(f"  Domains: {', '.join(d.value for d in KnowledgeDomain)}", err=True)

    registry = get_registry()
    service = KnowledgeTransferService(inference_engine=registry.inference_engine)
    try:
        result = service.validate(
            merged_model=merged,
            source_model=source,
        )
    except Exception as e:
        error = ErrorDetail(
            code="MC-1031",
            title="Knowledge validation failed",
            detail=str(e),
            hint="Check model paths and ensure models are loaded correctly",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Display summary
    typer.echo("\nKnowledge Transfer Validation Complete!", err=True)
    typer.echo(f"  Overall Retention: {result.overall_retention:.1%}", err=True)
    typer.echo(f"  Probes Executed: {result.probes_executed}", err=True)
    typer.echo(f"  Time: {result.execution_time_seconds:.1f}s", err=True)

    typer.echo("\n  Per-Domain Retention:", err=True)
    for domain, domain_result in result.report.per_domain.items():
        # Show raw retention score - let user interpret based on their context
        typer.echo(
            f"    {domain.value}: {domain_result.retention_score:.1%} "
            f"({domain_result.probes_tested} probes)",
            err=True,
        )

    if result.warnings:
        typer.echo("\n  Warnings:", err=True)
        for warning in result.warnings:
            typer.echo(f"    - {warning}", err=True)

    # Save report if requested
    if report_path:
        from pathlib import Path

        Path(report_path).write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        typer.echo(f"  Report saved: {report_path}", err=True)

    write_output(result.to_dict(), context.output_format, context.pretty)


@app.command("analyze-alignment")
def model_analyze_alignment(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Analyze alignment drift between two models.

    Examples:
        mc model analyze-alignment --model-a ./base-model --model-b ./fine-tuned
    """
    context = _context(ctx)
    service = get_model_probe_service()
    try:
        result = service.analyze_alignment(model_a, model_b)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1003",
            title="Alignment analysis failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1003",
            title="Alignment analysis failed",
            detail=str(exc),
            hint="Check MLX runtime status (mc system status) and ensure MLX loads on this machine.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "driftMagnitude": result.drift_magnitude,
        "driftStd": result.drift_std,
        "driftMin": result.drift_min,
        "driftMax": result.drift_max,
        "driftP50": result.drift_p50,
        "driftP90": result.drift_p90,
        "commonLayerCount": result.common_layer_count,
        "comparableLayerCount": result.comparable_layer_count,
        "missingLayerCount": result.missing_layer_count,
        "layerDrifts": [
            {
                "layerName": drift.layer_name,
                "driftMagnitude": drift.drift_magnitude,
                "driftZScore": drift.drift_z_score,
                "comparable": drift.comparable,
            }
            for drift in result.layer_drifts[:20]  # Limit to first 20 layers
        ],
    }

    if context.output_format == "text":
        def _fmt(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"{value:.4f}"

        lines = [
            "ALIGNMENT ANALYSIS",
            f"Drift Magnitude: {_fmt(result.drift_magnitude)}",
            f"Drift Std: {_fmt(result.drift_std)}",
            f"Drift Min: {_fmt(result.drift_min)}",
            f"Drift Max: {_fmt(result.drift_max)}",
            f"Drift P50: {_fmt(result.drift_p50)}",
            f"Drift P90: {_fmt(result.drift_p90)}",
            f"Common Layers: {result.common_layer_count}",
            f"Comparable Layers: {result.comparable_layer_count}",
            f"Missing Layers: {result.missing_layer_count}",
        ]
        if result.layer_drifts:
            lines.append("")
            lines.append("Layer Drifts (top 10):")
            comparable_drifts = [d for d in result.layer_drifts if d.drift_magnitude is not None]
            for drift in sorted(comparable_drifts, key=lambda d: d.drift_magnitude, reverse=True)[:10]:
                lines.append(
                    f"  {drift.layer_name}: {drift.drift_magnitude:.4f} (z={_fmt(drift.drift_z_score)})"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("vocab-compare")
def model_vocab_compare(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Compare vocabularies between two models for cross-vocabulary merging.

    Analyzes tokenizer overlap and reports alignment statistics.

    Examples:
        mc model vocab-compare --model-a ./llama-3-8b --model-b ./qwen-2-7b
    """

    context = _context(ctx)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        error = ErrorDetail(
            code="MC-1020",
            title="Missing dependency",
            detail="transformers package required for vocabulary comparison",
            hint="Install with: pip install transformers",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.core.domain.vocabulary import (
            compare_tokenizers,
            format_comparison_report,
        )
    except ImportError as e:
        error = ErrorDetail(
            code="MC-1021",
            title="Vocabulary comparison not available",
            detail=str(e),
            hint="Ensure modelcypher is properly installed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load tokenizers
    typer.echo(f"Loading tokenizer from {model_a}...", err=True)
    try:
        tokenizer_a = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1022",
            title="Failed to load tokenizer",
            detail=f"Model A: {e}",
            hint="Ensure the path contains a valid tokenizer",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    typer.echo(f"Loading tokenizer from {model_b}...", err=True)
    try:
        tokenizer_b = AutoTokenizer.from_pretrained(model_b, trust_remote_code=True)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1022",
            title="Failed to load tokenizer",
            detail=f"Model B: {e}",
            hint="Ensure the path contains a valid tokenizer",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Perform comparison
    typer.echo("Analyzing vocabulary overlap...", err=True)
    result = compare_tokenizers(tokenizer_a, tokenizer_b)

    # Build payload
    payload = result.to_dict()
    payload["modelA"] = model_a
    payload["modelB"] = model_b
    payload["needsBridge"] = result.overlap_ratio < 1.0

    if context.output_format == "text":
        report = format_comparison_report(result)
        typer.echo("")
        typer.echo(report)
        return

    write_output(payload, context.output_format, context.pretty)


# Helper functions for model search


def _parse_model_search_library(value: str) -> ModelSearchLibraryFilter:
    normalized = value.lower()
    for member in ModelSearchLibraryFilter:
        if normalized == member.value.lower():
            return member
    valid = ", ".join(m.value for m in ModelSearchLibraryFilter)
    raise typer.BadParameter(f"Invalid library filter '{value}'. Valid options: {valid}")


@app.command("extract-anchors")
def model_extract_anchors(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Path to save anchors (JSON)"
    ),
) -> None:
    """Extract semantic anchors from model token embeddings.

    Uses Fréchet mean (geometric center on curved manifolds) to compute
    semantic anchor points from the UnifiedAtlas probes (~450 concepts).

    These anchors can be used for:
    - Model alignment comparison
    - Semantic drift detection
    - Cross-model concept mapping

    Examples:
        mc model extract-anchors ./models/llama-7b
        mc model extract-anchors ./models/qwen-7b --output anchors.json
    """
    import json as json_module
    from pathlib import Path

    from modelcypher.cli.composition import get_registry
    from modelcypher.core.use_cases.anchor_extractor import (
        AnchorExtractor,
        AnchorExtractorError,
    )

    context = _context(ctx)

    typer.echo(f"Extracting semantic anchors from: {model_path}", err=True)

    try:
        # Load model weights
        registry = get_registry()
        weights, _ = registry.model_loader.load_weights(model_path)

        # Extract anchors
        extractor = AnchorExtractor()
        anchors, confidences = extractor.extract(model_path, weights)

        typer.echo(f"Extracted {len(anchors)} anchors", err=True)

    except AnchorExtractorError as exc:
        error = ErrorDetail(
            code="MC-1040",
            title="Anchor extraction failed",
            detail=str(exc),
            hint="Ensure model has tokenizer.json and embed_tokens weights",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-1040",
            title="Anchor extraction failed",
            detail=str(exc),
            hint="Check model path and ensure weights are accessible",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Convert anchors to serializable format
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    anchor_data = {
        anchor_id: backend.tolist(anchor_vec)
        for anchor_id, anchor_vec in anchors.items()
    }

    payload = {
        "modelPath": model_path,
        "anchorCount": len(anchors),
        "anchors": anchor_data,
        "confidences": confidences,
    }

    # Save to file if requested
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json_module.dump(payload, f, indent=2)
        typer.echo(f"Saved anchors to: {output_path}", err=True)

    if context.output_format == "text":
        lines = [
            "ANCHOR EXTRACTION",
            f"Model: {model_path}",
            f"Anchors: {len(anchors)}",
            "",
            "Top anchors by confidence:",
        ]
        # Show top 10 by confidence
        sorted_anchors = sorted(
            confidences.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for anchor_id, conf in sorted_anchors:
            lines.append(f"  {anchor_id}: {conf:.3f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


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
        # Match against both the enum value and a normalized version
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
