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
- Model listing, addition, deletion
- Model search via HuggingFace Hub
- Model inspection for architecture details
- Model merge validation
- Alignment analysis between models

Commands:
    mc model list
    mc model add <repo_id|path> [--alias]
    mc model search <query>
    mc model info <path>
"""

from __future__ import annotations

import json
import math
import statistics
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

# Late imports to keep prevent_sleep() context manager lightweight
from modelcypher.cli.composition import (  # noqa: E402
    get_backend,
    get_model_probe_service,
    get_model_search_service,
    get_model_service,
)
from modelcypher.cli.context import CLIContext  # noqa: E402
from modelcypher.cli.output import write_error, write_output  # noqa: E402
from modelcypher.cli.presenters import model_payload, model_search_payload  # noqa: E402
from modelcypher.cli.warnings import warn_network  # noqa: E402
from modelcypher.core.domain.model_search import (  # noqa: E402
    MemoryFitStatus,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.utils.errors import ErrorDetail  # noqa: E402
from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code  # noqa: E402

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
    trajectory_metrics: dict | None = None,
) -> None:
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
        "layerCountTensors": len(result.layers),
        "layerCountConfig": _coerce_value(getattr(result, "layer_count_config", None)),
        "layers": [_layer_payload(layer) for layer in result.layers[:20]],
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

    # Add trajectory metrics to payload if provided
    if trajectory_metrics is not None:
        payload["trajectory"] = {
            "pathLengthRatio": trajectory_metrics.get("path_length_ratio"),
            "meanCurvature": trajectory_metrics.get("mean_curvature"),
            "effectiveRank": trajectory_metrics.get("effective_rank"),
            "spectralEntropy": trajectory_metrics.get("spectral_entropy"),
            "probes": trajectory_metrics.get("probes"),
            "tokens": trajectory_metrics.get("tokens"),
        }

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

        # Add trajectory analysis section if computed
        if trajectory_metrics is not None:
            import math

            lines.append("")
            lines.append("TRAJECTORY ANALYSIS")
            plr = trajectory_metrics.get("path_length_ratio")
            curv = trajectory_metrics.get("mean_curvature")
            eff_rank = trajectory_metrics.get("effective_rank")
            spec_ent = trajectory_metrics.get("spectral_entropy")
            probes = trajectory_metrics.get("probes", 0)
            tokens = trajectory_metrics.get("tokens", 0)

            if plr is not None and not math.isnan(plr):
                lines.append(f"  Path Length Ratio: {plr:.4f}")
            else:
                lines.append("  Path Length Ratio: n/a")

            if curv is not None and not math.isnan(curv):
                lines.append(f"  Mean Curvature: {curv:.4f} rad")
            else:
                lines.append("  Mean Curvature: n/a")

            if eff_rank is not None and not math.isnan(eff_rank):
                lines.append(f"  Effective Rank: {eff_rank:.2f}")
            else:
                lines.append("  Effective Rank: n/a")

            if spec_ent is not None and not math.isnan(spec_ent):
                lines.append(f"  Spectral Entropy: {spec_ent:.4f}")
            else:
                lines.append("  Spectral Entropy: n/a")

            lines.append(f"  Probes: {probes}, Tokens: {tokens}")

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

    warn_network(context, "Fetching model artifacts from Hugging Face Hub.")
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




def _compute_trajectory_metrics(model_path: str) -> dict:
    """Compute trajectory complexity metrics for a model.

    Loads the model, runs diagnostic prompts, and computes geometric
    trajectory metrics.

    Returns dict with:
        path_length_ratio: float
        mean_curvature: float
        effective_rank: float
        spectral_entropy: float
        probes: int (number of prompts used)
        tokens: int (total tokens processed)
    """
    from mlx_lm import load

    from modelcypher.cli.composition import get_registry
    from modelcypher.core.domain.geometry.trajectory_complexity import TrajectoryComplexity

    # Diverse prompts for trajectory sampling
    diagnostic_prompts = [
        "What is the capital of France?",
        "Explain step by step how to solve: 2 + 3 * 4",
        "If all cats are mammals and all mammals have hearts, do all cats have hearts?",
        "Write a short poem about the moon.",
    ]

    # Load model and tokenizer
    model, tokenizer = load(model_path)

    # Get activation provider
    registry = get_registry()
    provider = registry.activation_provider

    # Collect trajectory activations
    trajectory_result = provider.collect_trajectory_batch(
        model=model,
        tokenizer=tokenizer,
        texts=diagnostic_prompts,
    )

    # Compute trajectory complexity from hidden positions
    tc = TrajectoryComplexity(backend=registry.backend)
    complexity = tc.compute(trajectory_result.positions)

    return {
        "path_length_ratio": complexity.path_length_ratio,
        "mean_curvature": complexity.mean_curvature,
        "effective_rank": complexity.trajectory_effective_rank,
        "spectral_entropy": complexity.trajectory_spectral_entropy,
        "probes": len(diagnostic_prompts),
        "tokens": trajectory_result.total_tokens,
    }


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
    trajectory: bool = typer.Option(
        False,
        "--trajectory",
        "-t",
        help="Run diagnostic prompts and compute trajectory metrics",
    ),
) -> None:
    """Inspect a model and surface its stored identity profile.

    Examples:
        mc model info ./models/llama-7b
        mc model info ./models/llama-7b --trajectory
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

    # Compute trajectory metrics if requested
    trajectory_metrics: dict | None = None
    if trajectory:
        try:
            trajectory_metrics = _compute_trajectory_metrics(model_path)
        except Exception as exc:
            typer.echo(f"Warning: trajectory analysis failed: {exc}", err=True)

    _write_probe_output(
        result,
        context,
        include_profile=True,
        model_path=model_path,
        trajectory_metrics=trajectory_metrics,
    )




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
        warn_trust_remote_code()
        tokenizer_a = AutoTokenizer.from_pretrained(
            model_a, trust_remote_code=trust_remote_code_enabled()
        )
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
        warn_trust_remote_code()
        tokenizer_b = AutoTokenizer.from_pretrained(
            model_b, trust_remote_code=trust_remote_code_enabled()
        )
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
    backend = get_backend()
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


def _summarize_deltas(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "finite": 0, "infinite": 0}
    finite = [value for value in values if math.isfinite(value)]
    infinite = len(values) - len(finite)
    if not finite:
        return {"count": len(values), "finite": 0, "infinite": infinite}

    abs_vals = [abs(value) for value in finite]
    return {
        "count": len(values),
        "finite": len(finite),
        "infinite": infinite,
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "meanAbs": float(statistics.fmean(abs_vals)),
        "medianAbs": float(statistics.median(abs_vals)),
        "minAbs": float(min(abs_vals)),
        "maxAbs": float(max(abs_vals)),
    }


def _compare_geometric_profiles(base, candidate) -> dict[str, object]:
    base_layers = set(base.layer_profiles)
    candidate_layers = set(candidate.layer_profiles)
    common_layers = sorted(base_layers & candidate_layers)

    def _layer_deltas(field: str) -> list[float]:
        deltas: list[float] = []
        for idx in common_layers:
            base_val = getattr(base.layer_profiles[idx], field, None)
            cand_val = getattr(candidate.layer_profiles[idx], field, None)
            if base_val is None or cand_val is None:
                continue
            deltas.append(float(cand_val) - float(base_val))
        return deltas

    metrics = {
        "activationRank": _summarize_deltas(_layer_deltas("activation_rank")),
        "trajectoryRank": _summarize_deltas(_layer_deltas("trajectory_rank")),
        "signalRank": _summarize_deltas(_layer_deltas("signal_rank")),
        "nullRank": _summarize_deltas(_layer_deltas("null_rank")),
        "gramCondition": _summarize_deltas(_layer_deltas("gram_condition")),
        "trajectorySamples": _summarize_deltas(_layer_deltas("trajectory_samples")),
        "positionSamples": _summarize_deltas(_layer_deltas("position_samples")),
        "velocitySamples": _summarize_deltas(_layer_deltas("velocity_samples")),
    }

    base_saturated = sum(1 for lp in base.layer_profiles.values() if lp.saturated)
    cand_saturated = sum(1 for lp in candidate.layer_profiles.values() if lp.saturated)

    return {
        "layersCompared": len(common_layers),
        "metrics": metrics,
        "embedding": {
            "rankDelta": float(candidate.embedding_rank - base.embedding_rank),
            "gramConditionDelta": float(
                candidate.embedding_gram_condition - base.embedding_gram_condition
            ),
            "nProbesDelta": float(candidate.embedding_n_probes - base.embedding_n_probes),
        },
        "saturation": {
            "base": base_saturated,
            "candidate": cand_saturated,
            "delta": cand_saturated - base_saturated,
        },
    }


@app.command("profile")
def model_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-profile even if valid profile exists"
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Show existing profile instead of computing"
    ),
    max_batches: int | None = typer.Option(
        None, "--max-batches", help="Maximum batches for testing (None = run to saturation)"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Show per-layer intrinsic dimension trajectory"
    ),
    fingerprint: bool = typer.Option(
        False, "--fingerprint", help="Show geometric fingerprint (expansion_ratio by task)"
    ),
) -> None:
    """Compute geometric profile via trajectory-based manifold mapping.

    Uses domain-stratified sampling and rank saturation detection to
    fully map the model's activation manifold. This is 20x more efficient
    than per-probe profiling:
    - A 100-token text yields 199 samples (100 positions + 99 velocities)
    - Domain-stratified sampling ensures coverage of all 15 atlas domains
    - Rank saturation detection provides geometric termination

    Profile is stored in:
        {model_path}/.modelcypher/profile.json      (metadata)
        {model_path}/.modelcypher/activations.safetensors (activations)

    The --trajectory flag computes per-layer intrinsic dimension using TwoNN,
    showing the semantic highway pattern (compression → processing → recovery).

    The --fingerprint flag runs diverse task probes and shows expansion_ratio
    variance - a signature that distinguishes base models from specialists.

    Examples:
        mc model profile ./models/llama-7b              # Profile a model
        mc model profile ./models/llama-7b --force      # Force re-profile
        mc model profile ./models/llama-7b --show       # Show existing profile
        mc model profile ./models/llama-7b --trajectory # Show ID trajectory
        mc model profile ./models/llama-7b --fingerprint  # Show expansion_ratio
    """
    from modelcypher.cli.composition import get_registry
    from modelcypher.core.domain.profile import GeometricProfileStore
    from modelcypher.core.use_cases.profile_service import ProfileService

    context = _context(ctx)

    # Validate model path
    model_path_obj = Path(model_path).expanduser().resolve()
    if not model_path_obj.exists():
        error = ErrorDetail(
            code="MC-1050",
            title="Model not found",
            detail=f"Model path does not exist: {model_path}",
            hint="Ensure the path points to a valid model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    store = GeometricProfileStore()

    # Show existing profile
    if show:
        profile = store.load(model_path)
        if profile is None:
            error = ErrorDetail(
                code="MC-1051",
                title="Profile not found",
                detail=f"No profile found for model: {model_path}",
                hint="Run 'mc model profile <path>' to create a profile",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        payload = profile.to_dict()
        if context.output_format == "text":
            lines = [
                "GEOMETRIC PROFILE (TRAJECTORY-BASED)",
                f"Model: {profile.model_path}",
                f"Version: {profile.profile_version}",
                f"Created: {profile.created_at}",
                f"Probes: {profile.probe_count}",
                "",
                "Dimensions:",
                f"  Hidden: {profile.hidden_dim}",
                f"  Intermediate: {profile.intermediate_dim}",
                f"  Layers: {profile.num_layers}",
                f"  Vocab: {profile.vocab_size}",
                "",
                f"Layers Profiled: {len(profile.layer_profiles)}",
                f"Has Activations: {profile.has_activations}",
            ]

            # Add convergence metrics
            if profile.convergence:
                conv = profile.convergence
                lines.append("")
                lines.append("Convergence:")
                lines.append(f"  Batches: {conv.total_batches}")
                lines.append(f"  All Saturated: {conv.all_layers_saturated}")
                if conv.domains_covered:
                    lines.append(f"  Domains: {len(conv.domains_covered)} ({', '.join(sorted(conv.domains_covered)[:5])}...)")

            if profile.layer_profiles:
                lines.append("")
                lines.append("Layer Geometry (sample):")
                for idx in sorted(profile.layer_profiles.keys())[:5]:
                    lp = profile.layer_profiles[idx]
                    sat_str = "SAT" if lp.saturated else "---"
                    samples_str = f"{lp.trajectory_samples:,}" if lp.trajectory_samples else str(lp.n_probes)
                    lines.append(
                        f"  Layer {idx}: rank={lp.activation_rank}/{lp.hidden_dim}, "
                        f"null={lp.null_rank}, samples={samples_str}, [{sat_str}]"
                    )
                if len(profile.layer_profiles) > 5:
                    lines.append(f"  ... ({len(profile.layer_profiles) - 5} more layers)")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
        return

    # Compute profile (when show=False)
    typer.echo(f"Computing trajectory-based manifold map for: {model_path_obj}", err=True)
    if max_batches:
        typer.echo(f"  Max batches: {max_batches} (testing mode)", err=True)

    registry = get_registry()
    service = ProfileService(
        backend=registry.backend,
        model_loader=registry.model_loader,
        activation_provider=registry.activation_provider,
        store=store,
    )

    with prevent_sleep():
        try:
            result = service.compute_profile(model_path, force=force, max_batches=max_batches)
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1052",
                title="Profile computation failed",
                detail=str(exc),
                hint="Check model path and ensure the model is loadable",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    profile = result.profile

    # Build payload with trajectory-specific metrics
    payload = {
        "status": "cached" if result.from_cache else "computed",
        "modelPath": profile.model_path,
        "profileDir": str(result.profile_dir),
        "profileVersion": profile.profile_version,
        "probesProcessed": result.probes_processed,
        "probesFailed": result.probes_failed,
        "layersProfiled": result.layers_profiled,
        "hasActivations": profile.has_activations,
        "dimensions": {
            "hidden": profile.hidden_dim,
            "intermediate": profile.intermediate_dim,
            "layers": profile.num_layers,
            "vocab": profile.vocab_size,
        },
        "convergence": {
            "totalBatches": profile.convergence.total_batches,
            "allLayersSaturated": profile.convergence.all_layers_saturated,
            "domainsCovered": list(profile.convergence.domains_covered),
        },
    }

    # Compute trajectory (per-layer intrinsic dimension) if requested
    trajectory_data: dict[str, Any] = {}
    if trajectory:
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
        from modelcypher.core.domain.profile import load_activations

        backend = get_backend()
        id_estimator = IntrinsicDimension(backend)

        # Load cached activations from profile directory
        try:
            activations = load_activations(result.profile_dir, backend)
        except FileNotFoundError:
            activations = None
        if activations and activations.hidden:
            layer_ids: list[dict[str, Any]] = []
            for layer_idx in sorted(activations.hidden.keys()):
                act = activations.hidden[layer_idx]
                try:
                    estimate = id_estimator.compute(act)
                    layer_ids.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": estimate.intrinsic_dimension,
                        "sample_count": estimate.sample_count,
                    })
                except Exception:
                    layer_ids.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": float("nan"),
                        "sample_count": 0,
                    })

            valid_ids = [r["intrinsic_dimension"] for r in layer_ids
                         if r["intrinsic_dimension"] == r["intrinsic_dimension"]]
            if valid_ids:
                min_id = min(valid_ids)
                max_id = max(valid_ids)
                final_id = valid_ids[-1] if valid_ids else 0
                recovery_ratio = final_id / min_id if min_id > 0 else 0

                trajectory_data = {
                    "layerIntrinsicDims": layer_ids,
                    "minIntrinsicDim": min_id,
                    "maxIntrinsicDim": max_id,
                    "finalIntrinsicDim": final_id,
                    "recoveryRatio": recovery_ratio,
                }
                payload["trajectory"] = trajectory_data

    # Compute fingerprint (expansion_ratio by task) if requested
    fingerprint_data: dict[str, Any] = {}
    if fingerprint:
        from mlx_lm import load

        loaded_model, tokenizer = load(str(model_path_obj))
        task_results = {}
        for task_type, prompt in _FINGERPRINT_PROBES.items():
            norms = _trace_norm_trajectory(loaded_model, tokenizer, prompt)
            expansion_ratio = _compute_expansion_ratio(norms)
            task_results[task_type] = expansion_ratio

        ratio_values = list(task_results.values())
        ratio_mean = statistics.mean(ratio_values)
        ratio_variance = statistics.variance(ratio_values) if len(ratio_values) > 1 else 0.0

        fingerprint_data = {
            "expansionRatioMean": ratio_mean,
            "expansionRatioVariance": ratio_variance,
            "expansionRatioRange": [min(ratio_values), max(ratio_values)],
            "taskBreakdown": task_results,
        }
        payload["fingerprint"] = fingerprint_data

    if context.output_format == "text":
        status = "CACHED" if result.from_cache else "COMPUTED"
        sat_status = "ALL SATURATED" if profile.convergence.all_layers_saturated else "PARTIAL"
        typer.echo(f"\nProfile {status} for: {profile.model_path}", err=True)
        typer.echo(f"  Probes: {result.probes_processed}, Batches: {profile.convergence.total_batches}", err=True)
        typer.echo(f"  Layers: {result.layers_profiled}, Status: {sat_status}", err=True)
        typer.echo(f"  Profile saved: {result.profile_dir}", err=True)

        # Show trajectory if computed
        if trajectory_data:
            typer.echo("\nTrajectory (Per-Layer Intrinsic Dimension):", err=True)
            layer_ids_list = trajectory_data.get("layerIntrinsicDims", [])
            max_id_val = trajectory_data.get("maxIntrinsicDim", 1)
            for r in layer_ids_list:
                id_val = r["intrinsic_dimension"]
                if id_val == id_val:  # not NaN
                    bar_len = int(40 * id_val / max_id_val) if max_id_val > 0 else 0
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    typer.echo(f"  Layer {r['layer']:3d}: {id_val:5.1f}D |{bar}|", err=True)
            typer.echo(f"\n  Min ID: {trajectory_data.get('minIntrinsicDim', 0):.1f}", err=True)
            typer.echo(f"  Max ID: {trajectory_data.get('maxIntrinsicDim', 0):.1f}", err=True)
            typer.echo(f"  Recovery Ratio: {trajectory_data.get('recoveryRatio', 0):.2f}×", err=True)

        # Show fingerprint if computed
        if fingerprint_data:
            typer.echo("\nFingerprint (Expansion Ratio by Task):", err=True)
            for task, ratio in fingerprint_data.get("taskBreakdown", {}).items():
                typer.echo(f"  {task}: {ratio:.4f}", err=True)
            typer.echo(f"\n  Mean: {fingerprint_data.get('expansionRatioMean', 0):.4f}", err=True)
            typer.echo(f"  Variance: {fingerprint_data.get('expansionRatioVariance', 0):.6f}", err=True)
            range_vals = fingerprint_data.get("expansionRatioRange", [0, 0])
            typer.echo(f"  Range: [{range_vals[0]:.4f}, {range_vals[1]:.4f}]", err=True)

    write_output(payload, context.output_format, context.pretty)
    return


@app.command("quantize-sweep")
def model_quantize_sweep(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for quantized models (default: <model_path>/quantized)",
    ),
    bits: list[int] = typer.Option(
        None,
        "--bits",
        "-b",
        help="Bit widths to attempt (repeatable). Defaults to 8, 6, 4, 2, 1.",
    ),
    group_size: int | None = typer.Option(
        None,
        "--group-size",
        "-g",
        help="Quantization group size (required unless config.json provides one)",
    ),
    mode: str = typer.Option(
        "affine",
        "--mode",
        help="Quantization mode passed to MLX (affine, mxfp4, etc.)",
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Profile each quantized model after quantization",
    ),
    profile_base: bool = typer.Option(
        True,
        "--profile-base/--no-profile-base",
        help="Profile the full-precision model before the sweep",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing quantized weights if they already exist",
    ),
    force_profile: bool = typer.Option(
        False,
        "--force-profile",
        help="Recompute profiles even if cached profiles already exist",
    ),
    max_batches: int | None = typer.Option(
        None,
        "--max-batches",
        help="Maximum batches for profiling (None = run to saturation)",
    ),
) -> None:
    """Quantize a model across multiple bit widths and profile each variant."""
    from modelcypher.cli.composition import get_registry
    from modelcypher.core.domain.profile import GeometricProfileStore
    from modelcypher.core.use_cases.profile_service import ProfileService
    from modelcypher.core.use_cases.quantization_service import QuantizationService

    context = _context(ctx)
    model_path_obj = Path(model_path).expanduser().resolve()
    if not model_path_obj.exists():
        error = ErrorDetail(
            code="MC-1060",
            title="Model not found",
            detail=f"Model path does not exist: {model_path}",
            hint="Ensure the path points to a valid model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    output_root = Path(output_dir).expanduser().resolve() if output_dir else model_path_obj / "quantized"
    output_root.mkdir(parents=True, exist_ok=True)

    requested_bits = bits if bits else [8, 6, 4, 2, 1]
    seen_bits: set[int] = set()
    bits_list = [b for b in requested_bits if not (b in seen_bits or seen_bits.add(b))]

    if group_size is None:
        config_path = model_path_obj / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            quant_cfg = config.get("quantization_config") or config.get("quantization") or {}
            group_size = quant_cfg.get("group_size")

    if group_size is None:
        error = ErrorDetail(
            code="MC-1061",
            title="Missing group size",
            detail="Quantization requires --group-size or a quantization_config in config.json.",
            hint="Provide --group-size explicitly for full-precision models.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    registry = get_registry()
    quant_service = QuantizationService(registry.backend, registry.model_loader)
    profile_store = GeometricProfileStore()
    profile_service = ProfileService(
        backend=registry.backend,
        model_loader=registry.model_loader,
        activation_provider=registry.activation_provider,
        store=profile_store,
    )

    supported_bits = quant_service.detect_supported_bits(bits_list, group_size, mode)
    available_bits = [bit for bit in bits_list if supported_bits.get(bit) is None]

    base_profile = None
    base_profile_dir = None
    base_profile_summary = None

    def _profile_summary(result) -> dict[str, object]:
        profile = result.profile
        return {
            "profileDir": str(result.profile_dir),
            "layersProfiled": result.layers_profiled,
            "probesProcessed": result.probes_processed,
            "probesFailed": result.probes_failed,
            "fromCache": result.from_cache,
            "hasActivations": profile.has_activations,
            "convergence": {
                "totalBatches": profile.convergence.total_batches,
                "allLayersSaturated": profile.convergence.all_layers_saturated,
                "domainsCovered": list(profile.convergence.domains_covered),
            },
        }

    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    with prevent_sleep():
        if profile and profile_base:
            typer.echo(f"Profiling base model: {model_path_obj}", err=True)
            try:
                base_profile_result = profile_service.compute_profile(
                    model_path=str(model_path_obj),
                    force=force_profile,
                    max_batches=max_batches,
                )
                base_profile = base_profile_result.profile
                base_profile_dir = base_profile_result.profile_dir
                base_profile_summary = _profile_summary(base_profile_result)
            except Exception as exc:
                errors.append(
                    {
                        "stage": "profile_base",
                        "error": str(exc),
                    }
                )

        for bit in available_bits:
            run_label = f"{bit}bit"
            quant_dir = output_root / f"{model_path_obj.name}-{run_label}"
            typer.echo(f"Quantizing ({run_label}) -> {quant_dir}", err=True)
            try:
                quant_result = quant_service.quantize_model(
                    model_path=str(model_path_obj),
                    output_dir=str(quant_dir),
                    bits=bit,
                    group_size=group_size,
                    mode=mode,
                    overwrite=overwrite,
                )
            except Exception as exc:
                errors.append(
                    {
                        "stage": "quantize",
                        "bits": bit,
                        "error": str(exc),
                    }
                )
                continue

            entry: dict[str, object] = {
                "bits": bit,
                "quantization": quant_result.to_dict(),
            }

            if profile:
                typer.echo(f"Profiling quantized model ({run_label})", err=True)
                try:
                    quant_profile_result = profile_service.compute_profile(
                        model_path=str(quant_dir),
                        force=force_profile,
                        max_batches=max_batches,
                    )
                    entry["profile"] = _profile_summary(quant_profile_result)
                    if base_profile is not None:
                        entry["delta"] = _compare_geometric_profiles(
                            base_profile, quant_profile_result.profile
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "profile_quantized",
                            "bits": bit,
                            "error": str(exc),
                        }
                    )
            registry.backend.clear_cache()
            results.append(entry)

    payload = {
        "modelPath": str(model_path_obj),
        "outputRoot": str(output_root),
        "groupSize": group_size,
        "mode": mode,
        "bitsRequested": bits_list,
        "bitsSupported": [bit for bit, err in supported_bits.items() if err is None],
        "bitsUnsupported": {str(bit): err for bit, err in supported_bits.items() if err},
        "baseProfile": base_profile_summary,
        "baseProfileDir": str(base_profile_dir) if base_profile_dir else None,
        "runs": results,
        "errors": errors,
    }

    write_output(payload, context.output_format, context.pretty)

    # Compute profile
    typer.echo(f"Computing trajectory-based manifold map for: {model_path}", err=True)
    if max_batches:
        typer.echo(f"  Max batches: {max_batches} (testing mode)", err=True)

    registry = get_registry()
    service = ProfileService(
        backend=registry.backend,
        model_loader=registry.model_loader,
        activation_provider=registry.activation_provider,
        store=store,
    )

    with prevent_sleep():
        try:
            result = service.compute_profile(model_path, force=force, max_batches=max_batches)
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1052",
                title="Profile computation failed",
                detail=str(exc),
                hint="Check model path and ensure the model is loadable",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    profile = result.profile

    # Build payload with trajectory-specific metrics
    payload = {
        "status": "cached" if result.from_cache else "computed",
        "modelPath": profile.model_path,
        "profileDir": str(result.profile_dir),
        "profileVersion": profile.profile_version,
        "probesProcessed": result.probes_processed,
        "probesFailed": result.probes_failed,
        "layersProfiled": result.layers_profiled,
        "hasActivations": profile.has_activations,
        "dimensions": {
            "hidden": profile.hidden_dim,
            "intermediate": profile.intermediate_dim,
            "layers": profile.num_layers,
            "vocab": profile.vocab_size,
        },
        "convergence": {
            "totalBatches": profile.convergence.total_batches,
            "allLayersSaturated": profile.convergence.all_layers_saturated,
            "domainsCovered": list(profile.convergence.domains_covered),
        },
    }

    if context.output_format == "text":
        status = "CACHED" if result.from_cache else "COMPUTED"
        sat_status = "ALL SATURATED" if profile.convergence.all_layers_saturated else "PARTIAL"
        lines = [
            f"PROFILE {status} ({sat_status})",
            f"Model: {profile.model_path}",
            f"Profile Dir: {result.profile_dir}",
            f"Probes: {result.probes_processed}",
            f"Batches: {profile.convergence.total_batches}",
            f"Domains: {len(profile.convergence.domains_covered)}",
            f"Layers: {result.layers_profiled}",
            f"Activations Saved: {profile.has_activations}",
        ]

        # Show layer summary
        if profile.layer_profiles:
            total_samples = sum(lp.trajectory_samples for lp in profile.layer_profiles.values())
            sat_count = sum(1 for lp in profile.layer_profiles.values() if lp.saturated)
            lines.append("")
            lines.append(f"Total Samples: {total_samples:,}")
            lines.append(f"Saturated Layers: {sat_count}/{len(profile.layer_profiles)}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# --- Geometric Fingerprinting Commands ---

_FINGERPRINT_PROBES = {
    "retrieval": "What is the capital of France?",
    "arithmetic": "What is 7 + 5?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "logic": "If all cats are animals, and all animals need water, do cats need water?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
    "cot": "Let me think step by step about how to solve this problem: What is 15% of 80?",
}


def _trace_norm_trajectory(model, tokenizer, prompt: str) -> list[float]:
    """Trace L2 norm through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    base = getattr(model, "model", model)

    # Embedding
    hidden = base.embed_tokens(input_ids)
    mx.eval(hidden)

    norms = [float(mx.sqrt(mx.sum(hidden * hidden)).item())]

    # Each layer
    for layer in base.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norms.append(float(mx.sqrt(mx.sum(hidden * hidden)).item()))

    return norms


def _compute_expansion_ratio(norms: list[float]) -> float:
    """Compute expansion ratio from norm trajectory (peak/final)."""
    if len(norms) < 2:
        return 1.0
    peak = max(norms)
    final = norms[-1]
    if final < sys.float_info.epsilon:
        return float("inf")
    return peak / final


@app.command("fingerprint")
def model_fingerprint(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Compute geometric fingerprint metrics from norm trajectories.

    Runs diverse task probes and measures expansion ratio variance.

    Examples:
        mc model fingerprint /path/to/model
    """
    from mlx_lm import load

    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-4003",
            title="Model not found",
            detail=f"Model path not found: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    loaded_model, tokenizer = load(str(model_path))

    # Compute expansion ratio for each task type
    task_results = {}
    for task_type, prompt in _FINGERPRINT_PROBES.items():
        norms = _trace_norm_trajectory(loaded_model, tokenizer, prompt)
        expansion_ratio = _compute_expansion_ratio(norms)
        task_results[task_type] = {
            "expansion_ratio": expansion_ratio,
            "peak_norm": max(norms),
            "final_norm": norms[-1],
        }

    # Compute variance
    ratio_values = [r["expansion_ratio"] for r in task_results.values()]
    ratio_mean = statistics.mean(ratio_values)
    ratio_variance = statistics.variance(ratio_values) if len(ratio_values) > 1 else 0.0
    ratio_std = statistics.stdev(ratio_values) if len(ratio_values) > 1 else 0.0

    result = {
        "model": str(model_path),
        "metrics": {
            "expansion_ratio_mean": ratio_mean,
            "expansion_ratio_variance": ratio_variance,
            "expansion_ratio_std": ratio_std,
            "expansion_ratio_min": min(ratio_values),
            "expansion_ratio_max": max(ratio_values),
        },
        "task_breakdown": task_results,
    }

    if context.output_format == "text":
        lines = [
            "MODEL FINGERPRINT",
            f"Path: {model_path}",
            f"",
            f"Expansion Ratio Statistics:",
            f"  Mean: {ratio_mean:.4f}",
            f"  Variance: {ratio_variance:.6f}",
            f"  Std: {ratio_std:.4f}",
            f"  Range: [{min(ratio_values):.4f}, {max(ratio_values):.4f}]",
            "",
            "Per-Task Metrics:",
        ]
        for task, data in task_results.items():
            lines.append(
                f"  {task}: expansion_ratio={data['expansion_ratio']:.4f}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(result, context.output_format, context.pretty)


@app.command("weight-analysis")
def model_weight_analysis(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model directory"),
    layers: str = typer.Option(
        "final",
        "--layers",
        "-l",
        help="Which layers to analyze: 'final', 'all', or comma-separated indices",
    ),
) -> None:
    """Analyze weight matrix properties.

    Examines effective rank, sparsity, and singular value distribution
    of weight matrices. Specialist models show higher sparsity and lower
    effective rank in final layers.

    Examples:
        mc model weight-analysis /path/to/model
        mc model weight-analysis /path/to/model --layers all
        mc model weight-analysis /path/to/model --layers 20,21,22
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-4003",
            title="Model not found",
            detail=f"Model path not found: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    loaded_model, _ = load(str(model_path))
    base = getattr(loaded_model, "model", loaded_model)
    n_layers = len(base.layers)

    # Determine which layers to analyze
    if layers == "final":
        layer_indices = [n_layers - 1]
    elif layers == "all":
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(x.strip()) for x in layers.split(",")]

    def analyze_weight_matrix(w: mx.array, name: str) -> dict:
        """Analyze a single weight matrix."""
        # Convert to float32 for numpy compatibility (handles bfloat16)
        w_float = w.astype(mx.float32)
        mx.eval(w_float)
        w_np = np.array(w_float)

        # Sparsity (fraction of near-zero values)
        threshold = 1e-6
        sparsity = float(np.mean(np.abs(w_np) < threshold))

        # SVD for effective rank
        try:
            s = np.linalg.svd(w_np, compute_uv=False)
            s_normalized = s / (s.sum() + 1e-10)
            # Effective rank via participation ratio
            eff_rank = float(1.0 / (np.sum(s_normalized ** 2) + 1e-10))
            # Condition number
            condition = float(s[0] / (s[-1] + 1e-10)) if len(s) > 0 else float("inf")
            # Top singular value dominance
            top_sv_ratio = float(s[0] / (s.sum() + 1e-10)) if len(s) > 0 else 0.0
        except Exception:
            eff_rank = 0.0
            condition = float("inf")
            top_sv_ratio = 0.0

        return {
            "name": name,
            "shape": list(w_np.shape),
            "sparsity": sparsity,
            "effective_rank": eff_rank,
            "condition_number": condition,
            "top_sv_ratio": top_sv_ratio,
            "frobenius_norm": float(np.linalg.norm(w_np, "fro")),
        }

    layer_results = {}
    for idx in layer_indices:
        if idx >= n_layers:
            continue
        layer = base.layers[idx]
        layer_data = {}

        # Analyze key weight matrices in the layer
        # Handle different architectures (Llama-style, LFM2, etc.)

        # Standard transformer: self_attn + mlp
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if hasattr(attn, name):
                    w = getattr(attn, name).weight
                    layer_data[name] = analyze_weight_matrix(w, name)

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(mlp, name):
                    w = getattr(mlp, name).weight
                    layer_data[name] = analyze_weight_matrix(w, name)

        # LFM2-style: conv + feed_forward
        if hasattr(layer, "feed_forward"):
            ff = layer.feed_forward
            # Standard naming
            for name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(ff, name):
                    w = getattr(ff, name).weight
                    layer_data[f"ff_{name}"] = analyze_weight_matrix(w, name)
            # LFM2 naming (w1, w2, w3)
            for name in ["w1", "w2", "w3"]:
                if hasattr(ff, name):
                    w = getattr(ff, name).weight
                    layer_data[f"ff_{name}"] = analyze_weight_matrix(w, name)

        if hasattr(layer, "conv"):
            conv = layer.conv
            if hasattr(conv, "linear"):
                w = conv.linear.weight
                layer_data["conv_linear"] = analyze_weight_matrix(w, "conv_linear")

        # Fallback: scan for Linear modules with 2D weight matrices
        if not layer_data:
            for key in layer.keys() if hasattr(layer, "keys") else []:
                submod = layer[key]
                if hasattr(submod, "weight") and len(submod.weight.shape) == 2:
                    w = submod.weight
                    layer_data[key] = analyze_weight_matrix(w, key)

        layer_results[f"layer_{idx}"] = layer_data

    # Compute summary statistics
    all_sparsities = []
    all_eff_ranks = []
    for layer_data in layer_results.values():
        for matrix_data in layer_data.values():
            all_sparsities.append(matrix_data["sparsity"])
            all_eff_ranks.append(matrix_data["effective_rank"])

    summary = {
        "mean_sparsity": statistics.mean(all_sparsities) if all_sparsities else 0.0,
        "mean_effective_rank": statistics.mean(all_eff_ranks) if all_eff_ranks else 0.0,
        "layers_analyzed": len(layer_results),
        "matrices_analyzed": len(all_sparsities),
    }

    result = {
        "model": str(model_path),
        "total_layers": n_layers,
        "summary": summary,
        "layers": layer_results,
    }

    if context.output_format == "text":
        lines = [
            f"WEIGHT ANALYSIS",
            f"Model: {model_path}",
            f"Layers analyzed: {summary['layers_analyzed']} / {n_layers}",
            f"",
            f"Summary:",
            f"  Mean Sparsity: {summary['mean_sparsity']:.2%}",
            f"  Mean Effective Rank: {summary['mean_effective_rank']:.1f}",
            f"",
        ]
        for layer_name, layer_data in layer_results.items():
            lines.append(f"{layer_name}:")
            for matrix_name, matrix_data in layer_data.items():
                lines.append(
                    f"  {matrix_name}: sparsity={matrix_data['sparsity']:.2%}, "
                    f"eff_rank={matrix_data['effective_rank']:.1f}"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(result, context.output_format, context.pretty)
