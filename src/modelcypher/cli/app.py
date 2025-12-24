from __future__ import annotations

import json
import sys
import time
from pathlib import Path


import typer
from typer.core import TyperGroup

from modelcypher.cli.typer_compat import apply_typer_compat
apply_typer_compat()

from modelcypher.cli.composition import (
    get_compare_service,
    get_dataset_editor_service,
    get_dataset_service,
    get_evaluation_service,
    get_training_service,
)
from modelcypher.adapters.asif_packager import ASIFPackager
from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.commands.geometry import emotion as geometry_emotion_commands
from modelcypher.cli.presenters import (
    compare_detail_payload,
    compare_list_payload,
    dataset_convert_payload,
    dataset_edit_payload,
    dataset_payload,
    dataset_preview_payload,
    dataset_row_payload,
    doc_convert_payload,
    evaluation_detail_payload,
    evaluation_list_payload,
    model_search_payload,
    model_payload,
)
from modelcypher.cli.dataset_fields import parse_fields, parse_format, preview_line, pretty_fields
from modelcypher.core.domain.model_search import (
    MemoryFitStatus,
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.core.domain.training import Hyperparameters, LoRAConfig, TrainingConfig
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.compare_service import CompareService
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService
from modelcypher.core.use_cases.doc_service import DocService
from modelcypher.core.use_cases.evaluation_service import EvaluationService
from modelcypher.core.use_cases.export_service import ExportService
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService
from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService
from modelcypher.core.use_cases.geometry_persona_service import GeometryPersonaService
from modelcypher.core.use_cases.geometry_transport_service import GeometryTransportService, MergeConfig
from modelcypher.core.domain.geometry.refinement_density import (
    RefinementDensityAnalyzer,
    RefinementDensityConfig,
)
from modelcypher.cli.commands import entropy as entropy_commands
from modelcypher.cli.commands import agent_eval as agent_eval_commands
from modelcypher.cli.commands import thermo as thermo_commands
from modelcypher.cli.commands import train as train_commands
from modelcypher.cli.commands import job as job_commands
from modelcypher.cli.commands import model as model_commands
from modelcypher.cli.commands import dataset as dataset_commands
from modelcypher.cli.commands import system as system_commands
from modelcypher.cli.commands import eval as eval_commands
from modelcypher.cli.commands import adapter as adapter_commands
from modelcypher.cli.commands import safety as safety_commands
from modelcypher.cli.commands import agent as agent_commands
from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
from modelcypher.cli.commands.geometry import sparse as geometry_sparse_commands
from modelcypher.cli.commands.geometry import refusal as geometry_refusal_commands
from modelcypher.cli.commands.geometry import persona as geometry_persona_commands
from modelcypher.cli.commands.geometry import manifold as geometry_manifold_commands
from modelcypher.cli.commands.geometry import transport as geometry_transport_commands
from modelcypher.cli.commands.geometry import refinement as geometry_refinement_commands
from modelcypher.cli.commands.geometry import invariant as geometry_invariant_commands
from modelcypher.cli.commands.geometry import training as geometry_training_commands
from modelcypher.cli.commands.geometry import safety as geometry_safety_commands
from modelcypher.cli.commands.geometry import geom_adapter as geometry_adapter_commands
from modelcypher.cli.commands.geometry import primes as geometry_primes_commands
from modelcypher.cli.commands.geometry import stitch as geometry_stitch_commands
from modelcypher.cli.commands.geometry import crm as geometry_crm_commands
from modelcypher.cli.commands.geometry import path as geometry_path_commands
from modelcypher.cli.commands.geometry import merge_entropy as geometry_merge_entropy_commands
from modelcypher.cli.commands.geometry import transfer as geometry_transfer_cabe_commands
from modelcypher.cli.commands.geometry import spatial as geometry_spatial_commands
from modelcypher.cli.commands.geometry import social as geometry_social_commands
from modelcypher.cli.commands.geometry import temporal as geometry_temporal_commands
from modelcypher.cli.commands.geometry import moral as geometry_moral_commands
from modelcypher.cli.commands.geometry import waypoint as geometry_waypoint_commands
from modelcypher.cli.commands.geometry import interference as geometry_interference_commands
from modelcypher.cli.commands import research as research_commands
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.inventory_service import InventoryService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.model_merge_service import ModelMergeService
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
from modelcypher.core.use_cases.model_search_service import ModelSearchService
from modelcypher.core.use_cases.model_service import ModelService
from modelcypher.core.use_cases.system_service import SystemService
from modelcypher.core.use_cases.training_service import TrainingService
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json
from modelcypher.utils.logging import configure_logging
from modelcypher.utils.limits import MAX_FIELD_BYTES, MAX_PREVIEW_LINES, MAX_RAW_BYTES


_GLOBAL_FLAGS_WITH_VALUES = {"--output", "--log-level", "--trace-id"}
_GLOBAL_FLAG_ALIASES = {
    "--ai",
    "--output",
    "--quiet",
    "--very-quiet",
    "--yes",
    "--no-prompt",
    "--pretty",
    "--log-level",
    "--trace-id",
}


def _hoist_global_flags(args: list[str]) -> list[str]:
    """Allow global flags to appear anywhere in the command.

    Click/Typer only parse group-level options *before* the subcommand token.
    ModelCypher-style usage places flags at the end (e.g. `mc inventory --output json`).

    This pre-parser moves known global flags (and their values) to the front so the
    Typer app callback can consume them, without requiring every subcommand to
    re-declare the same options.
    """

    extracted: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            remaining.extend(args[i:])
            break

        if any(arg.startswith(f"{flag}=") for flag in _GLOBAL_FLAGS_WITH_VALUES):
            extracted.append(arg)
            i += 1
            continue

        if arg in _GLOBAL_FLAGS_WITH_VALUES:
            extracted.append(arg)
            if i + 1 < len(args):
                extracted.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue

        if arg in _GLOBAL_FLAG_ALIASES:
            extracted.append(arg)
            i += 1
            continue

        remaining.append(arg)
        i += 1

    return extracted + remaining


class _GlobalOptionsTyperGroup(TyperGroup):
    def parse_args(self, ctx, args: list[str]) -> list[str]:
        return super().parse_args(ctx, _hoist_global_flags(args))


app = typer.Typer(no_args_is_help=True, add_completion=False, cls=_GlobalOptionsTyperGroup)
doc_app = typer.Typer(no_args_is_help=True)
validate_app = typer.Typer(no_args_is_help=True)
estimate_app = typer.Typer(no_args_is_help=True)
geometry_app = typer.Typer(no_args_is_help=True)

app.add_typer(train_commands.train_app, name="train")
app.add_typer(job_commands.app, name="job")
app.add_typer(train_commands.checkpoint_app, name="checkpoint")
app.add_typer(model_commands.app, name="model")
app.add_typer(system_commands.app, name="system")
app.add_typer(dataset_commands.app, name="dataset")
app.add_typer(eval_commands.eval_app, name="eval")
app.add_typer(eval_commands.compare_app, name="compare")
app.add_typer(doc_app, name="doc")
app.add_typer(validate_app, name="validate")
app.add_typer(estimate_app, name="estimate")
app.add_typer(geometry_app, name="geometry")
geometry_app.add_typer(geometry_path_commands.app, name="path")
geometry_app.add_typer(geometry_training_commands.app, name="training")
geometry_app.add_typer(geometry_safety_commands.app, name="safety")
geometry_app.add_typer(geometry_adapter_commands.app, name="adapter")
geometry_app.add_typer(geometry_primes_commands.app, name="primes")
geometry_app.add_typer(geometry_stitch_commands.app, name="stitch")
geometry_app.add_typer(geometry_crm_commands.app, name="crm")
geometry_app.add_typer(geometry_metrics_commands.app, name="metrics")
geometry_app.add_typer(geometry_sparse_commands.app, name="sparse")
geometry_app.add_typer(geometry_refusal_commands.app, name="refusal")
geometry_app.add_typer(geometry_persona_commands.app, name="persona")
geometry_app.add_typer(geometry_manifold_commands.app, name="manifold")
geometry_app.add_typer(geometry_transport_commands.app, name="transport")
geometry_app.add_typer(geometry_refinement_commands.app, name="refinement")
geometry_app.add_typer(geometry_invariant_commands.app, name="invariant")
geometry_app.add_typer(geometry_emotion_commands.app, name="emotion")
geometry_app.add_typer(geometry_merge_entropy_commands.app, name="merge-entropy")
geometry_app.add_typer(geometry_transfer_cabe_commands.app, name="transfer")
geometry_app.add_typer(geometry_spatial_commands.app, name="spatial")
geometry_app.add_typer(geometry_social_commands.app, name="social")
geometry_app.add_typer(geometry_temporal_commands.app, name="temporal")
geometry_app.add_typer(geometry_moral_commands.app, name="moral")
geometry_app.add_typer(geometry_waypoint_commands.app, name="waypoint")
geometry_app.add_typer(geometry_interference_commands.app, name="interference")
app.add_typer(entropy_commands.app, name="entropy")
app.add_typer(adapter_commands.adapter_app, name="adapter")
app.add_typer(adapter_commands.calibration_app, name="calibration")
app.add_typer(thermo_commands.app, name="thermo")
app.add_typer(safety_commands.app, name="safety")
app.add_typer(agent_commands.app, name="agent")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    ai: bool | None = typer.Option(None, "--ai", help="AI mode: JSON output, no prompts"),
    output: str | None = typer.Option(None, "--output", help="Output format: json, yaml, text"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress info logs"),
    very_quiet: bool = typer.Option(False, "--very-quiet", help="Suppress all logs"),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm prompts"),
    no_prompt: bool = typer.Option(False, "--no-prompt", help="Fail if confirmation required"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print JSON output"),
    log_level: str = typer.Option("info", "--log-level", help="Log level: trace, debug, info, warn, error"),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Trace ID"),
) -> None:
    ai_mode = resolve_ai_mode(ai)
    output_format = resolve_output_format(ai_mode, output)
    quiet_mode = very_quiet or quiet or ai_mode
    effective_log_level = "error" if very_quiet else log_level
    configure_logging(effective_log_level, quiet=quiet_mode)

    ctx.obj = CLIContext(
        ai_mode=ai_mode,
        output_format=output_format,
        quiet=quiet,
        very_quiet=very_quiet,
        yes=yes or ai_mode,
        no_prompt=no_prompt or ai_mode,
        pretty=pretty,
        log_level=effective_log_level,
        trace_id=trace_id,
    )


@app.command("inventory")
def inventory(ctx: typer.Context) -> None:
    context = _context(ctx)
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory

    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)
    service = factory.inventory_service()
    write_output(service.inventory(), context.output_format, context.pretty)


@app.command("explain")
def explain(ctx: typer.Context, command: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()
    payload = service.explain(command)
    write_output(payload, context.output_format, context.pretty)


@doc_app.command("convert")
def doc_convert(
    ctx: typer.Context,
    input: list[str] = typer.Option(..., "--input"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
    chunk_size: int = typer.Option(2000, "--chunk-size"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap"),
    text_only: bool = typer.Option(True, "--text-only"),
    stream: bool = typer.Option(False, "--stream"),
    update_manifest: bool = typer.Option(False, "--update-manifest"),
) -> None:
    context = _context(ctx)
    service = DocService()
    result, events = service.convert(
        inputs=input,
        output_path=output_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        text_only=text_only,
        stream=stream,
        update_manifest=update_manifest,
    )
    if stream:
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
        return
    write_output(doc_convert_payload(result), context.output_format, context.pretty)


@doc_app.command("validate")
def doc_validate(ctx: typer.Context, path: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = get_dataset_service()
    result = service.validate_dataset(path)
    write_output({"valid": result["valid"], "samples": result["totalExamples"], "errors": result["errors"], "warnings": result["warnings"]}, context.output_format, context.pretty)


@validate_app.command("train")
def validate_train(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    batch_size: int = typer.Option(4, "--batch-size"),
    sequence_length: int = typer.Option(2048, "--sequence-length"),
    epochs: int = typer.Option(1, "--epochs"),
) -> None:
    context = _context(ctx)
    service = get_training_service()
    hyperparams = Hyperparameters(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        sequence_length=sequence_length,
    )
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        output_path=".",
        hyperparameters=hyperparams,
    )
    result = service.preflight(config)
    payload = {
        "valid": result["canProceed"],
        "model": {"id": model, "found": True, "architecture": None},
        "dataset": {"path": dataset, "exists": True, "readable": True},
        "memory": {
            "willFit": result["canProceed"],
            "recommendedBatchSize": result["predictedBatchSize"],
            "projectedPeakGB": None,
            "availableGB": None,
        },
        "config": {
            "batchSize": batch_size,
            "sequenceLength": sequence_length,
            "learningRate": learning_rate,
            "epochs": epochs,
        },
        "warnings": [],
        "errors": [] if result["canProceed"] else ["Configuration may not fit in memory"],
        "nextActions": [f"mc train start --model {model} --dataset {dataset}"],
    }
    write_output(payload, context.output_format, context.pretty)


@validate_app.command("dataset")
def validate_dataset(ctx: typer.Context, path: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = get_dataset_service()
    result = service.validate_dataset(path)
    payload = {
        "valid": result["valid"],
        "path": path,
        "exists": True,
        "readable": True,
        "format": "jsonl",
        "exampleCount": result["totalExamples"],
        "tokenStats": {
            "average": result["averageTokens"],
            "min": result["minTokens"],
            "max": result["maxTokens"],
        },
        "warnings": result["warnings"],
        "errors": result["errors"],
        "nextActions": ["mc train start --model <model> --dataset <dataset>"],
    }
    write_output(payload, context.output_format, context.pretty)


@estimate_app.command("train")
def estimate_train(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    batch_size: int = typer.Option(1, "--batch-size"),
    sequence_length: int = typer.Option(2048, "--sequence-length"),
    dtype: str = typer.Option("fp16", "--dtype"),
) -> None:
    context = _context(ctx)
    service = get_training_service()
    hyperparams = Hyperparameters(
        batch_size=batch_size,
        learning_rate=1e-5,
        epochs=1,
        sequence_length=sequence_length,
    )
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        output_path=".",
        hyperparameters=hyperparams,
    )
    result = service.preflight(config)
    payload = {
        "willFit": result["canProceed"],
        "recommendedBatchSize": result["predictedBatchSize"],
        "projectedPeakGB": result["estimatedVRAMUsageBytes"] / (1024**3) if result["estimatedVRAMUsageBytes"] else None,
        "availableGB": result["availableVRAMBytes"] / (1024**3) if result["availableVRAMBytes"] else None,
        "ttftSeconds": None,
        "tokensPerSecond": None,
        "tokensPerSecondMin": None,
        "tokensPerSecondMax": None,
        "confidence": "low",
        "powerSource": "unknown",
        "thermalState": "unknown",
        "etaSeconds": None,
        "notes": [f"dtype={dtype}"],
        "nextActions": [f"mc train start --model {model} --dataset {dataset} --batch-size {batch_size}"],
    }
    write_output(payload, context.output_format, context.pretty)


@geometry_app.command("validate")
def geometry_validate(
    ctx: typer.Context,
    include_fixtures: bool = typer.Option(False, "--include-fixtures"),
    file: str | None = typer.Option(None, "--file"),
) -> None:
    context = _context(ctx)
    embedder = EmbeddingDefaults.make_default_embedder()
    service = GeometryService(embedder=embedder)
    report = service.validate(include_fixtures=include_fixtures)
    payload = service.validation_payload(
        report,
        include_schema=context.output_format in {"json", "yaml"},
    )
    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        status = "PASS" if report.passed else "FAIL"
        lines = [
            f"Geometry validation: {status}",
            f"GW distance (perm): {report.gromov_wasserstein.distance_permutation:.6f}",
            f"GW symmetry delta: {report.gromov_wasserstein.symmetry_delta:.6f}",
            f"Traversal coherence (self): {report.traversal_coherence.self_correlation:.5f}",
            f"Path signature similarity: {report.path_signature.signature_similarity:.5f}",
            f"Frechet distance: {report.path_signature.frechet_distance:.6f}",
        ]
        if report.fixtures is not None:
            lines.append("Fixtures: included")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("infer")
def infer(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    prompt: str = typer.Option(..., "--prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.95, "--top-p"),
    scan: bool = typer.Option(False, "--scan", help="Run security scan on output"),
) -> None:
    context = _context(ctx)
    engine = LocalInferenceEngine()
    
    # Use the more capable 'run' method
    from dataclasses import asdict
    result = engine.run(
        model=model, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p,
        security_scan=scan
    )
    
    # Convert dataclass to dict for output
    payload = asdict(result)
    
    # Flatten security info for easier reading if present
    if result.security:
        payload["securityAssessment"] = result.security.security_assessment
        payload["securityAnomalies"] = result.security.anomaly_count
    
    write_output(payload, context.output_format, context.pretty)


# RAG commands
rag_app = typer.Typer(no_args_is_help=True)
app.add_typer(rag_app, name="rag")


def _expand_rag_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_path in paths:
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        if resolved.is_dir():
            for candidate in resolved.rglob("*"):
                if candidate.is_file():
                    expanded.append(str(candidate))
        else:
            expanded.append(str(resolved))
    if not expanded:
        raise ValueError("No files found to index.")
    return expanded


@rag_app.command("index")
def rag_index(
    ctx: typer.Context,
    documents: list[str] = typer.Option(..., "--document", help="Document paths"),
    output_path: str | None = typer.Option(None, "--output-path", help="Index output path"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size"),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap", help="Chunk overlap"),
) -> None:
    """Create a vector index from documents."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()

    try:
        result = service.index(documents, output_path, chunk_size, chunk_overlap)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1010",
            title="RAG indexing failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "indexId": result.index_id,
        "documentCount": result.document_count,
        "totalChunks": result.total_chunks,
        "indexPath": result.index_path,
        "createdAt": result.created_at,
        "embeddingModel": result.embedding_model,
        "embeddingDimension": result.embedding_dimension,
    }

    write_output(payload, context.output_format, context.pretty)


@rag_app.command("build")
def rag_build(
    ctx: typer.Context,
    index_name: str = typer.Option(..., "--index-name", help="Name for the RAG index"),
    paths: list[str] = typer.Option(..., "--path", help="Document files or directories"),
    model_path: str = typer.Option(..., "--model-path", help="Embedding model path"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap"),
) -> None:
    """Build a RAG index from documents."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()

    try:
        expanded_paths = _expand_rag_paths(paths)
        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.exists() or not resolved_model.is_dir():
            raise ValueError(f"Model path is not a directory: {resolved_model}")
        result = service.index(
            expanded_paths,
            output_path=None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_name=index_name,
            model_path=str(resolved_model),
            embedding_model=embedding_model,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1010",
            title="RAG build failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "taskId": f"rag-{result.index_id}",
        "status": "completed",
        "indexName": index_name,
    }

    write_output(payload, context.output_format, context.pretty)


@rag_app.command("query")
def rag_query(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Query string"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
) -> None:
    """Query the index for relevant documents."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()

    try:
        result = service.query(query, top_k)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2010",
            title="RAG query failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "query": result.query,
        "results": result.results,
        "totalResults": result.total_results,
        "queryTimeMs": result.query_time_ms,
    }

    write_output(payload, context.output_format, context.pretty)


@rag_app.command("list")
def rag_list(ctx: typer.Context) -> None:
    """List available RAG indexes."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()
    systems = service.list_indexes()

    payload = {
        "systems": [
            {
                "id": system.system_id,
                "name": system.name,
                "modelPath": system.model_path,
                "embeddingModel": system.embedding_model,
                "documentCount": system.document_count,
                "chunkCount": system.chunk_count,
                "createdAt": system.created_at,
            }
            for system in systems
        ],
        "count": len(systems),
    }

    write_output(payload, context.output_format, context.pretty)


@rag_app.command("delete")
def rag_delete(
    ctx: typer.Context,
    index_name: str = typer.Argument(..., help="Index id or name"),
) -> None:
    """Delete a RAG index by id or name."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()
    deleted = service.delete_index(index_name)
    if not deleted:
        error = ErrorDetail(
            code="MC-2011",
            title="RAG index not found",
            detail=f"Index '{index_name}' does not exist",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {"deleted": index_name}
    write_output(payload, context.output_format, context.pretty)


@rag_app.command("status")
def rag_status(ctx: typer.Context) -> None:
    """Get RAG index status and statistics."""
    context = _context(ctx)
    from modelcypher.core.use_cases.rag_service import RAGService

    service = RAGService()
    result = service.status()

    payload = {
        "indexId": result.index_id,
        "status": result.status,
        "documentCount": result.document_count,
        "chunkCount": result.chunk_count,
        "indexSizeBytes": result.index_size_bytes,
        "lastUpdated": result.last_updated,
        "embeddingModel": result.embedding_model,
    }

    write_output(payload, context.output_format, context.pretty)


# Stability commands
stability_app = typer.Typer(no_args_is_help=True)
app.add_typer(stability_app, name="stability")


@stability_app.command("run")
def stability_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    num_runs: int = typer.Option(10, "--num-runs", help="Number of test runs"),
    prompt_variations: int = typer.Option(5, "--prompt-variations", help="Prompt variations"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Execute stability suite on a model."""
    context = _context(ctx)
    from modelcypher.core.use_cases.stability_service import (
        StabilityConfig,
        StabilityService,
    )

    config = StabilityConfig(
        num_runs=num_runs,
        prompt_variations=prompt_variations,
        seed=seed,
    )
    service = StabilityService()

    try:
        result = service.run(model, config)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1011",
            title="Stability test failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "suiteId": result.suite_id,
        "modelPath": result.model_path,
        "status": result.status,
        "startedAt": result.started_at,
        "config": result.config,
        "summary": result.summary,
    }

    write_output(payload, context.output_format, context.pretty)


@stability_app.command("report")
def stability_report(
    ctx: typer.Context,
    suite_id: str = typer.Argument(..., help="Stability suite ID"),
) -> None:
    """Get detailed stability report."""
    context = _context(ctx)
    from modelcypher.core.use_cases.stability_service import StabilityService

    service = StabilityService()

    try:
        result = service.report(suite_id)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2011",
            title="Stability suite not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "suiteId": result.suite_id,
        "modelPath": result.model_path,
        "status": result.status,
        "startedAt": result.started_at,
        "completedAt": result.completed_at,
        "config": result.config,
        "metrics": result.metrics,
        "perPromptResults": result.per_prompt_results,
        "interpretation": result.interpretation,
        "recommendations": result.recommendations,
    }

    write_output(payload, context.output_format, context.pretty)


# Agent-eval commands (extracted to commands/agent_eval.py)
app.add_typer(agent_eval_commands.app, name="agent-eval")

# Dashboard commands
dashboard_app = typer.Typer(no_args_is_help=True)
app.add_typer(dashboard_app, name="dashboard")


@dashboard_app.command("metrics")
def dashboard_metrics(ctx: typer.Context) -> None:
    """Return current metrics in Prometheus format."""
    context = _context(ctx)
    from modelcypher.core.use_cases.dashboard_service import DashboardService

    service = DashboardService()
    metrics = service.metrics()

    if context.output_format == "json":
        # Parse prometheus format to JSON
        lines = metrics.strip().split("\n")
        metric_dict = {}
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(" ")
            if len(parts) >= 2:
                metric_dict[parts[0]] = parts[1]
        write_output(metric_dict, context.output_format, context.pretty)
    else:
        # Output raw prometheus format
        sys.stdout.write(metrics + "\n")


@dashboard_app.command("export")
def dashboard_export(
    ctx: typer.Context,
    format: str = typer.Option("prometheus", "--format", help="Export format"),
    output_path: str | None = typer.Option(None, "--output-path", help="Output path"),
) -> None:
    """Export dashboard data."""
    context = _context(ctx)
    from modelcypher.core.use_cases.dashboard_service import DashboardService

    service = DashboardService()

    try:
        result = service.export(format, output_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1013",
            title="Dashboard export failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "format": result.format,
        "exportPath": result.export_path,
        "exportedAt": result.exported_at,
        "metricsCount": result.metrics_count,
    }

    if context.output_format == "text" and not output_path:
        # Print content directly
        sys.stdout.write(result.content + "\n")
    else:
        write_output(payload, context.output_format, context.pretty)


# Help commands
help_app = typer.Typer(no_args_is_help=True)
app.add_typer(help_app, name="help")


@help_app.command("ask")
def help_ask(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Question about ModelCypher"),
) -> None:
    """Get contextual help for a question."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()
    result = service.ask(question)

    payload = {
        "question": result.question,
        "answer": result.answer,
        "relatedCommands": result.related_commands,
        "examples": result.examples,
        "docsUrl": result.docs_url,
    }

    if context.output_format == "text":
        lines = [
            f"Q: {result.question}",
            "",
            result.answer,
            "",
            "Related commands:",
        ]
        for cmd in result.related_commands:
            lines.append(f"  - {cmd}")
        lines.append("")
        lines.append("Examples:")
        for ex in result.examples:
            lines.append(f"  $ {ex}")
        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("completions")
def completions(
    ctx: typer.Context,
    shell: str = typer.Argument(..., help="Shell type: bash, zsh, fish"),
) -> None:
    """Generate shell completion script."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()

    try:
        script = service.completions(shell)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1014",
            title="Completions generation failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    sys.stdout.write(script)


@app.command("schema")
def schema(
    ctx: typer.Context,
    command: str = typer.Argument(..., help="Command name"),
) -> None:
    """Return JSON schema for command output."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()

    try:
        result = service.schema(command)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2014",
            title="Schema not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    write_output(result, context.output_format, context.pretty)


# Infer commands (additional batch/suite commands)
infer_app = typer.Typer(no_args_is_help=True)
app.add_typer(infer_app, name="infer")


@infer_app.command("run")
def infer_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter directory"),
    security_scan: bool = typer.Option(False, "--security-scan", help="Perform dual-path security analysis"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per response"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
) -> None:
    """Execute inference with optional adapter and security scanning."""
    context = _context(ctx)
    engine = LocalInferenceEngine()

    try:
        result = engine.run(
            model=model,
            prompt=prompt,
            adapter=adapter,
            security_scan=security_scan,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1015",
            title="Inference failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1017",
            title="Inference locked",
            detail=str(exc),
            hint="Wait for training to complete or cancel it",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "model": result.model,
        "prompt": result.prompt,
        "response": result.response,
        "tokenCount": result.token_count,
        "tokensPerSecond": result.tokens_per_second,
        "timeToFirstToken": result.time_to_first_token,
        "totalDuration": result.total_duration,
        "stopReason": result.stop_reason,
        "adapter": result.adapter,
    }

    if result.security:
        payload["security"] = {
            "securityAssessment": result.security.security_assessment,
            "anomalyCount": result.security.anomaly_count,
            "maxAnomalyScore": result.security.max_anomaly_score,
            "avgDelta": result.security.avg_delta,
            "disagreementRate": result.security.disagreement_rate,
            "circuitBreakerTripped": result.security.circuit_breaker_tripped,
            "circuitBreakerTripIndex": result.security.circuit_breaker_trip_index,
        }

    if context.output_format == "text":
        lines = [
            "INFERENCE RESULT",
            f"Model: {result.model}",
            f"Prompt: {result.prompt[:50]}...",
            f"Response: {result.response[:100]}...",
            f"Tokens: {result.token_count} ({result.tokens_per_second:.1f} tok/s)",
            f"Duration: {result.total_duration:.2f}s",
        ]
        if result.adapter:
            lines.append(f"Adapter: {result.adapter}")
        if result.security:
            lines.append(f"Security: {result.security.security_assessment}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@infer_app.command("suite")
def infer_suite(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    suite_file: str = typer.Option(..., "--suite", help="Path to suite file (.txt, .json, .jsonl)"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter directory"),
    security_scan: bool = typer.Option(False, "--security-scan", help="Perform security analysis"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Default max tokens"),
    temperature: float = typer.Option(0.7, "--temperature", help="Default temperature"),
) -> None:
    """Execute batched inference over a suite of prompts."""
    context = _context(ctx)
    engine = LocalInferenceEngine()

    try:
        result = engine.suite(
            model=model,
            suite_file=suite_file,
            adapter=adapter,
            security_scan=security_scan,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1016",
            title="Inference suite failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Convert cases to dict format
    cases_payload = []
    for case in result.cases:
        case_dict = {
            "name": case.name,
            "prompt": case.prompt,
            "response": case.response,
            "tokenCount": case.token_count,
            "duration": case.duration,
            "passed": case.passed,
            "expected": case.expected,
        }
        if case.error:
            case_dict["error"] = case.error
        cases_payload.append(case_dict)

    payload = {
        "model": result.model,
        "adapter": result.adapter,
        "suite": result.suite,
        "totalCases": result.total_cases,
        "passed": result.passed,
        "failed": result.failed,
        "totalDuration": result.total_duration,
        "summary": result.summary,
        "cases": cases_payload[:10],  # Limit cases in output
    }

    if context.output_format == "text":
        lines = [
            "INFERENCE SUITE RESULTS",
            f"Model: {result.model}",
            f"Suite: {result.suite}",
            f"Cases: {result.total_cases} ({result.passed} passed, {result.failed} failed)",
        ]
        if result.summary.get("pass_rate") is not None:
            lines.append(f"Pass Rate: {result.summary.get('pass_rate', 0) * 100:.1f}%")
        lines.extend([
            f"Duration: {result.total_duration:.2f}s",
            "",
            "Case Results:",
        ])
        for case in result.cases:
            if case.passed is not None:
                status = "✓" if case.passed else "✗"
            else:
                status = "○"
            lines.append(f"  {status} {case.name}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Storage commands
storage_app = typer.Typer(no_args_is_help=True)
app.add_typer(storage_app, name="storage")


@storage_app.command("status")
def storage_status(ctx: typer.Context) -> None:
    """Return storage usage breakdown by category."""
    context = _context(ctx)
    from modelcypher.core.use_cases.storage_service import StorageService

    service = StorageService()
    snapshot = service.compute_snapshot()
    usage = snapshot.usage
    disk = snapshot.disk

    payload = {
        "totalGb": usage.total_gb,
        "modelsGb": usage.models_gb,
        "checkpointsGb": usage.checkpoints_gb,
        "otherGb": usage.other_gb,
        "disk": {
            "totalBytes": disk.total_bytes,
            "freeBytes": disk.free_bytes,
        },
    }

    if context.output_format == "text":
        lines = [
            "STORAGE STATUS",
            f"Total Disk: {usage.total_gb:.2f} GB",
            f"Free Disk: {disk.free_bytes / (1024**3):.2f} GB",
            "",
            "Usage Breakdown:",
            f"  Models: {usage.models_gb:.2f} GB",
            f"  Checkpoints: {usage.checkpoints_gb:.2f} GB",
            f"  Other: {usage.other_gb:.2f} GB",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@storage_app.command("usage")
def storage_usage(ctx: typer.Context) -> None:
    """Alias for storage status to match MCP naming."""
    storage_status(ctx)


@storage_app.command("cleanup")
def storage_cleanup(
    ctx: typer.Context,
    targets: list[str] = typer.Option(..., "--target", help="Cleanup targets: caches, rag"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview cleanup without deleting"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Remove old artifacts and return freed space."""
    context = _context(ctx)
    from modelcypher.core.use_cases.storage_service import StorageService

    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        targets_str = ", ".join(targets)
        if not typer.confirm(f"Clean up {targets_str}? This cannot be undone."):
            raise typer.Exit(code=1)

    service = StorageService()

    # Get before snapshot for comparison
    before_snapshot = service.compute_snapshot()

    if dry_run:
        payload = {
            "dryRun": True,
            "targets": targets,
            "message": "Dry run - no files deleted",
        }
        write_output(payload, context.output_format, context.pretty)
        return

    try:
        cleared = service.cleanup(targets)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1018",
            title="Storage cleanup failed",
            detail=str(exc),
            hint="Valid targets are: caches, rag",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Get after snapshot
    after_snapshot = service.compute_snapshot()
    freed_bytes = before_snapshot.disk.free_bytes - after_snapshot.disk.free_bytes
    # freed_bytes can be negative if cleanup freed space
    freed_bytes = max(0, after_snapshot.disk.free_bytes - before_snapshot.disk.free_bytes)

    payload = {
        "freedBytes": freed_bytes,
        "freedGb": freed_bytes / (1024**3),
        "categoriesCleaned": cleared,
    }

    if context.output_format == "text":
        lines = [
            "STORAGE CLEANUP",
            f"Categories cleaned: {', '.join(cleared)}",
            f"Space freed: {freed_bytes / (1024**3):.2f} GB",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Ensemble commands
ensemble_app = typer.Typer(no_args_is_help=True)
app.add_typer(ensemble_app, name="ensemble")


@ensemble_app.command("create")
def ensemble_create(
    ctx: typer.Context,
    models: list[str] = typer.Option(..., "--model", help="Model paths to include in ensemble"),
    strategy: str = typer.Option("weighted", "--strategy", help="Routing strategy: weighted, routing, voting, cascade"),
    weights: list[float] | None = typer.Option(None, "--weight", help="Weights for weighted strategy (must sum to 1.0)"),
) -> None:
    """Create an ensemble configuration from multiple models."""
    context = _context(ctx)
    from modelcypher.core.use_cases.ensemble_service import EnsembleService

    service = EnsembleService()

    try:
        result = service.create(
            model_paths=models,
            strategy=strategy,
            weights=weights,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1019",
            title="Ensemble creation failed",
            detail=str(exc),
            hint="Ensure all model paths exist and strategy is valid",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "ensembleId": result.ensemble_id,
        "models": result.models,
        "routingStrategy": result.routing_strategy,
        "weights": result.weights,
        "createdAt": result.created_at,
        "configPath": result.config_path,
    }

    if context.output_format == "text":
        lines = [
            "ENSEMBLE CREATED",
            f"Ensemble ID: {result.ensemble_id}",
            f"Strategy: {result.routing_strategy}",
            f"Models: {len(result.models)}",
        ]
        for i, model in enumerate(result.models):
            weight = result.weights[i] if result.weights else 1.0 / len(result.models)
            lines.append(f"  - {model} (weight: {weight:.3f})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@ensemble_app.command("run")
def ensemble_run(
    ctx: typer.Context,
    ensemble_id: str = typer.Argument(..., help="Ensemble ID"),
    prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
) -> None:
    """Execute ensemble inference."""
    context = _context(ctx)
    from modelcypher.core.use_cases.ensemble_service import EnsembleService

    service = EnsembleService()

    try:
        result = service.run(
            ensemble_id=ensemble_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1020",
            title="Ensemble inference failed",
            detail=str(exc),
            hint="Ensure ensemble ID is valid and models are accessible",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "ensembleId": result.ensemble_id,
        "prompt": result.prompt[:100] if len(result.prompt) > 100 else result.prompt,
        "response": result.response,
        "modelContributions": result.model_contributions,
        "totalDuration": result.total_duration,
        "strategy": result.strategy,
        "modelsUsed": result.models_used,
        "aggregationMethod": result.aggregation_method,
    }

    if context.output_format == "text":
        lines = [
            "ENSEMBLE INFERENCE",
            f"Ensemble ID: {result.ensemble_id}",
            f"Strategy: {result.strategy}",
            f"Models used: {result.models_used}",
            f"Duration: {result.total_duration:.3f}s",
            "",
            "Response:",
            result.response,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@ensemble_app.command("list")
def ensemble_list(ctx: typer.Context) -> None:
    """List all ensemble configurations."""
    context = _context(ctx)
    from modelcypher.core.use_cases.ensemble_service import EnsembleService

    service = EnsembleService()
    ensembles = service.list_ensembles()

    payload = {
        "ensembles": [
            {
                "ensembleId": e.ensemble_id,
                "models": len(e.models),
                "strategy": e.routing_strategy,
                "createdAt": e.created_at,
            }
            for e in ensembles
        ],
        "count": len(ensembles),
    }

    if context.output_format == "text":
        if not ensembles:
            write_output("No ensembles found.", context.output_format, context.pretty)
            return
        lines = ["ENSEMBLES", ""]
        for e in ensembles:
            lines.append(f"  {e.ensemble_id}")
            lines.append(f"    Strategy: {e.routing_strategy}")
            lines.append(f"    Models: {len(e.models)}")
            lines.append(f"    Created: {e.created_at}")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@ensemble_app.command("delete")
def ensemble_delete(
    ctx: typer.Context,
    ensemble_id: str = typer.Argument(..., help="Ensemble ID to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Delete an ensemble configuration."""
    context = _context(ctx)
    from modelcypher.core.use_cases.ensemble_service import EnsembleService

    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete ensemble {ensemble_id}?"):
            raise typer.Exit(code=1)

    service = EnsembleService()
    deleted = service.delete(ensemble_id)

    if not deleted:
        error = ErrorDetail(
            code="MC-2005",
            title="Ensemble not found",
            detail=f"Ensemble '{ensemble_id}' does not exist",
            hint="Use 'mc ensemble list' to see available ensembles",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {"deleted": ensemble_id}

    if context.output_format == "text":
        write_output(f"Deleted ensemble: {ensemble_id}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Research commands
research_app = typer.Typer(no_args_is_help=True)
app.add_typer(research_app, name="research")
research_app.add_typer(research_commands.taxonomy_app, name="taxonomy")


@research_app.command("sparse-region")
def research_sparse_region(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Analyze sparse activation regions in a model."""
    context = _context(ctx)
    from modelcypher.core.use_cases.research_service import ResearchService

    service = ResearchService()

    try:
        result = service.sparse_region(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1021",
            title="Sparse region analysis failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": result.model_path,
        "totalSparsity": result.total_sparsity,
        "layerCount": result.layer_count,
        "regions": [
            {
                "layerName": r.layer_name,
                "startIndex": r.start_index,
                "endIndex": r.end_index,
                "sparsityRatio": r.sparsity_ratio,
                "activationPattern": r.activation_pattern,
            }
            for r in result.regions
        ],
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Model: {result.model_path}",
            f"Total Sparsity: {result.total_sparsity:.1%}",
            f"Layers Analyzed: {result.layer_count}",
            "",
            "Regions:",
        ]
        for r in result.regions[:10]:  # Limit to first 10 for readability
            lines.append(f"  {r.layer_name}")
            lines.append(f"    Sparsity: {r.sparsity_ratio:.1%}")
            lines.append(f"    Pattern: {r.activation_pattern}")
        if len(result.regions) > 10:
            lines.append(f"  ... and {len(result.regions) - 10} more regions")
        lines.append("")
        lines.append("Interpretation:")
        lines.append(result.interpretation)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@research_app.command("afm")
def research_afm(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Run activation function mapping analysis."""
    context = _context(ctx)
    from modelcypher.core.use_cases.research_service import ResearchService

    service = ResearchService()

    try:
        result = service.afm(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1022",
            title="AFM analysis failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": result.model_path,
        "dominantPatterns": result.dominant_patterns,
        "layerSummaries": [
            {
                "layerName": s.layer_name,
                "dominantPattern": s.dominant_pattern,
                "meanActivation": s.mean_activation,
                "maxActivation": s.max_activation,
            }
            for s in result.layer_summaries
        ],
        "activationMaps": {
            k: v[:5] for k, v in result.activation_maps.items()  # Limit values for output
        },
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "ACTIVATION FUNCTION MAPPING",
            f"Model: {result.model_path}",
            f"Dominant Patterns: {', '.join(result.dominant_patterns)}",
            "",
            "Layer Summaries:",
        ]
        for s in result.layer_summaries[:10]:  # Limit to first 10 for readability
            lines.append(f"  {s.layer_name}")
            lines.append(f"    Pattern: {s.dominant_pattern}")
            lines.append(f"    Mean Activation: {s.mean_activation:.4f}")
            lines.append(f"    Max Activation: {s.max_activation:.4f}")
        if len(result.layer_summaries) > 10:
            lines.append(f"  ... and {len(result.layer_summaries) - 10} more layers")
        lines.append("")
        lines.append("Interpretation:")
        lines.append(result.interpretation)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)

