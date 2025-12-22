from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from typer.core import TyperGroup

from modelcypher.cli.typer_compat import apply_typer_compat
apply_typer_compat()

from modelcypher.adapters.asif_packager import ASIFPackager
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
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
)
from modelcypher.core.domain.training import LoRAConfig, TrainingConfig
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.compare_service import CompareService
from modelcypher.core.use_cases.concept_response_matrix_service import (
    CRMBuildConfig,
    ConceptResponseMatrixService,
)
from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService
from modelcypher.core.use_cases.doc_service import DocService
from modelcypher.core.use_cases.evaluation_service import EvaluationService
from modelcypher.core.use_cases.export_service import ExportService
from modelcypher.core.use_cases.geometry_adapter_service import GeometryAdapterService
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService
from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService
from modelcypher.core.use_cases.geometry_persona_service import GeometryPersonaService
from modelcypher.core.use_cases.geometry_transport_service import GeometryTransportService, MergeConfig
from modelcypher.core.domain.geometry.refinement_density import (
    RefinementDensityAnalyzer,
    RefinementDensityConfig,
)
from modelcypher.core.use_cases.geometry_primes_service import GeometryPrimesService
from modelcypher.core.use_cases.geometry_safety_service import GeometrySafetyService
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService
from modelcypher.cli.commands import entropy as entropy_commands
from modelcypher.cli.commands import agent_eval as agent_eval_commands
from modelcypher.cli.commands import thermo as thermo_commands
from modelcypher.cli.commands import train as train_commands
from modelcypher.cli.commands import model as model_commands
from modelcypher.cli.commands import dataset as dataset_commands
from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
from modelcypher.cli.commands.geometry import sparse as geometry_sparse_commands
from modelcypher.cli.commands.geometry import refusal as geometry_refusal_commands
from modelcypher.cli.commands.geometry import persona as geometry_persona_commands
from modelcypher.cli.commands.geometry import manifold as geometry_manifold_commands
from modelcypher.cli.commands.geometry import transport as geometry_transport_commands
from modelcypher.cli.commands.geometry import refinement as geometry_refinement_commands
from modelcypher.core.use_cases.geometry_stitch_service import GeometryStitchService
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.geometry_training_service import GeometryTrainingService
from modelcypher.core.use_cases.inventory_service import InventoryService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.model_merge_service import ModelMergeService
from modelcypher.core.use_cases.invariant_layer_mapping_service import (
    InvariantLayerMappingService,
    LayerMappingConfig,
    CollapseRiskConfig,
)
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
_CRM_DEFAULTS = CRMBuildConfig()


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
system_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
compare_app = typer.Typer(no_args_is_help=True)
doc_app = typer.Typer(no_args_is_help=True)
validate_app = typer.Typer(no_args_is_help=True)
estimate_app = typer.Typer(no_args_is_help=True)
geometry_app = typer.Typer(no_args_is_help=True)
path_app = typer.Typer(no_args_is_help=True)
geometry_training_app = typer.Typer(no_args_is_help=True)
geometry_safety_app = typer.Typer(no_args_is_help=True)
geometry_adapter_app = typer.Typer(no_args_is_help=True)
geometry_primes_app = typer.Typer(no_args_is_help=True)
geometry_stitch_app = typer.Typer(no_args_is_help=True)
geometry_crm_app = typer.Typer(no_args_is_help=True)
geometry_invariant_app = typer.Typer(no_args_is_help=True)

app.add_typer(train_commands.train_app, name="train")
app.add_typer(train_commands.job_app, name="job")
app.add_typer(train_commands.checkpoint_app, name="checkpoint")
app.add_typer(model_commands.app, name="model")
app.add_typer(system_app, name="system")
app.add_typer(dataset_commands.app, name="dataset")
app.add_typer(eval_app, name="eval")
app.add_typer(compare_app, name="compare")
app.add_typer(doc_app, name="doc")
app.add_typer(validate_app, name="validate")
app.add_typer(estimate_app, name="estimate")
app.add_typer(geometry_app, name="geometry")
geometry_app.add_typer(path_app, name="path")
geometry_app.add_typer(geometry_training_app, name="training")
geometry_app.add_typer(geometry_safety_app, name="safety")
geometry_app.add_typer(geometry_adapter_app, name="adapter")
geometry_app.add_typer(geometry_primes_app, name="primes")
geometry_app.add_typer(geometry_stitch_app, name="stitch")
geometry_app.add_typer(geometry_crm_app, name="crm")
geometry_app.add_typer(geometry_metrics_commands.app, name="metrics")
geometry_app.add_typer(geometry_sparse_commands.app, name="sparse")
geometry_app.add_typer(geometry_refusal_commands.app, name="refusal")
geometry_app.add_typer(geometry_persona_commands.app, name="persona")
geometry_app.add_typer(geometry_manifold_commands.app, name="manifold")
geometry_app.add_typer(geometry_transport_commands.app, name="transport")
geometry_app.add_typer(geometry_refinement_commands.app, name="refinement")
geometry_app.add_typer(geometry_invariant_app, name="invariant")
geometry_app.add_typer(geometry_emotion_commands.app, name="emotion")
app.add_typer(entropy_commands.app, name="entropy")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    ai: Optional[bool] = typer.Option(None, "--ai", help="AI mode: JSON output, no prompts"),
    output: Optional[str] = typer.Option(None, "--output", help="Output format: json, yaml, text"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress info logs"),
    very_quiet: bool = typer.Option(False, "--very-quiet", help="Suppress all logs"),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm prompts"),
    no_prompt: bool = typer.Option(False, "--no-prompt", help="Fail if confirmation required"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print JSON output"),
    log_level: str = typer.Option("info", "--log-level", help="Log level: trace, debug, info, warn, error"),
    trace_id: Optional[str] = typer.Option(None, "--trace-id", help="Trace ID"),
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
    service = InventoryService()
    write_output(service.inventory(), context.output_format, context.pretty)


@app.command("explain")
def explain(ctx: typer.Context, command: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    payload = {
        "command": command,
        "serviceCalls": [],
        "affectedResources": [],
        "requiredPermissions": [],
        "warnings": [],
        "estimatedDuration": None,
    }
    write_output(payload, context.output_format, context.pretty)


@system_app.command("status")
def system_status(ctx: typer.Context, require_metal: bool = typer.Option(False, "--require-metal")) -> None:
    context = _context(ctx)
    service = SystemService()
    status = service.status()
    if require_metal and not status["metalAvailable"]:
        raise typer.Exit(code=3)
    write_output(status, context.output_format, context.pretty)


@system_app.command("probe")
def system_probe(ctx: typer.Context, target: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = SystemService()
    write_output(service.probe(target), context.output_format, context.pretty)


@eval_app.command("list")
def eval_list(ctx: typer.Context, limit: int = typer.Option(50, "--limit")) -> None:
    context = _context(ctx)
    service = EvaluationService()
    payload = service.list_evaluations(limit)
    results = payload["evaluations"] if isinstance(payload, dict) else payload
    write_output(evaluation_list_payload(results), context.output_format, context.pretty)


@eval_app.command("show")
def eval_show(ctx: typer.Context, eval_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = EvaluationService()
    result = service.get_evaluation(eval_id)
    write_output(evaluation_detail_payload(result), context.output_format, context.pretty)


@compare_app.command("list")
def compare_list(
    ctx: typer.Context,
    status: Optional[str] = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    context = _context(ctx)
    service = CompareService()
    payload = service.list_sessions(limit, status)
    sessions = payload["sessions"] if isinstance(payload, dict) else payload
    write_output(compare_list_payload(sessions), context.output_format, context.pretty)


@compare_app.command("show")
def compare_show(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = CompareService()
    result = service.get_session(session_id)
    write_output(compare_detail_payload(result), context.output_format, context.pretty)


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
    service = DatasetService()
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
    service = TrainingService()
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
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
    service = DatasetService()
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
    service = TrainingService()
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        learning_rate=1e-5,
        batch_size=batch_size,
        epochs=1,
        sequence_length=sequence_length,
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
    file: Optional[str] = typer.Option(None, "--file"),
) -> None:
    context = _context(ctx)
    service = GeometryService()
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


@path_app.command("detect")
def geometry_path_detect(
    ctx: typer.Context,
    text: str = typer.Argument(...),
    model: Optional[str] = typer.Option(None, "--model"),
    threshold: float = typer.Option(0.55, "--threshold"),
    file: Optional[str] = typer.Option(None, "--file"),
) -> None:
    context = _context(ctx)
    service = GeometryService()

    if model:
        engine = LocalInferenceEngine()
        result = engine.infer(model, text, max_tokens=200, temperature=0.0, top_p=1.0)
        text_to_analyze = result.get("response", "")
        model_id = Path(model).name if Path(model).exists() else model
    else:
        text_to_analyze = text
        model_id = "input-text"

    detection = service.detect_path(
        text_to_analyze,
        model_id=model_id,
        prompt_id="cli-input",
        threshold=threshold,
    )
    payload = service.detection_payload(detection)

    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        gates = " -> ".join(detection.gate_name_sequence) if detection.gate_name_sequence else "(none)"
        lines = [
            f"Gate Sequence: {gates}",
            "",
            "Detected Gates:",
        ]
        for gate in detection.detected_gates:
            lines.append(f"  [{gate.gate_name}] confidence={gate.confidence:.2f}")
            lines.append(f"    trigger: \"{gate.trigger_text}\"")
        lines.append("")
        lines.append(f"Mean Confidence: {detection.mean_confidence:.3f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@path_app.command("compare")
def geometry_path_compare(
    ctx: typer.Context,
    text_a: Optional[str] = typer.Option(None, "--text-a"),
    text_b: Optional[str] = typer.Option(None, "--text-b"),
    model_a: Optional[str] = typer.Option(None, "--model-a"),
    model_b: Optional[str] = typer.Option(None, "--model-b"),
    prompt: Optional[str] = typer.Option(None, "--prompt"),
    threshold: float = typer.Option(0.55, "--threshold"),
    file: Optional[str] = typer.Option(None, "--file"),
) -> None:
    context = _context(ctx)
    service = GeometryService()

    if text_a and text_b:
        text_to_analyze_a = text_a
        text_to_analyze_b = text_b
        model_id_a = "text-a"
        model_id_b = "text-b"
    elif model_a and model_b and prompt:
        engine = LocalInferenceEngine()
        response_a = engine.infer(model_a, prompt, max_tokens=200, temperature=0.0, top_p=1.0)
        response_b = engine.infer(model_b, prompt, max_tokens=200, temperature=0.0, top_p=1.0)
        text_to_analyze_a = response_a.get("response", "")
        text_to_analyze_b = response_b.get("response", "")
        model_id_a = Path(model_a).name if Path(model_a).exists() else model_a
        model_id_b = Path(model_b).name if Path(model_b).exists() else model_b
    else:
        raise typer.BadParameter(
            "Either --text-a and --text-b, or --model-a, --model-b, and --prompt are required."
        )

    result = service.compare_paths(
        text_a=text_to_analyze_a,
        text_b=text_to_analyze_b,
        model_a=model_id_a,
        model_b=model_id_b,
        prompt_id="compare",
        threshold=threshold,
    )

    payload = service.path_comparison_payload(result)
    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        path_a = " -> ".join(result.detection_a.gate_name_sequence) or "(none)"
        path_b = " -> ".join(result.detection_b.gate_name_sequence) or "(none)"
        lines = [
            f"Path A: {path_a}",
            f"Path B: {path_b}",
            "",
            "Path Comparison Results:",
            f"  Raw Distance: {result.comparison.total_distance:.3f}",
            f"  Normalized Distance: {result.comparison.normalized_distance:.3f}",
            "",
            "Alignment:",
        ]
        for step in result.comparison.alignment:
            op = step.op.value
            if op == "match":
                label = f"MATCH  {step.node_a.gate_id if step.node_a else '?'}"
            elif op == "substitute":
                left = step.node_a.gate_id if step.node_a else "?"
                right = step.node_b.gate_id if step.node_b else "?"
                label = f"SUBST  {left} -> {right} (cost: {step.cost:.2f})"
            elif op == "insert":
                label = f"INSERT {step.node_b.gate_id if step.node_b else '?'}"
            else:
                label = f"DELETE {step.node_a.gate_id if step.node_a else '?'}"
            lines.append(f"  {label}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_training_app.command("status")
def geometry_training_status(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
    format: str = typer.Option("full", "--format"),
    ai: bool = typer.Option(False, "--ai"),
) -> None:
    context = _context(ctx)
    if format not in {"full", "summary"}:
        raise typer.BadParameter("Format must be 'full' or 'summary'.")

    service = GeometryTrainingService()
    payload = service.training_status_payload(job_id, output_format=format, require_metrics=False)
    output = {
        "jobId": payload["jobId"],
        "step": payload["step"],
        "flatnessScore": payload["flatnessScore"],
        "flatnessAssessment": payload["flatnessAssessment"] if format == "full" else None,
        "gradientSNR": payload["gradientSNR"],
        "snrAssessment": payload["snrAssessment"] if format == "full" else None,
        "circuitBreakerSeverity": payload["circuitBreakerSeverity"],
        "circuitBreakerTripped": payload["circuitBreakerTripped"],
        "activeLayers": payload["activeLayers"],
        "perLayerGradientNorms": payload["perLayerGradientNorms"] if format == "full" else None,
        "nextActions": (
            [
                f"mc geometry training history --job {job_id}",
                f"mc geometry safety circuit-breaker --job {job_id}",
            ]
            if ai
            else None
        ),
    }

    if context.output_format == "text":
        lines = [
            "GEOMETRIC TRAINING STATUS",
            f"Job: {output['jobId']}",
            f"Step: {output['step']}",
        ]
        if output["flatnessScore"] is not None:
            assessment = output.get("flatnessAssessment") or ""
            lines.append(f"Flatness: {output['flatnessScore']:.3f} {f'({assessment})' if assessment else ''}".strip())
        if output["gradientSNR"] is not None:
            assessment = output.get("snrAssessment") or ""
            lines.append(
                f"Gradient SNR: {output['gradientSNR']:.2f} {f'({assessment})' if assessment else ''}".strip()
            )
        if output["circuitBreakerSeverity"] is not None:
            tripped = "TRIPPED" if output.get("circuitBreakerTripped") else "OK"
            lines.append(f"Circuit Breaker: {output['circuitBreakerSeverity']:.3f} ({tripped})")
        if output["activeLayers"]:
            lines.append(f"Active Layers: {', '.join(output['activeLayers'])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@geometry_training_app.command("history")
def geometry_training_history(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    context = _context(ctx)
    service = GeometryTrainingService()
    payload = service.training_history_payload(job_id)

    if context.output_format == "text":
        lines = ["GEOMETRIC TRAINING HISTORY", payload["interpretation"]]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_training_app.command("levels")
def geometry_training_levels(ctx: typer.Context) -> None:
    context = _context(ctx)
    levels = [
        {"name": level.value, "description": level.description, "metricsCollected": level.metrics_collected}
        for level in GeometricInstrumentationLevel
    ]
    payload = {"levels": levels}
    if context.output_format == "text":
        lines = ["GEOMETRIC INSTRUMENTATION LEVELS"]
        for level in levels:
            lines.append(f"\n{level['name']}: {level['description']}")
            lines.append(f"  Metrics: {', '.join(level['metricsCollected'])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    write_output(payload, context.output_format, context.pretty)


@geometry_safety_app.command("circuit-breaker")
def geometry_safety_circuit_breaker(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Option(None, "--job"),
    entropy: Optional[float] = typer.Option(None, "--entropy"),
    refusal_distance: Optional[float] = typer.Option(None, "--refusal-distance"),
    persona_drift: Optional[float] = typer.Option(None, "--persona-drift"),
    oscillation: bool = typer.Option(False, "--oscillation"),
) -> None:
    context = _context(ctx)
    service = GeometrySafetyService()
    state, _signals = service.evaluate_circuit_breaker(
        job_id=job_id,
        entropy_signal=entropy,
        refusal_distance=refusal_distance,
        persona_drift_magnitude=persona_drift,
        has_oscillation=oscillation,
    )

    output = {
        "tripped": state.is_tripped,
        "severity": state.severity,
        "state": "tripped" if state.is_tripped else ("warning" if state.severity >= 0.5 else "nominal"),
        "interpretation": state.interpretation,
        "recommendedAction": state.recommended_action.description,
    }

    if context.output_format == "text":
        lines = [
            "CIRCUIT BREAKER EVALUATION",
            f"State: {output['state'].upper()}",
            f"Severity: {output['severity']:.3f}",
            f"Interpretation: {output['interpretation']}",
            f"Recommended Action: {output['recommendedAction']}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@geometry_safety_app.command("persona")
def geometry_safety_persona(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    context = _context(ctx)
    service = GeometrySafetyService()
    drift_info = service.persona_drift(job_id)
    if drift_info is None:
        raise typer.BadParameter(f"Job '{job_id}' not found or has no persona drift metrics.")

    output = {
        "jobId": job_id,
        "overallDriftMagnitude": drift_info.overall_drift_magnitude,
        "driftAssessment": drift_info.assessment,
        "driftingTraits": drift_info.drifting_traits,
        "refusalDistance": drift_info.refusal_distance,
        "isApproachingRefusal": drift_info.is_approaching_refusal,
    }

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT ANALYSIS",
            f"Job: {output['jobId']}",
            f"Drift Magnitude: {output['overallDriftMagnitude']:.4f} ({output['driftAssessment']})",
        ]
        if output["driftingTraits"]:
            lines.append(f"Drifting Traits: {', '.join(output['driftingTraits'])}")
        if output["refusalDistance"] is not None:
            approaching = "YES" if output.get("isApproachingRefusal") else "NO"
            lines.append(f"Refusal Distance: {output['refusalDistance']:.4f} (Approaching: {approaching})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@geometry_safety_app.command("jailbreak-test")
def geometry_safety_jailbreak_test(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompts: Optional[str] = typer.Option(None, "--prompts", help="Path to prompts file (JSON array or newline-separated)"),
    prompt: Optional[list[str]] = typer.Option(None, "--prompt", help="Individual prompt(s) to test"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path to adapter to apply"),
) -> None:
    """Execute jailbreak entropy analysis to test model safety boundaries."""
    context = _context(ctx)
    
    # Collect prompts from file or individual --prompt flags
    prompt_list: list[str] = []
    if prompts:
        # Will be loaded from file by the service
        prompt_input: list[str] | str = prompts
    elif prompt:
        prompt_list = list(prompt)
        prompt_input = prompt_list
    else:
        raise typer.BadParameter("Provide either --prompts file or --prompt values")
    
    service = GeometrySafetyService()
    result = service.jailbreak_test(
        model_path=model,
        prompts=prompt_input,
        adapter_path=adapter,
    )
    
    output = {
        "modelPath": result.model_path,
        "adapterPath": result.adapter_path,
        "promptsTested": result.prompts_tested,
        "vulnerabilitiesFound": result.vulnerabilities_found,
        "overallAssessment": result.overall_assessment,
        "riskScore": result.risk_score,
        "processingTime": result.processing_time,
        "vulnerabilityDetails": [
            {
                "prompt": v.prompt[:100] + "..." if len(v.prompt) > 100 else v.prompt,
                "vulnerabilityType": v.vulnerability_type,
                "severity": v.severity,
                "baselineEntropy": v.baseline_entropy,
                "attackEntropy": v.attack_entropy,
                "deltaH": v.delta_h,
                "confidence": v.confidence,
                "attackVector": v.attack_vector,
                "mitigationHint": v.mitigation_hint,
            }
            for v in result.vulnerability_details
        ],
        "nextActions": [
            "mc geometry safety circuit-breaker for combined safety assessment",
            "mc thermo detect for detailed entropy analysis",
        ],
    }
    
    if context.output_format == "text":
        lines = [
            "JAILBREAK TEST RESULTS",
            f"Model: {result.model_path}",
        ]
        if result.adapter_path:
            lines.append(f"Adapter: {result.adapter_path}")
        lines.append(f"Prompts Tested: {result.prompts_tested}")
        lines.append(f"Vulnerabilities Found: {result.vulnerabilities_found}")
        lines.append(f"Overall Assessment: {result.overall_assessment.upper()}")
        lines.append(f"Risk Score: {result.risk_score:.2f}")
        lines.append(f"Processing Time: {result.processing_time:.2f}s")
        
        if result.vulnerability_details:
            lines.append("")
            lines.append("VULNERABILITY DETAILS:")
            for i, v in enumerate(result.vulnerability_details[:10], 1):  # Limit to 10 in text output
                lines.append(f"  {i}. [{v.severity.upper()}] {v.vulnerability_type} via {v.attack_vector}")
                lines.append(f"     Prompt: {v.prompt[:60]}...")
                lines.append(f"     Delta H: {v.delta_h:.3f}, Confidence: {v.confidence:.2f}")
                lines.append(f"     Hint: {v.mitigation_hint}")
        
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(output, context.output_format, context.pretty)


@geometry_safety_app.command("probe-redteam")
def geometry_safety_probe_redteam(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: Optional[str] = typer.Option(None, "--description", help="Adapter description"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: Optional[str] = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Scan adapter metadata for threat indicators (static analysis)."""
    context = _context(ctx)
    service = SafetyProbeService()

    indicators = service.scan_adapter_metadata(
        name=name,
        description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.threat_indicators_payload(indicators)
    payload["nextActions"] = [
        "mc geometry safety probe-behavioral for runtime safety checks",
        "mc geometry safety circuit-breaker for combined assessment",
    ]

    if context.output_format == "text":
        lines = [
            "RED TEAM STATIC ANALYSIS",
            f"Adapter: {name}",
            f"Status: {payload['status'].upper()}",
            f"Threat Indicators: {payload['count']}",
            f"Max Severity: {payload['maxSeverity']:.2f}",
        ]
        if indicators:
            lines.append("")
            lines.append("DETECTED THREATS:")
            for ind in indicators:
                lines.append(f"  [{ind.severity:.2f}] {ind.location}: {ind.description}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_safety_app.command("probe-behavioral")
def geometry_safety_probe_behavioral(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    tier: str = typer.Option("standard", "--tier", help="Safety tier: quick, standard, full"),
    description: Optional[str] = typer.Option(None, "--description", help="Adapter description"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: Optional[str] = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Run behavioral safety probes (requires inference hook for full analysis)."""
    import asyncio
    from modelcypher.core.domain.safety.behavioral_probes import AdapterSafetyTier

    context = _context(ctx)
    service = SafetyProbeService()

    tier_map = {
        "quick": AdapterSafetyTier.QUICK,
        "standard": AdapterSafetyTier.STANDARD,
        "full": AdapterSafetyTier.FULL,
    }
    safety_tier = tier_map.get(tier.lower(), AdapterSafetyTier.STANDARD)

    result = asyncio.run(service.run_behavioral_probes(
        adapter_name=name,
        tier=safety_tier,
        adapter_description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    ))

    payload = SafetyProbeService.composite_result_payload(result)
    payload["nextActions"] = [
        "mc geometry safety probe-redteam for static analysis",
        "mc geometry safety circuit-breaker for combined assessment",
    ]

    if context.output_format == "text":
        lines = [
            "BEHAVIORAL SAFETY PROBE RESULTS",
            f"Adapter: {name}",
            f"Tier: {tier.upper()}",
            f"Recommended Status: {payload['recommendedStatus'].upper()}",
            f"Aggregate Risk: {payload['aggregateRiskScore']:.2f}",
            f"Probes Run: {payload['probeCount']}",
        ]
        if payload["anyTriggered"]:
            lines.append("")
            lines.append("TRIGGERED PROBES:")
            for r in result.probe_results:
                if r.triggered:
                    lines.append(f"  [{r.risk_score:.2f}] {r.probe_name}: {r.details}")
        if payload["allFindings"]:
            lines.append("")
            lines.append("FINDINGS:")
            for finding in payload["allFindings"][:10]:
                lines.append(f"  - {finding}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_adapter_app.command("sparsity")
def geometry_adapter_sparsity(
    ctx: typer.Context,
    checkpoint_path: str = typer.Option(..., "--checkpoint"),
    base_path: Optional[str] = typer.Option(None, "--base"),
) -> None:
    context = _context(ctx)
    service = GeometryAdapterService()
    analysis = service.analyze_dare(checkpoint_path, base_path)

    interpretation = (
        f"Effective sparsity {analysis.effective_sparsity:.2%} "
        f"({analysis.quality_assessment.value}). Recommended drop rate "
        f"{analysis.recommended_drop_rate:.2f}."
    )
    output = {
        "checkpointPath": checkpoint_path,
        "baseModelPath": base_path,
        "effectiveSparsity": analysis.effective_sparsity,
        "qualityAssessment": analysis.quality_assessment.value,
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
        lines.append(f"Quality: {analysis.quality_assessment.value}")
        lines.append(f"Recommended Drop Rate: {analysis.recommended_drop_rate:.2f}")
        lines.append("")
        lines.append(interpretation)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@geometry_adapter_app.command("decomposition")
def geometry_adapter_decomposition(
    ctx: typer.Context,
    checkpoint_path: str = typer.Option(..., "--checkpoint"),
    base_path: Optional[str] = typer.Option(None, "--base"),
) -> None:
    context = _context(ctx)
    service = GeometryAdapterService()
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


@app.command("infer")
def infer(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    prompt: str = typer.Option(..., "--prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.95, "--top-p"),
) -> None:
    context = _context(ctx)
    engine = LocalInferenceEngine()
    result = engine.infer(model, prompt, max_tokens, temperature, top_p)
    write_output(result, context.output_format, context.pretty)


@geometry_primes_app.command("list")
def geometry_primes_list(ctx: typer.Context) -> None:
    """List all semantic prime anchors."""
    context = _context(ctx)
    service = GeometryPrimesService()
    primes = service.list_primes()
    
    payload = {
        "primes": [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "exponents": p.exponents,
            }
            for p in primes
        ],
        "count": len(primes),
    }
    
    if context.output_format == "text":
        lines = ["SEMANTIC PRIMES", f"Total: {len(primes)}", ""]
        for p in primes:
            lines.append(f"  {p.id}: {p.name} ({p.category})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@geometry_primes_app.command("probe")
def geometry_primes_probe(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Probe model for prime activation patterns."""
    context = _context(ctx)
    service = GeometryPrimesService()
    
    try:
        activations = service.probe(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1003",
            title="Prime probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    
    payload = {
        "modelPath": model_path,
        "activations": [
            {
                "primeId": a.prime_id,
                "activationStrength": a.activation_strength,
                "layerActivations": a.layer_activations,
            }
            for a in activations
        ],
        "count": len(activations),
    }
    
    if context.output_format == "text":
        lines = ["PRIME ACTIVATIONS", f"Model: {model_path}", ""]
        for a in activations[:20]:  # Limit output
            lines.append(f"  {a.prime_id}: {a.activation_strength:.3f}")
        if len(activations) > 20:
            lines.append(f"  ... and {len(activations) - 20} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@geometry_primes_app.command("compare")
def geometry_primes_compare(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Compare prime alignment between two models."""
    context = _context(ctx)
    service = GeometryPrimesService()
    
    try:
        result = service.compare(model_a, model_b)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1004",
            title="Prime comparison failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    
    payload = {
        "modelA": model_a,
        "modelB": model_b,
        "alignmentScore": result.alignment_score,
        "divergentPrimes": result.divergent_primes,
        "convergentPrimes": result.convergent_primes,
        "interpretation": result.interpretation,
    }
    
    if context.output_format == "text":
        lines = [
            "PRIME COMPARISON",
            f"Model A: {model_a}",
            f"Model B: {model_b}",
            f"Alignment Score: {result.alignment_score:.3f}",
            "",
            result.interpretation,
        ]
        if result.divergent_primes:
            lines.append(f"\nDivergent Primes: {', '.join(result.divergent_primes[:10])}")
        if result.convergent_primes:
            lines.append(f"Convergent Primes: {', '.join(result.convergent_primes[:10])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@geometry_crm_app.command("build")
def geometry_crm_build(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model", help="Path to model directory"),
    output_path: str = typer.Option(..., "--output-path", help="Output CRM JSON path"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Optional adapter directory"),
    include_primes: bool = typer.Option(
        True,
        "--include-primes/--no-include-primes",
        help="Include semantic prime anchors",
    ),
    include_gates: bool = typer.Option(
        True,
        "--include-gates/--no-include-gates",
        help="Include computational gate anchors",
    ),
    include_polyglot: bool = typer.Option(
        True,
        "--include-polyglot/--no-include-polyglot",
        help="Include multilingual prime variants",
    ),
    include_sequence_invariants: bool = typer.Option(
        True,
        "--include-sequence-invariants/--no-include-sequence-invariants",
        help="Include sequence invariant anchors (fibonacci, logic, causality, etc.)",
    ),
    sequence_families: Optional[str] = typer.Option(
        None,
        "--sequence-families",
        help="Comma-separated sequence families: fibonacci,lucas,tribonacci,primes,catalan,ramanujan,logic,ordering,arithmetic,causality",
    ),
    max_prompts_per_anchor: int = typer.Option(
        _CRM_DEFAULTS.max_prompts_per_anchor,
        "--max-prompts-per-anchor",
        help="Max prompts per anchor",
    ),
    max_polyglot_texts_per_language: int = typer.Option(
        _CRM_DEFAULTS.max_polyglot_texts_per_language,
        "--max-polyglot-texts-per-language",
        help="Max polyglot texts per language",
    ),
    anchor_prefixes: Optional[str] = typer.Option(
        None,
        "--anchor-prefixes",
        help="Comma-separated anchor prefixes (prime, gate)",
    ),
    max_anchors: Optional[int] = typer.Option(
        None,
        "--max-anchors",
        help="Limit number of anchors for quick runs",
    ),
) -> None:
    """Build a concept response matrix (CRM) for a model."""
    context = _context(ctx)
    service = ConceptResponseMatrixService(engine=LocalInferenceEngine())

    prefixes = None
    if anchor_prefixes:
        prefixes = [value.strip() for value in anchor_prefixes.split(",") if value.strip()]

    parsed_families: frozenset[SequenceFamily] | None = None
    if sequence_families:
        family_list = [val.strip().lower() for val in sequence_families.split(",") if val.strip()]
        family_set: set[SequenceFamily] = set()
        for name in family_list:
            try:
                family_set.add(SequenceFamily(name))
            except ValueError:
                pass  # Ignore invalid family names
        if family_set:
            parsed_families = frozenset(family_set)

    config = CRMBuildConfig(
        include_primes=include_primes,
        include_gates=include_gates,
        include_polyglot=include_polyglot,
        include_sequence_invariants=include_sequence_invariants,
        sequence_families=parsed_families,
        max_prompts_per_anchor=max_prompts_per_anchor,
        max_polyglot_texts_per_language=max_polyglot_texts_per_language,
        anchor_prefixes=prefixes,
        max_anchors=max_anchors,
    )

    try:
        summary = service.build(
            model_path=model_path,
            output_path=output_path,
            config=config,
            adapter=adapter,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1018",
            title="CRM build failed",
            detail=str(exc),
            hint="Ensure the model directory contains config.json and weights.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": summary.model_path,
        "outputPath": summary.output_path,
        "layerCount": summary.layer_count,
        "hiddenDim": summary.hidden_dim,
        "anchorCount": summary.anchor_count,
        "primeCount": summary.prime_count,
        "gateCount": summary.gate_count,
        "sequenceInvariantCount": summary.sequence_invariant_count,
    }

    if context.output_format == "text":
        lines = [
            "CONCEPT RESPONSE MATRIX",
            f"Model: {summary.model_path}",
            f"Output: {summary.output_path}",
            f"Layers: {summary.layer_count}",
            f"Hidden Dim: {summary.hidden_dim}",
            f"Anchors: {summary.anchor_count} (primes {summary.prime_count}, gates {summary.gate_count}, seq {summary.sequence_invariant_count})",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_crm_app.command("compare")
def geometry_crm_compare(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source CRM JSON path"),
    target: str = typer.Option(..., "--target", help="Target CRM JSON path"),
    include_matrix: bool = typer.Option(False, "--include-matrix", help="Include full CKA matrix"),
) -> None:
    """Compare two CRMs and compute layer correspondence via CKA."""
    context = _context(ctx)
    service = ConceptResponseMatrixService()

    try:
        summary = service.compare(source, target, include_matrix=include_matrix)
    except (ValueError, OSError) as exc:
        error = ErrorDetail(
            code="MC-1019",
            title="CRM comparison failed",
            detail=str(exc),
            hint="Ensure both CRM paths exist and are valid JSON exports.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "sourcePath": summary.source_path,
        "targetPath": summary.target_path,
        "commonAnchorCount": summary.common_anchor_count,
        "overallAlignment": summary.overall_alignment,
        "layerCorrespondence": summary.layer_correspondence,
    }
    if summary.cka_matrix is not None:
        payload["ckaMatrix"] = summary.cka_matrix

    if context.output_format == "text":
        lines = [
            "CRM COMPARISON",
            f"Source: {summary.source_path}",
            f"Target: {summary.target_path}",
            f"Common Anchors: {summary.common_anchor_count}",
            f"Overall Alignment: {summary.overall_alignment:.4f}",
        ]
        if summary.layer_correspondence:
            lines.append("")
            lines.append("Layer Correspondence (top 10):")
            for match in summary.layer_correspondence[:10]:
                lines.append(
                    f"  {match['sourceLayer']} -> {match['targetLayer']} (CKA {match['cka']:.4f})"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_crm_app.command("sequence-inventory")
def geometry_crm_sequence_inventory(
    ctx: typer.Context,
    family: Optional[str] = typer.Option(
        None,
        "--family",
        help="Filter by family: fibonacci, lucas, tribonacci, primes, catalan, ramanujan, logic, ordering, arithmetic, causality",
    ),
) -> None:
    """List available sequence invariant probes for CRM anchoring."""
    context = _context(ctx)

    family_filter: set[SequenceFamily] | None = None
    if family:
        try:
            family_filter = {SequenceFamily(family.strip().lower())}
        except ValueError:
            error = ErrorDetail(
                code="MC-1050",
                title="Invalid sequence family",
                detail=f"Unknown family '{family}'",
                hint="Valid families: fibonacci, lucas, tribonacci, primes, catalan, ramanujan, logic, ordering, arithmetic, causality",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    probes = SequenceInvariantInventory.probes_for_families(family_filter)
    counts = SequenceInvariantInventory.probe_count_by_family()

    probe_list = [
        {
            "id": probe.id,
            "family": probe.family.value,
            "domain": probe.domain.value,
            "name": probe.name,
            "description": probe.description,
            "weight": probe.cross_domain_weight,
        }
        for probe in probes
    ]

    payload = {
        "totalProbes": len(probes),
        "familyCounts": {fam.value: count for fam, count in counts.items()},
        "probes": probe_list,
    }

    if context.output_format == "text":
        lines = [
            "SEQUENCE INVARIANT INVENTORY",
            f"Total Probes: {len(probes)}",
            "",
            "Probes by Family:",
        ]
        for fam, count in sorted(counts.items(), key=lambda x: x[0].value):
            lines.append(f"  {fam.value}: {count}")
        lines.append("")
        lines.append("Probes (first 20):")
        for probe in probes[:20]:
            lines.append(f"  [{probe.family.value}] {probe.id}: {probe.name}")
        if len(probes) > 20:
            lines.append(f"  ... and {len(probes) - 20} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_stitch_app.command("analyze")
def geometry_stitch_analyze(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(..., "--checkpoint", help="Checkpoint paths (specify multiple times)"),
) -> None:
    """Analyze manifold stitching between checkpoints."""
    context = _context(ctx)
    service = GeometryStitchService()
    
    try:
        result = service.analyze(checkpoints)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1005",
            title="Stitch analysis failed",
            detail=str(exc),
            hint="Ensure checkpoint paths exist and contain valid model files",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    
    payload = {
        "checkpoints": checkpoints,
        "manifoldDistance": result.manifold_distance,
        "stitchingPoints": [
            {
                "layerName": sp.layer_name,
                "sourceDim": sp.source_dim,
                "targetDim": sp.target_dim,
                "qualityScore": sp.quality_score,
            }
            for sp in result.stitching_points
        ],
        "recommendedConfig": result.recommended_config,
        "interpretation": result.interpretation,
    }
    
    if context.output_format == "text":
        lines = [
            "STITCH ANALYSIS",
            f"Checkpoints: {len(checkpoints)}",
            f"Manifold Distance: {result.manifold_distance:.3f}",
            f"Stitching Points: {len(result.stitching_points)}",
            "",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@geometry_stitch_app.command("apply")
def geometry_stitch_apply(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source checkpoint path"),
    target: str = typer.Option(..., "--target", help="Target checkpoint path"),
    output: str = typer.Option(..., "--output", help="Output path for stitched model"),
    learning_rate: float = typer.Option(0.01, "--learning-rate"),
    max_iterations: int = typer.Option(500, "--max-iterations"),
) -> None:
    """Apply stitching operation between checkpoints."""
    context = _context(ctx)
    service = GeometryStitchService()
    
    config = {
        "learning_rate": learning_rate,
        "max_iterations": max_iterations,
        "use_procrustes_warm_start": True,
    }
    
    try:
        result = service.apply(source, target, output, config)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1006",
            title="Stitch apply failed",
            detail=str(exc),
            hint="Ensure source and target paths exist",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    
    payload = {
        "outputPath": result.output_path,
        "stitchedLayers": result.stitched_layers,
        "qualityScore": result.quality_score,
    }
    
    if context.output_format == "text":
        lines = [
            "STITCH APPLIED",
            f"Output: {result.output_path}",
            f"Stitched Layers: {result.stitched_layers}",
            f"Quality Score: {result.quality_score:.3f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)



@eval_app.command("run")
def eval_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset file"),
    batch_size: int = typer.Option(4, "--batch-size"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
) -> None:
    """Execute evaluation on model with dataset."""
    context = _context(ctx)
    from modelcypher.core.use_cases.evaluation_service import EvaluationService, EvalConfig
    
    service = EvaluationService()
    config = EvalConfig(batch_size=batch_size, max_samples=max_samples)
    
    result = service.run(model, dataset, config)
    
    payload = {
        "evalId": result.eval_id,
        "modelPath": result.model_path,
        "datasetPath": result.dataset_path,
        "averageLoss": result.average_loss,
        "perplexity": result.perplexity,
        "sampleCount": result.sample_count,
    }
    
    if context.output_format == "text":
        lines = [
            "EVALUATION COMPLETE",
            f"Eval ID: {result.eval_id}",
            f"Model: {result.model_path}",
            f"Dataset: {result.dataset_path}",
            f"Average Loss: {result.average_loss:.4f}",
            f"Perplexity: {result.perplexity:.4f}",
            f"Samples: {result.sample_count}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@compare_app.command("run")
def compare_run(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(..., "--checkpoint", help="Checkpoint paths"),
    prompt: str = typer.Option("Hello, how are you?", "--prompt"),
) -> None:
    """Execute A/B comparison between checkpoints."""
    context = _context(ctx)
    from modelcypher.core.use_cases.compare_service import CompareService, CompareConfig
    
    service = CompareService()
    config = CompareConfig(prompt=prompt)
    
    result = service.run(checkpoints, config)
    
    payload = {
        "comparisonId": result.comparison_id,
        "checkpoints": result.checkpoints,
        "prompt": result.prompt,
    }
    
    if context.output_format == "text":
        lines = [
            "COMPARISON STARTED",
            f"Comparison ID: {result.comparison_id}",
            f"Checkpoints: {len(result.checkpoints)}",
            f"Prompt: {result.prompt[:50]}...",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@compare_app.command("checkpoints")
def compare_checkpoints(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Compare checkpoints for a job."""
    context = _context(ctx)
    from modelcypher.core.use_cases.compare_service import CompareService
    
    service = CompareService()
    result = service.checkpoints(job_id)
    
    payload = {
        "jobId": result.job_id,
        "checkpoints": result.checkpoints,
        "comparisonMetrics": result.comparison_metrics,
    }
    
    write_output(payload, context.output_format, context.pretty)


@compare_app.command("baseline")
def compare_baseline(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """Establish baseline metrics for comparison."""
    context = _context(ctx)
    from modelcypher.core.use_cases.compare_service import CompareService
    
    service = CompareService()
    result = service.baseline(model)
    
    payload = {
        "model": result.model,
        "metrics": result.metrics,
        "timestamp": result.timestamp.isoformat(),
    }
    
    if context.output_format == "text":
        lines = [
            "BASELINE ESTABLISHED",
            f"Model: {result.model}",
            f"Perplexity: {result.metrics.get('perplexity', 'N/A')}",
            f"Latency: {result.metrics.get('latency_ms', 'N/A')}ms",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)


@compare_app.command("score")
def compare_score(
    ctx: typer.Context,
    comparison_id: str = typer.Argument(..., help="Comparison ID"),
) -> None:
    """Get aggregated comparison scores."""
    context = _context(ctx)
    from modelcypher.core.use_cases.compare_service import CompareService
    
    service = CompareService()
    result = service.score(comparison_id)
    
    payload = {
        "comparisonId": result.comparison_id,
        "scores": result.scores,
        "winner": result.winner,
    }
    
    if context.output_format == "text":
        lines = [
            "COMPARISON SCORES",
            f"Comparison ID: {result.comparison_id}",
            f"Quality: {result.scores.get('quality', 'N/A')}",
            f"Speed: {result.scores.get('speed', 'N/A')}",
            f"Overall: {result.scores.get('overall', 'N/A')}",
        ]
        if result.winner:
            lines.append(f"Winner: {result.winner}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    
    write_output(payload, context.output_format, context.pretty)



# Adapter commands
adapter_app = typer.Typer(no_args_is_help=True)
app.add_typer(adapter_app, name="adapter")


@adapter_app.command("inspect")
def adapter_inspect(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to adapter directory"),
) -> None:
    """Inspect adapter for detailed analysis."""
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
    """Project adapter to target space."""
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
    """Wrap adapter for MLX compatibility."""
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
    """Apply smoothing to adapter weights."""
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
    """Merge multiple LoRA adapters using TIES/DARE strategies."""
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


# Thermo commands (extracted to commands/thermo.py)
app.add_typer(thermo_commands.app, name="thermo")

# Calibration commands
calibration_app = typer.Typer(no_args_is_help=True)
app.add_typer(calibration_app, name="calibration")


@calibration_app.command("run")
def calibration_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to calibration dataset"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Max samples"),
    method: str = typer.Option("minmax", "--method", help="Calibration method"),
) -> None:
    """Execute calibration on a model with a dataset."""
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
    """Get status of a calibration operation."""
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
    """Apply calibration results to a model."""
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
    output_path: Optional[str] = typer.Option(None, "--output-path", help="Index output path"),
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
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model"),
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
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
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
    output_path: Optional[str] = typer.Option(None, "--output-path", help="Output path"),
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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path to adapter directory"),
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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path to adapter directory"),
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
    weights: Optional[list[float]] = typer.Option(None, "--weight", help="Weights for weighted strategy (must sum to 1.0)"),
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

