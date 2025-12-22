from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from typer.core import TyperGroup

from modelcypher.adapters.asif_packager import ASIFPackager
from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.typer_compat import apply_typer_compat
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
from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
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

apply_typer_compat()


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
dataset_app = typer.Typer(no_args_is_help=True)
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
geometry_sparse_app = typer.Typer(no_args_is_help=True)
geometry_refusal_app = typer.Typer(no_args_is_help=True)
geometry_persona_app = typer.Typer(no_args_is_help=True)
geometry_manifold_app = typer.Typer(no_args_is_help=True)
geometry_transport_app = typer.Typer(no_args_is_help=True)
geometry_refinement_app = typer.Typer(no_args_is_help=True)
geometry_invariant_app = typer.Typer(no_args_is_help=True)

app.add_typer(train_commands.train_app, name="train")
app.add_typer(train_commands.job_app, name="job")
app.add_typer(train_commands.checkpoint_app, name="checkpoint")
app.add_typer(model_commands.app, name="model")
app.add_typer(system_app, name="system")
app.add_typer(dataset_app, name="dataset")
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
geometry_app.add_typer(geometry_sparse_app, name="sparse")
geometry_app.add_typer(geometry_refusal_app, name="refusal")
geometry_app.add_typer(geometry_persona_app, name="persona")
geometry_app.add_typer(geometry_manifold_app, name="manifold")
geometry_app.add_typer(geometry_transport_app, name="transport")
geometry_app.add_typer(geometry_refinement_app, name="refinement")
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


@model_app.command("list")
def model_list(ctx: typer.Context) -> None:
    context = _context(ctx)
    service = ModelService()
    models = [model_payload(model) for model in service.list_models()]
    write_output(models, context.output_format, context.pretty)


@model_app.command("register")
def model_register(
    ctx: typer.Context,
    alias: str = typer.Argument(...),
    path: str = typer.Option(..., "--path"),
    architecture: str = typer.Option(..., "--architecture"),
    parameters: Optional[int] = typer.Option(None, "--parameters"),
    default_chat: bool = typer.Option(False, "--default-chat"),
) -> None:
    context = _context(ctx)
    service = ModelService()
    service.register_model(alias, path, architecture, parameters=parameters, default_chat=default_chat)
    write_output({"registered": alias}, context.output_format, context.pretty)


@model_app.command("merge")
def model_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source"),
    target: str = typer.Option(..., "--target"),
    output_dir: str = typer.Option(..., "--output-dir"),
    alpha: float = typer.Option(0.5, "--alpha"),
    rank: int = typer.Option(32, "--rank"),
    module_scope: Optional[str] = typer.Option(None, "--module-scope"),
    anchor_mode: str = typer.Option("semantic-primes", "--anchor-mode"),
    intersection: Optional[str] = typer.Option(None, "--intersection"),
    fisher_source: Optional[str] = typer.Option(None, "--fisher-source"),
    fisher_target: Optional[str] = typer.Option(None, "--fisher-target"),
    fisher_strength: float = typer.Option(0.0, "--fisher-strength"),
    fisher_epsilon: float = typer.Option(1e-6, "--fisher-epsilon"),
    adaptive_alpha: bool = typer.Option(False, "--adaptive-alpha"),
    source_crm: Optional[str] = typer.Option(None, "--source-crm"),
    target_crm: Optional[str] = typer.Option(None, "--target-crm"),
    transition_gate_strength: float = typer.Option(0.0, "--transition-gate-strength"),
    transition_gate_min_ratio: float = typer.Option(0.7, "--transition-gate-min-ratio"),
    transition_gate_max_ratio: float = typer.Option(1.3, "--transition-gate-max-ratio"),
    consistency_gate_strength: float = typer.Option(0.0, "--consistency-gate-strength"),
    consistency_gate_layer_samples: int = typer.Option(6, "--consistency-gate-layer-samples"),
    shared_subspace: bool = typer.Option(False, "--shared-subspace"),
    shared_subspace_method: str = typer.Option("cca", "--shared-subspace-method"),
    shared_subspace_blend: Optional[float] = typer.Option(None, "--shared-subspace-blend"),
    shared_subspace_per_layer: bool = typer.Option(
        True,
        "--shared-subspace-per-layer/--no-shared-subspace-per-layer",
    ),
    shared_subspace_anchor_prefixes: Optional[str] = typer.Option(
        None,
        "--shared-subspace-anchor-prefixes",
    ),
    shared_subspace_anchor_weights: Optional[str] = typer.Option(
        None,
        "--shared-subspace-anchor-weights",
    ),
    shared_subspace_pca_mode: Optional[str] = typer.Option(
        None,
        "--shared-subspace-pca-mode",
    ),
    shared_subspace_pca_variance: Optional[float] = typer.Option(
        None,
        "--shared-subspace-pca-variance",
    ),
    shared_subspace_variance_threshold: Optional[float] = typer.Option(
        None,
        "--shared-subspace-variance-threshold",
    ),
    shared_subspace_min_correlation: Optional[float] = typer.Option(
        None,
        "--shared-subspace-min-correlation",
    ),
    transport_guided: bool = typer.Option(False, "--use-transport-guided"),
    transport_coupling_threshold: float = typer.Option(0.001, "--transport-coupling-threshold"),
    transport_blend_alpha: float = typer.Option(0.5, "--transport-blend-alpha"),
    transport_min_samples: int = typer.Option(5, "--transport-min-samples"),
    transport_max_samples: int = typer.Option(32, "--transport-max-samples"),
    output_quant: Optional[str] = typer.Option(None, "--output-quant"),
    output_quant_group_size: Optional[int] = typer.Option(None, "--output-quant-group-size"),
    output_quant_mode: Optional[str] = typer.Option(None, "--output-quant-mode"),
    verbose: bool = typer.Option(False, "--verbose"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_path: Optional[str] = typer.Option(None, "--report-path"),
) -> None:
    context = _context(ctx)
    service = ModelMergeService(FileSystemStore())
    report = service.merge(
        source_id=source,
        target_id=target,
        output_dir=output_dir,
        alpha=alpha,
        alignment_rank=rank,
        module_scope=module_scope,
        anchor_mode=anchor_mode,
        intersection_path=intersection,
        fisher_source=fisher_source,
        fisher_target=fisher_target,
        fisher_strength=fisher_strength,
        fisher_epsilon=fisher_epsilon,
        adaptive_alpha=adaptive_alpha,
        source_crm=source_crm,
        target_crm=target_crm,
        transition_gate_strength=transition_gate_strength,
        transition_gate_min_ratio=transition_gate_min_ratio,
        transition_gate_max_ratio=transition_gate_max_ratio,
        consistency_gate_strength=consistency_gate_strength,
        consistency_gate_layer_samples=consistency_gate_layer_samples,
        shared_subspace=shared_subspace,
        shared_subspace_method=shared_subspace_method,
        shared_subspace_blend=shared_subspace_blend,
        shared_subspace_per_layer=shared_subspace_per_layer,
        shared_subspace_anchor_prefixes=shared_subspace_anchor_prefixes,
        shared_subspace_anchor_weights=shared_subspace_anchor_weights,
        shared_subspace_pca_mode=shared_subspace_pca_mode,
        shared_subspace_pca_variance=shared_subspace_pca_variance,
        shared_subspace_variance_threshold=shared_subspace_variance_threshold,
        shared_subspace_min_correlation=shared_subspace_min_correlation,
        transport_guided=transport_guided,
        transport_coupling_threshold=transport_coupling_threshold,
        transport_blend_alpha=transport_blend_alpha,
        transport_min_samples=transport_min_samples,
        transport_max_samples=transport_max_samples,
        dry_run=dry_run,
        output_quant=output_quant,
        output_quant_group_size=output_quant_group_size,
        output_quant_mode=output_quant_mode,
    )
    if report_path:
        from pathlib import Path

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_output(report, context.output_format, context.pretty)


@model_app.command("delete")
def model_delete(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = ModelService()
    service.delete_model(model_id)
    write_output({"deleted": model_id}, context.output_format, context.pretty)


@model_app.command("fetch")
def model_fetch(
    ctx: typer.Context,
    repo_id: str = typer.Argument(...),
    revision: str = typer.Option("main", "--revision"),
    auto_register: bool = typer.Option(False, "--auto-register"),
    alias: Optional[str] = typer.Option(None, "--alias"),
    architecture: Optional[str] = typer.Option(None, "--architecture"),
) -> None:
    context = _context(ctx)
    service = ModelService()
    result = service.fetch_model(repo_id, revision, auto_register, alias, architecture)
    write_output(result, context.output_format, context.pretty)


@model_app.command("search")
def model_search(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None),
    author: Optional[str] = typer.Option(None, "--author"),
    library: str = typer.Option("mlx", "--library"),
    quant: Optional[str] = typer.Option(None, "--quant"),
    sort: str = typer.Option("downloads", "--sort"),
    limit: int = typer.Option(20, "--limit"),
    cursor: Optional[str] = typer.Option(None, "--cursor"),
) -> None:
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

    service = ModelSearchService()
    try:
        page = service.search(filters, cursor)
    except ModelSearchError as exc:
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


@model_app.command("probe")
def model_probe(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Probe a model for architecture details."""
    context = _context(ctx)
    service = ModelProbeService()
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

    payload = {
        "architecture": result.architecture,
        "parameterCount": result.parameter_count,
        "vocabSize": result.vocab_size,
        "hiddenSize": result.hidden_size,
        "numAttentionHeads": result.num_attention_heads,
        "quantization": result.quantization,
        "layerCount": len(result.layers),
        "layers": [
            {
                "name": layer.name,
                "type": layer.type,
                "parameters": layer.parameters,
                "shape": layer.shape,
            }
            for layer in result.layers[:20]  # Limit to first 20 layers for readability
        ],
    }

    if context.output_format == "text":
        lines = [
            "MODEL PROBE",
            f"Architecture: {result.architecture}",
            f"Parameters: {result.parameter_count:,}",
            f"Vocab Size: {result.vocab_size:,}",
            f"Hidden Size: {result.hidden_size}",
            f"Attention Heads: {result.num_attention_heads}",
            f"Layers: {len(result.layers)}",
        ]
        if result.quantization:
            lines.append(f"Quantization: {result.quantization}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@model_app.command("validate-merge")
def model_validate_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Path to source model"),
    target: str = typer.Option(..., "--target", help="Path to target model"),
) -> None:
    """Validate merge compatibility between two models."""
    context = _context(ctx)
    service = ModelProbeService()
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

    payload = {
        "compatible": result.compatible,
        "architectureMatch": result.architecture_match,
        "vocabMatch": result.vocab_match,
        "dimensionMatch": result.dimension_match,
        "warnings": result.warnings,
    }

    if context.output_format == "text":
        status = "COMPATIBLE" if result.compatible else "INCOMPATIBLE"
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


@model_app.command("analyze-alignment")
def model_analyze_alignment(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Analyze alignment drift between two models."""
    context = _context(ctx)
    service = ModelProbeService()
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

    payload = {
        "driftMagnitude": result.drift_magnitude,
        "assessment": result.assessment,
        "interpretation": result.interpretation,
        "layerDrifts": [
            {
                "layerName": drift.layer_name,
                "driftMagnitude": drift.drift_magnitude,
                "direction": drift.direction,
            }
            for drift in result.layer_drifts[:20]  # Limit to first 20 layers
        ],
    }

    if context.output_format == "text":
        lines = [
            "ALIGNMENT ANALYSIS",
            f"Drift Magnitude: {result.drift_magnitude:.4f}",
            f"Assessment: {result.assessment}",
            f"Interpretation: {result.interpretation}",
        ]
        if result.layer_drifts:
            lines.append("")
            lines.append("Layer Drifts (top 10):")
            for drift in sorted(result.layer_drifts, key=lambda d: d.drift_magnitude, reverse=True)[:10]:
                lines.append(f"  {drift.layer_name}: {drift.drift_magnitude:.4f} ({drift.direction})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _parse_model_search_library(value: str) -> ModelSearchLibraryFilter:
    normalized = value.lower()
    if normalized == "mlx":
        return ModelSearchLibraryFilter.mlx
    if normalized == "safetensors":
        return ModelSearchLibraryFilter.safetensors
    if normalized == "pytorch":
        return ModelSearchLibraryFilter.pytorch
    if normalized == "any":
        return ModelSearchLibraryFilter.any
    raise typer.BadParameter("Invalid library filter. Use: mlx, safetensors, pytorch, or any.")


def _parse_model_search_quant(value: Optional[str]) -> ModelSearchQuantization | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "4bit":
        return ModelSearchQuantization.four_bit
    if normalized == "8bit":
        return ModelSearchQuantization.eight_bit
    if normalized == "any":
        return ModelSearchQuantization.any
    raise typer.BadParameter("Invalid quantization filter. Use: 4bit, 8bit, or any.")


def _parse_model_search_sort(value: str) -> ModelSearchSortOption:
    normalized = value.lower()
    if normalized == "downloads":
        return ModelSearchSortOption.downloads
    if normalized == "likes":
        return ModelSearchSortOption.likes
    if normalized in {"lastmodified", "last_modified"}:
        return ModelSearchSortOption.last_modified
    if normalized == "trending":
        return ModelSearchSortOption.trending
    raise typer.BadParameter("Invalid sort option. Use: downloads, likes, lastModified, or trending.")


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


@dataset_app.command("validate")
def dataset_validate(ctx: typer.Context, path: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = DatasetService()
    write_output(service.validate_dataset(path), context.output_format, context.pretty)


@dataset_app.command("preprocess")
def dataset_preprocess(
    ctx: typer.Context,
    input_path: str = typer.Argument(...),
    output_path: str = typer.Option(..., "--output-path", "-o", "--dataset-output", "--processed-output"),
    tokenizer: str = typer.Option(..., "--tokenizer"),
) -> None:
    context = _context(ctx)
    service = DatasetService()
    result = service.preprocess_dataset(input_path, output_path, tokenizer)
    write_output(result, context.output_format, context.pretty)


@dataset_app.command("convert")
def dataset_convert(
    ctx: typer.Context,
    input_path: str = typer.Argument(...),
    to_format: str = typer.Option(..., "--to"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    target_format = parse_format(to_format)
    result = service.convert_dataset(input_path, target_format, output_path)
    payload = dataset_convert_payload(result)
    if context.output_format == "text":
        lines: list[str] = [
            "DATASET CONVERT",
            f"Source: {payload['sourcePath']}",
            f"Output: {payload['outputPath']}",
            f"Target format: {payload['targetFormat']}",
            f"Lines written: {payload['lineCount']}",
        ]
        if payload["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in payload["warnings"])
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    write_output(payload, context.output_format, context.pretty)


@dataset_app.command("preview")
def dataset_preview(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    lines: int = typer.Option(5, "--lines"),
    format: str = typer.Option("json", "--format"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    requested = max(1, lines)
    limit = min(requested, MAX_PREVIEW_LINES)
    warnings: list[str] = []
    if requested > limit:
        warnings.append(f"Preview capped at {limit} lines (requested {requested}).")
    preview = service.preview(path, limit)

    if context.output_format == "text":
        if warnings:
            sys.stdout.write(f"Warning: {warnings[0]}\n")
        mode = format.lower()
        if mode == "table":
            rows = [
                f"{row.line_number}\t[{row.format.value}]\t{preview_line(row.raw)}"
                for row in preview.rows
            ]
            write_output("\n".join(rows), context.output_format, context.pretty)
            return
        rows = []
        for row in preview.rows:
            message = "none" if not row.validation_messages else "; ".join(row.validation_messages)
            rows.append(
                "\n".join(
                    [
                        f"Line {row.line_number} [{row.format.value}]",
                        pretty_fields(row.fields),
                        f"Validation: {message}",
                    ]
                )
            )
        write_output("\n\n".join(rows), context.output_format, context.pretty)
        return

    write_output(dataset_preview_payload(preview, warnings=warnings), context.output_format, context.pretty)


@dataset_app.command("get-row")
def dataset_get_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    row = service.get_row(path, line)

    if context.output_format == "text":
        lines: list[str] = []
        lines.append(f"Line {row.line_number} [{row.format.value}]")
        lines.append(pretty_fields(row.fields))
        if row.raw_truncated:
            lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
        if row.fields_truncated:
            joined = ", ".join(row.fields_truncated)
            lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
        if row.validation_messages:
            lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_row_payload(row), context.output_format, context.pretty)


@dataset_app.command("update-row")
def dataset_update_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
    content: str = typer.Option(..., "--content"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    fields = parse_fields(content, "--content")
    result = service.update_row(path, line, fields)

    if context.output_format == "text":
        lines: list[str] = [f"Updated line {line}"]
        if result.row:
            row = result.row
            lines.append(pretty_fields(row.fields))
            if row.raw_truncated:
                lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
            if row.fields_truncated:
                joined = ", ".join(row.fields_truncated)
                lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
            if row.validation_messages:
                lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@dataset_app.command("add-row")
def dataset_add_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    format: str = typer.Option(..., "--format"),
    fields: str = typer.Option(..., "--fields"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    format_enum = parse_format(format)
    parsed_fields = parse_fields(fields, "--fields")
    result = service.add_row(path, format_enum, parsed_fields)

    if context.output_format == "text":
        line_label = result.line_number or 0
        lines: list[str] = [f"Added line {line_label} [{format_enum.value}]"]
        if result.row:
            row = result.row
            lines.append(pretty_fields(row.fields))
            if row.raw_truncated:
                lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
            if row.fields_truncated:
                joined = ", ".join(row.fields_truncated)
                lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
            if row.validation_messages:
                lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@dataset_app.command("delete-row")
def dataset_delete_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    result = service.delete_row(path, line)

    if context.output_format == "text":
        lines: list[str] = [f"Deleted line {line}"]
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@dataset_app.command("list")
def dataset_list(ctx: typer.Context) -> None:
    context = _context(ctx)
    service = DatasetService()
    datasets = [dataset_payload(dataset) for dataset in service.list_datasets()]
    write_output(datasets, context.output_format, context.pretty)


@dataset_app.command("delete")
def dataset_delete(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
) -> None:
    context = _context(ctx)
    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete dataset {path}?"):
            raise typer.Exit(code=1)
    service = DatasetService()
    service.delete_dataset(path)
    write_output({"deleted": path}, context.output_format, context.pretty)


@dataset_app.command("pack-asif")
def dataset_pack_asif(
    ctx: typer.Context,
    source: str = typer.Argument(...),
    destination: str = typer.Option(..., "--destination"),
    headroom_percent: int = typer.Option(15, "--headroom-percent"),
    minimum_free_gib: int = typer.Option(2, "--minimum-free-gib"),
    filesystem: str = typer.Option("apfs", "--filesystem"),
    volume_name: str = typer.Option("DATASET", "--volume-name"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    context = _context(ctx)
    packager = ASIFPackager()
    result = packager.pack(
        source=source,
        destination=destination,
        headroom_percent=headroom_percent,
        minimum_free_gib=minimum_free_gib,
        filesystem=filesystem,
        volume_name=volume_name,
        overwrite=overwrite,
    )
    write_output(result, context.output_format, context.pretty)


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


# =============================================================================
# Geometry Sparse Region Commands
# =============================================================================

@geometry_sparse_app.command("domains")
def geometry_sparse_domains(ctx: typer.Context) -> None:
    """List all built-in sparse region domains."""
    context = _context(ctx)
    service = GeometrySparseService()
    domains = service.list_domains()
    payload = service.domains_payload(domains)

    if context.output_format == "text":
        lines = [
            "SPARSE REGION DOMAINS",
            f"Total: {payload['count']}",
            "",
        ]
        for d in payload["domains"]:
            range_str = ""
            if d["expectedLayerRange"]:
                range_str = f" (layers {d['expectedLayerRange'][0]:.0%}-{d['expectedLayerRange'][1]:.0%})"
            lines.append(f"  {d['name']}: {d['description']}{range_str}")
            lines.append(f"    Category: {d['category']}, Probes: {d['probeCount']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_sparse_app.command("locate")
def geometry_sparse_locate(
    ctx: typer.Context,
    domain_stats_file: str = typer.Argument(..., help="Path to domain layer stats JSON"),
    baseline_stats_file: str = typer.Argument(..., help="Path to baseline layer stats JSON"),
    domain_name: str = typer.Option("unknown", "--domain", help="Domain name"),
    base_rank: int = typer.Option(16, "--rank", help="Base LoRA rank"),
    sparsity_threshold: float = typer.Option(0.3, "--threshold", help="Sparsity threshold"),
) -> None:
    """
    Locate sparse regions for LoRA injection.

    Input files should contain JSON arrays of layer stats:
    [{"layer_index": 0, "mean_activation": 0.5, ...}, ...]
    """
    context = _context(ctx)
    service = GeometrySparseService()

    domain_stats = json.loads(Path(domain_stats_file).read_text())
    baseline_stats = json.loads(Path(baseline_stats_file).read_text())

    result = service.locate_sparse_regions(
        domain_stats=domain_stats,
        baseline_stats=baseline_stats,
        domain_name=domain_name,
        base_rank=base_rank,
        sparsity_threshold=sparsity_threshold,
    )

    payload = service.analysis_payload(result)
    payload["nextActions"] = [
        "mc geometry sparse domains to see available domain definitions",
        "mc geometry adapter sparsity for DARE analysis",
    ]

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Domain: {result.domain}",
            f"Sparse Layers: {len(result.sparse_layers)} {result.sparse_layers}",
            f"Skip Layers: {len(result.skip_layers)} {result.skip_layers}",
            "",
            "LORA RECOMMENDATION",
            f"  Quality: {result.recommendation.quality.value.upper()}",
            f"  Overall Rank: {result.recommendation.overall_rank}",
            f"  Alpha: {result.recommendation.alpha}",
            f"  Preservation: {result.recommendation.estimated_preservation:.0%}",
            "",
            result.recommendation.rationale,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# Geometry Refusal Direction Commands
# =============================================================================

@geometry_refusal_app.command("pairs")
def geometry_refusal_pairs(ctx: typer.Context) -> None:
    """List standard contrastive prompt pairs for refusal direction."""
    context = _context(ctx)
    service = GeometrySparseService()
    pairs = service.get_contrastive_pairs()
    payload = service.contrastive_pairs_payload(pairs)

    if context.output_format == "text":
        lines = [
            "CONTRASTIVE PROMPT PAIRS",
            f"Total: {payload['count']}",
            "",
        ]
        for i, p in enumerate(payload["pairs"], 1):
            lines.append(f"{i}. Harmful: {p['harmful'][:60]}...")
            lines.append(f"   Harmless: {p['harmless'][:60]}...")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_refusal_app.command("detect")
def geometry_refusal_detect(
    ctx: typer.Context,
    harmful_file: str = typer.Argument(..., help="Path to harmful activations JSON"),
    harmless_file: str = typer.Argument(..., help="Path to harmless activations JSON"),
    layer_index: int = typer.Option(..., "--layer", help="Layer index"),
    model_id: str = typer.Option("unknown", "--model-id", help="Model identifier"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize direction"),
) -> None:
    """
    Detect refusal direction from contrastive activations.

    Input files should contain JSON arrays of activation vectors:
    [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    """
    context = _context(ctx)
    service = GeometrySparseService()

    harmful = json.loads(Path(harmful_file).read_text())
    harmless = json.loads(Path(harmless_file).read_text())

    direction = service.detect_refusal_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
        layer_index=layer_index,
        model_id=model_id,
        normalize=normalize,
    )

    if direction is None:
        write_error(
            ErrorDetail(
                code="MC-4010",
                message="Failed to compute refusal direction",
                detail="Insufficient data or activation difference below threshold",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    payload = service.refusal_direction_payload(direction)
    payload["nextActions"] = [
        "mc geometry refusal pairs to see contrastive prompts",
        "mc geometry safety circuit-breaker for safety assessment",
    ]

    if context.output_format == "text":
        lines = [
            "REFUSAL DIRECTION",
            f"Model: {direction.model_id}",
            f"Layer: {direction.layer_index}",
            f"Hidden Size: {direction.hidden_size}",
            f"Strength: {direction.strength:.4f}",
            f"Explained Variance: {direction.explained_variance:.2%}",
            f"Computed At: {direction.computed_at.isoformat()}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY PERSONA COMMANDS
# ──────────────────────────────────────────────────────────────────────────────


@geometry_persona_app.command("traits")
def geometry_persona_traits(ctx: typer.Context):
    """List all standard persona traits for vector extraction."""
    context = _context(ctx)
    service = GeometryPersonaService()

    traits = service.list_traits()
    payload = service.traits_payload(traits)
    payload["nextActions"] = [
        "mc geometry persona extract to extract a persona vector",
        "mc geometry persona drift to measure drift during training",
    ]

    if context.output_format == "text":
        lines = ["PERSONA TRAITS", ""]
        for trait in traits:
            lines.append(f"  {trait.id}: {trait.name}")
            lines.append(f"    {trait.description}")
            lines.append(f"    Prompts: +{trait.positive_prompt_count} / -{trait.negative_prompt_count}")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_persona_app.command("extract")
def geometry_persona_extract(
    ctx: typer.Context,
    positive_file: Path = typer.Option(..., "--positive", "-p", help="JSON file with positive activations"),
    negative_file: Path = typer.Option(..., "--negative", "-n", help="JSON file with negative activations"),
    trait_id: str = typer.Option(..., "--trait", "-t", help="Trait ID (helpful, harmless, honest)"),
    layer_index: int = typer.Option(..., "--layer", "-l", help="Layer index"),
    model_id: str = typer.Option("unknown", "--model", "-m", help="Model identifier"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize direction vector"),
):
    """
    Extract a persona vector from contrastive activations.

    Positive activations from trait-positive prompts, negative from trait-negative.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    positive = json.loads(Path(positive_file).read_text())
    negative = json.loads(Path(negative_file).read_text())

    vector = service.extract_persona_vector(
        positive_activations=positive,
        negative_activations=negative,
        trait_id=trait_id,
        layer_index=layer_index,
        model_id=model_id,
        normalize=normalize,
    )

    if vector is None:
        write_error(
            ErrorDetail(
                code="MC-4011",
                message="Failed to extract persona vector",
                detail="Insufficient data or correlation below threshold",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    payload = service.persona_vector_payload(vector)
    payload["nextActions"] = [
        "mc geometry persona drift to measure training drift",
        "mc safety persona-drift for safety monitoring",
    ]

    if context.output_format == "text":
        lines = [
            "PERSONA VECTOR",
            f"Trait: {vector.name} ({vector.id})",
            f"Model: {vector.model_id}",
            f"Layer: {vector.layer_index}",
            f"Hidden Size: {vector.hidden_size}",
            f"Strength: {vector.strength:.4f}",
            f"Correlation: {vector.correlation_coefficient:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_persona_app.command("drift")
def geometry_persona_drift(
    ctx: typer.Context,
    positions_file: Path = typer.Option(..., "--positions", "-p", help="JSON file with position measurements"),
    step: int = typer.Option(..., "--step", "-s", help="Training step number"),
    threshold: float = typer.Option(0.2, "--threshold", "-t", help="Drift threshold"),
):
    """
    Compute drift metrics from position measurements during training.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    positions = json.loads(Path(positions_file).read_text())
    metrics = service.compute_drift(
        positions=positions,
        step=step,
        drift_threshold=threshold,
    )

    payload = service.drift_metrics_payload(metrics)
    payload["nextActions"] = [
        "mc safety circuit-breaker if drift is significant",
        "mc train pause to halt training if needed",
    ]

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT METRICS",
            f"Step: {metrics.step}",
            f"Overall Drift: {metrics.overall_drift_magnitude:.4f}",
            f"Significant Drift: {'Yes' if metrics.has_significant_drift else 'No'}",
            "",
            metrics.interpretation,
        ]
        if metrics.drifting_traits:
            lines.append(f"Drifting Traits: {', '.join(metrics.drifting_traits)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY MANIFOLD COMMANDS
# ──────────────────────────────────────────────────────────────────────────────


@geometry_manifold_app.command("cluster")
def geometry_manifold_cluster(
    ctx: typer.Context,
    points_file: Path = typer.Option(..., "--points", "-p", help="JSON file with manifold points"),
    epsilon: float = typer.Option(0.3, "--epsilon", "-e", help="DBSCAN epsilon (distance threshold)"),
    min_points: int = typer.Option(5, "--min-points", "-m", help="Minimum points per cluster"),
    compute_dimension: bool = typer.Option(True, "--dimension/--no-dimension", help="Compute intrinsic dimension"),
):
    """
    Cluster manifold points into regions using DBSCAN.

    Points should have entropy and gate features from thermo measurements.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.cluster_points(
        points=points,
        epsilon=epsilon,
        min_points=min_points,
        compute_dimension=compute_dimension,
    )

    payload = service.clustering_payload(result)
    payload["nextActions"] = [
        "mc geometry manifold dimension to estimate dimensionality",
        "mc geometry manifold query to classify new points",
    ]

    if context.output_format == "text":
        lines = [
            "MANIFOLD CLUSTERING",
            f"Regions: {len(result.regions)}",
            f"Noise Points: {len(result.noise_points)}",
            f"New Clusters: {result.new_clusters_formed}",
            "",
        ]
        for region in result.regions:
            lines.append(f"  Region {str(region.id)[:8]}:")
            lines.append(f"    Type: {region.region_type.value}")
            lines.append(f"    Members: {region.member_count}")
            if region.intrinsic_dimension is not None:
                lines.append(f"    Dimension: {region.intrinsic_dimension:.2f}")
            lines.append(f"    Dominant Gates: {', '.join(region.dominant_gates)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_manifold_app.command("dimension")
def geometry_manifold_dimension(
    ctx: typer.Context,
    points_file: Path = typer.Option(..., "--points", "-p", help="JSON file with point vectors"),
    bootstrap: int = typer.Option(0, "--bootstrap", "-b", help="Bootstrap samples (0 = none)"),
    regression: bool = typer.Option(True, "--regression/--no-regression", help="Use regression-based estimation"),
):
    """
    Estimate intrinsic dimension of a point cloud using TwoNN.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.estimate_dimension(
        points=points,
        bootstrap_samples=bootstrap,
        use_regression=regression,
    )

    payload = service.dimension_payload(result)
    payload["nextActions"] = [
        "mc geometry intrinsic-dimension for alternative estimation",
        "mc geometry manifold cluster to find regions",
    ]

    if context.output_format == "text":
        lines = [
            "INTRINSIC DIMENSION ESTIMATE",
            f"Dimension: {result.intrinsic_dimension:.2f}",
        ]
        if result.ci95_lower is not None and result.ci95_upper is not None:
            lines.append(f"95% CI: [{result.ci95_lower:.2f}, {result.ci95_upper:.2f}]")
        lines.append(f"Samples: {result.sample_count} ({result.usable_count} usable)")
        lines.append(f"Method: {'Regression' if result.uses_regression else 'MLE'}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_manifold_app.command("query")
def geometry_manifold_query(
    ctx: typer.Context,
    point_file: Path = typer.Option(..., "--point", "-p", help="JSON file with point to query"),
    regions_file: Path = typer.Option(..., "--regions", "-r", help="JSON file with regions"),
    epsilon: float = typer.Option(0.3, "--epsilon", "-e", help="Distance threshold"),
):
    """
    Query which region a point belongs to.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    point = json.loads(Path(point_file).read_text())
    regions = json.loads(Path(regions_file).read_text())

    result = service.query_region(
        point=point,
        regions=regions,
        epsilon=epsilon,
    )

    payload = service.region_query_payload(result)
    payload["nextActions"] = [
        "mc geometry manifold cluster to update clusters",
        "mc thermo measure to get point features",
    ]

    if context.output_format == "text":
        lines = [
            "REGION QUERY RESULT",
            f"Suggested Type: {result.suggested_type.value}",
            f"Within Region: {'Yes' if result.is_within_region else 'No'}",
            f"Distance: {result.distance:.4f}",
            f"Confidence: {result.confidence:.2%}",
        ]
        if result.nearest_region:
            lines.append(f"Nearest Region: {str(result.nearest_region.id)[:8]} ({result.nearest_region.region_type.value})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY TRANSPORT COMMANDS
# ──────────────────────────────────────────────────────────────────────────────


@geometry_transport_app.command("merge")
def geometry_transport_merge(
    ctx: typer.Context,
    source_file: Path = typer.Option(..., "--source", "-s", help="JSON file with source weights [N x D]"),
    target_file: Path = typer.Option(..., "--target", "-t", help="JSON file with target weights [M x D]"),
    plan_file: Path = typer.Option(..., "--plan", "-p", help="JSON file with transport plan [N x M]"),
    coupling_threshold: float = typer.Option(0.001, "--threshold", help="Minimum coupling to consider"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize transport plan rows"),
    blend_alpha: float = typer.Option(0.5, "--alpha", "-a", help="Blend factor with target (0 = transport-only)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for merged weights"),
):
    """
    Merge weights using a transport plan.

    Uses the transport plan π[i,j] to guide weighted averaging:
    W_merged[j,:] = Σ_i π[i,j] * W_source[i,:]
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source = json.loads(Path(source_file).read_text())
    target = json.loads(Path(target_file).read_text())
    plan = json.loads(Path(plan_file).read_text())

    merged = service.synthesize_weights(
        source_weights=source,
        target_weights=target,
        transport_plan=plan,
        coupling_threshold=coupling_threshold,
        normalize_rows=normalize,
        blend_alpha=blend_alpha,
    )

    if merged is None:
        write_error(
            ErrorDetail(
                code="MC-4012",
                message="Failed to merge weights",
                detail="Invalid dimensions or empty input",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(merged))

    payload = {
        "mergedShape": [len(merged), len(merged[0]) if merged else 0],
        "blendAlpha": blend_alpha,
        "couplingThreshold": coupling_threshold,
        "outputFile": str(output) if output else None,
        "nextActions": [
            "mc geometry gromov-wasserstein to compute transport plan",
            "mc model merge for full model merging",
        ],
    }

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED MERGE",
            f"Source: {len(source)} x {len(source[0]) if source else 0}",
            f"Target: {len(target)} x {len(target[0]) if target else 0}",
            f"Merged: {len(merged)} x {len(merged[0]) if merged else 0}",
            f"Blend Alpha: {blend_alpha}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@geometry_transport_app.command("synthesize")
def geometry_transport_synthesize(
    ctx: typer.Context,
    source_act_file: Path = typer.Option(..., "--source-act", help="JSON file with source activations"),
    target_act_file: Path = typer.Option(..., "--target-act", help="JSON file with target activations"),
    source_weights_file: Path = typer.Option(..., "--source-weights", help="JSON file with source weights"),
    target_weights_file: Path = typer.Option(..., "--target-weights", help="JSON file with target weights"),
    coupling_threshold: float = typer.Option(0.001, "--threshold", help="Minimum coupling to consider"),
    blend_alpha: float = typer.Option(0.5, "--alpha", "-a", help="Blend factor with target"),
    gw_epsilon: float = typer.Option(0.05, "--epsilon", "-e", help="GW entropic regularization"),
    gw_iterations: int = typer.Option(50, "--iterations", "-i", help="Max GW iterations"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for merged weights"),
):
    """
    Compute GW transport plan and synthesize merged weights.

    Computes pairwise distances from activations, solves for optimal
    transport using Gromov-Wasserstein, then applies transport-guided merging.
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source_act = json.loads(Path(source_act_file).read_text())
    target_act = json.loads(Path(target_act_file).read_text())
    source_weights = json.loads(Path(source_weights_file).read_text())
    target_weights = json.loads(Path(target_weights_file).read_text())

    config = MergeConfig(
        coupling_threshold=coupling_threshold,
        blend_alpha=blend_alpha,
        gw_epsilon=gw_epsilon,
        gw_max_iterations=gw_iterations,
    )

    result = service.synthesize_with_gw(
        source_activations=source_act,
        target_activations=target_act,
        source_weights=source_weights,
        target_weights=target_weights,
        config=config,
    )

    if result is None:
        write_error(
            ErrorDetail(
                code="MC-4013",
                message="Failed to synthesize with GW",
                detail="Insufficient samples or dimension mismatch",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(result.merged_weights))

    payload = service.merge_result_payload(result)
    payload["outputFile"] = str(output) if output else None
    payload["nextActions"] = [
        "mc geometry intrinsic-dimension to analyze merged space",
        "mc model merge for full model merging",
    ]

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED SYNTHESIS",
            f"GW Distance: {result.gw_distance:.4f}",
            f"Marginal Error: {result.marginal_error:.4f}",
            f"Effective Rank: {result.effective_rank}",
            f"Converged: {'Yes' if result.converged else 'No'} ({result.iterations} iterations)",
            f"Merged Shape: {len(result.merged_weights)} x {len(result.merged_weights[0]) if result.merged_weights else 0}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# GEOMETRY REFINEMENT COMMANDS
# =============================================================================

@geometry_refinement_app.command("analyze")
def geometry_refinement_analyze(
    ctx: typer.Context,
    base_model: str = typer.Argument(..., help="Path to base (target) model"),
    adapted_model: str = typer.Argument(..., help="Path to adapted (source/refined) model"),
    source_crm: Optional[str] = typer.Option(None, "--source-crm", help="Path to source CRM file"),
    target_crm: Optional[str] = typer.Option(None, "--target-crm", help="Path to target CRM file"),
    sparsity_weight: float = typer.Option(0.35, "--sparsity-weight", help="Weight for DARE sparsity contribution"),
    directional_weight: float = typer.Option(0.35, "--directional-weight", help="Weight for DoRA directional drift"),
    transition_weight: float = typer.Option(0.30, "--transition-weight", help="Weight for transition CKA"),
    hard_swap_threshold: float = typer.Option(0.80, "--hard-swap-threshold", help="Score threshold for hard swap"),
    mode: str = typer.Option("default", "--mode", help="Analysis mode: default, aggressive, conservative"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON result to file"),
) -> None:
    """Analyze refinement density between base and adapted models.

    Combines DARE sparsity, DoRA directional drift, and transition CKA
    to produce per-layer refinement scores and merge recommendations.

    Example:
        mc geometry refinement analyze ./base-model ./finetuned-model
        mc geometry refinement analyze ./base ./adapted --mode aggressive
    """
    context = _context(ctx)

    try:
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
            Configuration as DAREConfig,
        )
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )
        from modelcypher.core.domain.geometry.concept_response_matrix import (
            ConceptResponseMatrix,
        )
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        # Load models
        _, base_weights = mlx_load(base_model, lazy=True)
        _, adapted_weights = mlx_load(adapted_model, lazy=True)

        base_weights = dict(base_weights)
        adapted_weights = dict(adapted_weights)

        # Compute delta weights for DARE
        delta_weights = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base = base_weights[name]
            adapted = adapted_weights[name]
            if base.shape != adapted.shape:
                continue
            delta = adapted - base
            mx.eval(delta)
            delta_weights[name] = delta.flatten().tolist()

        sparsity_analysis = DARESparsityAnalyzer.analyze(
            delta_weights, DAREConfig(compute_per_layer_metrics=True)
        )

        # Compute DoRA decomposition
        base_mx = {}
        adapted_mx = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base_mx[name] = base_weights[name]
            adapted_mx[name] = adapted_weights[name]

        dora = DoRADecomposition()
        dora_result = dora.analyze_adapter(base_mx, adapted_mx)

        # Load CRMs if provided
        transition_experiment = None
        if source_crm and target_crm:
            source = ConceptResponseMatrix.load(source_crm)
            target = ConceptResponseMatrix.load(target_crm)
            transition_experiment = source.compute_transition_alignment(target)

        # Configure analyzer
        if mode == "aggressive":
            config = RefinementDensityConfig.aggressive()
        elif mode == "conservative":
            config = RefinementDensityConfig.conservative()
        else:
            config = RefinementDensityConfig(
                sparsity_weight=sparsity_weight,
                directional_weight=directional_weight,
                transition_weight=transition_weight,
                hard_swap_threshold=hard_swap_threshold,
            )

        analyzer = RefinementDensityAnalyzer(config)
        result = analyzer.analyze(
            source_model=adapted_model,
            target_model=base_model,
            sparsity_analysis=sparsity_analysis,
            dora_result=dora_result,
            transition_experiment=transition_experiment,
        )

        # Write to file if requested
        if output_file:
            Path(output_file).write_text(dump_json(result.to_dict()))

        payload = result.to_dict()
        payload["outputFile"] = output_file
        payload["nextActions"] = [
            "mc model merge with --use-refinement-density to apply recommendations",
            "mc geometry adapter sparsity for detailed DARE analysis",
            "mc geometry adapter decomposition for detailed DoRA analysis",
        ]

        if context.output_format == "text":
            lines = [
                "REFINEMENT DENSITY ANALYSIS",
                f"Source (refined): {adapted_model}",
                f"Target (base): {base_model}",
                "",
                f"Mean Composite Score: {result.mean_composite_score:.3f}",
                f"Max Composite Score: {result.max_composite_score:.3f}",
                "",
                f"Layers Above Hard Swap Threshold: {result.layers_above_hard_swap}",
                f"Layers Above High Alpha Threshold: {result.layers_above_high_alpha}",
                f"Layers Above Medium Alpha Threshold: {result.layers_above_medium_alpha}",
                "",
            ]

            if result.hard_swap_layers:
                lines.append(f"Recommended Hard Swap Layers: {result.hard_swap_layers}")

            components = []
            if result.has_sparsity_data:
                components.append("DARE")
            if result.has_directional_data:
                components.append("DoRA")
            if result.has_transition_data:
                components.append("Transition-CKA")
            lines.append(f"Data Sources: {', '.join(components) or 'None'}")

            if output_file:
                lines.append(f"\nOutput: {output_file}")

            # Show top layers by refinement
            sorted_scores = sorted(
                result.layer_scores.items(),
                key=lambda x: x[1].composite_score,
                reverse=True,
            )
            if sorted_scores:
                lines.append("\nTop Refined Layers:")
                for idx, score in sorted_scores[:5]:
                    lines.append(
                        f"  Layer {idx}: {score.composite_score:.3f} "
                        f"({score.refinement_level.value}) -> {score.merge_recommendation.value}"
                    )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@geometry_refinement_app.command("summary")
def geometry_refinement_summary(
    ctx: typer.Context,
    result_file: str = typer.Argument(..., help="Path to refinement analysis JSON"),
) -> None:
    """Display summary of a saved refinement density analysis.

    Example:
        mc geometry refinement summary ./analysis.json
    """
    context = _context(ctx)

    try:
        data = json.loads(Path(result_file).read_text())

        payload = {
            "sourceModel": data.get("sourceModel"),
            "targetModel": data.get("targetModel"),
            "meanCompositeScore": data.get("meanCompositeScore"),
            "maxCompositeScore": data.get("maxCompositeScore"),
            "layersAboveHardSwap": data.get("layersAboveHardSwap"),
            "hardSwapLayers": data.get("hardSwapLayers"),
            "alphaByLayer": data.get("alphaByLayer"),
        }

        if context.output_format == "text":
            lines = [
                "REFINEMENT DENSITY SUMMARY",
                f"Source: {data.get('sourceModel')}",
                f"Target: {data.get('targetModel')}",
                f"Mean Score: {data.get('meanCompositeScore', 0):.3f}",
                f"Max Score: {data.get('maxCompositeScore', 0):.3f}",
                f"Hard Swap Candidates: {data.get('layersAboveHardSwap', 0)}",
            ]

            hard_swap = data.get("hardSwapLayers", [])
            if hard_swap:
                lines.append(f"Hard Swap Layers: {hard_swap}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc
