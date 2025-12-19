from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from modelcypher.adapters.asif_packager import ASIFPackager
from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.presenters import (
    compare_detail_payload,
    compare_list_payload,
    dataset_payload,
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
from modelcypher.core.domain.training import LoRAConfig, TrainingConfig
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.compare_service import CompareService
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService
from modelcypher.core.use_cases.doc_service import DocService
from modelcypher.core.use_cases.evaluation_service import EvaluationService
from modelcypher.core.use_cases.export_service import ExportService
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.inventory_service import InventoryService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.model_merge_service import ModelMergeService
from modelcypher.core.use_cases.model_search_service import ModelSearchService
from modelcypher.core.use_cases.model_service import ModelService
from modelcypher.core.use_cases.system_service import SystemService
from modelcypher.core.use_cases.training_service import TrainingService
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json
from modelcypher.utils.logging import configure_logging
from modelcypher.utils.limits import MAX_FIELD_BYTES, MAX_PREVIEW_LINES, MAX_RAW_BYTES

app = typer.Typer(no_args_is_help=True, add_completion=False)
train_app = typer.Typer(no_args_is_help=True)
job_app = typer.Typer(no_args_is_help=True)
checkpoint_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
system_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
compare_app = typer.Typer(no_args_is_help=True)
doc_app = typer.Typer(no_args_is_help=True)
validate_app = typer.Typer(no_args_is_help=True)
estimate_app = typer.Typer(no_args_is_help=True)
geometry_app = typer.Typer(no_args_is_help=True)
path_app = typer.Typer(no_args_is_help=True)

app.add_typer(train_app, name="train")
app.add_typer(job_app, name="job")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(model_app, name="model")
app.add_typer(system_app, name="system")
app.add_typer(dataset_app, name="dataset")
app.add_typer(eval_app, name="eval")
app.add_typer(compare_app, name="compare")
app.add_typer(doc_app, name="doc")
app.add_typer(validate_app, name="validate")
app.add_typer(estimate_app, name="estimate")
app.add_typer(geometry_app, name="geometry")
geometry_app.add_typer(path_app, name="path")


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


@train_app.command("start")
def train_start(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    batch_size: int = typer.Option(4, "--batch-size"),
    epochs: int = typer.Option(3, "--epochs"),
    sequence_length: int = typer.Option(2048, "--sequence-length"),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    gradient_clip: Optional[float] = typer.Option(None, "--gradient-clip"),
    resume_from: Optional[str] = typer.Option(None, "--resume-from"),
    lora_rank: Optional[int] = typer.Option(None, "--lora-rank"),
    lora_alpha: Optional[float] = typer.Option(None, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout"),
    lora_targets: Optional[list[str]] = typer.Option(None, "--lora-targets"),
    lora_layers: Optional[int] = typer.Option(None, "--lora-layers"),
    out_dir: Optional[str] = typer.Option(None, "--out"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    deterministic: bool = typer.Option(False, "--deterministic"),
    detach: bool = typer.Option(False, "--detach"),
    stream: bool = typer.Option(False, "--stream"),
) -> None:
    context = _context(ctx)
    lora = None
    if lora_rank and lora_alpha:
        lora = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            targets=lora_targets or ["q_proj", "v_proj"],
            layers=lora_layers,
        )
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
        grad_accum=grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        resume_from=resume_from,
        lora=lora,
        out_dir=out_dir,
        seed=seed,
        deterministic=deterministic,
    )
    service = TrainingService()
    try:
        result, events = service.start(config, stream=stream)
    except Exception as exc:
        error = ErrorDetail(
            code="TC-5001",
            title="Training failed",
            detail=str(exc),
            hint="Verify model and dataset paths",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    if stream:
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
        return
    write_output(result, context.output_format, context.pretty)


@train_app.command("preflight")
def train_preflight(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    batch_size: int = typer.Option(4, "--batch-size"),
    epochs: int = typer.Option(3, "--epochs"),
    sequence_length: int = typer.Option(2048, "--sequence-length"),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    gradient_clip: Optional[float] = typer.Option(None, "--gradient-clip"),
    resume_from: Optional[str] = typer.Option(None, "--resume-from"),
    lora_rank: Optional[int] = typer.Option(None, "--lora-rank"),
    lora_alpha: Optional[float] = typer.Option(None, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout"),
    lora_targets: Optional[list[str]] = typer.Option(None, "--lora-targets"),
    lora_layers: Optional[int] = typer.Option(None, "--lora-layers"),
    out_dir: Optional[str] = typer.Option(None, "--out"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    deterministic: bool = typer.Option(False, "--deterministic"),
) -> None:
    context = _context(ctx)
    lora = None
    if lora_rank and lora_alpha:
        lora = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            targets=lora_targets or ["q_proj", "v_proj"],
            layers=lora_layers,
        )
    config = TrainingConfig(
        model_id=model,
        dataset_path=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
        grad_accum=grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        resume_from=resume_from,
        lora=lora,
        out_dir=out_dir,
        seed=seed,
        deterministic=deterministic,
    )
    service = TrainingService()
    result = service.preflight(config)
    write_output(result, context.output_format, context.pretty)


@train_app.command("status")
def train_status(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    follow: bool = typer.Option(False, "--follow"),
    stream: bool = typer.Option(False, "--stream"),
) -> None:
    context = _context(ctx)
    service = TrainingService()
    if stream:
        job_service = JobService()
        lines = job_service.attach(job_id)
        for line in lines:
            sys.stdout.write(line + "\n")
        return
    if follow:
        while True:
            status = service.status(job_id)
            write_output(status, context.output_format, context.pretty)
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(2)
        return
    write_output(service.status(job_id), context.output_format, context.pretty)


@train_app.command("pause")
def train_pause(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = TrainingService()
    write_output(service.pause(job_id), context.output_format, context.pretty)


@train_app.command("resume")
def train_resume(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = TrainingService()
    write_output(service.resume(job_id), context.output_format, context.pretty)


@train_app.command("cancel")
def train_cancel(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = TrainingService()
    write_output(service.cancel(job_id), context.output_format, context.pretty)


@train_app.command("export")
def train_export(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model"),
    job: Optional[str] = typer.Option(None, "--job"),
    export_format: str = typer.Option(..., "--format"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    context = _context(ctx)
    service = ExportService()
    if bool(model) == bool(job):
        raise typer.BadParameter("Provide exactly one of --model or --job")
    if model:
        result = service.export_model(model, export_format, output_path)
    else:
        result = service.export_job(job, export_format, output_path)
    write_output(result, context.output_format, context.pretty)


@train_app.command("logs")
def train_logs(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    tail: int = typer.Option(100, "--tail"),
    follow: bool = typer.Option(False, "--follow"),
) -> None:
    service = TrainingService()
    lines = service.logs(job_id, tail=tail)
    for line in lines:
        sys.stdout.write(line + "\n")
    if follow:
        while True:
            time.sleep(1)
            lines = service.logs(job_id, tail=1)
            for line in lines:
                sys.stdout.write(line + "\n")


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    status: Optional[str] = typer.Option(None, "--status"),
    active_only: bool = typer.Option(False, "--active-only"),
) -> None:
    context = _context(ctx)
    service = JobService()
    write_output(service.list_jobs(status=status, active_only=active_only), context.output_format, context.pretty)


@job_app.command("show")
def job_show(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    loss_history: bool = typer.Option(False, "--loss-history"),
) -> None:
    context = _context(ctx)
    service = JobService()
    write_output(service.show_job(job_id, include_loss_history=loss_history), context.output_format, context.pretty)


@job_app.command("attach")
def job_attach(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    replay: bool = typer.Option(False, "--replay"),
    since: Optional[str] = typer.Option(None, "--since"),
) -> None:
    service = JobService()
    lines = service.attach(job_id, since=since if replay else None)
    for line in lines:
        sys.stdout.write(line + "\n")


@job_app.command("delete")
def job_delete(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    service = JobService()
    write_output(service.delete_job(job_id), context.output_format, context.pretty)


@checkpoint_app.command("list")
def checkpoint_list(ctx: typer.Context, job: Optional[str] = typer.Option(None, "--job")) -> None:
    context = _context(ctx)
    service = CheckpointService()
    write_output(service.list_checkpoints(job), context.output_format, context.pretty)


@checkpoint_app.command("delete")
def checkpoint_delete(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
) -> None:
    context = _context(ctx)
    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete checkpoint {path}?"):
            raise typer.Exit(code=1)
    service = CheckpointService()
    write_output(service.delete_checkpoint(path), context.output_format, context.pretty)


@checkpoint_app.command("export")
def checkpoint_export(
    ctx: typer.Context,
    checkpoint_path: str = typer.Argument(...),
    export_format: str = typer.Option(..., "--format"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    context = _context(ctx)
    service = CheckpointService()
    write_output(service.export_checkpoint(checkpoint_path, export_format, output_path), context.output_format, context.pretty)


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
    verbose: bool = typer.Option(False, "--verbose"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_path: Optional[str] = typer.Option(None, "--report-path"),
) -> None:
    context = _context(ctx)
    service = ModelMergeService()
    report = service.merge(
        source_id=source,
        target_id=target,
        output_dir=output_dir,
        alpha=alpha,
        anchor_mode=anchor_mode,
        dry_run=dry_run,
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
            code="TC-5002",
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
    output_path: str = typer.Option(..., "--output", "-o", "--dataset-output", "--processed-output"),
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
    output: str = typer.Option(..., "--output"),
) -> None:
    context = _context(ctx)
    service = DatasetEditorService()
    target_format = parse_format(to_format)
    result = service.convert_dataset(input_path, target_format, output)
    write_output(result, context.output_format, context.pretty)


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
    if requested > limit:
        sys.stderr.write(f"Warning: preview capped at {limit} lines (requested {requested}).\n")
    preview = service.preview(path, limit)

    if context.output_format == "text":
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

    write_output(preview, context.output_format, context.pretty)


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

    write_output(row, context.output_format, context.pretty)


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

    write_output(result, context.output_format, context.pretty)


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

    write_output(result, context.output_format, context.pretty)


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

    write_output(result, context.output_format, context.pretty)


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
    output: str = typer.Option(..., "--output"),
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
        output_path=output,
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
        "nextActions": [f"tc train start --model {model} --dataset {dataset}"],
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
        "nextActions": ["tc train start --model <model> --dataset <dataset>"],
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
        "nextActions": [f"tc train start --model {model} --dataset {dataset} --batch-size {batch_size}"],
    }
    write_output(payload, context.output_format, context.pretty)


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
