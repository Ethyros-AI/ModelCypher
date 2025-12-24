"""Training, job, and checkpoint CLI commands.

Provides commands for:
- Training management: start, preflight, status, pause, resume, cancel, export, logs
- Job management: list, show, attach, delete
- Checkpoint management: list, delete, export

Commands:
    mc train start --model <model> --dataset <dataset>
    mc train status <job_id>
    mc job list
    mc checkpoint list
"""

from __future__ import annotations

import json
import sys
import time


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.training import LoRAConfig, TrainingConfig
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.export_service import ExportService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.training_service import TrainingService
from modelcypher.utils.errors import ErrorDetail

train_app = typer.Typer(no_args_is_help=True)
checkpoint_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@train_app.command("start")
def train_start(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    batch_size: int = typer.Option(4, "--batch-size"),
    epochs: int = typer.Option(3, "--epochs"),
    sequence_length: int = typer.Option(2048, "--sequence-length"),
    grad_accum: int | None = typer.Option(None, "--grad-accum"),
    warmup_steps: int | None = typer.Option(None, "--warmup-steps"),
    weight_decay: float | None = typer.Option(None, "--weight-decay"),
    gradient_clip: float | None = typer.Option(None, "--gradient-clip"),
    resume_from: str | None = typer.Option(None, "--resume-from"),
    lora_rank: int | None = typer.Option(None, "--lora-rank"),
    lora_alpha: float | None = typer.Option(None, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout"),
    lora_targets: list[str] | None = typer.Option(None, "--lora-targets"),
    lora_layers: int | None = typer.Option(None, "--lora-layers"),
    out_dir: str | None = typer.Option(None, "--out"),
    seed: int | None = typer.Option(None, "--seed"),
    deterministic: bool = typer.Option(False, "--deterministic"),
    detach: bool = typer.Option(False, "--detach"),
    stream: bool = typer.Option(False, "--stream"),
) -> None:
    """Start a training job.

    Examples:
        mc train start --model meta-llama/Llama-2-7b --dataset ./data.jsonl
        mc train start --model ./local-model --dataset ./data.jsonl --lora-rank 8 --lora-alpha 16
    """
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
        result, events = service.start(config, stream=stream, detach=detach)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-5001",
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
    grad_accum: int | None = typer.Option(None, "--grad-accum"),
    warmup_steps: int | None = typer.Option(None, "--warmup-steps"),
    weight_decay: float | None = typer.Option(None, "--weight-decay"),
    gradient_clip: float | None = typer.Option(None, "--gradient-clip"),
    resume_from: str | None = typer.Option(None, "--resume-from"),
    lora_rank: int | None = typer.Option(None, "--lora-rank"),
    lora_alpha: float | None = typer.Option(None, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout"),
    lora_targets: list[str] | None = typer.Option(None, "--lora-targets"),
    lora_layers: int | None = typer.Option(None, "--lora-layers"),
    out_dir: str | None = typer.Option(None, "--out"),
    seed: int | None = typer.Option(None, "--seed"),
    deterministic: bool = typer.Option(False, "--deterministic"),
) -> None:
    """Run preflight checks before training.

    Examples:
        mc train preflight --model meta-llama/Llama-2-7b --dataset ./data.jsonl
    """
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
    """Get training job status.

    Examples:
        mc train status abc123
        mc train status abc123 --follow
    """
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
    """Pause a training job.

    Examples:
        mc train pause abc123
    """
    context = _context(ctx)
    service = TrainingService()
    write_output(service.pause(job_id), context.output_format, context.pretty)


@train_app.command("resume")
def train_resume(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    """Resume a paused training job.

    Examples:
        mc train resume abc123
    """
    context = _context(ctx)
    service = TrainingService()
    write_output(service.resume(job_id), context.output_format, context.pretty)


@train_app.command("cancel")
def train_cancel(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    """Cancel a training job.

    Examples:
        mc train cancel abc123
    """
    context = _context(ctx)
    service = TrainingService()
    write_output(service.cancel(job_id), context.output_format, context.pretty)


@train_app.command("export")
def train_export(
    ctx: typer.Context,
    model: str | None = typer.Option(None, "--model"),
    job: str | None = typer.Option(None, "--job"),
    export_format: str = typer.Option(..., "--format"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Export a trained model or job.

    Examples:
        mc train export --model ./fine-tuned --format gguf --output-path ./model.gguf
        mc train export --job abc123 --format safetensors --output-path ./model
    """
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
    """View training logs.

    Examples:
        mc train logs abc123
        mc train logs abc123 --tail 50 --follow
    """
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


# Checkpoint commands


@checkpoint_app.command("list")
def checkpoint_list(ctx: typer.Context, job: str | None = typer.Option(None, "--job")) -> None:
    """List checkpoints.

    Examples:
        mc checkpoint list
        mc checkpoint list --job abc123
    """
    context = _context(ctx)
    service = CheckpointService()
    write_output(service.list_checkpoints(job), context.output_format, context.pretty)


@checkpoint_app.command("delete")
def checkpoint_delete(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Delete a checkpoint.

    Examples:
        mc checkpoint delete ./checkpoints/step-1000
        mc checkpoint delete ./checkpoints/step-1000 --force
    """
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
    """Export a checkpoint.

    Examples:
        mc checkpoint export ./checkpoints/step-1000 --format safetensors --output-path ./model
    """
    context = _context(ctx)
    service = CheckpointService()
    write_output(service.export_checkpoint(checkpoint_path, export_format, output_path), context.output_format, context.pretty)
