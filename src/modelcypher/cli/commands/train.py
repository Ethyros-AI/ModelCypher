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

"""Training, job, and checkpoint CLI commands.

Provides commands for:
- Training management: start, preflight, status, pause, resume, cancel, export, logs
- Self-reflection training: Train models for geometric alignment through self-reflection
- biLM probe training: Train bidirectional LM probes for token-level classification
- Job management: list, show, attach, delete
- Checkpoint management: list, delete, export

Commands:
    mc train start --model <model> --dataset <dataset>
    mc train self-reflection --model <model> --output <path>
    mc train bilm-probe --positive pos.jsonl --negative neg.jsonl --output probe.json
    mc train status <job_id>
    mc job list
    mc checkpoint list
"""

from __future__ import annotations

import json
import sys
import time

import typer

from modelcypher.cli.composition import (
    get_checkpoint_service,
    get_export_service,
    get_job_service,
    get_training_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
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
    resume_from: str | None = typer.Option(None, "--resume-from"),
    out_dir: str = typer.Option(..., "--out"),
    detach: bool = typer.Option(False, "--detach"),
    stream: bool = typer.Option(False, "--stream"),
) -> None:
    """Start a training job.

    Examples:
        mc train start --model meta-llama/Llama-2-7b --dataset ./data.jsonl
    """
    context = _context(ctx)
    service = get_training_service()
    try:
        config = service.derive_spec(
            model=model,
            dataset=dataset,
            output_path=out_dir,
            resume_from=resume_from,
        )
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
    resume_from: str | None = typer.Option(None, "--resume-from"),
    out_dir: str = typer.Option(..., "--out"),
) -> None:
    """Run preflight checks before training.

    Examples:
        mc train preflight --model meta-llama/Llama-2-7b --dataset ./data.jsonl
    """
    context = _context(ctx)
    service = get_training_service()
    config = service.derive_spec(
        model=model,
        dataset=dataset,
        output_path=out_dir,
        resume_from=resume_from,
    )
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
    service = get_training_service()
    if stream:
        job_service = get_job_service()
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
    service = get_training_service()
    write_output(service.pause(job_id), context.output_format, context.pretty)


@train_app.command("resume")
def train_resume(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    """Resume a paused training job.

    Examples:
        mc train resume abc123
    """
    context = _context(ctx)
    service = get_training_service()
    write_output(service.resume(job_id), context.output_format, context.pretty)


@train_app.command("cancel")
def train_cancel(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    """Cancel a training job.

    Examples:
        mc train cancel abc123
    """
    context = _context(ctx)
    service = get_training_service()
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
        mc train export --job abc123 --format safetensors --output-path ./model
    """
    context = _context(ctx)
    service = get_export_service()
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
    service = get_training_service()
    lines = service.logs(job_id, tail=tail)
    for line in lines:
        sys.stdout.write(line + "\n")
    if follow:
        while True:
            time.sleep(1)
            lines = service.logs(job_id, tail=1)
            for line in lines:
                sys.stdout.write(line + "\n")


@train_app.command("self-reflection")
def train_self_reflection(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to base model"),
    adapter_path: str = typer.Option("", "--adapter-path", "-o", help="Path to save LoRA adapters"),
    rank: int = typer.Option(8, "--rank", help="LoRA rank (default: 8)"),
    epochs: int = typer.Option(15, "--epochs", help="Training epochs (default: 15)"),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate (default: 1e-4)"),
    test: bool = typer.Option(True, "--test/--no-test", help="Run tests after training"),
    layer_start: int | None = typer.Option(None, "--layer-start", help="First layer index for LoRA (default: all)"),
    layer_end: int | None = typer.Option(None, "--layer-end", help="Last layer index for LoRA (default: all)"),
    entropy_probe_path: str = typer.Option("", "--entropy-probe-path", help="Path to probe prompts for entropy profiling"),
    entropy_profile_output: str = typer.Option("", "--entropy-profile-output", help="Path to save entropy profile JSON"),
    id_profile_output: str = typer.Option("", "--id-profile-output", help="Path to save intrinsic dimension profile JSON"),
    training_data: str = typer.Option("", "--training-data", "-d", help="Path to custom JSONL training data"),
) -> None:
    """Train model for self-reflection using LoRA.

    Self-reflection training teaches the model to extract core questions
    before answering, achieving φ resonance for optimal geometric processing.

    Research basis:
    - Question normalization improves φ alignment by 73%
    - Self-reflection achieves 100% accuracy on problems that trip up intuitive processing

    Examples:
        mc train self-reflection --model /path/to/model
        mc train self-reflection --model /path/to/model --adapter-path ./adapters --epochs 20
        mc train self-reflection --model /path/to/model --rank 16 --lr 5e-5
        mc train self-reflection --model /path/to/model --training-data data/training/phase1.jsonl
    """
    context = _context(ctx)

    # Convert empty string to None for optional output
    output_path = adapter_path if adapter_path else None

    try:
        from modelcypher.core.domain.training.self_reflection import (
            train_self_reflection_lora,
        )

        result = train_self_reflection_lora(
            model_path=model,
            output_path=output_path,
            rank=rank,
            num_epochs=epochs,
            learning_rate=learning_rate,
            run_tests=test,
            layer_start=layer_start,
            layer_end=layer_end,
            entropy_probe_path=entropy_probe_path or None,
            entropy_profile_output=entropy_profile_output or None,
            id_profile_output=id_profile_output or None,
            training_data_path=training_data or None,
        )
        write_output(result, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-5010",
            title="Self-reflection training failed",
            detail=str(exc),
            hint="Check model path and GPU memory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@train_app.command("expansion-aligned")
def train_expansion_aligned(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to base model"),
    adapter_path: str = typer.Option("", "--adapter-path", "-o", help="Path to save LoRA adapters"),
    expansion_weight: float = typer.Option(0.01, "--expansion-weight", help="Weight for expansion loss (default: 0.01)"),
    rank: int = typer.Option(8, "--rank", help="LoRA rank (default: 8)"),
    epochs: int = typer.Option(15, "--epochs", help="Training epochs (default: 15)"),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate (default: 1e-4)"),
    warmup_epochs: int = typer.Option(0, "--warmup-epochs", help="Epochs before expansion loss kicks in (default: 0)"),
    ramp_epochs: int = typer.Option(0, "--ramp-epochs", help="Epochs to ramp expansion loss weight (default: 0)"),
    test: bool = typer.Option(True, "--test/--no-test", help="Run tests after training"),
    training_data: str = typer.Option("", "--training-data", "-d", help="Path to custom JSONL training data"),
) -> None:
    """[EXPERIMENTAL] Train model with differentiable expansion loss for geometric alignment.

    WARNING: This training mode is EXPERIMENTAL. The assumption that expansion_ratio = 1.0
    is the optimal target for all tasks has NOT been validated across diverse inputs.
    Use scripts/measure_expansion_distribution.py to gather empirical data before training.

    Research questions that remain unanswered:
    - What is the natural expansion_ratio distribution for different task types?
    - Does the optimal value vary by model size or architecture?
    - Is there a single attractor or multiple basins for different processing modes?

    This training mode combines standard task loss (next-token prediction)
    with a differentiable expansion loss that encourages balanced expansion/compression
    geometry (expansion_ratio = 1.0).

    The expansion loss is: |expansion_ratio - 1.0|

    Where:
    - expansion_rate = (peak_norm - initial_norm) / peak_layer
    - compression_rate = (peak_norm - final_norm) / (n_layers - peak_layer)
    - expansion_ratio = compression_rate / expansion_rate

    Examples:
        mc train expansion-aligned --model /path/to/model
        mc train expansion-aligned --model /path/to/model --expansion-weight 0.02 --adapter-path ./adapters
        mc train expansion-aligned --model /path/to/model --warmup-epochs 2 --ramp-epochs 3
    """
    context = _context(ctx)

    # Print experimental warning
    import sys
    sys.stderr.write(
        "\n"
        "WARNING: expansion-aligned training is EXPERIMENTAL.\n"
        "The assumption that expansion_ratio = 1.0 is optimal for all tasks is UNVALIDATED.\n"
        "Consider running: python scripts/measure_expansion_distribution.py --model <model>\n"
        "to gather empirical data before training toward a specific target.\n"
        "\n"
    )

    # Convert empty string to None for optional output
    output_path = adapter_path if adapter_path else None

    try:
        from modelcypher.core.domain.training.self_reflection import (
            train_with_expansion_loss,
        )

        result = train_with_expansion_loss(
            model_path=model,
            output_path=output_path,
            expansion_weight=expansion_weight,
            rank=rank,
            num_epochs=epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            run_tests=test,
            training_data_path=training_data or None,
        )
        write_output(result, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-5011",
            title="Phi-aligned training failed",
            detail=str(exc),
            hint="Check model path and GPU memory. Ensure phi_weight is not too high (try 0.001-0.1).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@train_app.command("bilm-probe")
def train_bilm_probe(
    ctx: typer.Context,
    positive_file: str = typer.Option(..., "--positive", "-p", help="JSONL file with positive example activations"),
    negative_file: str = typer.Option(..., "--negative", "-n", help="JSONL file with negative example activations"),
    output_path: str = typer.Option("", "--output", "-o", help="Path to save trained probe weights"),
    val_split: float = typer.Option(0.1, "--val-split", help="Fraction of data for validation"),
    learning_rate: float = typer.Option(0.01, "--lr", help="Learning rate"),
    max_iterations: int = typer.Option(1000, "--max-iter", help="Maximum training iterations"),
) -> None:
    """Train a bidirectional LM probe for token-level classification.

    Trains a linear probe on concatenated forward and backward LM representations
    to classify tokens as belonging to a target domain or not.

    Implementation based on arXiv:2601.21571v1.

    Input files should be JSONL with records containing:
    - "forward": Forward LM hidden state [hidden_dim]
    - "backward": Backward LM hidden state [hidden_dim]

    Examples:
        mc train bilm-probe --positive domain.jsonl --negative general.jsonl -o probe.json
        mc train bilm-probe -p pos.jsonl -n neg.jsonl --lr 0.001 --max-iter 2000
    """
    context = _context(ctx)

    try:
        from modelcypher.backends import get_backend
        from modelcypher.core.use_cases.bilm_probe_service import BiLMProbeService

        backend = get_backend()
        service = BiLMProbeService(backend)

        # Load positive examples
        forward_pos = []
        backward_pos = []
        with open(positive_file, "r") as f:
            for line in f:
                record = json.loads(line)
                forward_pos.append(record["forward"])
                backward_pos.append(record["backward"])

        # Load negative examples
        forward_neg = []
        backward_neg = []
        with open(negative_file, "r") as f:
            for line in f:
                record = json.loads(line)
                forward_neg.append(record["forward"])
                backward_neg.append(record["backward"])

        if not forward_pos or not forward_neg:
            error = ErrorDetail(
                code="MC-5020",
                title="Empty training data",
                detail="Need both positive and negative examples",
                hint="Check that input files contain valid JSONL with 'forward' and 'backward' fields",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        summary, result = service.train(
            forward_positive=backend.array(forward_pos),
            backward_positive=backend.array(backward_pos),
            forward_negative=backend.array(forward_neg),
            backward_negative=backend.array(backward_neg),
            val_split=val_split,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            output_path=output_path if output_path else None,
        )

        payload = BiLMProbeService.training_payload(summary)

        if context.output_format == "text":
            lines = [
                "BILM PROBE TRAINING COMPLETE",
                f"Training accuracy: {summary.train_accuracy:.2%}",
                f"Training F1: {summary.train_f1:.4f}",
            ]
            if summary.val_accuracy is not None:
                lines.append(f"Validation accuracy: {summary.val_accuracy:.2%}")
                lines.append(f"Validation F1: {summary.val_f1:.4f}")
            lines.append(f"Training samples: {summary.n_train}")
            lines.append(f"Validation samples: {summary.n_val}")
            if summary.output_path:
                lines.append(f"Saved to: {summary.output_path}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-5021",
            title="biLM probe training failed",
            detail=str(exc),
            hint="Check input file format and learning rate",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


# Checkpoint commands


@checkpoint_app.command("list")
def checkpoint_list(ctx: typer.Context, job: str | None = typer.Option(None, "--job")) -> None:
    """List checkpoints.

    Examples:
        mc checkpoint list
        mc checkpoint list --job abc123
    """
    context = _context(ctx)
    service = get_checkpoint_service()
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
    service = get_checkpoint_service()
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
    service = get_checkpoint_service()
    write_output(
        service.export_checkpoint(checkpoint_path, export_format, output_path),
        context.output_format,
        context.pretty,
    )
