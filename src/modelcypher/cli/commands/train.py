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

"""Training CLI - ONE way to train.

Uses LoRAMemoryStore.train_step() - the geometric training loop.
All hyperparameters derived from model geometry, not heuristics.

Commands:
    mc train --agent <id> --model <path>           # Run training
    mc train status --agent <id> --model <path>    # Check training status
    mc train merge --agent <id> --model <path>     # Merge LoRA to base
    mc train export --agent <id> --model <path>    # Export LoRA weights
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

train_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _validate_model_path(model_path: Path, context: CLIContext) -> None:
    """Validate model path exists, exit with error if not."""
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-2001",
            title="Model not found",
            detail=f"Model path does not exist: {model_path}",
            hint="Provide a valid path to a model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@train_app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    agent: str = typer.Option(
        None, "--agent", "-a", help="Agent ID for training state"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Path to model directory"
    ),
    max_steps: int = typer.Option(
        None, "--max-steps", help="Maximum training steps (default: derived from buffer size)"
    ),
    batch_size: int = typer.Option(
        None, "--batch-size", help="Batch size per step (default: 1)"
    ),
    learning_rate: float = typer.Option(
        None, "--lr", help="Learning rate (default: derived from geometry)"
    ),
    convergence: float = typer.Option(
        None, "--convergence", help="Loss threshold for early stopping (default: sqrt(eps))"
    ),
) -> None:
    """Train LoRA adapters from accumulated events.

    Runs training steps on (hidden_state, delta) pairs accumulated
    during inference. Hyperparameters derived from model geometry.

    Example:
        mc train --agent agent-001 --model /path/to/model
    """
    # If no subcommand and no args, show help
    if ctx.invoked_subcommand is not None:
        return

    if agent is None or model is None:
        raise typer.BadParameter("--agent and --model are required for training")

    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.core.use_cases.lora_memory_service import LoRAMemoryService

    service = LoRAMemoryService()

    # Get or create store
    store = service.get_or_create_store(
        agent_id=agent,
        base_model_path=model_path,
    )

    if store.buffer_size == 0:
        error = ErrorDetail(
            code="MC-2011",
            title="No events to train",
            detail="Buffer is empty - no events have been accumulated",
            hint="Run inference first to accumulate events",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Derive parameters from geometry when not provided
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

    backend = get_default_backend()
    eps = backend.finfo().eps
    sqrt_eps = sqrt_scalar(eps, backend)

    if batch_size is None:
        batch_size = 1

    if convergence is None:
        convergence = sqrt_eps

    if max_steps is None:
        max_steps = max(1, store.buffer_size // batch_size)

    if learning_rate is None:
        learning_rate = sqrt_eps

    # Train
    train_result = service.train(
        agent_id=agent,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        convergence_threshold=convergence,
    )

    result = {
        "agent_id": agent,
        "training": train_result.to_dict(),
        "parameters": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "convergence": convergence,
        },
    }

    write_output(result, context.output_format, context.pretty)


@train_app.command("status")
def train_status(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for training state"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
) -> None:
    """Show training status for an agent.

    Displays buffer size, training progress, and merge history.

    Example:
        mc train status --agent agent-001 --model /path/to/model
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.core.use_cases.lora_memory_service import LoRAMemoryService

    service = LoRAMemoryService()

    # Get or create store to load status
    service.get_or_create_store(
        agent_id=agent,
        base_model_path=model_path,
    )

    status = service.status(agent)
    if status is None:
        error = ErrorDetail(
            code="MC-2010",
            title="Store not found",
            detail=f"No training state found for agent: {agent}",
            hint="Run training first to create state",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    result = {
        "agent_id": agent,
        "status": status.to_dict(),
    }

    write_output(result, context.output_format, context.pretty)


@train_app.command("merge")
def train_merge(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for training state"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output path for merged model"
    ),
    save_model: bool = typer.Option(
        False, "--save", help="Save the merged model"
    ),
    reset_after: bool = typer.Option(
        True, "--reset/--no-reset", help="Reset LoRA buffer after merge"
    ),
) -> None:
    """Merge LoRA adapters into base model weights.

    Transfers trained LoRA weights to base model via null-space projection.

    Example:
        mc train merge --agent agent-001 --model /path/to/model --save --output /path/to/merged
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    # Load model
    try:
        from modelcypher.adapters.inference_engine import load_model_and_tokenizer

        model_obj, tokenizer = load_model_and_tokenizer(model_path)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-2002",
            title="Model load failed",
            detail=str(exc),
            hint="Ensure the model path contains valid model files",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Get model config for null-space tracker
    base_model = getattr(model_obj, "model", model_obj)
    config = getattr(base_model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", getattr(base_model, "n_layers", 12))
    hidden_dim = getattr(config, "hidden_size", getattr(base_model, "hidden_size", 576))

    # Create null-space tracker
    from modelcypher.cli.composition import get_backend
    from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker

    b = get_backend()
    tracker = NullSpaceTracker(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        backend=b,
    )

    from modelcypher.core.use_cases.lora_memory_service import LoRAMemoryService

    service = LoRAMemoryService()

    # Get store
    service.get_or_create_store(
        agent_id=agent,
        base_model_path=model_path,
    )

    # Merge
    merge_result = service.merge_to_base(
        agent_id=agent,
        model=model_obj,
        null_space_tracker=tracker,
        save_merged=save_model,
        output_path=output,
    )

    # Reset if requested
    if merge_result.success and reset_after:
        service.reset_lora(agent)

    result = {
        "agent_id": agent,
        "merge": merge_result.to_dict(),
        "reset": reset_after and merge_result.success,
    }

    if save_model and output:
        result["saved_to"] = output

    write_output(result, context.output_format, context.pretty)


@train_app.command("export")
def train_export(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for training state"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output path for exported LoRA"
    ),
) -> None:
    """Export LoRA adapters to files.

    Exports the trained LoRA weights and metadata to a directory.

    Example:
        mc train export --agent agent-001 --model /path/to/model --output /path/to/export
    """
    context = _context(ctx)
    model_path = Path(model)
    output_path = Path(output)
    _validate_model_path(model_path, context)

    from modelcypher.core.use_cases.lora_memory_service import LoRAMemoryService

    service = LoRAMemoryService()

    # Get store
    service.get_or_create_store(
        agent_id=agent,
        base_model_path=model_path,
    )

    # Export
    export_result = service.export_lora(
        agent_id=agent,
        output_path=output_path,
    )

    if not export_result.success:
        error = ErrorDetail(
            code="MC-2012",
            title="Export failed",
            detail=export_result.error or "Unknown error",
            hint="Ensure the agent has trained LoRA weights",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    result = {
        "agent_id": agent,
        "export": export_result.to_dict(),
    }

    write_output(result, context.output_format, context.pretty)
