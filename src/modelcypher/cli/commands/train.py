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

Usage:
    mc train --agent <id> --model <path>
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

train_app = typer.Typer(no_args_is_help=True)
checkpoint_app = typer.Typer(no_args_is_help=True)  # Stub for compatibility


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@train_app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for training state"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
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
    context = _context(ctx)
    model_path = Path(model)

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
