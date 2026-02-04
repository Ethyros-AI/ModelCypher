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

"""Compression gate analysis CLI commands.

Analyzes which layers consistently expand vs compress representations.
Base models have "compression gate" layers; specialists do not.

Commands:
    mc geometry compression-gate --model <path>
    mc geometry compression-gate --trajectory <path>
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


# Task probes for analysis
TASK_PROBES = {
    "retrieval": "What is the capital of France?",
    "arithmetic": "What is 7 + 5?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "logic": "If all cats are animals, and all animals need water, do cats need water?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
    "cot": "Let me think step by step about how to solve this problem: What is 15% of 80?",
}


def _trace_trajectory(model, tokenizer, prompt: str, backend=None) -> list[float]:
    """Trace norm through all layers using Backend."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

    return backend.trace_norm_trajectory(model, tokenizer, prompt)


def _analyze_layer_roles(trajectories: dict[str, list[float]]) -> dict:
    """Analyze which layers consistently expand vs compress."""
    n_tasks = len(trajectories)
    if n_tasks == 0:
        return {}

    # Get number of layers from first trajectory
    first_norms = next(iter(trajectories.values()))
    n_layers = len(first_norms) - 1  # Subtract embedding

    # Track per-layer behavior
    layer_expansion_count = [0] * (n_layers + 1)
    layer_compression_count = [0] * (n_layers + 1)

    for norms in trajectories.values():
        for i in range(1, len(norms)):
            if norms[i] > norms[i - 1]:
                layer_expansion_count[i] += 1
            elif norms[i] < norms[i - 1]:
                layer_compression_count[i] += 1

    # Classify layers
    always_expand = []
    always_compress = []
    mixed = []

    for i in range(1, n_layers + 1):
        exp = layer_expansion_count[i]
        comp = layer_compression_count[i]

        if exp == n_tasks and comp == 0:
            always_expand.append(i)
        elif comp == n_tasks and exp == 0:
            always_compress.append(i)
        else:
            mixed.append(i)

    # Compute layer contributions
    layer_delta = {i: [] for i in range(1, n_layers + 1)}
    for norms in trajectories.values():
        for i in range(1, len(norms)):
            delta = (norms[i] - norms[i - 1]) / (norms[0] + 1e-10)
            layer_delta[i].append(delta)

    # Compute mean using Python (no numpy needed)
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    avg_delta = {i: _mean(layer_delta[i]) for i in layer_delta}

    # Find compression gate (strongest compressor)
    compression_gate = None
    max_compression = 0
    for i, delta in avg_delta.items():
        if delta < -max_compression:
            max_compression = -delta
            compression_gate = i

    # Compute gate strength
    gate_strength = max_compression if compression_gate else 0.0

    return {
        "n_layers": n_layers,
        "n_tasks": n_tasks,
        "always_expand": always_expand,
        "always_compress": always_compress,
        "mixed": mixed,
        "compression_gate_layer": compression_gate,
        "compression_gate_strength": gate_strength,
        "has_compression_gate": len(always_compress) > 0,
        "layer_avg_delta": avg_delta,
        "layer_expansion_rate": {
            i: layer_expansion_count[i] for i in range(1, n_layers + 1)
        },
        "layer_compression_rate": {
            i: layer_compression_count[i] for i in range(1, n_layers + 1)
        },
    }


@app.command("analyze")
def compression_gate_analyze(
    ctx: typer.Context,
    model: str = typer.Option(None, "--model", "-m", help="Path to model directory"),
    trajectory: str = typer.Option(
        None, "--trajectory", "-t", help="Path to trajectory JSON file"
    ),
) -> None:
    """Analyze compression gate in a model.

    Identifies which layers consistently compress vs expand representations.
    Base models have compression gates; specialist models do not.

    Examples:
        mc geometry compression-gate analyze --model ./my-model
        mc geometry compression-gate analyze --trajectory ./trajectory.json
    """
    context = _context(ctx)

    if not model and not trajectory:
        error = ErrorDetail(
            code="MC-4001",
            title="Missing input",
            detail="Provide either --model or --trajectory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    trajectories = {}

    if trajectory:
        # Load from file
        import json

        traj_path = Path(trajectory)
        if not traj_path.exists():
            error = ErrorDetail(
                code="MC-4002",
                title="File not found",
                detail=f"Trajectory file not found: {trajectory}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        with open(traj_path) as f:
            data = json.load(f)

        for t in data.get("trajectories", []):
            task_type = t.get("task_type", "unknown")
            norms = t.get("norms", [])
            trajectories[task_type] = norms
    else:
        # Generate from model
        from modelcypher.adapters.model_loader import ModelLoader

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

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        for task_type, prompt in TASK_PROBES.items():
            norms = _trace_trajectory(loaded_model, tokenizer, prompt)
            trajectories[task_type] = norms

    # Analyze
    result = _analyze_layer_roles(trajectories)

    # Add interpretation
    if result.get("has_compression_gate"):
        result["interpretation"] = "BASE_MODEL"
        result["description"] = (
            f"Model has compression gate at L{result['compression_gate_layer']} "
            f"(strength: {result['compression_gate_strength']:.3f}). "
            f"This indicates a base/general model with task differentiation."
        )
    else:
        result["interpretation"] = "SPECIALIST_MODEL"
        result["description"] = (
            "Model has no compression gate (no always-compress layers). "
            "This indicates a specialist model with constant geometry."
        )

    write_output(result, context.output_format, context.pretty)
