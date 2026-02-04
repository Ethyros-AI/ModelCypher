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

"""Continual learning CLI commands.

Commands for inference-time learning and manifold consolidation.

Commands:
    mc learn consolidate --model <path> [--session <file>]
    mc learn status --model <path>
    mc learn null-space --model <path>
    mc learn lora-status --agent <id> --model <path>
    mc learn lora-train --agent <id> --model <path>
    mc learn merge-lora --agent <id> --model <path>
    mc learn lora-export --agent <id> --output <path>
    mc learn benchmark --model <path> --capture --output <file>
    mc learn benchmark --model <path> --before <file> --output <file>
    mc learn monitor --model <path> --status
    mc learn monitor --model <path> --auto
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("consolidate")
def learn_consolidate(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    session: str | None = typer.Option(
        None, "--session", "-s", help="Path to session file with sparsity events (JSON)"
    ),
    max_steps: int = typer.Option(
        50, "--max-steps", help="Maximum consolidation steps"
    ),
    max_probes: int = typer.Option(
        100, "--max-probes", help="Maximum probe embeddings to generate"
    ),
    save_model: bool = typer.Option(
        False, "--save", help="Save consolidated model weights"
    ),
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Output path for consolidated model"
    ),
) -> None:
    """Run manifold consolidation on a model.

    Consolidation fills in sparse regions of the model's representational
    manifold, making it denser and more robust.

    If --session is provided, uses sparsity events from that file.
    Otherwise, generates synthetic probes for demonstration.

    Examples:

        # Basic consolidation with synthetic probes
        mc learn consolidate --model /path/to/smolLM

        # Consolidation with session data
        mc learn consolidate --model /path/to/smolLM --session /path/to/session.json

        # Save consolidated model
        mc learn consolidate --model /path/to/smolLM --save --output /path/to/output
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

    # Load model
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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

    # Get model config
    base_model = getattr(model_obj, "model", model_obj)
    config = getattr(base_model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", getattr(base_model, "n_layers", 12))
    hidden_dim = getattr(config, "hidden_size", getattr(base_model, "hidden_size", 576))

    # Create consolidation service
    from modelcypher.core.use_cases.consolidation_service import (
        ConsolidationConfig,
        ConsolidationService,
        ConsolidationStats,
        create_consolidation_service,
    )

    service = create_consolidation_service(
        model=model_obj,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    config_obj = ConsolidationConfig(
        max_probes=max_probes,
        max_completion_steps=max_steps,
        clear_queue_after=True,
    )

    stats: ConsolidationStats

    if session:
        # Load session with sparsity events
        session_path = Path(session)
        if not session_path.exists():
            error = ErrorDetail(
                code="MC-2003",
                title="Session file not found",
                detail=f"Session file does not exist: {session_path}",
                hint="Provide a valid session file path",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        try:
            with open(session_path) as f:
                session_data = json.load(f)

            # Create bridge and load events
            from modelcypher.core.use_cases.entropy_learning_bridge import (
                EntropyLearningBridge,
                SparsityEvent,
            )
            from modelcypher.core.use_cases.entropy_monitor import UncertaintyAction

            bridge = EntropyLearningBridge(hidden_dim=hidden_dim)

            # Parse sparsity events from session
            events = session_data.get("sparsity_events", [])
            for event_data in events:
                event = SparsityEvent(
                    token_index=event_data.get("token_index", 0),
                    eigenscore=event_data.get("eigenscore", 0.0),
                    refusal_projection=event_data.get("refusal_projection", 0.0),
                    action=UncertaintyAction(event_data.get("action", "WARN")),
                    hidden_state_hash=event_data.get("hidden_state_hash", 0),
                    layer_index=event_data.get("layer_index", -1),
                )
                bridge._sparsity_queue.append(event)

            stats = service.consolidate_from_bridge(bridge, config_obj)

        except json.JSONDecodeError as exc:
            error = ErrorDetail(
                code="MC-2004",
                title="Invalid session file",
                detail=str(exc),
                hint="Session file must be valid JSON",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
    else:
        # Generate synthetic probes for demonstration
        b = get_backend()

        # Generate random probes with varied density
        probes = b.random_normal((max_probes, hidden_dim))
        b.eval(probes)

        # Run consolidation stream
        completion_steps = 0
        encodings_applied = 0
        total_entropy_reduction = 0.0

        for step in service.consolidate_stream(probes, max_steps=max_steps):
            completion_steps += 1
            if step.encoding_applied:
                encodings_applied += 1
            total_entropy_reduction += step.entropy_reduction

            if step.converged:
                break

        # Build stats manually
        from modelcypher.core.use_cases.consolidation_service import (
            ConsolidationStatus,
        )

        stats = ConsolidationStats(
            status=ConsolidationStatus.done,
            sparsity_events_processed=0,
            probes_generated=max_probes,
            completion_steps=completion_steps,
            encodings_applied=encodings_applied,
            mean_entropy_before=0.0,  # Not tracked in stream mode
            mean_entropy_after=0.0,
            entropy_reduction=total_entropy_reduction,
            mean_preserved_fraction=0.0,
        )

    # Save model if requested
    if save_model:
        out_path = Path(output_path) if output_path else model_path / "consolidated"
        try:
            # Save model via Backend
            from modelcypher.cli.composition import get_backend

            backend = get_backend()
            weights = dict(model_obj.parameters())
            backend.save_safetensors(str(out_path / "model.safetensors"), weights)

            # Copy config files
            import shutil

            for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                src = model_path / config_file
                if src.exists():
                    out_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, out_path / config_file)

            stats_dict = stats.as_dict()
            stats_dict["saved_to"] = str(out_path)

        except Exception as exc:
            error = ErrorDetail(
                code="MC-2005",
                title="Save failed",
                detail=str(exc),
                hint="Model save may have partially succeeded",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    # Output results
    result = {
        "model": str(model_path),
        "consolidation": stats.as_dict(),
        "null_space": service.get_null_space_summary(),
    }

    write_output(result, context.output_format, context.pretty)


@app.command("status")
def learn_status(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
) -> None:
    """Show null-space capacity and consolidation status for a model.

    Returns per-layer statistics on used vs available dimensions.

    Example:

        mc learn status --model /path/to/smolLM
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

    # Load model
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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

    # Get model config
    base_model = getattr(model_obj, "model", model_obj)
    config = getattr(base_model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", getattr(base_model, "n_layers", 12))
    hidden_dim = getattr(config, "hidden_size", getattr(base_model, "hidden_size", 576))

    # Create tracker to inspect null-space
    from modelcypher.core.use_cases.consolidation_service import (
        create_consolidation_service,
    )

    service = create_consolidation_service(
        model=model_obj,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    result = {
        "model": str(model_path),
        "config": {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        },
        "status": service.get_status().value,
        "null_space": service.get_null_space_summary(),
        "last_consolidation": service.get_last_stats().as_dict(),
    }

    write_output(result, context.output_format, context.pretty)


@app.command("null-space")
def learn_null_space(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    layer: int | None = typer.Option(
        None, "--layer", "-l", help="Specific layer to inspect (default: all)"
    ),
    probe_samples: int = typer.Option(
        100, "--samples", "-n", help="Number of random samples for estimation"
    ),
) -> None:
    """Analyze null-space availability in a model.

    Generates random probe activations to estimate which dimensions
    are used vs available for knowledge encoding.

    Example:

        mc learn null-space --model /path/to/smolLM
        mc learn null-space --model /path/to/smolLM --layer 16
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

    # Load model
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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

    # Get model config
    base_model = getattr(model_obj, "model", model_obj)
    config = getattr(base_model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", getattr(base_model, "n_layers", 12))
    hidden_dim = getattr(config, "hidden_size", getattr(base_model, "hidden_size", 576))

    # Create null-space tracker
    from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker

    b = get_backend()
    tracker = NullSpaceTracker(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        backend=b,
    )

    # Generate random activations to populate the tracker
    for sample_idx in range(probe_samples):
        activation = b.random_normal((hidden_dim,))
        b.eval(activation)

        if layer is not None:
            tracker.add_activation(layer, activation)
        else:
            for layer_id in range(n_layers):
                tracker.add_activation(layer_id, activation)

    # Update SVD for all layers
    tracker.update_all_layers()

    # Collect results
    if layer is not None:
        state = tracker.get_layer_state(layer)
        layer_results = [state.as_dict()]
    else:
        layer_results = [
            tracker.get_layer_state(i).as_dict()
            for i in range(n_layers)
        ]

    model_state = tracker.get_model_state()

    result = {
        "model": str(model_path),
        "config": {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "probe_samples": probe_samples,
        },
        "model_summary": model_state.as_dict(),
        "layers": layer_results if layer is None else layer_results[0],
    }

    write_output(result, context.output_format, context.pretty)


# =============================================================================
# LoRA Memory Commands (Two-Tier Memory)
# =============================================================================


@app.command("lora-status")
def lora_status(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for LoRA memory store"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
) -> None:
    """Show LoRA memory status for an agent.

    Displays buffer size, training progress, and merge history.

    Example:

        mc learn lora-status --agent agent-001 --model /path/to/smolLM
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

    from modelcypher.core.use_cases.lora_memory_service import (
        LoRAMemoryService,
    )

    service = LoRAMemoryService()

    # Get or create store to load status
    store = service.get_or_create_store(
        agent_id=agent,
        base_model_path=model_path,
    )

    status = service.status(agent)
    if status is None:
        error = ErrorDetail(
            code="MC-2010",
            title="Store not found",
            detail=f"No LoRA memory store found for agent: {agent}",
            hint="Create a store first by running with --entropy-aware",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    result = {
        "agent_id": agent,
        "status": status.to_dict(),
    }

    write_output(result, context.output_format, context.pretty)


@app.command("lora-train")
def lora_train(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for LoRA memory store"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    max_steps: int = typer.Option(
        None, "--max-steps", help="Maximum training steps (default: derived from buffer size)"
    ),
    batch_size: int = typer.Option(
        None, "--batch-size", help="Batch size per step (default: 1, algebraic minimum)"
    ),
    learning_rate: float = typer.Option(
        None, "--lr", help="Learning rate (default: derived from model geometry)"
    ),
    convergence: float = typer.Option(
        None, "--convergence", help="Loss threshold for early stopping (default: sqrt(eps))"
    ),
) -> None:
    """Train LoRA adapters from accumulated events.

    Runs training steps on the (hidden_state, delta) pairs accumulated
    during inference. This is the "dreaming" phase of two-tier memory.

    Philosophy: Hyperparameters are MEASUREMENTS, not knobs.

    When parameters are not provided, they are derived from model geometry:
    - learning_rate: sqrt(eps) / param_rms (from numerical stability)
    - batch_size: 1 (algebraic minimum, no heuristic batching)
    - convergence: sqrt(eps) (numerical precision floor)
    - max_steps: buffer_size (one pass through data)

    Example:

        mc learn lora-train --agent agent-001 --model /path/to/smolLM

    With explicit overrides:

        mc learn lora-train --agent agent-001 --model /path/to/smolLM --lr 1e-5 --max-steps 50
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

    from modelcypher.core.use_cases.lora_memory_service import (
        LoRAMemoryService,
    )

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
            hint="Run inference with --entropy-aware first to accumulate events",
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

    # Default batch_size: 1 (algebraic minimum)
    if batch_size is None:
        batch_size = 1

    # Default convergence: sqrt(eps) (numerical precision floor)
    if convergence is None:
        convergence = sqrt_eps

    # Default max_steps: one pass through buffer
    if max_steps is None:
        max_steps = max(1, store.buffer_size // batch_size)

    # Default learning_rate: derived from model geometry
    # This requires loading model to compute param RMS
    if learning_rate is None:
        # Use conservative default based on dtype, actual derivation happens in service
        learning_rate = sqrt_eps

    # Run training
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
        "parameters_used": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "convergence": convergence,
            "note": "Parameters derived from geometry when not explicitly provided",
        },
    }

    write_output(result, context.output_format, context.pretty)


@app.command("merge-lora")
def merge_lora(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for LoRA memory store"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    output: str | None = typer.Option(
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

    This is the "sleep consolidation" phase - transferring hippocampus
    (LoRA) knowledge to neocortex (base weights) via null-space projection.

    Example:

        mc learn merge-lora --agent agent-001 --model /path/to/smolLM --save --output /path/to/merged
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

    # Load model
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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
    from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker

    b = get_backend()
    tracker = NullSpaceTracker(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        backend=b,
    )

    from modelcypher.core.use_cases.lora_memory_service import (
        LoRAMemoryService,
    )

    service = LoRAMemoryService()

    # Get store
    store = service.get_or_create_store(
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


@app.command("lora-export")
def lora_export(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent ID for LoRA memory store"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output path for exported LoRA"
    ),
) -> None:
    """Export LoRA adapters to files for sharing or backup.

    Exports the trained LoRA weights and metadata to a directory.

    Example:

        mc learn lora-export --agent agent-001 --model /path/to/smolLM --output /path/to/export
    """
    context = _context(ctx)
    model_path = Path(model)
    output_path = Path(output)

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

    from modelcypher.core.use_cases.lora_memory_service import (
        LoRAMemoryService,
    )

    service = LoRAMemoryService()

    # Get store
    store = service.get_or_create_store(
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


# =============================================================================
# Memory Benchmark Commands
# =============================================================================


@app.command("benchmark")
def learn_benchmark(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    capture: bool = typer.Option(
        False, "--capture", help="Capture a new snapshot (vs compare)"
    ),
    before_file: str | None = typer.Option(
        None, "--before", "-b", help="Path to 'before' snapshot for comparison"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output path for snapshot or comparison result"
    ),
    probes: str | None = typer.Option(
        None, "--probes", "-p", help="Comma-separated probe prompts for entropy measurement"
    ),
) -> None:
    """Capture geometric snapshots and compare before/after consolidation.

    Memory effectiveness is proven by geometric comparison:
    - delta_sparsity < 0: Sparse regions became dense
    - delta_intrinsic_dim > 0: Denser manifold uses more dimensions
    - delta_eigenscore < 0: Less geometric uncertainty
    - delta_entropy < 0: More confident on uncertain prompts

    All significance thresholds derived from sqrt(eps) - machine precision.

    Examples:

        # Capture 'before' snapshot
        mc learn benchmark --model /path/to/model --capture --output before.json

        # Run consolidation (mc learn consolidate ...)

        # Capture 'after' and compare
        mc learn benchmark --model /path/to/model --before before.json --output results.json

        # With probe prompts for entropy measurement
        mc learn benchmark --model /path/to/model --capture --probes "What is France?,Who is president?" --output before.json
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

    # Validate arguments
    if not capture and not before_file:
        error = ErrorDetail(
            code="MC-2020",
            title="Invalid arguments",
            detail="Must specify either --capture or --before for comparison",
            hint="Use --capture to capture a snapshot, or --before to compare against a previous snapshot",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load model
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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

    # Parse probes
    probe_list: list[str] | None = None
    if probes:
        probe_list = [p.strip() for p in probes.split(",") if p.strip()]

    # Create benchmark service
    from modelcypher.core.use_cases.memory_benchmark import MemoryBenchmarkService

    service = MemoryBenchmarkService()

    if capture:
        # Capture snapshot
        snapshot = service.capture_snapshot(
            model=model_obj,
            probes=probe_list,
            tokenizer=tokenizer,
            model_path=str(model_path),
        )

        # Save if output specified
        if output:
            output_path = Path(output)
            service.save_snapshot(snapshot, output_path)

        result: dict[str, Any] = {
            "mode": "capture",
            "model": str(model_path),
            "snapshot": snapshot.to_dict(),
        }
        if output:
            result["saved_to"] = str(output)

        write_output(result, context.output_format, context.pretty)

    else:
        # Compare mode
        before_path = Path(before_file)  # type: ignore[arg-type]
        if not before_path.exists():
            error = ErrorDetail(
                code="MC-2021",
                title="Before file not found",
                detail=f"Before snapshot file does not exist: {before_path}",
                hint="Capture a 'before' snapshot first with --capture",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        # Load before snapshot
        before_snapshot = service.load_snapshot(before_path)

        # Capture after snapshot
        after_snapshot = service.capture_snapshot(
            model=model_obj,
            probes=probe_list,
            tokenizer=tokenizer,
            model_path=str(model_path),
        )

        # Compare
        comparison = service.compare(before_snapshot, after_snapshot)

        # Save if output specified
        if output:
            output_path = Path(output)
            service.save_comparison(comparison, output_path)

        result = {
            "mode": "compare",
            "model": str(model_path),
            "comparison": comparison.to_dict(),
        }
        if output:
            result["saved_to"] = str(output)

        write_output(result, context.output_format, context.pretty)


# =============================================================================
# Background Consolidation Monitor Commands
# =============================================================================


@app.command("monitor")
def learn_monitor(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to model directory"
    ),
    check_interval: float = typer.Option(
        30.0, "--interval", help="Seconds between condition checks"
    ),
    max_queue: int = typer.Option(
        1000, "--max-queue", help="Max sparsity events before forced consolidation"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Enable automatic consolidation when conditions met"
    ),
    status_only: bool = typer.Option(
        False, "--status", help="Show current geometric conditions and exit"
    ),
) -> None:
    """Monitor geometric conditions for background consolidation.

    Consolidation triggers are geometry-based, NOT time-based:
    - event_count >= MIN_EVENTS (max(20, hidden_dim/32))
    - mean_eigenscore > 2 * sqrt(eps) (meaningful sparsity)
    - mean_capacity_fraction > sqrt(eps) (room in model)
    - system_idle (not already consolidating)

    All thresholds derived from sqrt(eps) - machine precision.

    Examples:

        # Check current geometric conditions
        mc learn monitor --model /path/to/model --status

        # Start monitoring with auto-consolidation
        mc learn monitor --model /path/to/model --auto

        # Monitor with custom interval
        mc learn monitor --model /path/to/model --auto --interval 60
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

    # Load model to get dimensions
    try:
        from modelcypher.adapters.local_inference import load_model_and_tokenizer

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

    # Get model config
    base_model = getattr(model_obj, "model", model_obj)
    config = getattr(base_model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", getattr(base_model, "n_layers", 12))
    hidden_dim = getattr(config, "hidden_size", getattr(base_model, "hidden_size", 576))

    # Create consolidation service
    from modelcypher.core.use_cases.consolidation_service import (
        create_consolidation_service,
    )

    consolidation_service = create_consolidation_service(
        model=model_obj,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    # Create monitor
    from modelcypher.core.use_cases.background_consolidation import (
        BackgroundConsolidationMonitor,
        MonitorConfig,
    )

    monitor_config = MonitorConfig(
        check_interval=check_interval,
        max_queue_size=max_queue,
        enabled=auto,
    )
    monitor = BackgroundConsolidationMonitor(
        consolidation_service=consolidation_service,
        hidden_dim=hidden_dim,
        config=monitor_config,
    )

    if status_only:
        # Just show current conditions
        conditions = monitor.get_conditions()
        result: dict[str, Any] = {
            "model": str(model_path),
            "config": {
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "min_events_required": conditions.min_events_required,
            },
            "conditions": conditions.to_dict(),
            "status": monitor.get_status().to_dict(),
        }
        write_output(result, context.output_format, context.pretty)
        return

    # Run monitor (blocking)
    import asyncio

    async def run_monitor() -> None:
        monitor.start()
        try:
            # Run until interrupted
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await monitor.stop()

    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        pass

    # Final status
    result = {
        "model": str(model_path),
        "final_status": monitor.get_status().to_dict(),
    }
    write_output(result, context.output_format, context.pretty)
