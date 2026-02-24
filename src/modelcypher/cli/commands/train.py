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

NB-LoRA: Cayley-parameterized, geometry-derived, bounds by construction.
All hyperparameters derived from model geometry, not heuristics.

Commands:
    mc train run --model <path> --data <path.jsonl>   # Strict dataset-driven NB-LoRA training
    mc train run-research --model <path> --data <path.jsonl>   # Research controls
    mc train status --agent <id> --model <path>       # Check training status
    mc train merge --agent <id> --model <path>        # Merge LoRA to base
    mc train export --agent <id> --model <path>       # Export LoRA weights
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
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


def _write_training_derivation_error(
    exc: TrainingDerivationError,
    context: CLIContext,
) -> None:
    """Emit structured strict-derivation failure and terminate command."""
    error = ErrorDetail(
        code="MC-2014",
        title="Training derivation failed",
        detail=exc.detail,
        hint="Resolve the diagnostics and re-run training.",
        trace_id=context.trace_id,
    ).as_dict()
    error["failure_class"] = exc.failure_class
    error["diagnostics"] = exc.diagnostics or {}
    write_error(error, context.output_format, context.pretty)
    raise typer.Exit(code=1)


@train_app.callback()
def train() -> None:
    """Training command group.

    Strict production path: `mc train run`.
    Research-only controls: `mc train run-research`.
    """


@train_app.command("run")
def train_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    data: str = typer.Option(..., "--data", "-d", help="Path to JSONL training dataset"),
    output: str = typer.Option(None, "--output", "-o", help="Output path for adapter"),
    eval_data: str = typer.Option(
        None,
        "--eval-data",
        help="Held-out eval JSONL (default: pilot-variance-derived split)",
    ),
) -> None:
    """Strict model+dataset-only training path.

    User selects model + dataset. Hyperparameters are derived automatically.
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()
    try:
        result = service.train_from_dataset_strict(
            model_path=model_path,
            dataset_path=data,
            output_path=output,
            eval_dataset_path=eval_data,
        )
    except TrainingDerivationError as exc:
        _write_training_derivation_error(exc, context)

    write_output(result.to_dict(), context.output_format, context.pretty)


@train_app.command("run-research")
def train_run_research(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    data: str = typer.Option(..., "--data", "-d", help="Path to JSONL training dataset"),
    output: str = typer.Option(
        None, "--output", "-o", help="Output path for adapter",
    ),
    no_save: bool = typer.Option(
        False, "--no-save",
        help="Run training without saving an adapter (useful for experiments)",
    ),
    eval_data: str = typer.Option(
        None,
        "--eval-data",
        help="Held-out eval JSONL (default: pilot-variance-derived split)",
    ),
    seq_length: int = typer.Option(None, "--seq-length", help="Sequence length (auto-derived from data when omitted)"),
    lr: float = typer.Option(None, "--lr", help="Override geometry-derived learning rate"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    topo_monitor: bool = typer.Option(
        False,
        "--topo-monitor/--no-topo-monitor",
        help="Track topological phase metrics per epoch (Betti numbers, Ricci curvature). Slower.",
    ),
    dim_monitor: bool = typer.Option(
        False,
        "--dim-monitor/--no-dim-monitor",
        help="Track dimensional expansion/contraction per epoch using TwoNN.",
    ),
    auto_regime: bool = typer.Option(
        True,
        "--auto-regime/--no-auto-regime",
        help="Derive training objective (CE/REINFORCE/hybrid) from baseline eval (default: on)",
    ),
) -> None:
    """Research path with explicit training controls."""
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()
    try:
        result = service.train_from_dataset_research(
            model_path=model_path,
            dataset_path=data,
            output_path=output,
            eval_dataset_path=eval_data,
            seq_length=seq_length,
            lr_override=lr,
            seed=seed,
            topo_monitor=topo_monitor,
            dim_monitor=dim_monitor,
            auto_regime=auto_regime,
            no_save=no_save,
        )
    except TrainingDerivationError as exc:
        _write_training_derivation_error(exc, context)

    write_output(result.to_dict(), context.output_format, context.pretty)


@train_app.command("validate-derived")
def train_validate_derived(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    data: str = typer.Option(..., "--data", "-d", help="Path to JSONL training dataset"),
    trials: int = typer.Option(
        ...,
        "--trials",
        min=1,
        help="Number of repeated trials for counterexample search",
    ),
    eval_data: str = typer.Option(
        None,
        "--eval-data",
        help="Held-out eval JSONL (default: pilot-variance-derived split)",
    ),
    base_seed: int = typer.Option(
        None,
        "--base-seed",
        help="Optional base seed; if omitted derive from model+dataset hash",
    ),
    seq_length: int = typer.Option(
        None,
        "--seq-length",
        help="Optional explicit sequence length (otherwise auto-derived from data)",
    ),
    report_path: str = typer.Option(
        None,
        "--report-path",
        help="Optional JSON output path for full validation report",
    ),
    fail_on_counterexample: bool = typer.Option(
        True,
        "--fail-on-counterexample/--no-fail-on-counterexample",
        help="Return non-zero exit code when any trial fails improvement checks",
    ),
) -> None:
    """Run repeated derived-training validation and capture counterexamples.

    Uses derived settings for each trial and records failures where
    post-training metrics do not improve over baseline.
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.cli.composition import get_backend, get_dataset_training_service
    from modelcypher.core.use_cases.derived_training_validation_service import (
        DerivedTrainingValidationService,
    )

    validator = DerivedTrainingValidationService(
        dataset_training_service=get_dataset_training_service(),
        backend=get_backend(),
    )
    result = validator.validate(
        model_path=model_path,
        dataset_path=data,
        eval_dataset_path=eval_data,
        trials=trials,
        base_seed=base_seed,
        seq_length=seq_length,
    )
    payload = result.to_dict()

    if report_path is not None:
        output_file = Path(report_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    write_output(payload, context.output_format, context.pretty)

    if fail_on_counterexample and not result.all_passed:
        raise typer.Exit(code=1)


@train_app.command("star")
def train_star(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to base JSONL training dataset (e.g., 440-sample paired set)",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for STaR run artifacts",
    ),
    eval_data: str = typer.Option(
        None,
        "--eval-data",
        help="Optional eval JSONL for training validation loss tracking",
    ),
    eval_suite: str = typer.Option(
        None,
        "--eval-suite",
        help="Optional JSONL inference suite path (default: data/eval_prompts/nblora_inference_tests.jsonl)",
    ),
    initial_adapter: str = typer.Option(
        None,
        "--initial-adapter",
        help="Optional starting adapter path for round-1 generation",
    ),
    rounds: int = typer.Option(3, "--rounds", help="Number of STaR rounds"),
    problems_per_round: int = typer.Option(
        500,
        "--problems-per-round",
        help="Novel generated problems per round (must be >= 500)",
    ),
    strategy: str = typer.Option(
        "fresh_base",
        "--strategy",
        help=(
            "Training strategy: "
            "fresh_base (train from base on cumulative data), "
            "cumulative_adapter (initialize from prior adapter and train on new data)"
        ),
    ),
    few_shot_examples: int = typer.Option(
        3,
        "--few-shot-examples",
        min=2,
        max=3,
        help="Few-shot demonstrations in generation prompts (2 or 3)",
    ),
    max_generation_tokens: int = typer.Option(
        None,
        "--max-generation-tokens",
        help="Optional generation cap. Omit to use backend default.",
    ),
    seed: int = typer.Option(42, "--seed", help="Base seed (all round seeds derive from this)"),
) -> None:
    """Run STaR (generate → verify → retrain) with geometric diagnostics.

    Uses DatasetTrainingService for each round's retraining step.
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    from modelcypher.cli.composition import get_star_training_service
    from modelcypher.core.use_cases.star_training_service import (
        STRATEGY_CUMULATIVE_ADAPTER,
        STRATEGY_FRESH_BASE,
    )

    strategy_normalized = strategy.strip().lower()
    if strategy_normalized not in {STRATEGY_FRESH_BASE, STRATEGY_CUMULATIVE_ADAPTER}:
        error = ErrorDetail(
            code="MC-2013",
            title="Invalid STaR strategy",
            detail=(
                f"Unknown strategy: {strategy}. "
                f"Use {STRATEGY_FRESH_BASE} or {STRATEGY_CUMULATIVE_ADAPTER}."
            ),
            hint="Re-run with --strategy fresh_base or --strategy cumulative_adapter",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    service = get_star_training_service()
    try:
        result = service.run(
            model_path=model_path,
            base_dataset_path=Path(data),
            output_root=Path(output),
            eval_dataset_path=Path(eval_data) if eval_data else None,
            eval_suite_path=Path(eval_suite) if eval_suite else None,
            initial_adapter_path=Path(initial_adapter) if initial_adapter else None,
            rounds=rounds,
            problems_per_round=problems_per_round,
            seed=seed,
            few_shot_examples=few_shot_examples,
            max_generation_tokens=max_generation_tokens,
            training_strategy=strategy_normalized,
        )
    except TrainingDerivationError as exc:
        _write_training_derivation_error(exc, context)

    write_output(result.to_dict(), context.output_format, context.pretty)


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

    from modelcypher.cli.composition import get_lora_memory_service

    service = get_lora_memory_service()

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

    from modelcypher.cli.composition import get_lora_memory_service
    service = get_lora_memory_service()
    tracker = service.create_null_space_tracker(model_obj)

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

    from modelcypher.cli.composition import get_lora_memory_service

    service = get_lora_memory_service()

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
