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

"""Training CLI.

NB-LoRA: Cayley-parameterized, geometry-derived, bounds by construction.
All training controls on the canonical path are derived from model geometry or
measured data; optional flags add instrumentation only.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.exit_codes import EXIT_INPUT, EXIT_RUNTIME
from modelcypher.cli.output import write_agent_output, write_error, write_output
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
    write_error(error, context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
    raise typer.Exit(code=EXIT_RUNTIME)


def _format_training_result_text(payload: dict[str, object]) -> str:
    """Render a concise human summary for text-mode training output."""
    lines = ["Training result"]
    derived_plan = payload.get("derived_plan")
    if isinstance(derived_plan, dict):
        data_plan = derived_plan.get("data_plan")
        if isinstance(data_plan, dict):
            lines.append(
                "Data: split="
                f"{data_plan.get('split_method', 'unknown')} | "
                f"train={data_plan.get('n_train', '?')} "
                f"eval={data_plan.get('n_eval', '?')} | "
                f"seq_length={data_plan.get('seq_length', '?')}"
            )
        adaptation_surface = derived_plan.get("adaptation_surface")
        if isinstance(adaptation_surface, dict):
            rank_range = adaptation_surface.get("rank_range", [0, 0])
            if not isinstance(rank_range, list) or len(rank_range) != 2:
                rank_range = [0, 0]
            lines.append(
                "Resolved surface: "
                f"{adaptation_surface.get('target_module_count', '?')} modules | "
                f"ranks={rank_range[0]}-{rank_range[1]} | "
                f"params~{adaptation_surface.get('estimated_trainable_params', '?')}"
            )
    lines.append(
        "Losses: "
        f"baseline={payload.get('baseline_loss', '?')} | "
        f"train_final={payload.get('final_loss', '?')} | "
        f"post_eval={payload.get('post_loss', '?')}"
    )
    lines.append(
        "Verification: "
        f"spectral_bounds_ok={payload.get('spectral_bounds_ok', '?')} | "
        f"min_cka={payload.get('min_cka', 'n/a')} | "
        f"pipeline_gate_passed={payload.get('pipeline_gate_passed', '?')}"
    )
    benchmark_delta = payload.get("benchmark_delta")
    if isinstance(benchmark_delta, dict) and benchmark_delta:
        overall_delta = benchmark_delta.get("overall")
        if isinstance(overall_delta, (int, float)):
            lines.append(f"Benchmark delta: overall={overall_delta:+.4f}")
        else:
            lines.append("Benchmark delta: available in structured output")
    adapter_path = payload.get("adapter_path")
    if adapter_path:
        lines.append(f"Adapter: {adapter_path}")
    else:
        lines.append("Adapter: not saved")
    return "\n".join(lines)


@train_app.callback()
def train() -> None:
    """Training command group.

    One training path: `mc train run`.
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
    benchmark: str = typer.Option(
        None,
        "--benchmark",
        help="Run benchmark suite before/after training (quick, reasoning, factual, comprehensive)",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Run training without saving an adapter",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Show the resolved training plan before training, then continue.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Derive and print the exact training plan without injecting adapters or training.",
    ),
    seq_length: int = typer.Option(
        None,
        "--seq-length",
        help="Sequence length (auto-derived from data when omitted)",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        help="Optional seed override (default: derived from model+dataset hash)",
    ),
    topo_monitor: bool = typer.Option(
        False,
        "--topo-monitor/--no-topo-monitor",
        help="Track topological phase metrics per epoch (slower)",
    ),
    dim_monitor: bool = typer.Option(
        False,
        "--dim-monitor/--no-dim-monitor",
        help="Track dimensional expansion/contraction per epoch",
    ),
    target_experts: list[str] | None = typer.Option(
        None,
        "--target-experts",
        help='MoE expert selectors (repeatable or comma-separated), e.g. "L5.E42".',
    ),
    entropy_reg: bool = typer.Option(
        False,
        "--entropy-reg/--no-entropy-reg",
        help="Enable entropy floor regularization during CE training (prevents overconfident logits)",
    ),
) -> None:
    """Canonical NB-LoRA training path.

    Trains an NB-LoRA adapter using Cayley-Stiefel optimization with all
    hyperparameters derived from model geometry by default. Optional flags
    expose instrumentation on the same path.

    Output fields (when --json):
        train_iters: Number of training iterations completed
        final_loss: Final training loss
        adapter_path: Path to saved adapter weights
        derived_plan: Exact resolved preflight plan used by the run
        benchmark_baseline: Pre-training benchmark scores (with --benchmark)
        benchmark_post: Post-training benchmark scores (with --benchmark)
        benchmark_delta: Score deltas (with --benchmark)

    Example:
        mc train run --model /path/to/model --data /path/to/data.jsonl
        mc train run -m /path/to/model -d /path/to/data.jsonl --explain --benchmark quick
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)
    if explain and plan_only:
        raise typer.BadParameter("--explain and --plan-only cannot be used together")

    from modelcypher.cli.composition import get_dataset_training_service

    import sys

    from modelcypher.cli.progress import ProgressReporter

    reporter = None
    if context.ai_mode or not sys.stderr.isatty():
        reporter = ProgressReporter()

    service = get_dataset_training_service()
    service._progress_reporter = reporter
    plan = None
    try:
        if explain or plan_only:
            plan = service.build_training_plan(
                model_path=model_path,
                dataset_path=data,
                output_path=output,
                eval_dataset_path=eval_data,
                seq_length=seq_length,
                seed=seed,
                no_save=no_save,
                target_experts=target_experts,
            )
            if plan_only:
                payload = {"derived_plan": plan.to_dict()}
                if context.output_format == "text":
                    write_output(plan.to_text_summary(), context.output_format, context.pretty)
                else:
                    write_output(payload, context.output_format, context.pretty)
                return
            if context.output_format == "text":
                typer.echo(plan.to_text_summary())
                typer.echo("")

        train_kwargs = {
            "model_path": model_path,
            "dataset_path": data,
            "output_path": output,
            "eval_dataset_path": eval_data,
            "seq_length": seq_length,
            "seed": seed,
            "topo_monitor": topo_monitor,
            "dim_monitor": dim_monitor,
            "no_save": no_save,
            "benchmark_suite": benchmark,
            "target_experts": target_experts,
            "entropy_regularization": entropy_reg,
        }
        if plan is not None:
            train_kwargs["plan"] = plan
        result = service.train_from_dataset(**train_kwargs)
    except TrainingDerivationError as exc:
        _write_training_derivation_error(exc, context)

    payload = result.to_dict()

    # Wrap in AgentEnvelope for structured agent-readable output
    from modelcypher.core.domain.agent_protocol import (
        AgentEnvelope,
        derived_eval_hash,
        file_hash,
        make_metadata,
        model_id,
    )
    from modelcypher.core.domain.training.diagnostics import (
        diagnose_training_result,
    )

    diagnostics = diagnose_training_result(
        payload,
        model_path=str(model_path),
        adapter_path=result.adapter_path,
    )

    # Compute eval data hash: explicit file hash if --eval-data was provided,
    # otherwise derive a stable identity from the auto-split parameters.
    # Use the resolved seed from the training plan (not the CLI arg, which is
    # None when the seed is auto-derived from model+dataset hash).
    derived_eval_hash_val: str | None = None
    if eval_data is None and result.validation_split:
        vs = result.validation_split
        n_eval = vs.get("n_eval")
        data_hash_val = file_hash(data) if data else None
        resolved_seed = (
            result.derived_plan.get("seed")
            if result.derived_plan
            else seed
        )
        if data_hash_val and n_eval is not None and resolved_seed is not None:
            derived_eval_hash_val = derived_eval_hash(
                data_hash_val, int(resolved_seed), int(n_eval),
            )

    gate_passed = result.pipeline_gate_passed
    status = "success" if gate_passed is not False else "partial"
    envelope = AgentEnvelope(
        command="mc train run",
        status=status,
        result=payload,
        diagnostics=diagnostics,
        metadata=make_metadata(
            model=str(model_path),
            adapter_path=result.adapter_path,
            duration_seconds=result.training_time_seconds,
            seed=seed,
            model_id_value=model_id(str(model_path)),
            data_path=data,
            eval_data_path=eval_data,
            eval_data_hash=derived_eval_hash_val,
            benchmark_suite=benchmark,
        ),
    )

    if context.ai_mode or context.output_format != "text":
        write_agent_output(envelope, context.output_format, context.pretty)
    else:
        text_result = _format_training_result_text(payload)
        write_agent_output(
            envelope, context.output_format, context.pretty, text_result=text_result,
        )


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

    Runs N independent training trials with geometry-derived settings, then
    checks whether post-training metrics improve over baseline. Any trial
    that fails improvement checks is recorded as a counterexample. Use this
    to validate that derived hyperparameters consistently improve outcomes.

    Output fields (when --json):
        trials: Number of trials executed
        passed: Number of trials that improved over baseline
        failed: Number of counterexample trials
        allPassed: Whether every trial passed
        counterexamples: List of failed trial details with diagnostics
        summary: Aggregate statistics across all trials

    Example:
        mc train validate-derived -m /path/to/model -d /path/to/data.jsonl --trials 5
        mc train validate-derived -m /path/to/model -d /path/to/data.jsonl --trials 10 --report-path report.json
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
    """Run STaR (generate, verify, retrain) with geometric diagnostics.

    Self-Taught Reasoner: generates novel problems, verifies solutions via
    execution, then retrains on verified correct traces. Each round produces
    new training data from the model's own reasoning. Uses DatasetTrainingService
    for each round's retraining step.

    Output fields (when --json):
        rounds: List of per-round results (generated, verified, retrained)
        totalGenerated: Total novel problems generated across all rounds
        totalVerified: Total verified-correct solutions
        finalAdapterPath: Path to final adapter after all rounds
        strategy: Training strategy used (fresh_base or cumulative_adapter)

    Example:
        mc train star -m /path/to/model -d /path/to/data.jsonl -o /path/to/output --rounds 3
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
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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


@train_app.command("evaluate")
def train_evaluate(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    adapter: str = typer.Option(
        None, "--adapter", "-a", help="Path to LoRA adapter directory"
    ),
    prompts: str = typer.Option(
        None,
        "--prompts",
        help='JSONL with {"prompt": "...", "reference": "..."} for inference comparison',
    ),
    data: str = typer.Option(
        None, "--data", "-d", help="JSONL dataset for loss/perplexity evaluation"
    ),
    benchmark: str = typer.Option(
        None,
        "--benchmark",
        help="lm-eval benchmark suite (quick, reasoning, factual, comprehensive)",
    ),
    max_tokens: int = typer.Option(
        256, "--max-tokens", help="Max tokens for inference generation"
    ),
) -> None:
    """Evaluate a trained adapter against base model.

    Three evaluation modes (specify exactly one):
      --prompts: Inference comparison — generate with base vs adapted, compare per-prompt
      --data: Loss evaluation — compute loss/perplexity on a dataset
      --benchmark: Benchmark suite — run lm-eval pre/post

    Output fields (when --json):
        mode: Evaluation mode used
        overall_verdict: "improved", "degraded", "neutral", or "degenerated"
        n_prompts: Number of prompts evaluated (inference mode)
        n_improved/n_degraded/n_degenerated: Per-prompt counts (inference mode)
        base_loss/adapted_loss: Loss values (loss mode)
        benchmark_results: Benchmark scores (benchmark mode)

    Examples:
        mc train evaluate -m /path/to/model -a /path/to/adapter --prompts eval.jsonl
        mc train evaluate -m /path/to/model -a /path/to/adapter -d val.jsonl
        mc train evaluate -m /path/to/model --benchmark quick
    """
    context = _context(ctx)
    model_path = Path(model)
    _validate_model_path(model_path, context)

    n_modes = sum(1 for x in [prompts, data, benchmark] if x is not None)
    if n_modes != 1:
        error = ErrorDetail(
            code="MC-2015",
            title="Invalid evaluation mode",
            detail="Specify exactly one of --prompts, --data, or --benchmark",
            hint="Use --prompts for inference, --data for loss, or --benchmark for lm-eval",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)

    if prompts is not None and adapter is None:
        error = ErrorDetail(
            code="MC-2019",
            title="Adapter required for prompt comparison",
            detail="--prompts mode compares base vs adapted output. Provide --adapter.",
            hint="mc train evaluate -m /model -a /adapter --prompts eval.jsonl",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)

    from modelcypher.cli.composition import get_backend
    from modelcypher.core.use_cases.standalone_evaluation_service import (
        StandaloneEvaluationService,
    )

    service = StandaloneEvaluationService(backend=get_backend())

    try:
        result = service.evaluate(
            model_path=model_path,
            adapter_path=Path(adapter) if adapter else None,
            prompts_path=Path(prompts) if prompts else None,
            data_path=Path(data) if data else None,
            benchmark_suite=benchmark,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-2016",
            title="Evaluation failed",
            detail=str(exc),
            hint="Check model path, adapter path, and evaluation data",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    from modelcypher.core.domain.agent_protocol import model_id as _model_id

    envelope = service.make_envelope(
        result,
        model_id_value=_model_id(str(model_path)),
        eval_data_path=data if data else (prompts if prompts else None),
        benchmark_suite=benchmark,
    )
    write_agent_output(envelope, context.output_format, context.pretty)


@train_app.command("compare")
def train_compare(
    ctx: typer.Context,
    model: str = typer.Option(
        None, "--model", "-m", help="Path to model (required for adapter comparison)"
    ),
    adapter_a: str = typer.Option(
        None, "--adapter-a", help="First adapter path"
    ),
    adapter_b: str = typer.Option(
        None, "--adapter-b", help="Second adapter path"
    ),
    result_a: str = typer.Option(
        None, "--result-a", help="First training result JSON path"
    ),
    result_b: str = typer.Option(
        None, "--result-b", help="Second training result JSON path"
    ),
    data: str = typer.Option(
        None, "--data", "-d", help="JSONL dataset for adapter comparison"
    ),
) -> None:
    """Compare two training runs or adapters side-by-side.

    Two comparison modes:
      --result-a/--result-b: Compare saved training result JSON files
      --adapter-a/--adapter-b with --model: Evaluate both adapters and compare

    Output fields (when --json):
        label_a/label_b: Labels for each run
        metrics: Per-metric comparison with delta and winner
        winner: Overall winner ("a", "b", or null if inconclusive)
        winner_reason: Explanation of winner determination

    Examples:
        mc train compare --result-a run1.json --result-b run2.json
        mc train compare -m /path/to/model --adapter-a /a1 --adapter-b /a2 -d val.jsonl
    """
    context = _context(ctx)

    from modelcypher.core.use_cases.training_comparison_service import (
        TrainingComparisonService,
    )

    service = TrainingComparisonService()

    try:
        if result_a and result_b:
            result = service.compare_results(
                Path(result_a), Path(result_b),
            )
        elif adapter_a and adapter_b and model:
            from modelcypher.cli.composition import get_backend

            result = service.compare_adapters(
                model_path=Path(model),
                adapter_a_path=Path(adapter_a),
                adapter_b_path=Path(adapter_b),
                data_path=Path(data) if data else None,
                backend=get_backend(),
            )
        else:
            error = ErrorDetail(
                code="MC-2017",
                title="Invalid comparison mode",
                detail="Provide either --result-a/--result-b or --model/--adapter-a/--adapter-b",
                hint="Compare result files: --result-a r1.json --result-b r2.json",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
            raise typer.Exit(code=EXIT_INPUT)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-2018",
            title="Comparison failed",
            detail=str(exc),
            hint="Check file paths and data format",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    envelope = service.make_envelope(result)
    write_agent_output(envelope, context.output_format, context.pretty)
