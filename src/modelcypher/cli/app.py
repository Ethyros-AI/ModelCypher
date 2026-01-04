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

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Suppress noisy third-party initialization warnings
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import typer
from typer.core import TyperGroup

from modelcypher.cli.typer_compat import apply_typer_compat

apply_typer_compat()

from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories

register_default_atlas_inventories()

from modelcypher.infrastructure.inference_engine_factory import get_inference_engine
from modelcypher.cli.commands import adapter as adapter_commands
from modelcypher.cli.commands import agent as agent_commands
from modelcypher.cli.commands import agent_eval as agent_eval_commands
from modelcypher.cli.commands import dashboard as dashboard_commands
from modelcypher.cli.commands import entropy as entropy_commands
from modelcypher.cli.commands import eval as eval_commands
from modelcypher.cli.commands import help_cmd as help_commands
from modelcypher.cli.commands import infer as infer_commands
from modelcypher.cli.commands import job as job_commands
from modelcypher.cli.commands import merge as merge_commands
from modelcypher.cli.commands import model as model_commands
from modelcypher.cli.commands import profile as profile_commands
from modelcypher.cli.commands import research as research_commands
from modelcypher.cli.commands import safety as safety_commands
from modelcypher.cli.commands import stability as stability_commands
from modelcypher.cli.commands import storage as storage_commands
from modelcypher.cli.commands import system as system_commands
from modelcypher.cli.commands import thermo as thermo_commands
from modelcypher.cli.commands import train as train_commands
from modelcypher.cli.commands.geometry import atlas as geometry_atlas_commands
from modelcypher.cli.commands.geometry import baseline as geometry_baseline_commands
from modelcypher.cli.commands.geometry import concept as geometry_concept_commands
from modelcypher.cli.commands.geometry import crm as geometry_crm_commands
from modelcypher.cli.commands.geometry import cross_cultural as geometry_cross_cultural_commands
from modelcypher.cli.commands.geometry import density as geometry_density_commands
from modelcypher.cli.commands.geometry import emotion as geometry_emotion_commands
from modelcypher.cli.commands.geometry import geom_adapter as geometry_adapter_commands
from modelcypher.cli.commands.geometry import invariant as geometry_invariant_commands
from modelcypher.cli.commands.geometry import manifold as geometry_manifold_commands
from modelcypher.cli.commands.geometry import metaphor as geometry_metaphor_commands
from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
from modelcypher.cli.commands.geometry import moral as geometry_moral_commands
from modelcypher.cli.commands.geometry import number_theory as geometry_number_theory_commands
from modelcypher.cli.commands.geometry import path as geometry_path_commands
from modelcypher.cli.commands.geometry import persona as geometry_persona_commands
from modelcypher.cli.commands.geometry import primes as geometry_primes_commands
from modelcypher.cli.commands.geometry import refusal as geometry_refusal_commands
from modelcypher.cli.commands.geometry import research as geometry_research_commands
from modelcypher.cli.commands.geometry import safety as geometry_safety_commands
from modelcypher.cli.commands.geometry import social as geometry_social_commands
from modelcypher.cli.commands.geometry import sparse as geometry_sparse_commands
from modelcypher.cli.commands.geometry import spatial as geometry_spatial_commands
from modelcypher.cli.commands.geometry import temporal as geometry_temporal_commands
from modelcypher.cli.commands.geometry import training as geometry_training_commands
from modelcypher.cli.commands.geometry import transfer as geometry_transfer_cabe_commands
from modelcypher.cli.commands.geometry import visualize as geometry_visualize_commands
from modelcypher.cli.commands.geometry import waypoint as geometry_waypoint_commands
from modelcypher.cli.commands.geometry import interference as geometry_interference_commands
from modelcypher.cli.commands.geometry import merge_entropy as geometry_merge_entropy_commands
from modelcypher.cli.commands.geometry import refinement as geometry_refinement_commands
from modelcypher.cli.composition import get_training_service
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json
from modelcypher.utils.logging import configure_logging

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
validate_app = typer.Typer(no_args_is_help=True, help="Validation utilities")
estimate_app = typer.Typer(no_args_is_help=True, help="Training resource estimates")
geometry_app = typer.Typer(no_args_is_help=True, help="Geometry analysis commands")

# Hidden dev group for diagnostic/internal commands
dev_app = typer.Typer(no_args_is_help=True, hidden=True)

app.add_typer(train_commands.train_app, name="train", help="Training jobs and lifecycle")
app.add_typer(job_commands.app, name="job", help="Job management")
app.add_typer(train_commands.checkpoint_app, name="checkpoint", help="Checkpoint management")
app.add_typer(merge_commands.app, name="merge", help="Model merge pipeline")
app.add_typer(model_commands.app, name="model", help="Model registry and validation")
app.add_typer(system_commands.app, name="system", help="System probes and benchmarks")
app.add_typer(eval_commands.eval_app, name="eval", help="Evaluation runs")
app.add_typer(eval_commands.compare_app, name="compare", help="Checkpoint comparisons")
app.add_typer(validate_app, name="validate", help="Validation utilities")
app.add_typer(estimate_app, name="estimate", help="Training resource estimates")
app.add_typer(geometry_app, name="geometry", help="Geometry analysis commands")
geometry_app.add_typer(geometry_path_commands.app, name="path", help="Path geometry detection")
geometry_app.add_typer(geometry_training_commands.app, name="training", help="Training geometry metrics")
geometry_app.add_typer(geometry_safety_commands.app, name="safety", help="Safety geometry probes")
geometry_app.add_typer(geometry_adapter_commands.app, name="adapter", help="Adapter geometry analysis")
geometry_app.add_typer(geometry_atlas_commands.app, name="atlas", help="Atlas dimensionality studies")
geometry_app.add_typer(geometry_baseline_commands.app, name="baseline", help="Geometry baseline profiles")
geometry_app.add_typer(geometry_primes_commands.app, name="primes", help="Semantic primes analysis")
geometry_app.add_typer(geometry_crm_commands.app, name="crm", help="Concept Response Matrix")
geometry_app.add_typer(geometry_metrics_commands.app, name="metrics", help="Point-cloud geometry metrics")
geometry_app.add_typer(geometry_concept_commands.app, name="concept", help="Concept detection and comparison")
geometry_app.add_typer(geometry_cross_cultural_commands.app, name="cross-cultural", help="Cross-cultural analysis")
geometry_app.add_typer(geometry_density_commands.app, name="density", help="Knowledge density profiling")
geometry_app.add_typer(geometry_sparse_commands.app, name="sparse", help="Sparse domain analysis")
geometry_app.add_typer(geometry_refusal_commands.app, name="refusal", help="Refusal pair analysis")
geometry_app.add_typer(geometry_persona_commands.app, name="persona", help="Persona extraction")
geometry_app.add_typer(geometry_manifold_commands.app, name="manifold", help="Manifold operations")
geometry_app.add_typer(geometry_metaphor_commands.app, name="metaphor", help="Metaphor detection")
geometry_app.add_typer(geometry_invariant_commands.app, name="invariant", help="Invariant analysis")
geometry_app.add_typer(geometry_emotion_commands.app, name="emotion", help="Emotion geometry")
geometry_app.add_typer(geometry_transfer_cabe_commands.app, name="transfer", help="Transfer geometry")
geometry_app.add_typer(geometry_spatial_commands.app, name="spatial", help="Spatial geometry")
geometry_app.add_typer(geometry_social_commands.app, name="social", help="Social geometry")
geometry_app.add_typer(geometry_temporal_commands.app, name="temporal", help="Temporal geometry")
geometry_app.add_typer(geometry_moral_commands.app, name="moral", help="Moral geometry")
geometry_app.add_typer(geometry_number_theory_commands.app, name="number-theory", help="Number theory geometry")
geometry_app.add_typer(geometry_research_commands.app, name="research", help="Research-only geometry commands")
geometry_app.add_typer(geometry_visualize_commands.app, name="visualize", help="Geometry visualization")
geometry_app.add_typer(geometry_waypoint_commands.app, name="waypoint", help="Merge guidance waypoints")
geometry_app.add_typer(geometry_interference_commands.app, name="interference", help="Interference prediction")
geometry_app.add_typer(geometry_merge_entropy_commands.app, name="merge-entropy", help="Entropy merge validation")
geometry_app.add_typer(geometry_refinement_commands.app, name="refinement", help="Refinement analysis")
app.add_typer(entropy_commands.app, name="entropy", help="Entropy analytics")
app.add_typer(adapter_commands.adapter_app, name="adapter", help="Adapter inspection and projection")
app.add_typer(adapter_commands.calibration_app, name="calibration", help="Calibration runs")
app.add_typer(thermo_commands.app, name="thermo", help="Thermodynamics metrics")
app.add_typer(safety_commands.app, name="safety", help="Safety probes")
app.add_typer(agent_commands.app, name="agent", help="Agent trace tools")
app.add_typer(stability_commands.app, name="stability", help="Stability suites")
app.add_typer(dashboard_commands.app, name="dashboard", help="Metrics export")
app.add_typer(storage_commands.app, name="storage", help="Storage status and cleanup")
app.add_typer(infer_commands.app, name="infer", help="Inference commands")
app.add_typer(help_commands.app, name="help", help="Contextual help and schemas")
app.add_typer(profile_commands.app, name="profile", help="Unified model profile operations")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    ai: bool | None = typer.Option(
        None, "--ai", help="AI mode: force JSON output, suppress prompts/logs (MC_AI_MODE=1)"
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output format: json, yaml, text (AI defaults to json)"
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress info logs (stderr)"),
    very_quiet: bool = typer.Option(False, "--very-quiet", help="Suppress all logs (stderr)"),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm prompts"),
    no_prompt: bool = typer.Option(False, "--no-prompt", help="Fail if confirmation required"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print structured output"),
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level: trace, debug, info, warn, error"
    ),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Trace ID for diagnostics"),
) -> None:
    ai_mode = resolve_ai_mode(ai)
    output_format = resolve_output_format(ai_mode, output)
    quiet_mode = very_quiet or quiet or ai_mode
    effective_log_level = "error" if very_quiet else log_level
    configure_logging(effective_log_level, quiet=quiet_mode)

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
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
        return

    from modelcypher.backends import detect_default_backend_type, get_backend
    from modelcypher.backends.lazy_backend import LazyBackend
    from modelcypher.core.domain._backend import set_default_backend

    set_default_backend(LazyBackend(lambda: get_backend(detect_default_backend_type())))

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
    """List registered services and inventories."""
    context = _context(ctx)
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory

    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)
    service = factory.inventory_service()
    write_output(service.inventory(), context.output_format, context.pretty)


@app.command("explain")
def explain(ctx: typer.Context, command: str = typer.Argument(...)) -> None:
    """Explain a CLI command and return structured guidance."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()
    payload = service.explain(command)
    write_output(payload, context.output_format, context.pretty)


@validate_app.command("train")
def validate_train(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    output_path: str = typer.Option(..., "--out"),
    learning_rate: float = typer.Option(..., "--learning-rate"),
    batch_size: int = typer.Option(..., "--batch-size"),
    sequence_length: int = typer.Option(..., "--sequence-length"),
    epochs: int = typer.Option(..., "--epochs"),
    grad_accum: int = typer.Option(..., "--grad-accum"),
    warmup_steps: int = typer.Option(..., "--warmup-steps"),
    weight_decay: float = typer.Option(..., "--weight-decay"),
    gradient_checkpointing: bool = typer.Option(
        ..., "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    mixed_precision: bool = typer.Option(..., "--mixed-precision/--no-mixed-precision"),
    compute_precision: str = typer.Option(..., "--compute-precision"),
    optimizer_type: str = typer.Option(..., "--optimizer-type"),
    seed: int = typer.Option(..., "--seed"),
    deterministic: bool = typer.Option(..., "--deterministic/--stochastic"),
) -> None:
    """Validate a training configuration without launching a job."""
    context = _context(ctx)
    service = get_training_service()
    from modelcypher.core.domain.training import (
        ComputePrecision,
        Hyperparameters,
        TrainingSpec,
    )

    try:
        precision = ComputePrecision(compute_precision)
    except ValueError as exc:
        valid = ", ".join(m.value for m in ComputePrecision)
        raise typer.BadParameter(
            f"Invalid compute-precision '{compute_precision}'. Valid options: {valid}"
        ) from exc
    if optimizer_type != "adamw":
        raise typer.BadParameter("optimizer-type must be adamw")
    hyperparams = Hyperparameters(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        sequence_length=sequence_length,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        mixed_precision=mixed_precision,
        compute_precision=precision,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed,
        deterministic=deterministic,
        optimizer_type=optimizer_type,
    )
    config = TrainingSpec(
        model_id=model,
        dataset_path=dataset,
        output_path=output_path,
        hyperparameters=hyperparams,
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
            "gradientAccumulationSteps": grad_accum,
            "gradientCheckpointing": gradient_checkpointing,
            "mixedPrecision": mixed_precision,
            "computePrecision": precision.value,
            "warmupSteps": warmup_steps,
            "weightDecay": weight_decay,
            "seed": seed,
            "deterministic": deterministic,
            "optimizerType": optimizer_type,
        },
        "warnings": [],
        "errors": [] if result["canProceed"] else ["Configuration may not fit in memory"],
    }
    write_output(payload, context.output_format, context.pretty)


@validate_app.command("suite")
def validate_suite(
    ctx: typer.Context,
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save results (default: temp directory)",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        "-c",
        help="Run only specific category (A=introspection, B=geometry, D=safety, G=inference)",
    ),
    model_filter: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Run only on specific model (M1, M2, M3, M4)",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Timeout in seconds per command",
    ),
) -> None:
    """Run comprehensive CLI validation suite.

    Tests CLI commands against multiple models and generates a validation report.
    Model paths are configured in the test definitions and can be rooted via
    MODELCYPHER_VALIDATE_ROOT.

    Categories:
        A: Model introspection (probe, vocab-compare, validate-merge)
        B: Geometry metrics (gromov-wasserstein, intrinsic-dimension, spatial)
        D: Safety & entropy (circuit-breaker, thermo)
        G: Inference (run prompts)

    Examples:
        mc validate suite
        mc validate suite --category B
        mc validate suite --model M1 --output-dir /path/to/results
    """
    import json
    import os
    import shlex
    import subprocess
    import tempfile
    from dataclasses import dataclass, field
    from datetime import datetime

    context = _context(ctx)

    @dataclass
    class TestResult:
        test_id: str
        category: str
        command: str
        model: str | None
        status: str
        output: dict | None
        error: str | None
        duration_seconds: float
        timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model definitions (root configurable via MODELCYPHER_VALIDATE_ROOT)
    models_root = Path(
        os.environ.get("MODELCYPHER_VALIDATE_ROOT", "~/.modelcypher/models")
    ).expanduser()
    MODELS = {
        "M1": str(models_root / "mlx-community" / "Qwen2.5-0.5B-Instruct-bf16"),
        "M2": str(models_root / "mlx-community" / "Qwen2.5-3B-Instruct-bf16"),
        "M3": str(models_root / "mlx-community" / "Qwen2.5-Coder-3B-Instruct-bf16"),
        "M4": str(models_root / "mlx-community" / "Mistral-7B-Instruct-v0.3-4bit"),
    }

    # Test definitions
    TESTS = {
        "A": {
            "A1": {"name": "Model Probe", "command": "poetry run mc model probe {model}", "per_model": True},
            "A2": {"name": "Vocab Compare Same Family", "command": "poetry run mc model vocab-compare {M1} {M2}", "per_model": False},
            "A3": {"name": "Vocab Compare Cross Family", "command": "poetry run mc model vocab-compare {M1} {M4}", "per_model": False},
        },
        "B": {
            "B1": {"name": "Gromov-Wasserstein", "command": "poetry run mc geometry metrics gromov-wasserstein {M1} {M2}", "per_model": False},
            "B2": {"name": "Intrinsic Dimension", "command": "poetry run mc geometry metrics intrinsic-dimension {model}", "per_model": True},
            "B4": {"name": "Spatial Probe", "command": "poetry run mc geometry spatial probe-model {model}", "per_model": True},
        },
        "D": {
            "D1": {"name": "Circuit Breaker", "command": 'poetry run mc geometry safety circuit-breaker --model {model} --prompt "Hello"', "per_model": True},
            "D2": {"name": "Thermo Measure", "command": 'poetry run mc thermo measure --model {model} --prompt "Hello"', "per_model": True},
        },
        "G": {
            "G1": {"name": "Basic Math", "command": 'poetry run mc infer run --model {model} --prompt "What is 2+2?"', "per_model": True},
        },
    }

    def run_cli_command(cmd: str, cmd_timeout: int) -> tuple[dict | None, str | None, float]:
        import time
        start = time.time()
        try:
            if "--output" not in cmd and "--ai" not in cmd:
                cmd = cmd + " --ai"
            # Use shlex.split for safe command parsing (no shell=True)
            cmd_parts = shlex.split(cmd)
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=cmd_timeout)
            duration = time.time() - start
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout), None, duration
                except json.JSONDecodeError:
                    return {"raw": result.stdout}, None, duration
            return None, result.stderr or result.stdout, duration
        except subprocess.TimeoutExpired:
            return None, f"Timeout after {cmd_timeout}s", time.time() - start
        except Exception as e:
            return None, str(e), time.time() - start

    # Determine output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(tempfile.mkdtemp(prefix="mc-validate-"))
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[TestResult] = []
    categories = [category] if category else list(TESTS.keys())
    models_filter = [model_filter] if model_filter else None

    # Check model availability
    available_models = {k: v for k, v in MODELS.items() if Path(v).exists()}

    for cat in categories:
        if cat not in TESTS:
            continue
        cat_dir = out_path / cat.lower()
        cat_dir.mkdir(exist_ok=True)

        for test_id, test_def in TESTS[cat].items():
            if test_def.get("per_model", False):
                for model_id, model_path in MODELS.items():
                    if models_filter and model_id not in models_filter:
                        continue
                    if model_id not in available_models:
                        results.append(TestResult(
                            test_id=f"{test_id}_{model_id}",
                            category=cat,
                            command="",
                            model=model_id,
                            status="skip",
                            output=None,
                            error="Model not found",
                            duration_seconds=0,
                        ))
                        continue

                    cmd = test_def["command"].format(model=model_path)
                    output, error, duration = run_cli_command(cmd, timeout)
                    status = "pass" if output else "error"

                    results.append(TestResult(
                        test_id=f"{test_id}_{model_id}",
                        category=cat,
                        command=cmd,
                        model=model_id,
                        status=status,
                        output=output,
                        error=error,
                        duration_seconds=duration,
                    ))
            else:
                cmd = test_def["command"]
                for key, path in MODELS.items():
                    cmd = cmd.replace(f"{{{key}}}", path)

                output, error, duration = run_cli_command(cmd, timeout)
                status = "pass" if output else "error"

                results.append(TestResult(
                    test_id=test_id,
                    category=cat,
                    command=cmd,
                    model=None,
                    status=status,
                    output=output,
                    error=error,
                    duration_seconds=duration,
                ))

    # Generate summary
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skip")
    total_duration = sum(r.duration_seconds for r in results)

    summary = {
        "_schema": "mc.validate.suite.v1",
        "outputDir": str(out_path),
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "passRate": round(passed / total * 100, 1) if total > 0 else 0,
        "totalDurationSeconds": round(total_duration, 1),
        "availableModels": list(available_models.keys()),
        "categoriesRun": categories,
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary
    summary_file = out_path / "summary.json"
    summary_file.write_text(dump_json(summary, pretty=True))

    # Save all results
    all_results = [
        {
            "testId": r.test_id,
            "category": r.category,
            "command": r.command,
            "model": r.model,
            "status": r.status,
            "error": r.error,
            "durationSeconds": round(r.duration_seconds, 2),
        }
        for r in results
    ]
    (out_path / "all_results.json").write_text(dump_json(all_results, pretty=True))

    write_output(summary, context.output_format, context.pretty)


@estimate_app.command("train")
def estimate_train(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset"),
    output_path: str = typer.Option(..., "--out"),
    batch_size: int = typer.Option(..., "--batch-size"),
    sequence_length: int = typer.Option(..., "--sequence-length"),
    learning_rate: float = typer.Option(..., "--learning-rate"),
    epochs: int = typer.Option(..., "--epochs"),
    grad_accum: int = typer.Option(..., "--grad-accum"),
    warmup_steps: int = typer.Option(..., "--warmup-steps"),
    weight_decay: float = typer.Option(..., "--weight-decay"),
    gradient_checkpointing: bool = typer.Option(
        ..., "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    mixed_precision: bool = typer.Option(..., "--mixed-precision/--no-mixed-precision"),
    compute_precision: str = typer.Option(..., "--compute-precision"),
    optimizer_type: str = typer.Option(..., "--optimizer-type"),
    seed: int = typer.Option(..., "--seed"),
    deterministic: bool = typer.Option(..., "--deterministic/--stochastic"),
) -> None:
    """Estimate training resource usage for a configuration."""
    context = _context(ctx)
    service = get_training_service()
    from modelcypher.core.domain.training import (
        ComputePrecision,
        Hyperparameters,
        TrainingSpec,
    )

    try:
        precision = ComputePrecision(compute_precision)
    except ValueError as exc:
        valid = ", ".join(m.value for m in ComputePrecision)
        raise typer.BadParameter(
            f"Invalid compute-precision '{compute_precision}'. Valid options: {valid}"
        ) from exc
    if optimizer_type != "adamw":
        raise typer.BadParameter("optimizer-type must be adamw")
    hyperparams = Hyperparameters(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        sequence_length=sequence_length,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        mixed_precision=mixed_precision,
        compute_precision=precision,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed,
        deterministic=deterministic,
        optimizer_type=optimizer_type,
    )
    config = TrainingSpec(
        model_id=model,
        dataset_path=dataset,
        output_path=output_path,
        hyperparameters=hyperparams,
    )
    result = service.preflight(config)
    payload = {
        "willFit": result["canProceed"],
        "recommendedBatchSize": result["predictedBatchSize"],
        "projectedPeakGB": result["estimatedVRAMUsageBytes"] / (1024**3)
        if result["estimatedVRAMUsageBytes"]
        else None,
        "availableGB": result["availableVRAMBytes"] / (1024**3)
        if result["availableVRAMBytes"]
        else None,
        "ttftSeconds": None,
        "tokensPerSecond": None,
        "tokensPerSecondMin": None,
        "tokensPerSecondMax": None,
        "powerSource": "unknown",
        "thermalState": "unknown",
        "etaSeconds": None,
        "notes": [f"computePrecision={precision.value}"],
    }
    write_output(payload, context.output_format, context.pretty)


@geometry_app.command("validate")
def geometry_validate(
    ctx: typer.Context,
    include_fixtures: bool = typer.Option(False, "--include-fixtures"),
    file: str | None = typer.Option(None, "--file"),
) -> None:
    """Validate geometry invariants on built-in fixtures."""
    context = _context(ctx)
    service = GeometryService()
    try:
        report = service.validate(include_fixtures=include_fixtures)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-2005",
            title="Geometry validation failed",
            detail=str(exc),
            hint="Check MLX runtime status (mc system status) and ensure MLX loads on this machine.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
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
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Agent-eval commands (extracted to commands/agent_eval.py)
app.add_typer(agent_eval_commands.app, name="agent-eval", help="Agent evaluation runs")


# Research commands (all commands in research_commands.app)
app.add_typer(research_commands.app, name="research", help="Research experiments")
