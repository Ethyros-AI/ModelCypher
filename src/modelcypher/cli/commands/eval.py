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

"""Evaluation and comparison CLI commands.

Provides commands for:
- Evaluation management: list, show, run, benchmark
- Model comparison: list, show, run, checkpoints, baseline, score

Commands:
    mc eval list
    mc eval show <id>
    mc eval run --model <path> --dataset <path>
    mc eval benchmark --model <path> --tasks gsm8k,hellaswag
    mc compare list
    mc compare run --checkpoint <path1> --checkpoint <path2>
"""

from __future__ import annotations

import typer

from modelcypher.cli.composition import get_compare_service, get_evaluation_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.presenters import (
    compare_detail_payload,
    compare_list_payload,
    evaluation_detail_payload,
    evaluation_list_payload,
)

eval_app = typer.Typer(no_args_is_help=True)
compare_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@eval_app.command("list")
def eval_list(ctx: typer.Context, limit: int = typer.Option(50, "--limit")) -> None:
    """List all evaluations.

    Examples:
        mc eval list
        mc eval list --limit 10
    """
    context = _context(ctx)
    service = get_evaluation_service()
    payload = service.list_evaluations(limit)
    results = payload["evaluations"] if isinstance(payload, dict) else payload
    write_output(evaluation_list_payload(results), context.output_format, context.pretty)


@eval_app.command("show")
def eval_show(ctx: typer.Context, eval_id: str = typer.Argument(...)) -> None:
    """Show evaluation details.

    Examples:
        mc eval show abc123
    """
    context = _context(ctx)
    service = get_evaluation_service()
    result = service.get_evaluation(eval_id)
    write_output(evaluation_detail_payload(result), context.output_format, context.pretty)


@eval_app.command("run")
def eval_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset file"),
    batch_size: int = typer.Option(4, "--batch-size"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
) -> None:
    """Execute evaluation on model with dataset.

    Examples:
        mc eval run --model ./model --dataset ./data.jsonl
        mc eval run --model ./model --dataset ./data.jsonl --batch-size 8
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.evaluation_service import EvalConfig

    service = get_evaluation_service()
    config = EvalConfig(batch_size=batch_size, max_samples=max_samples)

    result = service.run(model, dataset, config)

    payload = {
        "evalId": result.eval_id,
        "modelPath": result.model_path,
        "datasetPath": result.dataset_path,
        "averageLoss": result.average_loss,
        "perplexity": result.perplexity,
        "sampleCount": result.sample_count,
    }

    if context.output_format == "text":
        lines = [
            "EVALUATION COMPLETE",
            f"Eval ID: {result.eval_id}",
            f"Model: {result.model_path}",
            f"Dataset: {result.dataset_path}",
            f"Average Loss: {result.average_loss:.4f}",
            f"Perplexity: {result.perplexity:.4f}",
            f"Samples: {result.sample_count}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@eval_app.command("benchmark")
def eval_benchmark(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to MLX model directory"),
    tasks: str = typer.Option(
        "gsm8k,hellaswag,arc_easy,arc_challenge,winogrande",
        "--tasks",
        help="Comma-separated benchmark tasks",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit samples per task"),
    num_fewshot: int = typer.Option(0, "--num-fewshot", help="Number of few-shot examples"),
    output_path: str | None = typer.Option(None, "--output-path", help="Save results to file"),
) -> None:
    """Run lm-eval-harness benchmarks on an MLX model.

    Uses native MLX inference for accurate benchmarking on Apple Silicon.

    Examples:
        mc eval benchmark --model ./model --tasks gsm8k,hellaswag
        mc eval benchmark --model ./model --tasks mmlu --limit 100
        mc eval benchmark --model ./model --tasks arc_challenge --num-fewshot 5
    """
    import json
    from pathlib import Path

    from modelcypher.core.use_cases.mlx_lm_eval import run_benchmark

    context = _context(ctx)

    model_path = Path(model).resolve()
    if not model_path.exists():
        typer.echo(f"Error: Model path does not exist: {model_path}", err=True)
        raise typer.Exit(1)

    task_list = [t.strip() for t in tasks.split(",")]
    typer.echo(f"Running benchmarks on {model_path}")
    typer.echo(f"Tasks: {task_list}")

    try:
        results = run_benchmark(
            model_path=str(model_path),
            tasks=task_list,
            limit=limit,
            num_fewshot=num_fewshot,
            output_path=output_path,
        )

        # Format output
        if context.output_format == "text":
            typer.echo("\n=== BENCHMARK RESULTS ===")
            typer.echo(f"Model: {model_path}")

            if "results" in results:
                for task_name, task_results in results["results"].items():
                    typer.echo(f"\n{task_name}:")
                    for metric, value in task_results.items():
                        if isinstance(value, float):
                            typer.echo(f"  {metric}: {value:.4f}")
                        else:
                            typer.echo(f"  {metric}: {value}")
        else:
            payload = {
                "model": str(model_path),
                "tasks": task_list,
                "results": results.get("results", {}),
                "config": results.get("config", {}),
            }
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        typer.echo(f"Benchmark failed: {e}", err=True)
        raise typer.Exit(1)


@eval_app.command("domain")
def eval_domain(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to MLX model directory"),
    domains: list[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Knowledge domains to evaluate (repeatable)",
    ),
    suite: str = typer.Option(
        None,
        "--suite",
        "-s",
        help="Use a predefined suite: quick, standard, comprehensive, code, math, reasoning",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit samples per task"),
    num_fewshot: int = typer.Option(0, "--num-fewshot", help="Number of few-shot examples"),
    output_path: str | None = typer.Option(None, "--output-path", help="Save results to file"),
) -> None:
    """Run domain-specific benchmarks mapped to industry standards.

    Maps ModelCypher knowledge domains to industry benchmarks:
    - computational: HumanEval, MBPP (code generation)
    - mathematical: GSM8K, MathQA (math reasoning)
    - logical: ARC-Challenge, LogiQA (logical reasoning)
    - linguistic: HellaSwag, LAMBADA (language understanding)
    - relational: WinoGrande (coreference/relations)
    - moral: TruthfulQA (ethics/truthfulness)

    Examples:
        mc eval domain --model ./model --domain computational
        mc eval domain --model ./model --domain mathematical --domain logical
        mc eval domain --model ./model --suite standard
        mc eval domain --model ./model --suite comprehensive --limit 100
    """
    import json
    from pathlib import Path

    from modelcypher.core.domain.geometry.domain_benchmark_map import (
        BENCHMARK_SUITES,
        DOMAIN_BENCHMARK_MAP,
        EvalDomain,
        get_benchmarks_for_domains,
        get_suite,
    )
    from modelcypher.core.use_cases.mlx_lm_eval import run_benchmark

    context = _context(ctx)

    model_path = Path(model).resolve()
    if not model_path.exists():
        typer.echo(f"Error: Model path does not exist: {model_path}", err=True)
        raise typer.Exit(1)

    # Determine tasks to run
    tasks: list[str] = []

    if suite:
        tasks = get_suite(suite)
        if not tasks:
            available = ", ".join(BENCHMARK_SUITES.keys())
            typer.echo(f"Unknown suite: {suite}. Available: {available}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Using suite '{suite}': {tasks}")

    if domains:
        # Validate domains
        valid_domains = {d.value for d in EvalDomain}
        for d in domains:
            if d not in valid_domains:
                typer.echo(f"Unknown domain: {d}. Available: {', '.join(sorted(valid_domains))}", err=True)
                raise typer.Exit(1)
        domain_tasks = get_benchmarks_for_domains(domains)
        tasks = list(set(tasks) | set(domain_tasks))
        typer.echo(f"Domains {domains} mapped to: {domain_tasks}")

    if not tasks:
        # Default to standard suite
        tasks = get_suite("standard")
        typer.echo(f"No domains/suite specified, using 'standard': {tasks}")

    typer.echo(f"Running benchmarks on {model_path}")
    typer.echo(f"Tasks: {tasks}")

    try:
        results = run_benchmark(
            model_path=str(model_path),
            tasks=tasks,
            limit=limit,
            num_fewshot=num_fewshot,
            output_path=output_path,
        )

        # Format output with domain mappings
        if context.output_format == "text":
            typer.echo("\n=== DOMAIN EVALUATION RESULTS ===")
            typer.echo(f"Model: {model_path}")

            if "results" in results:
                # Group results by domain
                from modelcypher.core.domain.geometry.domain_benchmark_map import domain_from_benchmark

                domain_scores: dict[str, list[tuple[str, float]]] = {}

                for task_name, task_results in results["results"].items():
                    # Get primary metric
                    acc = task_results.get("acc,none", task_results.get("acc_norm,none", 0))
                    if isinstance(acc, (int, float)):
                        task_domains = domain_from_benchmark(task_name)
                        for domain in task_domains:
                            if domain.value not in domain_scores:
                                domain_scores[domain.value] = []
                            domain_scores[domain.value].append((task_name, float(acc)))

                typer.echo("\n--- By Domain ---")
                for domain in sorted(domain_scores.keys()):
                    scores = domain_scores[domain]
                    avg = sum(s for _, s in scores) / len(scores) if scores else 0
                    typer.echo(f"\n{domain.upper()} (avg: {avg:.4f}):")
                    for task, score in scores:
                        typer.echo(f"  {task}: {score:.4f}")

                typer.echo("\n--- All Results ---")
                for task_name, task_results in results["results"].items():
                    typer.echo(f"\n{task_name}:")
                    for metric, value in task_results.items():
                        if isinstance(value, float):
                            typer.echo(f"  {metric}: {value:.4f}")
        else:
            payload = {
                "model": str(model_path),
                "tasks": tasks,
                "domains": domains or [],
                "suite": suite,
                "results": results.get("results", {}),
            }
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        typer.echo(f"Domain evaluation failed: {e}", err=True)
        raise typer.Exit(1)


@compare_app.command("list")
def compare_list(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List all comparison sessions.

    Examples:
        mc compare list
        mc compare list --status completed --limit 10
    """
    context = _context(ctx)
    service = get_compare_service()
    payload = service.list_sessions(limit, status)
    sessions = payload["sessions"] if isinstance(payload, dict) else payload
    write_output(compare_list_payload(sessions), context.output_format, context.pretty)


@compare_app.command("show")
def compare_show(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Show comparison session details.

    Examples:
        mc compare show abc123
    """
    context = _context(ctx)
    service = get_compare_service()
    result = service.get_session(session_id)
    write_output(compare_detail_payload(result), context.output_format, context.pretty)


@compare_app.command("run")
def compare_run(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(..., "--checkpoint", help="Checkpoint paths"),
    prompt: str = typer.Option("Hello, how are you?", "--prompt"),
) -> None:
    """Execute A/B comparison between checkpoints.

    Examples:
        mc compare run --checkpoint ./ckpt1 --checkpoint ./ckpt2
        mc compare run --checkpoint ./ckpt1 --checkpoint ./ckpt2 --prompt "Test"
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.compare_service import CompareConfig

    service = get_compare_service()
    config = CompareConfig(prompt=prompt)

    result = service.run(checkpoints, config)

    payload = {
        "comparisonId": result.comparison_id,
        "checkpoints": result.checkpoints,
        "prompt": result.prompt,
    }

    if context.output_format == "text":
        lines = [
            "COMPARISON STARTED",
            f"Comparison ID: {result.comparison_id}",
            f"Checkpoints: {len(result.checkpoints)}",
            f"Prompt: {result.prompt[:50]}...",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@compare_app.command("checkpoints")
def compare_checkpoints(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Compare checkpoints for a job.

    Examples:
        mc compare checkpoints abc123
    """
    context = _context(ctx)
    service = get_compare_service()
    result = service.checkpoints(job_id)

    payload = {
        "jobId": result.job_id,
        "checkpoints": result.checkpoints,
        "comparisonMetrics": result.comparison_metrics,
    }

    write_output(payload, context.output_format, context.pretty)


@compare_app.command("baseline")
def compare_baseline(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """Establish baseline metrics for comparison.

    Examples:
        mc compare baseline --model ./model
    """
    context = _context(ctx)
    service = get_compare_service()
    result = service.baseline(model)

    payload = {
        "model": result.model,
        "metrics": result.metrics,
        "timestamp": result.timestamp.isoformat(),
    }

    if context.output_format == "text":
        lines = [
            "BASELINE ESTABLISHED",
            f"Model: {result.model}",
            f"Perplexity: {result.metrics.get('perplexity', 'N/A')}",
            f"Latency: {result.metrics.get('latency_ms', 'N/A')}ms",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@compare_app.command("score")
def compare_score(
    ctx: typer.Context,
    comparison_id: str = typer.Argument(..., help="Comparison ID"),
) -> None:
    """Get aggregated comparison scores.

    Examples:
        mc compare score abc123
    """
    context = _context(ctx)
    service = get_compare_service()
    result = service.score(comparison_id)

    payload = {
        "comparisonId": result.comparison_id,
        "scores": result.scores,
        "winner": result.winner,
    }

    if context.output_format == "text":
        lines = [
            "COMPARISON SCORES",
            f"Comparison ID: {result.comparison_id}",
            f"Quality: {result.scores.get('quality', 'N/A')}",
            f"Speed: {result.scores.get('speed', 'N/A')}",
            f"Overall: {result.scores.get('overall', 'N/A')}",
        ]
        if result.winner:
            lines.append(f"Winner: {result.winner}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
