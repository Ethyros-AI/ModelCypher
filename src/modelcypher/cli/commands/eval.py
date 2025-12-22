"""Evaluation and comparison CLI commands.

Provides commands for:
- Evaluation management: list, show, run
- Model comparison: list, show, run, checkpoints, baseline, score

Commands:
    mc eval list
    mc eval show <id>
    mc eval run --model <path> --dataset <path>
    mc compare list
    mc compare run --checkpoint <path1> --checkpoint <path2>
"""

from __future__ import annotations

from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.presenters import (
    compare_detail_payload,
    compare_list_payload,
    evaluation_detail_payload,
    evaluation_list_payload,
)
from modelcypher.core.use_cases.compare_service import CompareService
from modelcypher.core.use_cases.evaluation_service import EvaluationService

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
    service = EvaluationService()
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
    service = EvaluationService()
    result = service.get_evaluation(eval_id)
    write_output(evaluation_detail_payload(result), context.output_format, context.pretty)


@eval_app.command("run")
def eval_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset file"),
    batch_size: int = typer.Option(4, "--batch-size"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
) -> None:
    """Execute evaluation on model with dataset.

    Examples:
        mc eval run --model ./model --dataset ./data.jsonl
        mc eval run --model ./model --dataset ./data.jsonl --batch-size 8
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.evaluation_service import EvalConfig

    service = EvaluationService()
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


@compare_app.command("list")
def compare_list(
    ctx: typer.Context,
    status: Optional[str] = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List all comparison sessions.

    Examples:
        mc compare list
        mc compare list --status completed --limit 10
    """
    context = _context(ctx)
    service = CompareService()
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
    service = CompareService()
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

    service = CompareService()
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
    service = CompareService()
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
    service = CompareService()
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
    service = CompareService()
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
