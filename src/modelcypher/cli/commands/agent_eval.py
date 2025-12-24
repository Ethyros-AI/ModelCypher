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

"""Agent evaluation CLI commands.

Provides commands for running agent evaluations and retrieving results.

Commands:
    mc agent-eval run --model <path> --suite <name>
    mc agent-eval results <eval_id>
"""

from __future__ import annotations


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("run")
def agent_eval_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    eval_suite: str = typer.Option("default", "--suite", help="Evaluation suite"),
    max_turns: int = typer.Option(10, "--max-turns", help="Max conversation turns"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Execute agent evaluation."""
    context = _context(ctx)
    from modelcypher.core.use_cases.agent_eval_service import (
        AgentEvalConfig,
        AgentEvalService,
    )

    config = AgentEvalConfig(
        model_path=model,
        eval_suite=eval_suite,
        max_turns=max_turns,
        timeout_seconds=timeout,
        seed=seed,
    )
    service = AgentEvalService()

    try:
        result = service.run(config)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1012",
            title="Agent evaluation failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "evalId": result.eval_id,
        "modelPath": result.model_path,
        "evalSuite": result.eval_suite,
        "status": result.status,
        "startedAt": result.started_at,
        "config": result.config,
        "summary": result.summary,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("results")
def agent_eval_results(
    ctx: typer.Context,
    eval_id: str = typer.Argument(..., help="Evaluation ID"),
) -> None:
    """Get agent evaluation results."""
    context = _context(ctx)
    from modelcypher.core.use_cases.agent_eval_service import AgentEvalService

    service = AgentEvalService()

    try:
        result = service.results(eval_id)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2012",
            title="Agent evaluation not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "evalId": result.eval_id,
        "modelPath": result.model_path,
        "evalSuite": result.eval_suite,
        "status": result.status,
        "startedAt": result.started_at,
        "completedAt": result.completed_at,
        "config": result.config,
        "metrics": result.metrics,
        "taskResults": result.task_results,
        "interpretation": result.interpretation,
        "overallScore": result.overall_score,
    }

    write_output(payload, context.output_format, context.pretty)
