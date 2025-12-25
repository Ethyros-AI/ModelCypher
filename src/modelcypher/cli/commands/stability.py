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

"""Stability testing CLI commands."""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("run")
def stability_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    num_runs: int = typer.Option(10, "--num-runs", help="Number of test runs"),
    prompt_variations: int = typer.Option(5, "--prompt-variations", help="Prompt variations"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Execute stability suite on a model."""
    context = _context(ctx)
    from modelcypher.core.use_cases.stability_service import (
        StabilityConfig,
        StabilityService,
    )

    config = StabilityConfig(
        num_runs=num_runs,
        prompt_variations=prompt_variations,
        seed=seed,
    )
    service = StabilityService()

    try:
        result = service.run(model, config)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1011",
            title="Stability test failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "suiteId": result.suite_id,
        "modelPath": result.model_path,
        "status": result.status,
        "startedAt": result.started_at,
        "config": result.config,
        "summary": result.summary,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("report")
def stability_report(
    ctx: typer.Context,
    suite_id: str = typer.Argument(..., help="Stability suite ID"),
) -> None:
    """Get detailed stability report."""
    context = _context(ctx)
    from modelcypher.core.use_cases.stability_service import StabilityService

    service = StabilityService()

    try:
        result = service.report(suite_id)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2011",
            title="Stability suite not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "suiteId": result.suite_id,
        "modelPath": result.model_path,
        "status": result.status,
        "startedAt": result.started_at,
        "completedAt": result.completed_at,
        "config": result.config,
        "metrics": result.metrics,
        "perPromptResults": result.per_prompt_results,
        "interpretation": result.interpretation,
        "recommendations": result.recommendations,
    }

    write_output(payload, context.output_format, context.pretty)
