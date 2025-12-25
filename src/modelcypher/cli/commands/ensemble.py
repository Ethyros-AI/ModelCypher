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

"""Ensemble CLI commands."""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("create")
def ensemble_create(
    ctx: typer.Context,
    models: list[str] = typer.Option(..., "--model", help="Model paths to include in ensemble"),
    strategy: str = typer.Option(
        "weighted", "--strategy", help="Routing strategy: weighted, routing, voting, cascade"
    ),
    weights: list[float] | None = typer.Option(
        None, "--weight", help="Weights for weighted strategy (must sum to 1.0)"
    ),
) -> None:
    """Create an ensemble configuration from multiple models."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_ensemble_service

    service = get_ensemble_service()

    try:
        result = service.create(
            model_paths=models,
            strategy=strategy,
            weights=weights,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1019",
            title="Ensemble creation failed",
            detail=str(exc),
            hint="Ensure all model paths exist and strategy is valid",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "ensembleId": result.ensemble_id,
        "models": result.models,
        "routingStrategy": result.routing_strategy,
        "weights": result.weights,
        "createdAt": result.created_at,
        "configPath": result.config_path,
    }

    if context.output_format == "text":
        lines = [
            "ENSEMBLE CREATED",
            f"Ensemble ID: {result.ensemble_id}",
            f"Strategy: {result.routing_strategy}",
            f"Models: {len(result.models)}",
        ]
        for i, model in enumerate(result.models):
            weight = result.weights[i] if result.weights else 1.0 / len(result.models)
            lines.append(f"  - {model} (weight: {weight:.3f})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("run")
def ensemble_run(
    ctx: typer.Context,
    ensemble_id: str = typer.Argument(..., help="Ensemble ID"),
    prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
) -> None:
    """Execute ensemble inference."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_ensemble_service

    service = get_ensemble_service()

    try:
        result = service.run(
            ensemble_id=ensemble_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1020",
            title="Ensemble inference failed",
            detail=str(exc),
            hint="Ensure ensemble ID is valid and models are accessible",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "ensembleId": result.ensemble_id,
        "prompt": result.prompt[:100] if len(result.prompt) > 100 else result.prompt,
        "response": result.response,
        "modelContributions": result.model_contributions,
        "totalDuration": result.total_duration,
        "strategy": result.strategy,
        "modelsUsed": result.models_used,
        "aggregationMethod": result.aggregation_method,
    }

    if context.output_format == "text":
        lines = [
            "ENSEMBLE INFERENCE",
            f"Ensemble ID: {result.ensemble_id}",
            f"Strategy: {result.strategy}",
            f"Models used: {result.models_used}",
            f"Duration: {result.total_duration:.3f}s",
            "",
            "Response:",
            result.response,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("list")
def ensemble_list(ctx: typer.Context) -> None:
    """List all ensemble configurations."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_ensemble_service

    service = get_ensemble_service()
    ensembles = service.list_ensembles()

    payload = {
        "ensembles": [
            {
                "ensembleId": e.ensemble_id,
                "models": len(e.models),
                "strategy": e.routing_strategy,
                "createdAt": e.created_at,
            }
            for e in ensembles
        ],
        "count": len(ensembles),
    }

    if context.output_format == "text":
        if not ensembles:
            write_output("No ensembles found.", context.output_format, context.pretty)
            return
        lines = ["ENSEMBLES", ""]
        for e in ensembles:
            lines.append(f"  {e.ensemble_id}")
            lines.append(f"    Strategy: {e.routing_strategy}")
            lines.append(f"    Models: {len(e.models)}")
            lines.append(f"    Created: {e.created_at}")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("delete")
def ensemble_delete(
    ctx: typer.Context,
    ensemble_id: str = typer.Argument(..., help="Ensemble ID to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Delete an ensemble configuration."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_ensemble_service

    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete ensemble {ensemble_id}?"):
            raise typer.Exit(code=1)

    service = get_ensemble_service()
    deleted = service.delete(ensemble_id)

    if not deleted:
        error = ErrorDetail(
            code="MC-2005",
            title="Ensemble not found",
            detail=f"Ensemble '{ensemble_id}' does not exist",
            hint="Use 'mc ensemble list' to see available ensembles",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {"deleted": ensemble_id}

    if context.output_format == "text":
        write_output(f"Deleted ensemble: {ensemble_id}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
