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

"""Job management CLI commands.

Provides commands for:
- Listing training jobs with filtering
- Showing detailed job status and metrics
- Attaching to job logs
- Deleting jobs

Commands:
    mc job list [--status <status>] [--model <model_id>]
    mc job show <job_id>
    mc job attach <job_id>
    mc job delete <job_id>
"""

from __future__ import annotations

import sys

import typer

from modelcypher.cli.composition import get_job_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("list")
def job_list(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status"),
    active_only: bool = typer.Option(False, "--active-only"),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    """List all jobs.

    Examples:
        mc job list
        mc job list --status running
        mc job list --model my-model
    """
    context = _context(ctx)
    service = get_job_service()
    write_output(
        service.list_jobs(status=status, active_only=active_only, model_id=model),
        context.output_format,
        context.pretty,
    )


@app.command("show")
def job_show(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    loss_history: bool = typer.Option(False, "--loss-history"),
) -> None:
    """Show job details.

    Examples:
        mc job show abc123
        mc job show abc123 --loss-history
    """
    context = _context(ctx)
    service = get_job_service()
    write_output(
        service.show_job(job_id, include_loss_history=loss_history),
        context.output_format,
        context.pretty,
    )


@app.command("attach")
def job_attach(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    replay: bool = typer.Option(False, "--replay"),
    since: str | None = typer.Option(None, "--since"),
) -> None:
    """Attach to a running job's output stream.

    Examples:
        mc job attach abc123
        mc job attach abc123 --replay --since 2024-01-01T00:00:00
    """
    service = get_job_service()
    lines = service.attach(job_id, since=since if replay else None)
    for line in lines:
        sys.stdout.write(line + "\n")


@app.command("delete")
def job_delete(ctx: typer.Context, job_id: str = typer.Argument(...)) -> None:
    """Delete a job.

    Examples:
        mc job delete abc123
    """
    context = _context(ctx)
    service = get_job_service()
    write_output(service.delete_job(job_id), context.output_format, context.pretty)
