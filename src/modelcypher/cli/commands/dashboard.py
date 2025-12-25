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

"""Dashboard CLI commands."""

from __future__ import annotations

import sys

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("metrics")
def dashboard_metrics(ctx: typer.Context) -> None:
    """Return current metrics in Prometheus format."""
    context = _context(ctx)
    from modelcypher.core.use_cases.dashboard_service import DashboardService

    service = DashboardService()
    metrics = service.metrics()

    if context.output_format == "json":
        lines = metrics.strip().split("\n")
        metric_dict = {}
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(" ")
            if len(parts) >= 2:
                metric_dict[parts[0]] = parts[1]
        write_output(metric_dict, context.output_format, context.pretty)
    else:
        sys.stdout.write(metrics + "\n")


@app.command("export")
def dashboard_export(
    ctx: typer.Context,
    format: str = typer.Option("prometheus", "--format", help="Export format"),
    output_path: str | None = typer.Option(None, "--output-path", help="Output path"),
) -> None:
    """Export dashboard data."""
    context = _context(ctx)
    from modelcypher.core.use_cases.dashboard_service import DashboardService

    service = DashboardService()

    try:
        result = service.export(format, output_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1013",
            title="Dashboard export failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "format": result.format,
        "exportPath": result.export_path,
        "exportedAt": result.exported_at,
        "metricsCount": result.metrics_count,
    }

    if context.output_format == "text" and not output_path:
        sys.stdout.write(result.content + "\n")
    else:
        write_output(payload, context.output_format, context.pretty)
