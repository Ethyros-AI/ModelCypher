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

"""System CLI commands.

Provides commands for:
- System status and health checks
- System probing

Commands:
    mc system status
    mc system probe <target>
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.system_service import SystemService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def system_status(
    ctx: typer.Context, require_metal: bool = typer.Option(False, "--require-metal")
) -> None:
    """Get system status.

    Examples:
        mc system status
        mc system status --require-metal
    """
    context = _context(ctx)
    service = SystemService()
    status = service.status()
    if require_metal and not status["metalAvailable"]:
        raise typer.Exit(code=3)
    write_output(status, context.output_format, context.pretty)


@app.command("probe")
def system_probe(ctx: typer.Context, target: str = typer.Argument(...)) -> None:
    """Probe a system target.

    Examples:
        mc system probe gpu
    """
    context = _context(ctx)
    service = SystemService()
    write_output(service.probe(target), context.output_format, context.pretty)
