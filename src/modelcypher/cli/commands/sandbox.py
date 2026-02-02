# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later

"""QUARANTINED: Sandbox commands moved to experimental/.

The sandbox module contained claims that "balanced geometry is definitionally
aligned" without peer-reviewed support.

If you need this functionality for research:
1. Import from modelcypher.experimental.sandbox
2. Use the experimental CLI directly

See experimental/__init__.py for details.
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True, help="[QUARANTINED] Geometric self-study sandbox")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _quarantine_error(ctx: typer.Context, command: str) -> None:
    """Emit quarantine error and exit."""
    context = _context(ctx)
    error = ErrorDetail(
        code="MC-3099",
        title="Command quarantined",
        detail=(
            f"The 'sandbox {command}' command has been quarantined to experimental/. "
            "The sandbox module contained interpretive claims without peer-reviewed support. "
            "Import from modelcypher.experimental.sandbox if needed for research."
        ),
        trace_id=context.trace_id,
    )
    write_error(error.as_dict(), context.output_format, context.pretty)
    raise typer.Exit(code=1)


@app.command("explore")
def sandbox_explore(ctx: typer.Context) -> None:
    """[QUARANTINED] Interactive REPL with geometric feedback."""
    _quarantine_error(ctx, "explore")


@app.command("compare")
def sandbox_compare(ctx: typer.Context) -> None:
    """[QUARANTINED] Compare multiple reasoning approaches geometrically."""
    _quarantine_error(ctx, "compare")


@app.command("study")
def sandbox_study(ctx: typer.Context) -> None:
    """[QUARANTINED] Run automated self-study curriculum."""
    _quarantine_error(ctx, "study")


@app.command("attempt")
def sandbox_attempt(ctx: typer.Context) -> None:
    """[QUARANTINED] Generate a single response and show geometric feedback."""
    _quarantine_error(ctx, "attempt")
