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

"""Help CLI commands."""

from __future__ import annotations

import sys

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("ask")
def help_ask(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Question about ModelCypher"),
) -> None:
    """Get contextual help for a question."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()
    result = service.ask(question)

    payload = {
        "question": result.question,
        "answer": result.answer,
        "relatedCommands": result.related_commands,
        "examples": result.examples,
        "docsUrl": result.docs_url,
    }

    if context.output_format == "text":
        lines = [
            f"Q: {result.question}",
            "",
            result.answer,
            "",
            "Related commands:",
        ]
        for cmd in result.related_commands:
            lines.append(f"  - {cmd}")
        lines.append("")
        lines.append("Examples:")
        for ex in result.examples:
            lines.append(f"  $ {ex}")
        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("completions")
def help_completions(
    ctx: typer.Context,
    shell: str = typer.Argument(..., help="Shell type: bash, zsh, fish"),
) -> None:
    """Generate shell completion script."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()

    try:
        script = service.completions(shell)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1014",
            title="Completions generation failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    sys.stdout.write(script)


@app.command("schema")
def help_schema(
    ctx: typer.Context,
    command: str = typer.Argument(..., help="Command name"),
) -> None:
    """Return JSON schema for command output."""
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()

    try:
        result = service.schema(command)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-2014",
            title="Schema not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    write_output(result, context.output_format, context.pretty)
