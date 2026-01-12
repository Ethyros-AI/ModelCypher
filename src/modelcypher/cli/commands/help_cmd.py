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


@app.command("command")
def help_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Command name (e.g., 'merge', 'model', 'infer')"),
) -> None:
    """Show help for a specific CLI command.

    This provides a convenient way to get help for any top-level command.

    Examples:
        mc help command merge
        mc help command model
        mc help command geometry
    """
    import shutil
    import subprocess
    import sys

    context = _context(ctx)

    # Find the mc executable
    mc_path = shutil.which("mc")
    if mc_path is None:
        # Fallback: try to run mc directly (it might be available as an entry point)
        mc_path = "mc"

    # Run mc <name> --help
    result = subprocess.run(
        [mc_path, name, "--help"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        sys.stdout.write(result.stdout)
    else:
        # Check if it's a "command not found" error
        stderr = result.stderr
        if "No such command" in stderr:
            error = ErrorDetail(
                code="MC-1050",
                title="Command not found",
                detail=f"No such command: {name}",
                hint="Run 'mc --help' to see available commands.",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        else:
            # Some other error - could be valid help output on stderr
            if result.stdout:
                sys.stdout.write(result.stdout)
            if stderr:
                sys.stderr.write(stderr)
            raise typer.Exit(code=result.returncode)


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
