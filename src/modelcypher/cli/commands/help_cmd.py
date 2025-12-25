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

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

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
