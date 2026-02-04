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

"""Prompt input helpers for CLI commands."""

from __future__ import annotations

import sys

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error
from modelcypher.cli.input_validation import validate_file_exists
from modelcypher.utils.errors import ErrorDetail


def resolve_prompt_input(
    *,
    prompt: str | None,
    prompt_file: str | None,
    prompt_stdin: bool,
    context: CLIContext,
    label: str = "prompt",
    default: str | None = None,
    required: bool = True,
) -> str:
    """Resolve prompt input from direct text, file, or stdin."""
    if prompt is not None and (prompt_file or prompt_stdin):
        _fail_prompt_input(
            title="Conflicting prompt inputs",
            detail="Provide only one of --prompt, --prompt-file, or --prompt-stdin.",
            context=context,
        )
    if prompt_file and prompt_stdin:
        _fail_prompt_input(
            title="Conflicting prompt inputs",
            detail="Provide only one of --prompt-file or --prompt-stdin.",
            context=context,
        )

    content: str | None = None
    if prompt_file:
        path = validate_file_exists(prompt_file, description=f"{label} file", context=context)
        content = path.read_text(encoding="utf-8")
    elif prompt_stdin or (prompt is None and not sys.stdin.isatty()):
        content = sys.stdin.read()
    elif prompt is not None:
        content = prompt

    if content is None:
        if not required and default is not None:
            return default
        _fail_prompt_input(
            title=f"Missing {label}",
            detail=(
                f"Provide --{label}, --{label}-file, or --{label}-stdin."
            ),
            context=context,
        )

    if content == "":
        if not required and default is not None:
            return default
        _fail_prompt_input(
            title=f"Empty {label}",
            detail=f"No {label} content was provided.",
            context=context,
        )

    return content


def _fail_prompt_input(*, title: str, detail: str, context: CLIContext) -> None:
    error = ErrorDetail(
        code="MC-1080",
        title=title,
        detail=detail,
        trace_id=context.trace_id,
    )
    write_error(error.as_dict(), context.output_format, context.pretty)
    raise typer.Exit(code=1)
