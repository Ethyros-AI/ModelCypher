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

"""CLI input validation helpers.

Provides early validation for common input patterns to give users clear,
actionable error messages before expensive operations begin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from modelcypher.utils.errors import ErrorDetail
from modelcypher.cli.output import write_error


def validate_file_exists(
    path: str,
    *,
    description: str = "File",
    context: Any = None,
) -> Path:
    """Validate that a file exists and is readable.

    Args:
        path: Path to validate
        description: Human-readable description for error messages
        context: CLIContext for trace_id (optional)

    Returns:
        Path object if valid

    Raises:
        typer.Exit: If file doesn't exist or isn't readable
    """
    p = Path(path)
    if not p.exists():
        _fail(
            code="MC-1020",
            title=f"{description} not found",
            detail=f"Path does not exist: {path}",
            hint="Check the path for typos. Use absolute paths to avoid confusion.",
            context=context,
        )
    if not p.is_file():
        _fail(
            code="MC-1021",
            title=f"{description} is not a file",
            detail=f"Expected a file but found a directory: {path}",
            hint="Provide a path to a file, not a directory.",
            context=context,
        )
    return p


def validate_dir_exists(
    path: str,
    *,
    description: str = "Directory",
    context: Any = None,
) -> Path:
    """Validate that a directory exists.

    Args:
        path: Path to validate
        description: Human-readable description for error messages
        context: CLIContext for trace_id (optional)

    Returns:
        Path object if valid

    Raises:
        typer.Exit: If directory doesn't exist
    """
    p = Path(path)
    if not p.exists():
        _fail(
            code="MC-1022",
            title=f"{description} not found",
            detail=f"Path does not exist: {path}",
            hint="Check the path for typos. Use absolute paths to avoid confusion.",
            context=context,
        )
    if not p.is_dir():
        _fail(
            code="MC-1023",
            title=f"{description} is not a directory",
            detail=f"Expected a directory but found a file: {path}",
            hint="Provide a path to a directory, not a file.",
            context=context,
        )
    return p


def validate_model_path(
    path: str,
    *,
    context: Any = None,
) -> Path:
    """Validate that a path points to a valid model directory.

    A valid model directory contains config.json.

    Args:
        path: Path to model directory
        context: CLIContext for trace_id (optional)

    Returns:
        Path object if valid

    Raises:
        typer.Exit: If model directory is invalid
    """
    p = validate_dir_exists(path, description="Model directory", context=context)
    config_path = p / "config.json"
    if not config_path.exists():
        _fail(
            code="MC-1001",
            title="Invalid model directory",
            detail=f"config.json not found in: {path}",
            hint=(
                "Model directories must contain config.json. "
                "If using HuggingFace cache, point to the snapshot directory."
            ),
            context=context,
        )
    return p


def validate_json_file(
    path: str,
    *,
    description: str = "JSON file",
    context: Any = None,
) -> Any:
    """Validate and load a JSON file.

    Args:
        path: Path to JSON file
        description: Human-readable description for error messages
        context: CLIContext for trace_id (optional)

    Returns:
        Parsed JSON data

    Raises:
        typer.Exit: If file doesn't exist or contains invalid JSON
    """
    p = validate_file_exists(path, description=description, context=context)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        _fail(
            code="MC-1024",
            title=f"Invalid JSON in {description.lower()}",
            detail=f"Parse error at line {e.lineno}, column {e.colno}: {e.msg}",
            hint=f"Check the JSON syntax in: {path}",
            context=context,
        )


def _fail(
    *,
    code: str,
    title: str,
    detail: str,
    hint: str | None = None,
    context: Any = None,
) -> None:
    """Write error and exit."""
    trace_id = getattr(context, "trace_id", None) if context else None
    output_format = getattr(context, "output_format", "json") if context else "json"
    pretty = getattr(context, "pretty", False) if context else False

    error = ErrorDetail(
        code=code,
        title=title,
        detail=detail,
        hint=hint,
        trace_id=trace_id,
    )
    write_error(error.as_dict(), output_format, pretty)
    raise typer.Exit(code=1)
