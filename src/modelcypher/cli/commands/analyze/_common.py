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

"""Shared utilities for legacy expert analyze commands."""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.warnings import warn_network
from modelcypher.utils.errors import ErrorDetail


def get_context(ctx: typer.Context) -> CLIContext:
    """Extract CLIContext from typer context."""
    return ctx.obj


__all__ = [
    "Path",
    "typer",
    "CLIContext",
    "write_error",
    "write_output",
    "warn_network",
    "ErrorDetail",
    "get_context",
]
