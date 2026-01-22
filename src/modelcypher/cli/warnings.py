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

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.utils.security import TRUST_REMOTE_CODE_ENV, trust_remote_code_enabled


def warn_trust_remote_code(context: CLIContext) -> None:
    """Emit a CLI warning when trust_remote_code is enabled."""
    if context.ai_mode or context.quiet or context.very_quiet:
        return
    if not trust_remote_code_enabled():
        return
    typer.echo(
        f"SECURITY WARNING: {TRUST_REMOTE_CODE_ENV}=1 enables execution of model-supplied code. "
        "Only load models you trust.",
        err=True,
    )


def warn_network(context: CLIContext, message: str) -> None:
    """Emit a CLI warning before network access."""
    if context.ai_mode or context.quiet or context.very_quiet:
        return
    typer.echo(f"NETWORK WARNING: {message}", err=True)
