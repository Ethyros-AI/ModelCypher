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

from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
from modelcypher.cli.typer_compat import apply_typer_compat

apply_typer_compat()

app = typer.Typer(no_args_is_help=True, add_completion=False)
geometry_app = typer.Typer(no_args_is_help=True)

geometry_app.add_typer(geometry_metrics_commands.app, name="metrics")
app.add_typer(geometry_app, name="geometry")
