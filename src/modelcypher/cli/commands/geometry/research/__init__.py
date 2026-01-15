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

"""Geometry research CLI commands."""

from __future__ import annotations

import typer

from . import (
    curvature_cmds,
    density_cmds,
    diff_cmds,
    evidence_cmds,
    eval_cmds,
    graft_cmds,
    manifold_evidence_cmds,
    prompt_manifold_cmds,
    profile_cmds,
    shared_manifold_cmds,
    validation_cmds,
)

app = typer.Typer(no_args_is_help=True)


def _register() -> None:
    curvature_cmds.register(app)
    density_cmds.register(app)
    diff_cmds.register(app)
    evidence_cmds.register(app)
    graft_cmds.register(app)
    profile_cmds.register(app)
    eval_cmds.register(app)
    validation_cmds.register(app)
    shared_manifold_cmds.register(app)
    manifold_evidence_cmds.register(app)
    prompt_manifold_cmds.register(app)


_register()

__all__ = ["app"]
