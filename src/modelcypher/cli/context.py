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

import os
import sys
from dataclasses import dataclass


@dataclass
class CLIContext:
    ai_mode: bool
    output_format: str
    quiet: bool
    very_quiet: bool
    yes: bool
    no_prompt: bool
    pretty: bool
    log_level: str
    trace_id: str | None


def resolve_ai_mode(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("MC_NO_AI") == "1":
        return False
    if os.environ.get("MC_AI_MODE") == "1":
        return True
    return not sys.stdout.isatty()


def resolve_output_format(ai_mode: bool, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MC_OUTPUT")
    if env:
        return env
    return "json" if ai_mode else "text"
