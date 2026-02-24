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

"""Semantic exit codes for agent-friendly error classification.

Agents use exit codes to decide what to do next:
    0  Success         → proceed
    1  Input error     → retry with corrected input
    2  Cancelled       → ask user for confirmation
    3  Dependency      → report prerequisite (backend, volume, etc.)
    4  Runtime error   → report or retry after resource check
    5  Internal error  → report bug
"""

EXIT_OK = 0
EXIT_INPUT = 1
EXIT_CANCELLED = 2
EXIT_DEPENDENCY = 3
EXIT_RUNTIME = 4
EXIT_INTERNAL = 5
