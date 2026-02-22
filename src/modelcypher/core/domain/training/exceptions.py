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

from dataclasses import dataclass
from typing import Any


class CheckpointError(Exception):
    """Exception raised for errors during model checkpointing."""


@dataclass
class TrainingDerivationError(Exception):
    """Hard failure for strict geometry-derivation requirements.

    Attributes
    ----------
    failure_class:
        Stable classification key consumed by CLI/tests.
    detail:
        Human-readable failure explanation.
    diagnostics:
        Structured context for actionable troubleshooting.
    """

    failure_class: str
    detail: str
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__(self.detail)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for CLI error payloads."""
        return {
            "failure_class": self.failure_class,
            "detail": self.detail,
            "diagnostics": self.diagnostics or {},
        }
