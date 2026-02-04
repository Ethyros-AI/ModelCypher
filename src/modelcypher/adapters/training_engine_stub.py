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

import asyncio
from dataclasses import dataclass
from typing import Any


class BackendTrainingEngine:
    """Placeholder training engine for backend-selected workflows."""

    def __init__(self) -> None:
        self._paused_jobs: set[str] = set()
        self._pause_events: dict[str, asyncio.Event] = {}
        self._cancelled_jobs: set[str] = set()

    async def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Training engine is not implemented for this runtime."
        )


@dataclass
class BackendCheckpointManager:
    """Placeholder checkpoint manager."""

    max_checkpoints: int = 3

    def prune(self, checkpoints: list[str]) -> list[str]:
        return checkpoints[-self.max_checkpoints :]


class BackendLossLandscapeComputer:
    """Placeholder loss landscape computer."""

    def compute(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Loss landscape computation is not implemented for this runtime."
        )


__all__ = [
    "BackendTrainingEngine",
    "BackendCheckpointManager",
    "BackendLossLandscapeComputer",
]
