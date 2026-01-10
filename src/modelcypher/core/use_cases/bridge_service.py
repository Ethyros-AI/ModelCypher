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

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.domain.bridge.generator import (
        BridgeGeneratorResult,
        CrossModalBridge,
    )
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.bridge_store import BridgeStore


class BridgeService:
    """Use-case wrapper for bridge persistence."""

    def __init__(
        self,
        store: "BridgeStore",
        backend: "Backend | None" = None,
    ) -> None:
        self._store = store
        self._backend = backend

    def save(self, result: "BridgeGeneratorResult", path: str | Path) -> None:
        self._store.save(Path(path), result, backend=self._backend)

    def load(self, path: str | Path) -> "CrossModalBridge":
        return self._store.load(Path(path), backend=self._backend)


__all__ = ["BridgeService"]
