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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.core.domain.bridge.bridge_generator import (
        BridgeGeneratorResult,
        CrossModalBridge,
    )
    from modelcypher.ports.backend import Backend


@runtime_checkable
class BridgeStore(Protocol):
    """Port for persisting cross-modal bridge artifacts."""

    def save(
        self,
        path: Path,
        result: "BridgeGeneratorResult",
        backend: "Backend | None" = None,
    ) -> None:
        """Save a bridge artifact to disk."""
        ...

    def load(
        self,
        path: Path,
        backend: "Backend | None" = None,
    ) -> "CrossModalBridge":
        """Load a bridge artifact from disk."""
        ...


__all__ = ["BridgeStore"]
