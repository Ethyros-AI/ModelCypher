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
    from modelcypher.ports.backend import Array, Backend


@runtime_checkable
class ActivationStore(Protocol):
    """Port for persisting probe activation checkpoints."""

    def save_probe_activations(
        self,
        activation_path: Path,
        arrays: dict[str, "Array"],
        backend: "Backend",
    ) -> None:
        """Save probe activations to an on-disk checkpoint."""
        ...

    def load_probe_activations(
        self,
        activation_path: Path,
        backend: "Backend",
    ) -> dict[str, "Array"] | None:
        """Load probe activations from an on-disk checkpoint."""
        ...


__all__ = ["ActivationStore"]
