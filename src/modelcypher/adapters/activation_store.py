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

from modelcypher.ports.activation_store import ActivationStore

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class SafetensorsActivationStore(ActivationStore):
    """Persist probe activations using safetensors format (backend-agnostic)."""

    def save_probe_activations(
        self,
        activation_path: Path,
        arrays: dict[str, "Array"],
        backend: "Backend",
    ) -> None:
        if not arrays:
            return

        activation_path = Path(activation_path)
        # Use .safetensors extension
        if activation_path.suffix == ".npz":
            activation_path = activation_path.with_suffix(".safetensors")

        backend.save_safetensors(str(activation_path), arrays)

    def load_probe_activations(
        self,
        activation_path: Path,
        backend: "Backend",
    ) -> dict[str, "Array"] | None:
        activation_path = Path(activation_path)
        if not activation_path.exists():
            # Try .safetensors if .npz was requested
            if activation_path.suffix == ".npz":
                safetensors_path = activation_path.with_suffix(".safetensors")
                if safetensors_path.exists():
                    activation_path = safetensors_path
                else:
                    return None
            else:
                return None

        return backend.load_safetensors(str(activation_path))


# Backwards compatibility alias
NPZActivationStore = SafetensorsActivationStore


__all__ = ["SafetensorsActivationStore", "NPZActivationStore"]
