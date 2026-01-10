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


def _backend_is_mlx(backend: "Backend") -> bool:
    try:
        probe = backend.zeros((1,))
    except Exception:
        return False
    return type(probe).__module__.startswith("mlx")


class NPZActivationStore(ActivationStore):
    """Persist probe activations to NPZ with backend-aware conversion."""

    def save_probe_activations(
        self,
        activation_path: Path,
        arrays: dict[str, "Array"],
        backend: "Backend",
    ) -> None:
        if not arrays:
            return

        activation_path = Path(activation_path)
        temp_path = activation_path.with_suffix(".tmp.npz")

        if _backend_is_mlx(backend):
            try:
                import mlx.core as mx

                mx.savez(str(temp_path), **arrays)
                temp_path.rename(activation_path)
                return
            except Exception:
                pass

        import numpy as np

        np_arrays = {k: backend.to_numpy(v) for k, v in arrays.items()}
        np.savez_compressed(str(temp_path), **np_arrays)
        temp_path.rename(activation_path)

    def load_probe_activations(
        self,
        activation_path: Path,
        backend: "Backend",
    ) -> dict[str, "Array"] | None:
        activation_path = Path(activation_path)
        if not activation_path.exists():
            return None

        if _backend_is_mlx(backend):
            try:
                import mlx.core as mx

                loaded = mx.load(str(activation_path))
                return dict(loaded)
            except Exception:
                pass

        import numpy as np

        loaded = dict(np.load(str(activation_path)))
        return {k: backend.array(v) for k, v in loaded.items()}


__all__ = ["NPZActivationStore"]
