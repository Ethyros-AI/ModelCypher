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
from typing import TYPE_CHECKING, Any

from modelcypher.ports.adapter_weights import AdapterWeightsLoader

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class AutoAdapterWeightsLoader(AdapterWeightsLoader):
    """Load adapter weights across supported formats.

    Uses backend-native safetensors loading when available and falls back
    to torch for .bin/.pt files.
    """

    def load(self, weights_path: Path, backend: "Backend") -> dict[str, Any]:
        suffix = weights_path.suffix.lower()
        if suffix == ".safetensors":
            return backend.load_safetensors(str(weights_path))
        if suffix in (".bin", ".pt"):
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "torch is required to load .bin/.pt adapter files. "
                    "Install with: pip install torch"
                ) from exc
            raw_weights = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            return {key: backend.array(value) for key, value in raw_weights.items()}

        raise ValueError(f"Unsupported adapter weights format: {weights_path}")


__all__ = ["AutoAdapterWeightsLoader"]
