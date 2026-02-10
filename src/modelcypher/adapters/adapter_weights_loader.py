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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.adapter_weights import AdapterWeightsLoader

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class AutoAdapterWeightsLoader(AdapterWeightsLoader):
    """Load adapter weights across supported formats.

    Uses backend-native safetensors loading when available and falls back
    to backend binary loading for non-safetensors files.
    """

    def load(self, weights_path: Path, backend: "Backend") -> dict[str, Any]:
        suffix = weights_path.suffix.lower()
        if suffix == ".safetensors":
            return backend.load_safetensors(str(weights_path))
        if suffix in (".bin", ".pt"):
            return backend.load_binary_weights(str(weights_path))

        raise ValueError(f"Unsupported adapter weights format: {weights_path}")


def load_weights_from_paths(
    paths: list[Path],
    backend: "Backend",
    weights_loader: AdapterWeightsLoader | None = None,
) -> dict[str, Any]:
    """Load and merge weights from multiple paths.

    Paths are loaded in order; later files overwrite duplicate keys from earlier
    files, matching common shard loading semantics.
    """
    loader = weights_loader or AutoAdapterWeightsLoader()
    weights: dict[str, Any] = {}

    for path in paths:
        if not path.exists():
            continue
        file_weights = loader.load(path, backend)
        weights.update(file_weights)

    if weights:
        backend.eval(*weights.values())
    return weights


def load_safetensors_from_model_dir(
    model_dir: Path,
    backend: "Backend",
    required_keys: set[str] | None = None,
    weights_loader: AdapterWeightsLoader | None = None,
) -> dict[str, Any]:
    """Load model safetensors from a directory (single or sharded).

    If ``model.safetensors.index.json`` exists, shard resolution is index-aware.
    When ``required_keys`` is provided and index metadata is available, only the
    shards containing those keys are loaded.
    """
    model_dir = model_dir.expanduser().resolve()
    index_file = model_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        if required_keys is None:
            shard_files = sorted(set(weight_map.values()))
        else:
            shard_files = sorted(
                {weight_map[key] for key in required_keys if key in weight_map}
            )

        shard_paths = [model_dir / shard for shard in shard_files]
        weights = load_weights_from_paths(shard_paths, backend, weights_loader)
    else:
        safetensors_paths = sorted(model_dir.glob("*.safetensors"))
        weights = load_weights_from_paths(safetensors_paths, backend, weights_loader)

    if required_keys is None:
        return weights
    return {key: tensor for key, tensor in weights.items() if key in required_keys}


__all__ = [
    "AutoAdapterWeightsLoader",
    "load_weights_from_paths",
    "load_safetensors_from_model_dir",
]
