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

"""Unified model loading - ONE loader that uses Backend.

This is THE model loader. It delegates to Backend.load_model().
NO framework imports here - they live ONLY in backends/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.ports.model_loader import ModelLoaderPort

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class ModelLoader(ModelLoaderPort):
    """Unified model loader - delegates to Backend.

    All framework-specific code lives in the backend implementations.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize with optional backend.

        Args:
            backend: If None, auto-detects from platform.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self._backend = backend

    def load_model(
        self,
        model_path: str,
        adapter_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer via Backend.

        Args:
            model_path: Path to model directory
            adapter_path: Optional adapter directory to load

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path_obj = Path(model_path).expanduser().resolve()
        adapter_path_resolved = None
        if adapter_path:
            adapter_path_resolved = str(Path(adapter_path).expanduser().resolve())

        return self._backend.load_model(str(model_path_obj), adapter_path_resolved)

    def load_weights(self, model_path: str) -> dict[str, Any]:
        """Load model weights as native backend arrays.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to backend arrays
        """
        weights: dict[str, Any] = {}
        for name, tensor in self.iter_weights(model_path):
            weights[name] = tensor
        return weights

    def iter_weights(self, model_path: str) -> "Iterator[tuple[str, Any]]":
        """Stream model weights as (name, tensor) pairs.

        Yields weights in deterministic order:
        1. Safetensor file name order
        2. Tensor key order within each file
        """
        model_dir = Path(model_path).expanduser().resolve()
        safetensor_files = self._resolve_safetensor_files(model_dir)
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        for sf_path in safetensor_files:
            file_weights = self._backend.load_safetensors(str(sf_path))
            keys = sorted(file_weights.keys())
            if keys:
                self._backend.eval(*[file_weights[key] for key in keys])
            for key in keys:
                yield key, file_weights[key]

    def _resolve_safetensor_files(self, model_dir: Path) -> list[Path]:
        """Resolve safetensor files from shard files or index manifest."""
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        if safetensor_files:
            return safetensor_files

        index_file = model_dir / "model.safetensors.index.json"
        if not index_file.exists():
            return []

        with index_file.open(encoding="utf-8") as handle:
            index = json.load(handle)

        shard_files = sorted(set(index.get("weight_map", {}).values()))
        resolved = [model_dir / shard for shard in shard_files]
        return [path for path in resolved if path.exists()]

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """Generate text via Backend.

        Args:
            model: Model from load_model
            tokenizer: Tokenizer from load_model
            prompt: Input prompt
            max_tokens: Max tokens to generate

        Returns:
            Generated text
        """
        return self._backend.generate(model, tokenizer, prompt, max_tokens, **kwargs)


__all__ = ["ModelLoader"]
