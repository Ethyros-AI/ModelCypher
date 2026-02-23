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
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.ports.model_loader import ModelLoaderPort

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


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


# Convenience functions for backwards compatibility
def load_model(model_path: str | Path, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load model and tokenizer."""
    return ModelLoader().load_model(str(model_path), adapter_path)


def load_model_for_training(model_path: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load model for training (same as load_model)."""
    return ModelLoader().load_model(model_path, adapter_path)


def get_model_loader(backend: "Backend | None" = None) -> ModelLoader:
    """Get a model loader instance."""
    return ModelLoader(backend)


def load_model_weights_only(
    model_id: str,
    backend: "Backend | None" = None,
) -> dict[str, Any]:
    """Load only the weight tensors from a model without instantiation.

    This is a lightweight function for operations that only need the raw
    weight tensors (e.g., SVD computation for LoRA transfer) without loading
    the full model into memory.

    Args:
        model_id: HuggingFace model ID or local path to model directory

    Returns:
        Dict mapping weight names to backend-native arrays.

    Raises:
        FileNotFoundError: If model weights cannot be found
        RuntimeError: If weight loading fails
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()
    model_path = Path(model_id)

    def _load_paths(paths: list[Path]) -> dict[str, Any]:
        if not paths:
            return {}
        loaded_weights: dict[str, Any] = {}
        for path in paths:
            if not path.exists():
                continue
            shard_weights = b.load_safetensors(str(path))
            loaded_weights.update(shard_weights)
        if loaded_weights:
            b.eval(*loaded_weights.values())
        return loaded_weights

    if model_path.exists():
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index = json.load(f)
                weight_files = sorted(set(index.get("weight_map", {}).values()))
                safetensors_files = [model_path / wf for wf in weight_files]

        weights = _load_paths(safetensors_files)
        if not weights:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        return weights

    # Resolve from Hugging Face Hub into local cache and load through backend.
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_files = list_repo_files(model_id)
        safetensors_files = [f for f in repo_files if f.endswith(".safetensors")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_id}")

        local_paths = [Path(hf_hub_download(model_id, sf_file)) for sf_file in safetensors_files]
        weights = _load_paths(local_paths)
        if not weights:
            raise FileNotFoundError(f"Could not load safetensors for {model_id}")
        return weights
    except Exception as e:
        raise RuntimeError(f"Cannot load weights from {model_id}: {e}") from e


__all__ = [
    "ModelLoader",
    "load_model",
    "load_model_for_training",
    "get_model_loader",
    "load_model_weights_only",
]
