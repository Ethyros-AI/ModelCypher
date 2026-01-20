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

"""Factory for creating ModelLoaderPort implementations.

This factory handles platform detection and returns the appropriate
model loader for the current environment. It lives in infrastructure
(not ports) to avoid ports importing from adapters.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.model_loader import ModelLoaderPort


def get_model_loader() -> "ModelLoaderPort":
    """Get the appropriate model loader for the current platform.

    Auto-selects:
    - MLXModelLoader on macOS (Metal GPU)
    - CUDAModelLoader on Linux with CUDA
    - JAXModelLoader on Linux/TPU without CUDA

    Returns:
        A ModelLoaderPort instance for the current platform.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    allow_stub = os.environ.get("MC_ALLOW_STUB_INFERENCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if allow_stub:
        from modelcypher.core.domain._backend import get_default_backend

        class _StubTokenizer:
            def __init__(self, vocab_size: int = 16, model_max_length: int = 32) -> None:
                self.vocab_size = vocab_size
                self.model_max_length = model_max_length
                self.eos_token_id = vocab_size - 1

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                if not text.strip():
                    return [0] if add_special_tokens else []
                tokens = []
                for part in text.split():
                    token_id = sum(ord(ch) for ch in part) % (self.vocab_size - 1)
                    tokens.append(token_id)
                return tokens or ([0] if add_special_tokens else [])

            def decode(self, token_ids: list[int]) -> str:
                return " ".join(f"<t{token_id}>" for token_id in token_ids)

        class _StubModel:
            def __init__(self, backend, vocab_size: int) -> None:
                self._backend = backend
                self._vocab_size = vocab_size

            def __call__(self, input_ids):
                seq_len = int(input_ids.shape[1])
                vocab = self._backend.arange(self._vocab_size)
                vocab = vocab + 0.0
                logits = self._backend.tile(vocab, (seq_len, 1))
                return self._backend.expand_dims(logits, axis=0)

        class _StubModelLoader:
            def __init__(self) -> None:
                self._backend = get_default_backend()
                self._tokenizer = _StubTokenizer()
                self._model = _StubModel(self._backend, self._tokenizer.vocab_size)

            def load_model_for_training(
                self,
                model_path: str,
                lora_config: "LoRASettings | None" = None,
                adapter_path: str | None = None,
            ) -> tuple[object, object]:
                return self._model, self._tokenizer

            def load_weights(self, model_path: str) -> "dict[str, object]":
                return {}

        return _StubModelLoader()

    # macOS: Use MLX (Metal GPU)
    if sys.platform == "darwin":
        try:
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader

            return MLXModelLoader()
        except ImportError as e:
            raise RuntimeError(
                "MLX not available on macOS. Install with: pip install mlx mlx-lm"
            ) from e

    # Linux/other: Try CUDA first, then JAX
    try:
        import torch

        if torch.cuda.is_available():
            from modelcypher.adapters.cuda_model_loader import CUDAModelLoader

            return CUDAModelLoader()
    except ImportError:
        pass

    # Try JAX (works on CPU, TPU, and GPU)
    try:
        from modelcypher.adapters.jax_model_loader import JAXModelLoader

        loader = JAXModelLoader()
        if loader.available:
            return loader
    except ImportError:
        pass

    # No suitable backend found
    raise RuntimeError(
        "No model loader available. Install one of:\n"
        "  - macOS: pip install mlx mlx-lm\n"
        "  - CUDA: pip install torch transformers\n"
        "  - JAX/TPU: pip install jax jaxlib transformers"
    )


__all__ = ["get_model_loader"]
