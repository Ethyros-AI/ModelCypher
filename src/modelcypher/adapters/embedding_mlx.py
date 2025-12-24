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

from dataclasses import dataclass
from typing import Any

from modelcypher.backends.safe_gpu import SafeGPU
from modelcypher.ports.embedding import EmbeddingProvider


class MLXEmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MLXEmbeddingConfig:
    model_name: str = "mlx-community/all-MiniLM-L6-v2-4bit"
    max_length: int = 512


class MLXEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: MLXEmbeddingConfig) -> None:
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise MLXEmbeddingError("mlx is required for MLXEmbeddingProvider") from exc
        try:
            from mlx_embeddings.utils import load
        except ImportError as exc:
            raise MLXEmbeddingError("mlx-embeddings is required for MLXEmbeddingProvider") from exc

        self._mx = mx
        self._safe = SafeGPU(mx)
        self._config = config
        self._model, self._tokenizer = load(config.model_name)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise MLXEmbeddingError("Embedding dimension not available yet")
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        )
        outputs = self._model(**inputs)
        embeddings = self._extract_embeddings(outputs, inputs)
        self._safe.eval(embeddings)
        if embeddings.ndim != 2:
            raise MLXEmbeddingError(f"Unexpected embedding shape: {embeddings.shape}")
        if self._dimension is None:
            self._dimension = int(embeddings.shape[-1])
        return embeddings.tolist()

    def _extract_embeddings(self, outputs: Any, inputs: dict[str, Any]) -> Any:
        if hasattr(outputs, "text_embeds"):
            return outputs.text_embeds
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            return self._mean_pool(outputs.last_hidden_state, inputs.get("attention_mask"))
        raise MLXEmbeddingError("Unsupported embedding output structure")

    def _mean_pool(self, hidden: Any, attention_mask: Any | None) -> Any:
        if attention_mask is None:
            return self._mx.mean(hidden, axis=1)
        mask = attention_mask.astype(self._mx.float32)
        masked = hidden * mask[:, :, None]
        denom = self._mx.sum(mask, axis=1, keepdims=True) + 1e-8
        return self._mx.sum(masked, axis=1) / denom
