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

from typing import Any

from modelcypher.ports.embedding import EmbeddingProvider


class MLXEmbeddingError(RuntimeError):
    pass


class MLXEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "mlx-community/all-MiniLM-L6-v2-4bit",
        max_length: int = 512,
    ) -> None:
        from modelcypher.core.domain._backend import get_mlx_probe_error, probe_mlx_available

        if not probe_mlx_available(explicit=True):
            detail = get_mlx_probe_error() or "MLX probe failed"
            raise MLXEmbeddingError(
                "MLX backend unavailable for embeddings. "
                f"{detail} (run from Terminal.app or set MC_ALLOW_STUB_EMBEDDINGS=1)."
            )
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise MLXEmbeddingError("mlx is required for MLXEmbeddingProvider") from exc
        try:
            from mlx_embeddings.utils import load
        except ImportError as exc:
            raise MLXEmbeddingError("mlx-embeddings is required for MLXEmbeddingProvider") from exc

        self._mx = mx
        self._model_name = model_name
        self._max_length = max_length
        self._model, self._tokenizer = load(model_name)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise MLXEmbeddingError("Embedding dimension not available yet")
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text inputs using MLX backend.

        Parameters
        ----------
        texts : list of str
            Input texts to embed.

        Returns
        -------
        list of list of float
            Embeddings as nested lists of floats.
        """
        if not texts:
            return []
        # TokenizerWrapper from mlx_embeddings doesn't expose __call__
        # Access the underlying tokenizer and convert numpy to MLX
        raw_tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        np_inputs = raw_tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        # Convert numpy arrays to MLX arrays
        inputs = {k: self._mx.array(v) for k, v in np_inputs.items()}
        outputs = self._model(**inputs)
        embeddings = self._extract_embeddings(outputs, inputs)
        self._mx.eval(embeddings)
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
        # Use 1e-6 for mixed precision safety (1e-8 too small for float16)
        denom = self._mx.maximum(
            self._mx.sum(mask, axis=1, keepdims=True),
            self._mx.array(1e-6),
        )
        return self._mx.sum(masked, axis=1) / denom
