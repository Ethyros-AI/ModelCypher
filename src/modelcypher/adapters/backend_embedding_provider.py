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

"""Backend-based embedding provider using the Backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.ports.embedding import EmbeddingProvider

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class BackendEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that uses Backend protocol for model operations.

    Uses the default backend's model loading and embedding extraction.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: "Backend | None" = None,
    ):
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self._backend = backend
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._hidden_dim: int | None = None

    def _ensure_model(self) -> None:
        """Load model lazily on first use."""
        if self._model is not None:
            return
        if self._model_path is None:
            raise RuntimeError("No model path configured for embedding provider")
        self._model, self._tokenizer = self._backend.load_model(self._model_path)
        self._hidden_dim = self._backend.get_hidden_dim(self._model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the model's hidden states."""
        self._ensure_model()

        results = []
        for text in texts:
            # Get embedding activation (mean of hidden states)
            embedding = self._backend.collect_embedding_activations(
                self._model, self._tokenizer, text
            )
            self._backend.eval(embedding)
            # Convert to list
            embedding_list = self._backend.tolist(embedding)
            if isinstance(embedding_list, list) and embedding_list:
                # Flatten if needed
                if isinstance(embedding_list[0], list):
                    embedding_list = embedding_list[0]
            results.append(embedding_list)

        return results

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._ensure_model()
        return self._hidden_dim or 0


def get_embedding_provider(
    model_path: str | None = None,
    backend: "Backend | None" = None,
) -> EmbeddingProvider:
    """Get a Backend-based embedding provider.

    Args:
        model_path: Optional path to model. If None, must be set later.
        backend: Optional backend. Uses default if not provided.

    Returns:
        EmbeddingProvider instance.
    """
    return BackendEmbeddingProvider(model_path=model_path, backend=backend)
