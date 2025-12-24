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


from typing import Any
import mlx.core as mx
from modelcypher.ports.async_embeddings import EmbedderPort

class MockMLXEmbedder(EmbedderPort):
    """
    A mock embedder for testing/verification. 
    Returns random compatible vectors.
    """
    def __init__(self, dim: int = 768):
        self._dim = dim
        
    async def embed(self, texts: list[str]) -> Any:
        n = len(texts)
        if n == 0: return mx.zeros((0, self._dim))
        # Random vectors normalized
        vecs = mx.random.normal((n, self._dim))
        norms = mx.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
        
    async def dimension(self) -> int:
        return self._dim
