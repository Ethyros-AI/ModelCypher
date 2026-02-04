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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
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
        if n == 0:
            return mx.zeros((0, self._dim))
        # Random vectors normalized
        vecs = mx.random.normal((n, self._dim))
        backend = get_default_backend()
        norms = geodesic_norms(vecs, backend)
        norms = backend.reshape(norms, (-1, 1))
        # Guard against zero norms
        eps = division_epsilon(backend, norms)
        safe_norms = backend.maximum(norms, backend.full(norms.shape, eps))
        backend.eval(safe_norms)
        return vecs / safe_norms

    async def dimension(self) -> int:
        return self._dim
