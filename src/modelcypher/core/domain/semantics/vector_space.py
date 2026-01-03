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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.ports.backend import Array, Backend


@dataclass
class ConceptNode:
    id: str
    vector: Array
    metadata: dict[str, str]


class ConceptVectorSpace:
    """
    Manages a high-dimensional vector space for semantic concepts.
    Provides storage and similarity search operations.
    """

    def __init__(self, dimension: int = 4096, backend: Backend | None = None) -> None:
        self.dimension = dimension
        self.concepts: dict[str, ConceptNode] = {}
        self._backend = backend or get_default_backend()

    def add_concept(self, concept_id: str, vector: Array, metadata: dict | None = None) -> None:
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )

        # Normalize on insertion for cosine similarity
        norm = self._backend.norm(vector)
        self._backend.eval(norm)
        norm_val = float(self._backend.to_scalar(norm))
        div_eps = division_epsilon(self._backend, vector)
        # Only add epsilon when norm is near zero to preserve precision
        if norm_val < div_eps:
            normalized = vector / div_eps
        else:
            normalized = vector / norm

        self.concepts[concept_id] = ConceptNode(
            id=concept_id,
            vector=normalized,
            metadata=metadata or {},
        )

    def find_nearest_neighbors(self, query_vector: Array, k: int = 5) -> list[tuple[str, float]]:
        if not self.concepts:
            return []

        # 1. Prepare Query - normalize only if not already unit-length
        q_norm = self._backend.norm(query_vector)
        self._backend.eval(q_norm)
        q_norm_val = float(self._backend.to_scalar(q_norm))
        div_eps = division_epsilon(self._backend, query_vector)
        # Skip normalization if already unit-length (within machine epsilon)
        # This preserves precision when querying with a stored (normalized) vector
        if abs(q_norm_val - 1.0) < div_eps:
            q = query_vector  # Already normalized
        elif q_norm_val < div_eps:
            q = query_vector / div_eps
        else:
            q = query_vector / q_norm

        # 2. Stack Concept Vectors
        ids = list(self.concepts.keys())
        matrix = self._backend.stack([self.concepts[id].vector for id in ids])

        # 3. Compute Cosine Similarity (Dot product of normalized vectors)
        scores = self._backend.matmul(q, self._backend.transpose(matrix))

        # 4. Top K
        # argsort is ascending
        indices = self._backend.argsort(scores)
        # Take last k elements (highest scores) and reverse them
        top_k_indices = indices[-k:][::-1]

        top_scores = self._backend.take(scores, top_k_indices)
        self._backend.eval(top_scores, top_k_indices)

        # Use native tolist() for O(1) extraction
        indices_list = self._backend.tolist(top_k_indices)
        scores_list = self._backend.tolist(top_scores)
        results = [
            (ids[int(idx)], float(score))
            for idx, score in zip(indices_list, scores_list)
        ]

        return results

    def arithmetics(self, positive: list[str], negative: list[str]) -> Array:
        """
        Performs vector arithmetic: sum(pos) - sum(neg)
        """
        result = self._backend.zeros((self.dimension,))

        for p in positive:
            if p in self.concepts:
                result = result + self.concepts[p].vector

        for n in negative:
            if n in self.concepts:
                result = result - self.concepts[n].vector

        return result
