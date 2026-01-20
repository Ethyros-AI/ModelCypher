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
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_batch
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

    def __init__(self, backend: Backend | None = None) -> None:
        self.dimension: int | None = None
        self.concepts: dict[str, ConceptNode] = {}
        self._backend = backend or get_default_backend()

    def add_concept(self, concept_id: str, vector: Array, metadata: dict | None = None) -> None:
        if self.dimension is None:
            self.dimension = int(vector.shape[0])
        elif vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )

        self.concepts[concept_id] = ConceptNode(
            id=concept_id,
            vector=vector,
            metadata=metadata or {},
        )

    def find_nearest_neighbors(self, query_vector: Array) -> list[tuple[str, float]]:
        if not self.concepts:
            return []

        # 1. Stack Concept Vectors
        ids = list(self.concepts.keys())
        matrix = self._backend.stack([self.concepts[id].vector for id in ids])

        # 2. Compute Geodesic Cosine Similarity
        scores = geodesic_cosine_batch(query_vector, matrix, self._backend)
        self._backend.eval(scores)

        # 3. Select numerically significant neighbors
        max_score = self._backend.max(scores)
        self._backend.eval(max_score)
        max_score_val = float(self._backend.to_scalar(max_score))
        threshold = max_score_val * regularization_epsilon(self._backend, scores)
        if max_score_val <= threshold:
            return []
        mask = scores >= threshold
        count_arr = self._backend.sum(self._backend.astype(mask, "int32"))
        self._backend.eval(count_arr)
        k = int(self._backend.to_scalar(count_arr))
        if k <= 0:
            return []

        neg_scores = -scores
        kth = max(0, k - 1)
        partitioned = self._backend.argpartition(neg_scores, kth)
        top_k_indices = self._backend.take(partitioned, self._backend.arange(k), axis=0)
        top_scores = self._backend.take(scores, top_k_indices)
        order = self._backend.argsort(-top_scores)
        top_k_indices = self._backend.take(top_k_indices, order, axis=0)
        top_scores = self._backend.take(top_scores, order, axis=0)
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
        if self.dimension is None:
            raise ValueError("Concept space is empty; dimension is undefined.")
        result = self._backend.zeros((self.dimension,))

        for p in positive:
            if p in self.concepts:
                result = result + self.concepts[p].vector

        for n in negative:
            if n in self.concepts:
                result = result - self.concepts[n].vector

        return result
