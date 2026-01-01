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

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean


def _frechet_mean_from_ids(
    token_ids: list[int],
    embedding: "object",
    backend: "object",
) -> "object | None":
    if not token_ids:
        return None
    if len(token_ids) == 1:
        return embedding[token_ids[0]]
    idx = backend.array(token_ids)
    vectors = backend.take(embedding, idx, axis=0)
    return _frechet_mean_vectors(vectors, backend)


def _frechet_mean_from_bytes(
    byte_values: list[int],
    byte_map: dict[int, "object"],
    backend: "object",
) -> "object | None":
    if not byte_values:
        return None
    vectors = [byte_map[b] for b in byte_values if b in byte_map]
    if not vectors:
        return None
    if len(vectors) == 1:
        return vectors[0]
    stacked = backend.stack(vectors, axis=0)
    return _frechet_mean_vectors(stacked, backend)


def _frechet_mean_vectors(
    vectors: "object",
    backend: "object",
) -> "object":
    """Compute Frechet mean of vectors with full geodesic precision.

    For byte/vocabulary anchors, we need EXACT alignment - no approximations.
    """
    n_vectors = vectors.shape[0]
    if n_vectors <= 1:
        return vectors[0]

    # Check if vectors are identical (exact match, no computation needed)
    diff = vectors - vectors[:1]
    diff_norm = backend.norm(diff, axis=1)
    max_norm = backend.max(diff_norm)
    backend.eval(max_norm)
    # Dtype-derived epsilon for zero-check
    eps = machine_epsilon(backend, vectors)
    if float(max_norm) < eps:
        return vectors[0]

    k_neighbors = max(1, n_vectors - 1)
    mean = frechet_mean(
        vectors,
        backend=backend,
        k_neighbors=k_neighbors,
        max_k_neighbors=k_neighbors,
    )
    backend.eval(mean)
    return mean
