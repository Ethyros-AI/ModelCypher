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

"""Multi-metric topology for β₁ falsification experiments.

Extends existing topology infrastructure with multiple distance metrics
and a graph-based β₁ proxy. This module exists to test whether the
β₁ signal is robust to the choice of distance metric (falsification
test F1) and to evaluate a cheap graph-based proxy for inference-time use.

EXPERIMENTAL: Research code for falsification experiments.
Not promoted to CLI until β₁ survives all three falsification tests.

Distance metrics:
    - Euclidean: chord distance (L2 in ambient space)
    - Cosine: geodesic cosine dissimilarity max(0, 1 - cos_sim)
    - Spectral: Euclidean distance in spectral embedding space
      (effective resistance distance via Varadhan's formula)
    - Geodesic: kNN graph shortest path (current default in TopologicalFingerprint)

Graph β₁ proxy:
    Cycle rank of kNN graph: |E| - |V| + C
    O(n²) vs O(n³) for VR filtration. Viable if Spearman ρ > 0.5
    with VR β₁ across layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

METRIC_NAMES = ("euclidean", "cosine", "spectral", "geodesic")


@dataclass(frozen=True)
class MultiMetricBeta1:
    """β₁ computed under four distance metrics for a single point cloud.

    If all four agree on sign(Δβ₁), the signal is metric-robust.
    If they disagree, the signal may be a metric artifact.
    """

    euclidean_beta1: int
    cosine_beta1: int
    spectral_beta1: int
    geodesic_beta1: int

    def as_dict(self) -> dict[str, int]:
        return {
            "euclidean": self.euclidean_beta1,
            "cosine": self.cosine_beta1,
            "spectral": self.spectral_beta1,
            "geodesic": self.geodesic_beta1,
        }

    def by_name(self, name: str) -> int:
        return self.as_dict()[name]


@dataclass(frozen=True)
class GraphBeta1Result:
    """Cycle rank of kNN graph as cheap β₁ proxy.

    β₁ = |E| - |V| + C  (first Betti number of 1-skeleton)
    where E = edges, V = vertices, C = connected components.
    """

    beta1: int
    n_vertices: int
    n_edges: int
    n_components: int
    k_neighbors: int

    def as_dict(self) -> dict:
        return {
            "beta1": self.beta1,
            "n_vertices": self.n_vertices,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "k_neighbors": self.k_neighbors,
        }


@dataclass(frozen=True)
class MultiMetricTopologySignal:
    """Multi-metric topology analysis for a full trajectory.

    Contains per-layer β₁ under all four metrics plus graph proxy,
    and aggregate statistics for falsification testing.

    Attributes:
        beta1_by_metric_by_layer: Per-layer MultiMetricBeta1.
        delta_beta1_by_metric: metric_name -> Δβ₁ (late - early).
        sign_agreement: Fraction of metrics where sign(Δβ₁) agrees
            with the majority sign. 1.0 = all agree, 0.5 = split.
        graph_beta1_by_layer: Per-layer graph cycle rank.
        graph_delta_beta1: Graph proxy Δβ₁ (late - early).
        layer_indices: Which layers were analyzed.
    """

    beta1_by_metric_by_layer: tuple[MultiMetricBeta1, ...]
    delta_beta1_by_metric: dict[str, float]
    sign_agreement: float
    graph_beta1_by_layer: tuple[int, ...]
    graph_delta_beta1: float
    layer_indices: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "beta1_by_metric_by_layer": [
                m.as_dict() for m in self.beta1_by_metric_by_layer
            ],
            "delta_beta1_by_metric": dict(self.delta_beta1_by_metric),
            "sign_agreement": self.sign_agreement,
            "graph_beta1_by_layer": list(self.graph_beta1_by_layer),
            "graph_delta_beta1": self.graph_delta_beta1,
            "layer_indices": list(self.layer_indices),
        }


def _compute_euclidean_distances(
    points: "Array", backend: "Backend",
) -> list[list[float]]:
    """Pairwise Euclidean distance matrix as Python lists."""
    from modelcypher.core.domain.geometry.spectral_embedding import (
        _chord_distance_matrix,
    )

    dist = _chord_distance_matrix(points, backend)
    backend.eval(dist)
    return backend.tolist(dist)


def _compute_cosine_distances(
    points: "Array", backend: "Backend",
) -> list[list[float]]:
    """Cosine dissimilarity matrix: max(0, 1 - cos_sim) as Python lists.

    Uses geodesic cosine similarity from riemannian_utils, then converts
    similarity to distance via d = max(0, 1 - sim).
    """
    from modelcypher.core.domain.geometry.riemannian_utils import (
        geodesic_cosine_matrix,
    )

    n = int(points.shape[0])
    if n < 3:
        # geodesic_cosine_matrix requires n >= 3; fall back to chord cosine
        cos_sim = _chord_cosine_similarity(points, backend)
    else:
        cos_sim = geodesic_cosine_matrix(points, backend)

    backend.eval(cos_sim)

    # Convert similarity to distance, clamping to [0, 2]
    ones = backend.full(cos_sim.shape, 1.0)
    dist = ones - cos_sim
    zeros = backend.zeros_like(dist)
    dist = backend.maximum(dist, zeros)

    # Zero diagonal
    eye = backend.eye(n)
    dist = dist * (ones - eye)
    backend.eval(dist)

    return backend.tolist(dist)


def _chord_cosine_similarity(
    points: "Array", backend: "Backend",
) -> "Array":
    """Simple chord cosine similarity for small point sets (n < 3)."""
    n = int(points.shape[0])
    eps = machine_epsilon(backend, points)

    norms = backend.sqrt(backend.sum(points * points, axis=1))
    norms = backend.maximum(norms, backend.full(norms.shape, eps))
    backend.eval(norms)

    normalized = points / backend.reshape(norms, (n, 1))
    backend.eval(normalized)

    sim = backend.matmul(normalized, backend.transpose(normalized))

    # Clamp to [-1, 1]
    ones = backend.full(sim.shape, 1.0)
    neg_ones = backend.full(sim.shape, -1.0)
    sim = backend.minimum(sim, ones)
    sim = backend.maximum(sim, neg_ones)
    backend.eval(sim)

    return sim


def _compute_spectral_distances(
    points: "Array", backend: "Backend",
) -> list[list[float]]:
    """Spectral (effective resistance) distance matrix as Python lists.

    Computes spectral embedding, then Euclidean distances in that space.
    Effective resistance captures graph-theoretic distance structure.
    """
    from modelcypher.core.domain.geometry.spectral_embedding import (
        compute_spectral_embedding,
        geodesic_distances_from_embedding,
    )

    n = int(points.shape[0])
    if n < 3:
        # Not enough points for spectral embedding; fall back to Euclidean
        return _compute_euclidean_distances(points, backend)

    embedding_result = compute_spectral_embedding(points, backend)
    dist = geodesic_distances_from_embedding(embedding_result.embedding, backend)
    backend.eval(dist)

    return backend.tolist(dist)


def _compute_geodesic_distances(
    points: "Array", backend: "Backend",
) -> list[list[float]]:
    """Geodesic (kNN graph shortest path) distance matrix as Python lists.

    This is the current default used by TopologicalFingerprint.
    """
    from modelcypher.core.domain.geometry.riemannian_utils import (
        geodesic_distance_matrix,
    )

    n = int(points.shape[0])
    k_neighbors = max(1, n - 1)
    dist = geodesic_distance_matrix(points, k_neighbors=k_neighbors, backend=backend)
    backend.eval(dist)

    return backend.tolist(dist)


def _beta1_from_distance_matrix(distances: list[list[float]]) -> int:
    """Extract β₁ from a distance matrix via VR filtration.

    Reuses TopologicalFingerprint._vietoris_rips_filtration() which accepts
    raw list[list[float]] distance matrices.
    """
    n = len(distances)
    if n < 3:
        return 0

    backend = get_default_backend()

    # Find min/max filtration values
    max_dist = 0.0
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = distances[i][j]
            if d > max_dist:
                max_dist = d
            if 0 < d < min_dist:
                min_dist = d

    if max_dist == 0.0:
        return 0
    if min_dist == float("inf"):
        min_dist = 0.0

    diagram = TopologicalFingerprint._vietoris_rips_filtration(
        distances=distances,
        min_filtration=min_dist,
        max_filtration=max_dist,
    )

    # Use dtype-derived persistence threshold (same as TopologicalFingerprint.compute)
    dist_arr = backend.array(distances)
    eps = machine_epsilon(backend, dist_arr)
    threshold = max_dist * eps

    betti = diagram.betti_numbers(persistence_threshold=threshold)
    return betti.get(1, 0)


def compute_beta1_multi_metric(
    points: "Array",
    backend: "Backend | None" = None,
) -> MultiMetricBeta1:
    """Compute β₁ under four distance metrics for a single point cloud.

    Args:
        points: Point cloud [n, d].
        backend: Backend for tensor operations.

    Returns:
        MultiMetricBeta1 with β₁ under each metric.
    """
    backend = backend or get_default_backend()
    points = backend.astype(points, precision_dtype(backend, reference=points))
    backend.eval(points)

    n = int(points.shape[0])
    if n < 3:
        return MultiMetricBeta1(
            euclidean_beta1=0,
            cosine_beta1=0,
            spectral_beta1=0,
            geodesic_beta1=0,
        )

    euclidean_dist = _compute_euclidean_distances(points, backend)
    cosine_dist = _compute_cosine_distances(points, backend)
    spectral_dist = _compute_spectral_distances(points, backend)
    geodesic_dist = _compute_geodesic_distances(points, backend)

    return MultiMetricBeta1(
        euclidean_beta1=_beta1_from_distance_matrix(euclidean_dist),
        cosine_beta1=_beta1_from_distance_matrix(cosine_dist),
        spectral_beta1=_beta1_from_distance_matrix(spectral_dist),
        geodesic_beta1=_beta1_from_distance_matrix(geodesic_dist),
    )


def compute_graph_beta1(
    points: "Array",
    k_neighbors: int | None = None,
    backend: "Backend | None" = None,
) -> GraphBeta1Result:
    """Compute graph-based β₁ proxy (cycle rank of kNN graph).

    Cycle rank = |E| - |V| + C where:
        E = edges in the symmetric kNN graph
        V = vertices (points)
        C = connected components

    This is the first Betti number of the 1-skeleton (the graph itself),
    not of the full Rips complex. O(n²) vs O(n³) for VR filtration.

    Args:
        points: Point cloud [n, d].
        k_neighbors: Number of neighbors. If None, uses MST max degree.
        backend: Backend for tensor operations.

    Returns:
        GraphBeta1Result with cycle rank and graph statistics.
    """
    from modelcypher.core.domain.geometry.mst import minimum_k_from_mst
    from modelcypher.core.domain.geometry.spectral_embedding import (
        _chord_distance_matrix,
    )

    backend = backend or get_default_backend()
    points = backend.astype(points, precision_dtype(backend, reference=points))
    backend.eval(points)

    n = int(points.shape[0])
    if n < 2:
        return GraphBeta1Result(
            beta1=0, n_vertices=n, n_edges=0, n_components=n,
            k_neighbors=0,
        )

    # Compute Euclidean distances
    dist = _chord_distance_matrix(points, backend)
    backend.eval(dist)

    # Determine k
    if k_neighbors is None:
        k_neighbors, _ = minimum_k_from_mst(dist, backend)
    k_neighbors = max(1, min(k_neighbors, n - 1))

    # Build symmetric kNN adjacency
    # For each point, find k nearest neighbors
    n_edges = 0
    n_components = 0
    adj = [[False] * n for _ in range(n)]

    dist_list = backend.tolist(dist)

    for i in range(n):
        # Get distances from point i to all others
        dists_i = [(dist_list[i][j], j) for j in range(n) if j != i]
        dists_i.sort()
        # Connect to k nearest
        for idx in range(min(k_neighbors, len(dists_i))):
            j = dists_i[idx][1]
            if not adj[i][j]:
                adj[i][j] = True
                adj[j][i] = True  # symmetric

    # Count edges (each undirected edge counted once)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                n_edges += 1

    # Count connected components via BFS
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            n_components += 1
            stack = [i]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                for j in range(n):
                    if adj[v][j] and not visited[j]:
                        stack.append(j)

    # Cycle rank: |E| - |V| + C
    beta1 = n_edges - n + n_components

    return GraphBeta1Result(
        beta1=max(0, beta1),
        n_vertices=n,
        n_edges=n_edges,
        n_components=n_components,
        k_neighbors=k_neighbors,
    )


def _pca_reduce(
    data: "Array", target_dim: int, backend: "Backend",
) -> "Array":
    """PCA dimensionality reduction (center, SVD, project).

    Replicates ReasoningGeometryAnalyzer._pca_reduce pattern.
    """
    mean = backend.mean(data, axis=0)
    backend.eval(mean)
    centered = data - mean
    backend.eval(centered)

    U, S, Vt = backend.svd(centered, full_matrices=False)
    backend.eval(U, S, Vt)

    U_k = U[:, :target_dim]
    S_k = S[:target_dim]
    projected = U_k * S_k
    backend.eval(projected)

    return projected


def _compute_delta(values: list[int | float]) -> float:
    """Compute Δ = mean(late 1/3) - mean(early 1/3)."""
    n = len(values)
    if n < 2:
        return 0.0
    third = max(1, n // 3)
    early = values[:third]
    late = values[-third:]
    mean_early = sum(early) / len(early)
    mean_late = sum(late) / len(late)
    return mean_late - mean_early


def _compute_sign_agreement(deltas: dict[str, float]) -> float:
    """Fraction of metrics whose sign agrees with the majority.

    Returns 1.0 if all metrics agree, lower if they disagree.
    Zero deltas are excluded from voting.
    """
    signs = []
    for d in deltas.values():
        if d > 0:
            signs.append(1)
        elif d < 0:
            signs.append(-1)
        # d == 0 is ambiguous, skip

    if not signs:
        return 1.0  # no signal to disagree on

    n_pos = signs.count(1)
    n_neg = signs.count(-1)
    majority_count = max(n_pos, n_neg)

    return majority_count / len(signs)


def compute_multi_metric_topology_signal(
    layer_hidden_states_sequence: list[dict[int, "Array"]],
    layer_indices: list[int],
    backend: "Backend | None" = None,
    max_tda_points: int = 50,
) -> MultiMetricTopologySignal:
    """Compute multi-metric topology signal for a full trajectory.

    For each layer, collects token hidden states into a point cloud,
    applies PCA if needed, then computes β₁ under all four distance
    metrics plus graph proxy.

    Follows the same pattern as ReasoningGeometryAnalyzer._compute_topology_signal
    but extends to multiple metrics.

    Args:
        layer_hidden_states_sequence: Per-token hidden states. Each entry
            maps layer_idx -> Array[hidden_dim].
        layer_indices: Which layers to analyze.
        backend: Backend for tensor operations.
        max_tda_points: PCA target dimension for high-dimensional data.

    Returns:
        MultiMetricTopologySignal with per-layer multi-metric β₁.
    """
    backend = backend or get_default_backend()
    n_tokens = len(layer_hidden_states_sequence)

    empty = MultiMetricTopologySignal(
        beta1_by_metric_by_layer=(),
        delta_beta1_by_metric={m: 0.0 for m in METRIC_NAMES},
        sign_agreement=1.0,
        graph_beta1_by_layer=(),
        graph_delta_beta1=0.0,
        layer_indices=tuple(layer_indices),
    )

    if n_tokens < 2 or not layer_indices:
        return empty

    multi_beta1_by_layer: list[MultiMetricBeta1] = []
    graph_beta1_by_layer: list[int] = []

    for layer_idx in layer_indices:
        # Collect hidden states at this layer across all tokens
        points_arrays = []
        for token_states in layer_hidden_states_sequence:
            if layer_idx in token_states:
                arr = token_states[layer_idx]
                if not hasattr(arr, "shape"):
                    arr = backend.array(arr)
                backend.eval(arr)
                points_arrays.append(arr)

        if len(points_arrays) < 3:
            # Need >= 3 points for meaningful topology
            multi_beta1_by_layer.append(MultiMetricBeta1(0, 0, 0, 0))
            graph_beta1_by_layer.append(0)
            continue

        # Stack into [n_tokens, hidden_dim]
        point_cloud = backend.stack(points_arrays, axis=0)
        backend.eval(point_cloud)

        # PCA reduction if needed
        n_points = int(backend.shape(point_cloud)[0])
        hidden_dim = int(backend.shape(point_cloud)[1])
        target_dim = min(max_tda_points, n_points - 1, hidden_dim)

        if hidden_dim > target_dim and target_dim > 0:
            point_cloud = _pca_reduce(point_cloud, target_dim, backend)

        # Multi-metric β₁
        mm_beta1 = compute_beta1_multi_metric(point_cloud, backend)
        multi_beta1_by_layer.append(mm_beta1)

        # Graph β₁ proxy
        graph_result = compute_graph_beta1(point_cloud, backend=backend)
        graph_beta1_by_layer.append(graph_result.beta1)

    # Compute Δβ₁ per metric
    delta_by_metric: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        values = [m.by_name(metric_name) for m in multi_beta1_by_layer]
        delta_by_metric[metric_name] = _compute_delta(values)

    # Sign agreement across metrics
    sign_agreement = _compute_sign_agreement(delta_by_metric)

    # Graph proxy Δβ₁
    graph_delta = _compute_delta(graph_beta1_by_layer)

    return MultiMetricTopologySignal(
        beta1_by_metric_by_layer=tuple(multi_beta1_by_layer),
        delta_beta1_by_metric=delta_by_metric,
        sign_agreement=sign_agreement,
        graph_beta1_by_layer=tuple(graph_beta1_by_layer),
        graph_delta_beta1=graph_delta,
        layer_indices=tuple(layer_indices),
    )


__all__ = [
    "METRIC_NAMES",
    "MultiMetricBeta1",
    "GraphBeta1Result",
    "MultiMetricTopologySignal",
    "compute_beta1_multi_metric",
    "compute_graph_beta1",
    "compute_multi_metric_topology_signal",
]
