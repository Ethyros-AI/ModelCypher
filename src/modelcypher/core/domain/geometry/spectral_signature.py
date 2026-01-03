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

"""Graph spectral signatures for point cloud manifolds.

Builds a k-NN graph Laplacian from pairwise distances and reports raw spectral
metrics (eigenvalues, heat trace, entropy) without interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
    regularization_epsilon,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class SpectralSignatureConfig:
    """Configuration for spectral signature computation."""

    k_neighbors: int | None = None
    kernel_bandwidth: float | None = None
    normalized_laplacian: bool = True
    heat_trace_times: tuple[float, ...] = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class SpectralSignatureResult:
    """Raw spectral signature metrics."""

    eigenvalues: list[float]
    heat_trace: list[float]
    heat_times: list[float]
    spectral_entropy: float
    algebraic_connectivity: float
    component_count: int
    node_count: int
    edge_count: int
    k_neighbors: int
    kernel_bandwidth: float
    normalized_laplacian: bool
    connected: bool


class SpectralSignature:
    """Compute graph spectral signatures for point clouds."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(
        self,
        points: list[list[float]],
        config: SpectralSignatureConfig | None = None,
    ) -> SpectralSignatureResult:
        if config is None:
            config = SpectralSignatureConfig()

        backend = self._backend
        points_arr = backend.array(points)
        backend.eval(points_arr)

        n = int(points_arr.shape[0]) if len(points_arr.shape) > 0 else 0
        heat_times = list(config.heat_trace_times)

        if n == 0:
            return SpectralSignatureResult(
                eigenvalues=[],
                heat_trace=[0.0 for _ in heat_times],
                heat_times=heat_times,
                spectral_entropy=0.0,
                algebraic_connectivity=0.0,
                component_count=0,
                node_count=0,
                edge_count=0,
                k_neighbors=0,
                kernel_bandwidth=0.0,
                normalized_laplacian=config.normalized_laplacian,
                connected=True,
            )

        if n == 1:
            return SpectralSignatureResult(
                eigenvalues=[0.0],
                heat_trace=[1.0 for _ in heat_times],
                heat_times=heat_times,
                spectral_entropy=0.0,
                algebraic_connectivity=0.0,
                component_count=1,
                node_count=1,
                edge_count=0,
                k_neighbors=0,
                kernel_bandwidth=0.0,
                normalized_laplacian=config.normalized_laplacian,
                connected=True,
            )

        # Local edges are Euclidean distances; on the k-NN graph these are exact
        # geodesic segments, and the global geodesic is the shortest path on this graph.
        adjacency, euclidean_dist, inf_value, k_neighbors, neighbor_indices = self._build_knn_adjacency(
            points_arr, config.k_neighbors
        )
        backend.eval(adjacency, euclidean_dist, neighbor_indices)
        edge_mask = adjacency < inf_value * 0.9
        edge_count_total = int(
            backend.to_scalar(backend.sum(backend.astype(edge_mask, "int32")))
        )
        edge_count = max(0, (edge_count_total - n) // 2)

        neighbor_dists = backend.take(euclidean_dist, neighbor_indices, axis=1)
        backend.eval(neighbor_dists)

        kernel_bandwidth = config.kernel_bandwidth
        if kernel_bandwidth is None:
            kernel_bandwidth = _median_flattened(neighbor_dists, backend)

        bandwidth_floor = tiny_value(backend, euclidean_dist)
        if kernel_bandwidth <= bandwidth_floor:
            kernel_bandwidth = bandwidth_floor

        weights_arr = backend.zeros((n, n), dtype="float32")
        if edge_count > 0:
            sigma_sq = kernel_bandwidth * kernel_bandwidth * 2.0
            edge_mask = backend.where(
                adjacency < inf_value * 0.9,
                backend.ones_like(adjacency),
                backend.zeros_like(adjacency),
            )
            weights_arr = backend.exp(-(euclidean_dist * euclidean_dist) / sigma_sq)
            weights_arr = weights_arr * edge_mask
            weights_arr = weights_arr * (1.0 - backend.eye(n))
            weights_arr = backend.astype(weights_arr, "float32")

        backend.eval(weights_arr)
        degree = backend.sum(weights_arr, axis=1)
        backend.eval(degree)

        laplacian = self._build_laplacian(weights_arr, degree, normalized=config.normalized_laplacian)
        backend.eval(laplacian)

        eigvals, _ = backend.eigh(laplacian)
        backend.eval(eigvals)
        eig_list = sorted([float(x) for x in backend.tolist(eigvals)])

        spectral_entropy = _spectral_entropy(eig_list, regularization_epsilon(backend, eigvals))
        algebraic_connectivity = eig_list[1] if len(eig_list) > 1 else 0.0
        neighbor_indices_list = [[int(x) for x in row] for row in backend.tolist(neighbor_indices)]
        component_count = _count_components_from_neighbors(neighbor_indices_list, n)
        connected = component_count == 1

        eig_arr = backend.array(eig_list, dtype="float32")
        backend.eval(eig_arr)
        heat_trace = _heat_trace(backend, eig_arr, heat_times)

        return SpectralSignatureResult(
            eigenvalues=eig_list,
            heat_trace=heat_trace,
            heat_times=heat_times,
            spectral_entropy=spectral_entropy,
            algebraic_connectivity=algebraic_connectivity,
            component_count=component_count,
            node_count=n,
            edge_count=edge_count,
            k_neighbors=k_neighbors,
            kernel_bandwidth=float(kernel_bandwidth),
            normalized_laplacian=config.normalized_laplacian,
            connected=connected,
        )

    def _build_laplacian(
        self,
        weights: "Array",
        degree: "Array",
        *,
        normalized: bool,
    ) -> "Array":
        backend = self._backend
        n = int(weights.shape[0])
        if normalized:
            eps = division_epsilon(backend, degree)
            safe_degree = backend.maximum(degree, eps)
            inv_sqrt = 1.0 / backend.sqrt(safe_degree)
            d_inv_sqrt = backend.diag(inv_sqrt)
            return backend.eye(n) - backend.matmul(d_inv_sqrt, backend.matmul(weights, d_inv_sqrt))
        diag = backend.diag(degree)
        return diag - weights

    def _build_knn_adjacency(
        self,
        points: "Array",
        k_neighbors: int | None,
    ) -> tuple["Array", "Array", float, int, "Array"]:
        backend = self._backend
        n = int(points.shape[0])
        if k_neighbors is None:
            # Use connectivity-based k selection (Berry & Sauer 2016)
            from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

            rg = RiemannianGeometry(backend)
            geo_result = rg.geodesic_distances(points, k_neighbors=None)
            k_neighbors = geo_result.k_neighbors
        k_neighbors = max(1, min(k_neighbors, n - 1))

        inf_val = float(backend.finfo().max) * 0.25
        euclidean_dist = self._euclidean_distance_matrix(points)
        backend.eval(euclidean_dist)

        self_mask = backend.eye(n) > 0.0
        dist_no_self = backend.where(self_mask, inf_val, euclidean_dist)
        neighbor_order = backend.argsort(dist_no_self, axis=1)
        neighbor_indices = neighbor_order[:, :k_neighbors]
        backend.eval(neighbor_order, neighbor_indices)

        adj = backend.full((n, n), inf_val)
        diag_mask = backend.eye(n) > 0.0
        adj = backend.where(diag_mask, backend.zeros_like(adj), adj)

        edge_eps = float(division_epsilon(backend, euclidean_dist))
        edge_eps_arr = backend.full(euclidean_dist.shape, edge_eps)
        weights = backend.maximum(euclidean_dist, edge_eps_arr)

        # Vectorized k-NN mask: rank(i, j) < k_neighbors
        rank = backend.argsort(neighbor_order, axis=1)
        mask = rank < k_neighbors
        mask = backend.where(diag_mask, backend.zeros_like(mask), mask)
        mask_float = backend.astype(mask, "float32")
        mask_sym = backend.maximum(mask_float, backend.transpose(mask_float))

        adj = backend.where(mask_sym > 0.0, weights, adj)

        return adj, euclidean_dist, inf_val, k_neighbors, neighbor_indices

    def _euclidean_distance_matrix(self, points: "Array") -> "Array":
        """Compute pairwise distances using stable direct difference formula."""
        backend = self._backend

        # Force float32 to avoid bfloat16 precision issues
        if hasattr(backend, 'astype'):
            points = backend.astype(points, "float32")

        n = int(points.shape[0])
        d = int(points.shape[1]) if len(points.shape) > 1 else 1

        # Direct difference formula: ||a - b||² = Σ(aᵢ - bᵢ)²
        # This is rotation-invariant and avoids catastrophic cancellation
        points_i = backend.reshape(points, (n, 1, d))
        points_j = backend.reshape(points, (1, n, d))
        diffs = points_i - points_j
        dist_sq = backend.sum(diffs * diffs, axis=2)
        backend.eval(dist_sq)
        dist_sq = backend.maximum(dist_sq, 0.0)
        return backend.sqrt(dist_sq)


def _median_flattened(values: "Array", backend: "Backend") -> float:
    flat = backend.reshape(values, (-1,))
    count = int(flat.shape[0])
    if count == 0:
        return 0.0
    sorted_vals = backend.sort(flat)
    backend.eval(sorted_vals)
    # Use native tolist() for O(1) extraction
    sorted_list = backend.tolist(sorted_vals)
    mid = count // 2
    if count % 2 == 1:
        return float(sorted_list[mid])
    return 0.5 * (float(sorted_list[mid - 1]) + float(sorted_list[mid]))


def _spectral_entropy(eigenvalues: list[float], eps: float) -> float:
    total = sum(eigenvalues)
    if total <= eps:
        return 0.0
    _b = get_default_backend()
    entropy = 0.0
    for value in eigenvalues:
        if value > eps:
            prob = value / total
            entropy -= prob * log_scalar(prob, _b)
    return entropy


def _heat_trace(backend: "Backend", eigenvalues: "Array", times: list[float]) -> list[float]:
    trace_values: list[float] = []
    for t in times:
        heat = backend.sum(backend.exp(-t * eigenvalues))
        backend.eval(heat)
        trace_values.append(float(backend.to_scalar(heat)))
    return trace_values


def _count_components_from_neighbors(neighbors: list[list[int]], n: int) -> int:
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            j_idx = int(j)
            adjacency[i].add(j_idx)
            adjacency[j_idx].add(i)

    visited = [False] * n
    components = 0
    for i in range(n):
        if visited[i]:
            continue
        components += 1
        stack = [i]
        visited[i] = True
        while stack:
            node = stack.pop()
            for j in adjacency[node]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
    return components
