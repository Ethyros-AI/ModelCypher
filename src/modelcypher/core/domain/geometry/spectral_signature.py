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

"""Geodesic spectral signatures for point cloud manifolds.

Builds a graph Laplacian from geodesic distances and reports raw spectral
metrics (eigenvalues, heat trace, entropy) without interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import _set_matrix_element

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
    """Compute geodesic spectral signatures for point clouds."""

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

        adjacency, euclidean_dist, inf_value, k_neighbors = self._build_knn_adjacency(
            points_arr, config.k_neighbors
        )
        backend.eval(adjacency, euclidean_dist)

        euclidean_np = backend.to_numpy(euclidean_dist)
        adj_np = backend.to_numpy(adjacency)

        edge_distances: list[float] = []
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_np[i, j] < inf_value * 0.9:
                    d = float(euclidean_np[i, j])
                    if math.isfinite(d):
                        edge_distances.append(d)
                        edge_count += 1

        kernel_bandwidth = config.kernel_bandwidth
        if kernel_bandwidth is None:
            kernel_bandwidth = _median(edge_distances)

        bandwidth_floor = tiny_value(backend, geo_dist)
        if kernel_bandwidth <= bandwidth_floor:
            kernel_bandwidth = bandwidth_floor

        weights = [[0.0 for _ in range(n)] for _ in range(n)]
        if edge_distances:
            sigma_sq = kernel_bandwidth * kernel_bandwidth * 2.0
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_np[i, j] < inf_value * 0.9:
                        d = float(euclidean_np[i, j])
                        if math.isfinite(d):
                            weight = math.exp(-(d * d) / sigma_sq)
                            weights[i][j] = weight
                            weights[j][i] = weight

        weights_arr = backend.array(weights, dtype="float32")
        backend.eval(weights_arr)
        degree = backend.sum(weights_arr, axis=1)
        backend.eval(degree)

        laplacian = self._build_laplacian(weights_arr, degree, normalized=config.normalized_laplacian)
        backend.eval(laplacian)

        eigvals, _ = backend.eigh(laplacian)
        backend.eval(eigvals)
        eig_np = [float(v) for v in backend.to_numpy(eigvals).tolist()]
        eig_np.sort()

        spectral_entropy = _spectral_entropy(eig_np, regularization_epsilon(backend, eigvals))
        algebraic_connectivity = eig_np[1] if len(eig_np) > 1 else 0.0
        component_count = _count_components(adj_np, inf_value)
        connected = component_count == 1

        eig_arr = backend.array(eig_np, dtype="float32")
        backend.eval(eig_arr)
        heat_trace = _heat_trace(backend, eig_arr, heat_times)

        return SpectralSignatureResult(
            eigenvalues=eig_np,
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
    ) -> tuple["Array", "Array", float, int]:
        backend = self._backend
        n = int(points.shape[0])
        if k_neighbors is None:
            k_neighbors = min(10, n - 1)
        k_neighbors = max(1, min(k_neighbors, n - 1))

        euclidean_dist = self._euclidean_distance_matrix(points)
        backend.eval(euclidean_dist)
        euclidean_np = backend.to_numpy(euclidean_dist)

        inf_val = float(backend.finfo().max) * 0.25
        adj = backend.full((n, n), inf_val)
        for i in range(n):
            adj = _set_matrix_element(backend, adj, i, i, 0.0)

        edge_eps = float(division_epsilon(backend, euclidean_dist))
        for i in range(n):
            dists = euclidean_np[i, :].tolist()
            other_pairs = [(j, dists[j]) for j in range(n) if j != i]
            sorted_pairs = sorted(other_pairs, key=lambda x: x[1])
            nearest_indices = [p[0] for p in sorted_pairs[:k_neighbors]]
            for j in nearest_indices:
                edge_weight = max(dists[j], edge_eps)
                adj = _set_matrix_element(backend, adj, i, j, edge_weight)
                adj = _set_matrix_element(backend, adj, j, i, edge_weight)

        return adj, euclidean_dist, inf_val, k_neighbors

    def _euclidean_distance_matrix(self, points: "Array") -> "Array":
        backend = self._backend
        norms = backend.sum(points * points, axis=1, keepdims=True)
        dist_sq = norms + backend.transpose(norms) - 2.0 * backend.matmul(
            points, backend.transpose(points)
        )
        dist_sq = backend.maximum(dist_sq, 0.0)
        return backend.sqrt(dist_sq)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 1:
        return values_sorted[mid]
    return 0.5 * (values_sorted[mid - 1] + values_sorted[mid])


def _spectral_entropy(eigenvalues: list[float], eps: float) -> float:
    total = sum(eigenvalues)
    if total <= eps:
        return 0.0
    entropy = 0.0
    for value in eigenvalues:
        if value > eps:
            prob = value / total
            entropy -= prob * math.log(prob)
    return entropy


def _heat_trace(backend: "Backend", eigenvalues: "Array", times: list[float]) -> list[float]:
    trace_values: list[float] = []
    for t in times:
        heat = backend.sum(backend.exp(-t * eigenvalues))
        backend.eval(heat)
        trace_values.append(float(backend.to_numpy(heat).item()))
    return trace_values


def _count_components(adjacency: "Array", inf_value: float) -> int:
    n = int(adjacency.shape[0])
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
            for j in range(n):
                if adjacency[node, j] < inf_value * 0.9 and not visited[j]:
                    visited[j] = True
                    stack.append(j)
    return components
