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

"""Ollivier-Ricci curvature on discrete k-NN manifolds.

Ollivier-Ricci curvature measures how probability mass from neighboring
nodes overlaps, providing a discrete analog of Ricci curvature.

For each edge (x, y) in the k-NN graph:
    kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)

where m_x is a lazy random walk distribution centered at x.

References:
    - Ollivier (2009): "Ricci curvature of Markov chains on metric spaces"
    - NeurIPS 2024: "Exploring Geometric Representational Alignment
      through Ollivier-Ricci Curvature"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_finite,
    log_scalar,
    sqrt_scalar,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeCurvature:
    """Ollivier-Ricci curvature for a single edge in the k-NN graph."""

    source_idx: int
    target_idx: int
    curvature: float  # kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)
    wasserstein_distance: float  # W_1(m_x, m_y)
    edge_weight: float  # d(x, y) from geodesic distance


@dataclass(frozen=True)
class NodeRicciCurvature:
    """Aggregated Ollivier-Ricci curvature at a node."""

    node_idx: int
    mean_curvature: float  # Average over incident edges
    min_curvature: float
    max_curvature: float
    num_edges: int


@dataclass(frozen=True)
class OllivierRicciResult:
    """Complete Ollivier-Ricci curvature analysis of a manifold.

    All values are raw geometric measurements:
    - mean_edge_curvature: Negative = hyperbolic, Positive = spherical
    - std_edge_curvature: Standard deviation for uncertainty estimation
    - mean_node_curvature: Per-node aggregated curvature
    """

    edge_curvatures: list[EdgeCurvature]
    node_curvatures: list[NodeRicciCurvature]
    mean_edge_curvature: float
    std_edge_curvature: float
    mean_node_curvature: float
    k_neighbors: int  # Derived from intrinsic dimension
    n_points: int


class OllivierRicciCurvature:
    """Compute Ollivier-Ricci curvature on the discrete k-NN manifold.

    All parameters are derived from data - no configuration needed.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize Ollivier-Ricci curvature."""
        self._backend = backend or get_default_backend()
        self._derived_base_alpha = 0.5
        self._derived_adaptive_strength = 0.5

    def compute(self, points: "Array") -> OllivierRicciResult:
        """Compute Ollivier-Ricci curvature for all edges in the k-NN graph.

        All parameters are derived from data:
        - k_neighbors: from connectivity (Berry & Sauer 2016)
        - alpha: from graph density
        - adaptive_strength: from degree variance
        """
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n < 2:
            return OllivierRicciResult(
                edge_curvatures=[],
                node_curvatures=[],
                mean_edge_curvature=0.0,
                std_edge_curvature=0.0,
                mean_node_curvature=0.0,
                k_neighbors=1,
                n_points=n,
            )

        # k is derived from connectivity (Berry & Sauer 2016)
        k = None  # Triggers automatic selection

        # 1. Compute geodesic distances
        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        k = geo_result.k_neighbors
        backend.eval(geo_result.distances)

        # 2. Build adjacency list from k-NN graph
        adjacency_list = self._build_adjacency_list(geo_result, k, n)

        # Compute max degree for adaptive alpha
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values()) if adjacency_list else 1

        # Derive base_alpha and adaptive_strength from data
        degrees = [len(neighbors) for neighbors in adjacency_list.values()] if adjacency_list else [1]
        avg_degree = sum(degrees) / len(degrees) if degrees else 1.0

        # Derive from average degree: denser graphs need more self-weight
        self._derived_base_alpha = 1.0 / (1.0 + avg_degree)

        # Derive from coefficient of variation of degrees
        if len(degrees) > 1 and avg_degree > 0:
            deg_var = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
            deg_std = deg_var ** 0.5
            self._derived_adaptive_strength = min(1.0, deg_std / avg_degree)
        else:
            self._derived_adaptive_strength = 0.0

        measures = self._build_lazy_measures(adjacency_list, max_degree, n)
        kernel = self._prepare_sinkhorn_kernel(geo_result.distances)
        cost_matrix = kernel[0]

        # 3. Compute edge curvatures
        edge_curvatures: list[EdgeCurvature] = []
        processed_edges: set[tuple[int, int]] = set()
        eps = division_epsilon(backend, geo_result.distances)

        for source_idx in range(n):
            targets = adjacency_list.get(source_idx, [])
            if not targets:
                continue

            batch_targets: list[int] = []
            for target_idx in targets:
                edge_key = (min(source_idx, target_idx), max(source_idx, target_idx))
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)
                batch_targets.append(target_idx)

            if not batch_targets:
                continue

            targets_arr = backend.array(batch_targets)
            row = backend.take(cost_matrix, backend.array([source_idx]), axis=0)
            row = backend.squeeze(row, axis=0)
            edge_weights_arr = backend.take(row, targets_arr, axis=0)
            finite_mask = backend.isfinite(edge_weights_arr)
            weight_mask = edge_weights_arr >= eps
            valid_mask = finite_mask * weight_mask
            mask_int = backend.astype(valid_mask, "int32")
            count_arr = backend.sum(mask_int)
            backend.eval(edge_weights_arr, count_arr)
            count = int(backend.to_scalar(count_arr))
            if count == 0:
                continue

            neg_mask = -mask_int
            kth = max(0, count - 1)
            partitioned = backend.argpartition(neg_mask, kth)
            valid_idx = backend.take(partitioned, backend.arange(count), axis=0)
            valid_idx = backend.sort(valid_idx)

            valid_targets_arr = backend.take(targets_arr, valid_idx, axis=0)
            valid_weights_arr = backend.take(edge_weights_arr, valid_idx, axis=0)
            source_idx_arr = backend.full((count,), source_idx, dtype="int32")
            mu_batch = backend.take(measures, source_idx_arr, axis=0)
            nu_batch = backend.take(measures, valid_targets_arr, axis=0)
            w1_batch = self._compute_wasserstein_1_batch(
                mu_batch, nu_batch, cost_matrix, kernel=kernel
            )
            backend.eval(w1_batch, valid_targets_arr, valid_weights_arr)
            w1_list = [float(x) for x in backend.tolist(w1_batch)]
            valid_targets = backend.tolist(valid_targets_arr)
            valid_weights = backend.tolist(valid_weights_arr)
            if not isinstance(valid_targets, list):
                valid_targets = [int(valid_targets)]
            if not isinstance(valid_weights, list):
                valid_weights = [float(valid_weights)]

            for target_idx, edge_weight, w1 in zip(valid_targets, valid_weights, w1_list):
                curvature = 1.0 - w1 / edge_weight
                edge_curvatures.append(
                    EdgeCurvature(
                        source_idx=source_idx,
                        target_idx=int(target_idx),
                        curvature=curvature,
                        wasserstein_distance=w1,
                        edge_weight=float(edge_weight),
                    )
                )

        # 4. Aggregate to nodes
        node_curvatures = self._aggregate_to_nodes(edge_curvatures, n)

        # 5. Compute global statistics
        if edge_curvatures:
            edge_kappas = [ec.curvature for ec in edge_curvatures]
            mean_edge = sum(edge_kappas) / len(edge_kappas)
            variance = sum((kappa - mean_edge) ** 2 for kappa in edge_kappas) / len(edge_kappas)
            std_edge = sqrt_scalar(variance, backend)
        else:
            mean_edge = 0.0
            std_edge = 0.0

        if node_curvatures:
            node_means = [nc.mean_curvature for nc in node_curvatures]
            mean_node = sum(node_means) / len(node_means)
        else:
            mean_node = 0.0

        return OllivierRicciResult(
            edge_curvatures=edge_curvatures,
            node_curvatures=node_curvatures,
            mean_edge_curvature=mean_edge,
            std_edge_curvature=std_edge,
            mean_node_curvature=mean_node,
            k_neighbors=k,
            n_points=n,
        )

    def _build_adjacency_list(
        self,
        geo_result: "Any",
        k: int,
        n: int,
    ) -> dict[int, list[int]]:
        """Build adjacency list from geodesic distance result."""
        backend = self._backend
        adjacency: dict[int, list[int]] = {}
        if n <= 0 or k <= 0:
            return {i: [] for i in range(n)}

        distances = geo_result.distances
        dtype = distances.dtype if hasattr(distances, "dtype") else None
        inf_val = float(backend.finfo(dtype).max)

        eye = backend.eye(n)
        row_masked = backend.where(
            eye > 0, backend.full((n, n), inf_val), distances
        )
        kth = max(0, min(k - 1, n - 1))
        partitioned = backend.argpartition(row_masked, kth, axis=1)
        neighbors_arr = partitioned[:, :k]

        row_indices = backend.reshape(backend.arange(n), (n, 1))
        flat_idx = row_indices * n + neighbors_arr
        flat = backend.reshape(row_masked, (-1,))
        neighbor_dists = backend.take(flat, flat_idx, axis=0)
        backend.eval(neighbors_arr, neighbor_dists)

        idx_list = backend.tolist(neighbors_arr)
        dist_list = backend.tolist(neighbor_dists)
        for i in range(n):
            neighbors = [
                int(idx)
                for idx, dist in zip(idx_list[i], dist_list[i])
                if is_finite(float(dist), backend)
            ]
            adjacency[i] = neighbors

        return adjacency

    def _build_lazy_measure(
        self,
        node_idx: int,
        adjacency_list: dict[int, list[int]],
        max_degree: int,
        n_points: int,
    ) -> "Array":
        """Build lazy random walk probability measure at a node.

        m_x = (1-alpha) * delta_x + alpha * uniform(neighbors(x))
        """
        backend = self._backend

        neighbors = adjacency_list.get(node_idx, [])
        degree = len(neighbors)

        base_alpha = self._derived_base_alpha
        adaptive_strength = self._derived_adaptive_strength

        if max_degree > 0:
            alpha = base_alpha * (1.0 - (degree / max_degree) * adaptive_strength)
        else:
            alpha = base_alpha

        alpha = max(0.0, min(1.0, alpha))

        measure_list = [0.0] * n_points
        measure_list[node_idx] = 1.0 - alpha

        if degree > 0:
            neighbor_weight = alpha / degree
            for neighbor_idx in neighbors:
                measure_list[neighbor_idx] += neighbor_weight
        else:
            measure_list[node_idx] = 1.0

        return backend.array(measure_list)

    def _build_lazy_measures(
        self,
        adjacency_list: dict[int, list[int]],
        max_degree: int,
        n_points: int,
    ) -> "Array":
        """Build lazy random walk measures for all nodes."""
        backend = self._backend
        measures = []
        for node_idx in range(n_points):
            measures.append(
                self._build_lazy_measure(node_idx, adjacency_list, max_degree, n_points)
            )
        stacked = backend.stack(measures, axis=0)
        backend.eval(stacked)
        return stacked

    def _prepare_sinkhorn_kernel(
        self,
        cost_matrix: "Array",
    ) -> tuple["Array", "Array", "Array", float, float]:
        """Prepare Sinkhorn kernel from a fixed cost matrix."""
        backend = self._backend
        cost_matrix = backend.array(cost_matrix)

        eps = division_epsilon(backend, cost_matrix)
        floor = tiny_value(backend, cost_matrix)

        finite_mask = backend.isfinite(cost_matrix)
        dtype = cost_matrix.dtype if hasattr(cost_matrix, "dtype") else None
        neg_inf = -float(backend.finfo(dtype).max)
        finite_vals = backend.where(
            finite_mask, cost_matrix, backend.full(cost_matrix.shape, neg_inf)
        )
        finite_max_arr = backend.max(finite_vals)
        backend.eval(finite_max_arr)
        finite_max = float(backend.to_scalar(finite_max_arr))
        finite_max = max(finite_max, eps)
        cost_matrix = backend.where(
            finite_mask,
            cost_matrix,
            backend.full(cost_matrix.shape, finite_max),
        )

        cost_max_arr = backend.max(cost_matrix)
        backend.eval(cost_max_arr)
        cost_max = float(backend.to_scalar(cost_max_arr))
        epsilon = cost_max * eps if cost_max > 0 else eps
        threshold = eps

        cost_min = backend.min(cost_matrix, axis=1, keepdims=True)
        cost_centered = cost_matrix - cost_min
        log_K = -cost_centered / max(epsilon, eps)

        min_log = log_scalar(floor, backend)
        log_K = backend.maximum(log_K, backend.full(log_K.shape, min_log))
        K = backend.exp(log_K)
        K = backend.maximum(K, backend.full(K.shape, floor))
        K_T = backend.transpose(K)
        backend.eval(K, K_T)
        return cost_matrix, K, K_T, floor, threshold

    def _compute_wasserstein_1_batch(
        self,
        mu_batch: "Array",
        nu_batch: "Array",
        cost_matrix: "Array",
        kernel: tuple["Array", "Array", "Array", float, float] | None = None,
    ) -> "Array":
        """Compute Wasserstein-1 distances for a batch of marginals."""
        backend = self._backend
        mu_batch = backend.array(mu_batch)
        nu_batch = backend.array(nu_batch)
        cost_matrix = backend.array(cost_matrix)

        batch = int(mu_batch.shape[0])
        n = int(mu_batch.shape[1])
        if batch == 0 or n == 0:
            return backend.zeros((batch,))

        if kernel is None:
            cost_matrix, K, K_T, floor, threshold = self._prepare_sinkhorn_kernel(
                cost_matrix
            )
        else:
            cost_matrix, K, K_T, floor, threshold = kernel

        # Sinkhorn iteration bound: O(n log n) for transport on n points
        # Derived from theoretical convergence rate of entropy-regularized OT
        max_iter = n * max(1, int(math.ceil(math.log(n + 1))))

        u = backend.ones((batch, n))
        v = backend.ones((batch, n))

        for _ in range(max_iter):
            Kv = backend.matmul(K, backend.reshape(v, (batch, n, 1)))
            Kv = backend.squeeze(Kv, axis=2)
            Kv = backend.maximum(Kv, backend.full(Kv.shape, floor))
            u_new = mu_batch / Kv

            Ktu = backend.matmul(K_T, backend.reshape(u_new, (batch, n, 1)))
            Ktu = backend.squeeze(Ktu, axis=2)
            Ktu = backend.maximum(Ktu, backend.full(Ktu.shape, floor))
            v_new = nu_batch / Ktu

            if threshold > 0:
                u_diff = backend.max(backend.abs(u_new - u), axis=1)
                v_diff = backend.max(backend.abs(v_new - v), axis=1)
                diff_max = backend.maximum(u_diff, v_diff)
                diff_max_arr = backend.max(diff_max)
                backend.eval(diff_max_arr)
                if float(backend.to_scalar(diff_max_arr)) < threshold:
                    u, v = u_new, v_new
                    break

            u, v = u_new, v_new

        gamma = (
            K
            * backend.reshape(u, (batch, n, 1))
            * backend.reshape(v, (batch, 1, n))
        )
        w1 = backend.sum(cost_matrix * gamma, axis=(1, 2))
        backend.eval(w1)
        return w1

    def _aggregate_to_nodes(
        self,
        edge_curvatures: list[EdgeCurvature],
        n_points: int,
    ) -> list[NodeRicciCurvature]:
        """Aggregate edge curvatures to node curvatures."""
        node_curvatures_map: dict[int, list[float]] = {i: [] for i in range(n_points)}

        for ec in edge_curvatures:
            node_curvatures_map[ec.source_idx].append(ec.curvature)
            node_curvatures_map[ec.target_idx].append(ec.curvature)

        result: list[NodeRicciCurvature] = []
        for node_idx in range(n_points):
            curvatures = node_curvatures_map[node_idx]
            if curvatures:
                result.append(
                    NodeRicciCurvature(
                        node_idx=node_idx,
                        mean_curvature=sum(curvatures) / len(curvatures),
                        min_curvature=min(curvatures),
                        max_curvature=max(curvatures),
                        num_edges=len(curvatures),
                    )
                )
            else:
                result.append(
                    NodeRicciCurvature(
                        node_idx=node_idx,
                        mean_curvature=0.0,
                        min_curvature=0.0,
                        max_curvature=0.0,
                        num_edges=0,
                    )
                )

        return result
