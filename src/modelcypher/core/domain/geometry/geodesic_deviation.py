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

"""Geodesic deviation (Jacobi fields) for trajectory stability analysis.

Measures how nearby geodesics diverge or converge, providing a leading
indicator of trajectory instability before it manifests as high entropy.

Mathematical background:
    The Jacobi equation describes geodesic deviation:
        D²J/dt² + R(J, γ')γ' = 0

    where J is the Jacobi field (deviation vector), γ' is the tangent to
    the reference geodesic, and R is the Riemann curvature tensor.

    In positive curvature regions, geodesics converge (stable).
    In negative curvature regions, geodesics diverge (unstable).

    The deviation rate λ where ||J(t)|| ~ exp(λt) characterizes stability.

Implementation:
    - Shortest paths via Dijkstra's algorithm on k-NN graph (true geodesics)
    - Perturbation evolution via local tangent frame parallel transport
    - Correlation via sign-based proxy (not true Pearson - see docstring)

References:
    - do Carmo, M. P. (1992). "Riemannian Geometry." Chapter 5.
    - O'Neill, B. (1983). "Semi-Riemannian Geometry." Chapter 10.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeodesicDeviationResult:
    """Result of geodesic deviation analysis.

    Attributes:
        reference_path: Indices of the reference geodesic.
        deviation_rates: Exponential rates for each perturbation [n_perturbations].
        mean_deviation_rate: Mean of deviation rates.
        arc_lengths: Cumulative arc length at each step [n_steps].
        separations: Distance from reference at each step [n_perturbations, n_steps].
        curvature_correlation: Sign-based correlation with local Ricci curvature.
    """

    reference_path: tuple[int, ...]
    deviation_rates: "Array"
    mean_deviation_rate: float
    arc_lengths: "Array"
    separations: "Array"
    curvature_correlation: float


class GeodesicDeviationAnalyzer:
    """Analyze geodesic deviation for trajectory stability assessment.

    Measures how nearby geodesics diverge from a reference path, providing
    a leading indicator of geometric instability.

    Uses Dijkstra's algorithm for true shortest paths on the k-NN graph,
    and local tangent frame rotation for parallel transport of perturbation vectors.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_deviation(
        self,
        points: "Array",
        start_idx: int,
        end_idx: int,
        n_perturbations: int = 8,
        perturbation_scale: float | None = None,
    ) -> GeodesicDeviationResult:
        """Compute geodesic deviation along a reference geodesic.

        Algorithm:
        1. Build k-NN graph and find shortest path via Dijkstra
        2. Perturb the start point in random tangent directions
        3. Parallel transport perturbations along the geodesic
        4. Measure separation vs arc-length along the path
        5. Fit exponential to get deviation rate

        Args:
            points: Point cloud [n, d].
            start_idx: Index of starting point.
            end_idx: Index of ending point.
            n_perturbations: Number of random perturbations.
            perturbation_scale: Scale of perturbations. If None, derived from
                local edge length scaled by sqrt(machine_epsilon) for numerical
                stability in the Jacobi linearization.

        Returns:
            GeodesicDeviationResult with deviation rates and metrics.
        """
        b = self._backend

        points = b.array(points) if not hasattr(points, "shape") else points
        b.eval(points)

        n = int(b.shape(points)[0])
        d = int(b.shape(points)[1])

        eps = division_epsilon(b, points)

        # Build k-NN graph for geodesic computation
        # Use k = ceil(log2(n)) as baseline, with minimum of 3 for connectivity
        # This scales sublinearly with n and provides good graph connectivity
        import math
        k = max(3, min(n - 1, int(math.ceil(math.log2(max(n, 2))))))
        adjacency, edge_lengths = self._build_knn_graph(points, k, b)

        # Find shortest path via Dijkstra
        reference_path, path_distances = self._dijkstra_path(
            adjacency, edge_lengths, start_idx, end_idx, n
        )

        if len(reference_path) < 2:
            # Degenerate case - no path found
            empty_rates = b.zeros((n_perturbations,))
            empty_arc = b.zeros((1,))
            empty_sep = b.zeros((n_perturbations, 1))
            return GeodesicDeviationResult(
                reference_path=(start_idx, end_idx),
                deviation_rates=empty_rates,
                mean_deviation_rate=0.0,
                arc_lengths=empty_arc,
                separations=empty_sep,
                curvature_correlation=0.0,
            )

        # Compute arc lengths along reference path
        arc_lengths = self._compute_arc_lengths(points, reference_path, b)

        # Derive perturbation scale from local geometry
        if perturbation_scale is None:
            # Use local edge length scaled by sqrt(eps) for numerical stability.
            # sqrt(eps) ~ 1e-4 for float32 ensures:
            # 1. Large enough to measure above numerical noise
            # 2. Small enough for Jacobi linearization to hold
            local_edge = self._get_local_edge_length(points, start_idx, adjacency, b)
            sqrt_eps = sqrt_scalar(machine_epsilon(b, points), b)
            perturbation_scale = local_edge * sqrt_eps

        perturbation_scale = max(perturbation_scale, eps)

        # Import parallel transport
        from modelcypher.core.domain.geometry.parallel_transport import (
            ParallelTransporter,
        )

        transporter = ParallelTransporter(b)

        # Generate random perturbation directions
        perturbation_dirs = self._generate_random_directions(d, n_perturbations, b)

        # Get start point
        start_point = points[start_idx]
        b.eval(start_point)

        # Project perturbation directions into tangent subspace at start point
        # This avoids spurious drift from null-space rotation when r < d
        distances = transporter._compute_distance_matrix(points, b)
        frame_result = transporter._compute_local_frame(
            points, start_idx, distances, b, reference_direction=None
        )
        if frame_result is not None:
            frame, eigenvalues = frame_result
            # Estimate intrinsic dimension from eigenvalue spectrum
            r = transporter._estimate_intrinsic_dim_from_eigenvalues(eigenvalues, b)
            if 0 < r < d:
                # Extract tangent basis (first r columns)
                # Guard: r == 0 means degenerate manifold, skip projection
                tangent_basis = frame[:, :r]  # [d, r]
                b.eval(tangent_basis)

                # Project each direction into tangent subspace and re-normalize
                projected_dirs = []
                for p in range(n_perturbations):
                    dir_p = perturbation_dirs[p]
                    b.eval(dir_p)
                    # Project: v_tangent = tangent @ tangent.T @ v
                    proj = b.matmul(tangent_basis, b.matmul(b.transpose(tangent_basis), dir_p))
                    b.eval(proj)
                    # Re-normalize
                    proj_norm = b.sqrt(b.sum(proj * proj))
                    b.eval(proj_norm)
                    proj_norm_val = float(b.to_scalar(proj_norm))
                    if proj_norm_val > eps:
                        proj = proj / proj_norm
                    b.eval(proj)
                    projected_dirs.append(proj)
                perturbation_dirs = b.stack(projected_dirs, axis=0)
                b.eval(perturbation_dirs)

        # For each perturbation, compute deviation trajectory
        separations_list = []
        deviation_rates_list = []
        local_deviation_list = []

        for p in range(n_perturbations):
            pert_dir = perturbation_dirs[p]
            initial_perturbation = perturbation_scale * pert_dir
            b.eval(initial_perturbation)

            # Parallel transport the perturbation along the geodesic
            seps = []
            current_perturbation = initial_perturbation

            for step, ref_idx in enumerate(reference_path):
                # Compute separation at this point
                sep = b.sqrt(b.sum(current_perturbation * current_perturbation))
                b.eval(sep)
                seps.append(float(b.to_scalar(sep)))

                # Transport to next point if not at end
                if step < len(reference_path) - 1:
                    # Use local tangent frame rotation for transport
                    # Create a 2-point path for single step transport
                    transport_result = transporter.transport_along_path(
                        points,
                        [ref_idx, reference_path[step + 1]],
                        current_perturbation,
                    )
                    current_perturbation = transport_result.transported_vector
                    b.eval(current_perturbation)

            seps_arr = b.array(seps)
            separations_list.append(seps_arr)

            # Fit exponential deviation rate
            rate = self._fit_deviation_rate(seps, arc_lengths, b)
            deviation_rates_list.append(rate)

            # Track local deviation for correlation
            if len(seps) >= 2:
                local_deviation_list.append(seps[-1] - seps[0])

        # Stack results
        separations = b.stack(separations_list, axis=0)
        deviation_rates = b.array(deviation_rates_list)
        b.eval(separations, deviation_rates)

        mean_rate = float(b.to_scalar(b.mean(deviation_rates)))

        # Compute sign-based correlation with local curvature
        curvature_correlation = self._compute_curvature_correlation(
            points, reference_path, deviation_rates, b
        )

        arc_lengths_arr = b.array(arc_lengths)
        b.eval(arc_lengths_arr)

        return GeodesicDeviationResult(
            reference_path=tuple(reference_path),
            deviation_rates=deviation_rates,
            mean_deviation_rate=mean_rate,
            arc_lengths=arc_lengths_arr,
            separations=separations,
            curvature_correlation=curvature_correlation,
        )

    def _build_knn_graph(
        self, points: "Array", k: int, backend: "Backend"
    ) -> tuple[dict[int, list[int]], dict[tuple[int, int], float]]:
        """Build k-NN graph with edge lengths.

        Args:
            points: Point cloud [n, d].
            k: Number of neighbors.
            backend: Backend for operations.

        Returns:
            adjacency: Dict mapping node -> list of neighbors
            edge_lengths: Dict mapping (i, j) -> distance
        """
        b = backend
        n = int(b.shape(points)[0])

        # Compute all pairwise distances
        # Expand dims for broadcasting: [n, 1, d] - [1, n, d] = [n, n, d]
        p1 = b.expand_dims(points, 1)  # [n, 1, d]
        p2 = b.expand_dims(points, 0)  # [1, n, d]
        diff = p1 - p2  # [n, n, d]
        distances = b.sqrt(b.sum(diff * diff, axis=2))  # [n, n]
        b.eval(distances)

        adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        edge_lengths: dict[tuple[int, int], float] = {}

        for i in range(n):
            row = distances[i]
            b.eval(row)

            # Get k nearest neighbors (excluding self)
            # Set self-distance to infinity
            row_mod = b.where(
                b.arange(0, n) == i,
                b.full((n,), float("inf")),
                row,
            )
            b.eval(row_mod)

            # Sort and take k smallest
            sorted_idx = b.argsort(row_mod)
            b.eval(sorted_idx)

            for j_pos in range(min(k, n - 1)):
                j = int(b.to_scalar(sorted_idx[j_pos]))
                dist = float(b.to_scalar(row[j]))

                adjacency[i].append(j)
                edge_lengths[(i, j)] = dist
                # Make symmetric (add reverse edge if not already present)
                if i not in adjacency[j]:
                    adjacency[j].append(i)
                edge_lengths[(j, i)] = dist

        return adjacency, edge_lengths

    def _dijkstra_path(
        self,
        adjacency: dict[int, list[int]],
        edge_lengths: dict[tuple[int, int], float],
        start: int,
        end: int,
        n: int,
    ) -> tuple[list[int], list[float]]:
        """Find shortest path using Dijkstra's algorithm.

        Args:
            adjacency: Node -> neighbors mapping.
            edge_lengths: Edge -> distance mapping.
            start: Starting node.
            end: Ending node.
            n: Number of nodes.

        Returns:
            path: List of node indices from start to end.
            distances: Distance to each node in path.
        """
        if start == end:
            return [start], [0.0]

        # Dijkstra's algorithm
        dist = {i: float("inf") for i in range(n)}
        dist[start] = 0.0
        predecessor: dict[int, int | None] = {i: None for i in range(n)}

        # Priority queue: (distance, node)
        pq: list[tuple[float, int]] = [(0.0, start)]
        visited: set[int] = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == end:
                break

            for v in adjacency.get(u, []):
                if v in visited:
                    continue

                edge_key = (u, v)
                if edge_key not in edge_lengths:
                    continue

                alt = dist[u] + edge_lengths[edge_key]
                if alt < dist[v]:
                    dist[v] = alt
                    predecessor[v] = u
                    heapq.heappush(pq, (alt, v))

        # Reconstruct path
        if predecessor[end] is None and start != end:
            # No path found
            return [], []

        path = []
        current: int | None = end
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()

        # Get distances along path
        path_distances = [dist[node] for node in path]

        return path, path_distances

    def _get_local_edge_length(
        self,
        points: "Array",
        idx: int,
        adjacency: dict[int, list[int]],
        backend: "Backend",
    ) -> float:
        """Get local edge length (nearest neighbor distance).

        Args:
            points: Point cloud.
            idx: Point index.
            adjacency: Adjacency dict.
            backend: Backend.

        Returns:
            Minimum edge length from this point.
        """
        b = backend
        p = points[idx]
        b.eval(p)

        neighbors = adjacency.get(idx, [])
        if not neighbors:
            raise ValueError(
                f"k-NN graph has no neighbors for point {idx}; local edge length is undefined."
            )

        min_dist = float("inf")
        for j in neighbors:
            q = points[j]
            diff = p - q
            dist = b.sqrt(b.sum(diff * diff))
            b.eval(dist)
            d = float(b.to_scalar(dist))
            if d < min_dist:
                min_dist = d

        if min_dist == float("inf"):
            raise ValueError(
                f"k-NN graph distances for point {idx} are undefined; local edge length is undefined."
            )
        return min_dist

    def _compute_arc_lengths(
        self,
        points: "Array",
        path: list[int],
        backend: "Backend",
    ) -> list[float]:
        """Compute cumulative arc length along a path.

        Args:
            points: Point cloud [n, d].
            path: List of indices.
            backend: Backend for operations.

        Returns:
            List of cumulative arc lengths (starting from 0).
        """
        b = backend

        arc_lengths = [0.0]
        cumulative = 0.0

        for i in range(len(path) - 1):
            p1 = points[path[i]]
            p2 = points[path[i + 1]]
            diff = p2 - p1
            dist = b.sqrt(b.sum(diff * diff))
            b.eval(dist)
            cumulative += float(b.to_scalar(dist))
            arc_lengths.append(cumulative)

        return arc_lengths

    def _generate_random_directions(
        self,
        d: int,
        n: int,
        backend: "Backend",
    ) -> "Array":
        """Generate n random unit vectors in d dimensions.

        Uses deterministic seed based on dimension for reproducibility.

        Args:
            d: Dimension.
            n: Number of directions.
            backend: Backend for operations.

        Returns:
            Array of unit vectors [n, d].
        """
        b = backend

        # Generate pseudo-random directions using golden angle spacing
        # This provides better coverage than pure random
        directions = []
        phi = (1.0 + 5.0**0.5) / 2.0  # Golden ratio

        for i in range(n):
            vec = []
            for j in range(d):
                # Use golden angle based sequence
                angle = 2.0 * 3.14159265 * ((i * phi + j * 0.5772156649) % 1.0)
                vec.append(float(b.to_scalar(b.sin(b.array([angle])))))

            vec_arr = b.array(vec)
            b.eval(vec_arr)

            # Normalize
            eps = division_epsilon(b, vec_arr)
            norm = b.sqrt(b.sum(vec_arr * vec_arr))
            b.eval(norm)
            norm_val = float(b.to_scalar(norm))

            if norm_val > eps:
                vec_arr = vec_arr / norm
            else:
                # Fallback to basis vector
                basis = [0.0] * d
                basis[i % d] = 1.0
                vec_arr = b.array(basis)

            b.eval(vec_arr)
            directions.append(vec_arr)

        result = b.stack(directions, axis=0)
        b.eval(result)
        return result

    def _fit_deviation_rate(
        self,
        separations: list[float],
        arc_lengths: list[float],
        backend: "Backend",
    ) -> float:
        """Fit exponential deviation rate using linear regression on log.

        Args:
            separations: Separation distances at each step.
            arc_lengths: Arc lengths at each step.
            backend: Backend for operations.

        Returns:
            Deviation rate λ.
        """
        b = backend
        eps = float(machine_epsilon(b, b.array([1.0])))

        # Filter positive separations
        valid_pairs = [
            (s, t) for s, t in zip(separations, arc_lengths) if s > eps and t >= 0
        ]

        if len(valid_pairs) < 2:
            return 0.0

        # Linear regression: log(s) = log(s0) + λ * t
        import math

        log_s = [math.log(max(s, eps)) for s, _ in valid_pairs]
        t_vals = [t for _, t in valid_pairs]

        n = len(valid_pairs)
        sum_t = sum(t_vals)
        sum_log_s = sum(log_s)
        sum_t_log_s = sum(t * ls for t, ls in zip(t_vals, log_s))
        sum_t2 = sum(t * t for t in t_vals)

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < eps:
            return 0.0

        # Slope = deviation rate
        rate = (n * sum_t_log_s - sum_t * sum_log_s) / denom

        return rate

    def _compute_curvature_correlation(
        self,
        points: "Array",
        path: list[int],
        deviation_rates: "Array",
        backend: "Backend",
    ) -> float:
        """Compute sign-based correlation between deviation and curvature.

        Theory: positive curvature -> convergent (negative rate)
                negative curvature -> divergent (positive rate)

        This is NOT a true Pearson correlation because the observations
        are not paired (n_perturbations rates vs len(path) curvatures).
        Instead, we compute whether mean curvature and mean deviation rate
        have the expected opposite-sign relationship.

        Args:
            points: Point cloud.
            path: Reference path indices.
            deviation_rates: Deviation rates for each perturbation.
            backend: Backend for operations.

        Returns:
            Sign-based correlation proxy in [-1, 1]:
            -1.0 if theory holds (opposite signs)
            +1.0 if theory violated (same signs)
            0.0 if insufficient data or near-zero values
        """
        b = backend

        n = int(b.shape(points)[0])
        d = int(b.shape(points)[1])

        # Minimum requirements for meaningful curvature estimation
        min_points_for_curvature = d + 1
        if n < min_points_for_curvature or len(path) < 3:
            return 0.0

        # Compute curvature at each path point
        curvatures = []
        for idx in path:
            try:
                from modelcypher.core.domain.geometry.riemannian_utils import (
                    RiemannianGeometry,
                )

                rg = RiemannianGeometry(b)
                curv_result = rg.estimate_local_curvature(points, idx)
                if curv_result.sectional_curvature is not None:
                    curvatures.append(curv_result.sectional_curvature)
            except Exception as exc:
                logger.debug("Failed to estimate curvature at index %s: %s", idx, exc)

        if len(curvatures) < 2:
            return 0.0

        # Get deviation rates as list
        dev_rates_list = []
        for i in range(int(b.shape(deviation_rates)[0])):
            dev_rates_list.append(float(b.to_scalar(deviation_rates[i])))

        if len(dev_rates_list) < 2:
            return 0.0

        # Compute sign-based correlation between mean deviation rate and curvatures
        # Since we have n_perturbations deviation rates but len(path) curvatures,
        # we correlate mean curvature with each deviation rate, then average

        mean_curv = sum(curvatures) / len(curvatures)
        mean_rate = sum(dev_rates_list) / len(dev_rates_list)

        # Compute variance of curvatures
        var_curv = sum((c - mean_curv) ** 2 for c in curvatures) / len(curvatures)

        # For a proper correlation, we'd need paired observations
        # Since we have one curvature profile and multiple deviation rates,
        # compute correlation of deviation rates with their expected values
        # based on curvature (theory: positive curv -> negative rate)

        # Use dtype-derived epsilon for numerical comparisons
        # Create dummy array to get epsilon from backend
        dummy_arr = backend.array([1.0])
        div_eps = division_epsilon(backend, dummy_arr)

        if var_curv < div_eps or len(dev_rates_list) < 2:
            return 0.0

        var_rate = sum((r - mean_rate) ** 2 for r in dev_rates_list) / len(
            dev_rates_list
        )
        if var_rate < div_eps:
            return 0.0

        # Use sign relationship as proxy for correlation
        # Theory: positive curvature -> convergent (negative rate)
        # So correlation should be negative
        # Compute normalized dot product of (curv - mean) with (rate - mean)

        # Since dimensions don't match, use mean values
        # Positive correlation means they move together (unexpected)
        # Negative correlation means opposite directions (expected)

        # Simple approach: if mean_curv and mean_rate have opposite signs, return -1
        # If same signs, return +1. Scale by confidence based on magnitudes.

        eps = div_eps
        if abs(mean_curv) < eps or abs(mean_rate) < eps:
            return 0.0

        # Normalized product gives correlation-like measure
        # Range: approximately [-1, 1]
        var_curv**0.5
        var_rate**0.5

        # Correlation proxy
        corr = (mean_curv * mean_rate) / (abs(mean_curv) * abs(mean_rate) + eps)

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, corr))


def compute_geodesic_deviation(
    points: "Array",
    start_idx: int,
    end_idx: int,
    n_perturbations: int = 8,
    backend: "Backend | None" = None,
) -> GeodesicDeviationResult:
    """Compute geodesic deviation along a path (convenience function).

    Args:
        points: Point cloud [n, d].
        start_idx: Starting point index.
        end_idx: Ending point index.
        n_perturbations: Number of perturbations.
        backend: Backend for tensor operations.

    Returns:
        GeodesicDeviationResult with deviation analysis.
    """
    analyzer = GeodesicDeviationAnalyzer(backend)
    return analyzer.compute_deviation(points, start_idx, end_idx, n_perturbations)


__all__ = [
    "GeodesicDeviationResult",
    "GeodesicDeviationAnalyzer",
    "compute_geodesic_deviation",
]
