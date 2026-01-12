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

"""Riemannian sampling mixin for manifold coverage operations.

This mixin provides farthest point sampling and directional coverage methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    is_inf,
    pi_value,
)
from modelcypher.core.domain.geometry.riemannian_types import (
    DirectionalCoverage,
    FarthestPointSamplingResult,
)
from modelcypher.core.domain.geometry.riemannian_validation import (
    derive_k_neighbors,
    set_matrix_element,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class RiemannianSamplingMixin:
    """Mixin providing sampling and coverage analysis methods.

    Requires the base class to have:
    - self._backend: Backend instance
    - self.geodesic_distances(points, k_neighbors) method
    """

    _backend: "Backend"

    def farthest_point_sampling(
        self,
        points: "Array",
        n_samples: int,
    ) -> FarthestPointSamplingResult:
        """
        Select points via geodesic farthest point sampling (maximin design).

        FPS iteratively selects the point that maximizes the minimum geodesic
        distance to the already-selected set. This provides optimal coverage
        of the manifold with a given number of samples.

        Algorithm (O(n * n_samples) with precomputed geodesic matrix):
            1. Start with seed point
            2. For each new sample:
               - Compute min geodesic distance from each point to selected set
               - Select the point with maximum min-distance
            3. Return selected indices

        This is the geodesic analog of ambient FPS, respecting the
        manifold's intrinsic geometry.

        Args:
            points: Point cloud [n, d]
            n_samples: Number of points to select

        Returns:
            FarthestPointSamplingResult with selected indices and coverage stats
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n == 0:
            return FarthestPointSamplingResult(
                selected_indices=[],
                min_distances=backend.zeros((0,)),
                coverage_radius=0.0,
            )

        n_samples = max(1, min(n_samples, n))
        # Compute geodesic distances (cached)
        geo_result = self.geodesic_distances(points, k_neighbors=None)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        finite_mask = backend.isfinite(geo_dist)
        finite_sum = backend.sum(
            backend.where(finite_mask, geo_dist, backend.zeros_like(geo_dist)), axis=1
        )
        finite_count = backend.sum(backend.astype(finite_mask, "int32"), axis=1)
        finite_count = backend.maximum(finite_count, backend.ones_like(finite_count))
        mean_dist = finite_sum / finite_count
        backend.eval(mean_dist)
        seed_idx_arr = backend.argmax(mean_dist)
        backend.eval(seed_idx_arr)
        seed_idx = int(backend.to_scalar(seed_idx_arr))

        # Initialize: select seed
        selected = [seed_idx]
        index_grid = backend.arange(0, n)
        mask = backend.astype(index_grid == seed_idx, "float32")
        backend.eval(mask)

        # Min distance from each point to the selected set
        # Initially, just distance to seed
        min_distances = geo_dist[seed_idx]
        backend.eval(min_distances)

        # Iteratively select farthest point
        for _ in range(n_samples - 1):
            # Find point with maximum min-distance to selected set
            neg_inf = backend.full((n,), float("-inf"))
            masked = backend.where(mask > 0, neg_inf, min_distances)
            masked = backend.where(backend.isfinite(masked), masked, neg_inf)
            backend.eval(masked)

            farthest_idx_arr = backend.argmax(masked)
            backend.eval(farthest_idx_arr)
            farthest_idx = int(backend.to_scalar(farthest_idx_arr))
            selected.append(farthest_idx)

            # Update mask for selected points
            one_hot = backend.astype(index_grid == farthest_idx, "float32")
            mask = backend.minimum(mask + one_hot, backend.ones_like(mask))

            # Update min distances: element-wise minimum with new point's distances
            new_dists = geo_dist[farthest_idx]
            min_distances = backend.minimum(min_distances, new_dists)
            backend.eval(min_distances, mask)

        # Compute coverage radius (max of final min-distances, excluding selected)
        neg_inf = backend.full((n,), float("-inf"))
        masked = backend.where(mask > 0, neg_inf, min_distances)
        masked = backend.where(backend.isfinite(masked), masked, neg_inf)
        backend.eval(masked)
        max_val_arr = backend.max(masked)
        backend.eval(max_val_arr)
        max_val = float(backend.to_scalar(max_val_arr))
        coverage_radius = 0.0 if is_inf(max_val, backend) else max(0.0, max_val)

        return FarthestPointSamplingResult(
            selected_indices=selected,
            min_distances=min_distances,
            coverage_radius=coverage_radius,
        )

    def directional_coverage(
        self,
        point_idx: int,
        points: "Array",
    ) -> DirectionalCoverage:
        """
        Analyze directional coverage in tangent space at a point.

        Finds the most under-sampled direction by analyzing the angular
        distribution of neighbors projected onto the tangent sphere.

        Algorithm:
            1. Get k nearest neighbors (by geodesic distance)
            2. Compute tangent vectors to each neighbor
            3. Normalize to unit sphere (tangent sphere S^{d-1})
            4. Find largest angular gap via candidate sampling
            5. Return the sparse direction and coverage metrics

        The sparse direction identifies where to explore for increased coverage.

        Args:
            point_idx: Index of the center point
            points: Point cloud [n, d]
        Returns:
            DirectionalCoverage with sparse direction and metrics
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        d = int(points.shape[1])
        point_idx = max(0, min(point_idx, n - 1))
        k = derive_k_neighbors(points, backend)
        k = min(k, n - 1)

        # Early exit for isolated point (no neighbors possible)
        if k == 0:
            sparse_dir = backend.zeros((d,))
            if d > 0:
                sparse_dir = set_matrix_element(
                    backend, backend.reshape(sparse_dir, (1, d)), 0, 0, 1.0
                )
                sparse_dir = backend.reshape(sparse_dir, (d,))
            return DirectionalCoverage(
                sparse_direction=sparse_dir,
                max_gap_angle=pi_value(backend),  # Full hemisphere is empty
                coverage_variance=float("nan"),
                neighbor_directions=backend.zeros((0, d)),
                point_idx=point_idx,
            )

        center = points[point_idx]

        # Get geodesic distances for neighbor selection
        geo_result = self.geodesic_distances(points, k_neighbors=k)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        # Find k nearest neighbors by geodesic distance
        row = backend.take(geo_dist, backend.array([point_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        inf = float("inf")
        row_masked = backend.where(
            backend.arange(0, n) == point_idx,
            backend.full((n,), inf),
            row,
        )
        kth = max(0, min(k - 1, n - 1))
        partitioned = backend.argpartition(row_masked, kth)
        neighbors = partitioned[:k]

        if int(neighbors.shape[0]) == 0:
            # Isolated point - any direction is sparse
            sparse_dir = backend.zeros((d,))
            if d > 0:
                sparse_dir = set_matrix_element(
                    backend, backend.reshape(sparse_dir, (1, d)), 0, 0, 1.0
                )
                sparse_dir = backend.reshape(sparse_dir, (d,))
            return DirectionalCoverage(
                sparse_direction=sparse_dir,
                max_gap_angle=pi_value(backend),  # Full hemisphere is empty
                coverage_variance=float("nan"),
                neighbor_directions=backend.zeros((0, d)),
                point_idx=point_idx,
            )

        # Compute tangent vectors to neighbors
        neighbor_pts = backend.take(points, neighbors, axis=0)
        tangent_vecs = neighbor_pts - backend.reshape(center, (1, d))

        # Normalize to unit tangent sphere.
        # INTENTIONAL TANGENT: The tangent space is the local linear
        # approximation to the manifold. Normalizing tangent vectors uses the
        # inherited tangent metric in the embedding coordinates.
        norms = backend.sqrt(backend.sum(tangent_vecs * tangent_vecs, axis=1, keepdims=True))
        eps = division_epsilon(backend, norms)
        norms_safe = backend.maximum(norms, backend.full(norms.shape, eps))
        tangent_dirs = tangent_vecs / norms_safe
        backend.eval(tangent_dirs)

        # Candidate directions derived from neighbor directions and tangent covariance.
        cov = backend.matmul(backend.transpose(tangent_dirs), tangent_dirs) / float(k)
        eigvals, eigvecs = backend.eigh(cov)
        backend.eval(eigvecs)
        eig_dirs = backend.transpose(eigvecs)

        candidates = backend.concatenate(
            [
                tangent_dirs,
                -tangent_dirs,
                eig_dirs,
                -eig_dirs,
            ],
            axis=0,
        )
        cand_norms = backend.sqrt(
            backend.sum(candidates * candidates, axis=1, keepdims=True)
        )
        cand_norms_safe = backend.maximum(cand_norms, backend.full(cand_norms.shape, eps))
        candidates = candidates / cand_norms_safe
        backend.eval(candidates)

        # For each candidate, find minimum cosine similarity to any neighbor direction
        # (cosine = 1 means same direction, -1 means opposite)
        # We want the candidate with the smallest maximum similarity (furthest from all)
        # Equivalently: largest angular gap

        # Compute dot products: candidates @ tangent_dirs.T -> [n_candidates, k_actual]
        similarities = backend.matmul(candidates, backend.transpose(tangent_dirs))
        backend.eval(similarities)

        # For each candidate, find the maximum similarity (closest neighbor direction)
        max_sims = backend.max(similarities, axis=1)  # [n_candidates]
        backend.eval(max_sims)

        # The sparse direction is the candidate with minimum max-similarity
        min_max_sim_arr = backend.min(max_sims)
        sparse_idx_arr = backend.argmin(max_sims)
        backend.eval(min_max_sim_arr, sparse_idx_arr)
        min_max_sim = float(backend.to_scalar(min_max_sim_arr))
        sparse_idx = int(backend.to_scalar(sparse_idx_arr))

        sparse_direction = candidates[sparse_idx]

        # Convert max similarity to angle: theta = arccos(similarity)
        # The "gap" is the angle to the nearest neighbor direction
        # Fix: arccos domain is [-1, 1]
        clamped_sim = max(-1.0, min(1.0, min_max_sim))
        max_gap_angle = acos_scalar(clamped_sim, backend)

        sim_mean = backend.mean(max_sims)
        sim_var = backend.mean((max_sims - sim_mean) ** 2)
        backend.eval(sim_var)
        coverage_variance = float(backend.to_scalar(sim_var))

        return DirectionalCoverage(
            sparse_direction=sparse_direction,
            max_gap_angle=max_gap_angle,
            coverage_variance=coverage_variance,
            neighbor_directions=tangent_dirs,
            point_idx=point_idx,
        )

    def propose_in_sparse_direction(
        self,
        point_idx: int,
        points: "Array",
    ) -> "Array":
        """
        Propose a new point by selecting the neighbor most aligned with
        the sparsest tangent direction.

        Args:
            point_idx: Index of the base point
            points: Point cloud [n, d]
        Returns:
            Proposed new point [d]
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        point_idx = max(0, min(point_idx, n - 1))

        # Get directional coverage analysis
        coverage = self.directional_coverage(point_idx, points)

        # Base point
        base = points[point_idx]

        k = derive_k_neighbors(points, backend)
        k = min(k, n - 1)

        if k == 0:
            return base

        # Choose existing neighbor most aligned with sparse direction.
        geo_result = self.geodesic_distances(points, k_neighbors=k)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        row = backend.take(geo_dist, backend.array([point_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        inf = float("inf")
        row_masked = backend.where(
            backend.arange(0, n) == point_idx,
            backend.full((n,), inf),
            row,
        )
        kth = max(0, min(k - 1, n - 1))
        partitioned = backend.argpartition(row_masked, kth)
        neighbors = partitioned[:k]
        neighbor_pts = backend.take(points, neighbors, axis=0)
        tangent_vecs = neighbor_pts - backend.reshape(base, (1, -1))
        norms = backend.sqrt(backend.sum(tangent_vecs * tangent_vecs, axis=1, keepdims=True))
        eps = division_epsilon(backend, norms)
        norms_safe = backend.maximum(norms, backend.full(norms.shape, eps))
        tangent_dirs = tangent_vecs / norms_safe
        backend.eval(tangent_dirs)

        similarities = backend.matmul(
            tangent_dirs, backend.reshape(coverage.sparse_direction, (-1, 1))
        )
        similarities = backend.squeeze(similarities, axis=1)
        backend.eval(similarities)
        aligned_idx_arr = backend.argmax(similarities)
        backend.eval(aligned_idx_arr)
        aligned_idx = int(backend.to_scalar(aligned_idx_arr))
        proposed = neighbor_pts[aligned_idx]

        return proposed


__all__ = ["RiemannianSamplingMixin"]
