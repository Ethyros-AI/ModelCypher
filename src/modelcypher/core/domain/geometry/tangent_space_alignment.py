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

"""
Tangent Space Alignment for Local Geometry Comparison.

Ported 1:1 from the reference Swift implementation.

Measures local geometric agreement by comparing tangent spaces around shared anchors.
Uses principal angles (canonical correlations) between local tangent bases.

Key concepts:
- Tangent space: Local linear approximation of the manifold at a point
- Principal angles: Canonical correlations between subspaces
- Agreement: High cosine = similar local structure, low = different local structure
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median as stats_median
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    geodesic_svd,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_distance_matrix

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Minimum anchors required for statistical variance estimation
MIN_ANCHOR_COUNT = 3


@dataclass(frozen=True)
class LayerResult:
    """Tangent alignment metrics for a single layer pair."""

    source_layer: int
    target_layer: int
    anchor_count: int
    neighbor_count: int
    tangent_rank: int
    mean_cosine: float
    min_cosine: float
    max_cosine: float
    mean_angle_radians: float
    median_angle_radians: float
    coverage: float


@dataclass(frozen=True)
class TangentAlignmentReport:
    """Complete tangent space alignment report."""

    source_model: str
    target_model: str
    timestamp: datetime
    layer_results: list[LayerResult]
    mean_cosine: float
    mean_angle_radians: float
    anchor_count: int
    layer_count: int


class TangentSpaceAlignment:
    """
    Measures local geometric agreement via tangent space comparison.

    For each shared anchor point between two models, computes:
    1. Local tangent basis from k-nearest neighbors
    2. Principal angles between tangent bases
    3. Cosine statistics (mean, min, max)

    All parameters (neighbor_count, tangent_rank, epsilon) are derived
    from the data. No configuration needed.

    Usage:
        aligner = TangentSpaceAlignment()
        result = aligner.compute_layer_metrics(source_points, target_points)
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize tangent space alignment.

        Args:
            backend: Optional backend for array operations.
        """
        self._backend = backend or get_default_backend()

    def compute_layer_metrics(
        self,
        source_points: "Array",
        target_points: "Array",
        source_layer: int = 0,
        target_layer: int = 0,
    ) -> LayerResult | None:
        """
        Compute tangent alignment for a single layer pair.

        All parameters are derived from data:
        - neighbor_count: sqrt(n_anchors), clamped to [2, n_anchors-1]
        - tangent_rank: neighbor_count // 2, clamped to [1, neighbor_count]
        - epsilon: dtype-derived numerical stability threshold

        Args:
            source_points: [n_anchors, dim] source activations
            target_points: [n_anchors, dim] target activations
            source_layer: Source layer index
            target_layer: Target layer index

        Returns:
            LayerResult or None if insufficient data
        """
        n_anchors = min(source_points.shape[0], target_points.shape[0])

        if n_anchors < MIN_ANCHOR_COUNT:
            return None
        if source_points.shape[0] != target_points.shape[0]:
            return None

        # Derive neighbor_count from sqrt(n)
        neighbor_count = min(
            max(2, int(sqrt_scalar(float(n_anchors), self._backend))),
            n_anchors - 1,
        )

        # Derive tangent_rank from neighbor_count // 2
        tangent_rank = min(max(1, neighbor_count // 2), neighbor_count)

        # Derive epsilon from data
        eps = division_epsilon(self._backend, source_points)

        b = self._backend
        # Compute k-nearest neighbors for each point
        source_neighbors = self._compute_neighbors(source_points, neighbor_count)
        target_neighbors = self._compute_neighbors(target_points, neighbor_count)

        cosines: list[float] = []
        angles: list[float] = []
        used_anchors = 0

        use_batch = False
        source_bases = None
        target_bases = None
        source_counts: list[int] = []
        target_counts: list[int] = []
        try:
            source_batch = self._compute_tangent_bases_batch(
                source_points, source_neighbors, tangent_rank, eps
            )
            target_batch = self._compute_tangent_bases_batch(
                target_points, target_neighbors, tangent_rank, eps
            )
            if source_batch is not None and target_batch is not None:
                source_bases, source_counts_arr = source_batch
                target_bases, target_counts_arr = target_batch
                b.eval(source_counts_arr, target_counts_arr)
                source_counts = [int(x) for x in b.tolist(source_counts_arr)]
                target_counts = [int(x) for x in b.tolist(target_counts_arr)]
                use_batch = True
        except Exception:
            use_batch = False

        if use_batch:
            rank = int(source_bases.shape[2])
            m_batch = b.matmul(
                b.transpose(source_bases, axes=(0, 2, 1)), target_bases
            )
            s_batch = b.svd(m_batch, compute_uv=False)
            b.eval(s_batch)
            s_list = b.tolist(s_batch)
            for i in range(n_anchors):
                count_a = source_counts[i] if i < len(source_counts) else 0
                count_b = target_counts[i] if i < len(target_counts) else 0
                if count_a <= 0 or count_b <= 0:
                    continue
                limit = min(count_a, count_b, rank)
                if limit <= 0:
                    continue
                principal_cosines = [
                    max(0.0, min(1.0 + eps, float(val)))
                    for val in s_list[i][:limit]
                ]
                if not principal_cosines:
                    continue

                for cos in principal_cosines:
                    clamped = max(0.0, min(1.0, cos))
                    cosines.append(clamped)
                    angles.append(acos_scalar(clamped, self._backend))

                used_anchors += 1
        else:
            source_neighbors_list = b.tolist(source_neighbors)
            target_neighbors_list = b.tolist(target_neighbors)

            for i in range(n_anchors):
                source_basis = self._compute_tangent_basis(
                    source_points, i, source_neighbors_list[i], tangent_rank, eps
                )
                target_basis = self._compute_tangent_basis(
                    target_points, i, target_neighbors_list[i], tangent_rank, eps
                )

                if source_basis is None or target_basis is None:
                    continue

                principal_cosines = self._principal_cosines(source_basis, target_basis, eps)
                if not principal_cosines:
                    continue

                for cos in principal_cosines:
                    clamped = max(0.0, min(1.0, cos))
                    cosines.append(clamped)
                    angles.append(acos_scalar(clamped, self._backend))

                used_anchors += 1

        if not cosines or not angles:
            return None

        coverage = used_anchors / n_anchors

        return LayerResult(
            source_layer=source_layer,
            target_layer=target_layer,
            anchor_count=n_anchors,
            neighbor_count=neighbor_count,
            tangent_rank=tangent_rank,
            mean_cosine=sum(cosines) / len(cosines),
            min_cosine=min(cosines),
            max_cosine=max(cosines),
            mean_angle_radians=sum(angles) / len(angles),
            median_angle_radians=self._median(angles),
            coverage=coverage,
        )

    def _compute_neighbors(
        self,
        points: "Array",
        k: int,
    ) -> "Array":
        """Compute k-nearest neighbor indices for each point."""
        n = int(points.shape[0])
        b = self._backend
        if n == 0:
            return b.zeros((0, 0), dtype="int32")

        k = max(0, int(k))
        if k == 0:
            return b.zeros((n, 0), dtype="int32")

        # Use geodesic distances to respect manifold curvature.
        distances = geodesic_distance_matrix(points, backend=b)
        b.eval(distances)

        max_dist_arr = b.max(distances)
        b.eval(max_dist_arr)
        max_dist = float(b.to_scalar(max_dist_arr))
        eps = division_epsilon(b, distances)
        base = max(max_dist, eps)
        inf_val = min(base / eps, b.finfo(distances.dtype).max)
        dist_no_self = distances + b.eye(n) * inf_val

        kth = max(0, min(k - 1, n - 1))
        partitioned = b.argpartition(dist_no_self, kth, axis=1)
        neighbor_block = partitioned[:, :k]
        b.eval(neighbor_block)
        return b.astype(neighbor_block, "int32")

    def _compute_tangent_bases_batch(
        self,
        points: "Array",
        neighbor_indices: "Array",
        rank: int,
        epsilon: float,
    ) -> tuple["Array", "Array"] | None:
        """Compute tangent bases for all anchors in a batch."""
        b = self._backend
        n = int(points.shape[0])
        if n == 0:
            return None

        shape_neighbors = b.shape(neighbor_indices)
        if len(shape_neighbors) != 2 or shape_neighbors[1] < 2:
            return None

        dim = int(points.shape[1])
        k = int(shape_neighbors[1])
        rank = min(int(rank), dim)
        if rank <= 0:
            return None

        neighbors_flat = b.reshape(neighbor_indices, (-1,))
        neighbor_points_flat = b.take(points, neighbors_flat, axis=0)
        neighbor_points = b.reshape(neighbor_points_flat, (n, k, dim))
        anchors = b.reshape(points, (n, 1, dim))
        delta = neighbor_points - anchors

        cov = b.matmul(b.transpose(delta, axes=(0, 2, 1)), delta)
        # Use eigh for symmetric covariance (more efficient than SVD for symmetric PSD)
        # eigh returns eigenvalues in ascending order, eigenvectors as columns
        eigenvalues, eigenvectors = b.eigh(cov)
        b.eval(eigenvalues, eigenvectors)

        # For symmetric PSD matrices, eigenvalues = singular values
        # Take the largest eigenvalues (from the end due to ascending order)
        s_max = b.max(eigenvalues, axis=1, keepdims=True)
        threshold = s_max * epsilon
        # Check the top rank eigenvalues (from the end)
        mask = eigenvalues[:, -rank:] > threshold
        valid_counts = b.sum(b.astype(mask, "int32"), axis=1)
        b.eval(valid_counts)
        # Return eigenvectors for the largest eigenvalues
        return eigenvectors[:, :, -rank:], valid_counts

    def _compute_tangent_basis(
        self,
        points: "Array",
        anchor_idx: int,
        neighbor_indices: list[int],
        rank: int,
        epsilon: float,
    ) -> "Array | None":
        """
        Compute local tangent basis at an anchor point.

        Uses geodesic SVD on difference vectors to neighbors.
        """
        if len(neighbor_indices) < 2:
            return None

        b = self._backend
        anchor = points[anchor_idx]

        # Compute difference vectors
        idx_arr = b.array(neighbor_indices)
        neighbor_points = b.take(points, idx_arr, axis=0)
        delta_matrix = neighbor_points - anchor  # [k, dim]

        # Covariance matrix
        cov = b.matmul(b.transpose(delta_matrix), delta_matrix)  # [dim, dim]

        # SVD
        try:
            u, s, _ = geodesic_svd(b, cov, k=rank)
            b.eval(u, s)

            # Filter by eigenvalue threshold (relative to max singular value)
            if int(s.shape[0]) > 0:
                s_max_arr = b.max(s)
                b.eval(s_max_arr)
                s_max = float(b.to_scalar(s_max_arr))
            else:
                s_max = 0.0

            # Use relative threshold: eigenvalue must be > epsilon * max_eigenvalue
            # This handles varying scales in the data
            if s_max <= 0:
                return None

            relative_threshold = epsilon * s_max
            mask = s[:rank] > relative_threshold
            valid_count_arr = b.sum(b.astype(mask, "int32"))
            b.eval(valid_count_arr)
            valid_count = int(b.to_scalar(valid_count_arr))

            if valid_count == 0:
                return None

            # Take top-k eigenvectors as basis
            basis = u[:, :valid_count]  # [dim, rank]
            return basis

        except Exception:
            return None

    def _principal_cosines(
        self,
        basis_a: "Array",
        basis_b: "Array",
        epsilon: float,
    ) -> list[float]:
        """
        Compute principal cosines (canonical correlations) between two bases.

        Uses geodesic SVD of B_a^T @ B_b.
        """
        if basis_a.shape[0] != basis_b.shape[0]:
            return []

        rank_a = basis_a.shape[1]
        rank_b = basis_b.shape[1]
        rank = min(rank_a, rank_b)

        if rank == 0:
            return []

        b = self._backend
        # Compute inner products between bases
        m = b.matmul(b.transpose(basis_a[:, :rank]), basis_b[:, :rank])

        try:
            _, s, _ = geodesic_svd(b, m, k=rank)
            b.eval(s)

            cosines = [float(x) for x in b.tolist(s)][:rank]
            return [max(0.0, min(1.0 + epsilon, c)) for c in cosines]

        except Exception:
            return []

    def _median(self, values: list[float]) -> float:
        """Compute median of values.

        Args:
            values: List of float values.

        Returns:
            Median value, or 0.0 if list is empty.
        """
        if not values:
            return 0.0
        return stats_median(values)


# =============================================================================
# Batch Processing
# =============================================================================


def compute_alignment_for_layers(
    source_activations: "dict[int, Array]",
    target_activations: "dict[int, Array]",
    layer_mappings: list[tuple[int, int]],
    backend: "Backend | None" = None,
) -> TangentAlignmentReport:
    """
    Compute tangent alignment across multiple layer pairs.

    All parameters (neighbor_count, tangent_rank, epsilon) are derived
    from the data. No configuration needed.

    Args:
        source_activations: Dict mapping layer index to activation matrix
        target_activations: Dict mapping layer index to activation matrix
        layer_mappings: List of (source_layer, target_layer) pairs
        backend: Optional backend for array operations

    Returns:
        TangentAlignmentReport with all layer results
    """
    aligner = TangentSpaceAlignment(backend)
    results: list[LayerResult] = []
    anchor_count = 0

    for src_layer, tgt_layer in layer_mappings:
        src_pts = source_activations.get(src_layer)
        tgt_pts = target_activations.get(tgt_layer)

        if src_pts is None or tgt_pts is None:
            continue

        result = aligner.compute_layer_metrics(src_pts, tgt_pts, src_layer, tgt_layer)
        if result is not None:
            results.append(result)
            anchor_count = max(anchor_count, result.anchor_count)

    if not results:
        return TangentAlignmentReport(
            source_model="",
            target_model="",
            timestamp=datetime.now(),
            layer_results=[],
            mean_cosine=0.0,
            mean_angle_radians=0.0,
            anchor_count=0,
            layer_count=0,
        )

    mean_cos = sum(r.mean_cosine for r in results) / len(results)
    mean_angle = sum(r.mean_angle_radians for r in results) / len(results)

    return TangentAlignmentReport(
        source_model="source",
        target_model="target",
        timestamp=datetime.now(),
        layer_results=results,
        mean_cosine=mean_cos,
        mean_angle_radians=mean_angle,
        anchor_count=anchor_count,
        layer_count=len(results),
    )
