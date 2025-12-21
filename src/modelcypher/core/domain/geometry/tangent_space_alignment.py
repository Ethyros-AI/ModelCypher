"""
Tangent Space Alignment for Local Geometry Comparison.

Ported 1:1 from TrainingCypher/Domain/Geometry/TangentSpaceAlignment.swift.

Measures local geometric agreement by comparing tangent spaces around shared anchors.
Uses principal angles (canonical correlations) between local tangent bases.

Key concepts:
- Tangent space: Local linear approximation of the manifold at a point
- Principal angles: Canonical correlations between subspaces
- Agreement: High cosine = similar local structure, low = different local structure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlx.core as mx


@dataclass
class TangentConfig:
    """Configuration for tangent space alignment."""
    neighbor_count: int = 8
    tangent_rank: int = 4
    min_anchor_count: int = 8
    epsilon: float = 1e-6

    @classmethod
    def default(cls) -> "TangentConfig":
        return cls()


@dataclass
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


@dataclass
class TangentAlignmentReport:
    """Complete tangent space alignment report."""
    source_model: str
    target_model: str
    timestamp: datetime
    config: TangentConfig
    layer_results: List[LayerResult]
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

    Usage:
        aligner = TangentSpaceAlignment()
        result = aligner.compute_layer_metrics(source_points, target_points)
    """

    def __init__(self, config: Optional[TangentConfig] = None):
        self.config = config or TangentConfig.default()

    def compute_layer_metrics(
        self,
        source_points: mx.array,
        target_points: mx.array,
        source_layer: int = 0,
        target_layer: int = 0,
    ) -> Optional[LayerResult]:
        """
        Compute tangent alignment for a single layer pair.

        Args:
            source_points: [n_anchors, dim] source activations
            target_points: [n_anchors, dim] target activations
            source_layer: Source layer index
            target_layer: Target layer index

        Returns:
            LayerResult or None if insufficient data
        """
        n_anchors = min(source_points.shape[0], target_points.shape[0])
        if n_anchors < max(self.config.min_anchor_count, 3):
            return None
        if source_points.shape[0] != target_points.shape[0]:
            return None

        neighbor_count = min(
            max(2, self.config.neighbor_count),
            n_anchors - 1
        )
        tangent_rank = min(
            max(1, self.config.tangent_rank),
            neighbor_count
        )

        # Compute k-nearest neighbors for each point
        source_neighbors = self._compute_neighbors(source_points, neighbor_count)
        target_neighbors = self._compute_neighbors(target_points, neighbor_count)

        cosines: List[float] = []
        angles: List[float] = []
        used_anchors = 0

        for i in range(n_anchors):
            source_basis = self._compute_tangent_basis(
                source_points, i, source_neighbors[i], tangent_rank
            )
            target_basis = self._compute_tangent_basis(
                target_points, i, target_neighbors[i], tangent_rank
            )

            if source_basis is None or target_basis is None:
                continue

            principal_cosines = self._principal_cosines(source_basis, target_basis)
            if not principal_cosines:
                continue

            for cos in principal_cosines:
                clamped = max(0.0, min(1.0, cos))
                cosines.append(clamped)
                angles.append(math.acos(clamped))

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
        points: mx.array,
        k: int,
    ) -> List[List[int]]:
        """Compute k-nearest neighbor indices for each point."""
        n = points.shape[0]
        if n == 0:
            return []

        # Compute pairwise squared distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        sq_norms = mx.sum(points ** 2, axis=1, keepdims=True)
        dots = points @ points.T
        distances = sq_norms + sq_norms.T - 2 * dots
        mx.eval(distances)

        # Convert to Python for neighbor selection
        dist_np = distances.tolist()

        neighbors: List[List[int]] = []
        for i in range(n):
            pairs = [(dist_np[i][j], j) for j in range(n) if j != i]
            pairs.sort(key=lambda x: x[0])
            neighbors.append([p[1] for p in pairs[:k]])

        return neighbors

    def _compute_tangent_basis(
        self,
        points: mx.array,
        anchor_idx: int,
        neighbor_indices: List[int],
        rank: int,
    ) -> Optional[mx.array]:
        """
        Compute local tangent basis at an anchor point.

        Uses SVD on difference vectors to neighbors.
        """
        if len(neighbor_indices) < 2:
            return None

        anchor = points[anchor_idx]
        dim = points.shape[1]

        # Compute difference vectors
        deltas = []
        for idx in neighbor_indices:
            delta = points[idx] - anchor
            deltas.append(delta)

        delta_matrix = mx.stack(deltas)  # [k, dim]

        # Covariance matrix
        cov = delta_matrix.T @ delta_matrix  # [dim, dim]

        # SVD
        try:
            u, s, _ = mx.linalg.svd(cov, stream=mx.cpu)
            mx.eval(u, s)

            # Filter by eigenvalue threshold
            s_list = s.tolist()
            valid_count = sum(1 for v in s_list[:rank] if v > self.config.epsilon)

            if valid_count == 0:
                return None

            # Take top-k eigenvectors as basis
            basis = u[:, :valid_count]  # [dim, rank]
            return basis

        except Exception:
            return None

    def _principal_cosines(
        self,
        basis_a: mx.array,
        basis_b: mx.array,
    ) -> List[float]:
        """
        Compute principal cosines (canonical correlations) between two bases.

        Uses SVD of B_a^T @ B_b.
        """
        if basis_a.shape[0] != basis_b.shape[0]:
            return []

        rank_a = basis_a.shape[1]
        rank_b = basis_b.shape[1]
        rank = min(rank_a, rank_b)

        if rank == 0:
            return []

        # Compute inner products between bases
        m = basis_a[:, :rank].T @ basis_b[:, :rank]

        try:
            _, s, _ = mx.linalg.svd(m, stream=mx.cpu)
            mx.eval(s)

            cosines = s.tolist()
            return [max(0.0, min(1.0 + self.config.epsilon, c)) for c in cosines[:rank]]

        except Exception:
            return []

    def _median(self, values: List[float]) -> float:
        """Compute median of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]


# =============================================================================
# Batch Processing
# =============================================================================

def compute_alignment_for_layers(
    source_activations: Dict[int, mx.array],
    target_activations: Dict[int, mx.array],
    layer_mappings: List[Tuple[int, int]],
    config: Optional[TangentConfig] = None,
) -> TangentAlignmentReport:
    """
    Compute tangent alignment across multiple layer pairs.

    Args:
        source_activations: Dict mapping layer index to activation matrix
        target_activations: Dict mapping layer index to activation matrix
        layer_mappings: List of (source_layer, target_layer) pairs
        config: Alignment configuration

    Returns:
        TangentAlignmentReport with all layer results
    """
    aligner = TangentSpaceAlignment(config)
    results: List[LayerResult] = []
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
            config=config or TangentConfig.default(),
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
        config=config or TangentConfig.default(),
        layer_results=results,
        mean_cosine=mean_cos,
        mean_angle_radians=mean_angle,
        anchor_count=anchor_count,
        layer_count=len(results),
    )
