"""
Manifold Fidelity Sweep: Optimal Subspace Dimension Search.

Ported from TrainingCypher/Domain/Geometry/ManifoldFidelitySweep.swift.

Sweeps alignment ranks to estimate the smallest subspace that preserves manifold fidelity.
Uses multiple metrics to find the "elbow" where additional dimensions provide diminishing returns.

Metrics computed at each rank:
- CKA (Centered Kernel Alignment)
- Procrustes Error
- k-NN Overlap
- Distance Correlation
- Variance Captured
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlx.core as mx


@dataclass
class SweepConfig:
    """Configuration for manifold fidelity sweep."""
    ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    neighbor_count: int = 8
    min_anchor_count: int = 8
    plateau_epsilon: float = 0.02

    @classmethod
    def default(cls) -> "SweepConfig":
        return cls()


@dataclass
class RankMetrics:
    """Metrics for a single rank level."""
    rank: int
    anchor_count: int
    cka: float
    procrustes_error: float
    knn_overlap: float
    distance_correlation: float
    variance_captured_source: float
    variance_captured_target: float


@dataclass
class PlateauSummary:
    """Plateau ranks for each metric."""
    cka: Optional[int] = None
    procrustes_error: Optional[int] = None
    knn_overlap: Optional[int] = None
    distance_correlation: Optional[int] = None
    variance_captured: Optional[int] = None


@dataclass
class LayerSweep:
    """Sweep results for a single layer pair."""
    source_layer: int
    target_layer: int
    anchor_count: int
    metrics: List[RankMetrics]
    plateau: PlateauSummary


@dataclass
class SweepReport:
    """Complete sweep report."""
    source_model: str
    target_model: str
    timestamp: datetime
    config: SweepConfig
    anchor_count: int
    layer_count: int
    ranks: List[int]
    layer_sweeps: List[LayerSweep]
    plateau: PlateauSummary


class ManifoldFidelitySweep:
    """
    Sweeps alignment ranks to find optimal subspace dimension.

    For each layer pair, projects activations to progressively
    higher-dimensional subspaces and measures alignment quality.
    """

    def __init__(self, config: Optional[SweepConfig] = None):
        self.config = config or SweepConfig.default()

    def run_sweep(
        self,
        source_activations: mx.array,
        target_activations: mx.array,
        source_layer: int = 0,
        target_layer: int = 0,
    ) -> Optional[LayerSweep]:
        """
        Run sweep for a single layer pair.

        Args:
            source_activations: [n_anchors, dim] source activations
            target_activations: [n_anchors, dim] target activations
            source_layer: Source layer index
            target_layer: Target layer index

        Returns:
            LayerSweep with metrics at each rank level
        """
        n_anchors = min(source_activations.shape[0], target_activations.shape[0])
        if n_anchors < self.config.min_anchor_count:
            return None

        # Center matrices
        source_centered = self._center(source_activations)
        target_centered = self._center(target_activations)

        # SVD for projection
        source_svd = self._compute_svd(source_centered)
        target_svd = self._compute_svd(target_centered)

        if source_svd is None or target_svd is None:
            return None

        max_rank = min(
            source_svd[1].shape[0],
            target_svd[1].shape[0],
            n_anchors,
        )

        valid_ranks = [r for r in self.config.ranks if r <= max_rank]
        if not valid_ranks:
            return None

        metrics_list: List[RankMetrics] = []

        for rank in valid_ranks:
            # Project to rank-dimensional subspace
            source_proj = self._project(source_centered, source_svd, rank)
            target_proj = self._project(target_centered, target_svd, rank)

            # Compute metrics
            cka = self._compute_cka(source_proj, target_proj)
            procrustes = self._compute_procrustes_error(source_proj, target_proj)
            knn = self._compute_knn_overlap(source_proj, target_proj)
            dist_corr = self._compute_distance_correlation(source_proj, target_proj)
            var_src = self._variance_ratio(source_svd[0], rank)
            var_tgt = self._variance_ratio(target_svd[0], rank)

            metrics_list.append(RankMetrics(
                rank=rank,
                anchor_count=n_anchors,
                cka=cka,
                procrustes_error=procrustes,
                knn_overlap=knn,
                distance_correlation=dist_corr,
                variance_captured_source=var_src,
                variance_captured_target=var_tgt,
            ))

        plateau = self._compute_plateau(metrics_list)

        return LayerSweep(
            source_layer=source_layer,
            target_layer=target_layer,
            anchor_count=n_anchors,
            metrics=metrics_list,
            plateau=plateau,
        )

    def _center(self, x: mx.array) -> mx.array:
        """Center columns to zero mean."""
        return x - mx.mean(x, axis=0, keepdims=True)

    def _compute_svd(self, x: mx.array) -> Optional[Tuple[mx.array, mx.array]]:
        """Compute SVD, return (s, vT)."""
        try:
            _, s, vT = mx.linalg.svd(x, stream=mx.cpu)
            mx.eval(s, vT)
            return (s, vT)
        except Exception:
            return None

    def _project(
        self,
        x: mx.array,
        svd: Tuple[mx.array, mx.array],
        rank: int,
    ) -> mx.array:
        """Project to top-k dimensions using right singular vectors."""
        _, vT = svd
        v_k = vT[:rank].T  # [dim, rank]
        return x @ v_k

    def _variance_ratio(self, s: mx.array, rank: int) -> float:
        """Compute variance explained by top-k singular values."""
        s_sq = s ** 2
        total = float(mx.sum(s_sq).item())
        if total < 1e-10:
            return 0.0
        captured = float(mx.sum(s_sq[:rank]).item())
        return captured / total

    def _compute_cka(self, x: mx.array, y: mx.array) -> float:
        """Linear CKA (Centered Kernel Alignment)."""
        # Gram matrices
        kx = x @ x.T
        ky = y @ y.T

        # HSIC
        hsic_xy = float(mx.sum(kx * ky).item())
        hsic_xx = float(mx.sum(kx * kx).item())
        hsic_yy = float(mx.sum(ky * ky).item())

        denom = math.sqrt(hsic_xx * hsic_yy)
        return hsic_xy / denom if denom > 1e-10 else 0.0

    def _compute_procrustes_error(self, x: mx.array, y: mx.array) -> float:
        """Procrustes distance (normalized reconstruction error)."""
        # M = X^T Y
        m = x.T @ y

        try:
            u, _, vT = mx.linalg.svd(m, stream=mx.cpu)
            mx.eval(u, vT)

            # Optimal rotation: Omega = U @ V^T
            omega = u @ vT

            # Rotated source
            x_rotated = x @ omega

            # Normalized error
            diff = x_rotated - y
            error = float(mx.sum(diff ** 2).item())
            norm_y = float(mx.sum(y ** 2).item())

            return math.sqrt(error / norm_y) if norm_y > 1e-10 else 0.0

        except Exception:
            return 0.0

    def _compute_knn_overlap(self, x: mx.array, y: mx.array, k: int = None) -> float:
        """k-NN neighborhood preservation."""
        if k is None:
            k = min(self.config.neighbor_count, x.shape[0] - 1)

        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute pairwise distances
        def pairwise_dist(pts):
            sq_norms = mx.sum(pts ** 2, axis=1, keepdims=True)
            return sq_norms + sq_norms.T - 2 * (pts @ pts.T)

        dx = pairwise_dist(x)
        dy = pairwise_dist(y)
        mx.eval(dx, dy)

        # Get k-nearest neighbors
        dx_np = dx.tolist()
        dy_np = dy.tolist()

        overlap_sum = 0.0
        for i in range(n):
            x_neighbors = sorted(range(n), key=lambda j: dx_np[i][j] if j != i else float('inf'))[:k]
            y_neighbors = sorted(range(n), key=lambda j: dy_np[i][j] if j != i else float('inf'))[:k]
            overlap = len(set(x_neighbors) & set(y_neighbors))
            overlap_sum += overlap / k

        return overlap_sum / n

    def _compute_distance_correlation(self, x: mx.array, y: mx.array) -> float:
        """Pearson correlation of pairwise distances."""
        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute upper triangular pairwise distances
        def upper_tri_distances(pts):
            dists = []
            pts_list = pts.tolist()
            for i in range(n):
                for j in range(i + 1, n):
                    d = sum((pts_list[i][k] - pts_list[j][k])**2 for k in range(len(pts_list[i])))
                    dists.append(math.sqrt(d))
            return dists

        dx = upper_tri_distances(x)
        dy = upper_tri_distances(y)

        if len(dx) < 2:
            return 0.0

        # Pearson correlation
        mean_x = sum(dx) / len(dx)
        mean_y = sum(dy) / len(dy)

        cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(dx, dy))
        var_x = sum((a - mean_x)**2 for a in dx)
        var_y = sum((b - mean_y)**2 for b in dy)

        denom = math.sqrt(var_x * var_y)
        return cov / denom if denom > 1e-10 else 0.0

    def _compute_plateau(self, metrics: List[RankMetrics]) -> PlateauSummary:
        """Find plateau ranks where metrics stop improving."""
        if not metrics:
            return PlateauSummary()

        sorted_metrics = sorted(metrics, key=lambda m: m.rank)
        eps = self.config.plateau_epsilon

        def find_plateau(values: List[float], higher_better: bool) -> Optional[int]:
            if len(values) < 2:
                return sorted_metrics[0].rank if values else None

            best_idx = 0
            for i in range(1, len(values)):
                if higher_better:
                    improvement = values[i] - values[best_idx]
                else:
                    improvement = values[best_idx] - values[i]

                if improvement > eps:
                    best_idx = i
                # If improvement is small, previous rank was sufficient

            return sorted_metrics[best_idx].rank

        return PlateauSummary(
            cka=find_plateau([m.cka for m in sorted_metrics], higher_better=True),
            procrustes_error=find_plateau([m.procrustes_error for m in sorted_metrics], higher_better=False),
            knn_overlap=find_plateau([m.knn_overlap for m in sorted_metrics], higher_better=True),
            distance_correlation=find_plateau([m.distance_correlation for m in sorted_metrics], higher_better=True),
            variance_captured=find_plateau(
                [0.5 * (m.variance_captured_source + m.variance_captured_target) for m in sorted_metrics],
                higher_better=True
            ),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def find_optimal_rank(
    source_activations: mx.array,
    target_activations: mx.array,
    metric: str = "cka",
    ranks: Optional[List[int]] = None,
) -> Optional[int]:
    """
    Find optimal alignment rank for given metric.

    Args:
        source_activations: Source activation matrix
        target_activations: Target activation matrix
        metric: Which metric to optimize ("cka", "procrustes", "knn", "distance")
        ranks: Ranks to try (default: [4, 8, 16, 32, 64])

    Returns:
        Optimal rank or None if sweep fails
    """
    config = SweepConfig(ranks=ranks or [4, 8, 16, 32, 64])
    sweep = ManifoldFidelitySweep(config)
    result = sweep.run_sweep(source_activations, target_activations)

    if result is None:
        return None

    plateau = result.plateau

    if metric == "cka":
        return plateau.cka
    elif metric == "procrustes":
        return plateau.procrustes_error
    elif metric == "knn":
        return plateau.knn_overlap
    elif metric == "distance":
        return plateau.distance_correlation
    elif metric == "variance":
        return plateau.variance_captured
    else:
        return plateau.cka
