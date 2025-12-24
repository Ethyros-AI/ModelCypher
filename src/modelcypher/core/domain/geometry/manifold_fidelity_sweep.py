"""
Manifold Fidelity Sweep: Optimal Subspace Dimension Search.

Ported from the reference Swift implementation.

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
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class SweepConfig:
    """Configuration for manifold fidelity sweep."""
    ranks: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
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
    cka: int | None = None
    procrustes_error: int | None = None
    knn_overlap: int | None = None
    distance_correlation: int | None = None
    variance_captured: int | None = None


@dataclass
class LayerSweep:
    """Sweep results for a single layer pair."""
    source_layer: int
    target_layer: int
    anchor_count: int
    metrics: list[RankMetrics]
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
    ranks: list[int]
    layer_sweeps: list[LayerSweep]
    plateau: PlateauSummary


class ManifoldFidelitySweep:
    """
    Sweeps alignment ranks to find optimal subspace dimension.

    For each layer pair, projects activations to progressively
    higher-dimensional subspaces and measures alignment quality.
    """

    def __init__(
        self,
        config: SweepConfig | None = None,
        backend: "Backend | None" = None,
    ):
        self.config = config or SweepConfig.default()
        self._backend = backend or get_default_backend()

    def run_sweep(
        self,
        source_activations: "Array",
        target_activations: "Array",
        source_layer: int = 0,
        target_layer: int = 0,
    ) -> LayerSweep | None:
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

        metrics_list: list[RankMetrics] = []

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

    def _center(self, x: "Array") -> "Array":
        """Center columns to zero mean."""
        b = self._backend
        return x - b.mean(x, axis=0, keepdims=True)

    def _compute_svd(self, x: "Array") -> "tuple[Array, Array] | None":
        """Compute SVD, return (s, vT)."""
        b = self._backend
        try:
            _, s, vT = b.svd(x)
            b.eval(s, vT)
            return (s, vT)
        except Exception:
            return None

    def _project(
        self,
        x: "Array",
        svd: "tuple[Array, Array]",
        rank: int,
    ) -> "Array":
        """Project to top-k dimensions using right singular vectors."""
        b = self._backend
        _, vT = svd
        v_k = b.transpose(vT[:rank])  # [dim, rank]
        return b.matmul(x, v_k)

    def _variance_ratio(self, s: "Array", rank: int) -> float:
        """Compute variance explained by top-k singular values."""
        b = self._backend
        s_sq = s ** 2
        total_arr = b.sum(s_sq)
        captured_arr = b.sum(s_sq[:rank])
        b.eval(total_arr, captured_arr)
        total = float(b.to_numpy(total_arr).item())
        if total < 1e-10:
            return 0.0
        captured = float(b.to_numpy(captured_arr).item())
        return captured / total

    def _compute_cka(self, x: "Array", y: "Array") -> float:
        """Linear CKA (Centered Kernel Alignment)."""
        b = self._backend
        # Gram matrices
        kx = b.matmul(x, b.transpose(x))
        ky = b.matmul(y, b.transpose(y))

        # HSIC
        hsic_xy_arr = b.sum(kx * ky)
        hsic_xx_arr = b.sum(kx * kx)
        hsic_yy_arr = b.sum(ky * ky)
        b.eval(hsic_xy_arr, hsic_xx_arr, hsic_yy_arr)

        hsic_xy = float(b.to_numpy(hsic_xy_arr).item())
        hsic_xx = float(b.to_numpy(hsic_xx_arr).item())
        hsic_yy = float(b.to_numpy(hsic_yy_arr).item())

        denom = math.sqrt(hsic_xx * hsic_yy)
        return hsic_xy / denom if denom > 1e-10 else 0.0

    def _compute_procrustes_error(self, x: "Array", y: "Array") -> float:
        """Procrustes distance (normalized reconstruction error)."""
        b = self._backend
        # M = X^T Y
        m = b.matmul(b.transpose(x), y)

        try:
            u, _, vT = b.svd(m)
            b.eval(u, vT)

            # Optimal rotation: Omega = U @ V^T
            omega = b.matmul(u, vT)

            # Rotated source
            x_rotated = b.matmul(x, omega)

            # Normalized error
            diff = x_rotated - y
            error_arr = b.sum(diff ** 2)
            norm_y_arr = b.sum(y ** 2)
            b.eval(error_arr, norm_y_arr)

            error = float(b.to_numpy(error_arr).item())
            norm_y = float(b.to_numpy(norm_y_arr).item())

            return math.sqrt(error / norm_y) if norm_y > 1e-10 else 0.0

        except Exception:
            return 0.0

    def _compute_knn_overlap(self, x: "Array", y: "Array", k: int = None) -> float:
        """k-NN neighborhood preservation."""
        b = self._backend
        if k is None:
            k = min(self.config.neighbor_count, x.shape[0] - 1)

        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute pairwise distances
        def pairwise_dist(pts: "Array") -> "Array":
            sq_norms = b.sum(pts ** 2, axis=1, keepdims=True)
            return sq_norms + b.transpose(sq_norms) - 2 * b.matmul(pts, b.transpose(pts))

        dx = pairwise_dist(x)
        dy = pairwise_dist(y)
        b.eval(dx, dy)

        # Get k-nearest neighbors
        dx_np = b.to_numpy(dx).tolist()
        dy_np = b.to_numpy(dy).tolist()

        overlap_sum = 0.0
        for i in range(n):
            x_neighbors = sorted(range(n), key=lambda j: dx_np[i][j] if j != i else float('inf'))[:k]
            y_neighbors = sorted(range(n), key=lambda j: dy_np[i][j] if j != i else float('inf'))[:k]
            overlap = len(set(x_neighbors) & set(y_neighbors))
            overlap_sum += overlap / k

        return overlap_sum / n

    def _compute_distance_correlation(self, x: "Array", y: "Array") -> float:
        """Pearson correlation of pairwise distances."""
        b = self._backend
        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute upper triangular pairwise distances
        def upper_tri_distances(pts: "Array") -> list[float]:
            dists = []
            pts_list = b.to_numpy(pts).tolist()
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

    def _compute_plateau(self, metrics: list[RankMetrics]) -> PlateauSummary:
        """Find plateau ranks where metrics stop improving."""
        if not metrics:
            return PlateauSummary()

        sorted_metrics = sorted(metrics, key=lambda m: m.rank)
        eps = self.config.plateau_epsilon

        def find_plateau(values: list[float], higher_better: bool) -> int | None:
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
    source_activations: "Array",
    target_activations: "Array",
    metric: str = "cka",
    ranks: list[int] | None = None,
    backend: "Backend | None" = None,
) -> int | None:
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
    sweep = ManifoldFidelitySweep(config, backend=backend)
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
