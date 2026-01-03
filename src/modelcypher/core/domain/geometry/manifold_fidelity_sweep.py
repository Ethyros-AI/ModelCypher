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

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class SweepConfig:
    """Configuration for manifold fidelity sweep.

    All parameters are derived from data geometry when None:
    - ranks: geometric progression up to data dimension
    - neighbor_count: sqrt(n) scaling
    - min_anchor_count: based on smallest rank
    - plateau_epsilon: from fidelity variance
    """

    # Ranks to sweep: None = derive geometric progression [4, 8, 16, ...] up to dim
    ranks: list[int] | None = None
    # Number of neighbors for k-NN metrics: None = derive from sqrt(n)
    neighbor_count: int | None = None
    # Minimum anchors required: None = derive from smallest rank
    min_anchor_count: int | None = None
    plateau_epsilon: float | None = None  # Derived from fidelity variance if not set

    @classmethod
    def with_parameters(
        cls,
        *,
        ranks: list[int] | None = None,
        neighbor_count: int | None = None,
        min_anchor_count: int | None = None,
        plateau_epsilon: float | None = None,
    ) -> "SweepConfig":
        """Create configuration with explicit parameters.

        Args:
            ranks: Rank levels to sweep (None = derive from data dimension).
            neighbor_count: Number of neighbors (None = derive from sqrt(n)).
            min_anchor_count: Minimum anchors required (None = derive from min rank).
            plateau_epsilon: Epsilon for plateau detection (None = derive from data).

        Returns:
            Configuration with specified parameters.
        """
        if ranks is not None:
            if len(ranks) == 0:
                raise ValueError("ranks must have at least one value")
            if any(r < 1 for r in ranks):
                raise ValueError("All ranks must be >= 1")
        if neighbor_count is not None and neighbor_count < 1:
            raise ValueError(f"neighbor_count must be >= 1, got {neighbor_count}")
        if min_anchor_count is not None and min_anchor_count < 2:
            raise ValueError(f"min_anchor_count must be >= 2, got {min_anchor_count}")
        if plateau_epsilon is not None and plateau_epsilon <= 0:
            raise ValueError(f"plateau_epsilon must be > 0, got {plateau_epsilon}")
        return cls(
            ranks=ranks,
            neighbor_count=neighbor_count,
            min_anchor_count=min_anchor_count,
            plateau_epsilon=plateau_epsilon,
        )


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
        config: SweepConfig,
        backend: "Backend | None" = None,
    ):
        """Initialize with explicit configuration.

        Args:
            config: Sweep configuration (use with_parameters() to create).
            backend: Optional backend for array operations.
        """
        self.config = config
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
        import math

        n_anchors = min(source_activations.shape[0], target_activations.shape[0])
        dim = min(source_activations.shape[1], target_activations.shape[1])

        # Derive ranks from data dimension when not specified
        # Geometric progression [4, 8, 16, ...] up to dimension
        if self.config.ranks is not None:
            ranks = self.config.ranks
        else:
            ranks = []
            r = 4
            while r <= dim:
                ranks.append(r)
                r *= 2
            if not ranks:
                ranks = [min(4, dim)]

        # Derive neighbor_count from sqrt(n) when not specified
        if self.config.neighbor_count is not None:
            neighbor_count = self.config.neighbor_count
        else:
            neighbor_count = max(2, int(math.sqrt(n_anchors)))

        # Derive min_anchor_count from smallest rank when not specified
        if self.config.min_anchor_count is not None:
            min_anchor_count = self.config.min_anchor_count
        else:
            min_anchor_count = max(2, min(ranks) if ranks else 4)

        if n_anchors < min_anchor_count:
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

        valid_ranks = [r for r in ranks if r <= max_rank]
        if not valid_ranks:
            return None

        # Store derived neighbor_count for use in knn computation
        self._derived_neighbor_count = neighbor_count

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

            metrics_list.append(
                RankMetrics(
                    rank=rank,
                    anchor_count=n_anchors,
                    cka=cka,
                    procrustes_error=procrustes,
                    knn_overlap=knn,
                    distance_correlation=dist_corr,
                    variance_captured_source=var_src,
                    variance_captured_target=var_tgt,
                )
            )

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
        s_sq = s**2
        total_arr = b.sum(s_sq)
        captured_arr = b.sum(s_sq[:rank])
        b.eval(total_arr, captured_arr)
        total = float(b.to_scalar(total_arr))
        eps = division_epsilon(b, s)
        if total < eps:
            return 0.0
        captured = float(b.to_scalar(captured_arr))
        return captured / total

    def _compute_cka(self, x: "Array", y: "Array") -> float:
        """Linear CKA (Centered Kernel Alignment).

        Delegates to the canonical Backend-aware CKA implementation in cka.py.
        """
        from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka_backend

        return compute_cka_backend(
            x,
            y,
            self._backend,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        )

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
            error_arr = b.sum(diff**2)
            norm_y_arr = b.sum(y**2)
            b.eval(error_arr, norm_y_arr)

            error = float(b.to_scalar(error_arr))
            norm_y = float(b.to_scalar(norm_y_arr))

            eps = division_epsilon(b, y)
            return sqrt_scalar(error / norm_y, b) if norm_y > eps else 0.0

        except Exception:
            return 0.0

    def _compute_knn_overlap(self, x: "Array", y: "Array", k: int = None) -> float:
        """k-NN neighborhood preservation using geodesic distances.

        Geodesic distances account for manifold curvature. Euclidean distance
        would give incorrect neighbor rankings in curved spaces.
        """
        from .riemannian_utils import RiemannianGeometry

        b = self._backend
        if k is None:
            # Use derived neighbor count (set in run_sweep)
            neighbor_count = getattr(self, "_derived_neighbor_count", None)
            if neighbor_count is None:
                # Fallback: derive from sqrt(n)
                import math
                neighbor_count = max(2, int(math.sqrt(x.shape[0])))
            k = min(neighbor_count, x.shape[0] - 1)

        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute geodesic pairwise distances (curvature is inherent in high-D)
        rg = RiemannianGeometry(b)
        k_geo = min(max(3, n // 3), n - 1)

        dx_result = rg.geodesic_distances(x, k_neighbors=k_geo)
        dy_result = rg.geodesic_distances(y, k_neighbors=k_geo)
        dx = dx_result.distances
        dy = dy_result.distances
        b.eval(dx, dy)

        # Get k-nearest neighbors
        inf = float("inf")
        eye = b.eye(n)
        dx_masked = b.where(eye > 0, b.full(dx.shape, inf), dx)
        dy_masked = b.where(eye > 0, b.full(dy.shape, inf), dy)
        x_neighbors = b.argsort(dx_masked, axis=1)[:, :k]
        y_neighbors = b.argsort(dy_masked, axis=1)[:, :k]
        b.eval(x_neighbors, y_neighbors)

        # Overlap per row via broadcast compare
        x_exp = b.expand_dims(x_neighbors, axis=2)
        y_exp = b.expand_dims(y_neighbors, axis=1)
        matches = x_exp == y_exp
        match_counts = b.sum(b.astype(matches, "float32"), axis=(1, 2))
        b.eval(match_counts)
        overlap_sum_arr = b.sum(match_counts)
        b.eval(overlap_sum_arr)
        overlap_sum_val = float(b.to_scalar(overlap_sum_arr))
        overlap_sum = overlap_sum_val / float(k)

        return overlap_sum / n

    def _compute_distance_correlation(self, x: "Array", y: "Array") -> float:
        """Pearson correlation of pairwise geodesic distances.

        Geodesic distances account for manifold curvature. Comparing Euclidean
        distances would give incorrect correlation in curved spaces.
        """
        from .riemannian_utils import RiemannianGeometry

        b = self._backend
        n = x.shape[0]
        if n < 2:
            return 0.0

        # Compute geodesic pairwise distances (curvature is inherent in high-D)
        rg = RiemannianGeometry(b)
        k_geo = min(max(3, n // 3), n - 1)

        dx_result = rg.geodesic_distances(x, k_neighbors=k_geo)
        dy_result = rg.geodesic_distances(y, k_neighbors=k_geo)
        dx_mat = dx_result.distances
        dy_mat = dy_result.distances
        b.eval(dx_mat, dy_mat)

        # Extract upper triangular pairwise distances
        off_diag = b.ones((n, n)) - b.eye(n)
        dx_vals = dx_mat * off_diag
        dy_vals = dy_mat * off_diag
        count = n * (n - 1)

        mean_x = b.sum(dx_vals) / float(count)
        mean_y = b.sum(dy_vals) / float(count)
        cov = b.sum((dx_vals - mean_x) * (dy_vals - mean_y))
        var_x = b.sum((dx_vals - mean_x) ** 2)
        var_y = b.sum((dy_vals - mean_y) ** 2)
        b.eval(cov, var_x, var_y)

        var_x_val = float(b.to_scalar(var_x))
        var_y_val = float(b.to_scalar(var_y))
        cov_val = float(b.to_scalar(cov))
        denom = sqrt_scalar(var_x_val * var_y_val, b)
        eps = division_epsilon(b, b.array([denom]))
        return cov_val / denom if denom > eps else 0.0

    def _compute_plateau(self, metrics: list[RankMetrics]) -> PlateauSummary:
        """Find plateau ranks where metrics stop improving."""
        if not metrics:
            return PlateauSummary()

        sorted_metrics = sorted(metrics, key=lambda m: m.rank)

        def find_plateau(values: list[float], higher_better: bool) -> int | None:
            # Derive epsilon from data when not provided
            if self.config.plateau_epsilon is not None:
                eps = self.config.plateau_epsilon
            elif len(values) >= 2:
                # Use sqrt(machine_epsilon) * range as threshold
                val_range = max(values) - min(values) if values else 0.0
                backend = get_default_backend()
                m_eps = float(machine_epsilon(backend, backend.array([1.0])))
                eps = val_range * (m_eps ** 0.5)
                if eps == 0:
                    eps = m_eps
            else:
                eps = 1e-10  # Fallback for degenerate case
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
            procrustes_error=find_plateau(
                [m.procrustes_error for m in sorted_metrics], higher_better=False
            ),
            knn_overlap=find_plateau([m.knn_overlap for m in sorted_metrics], higher_better=True),
            distance_correlation=find_plateau(
                [m.distance_correlation for m in sorted_metrics], higher_better=True
            ),
            variance_captured=find_plateau(
                [
                    0.5 * (m.variance_captured_source + m.variance_captured_target)
                    for m in sorted_metrics
                ],
                higher_better=True,
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
        ranks: Ranks to try (None = derived from data dimension)

    Returns:
        Optimal rank or None if sweep fails
    """
    # Pass ranks as-is; None will be derived from data in run_sweep
    config = SweepConfig(ranks=ranks)
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
