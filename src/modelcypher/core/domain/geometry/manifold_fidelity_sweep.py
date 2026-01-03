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
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


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

    All parameters are derived from data geometry:
    - ranks: geometric progression [4, 8, 16, ...] up to data dimension
    - neighbor_count: sqrt(n) scaling
    - min_anchor_count: based on smallest rank
    - plateau_epsilon: from fidelity variance and machine epsilon
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ):
        """Initialize sweep.

        Args:
            backend: Optional backend for array operations.
        """
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

        All parameters are derived from data geometry:
        - ranks: geometric progression [4, 8, 16, ...] up to dimension
        - neighbor_count: sqrt(n) scaling
        - min_anchor_count: based on smallest rank

        Args:
            source_activations: [n_anchors, dim] source activations
            target_activations: [n_anchors, dim] target activations
            source_layer: Source layer index
            target_layer: Target layer index

        Returns:
            LayerSweep with metrics at each rank level
        """
        n_anchors = min(source_activations.shape[0], target_activations.shape[0])
        dim = min(source_activations.shape[1], target_activations.shape[1])

        # Derive ranks from data dimension
        # Geometric progression [4, 8, 16, ...] up to dimension
        ranks = []
        r = 4
        while r <= dim:
            ranks.append(r)
            r *= 2
        if not ranks:
            ranks = [min(4, dim)]

        # Derive neighbor_count from sqrt(n)
        neighbor_count = max(2, int(sqrt_scalar(float(n_anchors), self._backend)))

        # Derive min_anchor_count from smallest rank
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

            # Normalized error using geodesic norms
            diff = x_rotated - y
            diff_flat = b.reshape(diff, (1, -1))
            y_flat = b.reshape(y, (1, -1))
            error_norm = geodesic_norms(diff_flat, b)
            norm_y_arr = geodesic_norms(y_flat, b)
            b.eval(error_norm, norm_y_arr)

            error = float(b.to_scalar(error_norm))
            norm_y = float(b.to_scalar(norm_y_arr))

            eps = division_epsilon(b, y)
            # Return squared error ratio (geodesic norm squared)
            return (error * error) / (norm_y * norm_y) if norm_y > eps else 0.0

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
                neighbor_count = max(2, int(sqrt_scalar(float(x.shape[0]), b)))
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
        """Find plateau ranks where metrics stop improving.

        Plateau epsilon is derived from data:
        - Uses sqrt(machine_epsilon) * value_range as threshold
        - This captures when improvement is at numerical noise level
        """
        if not metrics:
            return PlateauSummary()

        sorted_metrics = sorted(metrics, key=lambda m: m.rank)

        def find_plateau(values: list[float], higher_better: bool) -> int | None:
            if len(values) < 2:
                return sorted_metrics[0].rank if values else None

            # Derive epsilon from data: sqrt(machine_epsilon) * range
            val_range = max(values) - min(values) if values else 0.0
            backend = get_default_backend()
            m_eps = float(machine_epsilon(backend, backend.array([1.0])))
            eps = val_range * (m_eps ** 0.5)
            if eps == 0:
                eps = m_eps

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
    backend: "Backend | None" = None,
) -> int | None:
    """
    Find optimal alignment rank for given metric.

    All parameters are derived from data geometry. Ranks are computed as
    geometric progression [4, 8, 16, ...] up to the data dimension.

    Args:
        source_activations: Source activation matrix
        target_activations: Target activation matrix
        metric: Which metric to optimize ("cka", "procrustes", "knn", "distance", "variance")
        backend: Optional backend for array operations

    Returns:
        Optimal rank or None if sweep fails
    """
    sweep = ManifoldFidelitySweep(backend=backend)
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
