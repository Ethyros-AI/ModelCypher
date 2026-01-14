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

"""Manifold curvature estimation for latent representation spaces.

Provides sectional curvature estimation for understanding the geometry
of representation manifolds. Used by Riemannian density estimator for
curvature-aware covariance computation.

Mathematical Background:
- Sectional curvature K(σ) measures how geodesics diverge/converge
- Positive curvature: geodesics converge (like a sphere)
- Negative curvature: geodesics diverge (like a saddle)
- Zero curvature: geodesics stay parallel (flat)

For neural network latent spaces, curvature indicates:
- High positive: concepts are tightly clustered, interference likely
- Negative: concepts have room to spread, safer for merging
- Variable: complex topology, needs careful analysis

References:
    - do Carmo, M. P. (1992). "Riemannian Geometry." Birkhäuser.
      ISBN: 978-0-8176-3490-2
    - Fefferman, C., Mitter, S., & Narayanan, H. (2016).
      "Testing the Manifold Hypothesis." J. American Mathematical Society.
      https://doi.org/10.1090/jams/852
    - Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000).
      "A Global Geometric Framework for Nonlinear Dimensionality Reduction."
      Science 290(5500):2319-2323. https://doi.org/10.1126/science.290.5500.2319
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    power_iteration_eigh,
    precision_dtype,
    regularization_epsilon,
    safe_inverse,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_distance_matrix,
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _array_to_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)


class CurvatureSign(str, Enum):
    """Classification of local curvature."""

    POSITIVE = "positive"  # Spherical, converging
    NEGATIVE = "negative"  # Hyperbolic, diverging
    FLAT = "flat"  # flat
    MIXED = "mixed"  # Variable sign in neighborhood




# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from the data:
# - epsilon: from data scale (mean neighbor distance)
# - num_directions: from dimension d (at least d, at most d(d-1)/2 for full coverage)
# - use_parallel_transport: always True (geometrically correct)
#
# There is exactly ONE correct way to estimate curvature.
# =============================================================================


@dataclass
class LocalCurvature:
    """Curvature information at a single point."""

    # The point where curvature was measured
    point: "Array"
    # Mean sectional curvature across sampled directions
    mean_sectional: float
    # Variance of sectional curvature (indicates isotropy)
    variance_sectional: float
    # Minimum sectional curvature (most negative direction)
    min_sectional: float
    # Maximum sectional curvature (most positive direction)
    max_sectional: float
    # Principal curvature directions (eigenvectors of shape operator)
    principal_directions: "Array | None"
    # Principal curvatures (eigenvalues)
    principal_curvatures: "Array | None"
    # Classification of curvature sign
    sign: CurvatureSign
    # Scalar curvature (trace of Ricci tensor, sum of sectional)
    scalar_curvature: float
    # True Ollivier-Ricci curvature at this point (aggregated from incident edges)
    # Computed via optimal transport: kappa = 1 - W_1(m_x, m_y) / d(x, y)
    # None if not yet computed (use OllivierRicciCurvature.compute() to populate)
    ollivier_ricci: float | None = None
    # Legacy: Principal curvature proxy (deprecated, use ollivier_ricci instead)
    principal_curvature_proxy: "Array | None" = None

    @property
    def is_positively_curved(self) -> bool:
        """Check if predominantly positive curvature."""
        return self.mean_sectional > 0 and self.sign in (
            CurvatureSign.POSITIVE,
            CurvatureSign.MIXED,
        )

    @property
    def is_negatively_curved(self) -> bool:
        """Check if predominantly negative curvature."""
        return self.mean_sectional < 0 and self.sign in (
            CurvatureSign.NEGATIVE,
            CurvatureSign.MIXED,
        )

    @property
    def curvature_anisotropy(self) -> float:
        """Measure of curvature variation across directions (0=isotropic)."""
        if self.max_sectional == self.min_sectional:
            return 0.0
        backend = get_default_backend()
        eps = division_epsilon(
            backend,
            backend.array([self.max_sectional, self.min_sectional]),
        )
        return (self.max_sectional - self.min_sectional) / (
            abs(self.max_sectional) + abs(self.min_sectional) + eps
        )


@dataclass
class ManifoldCurvatureProfile:
    """Curvature profile across multiple points on the manifold."""

    # Per-point curvature measurements
    local_curvatures: list[LocalCurvature]
    # Global mean curvature
    global_mean: float
    # Global curvature variance
    global_variance: float
    # Fraction of points with each curvature sign
    sign_distribution: dict[CurvatureSign, float]
    # Overall curvature classification
    dominant_sign: CurvatureSign
    # Estimated intrinsic dimension from curvature
    estimated_dimension: float | None

    def get_high_curvature_regions(self) -> list[int]:
        """Get indices of points with unusually large curvature magnitude.

        Uses a data-derived absolute cutoff: abs(mean) + std.
        """
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([self.global_mean]))
        mean_abs = abs(self.global_mean + eps)
        std_val = sqrt_scalar(self.global_variance, backend)
        abs_threshold = mean_abs + std_val
        return [
            i
            for i, lc in enumerate(self.local_curvatures)
            if abs(lc.mean_sectional) > abs_threshold
        ]

    def curvature_at_point(self, point: "Array") -> LocalCurvature | None:
        """Find curvature at nearest measured point (k-NN interpolation).

        Uses geodesic distances for neighbor finding - chord distance
        is incorrect in curved spaces where curvature itself affects distances.
        """
        if not self.local_curvatures:
            return None

        # Build point matrix for geodesic distance computation
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )
        from modelcypher.core.domain.geometry.riemannian_validation import (
            derive_k_neighbors,
        )

        backend = get_default_backend()

        # Stack all local curvature points
        point_list = [lc.point for lc in self.local_curvatures]
        all_points = backend.stack(point_list, axis=0)

        # Add query point (convert to backend array if needed)
        point_arr = backend.array(point)
        query_reshaped = backend.reshape(point_arr, (1, -1))
        all_points_with_query = backend.concatenate([all_points, query_reshaped], axis=0)
        pts_arr = backend.astype(
            all_points_with_query, precision_dtype(backend, reference=all_points_with_query)
        )

        # Geodesic distance matrix (k=None triggers connectivity-based selection)
        # Last row contains distances from query to all points
        geo_dist = geodesic_distance_matrix(pts_arr, k_neighbors=None, backend=backend)
        backend.eval(geo_dist)

        # Extract distances from query point (last row) to all measured points
        last_row_idx = backend.array([len(self.local_curvatures)])
        row = backend.take(geo_dist, last_row_idx, axis=0)
        row = backend.squeeze(row, axis=0)
        row = backend.take(row, backend.arange(0, len(self.local_curvatures)), axis=0)

        total = int(row.shape[0])
        k = derive_k_neighbors(all_points, backend)
        k = max(1, min(k, total))

        # If query matches an existing point (within precision), return it directly
        eps = division_epsilon(backend, row)
        min_dist_arr = backend.min(row)
        backend.eval(min_dist_arr)
        min_dist = float(backend.to_scalar(min_dist_arr))
        if min_dist <= eps:
            nearest_idx_arr = backend.argmin(row)
            backend.eval(nearest_idx_arr)
            nearest_idx = int(backend.to_scalar(nearest_idx_arr))
            nearest = self.local_curvatures[nearest_idx]
            return LocalCurvature(
                point=point,
                mean_sectional=nearest.mean_sectional,
                variance_sectional=nearest.variance_sectional,
                min_sectional=nearest.min_sectional,
                max_sectional=nearest.max_sectional,
                principal_directions=nearest.principal_directions,
                principal_curvatures=nearest.principal_curvatures,
                sign=nearest.sign,
                scalar_curvature=nearest.scalar_curvature,
                principal_curvature_proxy=nearest.principal_curvature_proxy,
            )

        kth = max(0, k - 1)
        partitioned = backend.argpartition(row, kth)
        nearest_indices_arr = partitioned[:k]
        nearest_dists = backend.take(row, nearest_indices_arr, axis=0)
        backend.eval(nearest_indices_arr, nearest_dists)
        nearest_indices = [
            int(val) for val in _array_to_list(backend, nearest_indices_arr)
        ]
        nearest_local_idx_arr = backend.argmin(nearest_dists)
        backend.eval(nearest_local_idx_arr)
        nearest_local_idx = int(backend.to_scalar(nearest_local_idx_arr))
        nearest_idx = nearest_indices[nearest_local_idx]

        # Weighted average by inverse geodesic distance
        eps = division_epsilon(backend, nearest_dists)
        weights_arr = 1.0 / (nearest_dists + eps)
        weights_arr = weights_arr / backend.sum(weights_arr)
        backend.eval(weights_arr)
        weights = [float(w) for w in _array_to_list(backend, weights_arr)]

        # Interpolate curvature values
        mean_sectional = sum(
            w * self.local_curvatures[idx].mean_sectional
            for w, idx in zip(weights, nearest_indices)
        )

        # Return nearest for full structure, but with interpolated mean
        nearest = self.local_curvatures[nearest_idx]
        return LocalCurvature(
            point=point,
            mean_sectional=mean_sectional,
            variance_sectional=nearest.variance_sectional,
            min_sectional=nearest.min_sectional,
            max_sectional=nearest.max_sectional,
            principal_directions=nearest.principal_directions,
            principal_curvatures=nearest.principal_curvatures,
            sign=nearest.sign,
            scalar_curvature=nearest.scalar_curvature,
            principal_curvature_proxy=nearest.principal_curvature_proxy,
        )


class SectionalCurvatureEstimator:
    """Estimates sectional curvature using geodesic deviation.

    For a Riemannian manifold embedded in ambient space, we estimate
    sectional curvature by measuring how neighboring geodesics deviate.

    Method:
    1. At point p, sample pairs of orthogonal directions (u, v)
    2. Compute metric tensor g_ij in local coordinates
    3. Estimate Christoffel symbols from metric gradient
    4. Compute Riemann tensor R^i_jkl
    5. Sectional curvature K(u,v) = R(u,v,v,u) / (|u|^2|v|^2 - <u,v>^2)
    """

    def __init__(self) -> None:
        """Initialize curvature estimator. All parameters derived from data."""
        pass

    def estimate_local_curvature(
        self,
        point: "Array",
        neighbors: "Array",
        metric_fn: Callable[["Array"], "Array"] | None = None,
    ) -> LocalCurvature:
        """Estimate curvature at a single point using neighborhood.

        Args:
            point: The point to estimate curvature at (d-dimensional)
            neighbors: Nearby points for local geometry (n x d array)
            metric_fn: Optional function returning metric tensor at a point.
                       If None, derives metric from local covariance.

        Returns:
            LocalCurvature with all curvature measurements
        """
        backend = get_default_backend()
        # Convert inputs to backend arrays if needed (handles numpy arrays from tests)
        point = backend.array(point)
        neighbors = backend.array(neighbors)
        backend.eval(point, neighbors)

        d = int(point.shape[0])
        n = int(neighbors.shape[0])

        # Need at least 2 neighbors for covariance estimation
        # When n < d, we use regularized pseudoinverse for metric tensor
        if n < 2:
            logger.warning(f"Insufficient neighbors ({n}) for curvature estimation")
            return self._flat_curvature(point)

        # Center neighbors around point
        centered = neighbors - point

        # Compute local metric tensor using covariance
        if metric_fn is not None:
            metric = metric_fn(point)
        else:
            # Approximate metric from local covariance structure
            metric = self._estimate_metric_tensor(centered, backend)

        # Estimate Christoffel symbols via finite differences
        christoffel_result = self._estimate_christoffel_symbols(point, neighbors, metric_fn, backend)

        # Handle reduced representation for high dimensions
        is_reduced = isinstance(christoffel_result, tuple)
        if is_reduced:
            christoffel, V_reduced = christoffel_result
            d_reduced = int(christoffel.shape[0])
            # Metric needs to be in reduced space too
            centered_reduced = backend.matmul(centered, backend.transpose(V_reduced))
            metric_reduced = self._estimate_metric_tensor(centered_reduced, backend)
        else:
            christoffel = christoffel_result
            d_reduced = d
            V_reduced = None
            metric_reduced = metric

        # Compute sectional curvatures for sampled direction pairs
        sectional_curvatures = []

        # Derive num_directions from dimension (data-derived)
        # For d dimensions, there are d(d-1)/2 possible 2-planes
        # Sample at least d directions for coverage, capped at full coverage
        # Use reduced dimension for high-d cases
        max_planes = d_reduced * (d_reduced - 1) // 2 if d_reduced > 1 else 1
        num_directions = min(max(d_reduced, 5), max_planes)

        backend.random_seed(42)  # Reproducible random directions
        eps = division_epsilon(backend, centered)

        # Batch-generate all random direction vectors at once
        # For high-d, generate in reduced space directly
        U = backend.random_normal((num_directions, d_reduced))
        V = backend.random_normal((num_directions, d_reduced))
        backend.eval(U, V)

        # Batch normalize U: U / ||U||
        U_norms = geodesic_norms(U, backend)
        backend.eval(U_norms)
        U_norms_safe = backend.maximum(U_norms, backend.zeros_like(U_norms) + eps)
        U = U / backend.reshape(U_norms_safe, (-1, 1))
        backend.eval(U)

        # Batch Gram-Schmidt: V = V - (U·V)·U, then V / ||V||
        # Compute dot products: (U * V).sum(axis=1)
        dot_UV = backend.sum(U * V, axis=1)
        backend.eval(dot_UV)
        V = V - backend.reshape(dot_UV, (-1, 1)) * U
        backend.eval(V)

        V_norms = geodesic_norms(V, backend)
        backend.eval(V_norms)
        V_norms_safe = backend.maximum(V_norms, backend.full(V_norms.shape, eps))
        V_unit = V / backend.reshape(V_norms_safe, (-1, 1))
        backend.eval(V_unit)

        curvature_arr = self._sectional_curvature_batch(
            U, V_unit, metric_reduced, christoffel, backend
        )
        valid_mask = V_norms > eps
        backend.eval(curvature_arr, valid_mask)

        # Compute count of valid curvatures on backend
        valid_float = backend.astype(
            valid_mask, precision_dtype(backend, reference=valid_mask)
        )
        valid_count_arr = backend.sum(valid_float)
        backend.eval(valid_count_arr)
        valid_count = int(backend.to_scalar(valid_count_arr))

        if valid_count == 0:
            return self._flat_curvature(point)

        # Compute statistics on backend using masked operations (no .tolist())
        zeros = backend.zeros_like(curvature_arr)
        valid_curvatures = backend.where(valid_mask, curvature_arr, zeros)

        # Mean
        sum_arr = backend.sum(valid_curvatures)
        backend.eval(sum_arr)
        mean_arr = sum_arr / valid_count
        backend.eval(mean_arr)
        mean_sectional = float(backend.to_scalar(mean_arr))

        # Variance = E[x^2] - E[x]^2
        sq_curvatures = backend.where(valid_mask, curvature_arr * curvature_arr, zeros)
        sum_sq_arr = backend.sum(sq_curvatures)
        backend.eval(sum_sq_arr)
        variance_sectional = float(backend.to_scalar(sum_sq_arr)) / valid_count - mean_sectional * mean_sectional

        # Min/Max using inf masking
        neg_inf = -float(backend.finfo().max)
        pos_inf = float(backend.finfo().max)
        max_masked = backend.where(valid_mask, curvature_arr, backend.full(curvature_arr.shape, neg_inf))
        min_masked = backend.where(valid_mask, curvature_arr, backend.full(curvature_arr.shape, pos_inf))
        max_arr = backend.max(max_masked)
        min_arr = backend.min(min_masked)
        backend.eval(max_arr, min_arr)
        max_sectional = float(backend.to_scalar(max_arr))
        min_sectional = float(backend.to_scalar(min_arr))

        # Compute principal curvatures via shape operator
        principal_dirs, principal_curvs = self._compute_principal_curvatures(
            point, neighbors, metric, backend
        )

        # Classify curvature sign using backend array + mask
        sign = self._classify_sign_backend(curvature_arr, valid_mask, backend)

        if principal_curvs is not None:
            scalar_curv_arr = backend.sum(principal_curvs)
            backend.eval(scalar_curv_arr)
            scalar_curv = float(backend.to_scalar(scalar_curv_arr))
        else:
            scalar_curv = float(sum(sectional_curvatures))

        return LocalCurvature(
            point=point,
            mean_sectional=mean_sectional,
            variance_sectional=variance_sectional,
            min_sectional=min_sectional,
            max_sectional=max_sectional,
            principal_directions=principal_dirs,
            principal_curvatures=principal_curvs,
            sign=sign,
            scalar_curvature=scalar_curv,
            principal_curvature_proxy=principal_curvs,
        )

    def estimate_manifold_profile(
        self,
        points: "Array",
        metric_fn: Callable[["Array"], "Array"] | None = None,
    ) -> ManifoldCurvatureProfile:
        """Estimate curvature profile across all points.

        Uses geodesic distances for neighbor finding - this is critical
        because chord k-NN in curved spaces will systematically
        misidentify neighbors, leading to incorrect curvature estimates.

        k_neighbors is derived from data via Berry & Sauer (2016):
        minimum k for graph connectivity, plus log(n) for local structure.

        Args:
            points: Points on the manifold (n x d array)
            metric_fn: Optional metric tensor function

        Returns:
            ManifoldCurvatureProfile with global statistics
        """
        import math

        from modelcypher.core.domain.geometry.riemannian_utils import (
            RiemannianGeometry,
            geodesic_distance_matrix,
        )

        backend = get_default_backend()
        # Convert inputs to backend arrays if needed (handles numpy arrays from tests)
        points = backend.array(points)
        backend.eval(points)
        n = int(points.shape[0])
        d = int(points.shape[1])

        # Derive k_neighbors from data:
        # 1. Minimum k for connectivity (Berry & Sauer 2016)
        # 2. At least ceil(log(n)) for local structure
        # 3. Cap at n-1 (can't have more neighbors than other points exist)
        #
        # Note: The d+1 requirement was mathematically incorrect.
        # Sectional curvature estimation needs enough neighbors for local
        # covariance estimation, not d+1 neighbors. With k < d, we use
        # regularized pseudoinverse for metric tensor estimation.
        riemannian = RiemannianGeometry(backend=backend)
        result = riemannian.geodesic_distances(points, k_neighbors=None)
        k_connectivity = result.k_neighbors
        k_local = max(2, int(math.ceil(math.log(max(n, 2)))))
        k_neighbors = max(k_connectivity, k_local)
        k_neighbors = min(k_neighbors, n - 1)  # Can't exceed available points

        # Compute full geodesic distance matrix once
        pts_arr = backend.astype(points, precision_dtype(backend, reference=points))
        geo_dist = geodesic_distance_matrix(
            pts_arr, k_neighbors=min(k_neighbors, n - 1), backend=backend
        )
        backend.eval(geo_dist)

        local_curvatures = []
        dtype = geo_dist.dtype if hasattr(geo_dist, "dtype") else None
        inf_val = float(backend.finfo(dtype).max)
        eye = backend.eye(n)
        row_masked = backend.where(eye > 0, backend.full((n, n), inf_val), geo_dist)
        kth = max(0, min(k_neighbors - 1, n - 1))
        partitioned = backend.argpartition(row_masked, kth, axis=1)
        neighbor_indices_all = partitioned[:, :k_neighbors]
        backend.eval(neighbor_indices_all)

        # Batch gather all neighbors at once: [n, k, d]
        # This replaces n separate backend.take calls with one batched operation
        flat_indices = backend.reshape(neighbor_indices_all, (-1,))
        flat_indices_int = backend.astype(flat_indices, "int32")
        all_neighbors_flat = backend.take(points, flat_indices_int, axis=0)
        all_neighbors = backend.reshape(all_neighbors_flat, (n, k_neighbors, d))
        backend.eval(all_neighbors)

        for i in range(n):
            point = points[i]

            # Check if we have enough neighbors (need at least 2)
            if k_neighbors < 2:
                local_curvatures.append(self._flat_curvature(point))
                continue

            # Get pre-gathered neighbors for this point
            neighbors = all_neighbors[i]

            # Compute local curvature - no fallback to flat
            # If this fails, it's a bug we need to fix, not hide
            lc = self.estimate_local_curvature(point, neighbors, metric_fn)
            local_curvatures.append(lc)

        # Compute global statistics using pure Python
        mean_sectionals = [lc.mean_sectional for lc in local_curvatures]
        global_mean = sum(mean_sectionals) / len(mean_sectionals) if mean_sectionals else 0.0
        global_variance = sum((m - global_mean) ** 2 for m in mean_sectionals) / len(mean_sectionals) if mean_sectionals else 0.0

        # Sign distribution
        sign_counts = {s: 0 for s in CurvatureSign}
        for lc in local_curvatures:
            sign_counts[lc.sign] += 1

        total = len(local_curvatures)
        sign_distribution = {s: c / total for s, c in sign_counts.items()}

        # Dominant sign
        dominant_sign = max(sign_distribution, key=sign_distribution.get)

        # Estimate intrinsic dimension from curvature
        estimated_dimension = self._estimate_dimension(local_curvatures, d)

        return ManifoldCurvatureProfile(
            local_curvatures=local_curvatures,
            global_mean=global_mean,
            global_variance=global_variance,
            sign_distribution=sign_distribution,
            dominant_sign=dominant_sign,
            estimated_dimension=estimated_dimension,
        )

    def _estimate_metric_tensor(self, centered_neighbors: "Array", backend: "Backend") -> "Array":
        """Estimate local metric tensor from neighborhood covariance."""
        # Compute covariance using backend
        backend.eval(centered_neighbors)
        n = int(centered_neighbors.shape[0])
        d = int(centered_neighbors.shape[1])

        if n < 2:
            return backend.eye(d)

        # Center the data
        mean = backend.mean(centered_neighbors, axis=0)
        centered = centered_neighbors - mean

        # Covariance = X^T @ X / (n-1)
        cov = backend.matmul(backend.transpose(centered), centered) / (n - 1)
        backend.eval(cov)

        # Metric is inverse of covariance (Fisher information interpretation)
        # Use safe_inverse for condition-checked inversion with auto-regularization
        metric, _ = safe_inverse(backend, cov, regularize=True)

        return metric

    def _compute_adaptive_epsilon(
        self,
        neighbors: "Array",
        backend: "Backend",
    ) -> float:
        """
        Compute adaptive epsilon based on data characteristics.

        The optimal step size for finite differences balances truncation error
        (decreases with smaller epsilon) against numerical precision limits
        (increases with smaller epsilon due to floating point cancellation).

        Formula: epsilon = scale * sqrt(machine_epsilon) * d^0.25
        where scale is the characteristic length scale of the data.

        Args:
            neighbors: Nearby points used for curvature estimation
            backend: Computational backend

        Returns:
            Adaptive epsilon value
        """
        backend.eval(neighbors)
        n = int(neighbors.shape[0])
        d = int(neighbors.shape[1])

        if n < 2:
            return division_epsilon(backend, neighbors)

        # Compute characteristic scale as median neighbor distance
        # Use ALL points - no subsampling. Geodesic diagnostics need full coverage.
        subset = neighbors

        # Compute pairwise geodesic distances for scale estimation
        m = int(subset.shape[0])
        dists = geodesic_distance_matrix(subset, k_neighbors=None, backend=backend)
        backend.eval(dists)

        # Extract upper triangle (excluding diagonal) for median using backend ops
        idx = backend.arange(m)
        row = backend.reshape(idx, (m, 1))
        col = backend.reshape(idx, (1, m))
        upper_mask = col > row
        upper_count_arr = backend.sum(backend.astype(upper_mask, "int32"))
        backend.eval(upper_count_arr)
        upper_count = int(backend.to_scalar(upper_count_arr))
        if upper_count == 0:
            return division_epsilon(backend, neighbors)

        pos_inf = backend.full(dists.shape, float("inf"))
        upper_vals = backend.where(upper_mask, dists, pos_inf)
        flat = backend.reshape(upper_vals, (-1,))
        median_idx = upper_count // 2
        partitioned = backend.argpartition(flat, median_idx)
        prefix = backend.take(partitioned, backend.arange(median_idx + 1), axis=0)
        median_arr = backend.max(backend.take(flat, prefix, axis=0))
        backend.eval(median_arr)
        median_dist = float(backend.to_scalar(median_arr))

        # Adaptive epsilon formula with dimensional scaling
        eps = division_epsilon(backend, neighbors)
        epsilon = median_dist * eps * (d ** 0.25)
        max_dist_arr = backend.max(backend.where(upper_mask, dists, backend.zeros_like(dists)))
        backend.eval(max_dist_arr)
        max_dist = float(backend.to_scalar(max_dist_arr))
        if max_dist > 0:
            epsilon = min(epsilon, max_dist)
        epsilon = max(eps, epsilon)

        logger.debug(
            f"Adaptive epsilon: {epsilon:.2e} (scale={median_dist:.2e}, d={d})"
        )

        return epsilon

    def _estimate_christoffel_symbols(
        self,
        point: "Array",
        neighbors: "Array",
        metric_fn: Callable[["Array"], "Array"] | None,
        backend: "Backend",
    ) -> "Array":
        """Estimate Christoffel symbols via finite differences.

        Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)

        Uses adaptive epsilon derived from neighbor distances.

        For high dimensions (d > 256), uses a reduced representation
        to avoid O(d³) memory. The full Christoffel tensor would be
        1024³ * 4 bytes = 4GB for d=1024, which is prohibitive.
        Instead, we use local PCA to work in a reduced tangent space.
        """
        backend.eval(point)
        d = int(point.shape[0])

        # For high dimensions, use reduced tangent space approach
        # This avoids O(d³) memory which would be 4GB+ for d=1024
        max_dim_for_full_tensor = 256
        if d > max_dim_for_full_tensor:
            # Use local PCA to reduce dimension
            # Work in a reduced tangent space (max 256 dims)
            centered = neighbors - point
            n_neighbors = int(centered.shape[0])

            # Compute SVD of centered neighbors to find principal directions
            U, S, Vt = geodesic_svd(backend, centered)
            backend.eval(S, Vt)

            # Use up to max_dim_for_full_tensor principal directions
            k_dim = min(max_dim_for_full_tensor, n_neighbors, d)
            V_reduced = Vt[:k_dim]  # [k_dim, d]

            # Project point and neighbors to reduced space
            point_reduced = backend.matmul(V_reduced, backend.reshape(point, (d, 1)))
            point_reduced = backend.reshape(point_reduced, (k_dim,))

            # Recursively compute Christoffel in reduced space
            neighbors_reduced = backend.matmul(centered, backend.transpose(V_reduced))  # [n, k_dim]
            neighbors_reduced = neighbors_reduced + point_reduced

            # Compute reduced Christoffel tensor (k_dim³ instead of d³)
            christoffel_reduced = self._estimate_christoffel_symbols_direct(
                point_reduced, neighbors_reduced, None, backend, k_dim
            )

            # Embed back to full dimension using projection
            # For curvature estimation, we only need the effect in sampled directions
            # so we store the reduced tensor and projection matrix
            return (christoffel_reduced, V_reduced)

        return self._estimate_christoffel_symbols_direct(
            point, neighbors, metric_fn, backend, d
        )

    def _estimate_christoffel_symbols_direct(
        self,
        point: "Array",
        neighbors: "Array",
        metric_fn: Callable[["Array"], "Array"] | None,
        backend: "Backend",
        d: int,
    ) -> "Array":
        """Direct Christoffel computation for dimensions <= 256."""
        # Epsilon is always derived from data (no configuration)
        eps = self._compute_adaptive_epsilon(neighbors, backend)

        # Get metric at point and perturbed points
        if metric_fn is not None:
            g = metric_fn(point)
            # Initialize gradient tensor
            dg_list = []
            eye = backend.eye(d)
            for k in range(d):
                delta = eye[k]
                perturbed_plus = point + delta * eps
                perturbed_minus = point - delta * eps

                g_plus = metric_fn(perturbed_plus)
                g_minus = metric_fn(perturbed_minus)
                dg_k = (g_plus - g_minus) / (2 * eps)
                dg_list.append(dg_k)
            dg = backend.stack(dg_list, axis=0)
        else:
            # Approximate from neighbors
            g = self._estimate_metric_tensor(neighbors - point, backend)
            # For approximation, use zeros (first-order approximation)
            dg = backend.zeros((d, d, d))

        # Compute Christoffel symbols
        # Use safe_inverse for condition-checked inversion with auto-regularization
        g_inv, _ = safe_inverse(backend, g, regularize=True)

        # Build christoffel tensor using backend ops
        backend.eval(g_inv, dg)
        dg_ij_l = dg
        dg_ji_l = backend.transpose(dg, (1, 0, 2))
        dg_l_ij = backend.transpose(dg, (2, 0, 1))
        term = dg_ij_l + dg_ji_l - dg_l_ij
        term_flat = backend.reshape(term, (d * d, d))
        term_flat_t = backend.transpose(term_flat)
        christoffel_flat = backend.matmul(g_inv, term_flat_t)
        christoffel = backend.reshape(christoffel_flat, (d, d, d)) * 0.5
        backend.eval(christoffel)

        return christoffel

    def _sectional_curvature(
        self,
        u: "Array",
        v: "Array",
        metric: "Array",
        christoffel: "Array",
        backend: "Backend",
    ) -> float:
        """Compute sectional curvature K(u, v).

        K(u,v) = R(u,v,v,u) / (g(u,u)g(v,v) - g(u,v)^2)

        where R is the Riemann curvature tensor.
        """
        backend.eval(u, v, metric, christoffel)
        d = int(u.shape[0])

        # Compute Riemann tensor component via backend contractions
        # K ≈ (Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik) u^i v^j v^k u^l
        u_col = backend.reshape(u, (d, 1))
        v_col = backend.reshape(v, (d, 1))
        v_row = backend.reshape(v, (1, d))

        # A_{l,m} = Σ_i Γ^l_im u_i
        u_weight = backend.reshape(u, (1, d, 1))
        A = backend.sum(christoffel * u_weight, axis=1)

        # C_{l,m} = Σ_j Γ^l_jm v_j
        v_weight = backend.reshape(v, (1, d, 1))
        C = backend.sum(christoffel * v_weight, axis=1)

        # B_m = Σ_{j,k} Γ^m_jk v_j v_k
        outer_vv = backend.matmul(v_col, v_row)
        B = backend.sum(christoffel * outer_vv, axis=(1, 2))

        # D_m = Σ_{i,k} Γ^m_ik u_i v_k
        outer_uv = backend.matmul(u_col, v_row)
        D = backend.sum(christoffel * outer_uv, axis=(1, 2))

        B_col = backend.reshape(B, (d, 1))
        D_col = backend.reshape(D, (d, 1))
        term1 = backend.sum(backend.matmul(A, B_col) * u_col)
        term2 = backend.sum(backend.matmul(C, D_col) * u_col)
        riemann_component_arr = term1 - term2
        backend.eval(riemann_component_arr)
        riemann_component = float(backend.to_scalar(riemann_component_arr))

        # Denominator: g(u,u)g(v,v) - g(u,v)^2
        u_vec = backend.reshape(u, (d, 1))
        v_vec = backend.reshape(v, (d, 1))
        g_u = backend.matmul(metric, u_vec)
        g_v = backend.matmul(metric, v_vec)
        g_uu_arr = backend.matmul(backend.transpose(u_vec), g_u)
        g_vv_arr = backend.matmul(backend.transpose(v_vec), g_v)
        g_uv_arr = backend.matmul(backend.transpose(u_vec), g_v)
        backend.eval(g_uu_arr, g_vv_arr, g_uv_arr)
        g_uu = float(backend.to_scalar(g_uu_arr))
        g_vv = float(backend.to_scalar(g_vv_arr))
        g_uv = float(backend.to_scalar(g_uv_arr))

        denom = g_uu * g_vv - g_uv * g_uv

        # Use precision-aware epsilon for denominator check
        eps = float(division_epsilon(backend, metric))
        if abs(denom) < eps:
            return 0.0

        return riemann_component / denom

    def _sectional_curvature_batch(
        self,
        u: "Array",
        v: "Array",
        metric: "Array",
        christoffel: "Array",
        backend: "Backend",
    ) -> "Array":
        """Compute sectional curvature K(u, v) for a batch of direction pairs."""
        backend.eval(u, v, metric, christoffel)
        batch = int(u.shape[0])
        d = int(u.shape[1])

        u_col = backend.reshape(u, (batch, d, 1))
        v_col = backend.reshape(v, (batch, d, 1))
        v_row = backend.reshape(v, (batch, 1, d))

        christ = backend.reshape(christoffel, (1, d, d, d))
        u_weight = backend.reshape(u, (batch, 1, d, 1))
        v_weight = backend.reshape(v, (batch, 1, d, 1))

        A = backend.sum(christ * u_weight, axis=2)
        C = backend.sum(christ * v_weight, axis=2)

        outer_vv = backend.matmul(v_col, v_row)
        outer_vv = backend.reshape(outer_vv, (batch, 1, d, d))
        B = backend.sum(christ * outer_vv, axis=(2, 3))

        outer_uv = backend.matmul(u_col, v_row)
        outer_uv = backend.reshape(outer_uv, (batch, 1, d, d))
        D = backend.sum(christ * outer_uv, axis=(2, 3))

        B_col = backend.reshape(B, (batch, d, 1))
        D_col = backend.reshape(D, (batch, d, 1))
        term1 = backend.sum(backend.matmul(A, B_col) * u_col, axis=(1, 2))
        term2 = backend.sum(backend.matmul(C, D_col) * u_col, axis=(1, 2))
        riemann_component = term1 - term2

        metric_exp = backend.reshape(metric, (1, d, d))
        g_u = backend.matmul(metric_exp, u_col)
        g_v = backend.matmul(metric_exp, v_col)
        u_vec_t = backend.transpose(u_col, (0, 2, 1))
        v_vec_t = backend.transpose(v_col, (0, 2, 1))
        g_uu = backend.reshape(backend.matmul(u_vec_t, g_u), (batch,))
        g_vv = backend.reshape(backend.matmul(v_vec_t, g_v), (batch,))
        g_uv = backend.reshape(backend.matmul(u_vec_t, g_v), (batch,))

        denom = g_uu * g_vv - g_uv * g_uv
        eps = float(division_epsilon(backend, metric))
        denom_abs = backend.abs(denom)
        valid = denom_abs >= eps
        safe_denom = backend.where(valid, denom, backend.ones_like(denom))
        return backend.where(valid, riemann_component / safe_denom, backend.zeros_like(denom))

    def _compute_principal_curvatures(
        self,
        point: "Array",
        neighbors: "Array",
        metric: "Array",
        backend: "Backend",
    ) -> tuple["Array | None", "Array | None"]:
        """Compute principal curvatures via shape operator approximation."""
        backend.eval(point, neighbors, metric)
        d = int(point.shape[0])
        n = int(neighbors.shape[0])

        if n < d:
            return None, None

        # Fit local quadratic form to approximate second fundamental form
        centered = neighbors - point

        try:
            # Use geodesic SVD for robust linear least squares (GPU-only)
            U, S, Vt = geodesic_svd(backend, centered)
            backend.eval(S, Vt)

            if int(S.shape[0]) < d:
                return None, None

            # Normal direction is smallest singular vector
            normal = Vt[-1]

            # Compute heights
            heights = backend.matmul(centered, backend.reshape(normal, (-1, 1)))
            heights = backend.reshape(heights, (-1,))

            # For simplified version, estimate Hessian from centered data
            # Using backend operations
            backend.eval(heights, centered)

            # Build design matrix on backend: [centered, upper-triangular quadratic terms]
            row_col = backend.reshape(centered, (n, d, 1))
            row_row = backend.reshape(centered, (n, 1, d))
            outer = row_col * row_row  # [n, d, d]
            upper_idx = [i * d + j for i in range(d) for j in range(i, d)]
            upper_idx_arr = backend.array(upper_idx, dtype="int32")
            outer_flat = backend.reshape(outer, (n, d * d))
            upper_terms = backend.take(outer_flat, upper_idx_arr, axis=1)
            design = backend.concatenate([centered, upper_terms], axis=1)

            # Solve least squares using backend
            # Use pinv for robust solution
            design_pinv = backend.pinv(design)
            heights_arr = backend.reshape(heights, (-1,))
            coeffs = backend.matmul(design_pinv, heights_arr)
            backend.eval(coeffs)

            # Extract Hessian (second fundamental form)
            # Use tolist() for O(1) extraction
            coeffs_list = backend.tolist(coeffs)
            hessian_list = [[0.0] * d for _ in range(d)]
            idx = d
            for i in range(d):
                for j in range(i, d):
                    if idx < len(coeffs_list):
                        coeff_val = float(coeffs_list[idx])
                        hessian_list[i][j] = coeff_val
                        hessian_list[j][i] = coeff_val
                    idx += 1

            hessian = backend.array(hessian_list)

            # Shape operator = g^{-1} @ H
            # Use safe_inverse for condition-checked inversion with auto-regularization
            metric_inv, _ = safe_inverse(backend, metric, regularize=True)
            shape_op = backend.matmul(metric_inv, hessian)

            # Principal curvatures are eigenvalues (geodesic - GPU-only)
            d_shape = int(shape_op.shape[0])
            eigenvalues, eigenvectors = power_iteration_eigh(backend, shape_op, k=d_shape)

            return eigenvectors, eigenvalues

        except Exception:
            return None, None

    def _classify_sign(self, sectional_curvatures: list[float]) -> CurvatureSign:
        """Classify curvature sign from sectional curvature samples."""
        if not sectional_curvatures:
            return CurvatureSign.FLAT

        backend = get_default_backend()
        curv_arr = backend.array(sectional_curvatures)
        backend.eval(curv_arr)
        eps = division_epsilon(backend, curv_arr)
        has_pos_arr = backend.sum(backend.astype(curv_arr > eps, "int32"))
        has_neg_arr = backend.sum(backend.astype(curv_arr < -eps, "int32"))
        backend.eval(has_pos_arr, has_neg_arr)
        has_pos = bool(backend.to_scalar(has_pos_arr))
        has_neg = bool(backend.to_scalar(has_neg_arr))

        if has_pos and has_neg:
            return CurvatureSign.MIXED
        if has_pos:
            return CurvatureSign.POSITIVE
        if has_neg:
            return CurvatureSign.NEGATIVE
        return CurvatureSign.FLAT

    def _classify_sign_backend(
        self,
        curvature_arr: "Array",
        valid_mask: "Array",
        backend: "Backend",
    ) -> CurvatureSign:
        """Classify curvature sign directly from backend arrays (no .tolist())."""
        backend.eval(curvature_arr, valid_mask)
        eps = division_epsilon(backend, curvature_arr)

        # Count valid curvatures that are positive / negative
        valid_int = backend.astype(valid_mask, "int32")
        pos_mask = backend.astype(curvature_arr > eps, "int32") * valid_int
        neg_mask = backend.astype(curvature_arr < -eps, "int32") * valid_int

        has_pos_arr = backend.sum(pos_mask)
        has_neg_arr = backend.sum(neg_mask)
        backend.eval(has_pos_arr, has_neg_arr)
        has_pos = bool(backend.to_scalar(has_pos_arr))
        has_neg = bool(backend.to_scalar(has_neg_arr))

        if has_pos and has_neg:
            return CurvatureSign.MIXED
        if has_pos:
            return CurvatureSign.POSITIVE
        if has_neg:
            return CurvatureSign.NEGATIVE
        return CurvatureSign.FLAT

    def _flat_curvature(self, point: "Array") -> LocalCurvature:
        """Return flat curvature for edge cases."""
        return LocalCurvature(
            point=point,
            mean_sectional=0.0,
            variance_sectional=0.0,
            min_sectional=0.0,
            max_sectional=0.0,
            principal_directions=None,
            principal_curvatures=None,
            sign=CurvatureSign.FLAT,
            scalar_curvature=0.0,
            principal_curvature_proxy=None,
        )

    def _estimate_dimension(
        self,
        local_curvatures: list[LocalCurvature],
        ambient_dim: int,
    ) -> float | None:
        """Estimate intrinsic dimension from curvature distribution.

        Uses relationship between scalar curvature and dimension:
        For n-sphere of radius r, scalar curvature = n(n-1)/r^2
        """
        # Get scalar curvatures
        scalars = [lc.scalar_curvature for lc in local_curvatures if lc.scalar_curvature != 0]

        if len(scalars) < 3:
            return None

        mean_scalar = sum(scalars) / len(scalars)
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([mean_scalar]))
        if abs(mean_scalar) < eps:
            return None

        # For positive curvature, estimate dimension from sphere formula
        if mean_scalar > 0:
            # n(n-1)/r^2 = S => n ≈ (1 + sqrt(1 + 4*S*r^2)) / 2
            # Assume unit radius for simplicity
            discriminant = 1 + 4 * mean_scalar
            if discriminant > 0:
                n_est = (1 + sqrt_scalar(discriminant, backend)) / 2
                return min(n_est, ambient_dim)

        # For negative curvature, use hyperbolic formula
        # Scalar curvature of n-dim hyperbolic space = -n(n-1)
        if mean_scalar < 0:
            n_est = (1 + sqrt_scalar(1 - 4 * mean_scalar, backend)) / 2
            return min(n_est, ambient_dim)

        return None


def compute_curvature_divergence(
    profile_a: ManifoldCurvatureProfile,
    profile_b: ManifoldCurvatureProfile,
) -> float:
    """Compute divergence between curvature profiles of two manifolds.

    Useful for predicting merge compatibility - similar curvature
    profiles suggest compatible geometric structure.

    Returns:
        Divergence score (0 = identical, higher = more different)
    """
    # Compare global statistics
    mean_diff = abs(profile_a.global_mean - profile_b.global_mean)
    var_diff = abs(profile_a.global_variance - profile_b.global_variance)

    # Compare sign distributions
    sign_diff = 0.0
    for sign in CurvatureSign:
        diff = abs(
            profile_a.sign_distribution.get(sign, 0) - profile_b.sign_distribution.get(sign, 0)
        )
        sign_diff += diff

    # Equal weights for divergence components
    divergence = (mean_diff + var_diff + sign_diff) / 3.0

    return divergence
