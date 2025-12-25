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
- Zero curvature: geodesics stay parallel (Euclidean)

For neural network latent spaces, curvature indicates:
- High positive: concepts are tightly clustered, interference likely
- Negative: concepts have room to spread, safer for merging
- Variable: complex topology, needs careful analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class CurvatureSign(str, Enum):
    """Classification of local curvature."""

    POSITIVE = "positive"  # Spherical, converging
    NEGATIVE = "negative"  # Hyperbolic, diverging
    FLAT = "flat"  # Euclidean
    MIXED = "mixed"  # Variable sign in neighborhood


@dataclass(frozen=True)
class CurvatureConfig:
    """Configuration for curvature estimation."""

    # Finite difference step size for gradient estimation
    # If None, epsilon is computed adaptively based on data scale
    epsilon: float | None = None
    # Number of random directions to sample for sectional curvature
    num_directions: int = 10
    # Threshold for considering curvature as flat
    flat_threshold: float = 1e-6
    # Whether to use parallel transport correction
    use_parallel_transport: bool = True
    # Neighborhood radius for local averaging
    neighborhood_radius: float = 0.1


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
    # Principal curvature proxy - NOT true Ricci curvature
    # This stores the principal curvatures as a proxy for Ricci-like information.
    # True Ricci curvature requires computing the Ricci tensor via contractions
    # of the Riemann tensor, which is computationally expensive for discrete manifolds.
    # For interference prediction, this proxy is sufficient.
    # TODO: Implement Ollivier-Ricci curvature for true discrete Ricci curvature.
    principal_curvature_proxy: "Array | None"

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
        return (self.max_sectional - self.min_sectional) / (
            abs(self.max_sectional) + abs(self.min_sectional) + 1e-10
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

    def get_high_curvature_regions(self, threshold: float = 2.0) -> list[int]:
        """Get indices of points with curvature magnitude above threshold."""
        return [
            i
            for i, lc in enumerate(self.local_curvatures)
            if abs(lc.mean_sectional) > threshold * abs(self.global_mean + 1e-10)
        ]

    def curvature_at_point(self, point: "Array", k: int = 3) -> LocalCurvature | None:
        """Find curvature at nearest measured point (k-NN interpolation).

        Uses geodesic distances for neighbor finding - Euclidean distance
        is incorrect in curved spaces where curvature itself affects distances.
        """
        if not self.local_curvatures:
            return None

        # Build point matrix for geodesic distance computation
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        backend = get_default_backend()

        # Stack all local curvature points
        point_list = [lc.point for lc in self.local_curvatures]
        all_points = backend.stack(point_list, axis=0)

        # Add query point (convert to backend array if needed)
        point_arr = backend.array(point)
        query_reshaped = backend.reshape(point_arr, (1, -1))
        all_points_with_query = backend.concatenate([all_points, query_reshaped], axis=0)
        pts_arr = backend.astype(all_points_with_query, "float32")

        # Geodesic distance matrix - last row contains distances from query to all points
        geo_dist = geodesic_distance_matrix(
            pts_arr, k_neighbors=min(10, len(all_points) - 1), backend=backend
        )
        backend.eval(geo_dist)
        geo_dist_np = backend.to_numpy(geo_dist)

        # Extract distances from query point (last row) to all measured points
        distances = geo_dist_np[-1, :-1].tolist()  # Exclude self-distance

        # Sort by distance to find k nearest
        indexed_distances = list(enumerate(distances))
        indexed_distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, _ in indexed_distances[:k]]

        # Weighted average by inverse geodesic distance
        weights = []
        for idx in nearest_indices:
            d = distances[idx]
            weights.append(1.0 / (d + 1e-10))

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Interpolate curvature values
        mean_sectional = sum(
            w * self.local_curvatures[idx].mean_sectional
            for w, idx in zip(weights, nearest_indices)
        )

        # Return nearest for full structure, but with interpolated mean
        nearest = self.local_curvatures[nearest_indices[0]]
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

    For a Riemannian manifold embedded in Euclidean space, we estimate
    sectional curvature by measuring how neighboring geodesics deviate.

    Method:
    1. At point p, sample pairs of orthogonal directions (u, v)
    2. Compute metric tensor g_ij in local coordinates
    3. Estimate Christoffel symbols from metric gradient
    4. Compute Riemann tensor R^i_jkl
    5. Sectional curvature K(u,v) = R(u,v,v,u) / (|u|^2|v|^2 - <u,v>^2)
    """

    def __init__(self, config: CurvatureConfig | None = None):
        self.config = config or CurvatureConfig()

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
                       If None, uses Euclidean metric approximation.

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

        if n < d + 1:
            logger.warning(f"Insufficient neighbors ({n}) for dimension {d}")
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
        christoffel = self._estimate_christoffel_symbols(point, neighbors, metric_fn, backend)

        # Compute sectional curvatures for sampled direction pairs
        sectional_curvatures = []
        directions_used = []

        backend.random_seed(42)  # Reproducible random directions
        for _ in range(self.config.num_directions):
            # Sample random orthonormal pair
            u = backend.random_normal((d,))
            backend.eval(u)
            u_norm = backend.norm(u)
            backend.eval(u_norm)
            u = u / (float(backend.to_numpy(u_norm)) + 1e-10)

            v = backend.random_normal((d,))
            backend.eval(u, v)
            # Gram-Schmidt
            u_np = backend.to_numpy(u)
            v_np = backend.to_numpy(v)
            dot_uv = sum(float(ui) * float(vi) for ui, vi in zip(u_np.flatten(), v_np.flatten()))
            v = v - dot_uv * u
            backend.eval(v)
            v_norm = backend.norm(v)
            backend.eval(v_norm)
            v_norm_val = float(backend.to_numpy(v_norm))
            if v_norm_val < 1e-10:
                continue
            v = v / v_norm_val

            # Compute sectional curvature K(u, v)
            K = self._sectional_curvature(u, v, metric, christoffel, backend)
            sectional_curvatures.append(K)
            directions_used.append((u, v))

        if not sectional_curvatures:
            return self._flat_curvature(point)

        # Compute principal curvatures via shape operator
        principal_dirs, principal_curvs = self._compute_principal_curvatures(
            point, neighbors, metric, backend
        )

        # Classify curvature sign
        sign = self._classify_sign(sectional_curvatures)

        # Compute statistics using pure Python
        mean_sectional = sum(sectional_curvatures) / len(sectional_curvatures)
        variance_sectional = sum((s - mean_sectional) ** 2 for s in sectional_curvatures) / len(sectional_curvatures)
        min_sectional = min(sectional_curvatures)
        max_sectional = max(sectional_curvatures)

        if principal_curvs is not None:
            backend.eval(principal_curvs)
            pc_np = backend.to_numpy(principal_curvs)
            scalar_curv = float(sum(float(x) for x in pc_np.flatten()))
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
        k_neighbors: int = 20,
        metric_fn: Callable[["Array"], "Array"] | None = None,
    ) -> ManifoldCurvatureProfile:
        """Estimate curvature profile across all points.

        Uses geodesic distances for neighbor finding - this is critical
        because Euclidean k-NN in curved spaces will systematically
        misidentify neighbors, leading to incorrect curvature estimates.

        Args:
            points: Points on the manifold (n x d array)
            k_neighbors: Number of neighbors for local estimation
            metric_fn: Optional metric tensor function

        Returns:
            ManifoldCurvatureProfile with global statistics
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        backend = get_default_backend()
        # Convert inputs to backend arrays if needed (handles numpy arrays from tests)
        points = backend.array(points)
        backend.eval(points)
        n = int(points.shape[0])
        d = int(points.shape[1])

        # Compute full geodesic distance matrix once
        pts_arr = backend.astype(points, "float32")
        geo_dist = geodesic_distance_matrix(
            pts_arr, k_neighbors=min(k_neighbors, n - 1), backend=backend
        )
        backend.eval(geo_dist)
        geo_dist_np = backend.to_numpy(geo_dist)

        local_curvatures = []

        for i in range(n):
            point = points[i]

            # Find k nearest neighbors using geodesic distances (excluding self)
            distances = geo_dist_np[i].tolist()
            # Sort by geodesic distance, exclude self (distance 0)
            indexed_distances = list(enumerate(distances))
            indexed_distances.sort(key=lambda x: x[1])
            neighbor_indices = [idx for idx, _ in indexed_distances if idx != i][:k_neighbors]

            if len(neighbor_indices) < d:
                local_curvatures.append(self._flat_curvature(point))
                continue

            # Gather neighbors
            neighbor_list = [points[idx] for idx in neighbor_indices]
            neighbors = backend.stack(neighbor_list, axis=0)

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

        # Regularize for numerical stability
        cov = cov + 1e-6 * backend.eye(d)

        # Metric is inverse of covariance (Fisher information interpretation)
        # Use regularized inverse - no fallback to identity (Euclidean)
        # Regularization handles near-singular cases while preserving geometry
        metric = backend.inv(cov)  # cov already regularized above

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

        For float32: sqrt(1e-7) ≈ 3e-4
        For float64: sqrt(1e-16) ≈ 1e-8

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
            return 1e-4  # Default fallback

        # Compute characteristic scale as median neighbor distance
        # Subsample for efficiency if many points
        if n > 50:
            indices = list(range(0, n, n // 50))
            subset_list = [neighbors[i] for i in indices]
            subset = backend.stack(subset_list, axis=0)
        else:
            subset = neighbors

        # Compute pairwise distances for scale estimation
        m = int(subset.shape[0])
        norms = backend.sum(subset * subset, axis=1, keepdims=True)
        dots = backend.matmul(subset, backend.transpose(subset))
        dist_sq = norms + backend.transpose(norms) - 2.0 * dots
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        dists = backend.sqrt(dist_sq)

        # Extract upper triangle (excluding diagonal) for median
        backend.eval(dists)
        dists_np = backend.to_numpy(dists)
        upper_tri = []
        for i in range(m):
            for j in range(i + 1, m):
                upper_tri.append(float(dists_np[i, j]))

        if not upper_tri:
            return 1e-4  # Default fallback

        upper_tri.sort()
        median_dist = upper_tri[len(upper_tri) // 2]

        # Adaptive epsilon formula
        # sqrt(machine_epsilon) for float32 ≈ 3e-4
        # Factor of d^0.25 accounts for high dimensionality
        machine_eps = 1e-7  # float32 machine epsilon
        epsilon = median_dist * (machine_eps ** 0.5) * (d ** 0.25)

        # Clamp to reasonable range
        epsilon = max(1e-8, min(epsilon, 0.1))

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

        Uses adaptive epsilon when config.epsilon is None.
        """
        backend.eval(point)
        d = int(point.shape[0])

        # Use adaptive epsilon if not specified
        if self.config.epsilon is None:
            eps = self._compute_adaptive_epsilon(neighbors, backend)
        else:
            eps = self.config.epsilon

        # Get metric at point and perturbed points
        if metric_fn is not None:
            g = metric_fn(point)
            # Initialize gradient tensor
            dg_list = []
            for k in range(d):
                point_np = backend.to_numpy(point).flatten()
                perturbed_plus = list(point_np)
                perturbed_plus[k] += eps
                perturbed_minus = list(point_np)
                perturbed_minus[k] -= eps

                g_plus = metric_fn(backend.array(perturbed_plus))
                g_minus = metric_fn(backend.array(perturbed_minus))
                dg_k = (g_plus - g_minus) / (2 * eps)
                dg_list.append(dg_k)
            dg = backend.stack(dg_list, axis=0)
        else:
            # Approximate from neighbors
            g = self._estimate_metric_tensor(neighbors - point, backend)
            # For approximation, use zeros (first-order approximation)
            dg = backend.zeros((d, d, d))

        # Compute Christoffel symbols
        # Regularize metric for stable inversion - no fallback to identity
        g_reg = g + 1e-8 * backend.eye(d)
        g_inv = backend.inv(g_reg)

        # Build christoffel tensor using pure Python loops (convert to numpy for indexing)
        backend.eval(g_inv, dg)
        g_inv_np = backend.to_numpy(g_inv)
        dg_np = backend.to_numpy(dg)

        christoffel_list = []
        for k in range(d):
            row_list = []
            for i in range(d):
                col_list = []
                for j in range(d):
                    total = 0.0
                    for idx_l in range(d):
                        if dg_np.ndim == 3:
                            total += float(g_inv_np[k, idx_l]) * (
                                float(dg_np[i, j, idx_l]) + float(dg_np[j, i, idx_l]) - float(dg_np[idx_l, i, j])
                            )
                    col_list.append(0.5 * total)
                row_list.append(col_list)
            christoffel_list.append(row_list)

        return backend.array(christoffel_list)

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
        u_np = backend.to_numpy(u).flatten()
        v_np = backend.to_numpy(v).flatten()
        metric_np = backend.to_numpy(metric)
        christoffel_np = backend.to_numpy(christoffel)

        d = len(u_np)

        # Compute Riemann tensor R^l_ijk
        # Simplified: use approximate formula for nearly flat spaces
        # K ≈ (Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik) u^i v^j v^k u^l

        riemann_component = 0.0

        for idx_l in range(d):
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        term1 = 0.0
                        term2 = 0.0
                        for m in range(d):
                            term1 += float(christoffel_np[idx_l, i, m]) * float(christoffel_np[m, j, k])
                            term2 += float(christoffel_np[idx_l, j, m]) * float(christoffel_np[m, i, k])

                        riemann_component += (term1 - term2) * float(u_np[i]) * float(v_np[j]) * float(v_np[k]) * float(u_np[idx_l])

        # Denominator: g(u,u)g(v,v) - g(u,v)^2
        # Compute using pure Python
        g_uu = sum(float(u_np[i]) * sum(float(metric_np[i, j]) * float(u_np[j]) for j in range(d)) for i in range(d))
        g_vv = sum(float(v_np[i]) * sum(float(metric_np[i, j]) * float(v_np[j]) for j in range(d)) for i in range(d))
        g_uv = sum(float(u_np[i]) * sum(float(metric_np[i, j]) * float(v_np[j]) for j in range(d)) for i in range(d))

        denom = g_uu * g_vv - g_uv * g_uv

        if abs(denom) < 1e-10:
            return 0.0

        return riemann_component / denom

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
            # Use SVD for robust linear least squares
            U, S, Vt = backend.svd(centered)
            backend.eval(S, Vt)

            S_np = backend.to_numpy(S)
            if len(S_np) < d:
                return None, None

            # Normal direction is smallest singular vector
            normal = Vt[-1]

            # Compute heights
            heights = backend.matmul(centered, backend.reshape(normal, (-1, 1)))
            heights = backend.reshape(heights, (-1,))

            # For simplified version, estimate Hessian from centered data
            # Using backend operations
            backend.eval(heights, centered)
            heights_np = backend.to_numpy(heights).flatten()
            centered_np = backend.to_numpy(centered)

            # Build design matrix in Python
            num_quad = d * (d + 1) // 2
            design_list = []
            for row_idx in range(n):
                row = list(centered_np[row_idx])
                # Add quadratic terms
                for i in range(d):
                    for j in range(i, d):
                        row.append(float(centered_np[row_idx, i] * centered_np[row_idx, j]))
                design_list.append(row)

            design = backend.array(design_list)

            # Solve least squares using backend
            # Use pinv for robust solution
            design_pinv = backend.pinv(design)
            heights_arr = backend.array(heights_np)
            coeffs = backend.matmul(design_pinv, heights_arr)
            backend.eval(coeffs)
            coeffs_np = backend.to_numpy(coeffs).flatten()

            # Extract Hessian (second fundamental form)
            hessian_list = [[0.0] * d for _ in range(d)]
            idx = d
            for i in range(d):
                for j in range(i, d):
                    if idx < len(coeffs_np):
                        hessian_list[i][j] = float(coeffs_np[idx])
                        hessian_list[j][i] = float(coeffs_np[idx])
                    idx += 1

            hessian = backend.array(hessian_list)

            # Shape operator = g^{-1} @ H
            # Regularize metric for stable inversion - no fallback
            metric_reg = metric + 1e-8 * backend.eye(d)
            metric_inv = backend.inv(metric_reg)
            shape_op = backend.matmul(metric_inv, hessian)

            # Principal curvatures are eigenvalues
            eigenvalues, eigenvectors = backend.eigh(shape_op)

            return eigenvectors, eigenvalues

        except Exception:
            return None, None

    def _classify_sign(self, sectional_curvatures: list[float]) -> CurvatureSign:
        """Classify curvature sign from sectional curvature samples."""
        threshold = self.config.flat_threshold

        pos_count = sum(1 for s in sectional_curvatures if s > threshold)
        neg_count = sum(1 for s in sectional_curvatures if s < -threshold)
        total = len(sectional_curvatures)

        if pos_count > 0.8 * total:
            return CurvatureSign.POSITIVE
        elif neg_count > 0.8 * total:
            return CurvatureSign.NEGATIVE
        elif pos_count + neg_count < 0.2 * total:
            return CurvatureSign.FLAT
        else:
            return CurvatureSign.MIXED

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
        if abs(mean_scalar) < 1e-10:
            return None

        # For positive curvature, estimate dimension from sphere formula
        if mean_scalar > 0:
            # n(n-1)/r^2 = S => n ≈ (1 + sqrt(1 + 4*S*r^2)) / 2
            # Assume unit radius for simplicity
            discriminant = 1 + 4 * mean_scalar
            if discriminant > 0:
                n_est = (1 + math.sqrt(discriminant)) / 2
                return min(n_est, ambient_dim)

        # For negative curvature, use hyperbolic formula
        # Scalar curvature of n-dim hyperbolic space = -n(n-1)
        if mean_scalar < 0:
            n_est = (1 + math.sqrt(1 - 4 * mean_scalar)) / 2
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

    # Normalize
    divergence = mean_diff + 0.5 * var_diff + 0.25 * sign_diff

    return divergence
