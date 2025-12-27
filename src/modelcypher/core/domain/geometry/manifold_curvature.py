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
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class CurvatureSign(str, Enum):
    """Classification of local curvature."""

    POSITIVE = "positive"  # Spherical, converging
    NEGATIVE = "negative"  # Hyperbolic, diverging
    FLAT = "flat"  # Euclidean
    MIXED = "mixed"  # Variable sign in neighborhood


class ManifoldHealth(str, Enum):
    """Health classification of a representation manifold based on Ricci curvature.

    For neural network representation spaces:
    - HEALTHY: Negative Ricci curvature (hyperbolic geometry) - normal for LLMs
    - DEGENERATE: Near-flat curvature - loss of geometric structure
    - COLLAPSED: Positive Ricci curvature - potential representation collapse
    """

    HEALTHY = "healthy"  # mean_ricci < -0.1 (hyperbolic, expected for LLMs)
    DEGENERATE = "degenerate"  # -0.1 <= mean_ricci <= 0.1 (nearly flat)
    COLLAPSED = "collapsed"  # mean_ricci > 0.1 (spherical, representation collapse)


@dataclass(frozen=True)
class OllivierRicciConfig:
    """Configuration for Ollivier-Ricci curvature computation.

    The Ollivier-Ricci curvature uses optimal transport (Wasserstein-1 distance)
    between lazy random walk distributions to measure discrete curvature.

    Adaptive alpha: When enabled, alpha varies per node based on degree.
    High-degree nodes get lower alpha (rely more on self, neighborhood is crowded).
    Low-degree nodes get higher alpha (spread mass to sparse neighborhood).
    """

    # Base lazy random walk parameter from Ollivier (2009)
    base_alpha: float = 0.5

    # Adaptive alpha: varies per node based on degree
    adaptive_alpha: bool = True

    # How much degree affects alpha: alpha = base * (1 - degree/max_degree * strength)
    adaptive_strength: float = 0.3

    # Sinkhorn regularization for W_1 approximation
    sinkhorn_epsilon: float = 0.001
    sinkhorn_iterations: int = 100
    sinkhorn_threshold: float = 1e-8

    # Number of neighbors for k-NN graph
    k_neighbors: int = 10

    # Whether to symmetrize curvature: kappa(x,y) = (kappa(x,y) + kappa(y,x)) / 2
    # Ollivier-Ricci is naturally asymmetric; averaging symmetrizes
    symmetrize: bool = True


@dataclass(frozen=True)
class EdgeCurvature:
    """Ollivier-Ricci curvature for a single edge in the k-NN graph."""

    source_idx: int
    target_idx: int
    curvature: float  # kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)
    wasserstein_distance: float  # W_1(m_x, m_y)
    edge_weight: float  # d(x, y) from geodesic distance


@dataclass(frozen=True)
class NodeRicciCurvature:
    """Aggregated Ollivier-Ricci curvature at a node."""

    node_idx: int
    mean_curvature: float  # Average over incident edges
    min_curvature: float
    max_curvature: float
    num_edges: int


@dataclass(frozen=True)
class OllivierRicciResult:
    """Complete Ollivier-Ricci curvature analysis of a manifold."""

    # Per-edge curvatures
    edge_curvatures: list[EdgeCurvature]

    # Per-node curvatures (aggregated from edges)
    node_curvatures: list[NodeRicciCurvature]

    # Global edge statistics
    mean_edge_curvature: float
    std_edge_curvature: float

    # Global node statistics
    mean_node_curvature: float

    # Health classification
    health: ManifoldHealth

    # Configuration used
    config: OllivierRicciConfig
    k_neighbors: int
    n_points: int


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

        # Use precision-aware epsilon for denominator check
        eps = float(division_epsilon(backend, metric))
        if abs(denom) < eps:
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


class OllivierRicciCurvature:
    """Compute Ollivier-Ricci curvature on the discrete k-NN manifold.

    Ollivier-Ricci curvature measures how probability mass from neighboring
    nodes overlaps, providing a discrete analog of Ricci curvature.

    For each edge (x, y) in the k-NN graph:
        kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)

    where m_x is a lazy random walk distribution centered at x:
        m_x = (1-alpha) * delta_x + alpha * uniform(neighbors(x))

    With adaptive alpha (default), alpha varies per node based on degree:
        alpha(node) = base_alpha * (1 - degree/max_degree * adaptive_strength)

    High-degree nodes get lower alpha (rely more on self, neighborhood is crowded).
    Low-degree nodes get higher alpha (spread mass to sparse neighborhood).

    Mathematical background:
    - Ollivier (2009) "Ricci curvature of Markov chains on metric spaces"
    - Curvature for graphs provides structural information:
      - Positive: clustered/spherical regions
      - Negative: bridges/bottlenecks between clusters
      - Flat: uniform connectivity

    For neural network representations:
    - Healthy manifolds have negative Ricci curvature (hyperbolic)
    - Near-zero indicates degenerate (flat) representations
    - Positive indicates potential representation collapse

    References:
        - Ollivier (2009): Original formulation
        - GraphRicciCurvature library: github.com/saibalmars/GraphRicciCurvature
        - NeurIPS 2024: "Exploring Geometric Representational Alignment
          through Ollivier-Ricci Curvature"
    """

    def __init__(
        self,
        config: OllivierRicciConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or OllivierRicciConfig()
        self._backend = backend or get_default_backend()

    def compute(
        self,
        points: "Array",
        k_neighbors: int | None = None,
    ) -> OllivierRicciResult:
        """Compute Ollivier-Ricci curvature for all edges in the k-NN graph.

        Args:
            points: Point cloud [n, d] on the manifold
            k_neighbors: Override config k_neighbors if provided

        Returns:
            OllivierRicciResult with edge/node curvatures and health classification
        """
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
        from modelcypher.core.domain.geometry.numerical_stability import tiny_value

        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        k = k_neighbors or self.config.k_neighbors
        k = min(k, n - 1)

        if n < 2:
            # Trivial case: no edges
            return OllivierRicciResult(
                edge_curvatures=[],
                node_curvatures=[],
                mean_edge_curvature=0.0,
                std_edge_curvature=0.0,
                mean_node_curvature=0.0,
                health=ManifoldHealth.DEGENERATE,
                config=self.config,
                k_neighbors=k,
                n_points=n,
            )

        # 1. Compute geodesic distances using existing infrastructure
        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        backend.eval(geo_result.distances)

        # 2. Build adjacency list from k-NN graph
        adjacency_list = self._build_adjacency_list(geo_result, k, n)

        # Compute max degree for adaptive alpha
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values()) if adjacency_list else 1

        # 3. Compute edge curvatures
        edge_curvatures: list[EdgeCurvature] = []
        processed_edges: set[tuple[int, int]] = set()

        geo_np = backend.to_numpy(geo_result.distances)

        for source_idx in range(n):
            for target_idx in adjacency_list.get(source_idx, []):
                # Track edges to avoid duplicates (for symmetrized curvature)
                edge_key = (min(source_idx, target_idx), max(source_idx, target_idx))
                if edge_key in processed_edges and self.config.symmetrize:
                    continue
                processed_edges.add(edge_key)

                ec = self._compute_edge_curvature(
                    source_idx,
                    target_idx,
                    geo_result,
                    geo_np,
                    adjacency_list,
                    max_degree,
                    n,
                )
                if ec is not None:
                    edge_curvatures.append(ec)

        # 4. Aggregate to nodes
        node_curvatures = self._aggregate_to_nodes(edge_curvatures, n)

        # 5. Compute global statistics
        if edge_curvatures:
            edge_kappas = [ec.curvature for ec in edge_curvatures]
            mean_edge = sum(edge_kappas) / len(edge_kappas)
            variance = sum((kappa - mean_edge) ** 2 for kappa in edge_kappas) / len(edge_kappas)
            std_edge = math.sqrt(variance)
        else:
            mean_edge = 0.0
            std_edge = 0.0

        if node_curvatures:
            node_means = [nc.mean_curvature for nc in node_curvatures]
            mean_node = sum(node_means) / len(node_means)
        else:
            mean_node = 0.0

        # 6. Classify health
        health = self._classify_health(mean_node)

        return OllivierRicciResult(
            edge_curvatures=edge_curvatures,
            node_curvatures=node_curvatures,
            mean_edge_curvature=mean_edge,
            std_edge_curvature=std_edge,
            mean_node_curvature=mean_node,
            health=health,
            config=self.config,
            k_neighbors=k,
            n_points=n,
        )

    def _build_adjacency_list(
        self,
        geo_result: "Any",
        k: int,
        n: int,
    ) -> dict[int, list[int]]:
        """Build adjacency list from geodesic distance result.

        Uses the k nearest neighbors for each point based on geodesic distances.
        """
        backend = self._backend
        geo_np = backend.to_numpy(geo_result.distances)

        adjacency: dict[int, list[int]] = {}

        for i in range(n):
            # Get distances from node i to all other nodes
            distances = [(j, geo_np[i, j]) for j in range(n) if j != i]

            # Filter out infinite distances (disconnected)
            finite_distances = [(j, d) for j, d in distances if math.isfinite(d)]

            # Sort by distance and take k nearest
            finite_distances.sort(key=lambda x: x[1])
            neighbors = [j for j, _ in finite_distances[:k]]

            adjacency[i] = neighbors

        return adjacency

    def _compute_edge_curvature(
        self,
        source_idx: int,
        target_idx: int,
        geo_result: "Any",
        geo_np: "Any",
        adjacency_list: dict[int, list[int]],
        max_degree: int,
        n: int,
    ) -> EdgeCurvature | None:
        """Compute Ollivier-Ricci curvature for a single edge.

        kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)
        """
        backend = self._backend

        # Edge weight from geodesic distance matrix
        edge_weight = float(geo_np[source_idx, target_idx])

        # Skip infinite or zero edge weights
        eps = division_epsilon(backend, geo_result.distances)
        if not math.isfinite(edge_weight) or edge_weight < eps:
            return None

        # Build lazy random walk measures with adaptive alpha
        mu = self._build_lazy_measure(source_idx, adjacency_list, max_degree, n)
        nu = self._build_lazy_measure(target_idx, adjacency_list, max_degree, n)

        # Cost matrix is the geodesic distance matrix
        cost_matrix = geo_result.distances

        # Compute W_1 distance
        w1 = self._compute_wasserstein_1(mu, nu, cost_matrix)

        # Ollivier-Ricci curvature
        curvature = 1.0 - w1 / edge_weight

        return EdgeCurvature(
            source_idx=source_idx,
            target_idx=target_idx,
            curvature=curvature,
            wasserstein_distance=w1,
            edge_weight=edge_weight,
        )

    def _build_lazy_measure(
        self,
        node_idx: int,
        adjacency_list: dict[int, list[int]],
        max_degree: int,
        n_points: int,
    ) -> "Array":
        """Build lazy random walk probability measure at a node.

        m_x = (1-alpha) * delta_x + alpha * uniform(neighbors(x))

        With adaptive alpha:
            degree = len(neighbors)
            alpha = base_alpha * (1 - degree/max_degree * adaptive_strength)

        High-degree nodes get lower alpha (rely more on self).
        Low-degree nodes get higher alpha (spread mass to sparse neighborhood).

        Args:
            node_idx: Index of the node
            adjacency_list: Mapping from node to its neighbors
            max_degree: Maximum degree in the graph (for normalization)
            n_points: Total number of points

        Returns:
            Probability distribution over all nodes [n_points]
        """
        backend = self._backend

        neighbors = adjacency_list.get(node_idx, [])
        degree = len(neighbors)

        # Compute adaptive alpha
        if self.config.adaptive_alpha and max_degree > 0:
            alpha = self.config.base_alpha * (
                1.0 - (degree / max_degree) * self.config.adaptive_strength
            )
        else:
            alpha = self.config.base_alpha

        # Clamp alpha to [0, 1]
        alpha = max(0.0, min(1.0, alpha))

        # Initialize measure as zeros
        measure_list = [0.0] * n_points

        # Lazy component: (1-alpha) at node itself
        measure_list[node_idx] = 1.0 - alpha

        # Uniform over neighbors: alpha / degree each
        if degree > 0:
            neighbor_weight = alpha / degree
            for neighbor_idx in neighbors:
                measure_list[neighbor_idx] += neighbor_weight
        else:
            # Isolated node: all mass at self
            measure_list[node_idx] = 1.0

        return backend.array(measure_list)

    def _compute_wasserstein_1(
        self,
        mu: "Array",
        nu: "Array",
        cost_matrix: "Array",
    ) -> float:
        """Compute Wasserstein-1 distance using Sinkhorn algorithm.

        W_1(mu, nu) = min_gamma <cost, gamma>
        subject to: gamma @ 1 = mu, gamma^T @ 1 = nu

        With entropic regularization:
        W_1^eps(mu, nu) = min_gamma <cost, gamma> + eps * H(gamma)

        This follows the same pattern as gromov_wasserstein.py:_solve_linear_ot()
        """
        from modelcypher.core.domain.geometry.numerical_stability import tiny_value

        backend = self._backend
        n = int(mu.shape[0])

        epsilon = self.config.sinkhorn_epsilon
        max_iter = self.config.sinkhorn_iterations
        threshold = self.config.sinkhorn_threshold

        # Get precision-aware values
        eps = division_epsilon(backend, cost_matrix)
        floor = tiny_value(backend, cost_matrix)

        # Stabilized Sinkhorn: K = exp(-cost / epsilon)
        cost_min = backend.min(cost_matrix, axis=1, keepdims=True)
        cost_centered = cost_matrix - cost_min
        log_K = -cost_centered / max(epsilon, eps)

        # Clamp to avoid underflow
        log_K = backend.maximum(log_K, backend.full(log_K.shape, -80.0))
        K = backend.exp(log_K)
        K = backend.maximum(K, backend.full(K.shape, floor))

        # Sinkhorn iterations
        u = backend.ones((n,))
        v = backend.ones((n,))

        for _ in range(max_iter):
            Kv = backend.matmul(K, v)
            Kv = backend.maximum(Kv, backend.full(Kv.shape, floor))
            u_new = mu / Kv

            Ktu = backend.matmul(backend.transpose(K), u_new)
            Ktu = backend.maximum(Ktu, backend.full(Ktu.shape, floor))
            v_new = nu / Ktu

            # Convergence check
            if threshold > 0:
                u_diff = backend.max(backend.abs(u_new - u))
                v_diff = backend.max(backend.abs(v_new - v))
                backend.eval(u_diff, v_diff)
                if max(float(backend.to_numpy(u_diff)), float(backend.to_numpy(v_diff))) < threshold:
                    u, v = u_new, v_new
                    break

            u, v = u_new, v_new

        # Recover transport plan: gamma = diag(u) @ K @ diag(v)
        gamma = K * backend.reshape(u, (n, 1)) * backend.reshape(v, (1, n))

        # W_1 = <cost, gamma>
        w1 = backend.sum(cost_matrix * gamma)
        backend.eval(w1)

        return float(backend.to_numpy(w1))

    def _aggregate_to_nodes(
        self,
        edge_curvatures: list[EdgeCurvature],
        n_points: int,
    ) -> list[NodeRicciCurvature]:
        """Aggregate edge curvatures to node curvatures.

        For each node, compute mean, min, max over all incident edges.
        """
        # Collect curvatures per node
        node_curvatures_map: dict[int, list[float]] = {i: [] for i in range(n_points)}

        for ec in edge_curvatures:
            node_curvatures_map[ec.source_idx].append(ec.curvature)
            node_curvatures_map[ec.target_idx].append(ec.curvature)

        # Build node curvature objects
        result: list[NodeRicciCurvature] = []
        for node_idx in range(n_points):
            curvatures = node_curvatures_map[node_idx]
            if curvatures:
                result.append(
                    NodeRicciCurvature(
                        node_idx=node_idx,
                        mean_curvature=sum(curvatures) / len(curvatures),
                        min_curvature=min(curvatures),
                        max_curvature=max(curvatures),
                        num_edges=len(curvatures),
                    )
                )
            else:
                # Isolated node
                result.append(
                    NodeRicciCurvature(
                        node_idx=node_idx,
                        mean_curvature=0.0,
                        min_curvature=0.0,
                        max_curvature=0.0,
                        num_edges=0,
                    )
                )

        return result

    def _classify_health(self, mean_ricci: float) -> ManifoldHealth:
        """Classify manifold health based on mean Ricci curvature.

        For LLM representation manifolds:
        - HEALTHY: mean_ricci < -0.1 (hyperbolic, expected geometry)
        - DEGENERATE: -0.1 <= mean_ricci <= 0.1 (nearly flat, loss of structure)
        - COLLAPSED: mean_ricci > 0.1 (spherical, representation collapse)
        """
        if mean_ricci < -0.1:
            return ManifoldHealth.HEALTHY
        elif mean_ricci > 0.1:
            return ManifoldHealth.COLLAPSED
        else:
            return ManifoldHealth.DEGENERATE
