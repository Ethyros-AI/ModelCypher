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
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    pass

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
    epsilon: float = 1e-4
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
    point: np.ndarray
    # Mean sectional curvature across sampled directions
    mean_sectional: float
    # Variance of sectional curvature (indicates isotropy)
    variance_sectional: float
    # Minimum sectional curvature (most negative direction)
    min_sectional: float
    # Maximum sectional curvature (most positive direction)
    max_sectional: float
    # Principal curvature directions (eigenvectors of shape operator)
    principal_directions: np.ndarray | None
    # Principal curvatures (eigenvalues)
    principal_curvatures: np.ndarray | None
    # Classification of curvature sign
    sign: CurvatureSign
    # Scalar curvature (trace of Ricci tensor, sum of sectional)
    scalar_curvature: float
    # Ricci curvature in principal directions
    ricci_curvature: np.ndarray | None

    @property
    def is_positively_curved(self) -> bool:
        """Check if predominantly positive curvature."""
        return self.mean_sectional > 0 and self.sign in (CurvatureSign.POSITIVE, CurvatureSign.MIXED)

    @property
    def is_negatively_curved(self) -> bool:
        """Check if predominantly negative curvature."""
        return self.mean_sectional < 0 and self.sign in (CurvatureSign.NEGATIVE, CurvatureSign.MIXED)

    @property
    def curvature_anisotropy(self) -> float:
        """Measure of curvature variation across directions (0=isotropic)."""
        if self.max_sectional == self.min_sectional:
            return 0.0
        return (self.max_sectional - self.min_sectional) / (abs(self.max_sectional) + abs(self.min_sectional) + 1e-10)


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
            i for i, lc in enumerate(self.local_curvatures)
            if abs(lc.mean_sectional) > threshold * abs(self.global_mean + 1e-10)
        ]

    def curvature_at_point(self, point: np.ndarray, k: int = 3) -> LocalCurvature | None:
        """Find curvature at nearest measured point (k-NN interpolation)."""
        if not self.local_curvatures:
            return None

        # Find k nearest measured points
        distances = [
            np.linalg.norm(lc.point - point)
            for lc in self.local_curvatures
        ]
        nearest_indices = np.argsort(distances)[:k]

        # Weighted average by inverse distance
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
            ricci_curvature=nearest.ricci_curvature,
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
        point: np.ndarray,
        neighbors: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray] | None = None,
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
        d = point.shape[0]
        n = neighbors.shape[0]

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
            metric = self._estimate_metric_tensor(centered)

        # Estimate Christoffel symbols via finite differences
        christoffel = self._estimate_christoffel_symbols(point, neighbors, metric_fn)

        # Compute sectional curvatures for sampled direction pairs
        sectional_curvatures = []
        directions_used = []

        for _ in range(self.config.num_directions):
            # Sample random orthonormal pair
            u = np.random.randn(d)
            u = u / (np.linalg.norm(u) + 1e-10)

            v = np.random.randn(d)
            v = v - np.dot(v, u) * u  # Gram-Schmidt
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                continue
            v = v / v_norm

            # Compute sectional curvature K(u, v)
            K = self._sectional_curvature(u, v, metric, christoffel)
            sectional_curvatures.append(K)
            directions_used.append((u, v))

        if not sectional_curvatures:
            return self._flat_curvature(point)

        sectional_array = np.array(sectional_curvatures)

        # Compute principal curvatures via shape operator
        principal_dirs, principal_curvs = self._compute_principal_curvatures(
            point, neighbors, metric
        )

        # Classify curvature sign
        sign = self._classify_sign(sectional_array)

        return LocalCurvature(
            point=point,
            mean_sectional=float(np.mean(sectional_array)),
            variance_sectional=float(np.var(sectional_array)),
            min_sectional=float(np.min(sectional_array)),
            max_sectional=float(np.max(sectional_array)),
            principal_directions=principal_dirs,
            principal_curvatures=principal_curvs,
            sign=sign,
            scalar_curvature=float(np.sum(principal_curvs)) if principal_curvs is not None else float(np.sum(sectional_array)),
            ricci_curvature=principal_curvs,
        )

    def estimate_manifold_profile(
        self,
        points: np.ndarray,
        k_neighbors: int = 20,
        metric_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> ManifoldCurvatureProfile:
        """Estimate curvature profile across all points.

        Args:
            points: Points on the manifold (n x d array)
            k_neighbors: Number of neighbors for local estimation
            metric_fn: Optional metric tensor function

        Returns:
            ManifoldCurvatureProfile with global statistics
        """
        from scipy.spatial import KDTree

        n, d = points.shape
        tree = KDTree(points)

        local_curvatures = []

        for i in range(n):
            point = points[i]

            # Find k nearest neighbors (excluding self)
            distances, indices = tree.query(point, k=min(k_neighbors + 1, n))
            neighbor_indices = [idx for idx in indices if idx != i][:k_neighbors]

            if len(neighbor_indices) < d:
                local_curvatures.append(self._flat_curvature(point))
                continue

            neighbors = points[neighbor_indices]

            try:
                lc = self.estimate_local_curvature(point, neighbors, metric_fn)
                local_curvatures.append(lc)
            except Exception as e:
                logger.warning(f"Curvature estimation failed at point {i}: {e}")
                local_curvatures.append(self._flat_curvature(point))

        # Compute global statistics
        mean_sectionals = [lc.mean_sectional for lc in local_curvatures]
        global_mean = float(np.mean(mean_sectionals))
        global_variance = float(np.var(mean_sectionals))

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

    def _estimate_metric_tensor(self, centered_neighbors: np.ndarray) -> np.ndarray:
        """Estimate local metric tensor from neighborhood covariance."""
        # Covariance gives inverse of metric in tangent space approximation
        cov = np.cov(centered_neighbors.T)
        if cov.ndim == 0:
            return np.array([[cov + 1e-10]])

        # Regularize for numerical stability
        cov = cov + 1e-6 * np.eye(cov.shape[0])

        # Metric is inverse of covariance (Fisher information interpretation)
        try:
            metric = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            metric = np.eye(cov.shape[0])

        return metric

    def _estimate_christoffel_symbols(
        self,
        point: np.ndarray,
        neighbors: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        """Estimate Christoffel symbols via finite differences.

        Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
        """
        d = point.shape[0]
        eps = self.config.epsilon

        # Get metric at point and perturbed points
        if metric_fn is not None:
            g = metric_fn(point)
            dg = np.zeros((d, d, d))  # ∂_k g_ij

            for k in range(d):
                perturbed = point.copy()
                perturbed[k] += eps
                g_plus = metric_fn(perturbed)
                perturbed[k] -= 2 * eps
                g_minus = metric_fn(perturbed)
                dg[k] = (g_plus - g_minus) / (2 * eps)
        else:
            # Approximate from neighbors
            g = self._estimate_metric_tensor(neighbors - point)
            dg = np.zeros((d, d, d))

            # Estimate gradient using local linear regression
            for k in range(d):
                # Find neighbors in positive and negative k direction
                centered = neighbors - point
                pos_mask = centered[:, k] > 0
                neg_mask = centered[:, k] < 0

                if np.sum(pos_mask) >= d and np.sum(neg_mask) >= d:
                    g_pos = self._estimate_metric_tensor(centered[pos_mask])
                    g_neg = self._estimate_metric_tensor(centered[neg_mask])
                    mean_dist = np.mean(np.abs(centered[:, k]))
                    if mean_dist > 1e-10:
                        dg[k] = (g_pos - g_neg) / (2 * mean_dist)

        # Compute Christoffel symbols
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            g_inv = np.eye(d)

        christoffel = np.zeros((d, d, d))  # Γ^k_ij

        for k in range(d):
            for i in range(d):
                for j in range(d):
                    total = 0.0
                    for l in range(d):
                        total += g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )
                    christoffel[k, i, j] = 0.5 * total

        return christoffel

    def _sectional_curvature(
        self,
        u: np.ndarray,
        v: np.ndarray,
        metric: np.ndarray,
        christoffel: np.ndarray,
    ) -> float:
        """Compute sectional curvature K(u, v).

        K(u,v) = R(u,v,v,u) / (g(u,u)g(v,v) - g(u,v)^2)

        where R is the Riemann curvature tensor.
        """
        d = len(u)

        # Compute Riemann tensor R^l_ijk
        # R^l_ijk = ∂_i Γ^l_jk - ∂_j Γ^l_ik + Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik
        # For sectional curvature, we only need R(u,v,v,u) = R^l_ijk u^i v^j v^k g_lm u^m

        # Simplified: use approximate formula for nearly flat spaces
        # K ≈ (Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik) u^i v^j v^k u^l

        riemann_component = 0.0

        for l in range(d):
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        term1 = 0.0
                        term2 = 0.0
                        for m in range(d):
                            term1 += christoffel[l, i, m] * christoffel[m, j, k]
                            term2 += christoffel[l, j, m] * christoffel[m, i, k]

                        riemann_component += (term1 - term2) * u[i] * v[j] * v[k] * u[l]

        # Denominator: g(u,u)g(v,v) - g(u,v)^2
        g_uu = np.dot(u, metric @ u)
        g_vv = np.dot(v, metric @ v)
        g_uv = np.dot(u, metric @ v)

        denom = g_uu * g_vv - g_uv * g_uv

        if abs(denom) < 1e-10:
            return 0.0

        return riemann_component / denom

    def _compute_principal_curvatures(
        self,
        point: np.ndarray,
        neighbors: np.ndarray,
        metric: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute principal curvatures via shape operator approximation."""
        d = point.shape[0]
        n = neighbors.shape[0]

        if n < d:
            return None, None

        # Fit local quadratic form to approximate second fundamental form
        centered = neighbors - point

        # Build design matrix for quadratic fit
        # Include linear and quadratic terms
        num_quad = d * (d + 1) // 2
        design = np.zeros((n, d + num_quad))

        design[:, :d] = centered

        idx = d
        for i in range(d):
            for j in range(i, d):
                design[:, idx] = centered[:, i] * centered[:, j]
                idx += 1

        # Estimate height function (distance from tangent plane)
        try:
            # Use SVD for robust linear least squares
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)

            # Normal direction is smallest singular vector
            if len(s) >= d:
                normal = Vt[-1]
            else:
                return None, None

            heights = centered @ normal

            # Fit quadratic to heights
            coeffs, _, _, _ = np.linalg.lstsq(design, heights, rcond=None)

            # Extract Hessian (second fundamental form)
            hessian = np.zeros((d, d))
            idx = d
            for i in range(d):
                for j in range(i, d):
                    hessian[i, j] = coeffs[idx]
                    hessian[j, i] = coeffs[idx]
                    idx += 1

            # Shape operator = g^{-1} @ H
            try:
                shape_op = np.linalg.solve(metric, hessian)
            except np.linalg.LinAlgError:
                shape_op = hessian

            # Principal curvatures are eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(shape_op)

            return eigenvectors, eigenvalues

        except Exception:
            return None, None

    def _classify_sign(self, sectional_curvatures: np.ndarray) -> CurvatureSign:
        """Classify curvature sign from sectional curvature samples."""
        threshold = self.config.flat_threshold

        pos_count = np.sum(sectional_curvatures > threshold)
        neg_count = np.sum(sectional_curvatures < -threshold)
        total = len(sectional_curvatures)

        if pos_count > 0.8 * total:
            return CurvatureSign.POSITIVE
        elif neg_count > 0.8 * total:
            return CurvatureSign.NEGATIVE
        elif pos_count + neg_count < 0.2 * total:
            return CurvatureSign.FLAT
        else:
            return CurvatureSign.MIXED

    def _flat_curvature(self, point: np.ndarray) -> LocalCurvature:
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
            ricci_curvature=None,
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

        mean_scalar = np.mean(scalars)
        if abs(mean_scalar) < 1e-10:
            return None

        # For positive curvature, estimate dimension from sphere formula
        if mean_scalar > 0:
            # n(n-1)/r^2 = S => n ≈ (1 + sqrt(1 + 4*S*r^2)) / 2
            # Assume unit radius for simplicity
            discriminant = 1 + 4 * mean_scalar
            if discriminant > 0:
                n_est = (1 + np.sqrt(discriminant)) / 2
                return min(n_est, ambient_dim)

        # For negative curvature, use hyperbolic formula
        # Scalar curvature of n-dim hyperbolic space = -n(n-1)
        if mean_scalar < 0:
            n_est = (1 + np.sqrt(1 - 4 * mean_scalar)) / 2
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
        diff = abs(profile_a.sign_distribution.get(sign, 0) - profile_b.sign_distribution.get(sign, 0))
        sign_diff += diff

    # Normalize
    divergence = mean_diff + 0.5 * var_diff + 0.25 * sign_diff

    return divergence
