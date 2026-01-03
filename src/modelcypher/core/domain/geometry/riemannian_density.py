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

"""Riemannian density estimation for concept manifolds.

Models concepts as probability distributions over the representation manifold.
Provides volume-based overlap for interference prediction and curvature-aware
covariance estimation.

Notes
-----
Each concept is modeled as a Riemannian Gaussian: a normal distribution on
the curved manifold where covariance accounts for the local metric tensor.
Geodesic radius measures extent along the manifold, not Euclidean distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    e_value,
    exp_scalar,
    inf_value,
    infinity_threshold,
    lgamma_scalar,
    log_scalar,
    machine_epsilon,
    pi_value,
    regularization_epsilon,
    sqrt_scalar,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Array, Backend

from .manifold_curvature import (
    LocalCurvature,
    SectionalCurvatureEstimator,
)
from .riemannian_utils import GeodesicDistanceResult, RiemannianGeometry

logger = logging.getLogger(__name__)

_cache = ComputationCache.shared()


def _find_k_elbow(activations: "Array", backend: "Backend") -> int:
    """Find optimal k for k-NN using elbow detection on the k-distance curve.

    The elbow method finds where adding more neighbors gives diminishing returns.
    This is a geometric feature of the data - the point of maximum curvature
    in the curve of mean k-th neighbor distance vs k.

    Algorithm:
    1. Compute pairwise distances between all points
    2. For each k from 1 to n-1, compute mean k-th neighbor distance
    3. Find the elbow (maximum curvature) in this curve
    4. Return that k value

    The curvature at each point is computed as the discrete second derivative
    normalized by arc length: curvature = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)

    Args:
        activations: Array of activation vectors (n x d)
        backend: Backend instance for tensor operations

    Returns:
        Optimal k value derived from data geometry
    """
    shape = activations.shape
    n = int(shape[0])

    if n <= 2:
        return 1

    # Compute pairwise geodesic distances from the k-NN graph.
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(activations, k_neighbors=None)
    dists = geo_result.distances
    backend.eval(dists)

    # Guard against any sentinel values during sorting.
    inf_thresh = infinity_threshold(backend, dists)
    dists = backend.where(
        dists >= inf_thresh, backend.full(dists.shape, inf_thresh), dists
    )

    # For each k, compute mean of k-th nearest neighbor distances
    # Sort distances for each point (rows), exclude self (col 0)
    sorted_dists = backend.sort(dists, axis=1)
    mean_k_dists = backend.mean(sorted_dists[:, 1:], axis=0)
    backend.eval(mean_k_dists)

    if int(mean_k_dists.shape[0]) < 3:
        # Not enough points to compute elbow - use all neighbors
        return n - 1

    # Find elbow using discrete curvature
    # Curvature at point i ≈ |f''(i)| / (1 + f'(i)²)^(3/2)
    # where f'(i) ≈ (f(i+1) - f(i-1)) / 2
    # and f''(i) ≈ f(i+1) - 2*f(i) + f(i-1)

    # Derive tolerance from input array dtype
    div_eps = division_epsilon(backend, activations)

    y_prev = mean_k_dists[:-2]
    y_curr = mean_k_dists[1:-1]
    y_next = mean_k_dists[2:]

    dy = (y_next - y_prev) / 2.0
    d2y = y_next - 2.0 * y_curr + y_prev

    denom = backend.sqrt(1.0 + dy * dy)
    denom = denom * denom * denom
    denom = backend.maximum(denom, backend.full(denom.shape, div_eps))
    curvatures = backend.abs(d2y) / denom
    backend.eval(curvatures)

    # Find k with maximum curvature (the elbow)
    # k corresponds to mean_k_dists index (offset by +2 for central differencing)
    best_idx_arr = backend.argmax(curvatures)
    backend.eval(best_idx_arr)
    best_idx = int(backend.to_scalar(best_idx_arr))
    return best_idx + 2


class InfluenceType(str, Enum):
    """Type of probability density function for concept influence."""

    GAUSSIAN = "gaussian"  # Standard Riemannian Gaussian
    LAPLACIAN = "laplacian"  # Heavy-tailed, robust to outliers
    STUDENT_T = "student_t"  # Even heavier tails, df parameter
    UNIFORM = "uniform"  # Uniform within geodesic ball


@dataclass(frozen=True)
class RiemannianDensityConfig:
    """Configuration for Riemannian density estimation.

    All parameters are derived from observed data geometry:
    - student_t_df: from data kurtosis (df = 4 + 6/(kurtosis - 3))
    - covariance_regularization: from machine epsilon
    - k_neighbors: from elbow detection on k-NN distances
    """

    # Influence function type
    influence_type: InfluenceType = InfluenceType.GAUSSIAN
    # Degrees of freedom for Student-t (ignored for other types)
    # None = derived from data kurtosis: df = 4 + 6/(kurtosis - 3)
    student_t_df: float | None = None
    # Regularization for covariance estimation (numerical precision floor)
    covariance_regularization: float | None = None
    # Whether to use curvature correction for covariance
    use_curvature_correction: bool = True
    # Number of neighbors for local density estimation
    # None = derive from elbow detection on k-NN distances
    k_neighbors: int | None = None


@dataclass
class ConceptVolume:
    """A concept modeled as a probability distribution on the manifold.

    Attributes
    ----------
    concept_id : str
        Identifier for this concept.
    centroid : Array
        Mean position in activation space.
    covariance : Array
        Covariance matrix (curvature-corrected if configured).
    geodesic_radius : float
        Extent along manifold (accounts for curvature).
    local_curvature : LocalCurvature or None
        Estimated curvature at centroid.
    num_samples : int
        Number of activations used to estimate volume.
    influence_type : InfluenceType
        Type of influence function (gaussian, laplacian, etc).
    """

    concept_id: str
    centroid: "Array"
    covariance: "Array"
    geodesic_radius: float
    local_curvature: LocalCurvature | None
    num_samples: int
    influence_type: InfluenceType = InfluenceType.GAUSSIAN
    student_t_df: float = 3.0

    # Optional raw activations for cross-dimensional CKA comparison
    # When comparing volumes of different dimensions, CKA uses these
    # to compute Gram matrices (n x n) which are dimension-agnostic.
    raw_activations: "Array | None" = field(default=None, repr=False)

    # Geodesic context for proper log map computation in density_at
    # Stores k-NN graph and distance matrix for curvature-aware density
    _geodesic_context: "GeodesicDistanceResult | None" = field(default=None, repr=False)

    # Cached values for efficiency
    _precision: "Array | None" = field(default=None, repr=False)
    _log_det_cov: float | None = field(default=None, repr=False)

    @property
    def dimension(self) -> int:
        """Dimensionality of the concept space."""
        shape = self.centroid.shape
        return int(shape[0]) if len(shape) == 1 else int(shape[-1])

    @property
    def precision(self) -> "Array":
        """Precision matrix (inverse covariance)."""
        if self._precision is None:
            backend = get_default_backend()
            try:
                precision = backend.inv(self.covariance)
                backend.eval(precision)
                object.__setattr__(self, "_precision", precision)
            except Exception:
                # Regularize if singular
                reg_eps = regularization_epsilon(backend, self.covariance)
                reg_cov = self.covariance + reg_eps * backend.eye(self.dimension)
                precision = backend.inv(reg_cov)
                backend.eval(precision)
                object.__setattr__(self, "_precision", precision)
        return self._precision

    @property
    def log_det_covariance(self) -> float:
        """Log determinant of covariance for normalization."""
        if self._log_det_cov is None:
            backend = get_default_backend()
            # slogdet returns (sign, logdet) - we compute via eigenvalues
            eigenvalues = backend.eigh(self.covariance)[0]
            backend.eval(eigenvalues)
            min_eig = backend.min(eigenvalues)
            backend.eval(min_eig)
            if float(backend.to_scalar(min_eig)) <= 0.0:
                logdet = -inf_value(backend)
            else:
                logdet_arr = backend.sum(backend.log(eigenvalues))
                backend.eval(logdet_arr)
                logdet = float(backend.to_scalar(logdet_arr))
            object.__setattr__(self, "_log_det_cov", logdet)
        return self._log_det_cov

    @property
    def volume(self) -> float:
        """Approximate volume of the concept region.

        For Gaussian, this is sqrt(det(2*pi*e*Cov)) ≈ exp(0.5 * log_det + d/2 * log(2*pi*e))
        For uniform ball, this is volume of d-dimensional sphere with geodesic_radius.
        """
        backend = get_default_backend()
        d = self.dimension
        if self.influence_type == InfluenceType.UNIFORM:
            # Volume of d-sphere: (pi^(d/2) / Gamma(d/2 + 1)) * r^d
            if self.geodesic_radius <= 0:
                return 0.0
            pi = pi_value(backend)
            log_vol = (
                (d / 2) * log_scalar(pi, backend)
                - lgamma_scalar(d / 2 + 1.0, backend)
                + d * log_scalar(self.geodesic_radius, backend)
            )
            return exp_scalar(log_vol, backend)
        else:
            # Gaussian effective volume
            pi = pi_value(backend)
            e = e_value(backend)
            return exp_scalar(0.5 * self.log_det_covariance + d / 2 * log_scalar(2 * pi * e, backend), backend)

    @property
    def effective_radius(self) -> float:
        """Effective radius from covariance (geometric mean of eigenvalues)."""
        backend = get_default_backend()
        eigenvalues = backend.eigh(self.covariance)[0]
        backend.eval(eigenvalues)
        # Geometric mean via log: exp(mean(log(max(eig, tiny))))
        tiny = tiny_value(backend, eigenvalues)
        clamped = backend.maximum(eigenvalues, tiny)
        mean_log = backend.mean(backend.log(clamped))
        backend.eval(mean_log)
        return exp_scalar(0.5 * float(backend.to_scalar(mean_log)), backend)

    def _compute_tangent_vector(self, point: "Array") -> "Array":
        """Compute tangent vector from centroid to point using log map.

        Requires geodesic context; Euclidean fallback is invalid on curved manifolds.

        Args:
            point: Point in activation space

        Returns:
            Tangent vector at centroid pointing toward point
        """
        backend = get_default_backend()
        point_arr = backend.array(point)
        centroid_arr = backend.array(self.centroid)
        diff = point_arr - centroid_arr

        # Geodesic context is mandatory for curvature-correct log maps
        if self._geodesic_context is None or self.raw_activations is None:
            raise ValueError(
                "Geodesic context required for Riemannian log map. "
                "Build volumes with store_raw_activations=True."
            )

        # Compute geodesic distance from centroid to point
        rg = RiemannianGeometry(backend)

        # Attach point to the manifold graph
        geo_from_point = rg._geodesic_distances_from_query(
            self.raw_activations,
            point_arr,
            geo_result=self._geodesic_context,
        )

        # Also get distances from centroid to all activations
        geo_from_centroid = rg._geodesic_distances_from_query(
            self.raw_activations,
            centroid_arr,
            geo_result=self._geodesic_context,
        )
        backend.eval(geo_from_point, geo_from_centroid)

        # Geodesic distance from centroid to point is approximated by
        # finding the path through the graph: min_i(geo_centroid_to_i + geo_i_to_point)
        total_dists = geo_from_centroid + geo_from_point
        geo_dist = backend.min(total_dists)
        backend.eval(geo_dist)
        geo_dist_float = float(backend.to_scalar(geo_dist))

        # Euclidean distance
        euc_dist_sq = backend.sum(diff * diff)
        euc_dist_arr = backend.sqrt(euc_dist_sq)
        backend.eval(euc_dist_arr)
        euc_dist = float(backend.to_scalar(euc_dist_arr))

        # Scale factor: geodesic / euclidean
        # Use machine_epsilon for near-zero check
        if euc_dist < machine_epsilon(backend, diff):
            return diff  # Point is at centroid
        scale = geo_dist_float / euc_dist

        return diff * scale

    def density_at(self, point: "Array") -> float:
        """Compute probability density at a point.

        Uses proper Riemannian log map when geodesic context is available,
        scaling the tangent vector by geodesic/Euclidean ratio.

        Args:
            point: Point in activation space (d-dimensional)

        Returns:
            Probability density value
        """
        backend = get_default_backend()

        # Compute tangent vector (log map if geodesic context available)
        tangent = self._compute_tangent_vector(point)

        # mahal_sq = tangent @ precision @ tangent
        temp = backend.matmul(tangent, self.precision)
        mahal_sq_arr = backend.matmul(temp, tangent)
        backend.eval(mahal_sq_arr)
        mahal_sq = float(backend.to_scalar(mahal_sq_arr))

        d = self.dimension
        pi = pi_value(backend)

        if self.influence_type == InfluenceType.GAUSSIAN:
            # Multivariate Gaussian
            log_norm = -0.5 * (d * log_scalar(2 * pi, backend) + self.log_det_covariance)
            return exp_scalar(log_norm - 0.5 * mahal_sq, backend)

        elif self.influence_type == InfluenceType.LAPLACIAN:
            # Multivariate Laplacian (product of univariate)
            mahal = sqrt_scalar(mahal_sq, backend)
            return exp_scalar(-mahal, backend) / (2**d)

        elif self.influence_type == InfluenceType.STUDENT_T:
            # Multivariate t-distribution
            nu = float(self.student_t_df)
            if nu <= 0:
                raise ValueError("student_t_df must be positive")
            log_norm = (
                lgamma_scalar((nu + d) / 2, backend)
                - lgamma_scalar(nu / 2, backend)
                - d / 2 * log_scalar(nu * pi, backend)
                - 0.5 * self.log_det_covariance
            )
            return exp_scalar(log_norm, backend) * (1 + mahal_sq / nu) ** (-(nu + d) / 2)

        elif self.influence_type == InfluenceType.UNIFORM:
            # Uniform ball
            if mahal_sq <= self.geodesic_radius**2:
                return 1.0 / self.volume
            return 0.0

        return 0.0

    def mahalanobis_distance(self, point: "Array") -> float:
        """Compute Mahalanobis distance from centroid to point."""
        backend = get_default_backend()
        diff = point - self.centroid
        # mahal_sq = diff @ precision @ diff
        temp = backend.matmul(diff, self.precision)
        mahal_sq_arr = backend.matmul(temp, diff)
        backend.eval(mahal_sq_arr)
        mahal_sq = float(backend.to_scalar(mahal_sq_arr))
        return sqrt_scalar(mahal_sq, backend)

    def geodesic_distance(self, point: "Array") -> float:
        """Compute geodesic distance from centroid to point.

        Uses k-NN graph shortest path estimation. This is the only correct
        distance metric in curved high-dimensional spaces.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        backend = get_default_backend()
        # Stack centroid and point into (2, d) array
        centroid_2d = backend.reshape(self.centroid, (1, -1))
        point_2d = backend.reshape(point, (1, -1))
        points = backend.concatenate([centroid_2d, point_2d], axis=0)
        points_arr = backend.astype(points, "float32")
        backend.eval(points_arr)

        # Compute geodesic distance
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=1, backend=backend)
        geo_elem = geo_dist[0, 1]
        backend.eval(geo_elem)
        return float(backend.to_scalar(geo_elem))

    def contains(self, point: "Array") -> bool:
        """Check if point is within concept volume.

        Uses geodesic radius criterion: point is inside if its geodesic
        distance from centroid is within the volume's extent. The geodesic
        radius is derived from the data during volume estimation.

        Args:
            point: Point to check

        Returns:
            True if point is within volume
        """
        dist = self.geodesic_distance_to(point)
        return dist <= self.geodesic_radius

    def mahalanobis_distance_batch(self, points: "Array") -> "Array":
        """Compute Mahalanobis distance from centroid to multiple points.

        Args:
            points: Array of points (n x d)

        Returns:
            Array of Mahalanobis distances (n,)
        """
        backend = get_default_backend()
        # diff: (n, d)
        diff = points - self.centroid
        # mahal_sq = sum((diff @ precision) * diff, axis=1)
        temp = backend.matmul(diff, self.precision)  # (n, d)
        mahal_sq = backend.sum(temp * diff, axis=1)  # (n,)
        backend.eval(mahal_sq)
        return backend.sqrt(backend.maximum(mahal_sq, backend.array(0.0)))

    def _compute_tangent_vectors_batch(self, points: "Array") -> "Array":
        """Compute tangent vectors from centroid to multiple points using log map.

        If geodesic context is available, scales by geodesic/Euclidean ratio.
        Otherwise returns raw Euclidean differences.

        Args:
            points: Array of points (n x d)

        Returns:
            Tangent vectors at centroid (n x d)
        """
        backend = get_default_backend()
        points_arr = backend.array(points)
        centroid_arr = backend.array(self.centroid)
        diff = points_arr - centroid_arr  # (n, d)

        # Geodesic context is required - no Euclidean fallback on curved manifolds
        if self._geodesic_context is None or self.raw_activations is None:
            raise ValueError(
                "Geodesic context required for Riemannian log map. "
                "Build volumes with store_raw_activations=True."
            )

        backend.eval(points_arr)
        n_points = int(points_arr.shape[0])
        rg = RiemannianGeometry(backend)

        # Compute geodesic distances from centroid to all activation points
        geo_from_centroid = rg._geodesic_distances_from_query(
            self.raw_activations,
            centroid_arr,
            geo_result=self._geodesic_context,
        )
        backend.eval(geo_from_centroid)

        # Compute geodesic distances for each query point
        # This is the expensive part - done point by point
        scales = []
        for i in range(n_points):
            point_i = points_arr[i]

            # Geodesic distances from this point to all activations
            geo_from_point = rg._geodesic_distances_from_query(
                self.raw_activations,
                point_i,
                geo_result=self._geodesic_context,
            )
            backend.eval(geo_from_point)

            # Geodesic distance to centroid via shortest path through graph
            total_dists = geo_from_centroid + geo_from_point
            geo_dist = backend.min(total_dists)
            backend.eval(geo_dist)
            geo_dist_float = float(backend.to_scalar(geo_dist))

            # Euclidean distance
            diff_i = diff[i]
            euc_dist_sq = backend.sum(diff_i * diff_i)
            euc_dist_arr = backend.sqrt(euc_dist_sq)
            backend.eval(euc_dist_arr)
            euc_dist = float(backend.to_scalar(euc_dist_arr))

            # Scale factor (use machine_epsilon for near-zero check)
            if euc_dist < machine_epsilon(backend, diff_i):
                scales.append(1.0)
            else:
                scales.append(geo_dist_float / euc_dist)

        # Apply scales to differences
        scales_arr = backend.array(scales)
        scales_col = backend.reshape(scales_arr, (-1, 1))
        tangent_vectors = diff * scales_col

        return tangent_vectors

    def density_at_batch(self, points: "Array") -> "Array":
        """Compute probability density at multiple points (vectorized).

        Uses proper Riemannian log map when geodesic context is available.

        Args:
            points: Array of points (n x d)

        Returns:
            Array of probability density values (n,)
        """
        backend = get_default_backend()
        d = self.dimension
        pi = pi_value(backend)

        # Compute tangent vectors (log map if geodesic context available)
        tangent = self._compute_tangent_vectors_batch(points)

        # Mahalanobis squared using tangent vectors
        temp = backend.matmul(tangent, self.precision)  # (n, d)
        mahal_sq = backend.sum(temp * tangent, axis=1)  # (n,)

        if self.influence_type == InfluenceType.GAUSSIAN:
            # Multivariate Gaussian: exp(log_norm - 0.5 * mahal_sq)
            log_norm = -0.5 * (d * log_scalar(2 * pi, backend) + self.log_det_covariance)
            densities = backend.exp(backend.array(log_norm) - 0.5 * mahal_sq)

        elif self.influence_type == InfluenceType.LAPLACIAN:
            # Multivariate Laplacian: exp(-sqrt(mahal_sq)) / 2^d
            mahal = backend.sqrt(backend.maximum(mahal_sq, backend.array(0.0)))
            densities = backend.exp(-mahal) / (2**d)

        elif self.influence_type == InfluenceType.STUDENT_T:
            # Multivariate t-distribution
            nu = float(self.student_t_df)
            log_norm = (
                lgamma_scalar((nu + d) / 2, backend)
                - lgamma_scalar(nu / 2, backend)
                - d / 2 * log_scalar(nu * pi, backend)
                - 0.5 * self.log_det_covariance
            )
            densities = backend.exp(backend.array(log_norm)) * backend.pow(
                1 + mahal_sq / nu, -(nu + d) / 2
            )

        elif self.influence_type == InfluenceType.UNIFORM:
            # Uniform ball: 1/volume if inside, 0 otherwise
            vol = self.volume
            inside = mahal_sq <= self.geodesic_radius**2
            densities = backend.where(
                inside,
                backend.full(mahal_sq.shape, 1.0 / vol if vol > 0 else 0.0),
                backend.zeros(mahal_sq.shape),
            )
        else:
            densities = backend.zeros(mahal_sq.shape)

        backend.eval(densities)
        return densities

    def contains_batch(self, points: "Array") -> "Array":
        """Check if multiple points are within concept volume (vectorized).

        Uses Mahalanobis distance criterion: points within geodesic_radius
        of centroid are inside. The geodesic_radius is derived from the
        data during volume estimation.

        Args:
            points: Array of points (n x d)

        Returns:
            Boolean array indicating membership (n,)
        """
        backend = get_default_backend()
        mahal_dist = self.mahalanobis_distance_batch(points)
        result = mahal_dist <= self.geodesic_radius
        backend.eval(result)
        return result


@dataclass
class ConceptVolumeRelation:
    """Relationship between two concept volumes."""

    volume_a: ConceptVolume
    volume_b: ConceptVolume
    # Overlap metrics
    overlap_coefficient: float  # Szymkiewicz-Simpson coefficient
    jaccard_index: float  # Intersection / Union volume ratio
    bhattacharyya_coefficient: float  # Distribution similarity
    # Distance metrics (both geodesic - the correct metric for curved manifolds)
    centroid_distance: float  # Geodesic between centroids
    geodesic_centroid_distance: float  # Geodesic between centroids (same value)
    mahalanobis_distance_ab: float  # Mahal from A's perspective
    mahalanobis_distance_ba: float  # Mahal from B's perspective
    # Curvature mismatch
    curvature_divergence: float
    # Dimensionality overlap
    subspace_alignment: float  # Alignment of principal axes


class RiemannianDensityEstimator:
    """Estimates concept volumes with curvature awareness.

    This is the core class for CABE-4, providing:
    1. ConceptVolume estimation from activations
    2. Curvature-corrected covariance
    3. Volume overlap computation
    4. Interference prediction foundation
    """

    def __init__(self, config: RiemannianDensityConfig | None = None):
        self.config = config or RiemannianDensityConfig()
        self.curvature_estimator = SectionalCurvatureEstimator()

    def estimate_concept_volume(
        self,
        concept_id: str,
        activations: "Array",
        metric_fn: Callable[["Array"], "Array"] | None = None,
        store_raw_activations: bool = True,
    ) -> ConceptVolume:
        """Estimate concept volume from activation samples.

        Args:
            concept_id: Identifier for the concept
            activations: Array of activation vectors (n x d)
            metric_fn: Optional metric tensor function for Riemannian geometry
            store_raw_activations: If True, store raw activations for CKA comparison
                                   across different dimensions (required for geodesic log map)

        Returns:
            ConceptVolume modeling the concept's distribution
        """
        backend = get_default_backend()
        # Convert to backend array if needed (handles numpy from tests)
        activations = backend.array(activations)
        backend.eval(activations)
        shape = activations.shape
        n, d = int(shape[0]), int(shape[1])

        if n < 2:
            # Single sample - return point mass
            geodesic_context = None
            if store_raw_activations:
                rg = RiemannianGeometry(backend)
                geodesic_context = rg.geodesic_distances(activations, k_neighbors=1)
            return ConceptVolume(
                concept_id=concept_id,
                centroid=activations[0],
                covariance=backend.eye(d) * regularization_epsilon(backend, activations),
                geodesic_radius=0.0,
                local_curvature=None,
                num_samples=n,
                influence_type=self.config.influence_type,
                student_t_df=self.config.student_t_df,
                raw_activations=activations if store_raw_activations else None,
                _geodesic_context=geodesic_context if store_raw_activations else None,
            )

        # Compute centroid using Fréchet mean - the only correct method on curved manifolds
        # Arithmetic mean is WRONG as it doesn't minimize squared geodesic distances
        # No fallback to arithmetic mean - if this fails, it's a bug we need to fix
        rg = RiemannianGeometry(backend)
        result = rg.frechet_mean(
            activations,
            max_iterations=50,
            tolerance=regularization_epsilon(backend, activations),
        )
        centroid = result.mean
        backend.eval(centroid)

        # Compute geodesic context for proper log map in density_at
        # k is derived from the data using elbow detection on k-NN distances
        if self.config.k_neighbors is not None:
            k_neighbors = min(self.config.k_neighbors, n - 1) if n > 1 else 1
        else:
            k_neighbors = _find_k_elbow(activations, backend)
        geodesic_context = rg.geodesic_distances(activations, k_neighbors=k_neighbors)

        # Estimate local curvature at centroid
        local_curvature = None
        if self.config.use_curvature_correction and n >= d + 2:
            try:
                local_curvature = self.curvature_estimator.estimate_local_curvature(
                    centroid, activations, metric_fn
                )
            except Exception as e:
                logger.warning(f"Curvature estimation failed for {concept_id}: {e}")

        # Compute covariance with curvature correction
        covariance = self._estimate_covariance(activations, centroid, local_curvature, metric_fn)

        # Compute geodesic radius (extent of activations from centroid)
        geodesic_radius = self._compute_geodesic_radius(activations, centroid)

        # Derive student_t_df from data kurtosis when not specified
        # Formula: df = 4 + 6/(kurtosis - 3), where kurtosis > 3 for heavy tails
        # Student-t with df=3 has kurtosis=inf, df→∞ approaches Gaussian (kurtosis=3)
        if self.config.student_t_df is not None:
            student_t_df = self.config.student_t_df
        else:
            # Compute excess kurtosis from centered activations
            centered = activations - centroid
            var = backend.mean(centered * centered)
            fourth = backend.mean(centered * centered * centered * centered)
            backend.eval(var, fourth)
            var_val = float(backend.to_scalar(var))
            fourth_val = float(backend.to_scalar(fourth))
            div_eps = division_epsilon(backend, activations)
            if var_val > div_eps:
                kurtosis = fourth_val / (var_val * var_val)
            else:
                kurtosis = 3.0  # Default to Gaussian kurtosis
            # df = 4 + 6/(kurtosis - 3), clamped to [2, 30]
            # kurtosis=3 → Gaussian, kurtosis>3 → heavy tails → lower df
            if kurtosis > 3.0 + div_eps:
                student_t_df = 4.0 + 6.0 / (kurtosis - 3.0)
                student_t_df = max(2.0, min(30.0, student_t_df))
            else:
                student_t_df = 30.0  # High df approaches Gaussian

        return ConceptVolume(
            concept_id=concept_id,
            centroid=centroid,
            covariance=covariance,
            geodesic_radius=geodesic_radius,
            local_curvature=local_curvature,
            num_samples=n,
            influence_type=self.config.influence_type,
            student_t_df=student_t_df,
            raw_activations=activations if store_raw_activations else None,
            _geodesic_context=geodesic_context if store_raw_activations else None,
        )

    def compute_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Compute relationship between two concept volumes.

        This is the foundation for interference prediction.

        Uses CKA (Centered Kernel Alignment) for all comparisons when raw
        activations are available. CKA computes Gram matrices (n x n) that are
        dimension-agnostic and GPU-accelerated. This is the correct approach:
        - Dimensions are compression/expansion choices, not fundamental structure
        - CKA captures the invariant representational geometry
        - Runs entirely on GPU (no SciPy/NumPy fallback)

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume

        Returns:
            ConceptVolumeRelation with all overlap/distance metrics
        """
        # Use CKA for all comparisons when raw_activations available
        # CKA is dimension-agnostic and GPU-accelerated
        if volume_a.raw_activations is not None and volume_b.raw_activations is not None:
            return self._compute_cka_relation(volume_a, volume_b)

        # Fallback to geodesic-based comparison only when raw_activations not available
        # (e.g., when loading cached volumes without activations)
        return self._compute_geodesic_relation(volume_a, volume_b)

    def _compute_geodesic_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Fallback: geodesic-based comparison for cached volumes without activations.

        Uses backend Floyd-Warshall for geodesic distance computation.
        Only used when raw_activations not available.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        backend = get_default_backend()

        # Must be same dimension for centroid comparison
        if volume_a.dimension != volume_b.dimension:
            raise ValueError(
                f"Geodesic comparison requires same dimensions. "
                f"Got {volume_a.dimension} vs {volume_b.dimension}. "
                f"Enable store_raw_activations=True for cross-dimensional comparison."
            )

        # Handle edge case: coincident centroids have geodesic distance 0 by definition
        diff = volume_a.centroid - volume_b.centroid
        diff_norm = backend.norm(diff)
        backend.eval(diff_norm)
        centroid_diff = float(backend.to_scalar(diff_norm))
        if centroid_diff < machine_epsilon(backend, diff):
            centroid_distance = 0.0
        else:
            centroid_a_2d = backend.reshape(volume_a.centroid, (1, -1))
            centroid_b_2d = backend.reshape(volume_b.centroid, (1, -1))
            centroids = backend.concatenate([centroid_a_2d, centroid_b_2d], axis=0)
            centroids_arr = backend.astype(centroids, "float32")

            geo_dist = geodesic_distance_matrix(centroids_arr, k_neighbors=1, backend=backend)
            geo_elem = geo_dist[0, 1]
            backend.eval(geo_elem)
            centroid_distance = float(backend.to_scalar(geo_elem))

        geodesic_centroid_distance = centroid_distance

        # Mahalanobis distances (asymmetric)
        mahal_ab = volume_a.mahalanobis_distance(volume_b.centroid)
        mahal_ba = volume_b.mahalanobis_distance(volume_a.centroid)

        # Bhattacharyya coefficient for Gaussians
        bhattacharyya = self._bhattacharyya_coefficient(volume_a, volume_b)

        # Overlap coefficient (Szymkiewicz-Simpson)
        overlap = self._overlap_coefficient(volume_a, volume_b)

        # Jaccard index
        jaccard = self._jaccard_index(volume_a, volume_b)

        # Curvature divergence
        curvature_div = self._curvature_divergence(volume_a, volume_b)

        # Subspace alignment
        subspace_align = self._subspace_alignment(volume_a, volume_b)

        return ConceptVolumeRelation(
            volume_a=volume_a,
            volume_b=volume_b,
            overlap_coefficient=overlap,
            jaccard_index=jaccard,
            bhattacharyya_coefficient=bhattacharyya,
            centroid_distance=centroid_distance,
            geodesic_centroid_distance=geodesic_centroid_distance,
            mahalanobis_distance_ab=mahal_ab,
            mahalanobis_distance_ba=mahal_ba,
            curvature_divergence=curvature_div,
            subspace_alignment=subspace_align,
        )

    def _compute_cka_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Compute relation between volumes using CKA (GPU-accelerated).

        CKA (Centered Kernel Alignment) computes Gram matrices (n x n) which
        are dimension-agnostic - it measures representational similarity
        regardless of dimensionality. This is the primary method:
        - Works for same or different dimensions
        - Runs entirely on GPU
        - Captures invariant representational geometry

        CKA = 1.0 means identical representational geometry (exact alignment)
        CKA = 0.0 means orthogonal representations (no overlap)

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume

        Returns:
            ConceptVolumeRelation with CKA-derived metrics
        """
        from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka_backend

        backend = get_default_backend()

        # CKA requires raw activations - this should not happen if caller
        # follows the contract (store_raw_activations=True)
        if volume_a.raw_activations is None or volume_b.raw_activations is None:
            raise ValueError(
                f"CKA comparison requires raw_activations. "
                f"Volume {volume_a.concept_id} missing activations. "
                f"Enable store_raw_activations=True when creating volumes."
            )

        # Compute CKA - this is dimension-agnostic
        # CKA uses Gram matrices K = X @ X.T (n x n) not raw dimensions
        # Use BIASED estimator to avoid eigh on potentially ill-conditioned matrices
        cka_similarity = compute_cka_backend(
            volume_a.raw_activations,
            volume_b.raw_activations,
            backend=backend,
            estimator=HSICEstimator.BIASED,
            feature_bias_correction=False,
        )

        # CKA measures representational similarity:
        # - CKA ~ 1.0 = same representational structure = high overlap
        # - CKA ~ 0.0 = different representations = no overlap
        # - CKA in between = partial alignment

        # Map CKA to our metrics:
        # - overlap_coefficient: CKA directly measures overlap in representation space
        # - bhattacharyya: CKA approximates distribution overlap
        # - jaccard: CKA approximates concept intersection
        # - subspace_alignment: CKA measures alignment directly
        overlap = cka_similarity
        bhattacharyya = cka_similarity
        jaccard = cka_similarity
        subspace_align = cka_similarity

        # Distance is inverse of similarity: CKA=1→distance=0, CKA=0→distance=1
        # This is a "representational distance" not Euclidean
        centroid_distance = 1.0 - cka_similarity
        geodesic_centroid_distance = centroid_distance

        # Mahalanobis doesn't apply across dimensions - use CKA-derived distance
        mahal_ab = centroid_distance
        mahal_ba = centroid_distance

        # Curvature: use average of local curvatures if available
        curvature_div = self._curvature_divergence(volume_a, volume_b)

        return ConceptVolumeRelation(
            volume_a=volume_a,
            volume_b=volume_b,
            overlap_coefficient=overlap,
            jaccard_index=jaccard,
            bhattacharyya_coefficient=bhattacharyya,
            centroid_distance=centroid_distance,
            geodesic_centroid_distance=geodesic_centroid_distance,
            mahalanobis_distance_ab=mahal_ab,
            mahalanobis_distance_ba=mahal_ba,
            curvature_divergence=curvature_div,
            subspace_alignment=subspace_align,
        )

    def _estimate_covariance(
        self,
        activations: "Array",
        centroid: "Array",
        local_curvature: LocalCurvature | None,
        metric_fn: Callable[["Array"], "Array"] | None,
    ) -> "Array":
        """Estimate covariance with optional curvature correction.

        Standard covariance assumes flat (Euclidean) space. In curved
        spaces, we compute covariance in the tangent space at the Fréchet mean,
        using the logarithmic map to project points onto the tangent space.

        This is the proper Riemannian covariance that respects manifold geometry.
        """
        backend = get_default_backend()
        shape = activations.shape
        d = int(shape[1])

        rg = RiemannianGeometry(backend)

        # Compute Riemannian covariance in tangent space at centroid
        # No fallback - if this fails, it's a bug we need to fix
        cov = rg.riemannian_covariance(
            activations,
            mean=centroid,
        )

        # Regularize
        reg_eps = (
            self.config.covariance_regularization
            if self.config.covariance_regularization is not None
            else regularization_epsilon(backend, cov)
        )
        cov = cov + reg_eps * backend.eye(d)
        backend.eval(cov)

        # Metric correction if available
        if metric_fn is not None:
            cov = self._apply_metric_correction(cov, centroid, metric_fn)

        return cov

    def _apply_metric_correction(
        self,
        cov: "Array",
        centroid: "Array",
        metric_fn: Callable[["Array"], "Array"],
    ) -> "Array":
        """Apply metric tensor correction to covariance.

        Transforms covariance to metric coordinates:
        Cov_metric = M^{-1/2} @ Cov @ M^{-1/2}
        """
        backend = get_default_backend()
        try:
            metric = metric_fn(centroid)
            eigenvalues, eigenvectors = backend.eigh(metric)
            backend.eval(eigenvalues, eigenvectors)

            # Compute inverse sqrt of eigenvalues (use tiny_value to prevent sqrt(0))
            tiny = tiny_value(backend, eigenvalues)
            inv_sqrt_eigs = 1.0 / backend.sqrt(backend.maximum(eigenvalues, backend.array(tiny)))
            inv_sqrt_metric = backend.matmul(
                eigenvectors,
                backend.matmul(backend.diag(inv_sqrt_eigs), backend.transpose(eigenvectors)),
            )

            result = backend.matmul(inv_sqrt_metric, backend.matmul(cov, inv_sqrt_metric))
            backend.eval(result)
            return result
        except Exception:
            return cov  # Keep uncorrected covariance

    def _apply_curvature_correction(
        self,
        covariance: "Array",
        local_curvature: LocalCurvature,
    ) -> "Array":
        """Apply curvature-based correction to covariance.

        In positively curved spaces, covariance underestimates spread.
        In negatively curved spaces, covariance overestimates spread.
        """
        backend = get_default_backend()
        K = local_curvature.mean_sectional

        # Use machine_epsilon for near-zero curvature check
        if abs(K) < machine_epsilon(backend, covariance):
            return covariance

        # Curvature correction factor
        # Based on comparison of volume elements in curved vs flat space
        # For sphere: dV_curved/dV_flat = (sin(r*sqrt(K))/(r*sqrt(K)))^(d-1)
        # For small curvature: ≈ 1 - K*r^2/6 for positive K

        # Use effective radius
        trace_val = backend.trace(covariance)
        shape = covariance.shape
        r_arr = backend.sqrt(trace_val / int(shape[0]))
        backend.eval(r_arr)
        r = float(backend.to_scalar(r_arr))

        if K > 0:
            # Positive curvature - expand covariance
            correction = 1.0 + K * r * r / 6
        else:
            # Negative curvature - shrink covariance
            correction = 1.0 / (1.0 - K * r * r / 6)

        # Clamp to a bounded stability range
        correction = max(0.5, min(2.0, correction))

        return covariance * correction

    def _compute_geodesic_radius(
        self,
        activations: "Array",
        centroid: "Array",
    ) -> float:
        """Compute geodesic radius (95th percentile distance from centroid).

        Uses geodesic distances via k-NN graph. No fallback to Euclidean -
        if this fails, it's a bug we need to fix.
        """
        backend = get_default_backend()
        shape = activations.shape
        n = int(shape[0])
        rg = RiemannianGeometry(backend)

        # Add centroid to points for distance computation
        centroid_2d = backend.reshape(centroid, (1, -1))
        points_with_centroid = backend.concatenate([centroid_2d, activations], axis=0)
        points_arr = backend.astype(points_with_centroid, "float32")
        backend.eval(points_arr)

        # Get geodesic distances from centroid (index 0) to all points
        k_neighbors = min(max(3, n // 3), n)
        geo_result = rg.geodesic_distances(points_arr, k_neighbors=k_neighbors)
        centroid_to_points = geo_result.distances[0, 1:]
        sorted_dists = backend.sort(centroid_to_points)
        backend.eval(sorted_dists)
        count = int(sorted_dists.shape[0])
        if count == 0:
            return 0.0
        idx = min(int(count * 0.95), count - 1)
        elem = sorted_dists[idx]
        backend.eval(elem)
        return float(backend.to_scalar(elem))

    def _bhattacharyya_coefficient(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute Bhattacharyya coefficient between two Gaussian volumes.

        BC = exp(-D_B), where D_B is Bhattacharyya distance.
        D_B = (1/8)(μ_a - μ_b)^T Σ^{-1} (μ_a - μ_b) + (1/2)ln(det(Σ)/sqrt(det(Σ_a)det(Σ_b)))
        where Σ = (Σ_a + Σ_b)/2
        """
        backend = get_default_backend()
        diff = volume_a.centroid - volume_b.centroid
        cov_avg = (volume_a.covariance + volume_b.covariance) / 2

        try:
            cov_avg_inv = backend.inv(cov_avg)
            backend.eval(cov_avg_inv)

            # term1 = 0.125 * diff @ cov_avg_inv @ diff
            temp = backend.matmul(diff, cov_avg_inv)
            term1_arr = backend.matmul(temp, diff)
            backend.eval(term1_arr)
            term1 = 0.125 * float(backend.to_scalar(term1_arr))

            # Compute log det of cov_avg via eigenvalues
            eigenvalues = backend.eigh(cov_avg)[0]
            backend.eval(eigenvalues)
            min_eig = backend.min(eigenvalues)
            backend.eval(min_eig)
            if float(backend.to_scalar(min_eig)) <= 0.0:
                return 0.0
            logdet_arr = backend.sum(backend.log(eigenvalues))
            backend.eval(logdet_arr)
            logdet_avg = float(backend.to_scalar(logdet_arr))

            term2 = 0.5 * (
                logdet_avg - 0.5 * (volume_a.log_det_covariance + volume_b.log_det_covariance)
            )

            db = term1 + term2
            return exp_scalar(-db, backend)

        except Exception:
            return 0.0

    def _overlap_coefficient(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Estimate Szymkiewicz-Simpson overlap coefficient.

        OC = |A ∩ B| / min(|A|, |B|)

        For Gaussian distributions, we approximate using Monte Carlo.
        """
        backend = get_default_backend()
        # Monte Carlo estimation with samples from both distributions
        n_samples = 1000
        d = volume_a.dimension

        # Sample from multivariate normal: X = mean + L @ Z where L is Cholesky of cov
        # Z ~ N(0, I)
        def sample_mvn(centroid: "Array", covariance: "Array", n: int) -> "Array":
            # Cholesky decomposition: cov = L @ L^T
            try:
                chol = backend.cholesky(covariance)
            except Exception:
                # If Cholesky fails, use regularized covariance
                reg_eps = regularization_epsilon(backend, covariance)
                reg_cov = covariance + reg_eps * backend.eye(d)
                chol = backend.cholesky(reg_cov)

            # Generate standard normal samples
            backend.random_seed(42)
            z = backend.random_normal((n, d))

            # Transform: samples = centroid + z @ chol^T
            samples = centroid + backend.matmul(z, backend.transpose(chol))
            backend.eval(samples)
            return samples

        # Sample from volume_a
        samples_a = sample_mvn(volume_a.centroid, volume_a.covariance, n_samples)

        # Sample from volume_b
        samples_b = sample_mvn(volume_b.centroid, volume_b.covariance, n_samples)

        # Count samples using vectorized batch operations (GPU-accelerated)
        # Membership uses geodesic_radius - derived from each volume's data
        a_in_b_mask = volume_b.contains_batch(samples_a)
        b_in_a_mask = volume_a.contains_batch(samples_b)

        # Sum boolean masks to get counts
        a_in_b_sum = backend.sum(backend.astype(a_in_b_mask, "float32"))
        b_in_a_sum = backend.sum(backend.astype(b_in_a_mask, "float32"))
        backend.eval(a_in_b_sum, b_in_a_sum)

        a_in_b = int(backend.to_scalar(a_in_b_sum))
        b_in_a = int(backend.to_scalar(b_in_a_sum))

        # Overlap coefficient
        return max(a_in_b, b_in_a) / n_samples

    def _jaccard_index(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Estimate Jaccard index (intersection over union).

        J = |A ∩ B| / |A ∪ B|
        """
        # Use Bhattacharyya as proxy for intersection
        bc = self._bhattacharyya_coefficient(volume_a, volume_b)

        # For Gaussians: J ≈ BC / (2 - BC)
        # When BC is within machine epsilon of 1.0, the distributions are
        # effectively identical, so Jaccard = 1.0
        backend = get_default_backend()
        eps = machine_epsilon(backend, volume_a.centroid)
        if bc > 1.0 - eps:
            return 1.0
        return bc / (2 - bc)

    def _curvature_divergence(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute curvature mismatch between two volumes."""
        if volume_a.local_curvature is None or volume_b.local_curvature is None:
            return 0.0

        K_a = volume_a.local_curvature.mean_sectional
        K_b = volume_b.local_curvature.mean_sectional

        # Normalized divergence (use division_epsilon for safe division)
        backend = get_default_backend()
        div_eps = division_epsilon(backend, volume_a.centroid)
        return abs(K_a - K_b) / (abs(K_a) + abs(K_b) + div_eps)

    def _subspace_alignment(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute alignment of principal subspaces.

        For same-dimension: Uses principal angles between covariance eigenspaces.
        For cross-dimension: Uses CKA on Gram matrices (dimension-agnostic).
        Returns value in [0, 1] where 1 = exactly aligned.
        """
        backend = get_default_backend()

        dim_a = int(volume_a.covariance.shape[0])
        dim_b = int(volume_b.covariance.shape[0])

        if dim_a == dim_b:
            # Same dimension: use principal angles
            _, Va = backend.eigh(volume_a.covariance)
            _, Vb = backend.eigh(volume_b.covariance)
            backend.eval(Va, Vb)

            # Compute singular values of Va^T @ Vb
            # These are cosines of principal angles
            M = backend.matmul(backend.transpose(Va), Vb)
            singular_values = backend.svd(M)[1]
            backend.eval(singular_values)

            sq_vals = singular_values * singular_values
            alignment = backend.mean(sq_vals)
            backend.eval(alignment)
            result = float(backend.to_scalar(alignment))
        else:
            # Cross-architecture: use CKA on Gram matrices (dimension-agnostic)
            # Gram matrix K = X @ X^T has shape [n_samples, n_samples]
            # This is the SAME size regardless of feature dimension!
            from modelcypher.core.domain.geometry.cka import compute_cka_from_grams

            # Must use raw_activations to compute proper Gram matrices
            if volume_a.raw_activations is None or volume_b.raw_activations is None:
                logger.warning(
                    "Cross-architecture subspace alignment requires raw_activations. "
                    "Enable store_raw_activations=True when creating volumes."
                )
                return 0.0

            # Compute Gram matrices: K = X @ X^T → [n_samples, n_samples]
            # These have the SAME dimensions regardless of feature dimension
            gram_a = backend.matmul(
                volume_a.raw_activations,
                backend.transpose(volume_a.raw_activations),
            )
            gram_b = backend.matmul(
                volume_b.raw_activations,
                backend.transpose(volume_b.raw_activations),
            )
            backend.eval(gram_a, gram_b)

            result = compute_cka_from_grams(gram_a, gram_b, backend=backend)

        # Clamp to [0, 1] to handle floating point precision
        return max(0.0, min(1.0, result))


def batch_estimate_volumes(
    estimator: RiemannianDensityEstimator,
    concept_activations: dict[str, "Array"],
    metric_fn: Callable[["Array"], "Array"] | None = None,
) -> dict[str, ConceptVolume]:
    """Estimate volumes for multiple concepts.

    Args:
        estimator: RiemannianDensityEstimator instance
        concept_activations: Dict mapping concept_id to activation array
        metric_fn: Optional metric tensor function

    Returns:
        Dict mapping concept_id to ConceptVolume
    """
    volumes = {}
    for concept_id, activations in concept_activations.items():
        try:
            volumes[concept_id] = estimator.estimate_concept_volume(
                concept_id, activations, metric_fn
            )
        except Exception as e:
            logger.warning(f"Volume estimation failed for {concept_id}: {e}")
    return volumes


def compute_pairwise_relations(
    estimator: RiemannianDensityEstimator,
    volumes: dict[str, ConceptVolume],
) -> dict[tuple[str, str], ConceptVolumeRelation]:
    """Compute relations between all pairs of volumes.

    Args:
        estimator: RiemannianDensityEstimator instance
        volumes: Dict of concept volumes

    Returns:
        Dict mapping (concept_a, concept_b) to relation
    """
    relations = {}
    concept_ids = list(volumes.keys())

    for i, id_a in enumerate(concept_ids):
        for id_b in concept_ids[i + 1 :]:
            relation = estimator.compute_relation(volumes[id_a], volumes[id_b])
            relations[(id_a, id_b)] = relation

    return relations
