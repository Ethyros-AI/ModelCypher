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
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend

# Check if scipy is available for special functions
try:
    from scipy.special import gamma as scipy_gamma

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_gamma = None

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Array, Backend

from .manifold_curvature import (
    CurvatureConfig,
    LocalCurvature,
    SectionalCurvatureEstimator,
)
from .riemannian_utils import RiemannianGeometry

logger = logging.getLogger(__name__)


class InfluenceType(str, Enum):
    """Type of probability density function for concept influence."""

    GAUSSIAN = "gaussian"  # Standard Riemannian Gaussian
    LAPLACIAN = "laplacian"  # Heavy-tailed, robust to outliers
    STUDENT_T = "student_t"  # Even heavier tails, df parameter
    UNIFORM = "uniform"  # Uniform within geodesic ball


@dataclass(frozen=True)
class RiemannianDensityConfig:
    """Configuration for Riemannian density estimation."""

    # Influence function type
    influence_type: InfluenceType = InfluenceType.GAUSSIAN
    # Degrees of freedom for Student-t (ignored for other types)
    student_t_df: float = 3.0
    # Regularization for covariance estimation
    covariance_regularization: float = 1e-6
    # Whether to use curvature correction for covariance
    use_curvature_correction: bool = True
    # Number of neighbors for local density estimation
    k_neighbors: int = 20
    # Threshold for considering activations as part of concept volume
    membership_threshold: float = 0.05
    # Curvature estimation config
    curvature_config: CurvatureConfig = field(default_factory=CurvatureConfig)


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

    # Optional raw activations for cross-dimensional CKA comparison
    # When comparing volumes of different dimensions, CKA uses these
    # to compute Gram matrices (n x n) which are dimension-agnostic.
    raw_activations: "Array | None" = field(default=None, repr=False)

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
                reg_cov = self.covariance + 1e-6 * backend.eye(self.dimension)
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
            eig_np = backend.to_numpy(eigenvalues)
            # Sum of log eigenvalues = log(det)
            if all(e > 0 for e in eig_np):
                logdet = float(sum(math.log(e) for e in eig_np))
            else:
                logdet = -math.inf
            object.__setattr__(self, "_log_det_cov", logdet)
        return self._log_det_cov

    @property
    def volume(self) -> float:
        """Approximate volume of the concept region.

        For Gaussian, this is sqrt(det(2*pi*e*Cov)) ≈ exp(0.5 * log_det + d/2 * log(2*pi*e))
        For uniform ball, this is volume of d-dimensional sphere with geodesic_radius.
        """
        d = self.dimension
        if self.influence_type == InfluenceType.UNIFORM:
            # Volume of d-sphere: (pi^(d/2) / Gamma(d/2 + 1)) * r^d
            if HAS_SCIPY and scipy_gamma is not None:
                return (math.pi ** (d / 2) / scipy_gamma(d / 2 + 1)) * (self.geodesic_radius**d)
            else:
                # Approximation using Stirling's formula for gamma
                # Gamma(n+1) ≈ sqrt(2*pi*n) * (n/e)^n
                n = d / 2
                if n > 0:
                    gamma_approx = math.sqrt(2 * math.pi * n) * (n / math.e) ** n
                else:
                    gamma_approx = 1.0
                return (math.pi ** (d / 2) / gamma_approx) * (self.geodesic_radius**d)
        else:
            # Gaussian effective volume
            return math.exp(0.5 * self.log_det_covariance + d / 2 * math.log(2 * math.pi * math.e))

    @property
    def effective_radius(self) -> float:
        """Effective radius from covariance (geometric mean of eigenvalues)."""
        backend = get_default_backend()
        eigenvalues = backend.eigh(self.covariance)[0]
        backend.eval(eigenvalues)
        eig_np = backend.to_numpy(eigenvalues)
        # Geometric mean via log: exp(mean(log(max(eig, 1e-10))))
        log_eigs = [math.log(max(e, 1e-10)) for e in eig_np]
        mean_log = sum(log_eigs) / len(log_eigs)
        return math.exp(mean_log) ** 0.5

    def density_at(self, point: "Array") -> float:
        """Compute probability density at a point.

        Args:
            point: Point in activation space (d-dimensional)

        Returns:
            Probability density value
        """
        backend = get_default_backend()
        # Convert to backend arrays if needed (handles numpy from tests)
        point_arr = backend.array(point)
        centroid_arr = backend.array(self.centroid)
        diff = point_arr - centroid_arr
        # mahal_sq = diff @ precision @ diff
        temp = backend.matmul(diff, self.precision)
        mahal_sq_arr = backend.matmul(temp, diff)
        backend.eval(mahal_sq_arr)
        mahal_sq = float(backend.to_numpy(mahal_sq_arr))

        d = self.dimension

        if self.influence_type == InfluenceType.GAUSSIAN:
            # Multivariate Gaussian
            log_norm = -0.5 * (d * math.log(2 * math.pi) + self.log_det_covariance)
            return math.exp(log_norm - 0.5 * mahal_sq)

        elif self.influence_type == InfluenceType.LAPLACIAN:
            # Multivariate Laplacian (product of univariate)
            mahal = math.sqrt(mahal_sq)
            return math.exp(-mahal) / (2**d)

        elif self.influence_type == InfluenceType.STUDENT_T:
            # Multivariate t-distribution
            if not HAS_SCIPY or scipy_gamma is None:
                # Fall back to Gaussian approximation
                log_norm = -0.5 * (d * math.log(2 * math.pi) + self.log_det_covariance)
                return math.exp(log_norm - 0.5 * mahal_sq)

            nu = 3.0  # degrees of freedom
            log_norm = (
                math.log(scipy_gamma((nu + d) / 2))
                - math.log(scipy_gamma(nu / 2))
                - d / 2 * math.log(nu * math.pi)
                - 0.5 * self.log_det_covariance
            )
            return math.exp(log_norm) * (1 + mahal_sq / nu) ** (-(nu + d) / 2)

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
        mahal_sq = float(backend.to_numpy(mahal_sq_arr))
        return math.sqrt(mahal_sq)

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
        backend.eval(geo_dist)
        geo_dist_np = backend.to_numpy(geo_dist)

        return float(geo_dist_np[0, 1])

    def contains(self, point: "Array", threshold: float = 0.05) -> bool:
        """Check if point is within concept volume.

        Args:
            point: Point to check
            threshold: Density threshold for membership

        Returns:
            True if density at point exceeds threshold
        """
        return self.density_at(point) >= threshold


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
        self.curvature_estimator = SectionalCurvatureEstimator(self.config.curvature_config)

    def estimate_concept_volume(
        self,
        concept_id: str,
        activations: "Array",
        metric_fn: Callable[["Array"], "Array"] | None = None,
        store_raw_activations: bool = False,
    ) -> ConceptVolume:
        """Estimate concept volume from activation samples.

        Args:
            concept_id: Identifier for the concept
            activations: Array of activation vectors (n x d)
            metric_fn: Optional metric tensor function for Riemannian geometry
            store_raw_activations: If True, store raw activations for CKA comparison
                                   across different dimensions

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
            return ConceptVolume(
                concept_id=concept_id,
                centroid=activations[0],
                covariance=backend.eye(d) * 1e-6,
                geodesic_radius=0.0,
                local_curvature=None,
                num_samples=n,
                influence_type=self.config.influence_type,
                raw_activations=activations if store_raw_activations else None,
            )

        # Compute centroid using Fréchet mean for curvature-aware estimation
        # On curved manifolds, arithmetic mean doesn't minimize squared geodesic distances
        if self.config.use_curvature_correction and n >= self.config.k_neighbors:
            try:
                rg = RiemannianGeometry(backend)

                # Compute Fréchet mean (always uses geodesic distances - curvature is inherent)
                result = rg.frechet_mean(
                    activations,
                    max_iterations=50,
                    tolerance=1e-5,
                )
                centroid = result.mean
            except Exception as e:
                logger.warning(f"Fréchet mean failed, using arithmetic mean: {e}")
                centroid = backend.mean(activations, axis=0)
        else:
            # Fallback to arithmetic mean for small samples or when curvature disabled
            centroid = backend.mean(activations, axis=0)
        backend.eval(centroid)

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

        return ConceptVolume(
            concept_id=concept_id,
            centroid=centroid,
            covariance=covariance,
            geodesic_radius=geodesic_radius,
            local_curvature=local_curvature,
            num_samples=n,
            influence_type=self.config.influence_type,
            raw_activations=activations if store_raw_activations else None,
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
        - Runs entirely on GPU (no scipy/numpy fallback)

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

        Uses scipy's floyd_warshall (CPU) for geodesic distance computation.
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
        centroid_diff = float(backend.to_numpy(diff_norm))
        if centroid_diff < 1e-10:
            centroid_distance = 0.0
        else:
            centroid_a_2d = backend.reshape(volume_a.centroid, (1, -1))
            centroid_b_2d = backend.reshape(volume_b.centroid, (1, -1))
            centroids = backend.concatenate([centroid_a_2d, centroid_b_2d], axis=0)
            centroids_arr = backend.astype(centroids, "float32")

            geo_dist = geodesic_distance_matrix(centroids_arr, k_neighbors=1, backend=backend)
            backend.eval(geo_dist)
            geo_dist_np = backend.to_numpy(geo_dist)
            centroid_distance = float(geo_dist_np[0, 1])

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

        CKA = 1.0 means identical representational geometry (perfect alignment)
        CKA = 0.0 means orthogonal representations (no overlap)

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume

        Returns:
            ConceptVolumeRelation with CKA-derived metrics
        """
        from modelcypher.core.domain.geometry.cka import compute_cka_backend

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
        cka_similarity = compute_cka_backend(
            volume_a.raw_activations,
            volume_b.raw_activations,
            backend=backend,
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
        cov = cov + self.config.covariance_regularization * backend.eye(d)
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

            # Compute inverse sqrt of eigenvalues
            inv_sqrt_eigs = 1.0 / backend.sqrt(backend.maximum(eigenvalues, backend.array(1e-10)))
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

        if abs(K) < 1e-10:
            return covariance

        # Curvature correction factor
        # Based on comparison of volume elements in curved vs flat space
        # For sphere: dV_curved/dV_flat = (sin(r*sqrt(K))/(r*sqrt(K)))^(d-1)
        # For small curvature: ≈ 1 - K*r^2/6 for positive K

        # Use effective radius
        trace_val = backend.trace(covariance)
        shape = covariance.shape
        backend.eval(trace_val)
        r = math.sqrt(float(backend.to_numpy(trace_val)) / int(shape[0]))

        if K > 0:
            # Positive curvature - expand covariance
            correction = 1.0 + K * r * r / 6
        else:
            # Negative curvature - shrink covariance
            correction = 1.0 / (1.0 - K * r * r / 6)

        # Clamp to reasonable range
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
        geo_np = backend.to_numpy(geo_result.distances)

        # Distances from centroid (row 0) to all activation points (rows 1:n+1)
        centroid_to_points = list(geo_np[0, 1:])

        # Use 95th percentile as radius (manual percentile calculation)
        sorted_dists = sorted(centroid_to_points)
        idx = int(len(sorted_dists) * 0.95)
        idx = min(idx, len(sorted_dists) - 1)
        return float(sorted_dists[idx])

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
            term1 = 0.125 * float(backend.to_numpy(term1_arr))

            # Compute log det of cov_avg via eigenvalues
            eigenvalues = backend.eigh(cov_avg)[0]
            backend.eval(eigenvalues)
            eig_np = backend.to_numpy(eigenvalues)
            if all(e > 0 for e in eig_np):
                logdet_avg = float(sum(math.log(e) for e in eig_np))
            else:
                return 0.0

            term2 = 0.5 * (
                logdet_avg - 0.5 * (volume_a.log_det_covariance + volume_b.log_det_covariance)
            )

            db = term1 + term2
            return math.exp(-db)

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
                reg_cov = covariance + 1e-6 * backend.eye(d)
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

        # Count samples from A that are in B's high-density region
        threshold = self.config.membership_threshold
        a_in_b = sum(volume_b.contains(samples_a[i], threshold) for i in range(n_samples))
        b_in_a = sum(volume_a.contains(samples_b[i], threshold) for i in range(n_samples))

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
        if bc > 0.999:
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

        # Normalized divergence
        return abs(K_a - K_b) / (abs(K_a) + abs(K_b) + 1e-10)

    def _subspace_alignment(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute alignment of principal subspaces.

        Uses principal angles between covariance eigenspaces.
        Returns value in [0, 1] where 1 = perfectly aligned.
        """
        backend = get_default_backend()

        # Get principal directions (eigenvectors of covariance)
        _, Va = backend.eigh(volume_a.covariance)
        _, Vb = backend.eigh(volume_b.covariance)
        backend.eval(Va, Vb)

        # Compute singular values of Va^T @ Vb
        # These are cosines of principal angles
        M = backend.matmul(backend.transpose(Va), Vb)
        singular_values = backend.svd(M)[1]  # S is the second element
        backend.eval(singular_values)

        # Average of squared cosines (like CKA)
        sq_vals = singular_values * singular_values
        alignment = backend.mean(sq_vals)
        backend.eval(alignment)

        return float(backend.to_numpy(alignment))


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
