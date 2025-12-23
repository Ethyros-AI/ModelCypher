"""Riemannian density estimation for concept manifolds.

Models concepts as probability distributions over the representation manifold
rather than single points. This enables:
- Volume-based overlap computation for interference prediction
- Curvature-aware covariance estimation
- Geodesic distance computation that respects manifold geometry

Key Insight: A concept in neural network latent space is not a single point,
but a distribution of activations that vary with input phrasing, context,
and model stochasticity. ConceptVolume captures this distributional nature.

Mathematical Background:
- Riemannian Gaussian: Normal distribution on curved manifold
- Covariance accounts for local metric tensor
- Geodesic radius measures extent along manifold, not Euclidean distance
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TYPE_CHECKING

import numpy as np

# Check if scipy is available for special functions
try:
    from scipy.special import gamma as scipy_gamma
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_gamma = None

if TYPE_CHECKING:
    pass

from .manifold_curvature import (
    LocalCurvature,
    ManifoldCurvatureProfile,
    SectionalCurvatureEstimator,
    CurvatureConfig,
)

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
    """A concept modeled as a volume of influence on the manifold.

    Rather than treating a concept as a single point (centroid), this
    models it as a probability distribution with extent and shape.

    Attributes:
        concept_id: Identifier for this concept
        centroid: Mean position in activation space
        covariance: Covariance matrix (curvature-corrected if configured)
        geodesic_radius: Extent along manifold (accounts for curvature)
        local_curvature: Curvature at centroid
        num_samples: Number of activations used to estimate volume
        influence_type: Type of influence function
    """
    concept_id: str
    centroid: np.ndarray
    covariance: np.ndarray
    geodesic_radius: float
    local_curvature: LocalCurvature | None
    num_samples: int
    influence_type: InfluenceType = InfluenceType.GAUSSIAN

    # Cached values for efficiency
    _precision: np.ndarray | None = field(default=None, repr=False)
    _log_det_cov: float | None = field(default=None, repr=False)

    @property
    def dimension(self) -> int:
        """Dimensionality of the concept space."""
        return len(self.centroid)

    @property
    def precision(self) -> np.ndarray:
        """Precision matrix (inverse covariance)."""
        if self._precision is None:
            try:
                object.__setattr__(self, '_precision', np.linalg.inv(self.covariance))
            except np.linalg.LinAlgError:
                # Regularize if singular
                reg_cov = self.covariance + 1e-6 * np.eye(self.dimension)
                object.__setattr__(self, '_precision', np.linalg.inv(reg_cov))
        return self._precision

    @property
    def log_det_covariance(self) -> float:
        """Log determinant of covariance for normalization."""
        if self._log_det_cov is None:
            sign, logdet = np.linalg.slogdet(self.covariance)
            object.__setattr__(self, '_log_det_cov', logdet if sign > 0 else -np.inf)
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
                return (np.pi ** (d / 2) / scipy_gamma(d / 2 + 1)) * (self.geodesic_radius ** d)
            else:
                # Approximation using Stirling's formula for gamma
                # Gamma(n+1) ≈ sqrt(2*pi*n) * (n/e)^n
                n = d / 2
                if n > 0:
                    gamma_approx = np.sqrt(2 * np.pi * n) * (n / np.e) ** n
                else:
                    gamma_approx = 1.0
                return (np.pi ** (d / 2) / gamma_approx) * (self.geodesic_radius ** d)
        else:
            # Gaussian effective volume
            return np.exp(0.5 * self.log_det_covariance + d / 2 * np.log(2 * np.pi * np.e))

    @property
    def effective_radius(self) -> float:
        """Effective radius from covariance (geometric mean of eigenvalues)."""
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return np.exp(np.mean(np.log(np.maximum(eigenvalues, 1e-10)))) ** 0.5

    def density_at(self, point: np.ndarray) -> float:
        """Compute probability density at a point.

        Args:
            point: Point in activation space (d-dimensional)

        Returns:
            Probability density value
        """
        diff = point - self.centroid
        mahal_sq = diff @ self.precision @ diff

        d = self.dimension

        if self.influence_type == InfluenceType.GAUSSIAN:
            # Multivariate Gaussian
            log_norm = -0.5 * (d * np.log(2 * np.pi) + self.log_det_covariance)
            return np.exp(log_norm - 0.5 * mahal_sq)

        elif self.influence_type == InfluenceType.LAPLACIAN:
            # Multivariate Laplacian (product of univariate)
            mahal = np.sqrt(mahal_sq)
            return np.exp(-mahal) / (2 ** d)

        elif self.influence_type == InfluenceType.STUDENT_T:
            # Multivariate t-distribution
            if not HAS_SCIPY or scipy_gamma is None:
                # Fall back to Gaussian approximation
                log_norm = -0.5 * (d * np.log(2 * np.pi) + self.log_det_covariance)
                return np.exp(log_norm - 0.5 * mahal_sq)

            nu = 3.0  # degrees of freedom
            log_norm = (
                np.log(scipy_gamma((nu + d) / 2))
                - np.log(scipy_gamma(nu / 2))
                - d / 2 * np.log(nu * np.pi)
                - 0.5 * self.log_det_covariance
            )
            return np.exp(log_norm) * (1 + mahal_sq / nu) ** (-(nu + d) / 2)

        elif self.influence_type == InfluenceType.UNIFORM:
            # Uniform ball
            if mahal_sq <= self.geodesic_radius ** 2:
                return 1.0 / self.volume
            return 0.0

        return 0.0

    def mahalanobis_distance(self, point: np.ndarray) -> float:
        """Compute Mahalanobis distance from centroid to point."""
        diff = point - self.centroid
        return np.sqrt(diff @ self.precision @ diff)

    def geodesic_distance(self, point: np.ndarray) -> float:
        """Approximate geodesic distance accounting for curvature.

        For small distances, geodesic ≈ Euclidean.
        For larger distances, curvature correction applies.
        """
        euclidean = np.linalg.norm(point - self.centroid)

        if self.local_curvature is None or euclidean < 1e-6:
            return euclidean

        # Curvature correction: for constant sectional curvature K,
        # geodesic distance s satisfies:
        # - K > 0: s = (1/sqrt(K)) * arcsin(sqrt(K) * euclidean)
        # - K < 0: s = (1/sqrt(-K)) * arcsinh(sqrt(-K) * euclidean)
        # - K = 0: s = euclidean

        K = self.local_curvature.mean_sectional

        if abs(K) < 1e-10:
            return euclidean

        if K > 0:
            # Positive curvature (spherical)
            sqrt_K = np.sqrt(K)
            arg = sqrt_K * euclidean
            if arg > 1:
                # Beyond antipodal point
                return np.pi / sqrt_K
            return np.arcsin(arg) / sqrt_K

        else:
            # Negative curvature (hyperbolic)
            sqrt_neg_K = np.sqrt(-K)
            return np.arcsinh(sqrt_neg_K * euclidean) / sqrt_neg_K

    def contains(self, point: np.ndarray, threshold: float = 0.05) -> bool:
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
    # Distance metrics
    centroid_distance: float  # Euclidean between centroids
    geodesic_centroid_distance: float  # Geodesic between centroids
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
        self.curvature_estimator = SectionalCurvatureEstimator(
            self.config.curvature_config
        )

    def estimate_concept_volume(
        self,
        concept_id: str,
        activations: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> ConceptVolume:
        """Estimate concept volume from activation samples.

        Args:
            concept_id: Identifier for the concept
            activations: Array of activation vectors (n x d)
            metric_fn: Optional metric tensor function for Riemannian geometry

        Returns:
            ConceptVolume modeling the concept's distribution
        """
        n, d = activations.shape

        if n < 2:
            # Single sample - return point mass
            return ConceptVolume(
                concept_id=concept_id,
                centroid=activations[0],
                covariance=np.eye(d) * 1e-6,
                geodesic_radius=0.0,
                local_curvature=None,
                num_samples=n,
                influence_type=self.config.influence_type,
            )

        # Compute centroid
        centroid = np.mean(activations, axis=0)

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
        covariance = self._estimate_covariance(
            activations, centroid, local_curvature, metric_fn
        )

        # Compute geodesic radius (extent of activations from centroid)
        geodesic_radius = self._estimate_geodesic_radius(
            activations, centroid, local_curvature
        )

        return ConceptVolume(
            concept_id=concept_id,
            centroid=centroid,
            covariance=covariance,
            geodesic_radius=geodesic_radius,
            local_curvature=local_curvature,
            num_samples=n,
            influence_type=self.config.influence_type,
        )

    def compute_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Compute relationship between two concept volumes.

        This is the foundation for interference prediction.

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume

        Returns:
            ConceptVolumeRelation with all overlap/distance metrics
        """
        # Centroid distances
        centroid_distance = np.linalg.norm(volume_a.centroid - volume_b.centroid)
        geodesic_centroid_distance = volume_a.geodesic_distance(volume_b.centroid)

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

    def _estimate_covariance(
        self,
        activations: np.ndarray,
        centroid: np.ndarray,
        local_curvature: LocalCurvature | None,
        metric_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        """Estimate covariance with optional curvature correction.

        Standard covariance assumes flat (Euclidean) space. In curved
        spaces, we need to account for the metric tensor.
        """
        n, d = activations.shape
        centered = activations - centroid

        # Standard sample covariance
        cov = np.cov(centered.T) if d > 1 else np.array([[np.var(centered)]])

        # Ensure 2D
        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Regularize
        cov = cov + self.config.covariance_regularization * np.eye(d)

        # Curvature correction
        if self.config.use_curvature_correction and local_curvature is not None:
            cov = self._apply_curvature_correction(cov, local_curvature)

        # Metric correction if available
        if metric_fn is not None:
            metric = metric_fn(centroid)
            # Transform to metric coordinates: Cov_metric = M^{-1/2} Cov M^{-1/2}
            try:
                # Compute M^{-1/2}
                eigenvalues, eigenvectors = np.linalg.eigh(metric)
                inv_sqrt_metric = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10))) @ eigenvectors.T
                cov = inv_sqrt_metric @ cov @ inv_sqrt_metric
            except np.linalg.LinAlgError:
                pass  # Keep uncorrected covariance

        return cov

    def _apply_curvature_correction(
        self,
        covariance: np.ndarray,
        local_curvature: LocalCurvature,
    ) -> np.ndarray:
        """Apply curvature-based correction to covariance.

        In positively curved spaces, covariance underestimates spread.
        In negatively curved spaces, covariance overestimates spread.
        """
        K = local_curvature.mean_sectional

        if abs(K) < 1e-10:
            return covariance

        # Curvature correction factor
        # Based on comparison of volume elements in curved vs flat space
        # For sphere: dV_curved/dV_flat = (sin(r*sqrt(K))/(r*sqrt(K)))^(d-1)
        # For small curvature: ≈ 1 - K*r^2/6 for positive K

        # Use effective radius
        r = np.sqrt(np.trace(covariance) / covariance.shape[0])

        if K > 0:
            # Positive curvature - expand covariance
            correction = 1.0 + K * r * r / 6
        else:
            # Negative curvature - shrink covariance
            correction = 1.0 / (1.0 - K * r * r / 6)

        # Clamp to reasonable range
        correction = np.clip(correction, 0.5, 2.0)

        return covariance * correction

    def _estimate_geodesic_radius(
        self,
        activations: np.ndarray,
        centroid: np.ndarray,
        local_curvature: LocalCurvature | None,
    ) -> float:
        """Estimate geodesic radius (max distance from centroid)."""
        distances = np.linalg.norm(activations - centroid, axis=1)
        euclidean_radius = np.percentile(distances, 95)  # 95th percentile

        if local_curvature is None:
            return euclidean_radius

        # Apply curvature correction
        K = local_curvature.mean_sectional

        if abs(K) < 1e-10:
            return euclidean_radius

        if K > 0:
            # Positive curvature - geodesic shorter than Euclidean
            sqrt_K = np.sqrt(K)
            arg = sqrt_K * euclidean_radius
            if arg < 1:
                return np.arcsin(arg) / sqrt_K
            return np.pi / (2 * sqrt_K)
        else:
            # Negative curvature - geodesic longer than Euclidean
            sqrt_neg_K = np.sqrt(-K)
            return np.arcsinh(sqrt_neg_K * euclidean_radius) / sqrt_neg_K

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
        diff = volume_a.centroid - volume_b.centroid
        cov_avg = (volume_a.covariance + volume_b.covariance) / 2

        try:
            cov_avg_inv = np.linalg.inv(cov_avg)
            term1 = 0.125 * (diff @ cov_avg_inv @ diff)

            sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
            term2 = 0.5 * (logdet_avg - 0.5 * (volume_a.log_det_covariance + volume_b.log_det_covariance))

            db = term1 + term2
            return np.exp(-db)

        except np.linalg.LinAlgError:
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
        # Monte Carlo estimation with samples from both distributions
        n_samples = 1000

        # Sample from volume_a
        samples_a = np.random.multivariate_normal(
            volume_a.centroid, volume_a.covariance, n_samples
        )

        # Sample from volume_b
        samples_b = np.random.multivariate_normal(
            volume_b.centroid, volume_b.covariance, n_samples
        )

        # Count samples from A that are in B's high-density region
        threshold = self.config.membership_threshold
        a_in_b = sum(volume_b.contains(s, threshold) for s in samples_a)
        b_in_a = sum(volume_a.contains(s, threshold) for s in samples_b)

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
        # Get principal directions (eigenvectors of covariance)
        _, Va = np.linalg.eigh(volume_a.covariance)
        _, Vb = np.linalg.eigh(volume_b.covariance)

        # Compute singular values of Va^T @ Vb
        # These are cosines of principal angles
        M = Va.T @ Vb
        singular_values = np.linalg.svd(M, compute_uv=False)

        # Average of squared cosines (like CKA)
        alignment = np.mean(singular_values ** 2)

        return alignment


def batch_estimate_volumes(
    estimator: RiemannianDensityEstimator,
    concept_activations: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], np.ndarray] | None = None,
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
        for id_b in concept_ids[i + 1:]:
            relation = estimator.compute_relation(volumes[id_a], volumes[id_b])
            relations[(id_a, id_b)] = relation

    return relations
