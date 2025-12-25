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

"""Cross-manifold projection via landmark MDS with geodesic distances.

Computes distance-preserving projections between representation manifolds
using anchor points as landmarks. This enables transferring concepts that
exist in one model's representation space to another model where they
may not have a direct counterpart.

Mathematical Framework:
    Given a concept X in source manifold M_s with anchor set A = {P_1, ..., P_n}:
    1. Compute anchor distance profile: d_s = [d(X, P_1), ..., d(X, P_n)]
    2. Find point X' in target manifold M_t minimizing stress:
       σ(X') = Σᵢ wᵢ |d_t(X', P_i) - d_s(X, P_i)|²

    Distances are computed as geodesics on the k-NN graph of the point cloud.
    Euclidean distance is incorrect in high-dimensional curved manifolds:
    - Positive curvature: Euclidean underestimates true distance
    - Negative curvature: Euclidean overestimates true distance

    The k-NN graph represents the discrete manifold structure. Geodesic
    distance = shortest path on this graph. This is exact for the
    discrete manifold (not an approximation).

References:
    - de Silva, V., & Tenenbaum, J. B. (2004). Sparse multidimensional scaling
      using landmark points. Stanford University Technical Report.
    - Cox, T. F., & Cox, M. A. A. (2000). Multidimensional Scaling (2nd ed.).
      Chapman and Hall/CRC. Chapter 4: Classical MDS.
    - Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global
      geometric framework for nonlinear dimensionality reduction. Science,
      290(5500), 2319-2323.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .manifold_curvature import (
    LocalCurvature,
    ManifoldCurvatureProfile,
    SectionalCurvatureEstimator,
)
from .riemannian_density import (
    ConceptVolume,
    RiemannianDensityEstimator,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ProjectionQuality(str, Enum):
    """Quality assessment of cross-manifold projection based on stress."""

    EXCELLENT = "excellent"  # Normalized stress < 0.1
    GOOD = "good"  # Normalized stress < 0.3
    MARGINAL = "marginal"  # Normalized stress < 0.5
    ACCEPTABLE = "acceptable"  # Alias for MARGINAL (backward compat)
    POOR = "poor"  # Normalized stress >= 0.5


# Backward-compatible alias for tests
TransferQuality = ProjectionQuality


@dataclass(frozen=True)
class CrossManifoldConfig:
    """Configuration for cross-manifold projection.

    Attributes:
        max_iterations: Maximum iterations for stress minimization.
        convergence_tolerance: Stop when stress change < this value.
        learning_rate: Step size for gradient descent.
        min_anchors: Minimum anchors required for reliable projection.
        distance_weight_decay: Controls anchor weighting by distance.
        use_curvature_correction: Whether to apply curvature-aware adjustments.
    """

    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    learning_rate: float = 0.01
    min_anchors: int = 10
    distance_weight_decay: float = 0.1
    use_curvature_correction: bool = True
    stress_regularization: float = 1e-8


@dataclass
class AnchorDistanceProfile:
    """Distance profile of a concept relative to landmark anchors.

    Captures the geodesic distances from a concept's centroid to each
    anchor in a fixed set of landmarks. This profile serves as a
    coordinate-free representation that can be used to locate the
    concept in a different manifold via stress minimization.

    This is analogous to the "landmark coordinates" in Landmark MDS
    (de Silva & Tenenbaum, 2004), but uses geodesic rather than
    Euclidean distances.

    Attributes:
        concept_id: Identifier for the concept.
        anchor_ids: Ordered list of anchor identifiers.
        distances: Geodesic distances to each anchor (n_anchors,).
        weights: Importance weights for each anchor (n_anchors,).
        source_curvature: Local curvature at concept position.
        source_volume: ConceptVolume if available.
    """

    concept_id: str
    anchor_ids: list[str]
    distances: np.ndarray
    weights: np.ndarray
    source_curvature: LocalCurvature | None
    source_volume: ConceptVolume | None

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_ids)

    @property
    def mean_distance(self) -> float:
        """Weighted mean distance to anchors."""
        return float(np.average(self.distances, weights=self.weights))

    @property
    def distance_variance(self) -> float:
        """Variance in anchor distances."""
        return float(np.var(self.distances))

    def distance_to(self, anchor_id: str) -> float | None:
        """Get distance to a specific anchor."""
        try:
            idx = self.anchor_ids.index(anchor_id)
            return float(self.distances[idx])
        except ValueError:
            return None


@dataclass
class TransferPoint:
    """A point computed via cross-manifold projection.

    Represents the result of projecting a concept from source manifold
    to target manifold by minimizing the stress of distance preservation
    to shared anchor points.

    The projection quality is measured by normalized stress:
        σ_norm = Σᵢ wᵢ (d_i - d̂_i)² / Σᵢ wᵢ d_i²

    where d_i are source distances and d̂_i are achieved target distances.

    Attributes:
        concept_id: Identifier matching the source concept.
        source_profile: The anchor distance profile from source.
        coordinates: Computed position in target space.
        projected_volume: ConceptVolume in target space (if computed).
        stress: Normalized stress of the projection.
        quality: Quality assessment based on stress.
        anchor_stress: Per-anchor stress breakdown.
        curvature_mismatch: Difference in local curvature.
        confidence: Overall confidence in the projection (0-1).
    """

    concept_id: str
    source_profile: AnchorDistanceProfile
    coordinates: np.ndarray
    projected_volume: ConceptVolume | None
    stress: float
    quality: ProjectionQuality
    anchor_stress: dict[str, float] = field(default_factory=dict)
    curvature_mismatch: float = 0.0
    confidence: float = 0.0

    @property
    def is_reliable(self) -> bool:
        """Check if projection is reliable enough for downstream use."""
        return self.quality in (ProjectionQuality.EXCELLENT, ProjectionQuality.GOOD)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "conceptId": self.concept_id,
            "coordinates": self.coordinates.tolist(),
            "stress": self.stress,
            "quality": self.quality.value,
            "curvatureMismatch": self.curvature_mismatch,
            "confidence": self.confidence,
            "numAnchors": self.source_profile.num_anchors,
            "meanSourceDistance": self.source_profile.mean_distance,
        }


@dataclass
class TransferReport:
    """Report on cross-manifold transfer for multiple concepts."""

    transfers: list[TransferPoint]
    source_model_id: str
    target_model_id: str
    mean_stress: float
    max_stress: float
    num_reliable: int
    num_unreliable: int
    source_mean_curvature: float | None
    target_mean_curvature: float | None

    @property
    def success_rate(self) -> float:
        """Fraction of reliable transfers."""
        total = len(self.transfers)
        return self.num_reliable / total if total > 0 else 0.0

    def get_reliable_transfers(self) -> list[TransferPoint]:
        """Get only reliable transfer points."""
        return [t for t in self.transfers if t.is_reliable]


class CrossManifoldProjector:
    """Projects concepts between manifolds via landmark MDS.

    Implements cross-manifold projection by:
    1. Computing anchor distance profiles in source manifold
    2. Finding stress-minimizing positions in target manifold
    3. Optionally projecting ConceptVolumes with curvature correction

    The algorithm is a form of weighted MDS where:
    - Anchors serve as shared landmarks between manifolds
    - Distances are computed along geodesics (curvature-aware)
    - Stress is minimized via gradient descent

    See: de Silva & Tenenbaum (2004) for the landmark MDS framework.
    """

    def __init__(self, config: CrossManifoldConfig | None = None):
        self.config = config or CrossManifoldConfig()
        self.density_estimator = RiemannianDensityEstimator()
        self.curvature_estimator = SectionalCurvatureEstimator()

    def compute_distance_profile(
        self,
        concept_activations: np.ndarray,
        concept_id: str,
        anchor_activations: dict[str, np.ndarray],
        manifold_profile: ManifoldCurvatureProfile | None = None,
    ) -> AnchorDistanceProfile:
        """Compute anchor distance profile for a concept.

        Computes geodesic distances from the concept's centroid to all anchor
        centroids via k-NN graph shortest paths. Geodesic distance is the
        correct metric in curved high-dimensional spaces.

        Args:
            concept_activations: Activation samples for the concept (n x d).
            concept_id: Identifier for the concept.
            anchor_activations: Dict mapping anchor_id -> activations (n x d).
            manifold_profile: Pre-computed curvature profile (optional, for metadata).

        Returns:
            AnchorDistanceProfile with geodesic distances to all anchors.
        """
        from .riemannian_utils import geodesic_distance_matrix
        from modelcypher.core.domain._backend import get_default_backend

        # Estimate concept volume
        concept_volume = self.density_estimator.estimate_concept_volume(
            concept_id, concept_activations
        )
        concept_centroid = concept_volume.centroid
        local_curvature = concept_volume.local_curvature

        # Compute anchor centroids
        anchor_ids = []
        anchor_centroids = []

        for anchor_id, anchor_acts in anchor_activations.items():
            if len(anchor_acts) > 0:
                anchor_ids.append(anchor_id)
                anchor_centroids.append(np.mean(anchor_acts, axis=0))

        if len(anchor_ids) < self.config.min_anchors:
            logger.warning(
                f"Only {len(anchor_ids)} anchors available, "
                f"minimum {self.config.min_anchors} recommended"
            )

        # Build combined point matrix: [concept_centroid, anchor_0, anchor_1, ...]
        all_points = np.vstack([concept_centroid.reshape(1, -1)] + [a.reshape(1, -1) for a in anchor_centroids])

        # Compute geodesic distances via k-NN graph
        backend = get_default_backend()
        points_arr = backend.array(all_points.astype(np.float32))
        n_points = len(anchor_ids) + 1
        k_neighbors = min(max(3, n_points // 3), n_points - 1)

        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
        backend.eval(geo_dist)
        geo_dist_np = backend.to_numpy(geo_dist)

        # Extract distances from concept (row 0) to each anchor
        distances = np.array([float(geo_dist_np[0, i + 1]) for i in range(len(anchor_ids))])

        # Weight by inverse distance (closer anchors more important)
        weights = 1.0 / (distances + self.config.distance_weight_decay)
        weights = weights / np.sum(weights)  # Normalize

        return AnchorDistanceProfile(
            concept_id=concept_id,
            anchor_ids=anchor_ids,
            distances=distances,
            weights=weights,
            source_curvature=local_curvature,
            source_volume=concept_volume,
        )

    def project(
        self,
        profile: AnchorDistanceProfile,
        target_anchor_activations: dict[str, np.ndarray],
        target_manifold_profile: ManifoldCurvatureProfile | None = None,
        initial_position: np.ndarray | None = None,
    ) -> TransferPoint:
        """Project a concept to target manifold via stress minimization.

        Finds position X' in target manifold minimizing:
            σ(X') = Σᵢ wᵢ |d_target(X', Pᵢ) - d_source(X, Pᵢ)|²

        Uses geodesic distances computed via k-NN graph shortest paths.
        Gradient descent operates in tangent space (local linear approximation).

        Args:
            profile: Anchor distance profile from source manifold.
            target_anchor_activations: Target model anchor activations.
            target_manifold_profile: Curvature profile of target (optional, for metadata).
            initial_position: Starting point for optimization (optional).

        Returns:
            TransferPoint with computed position and quality metrics.
        """
        from .riemannian_utils import geodesic_distance_matrix
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

        # Get target anchor centroids for matching anchors
        matching_anchor_ids = []
        target_centroids = []
        source_distances = []
        weights = []

        for i, anchor_id in enumerate(profile.anchor_ids):
            if anchor_id in target_anchor_activations:
                target_acts = target_anchor_activations[anchor_id]
                if len(target_acts) > 0:
                    matching_anchor_ids.append(anchor_id)
                    target_centroids.append(np.mean(target_acts, axis=0))
                    source_distances.append(profile.distances[i])
                    weights.append(profile.weights[i])

        if len(matching_anchor_ids) < self.config.min_anchors:
            logger.warning(
                f"Only {len(matching_anchor_ids)} matching anchors, projection may be unreliable"
            )

        target_centroids_arr = np.array(target_centroids)
        source_distances = np.array(source_distances)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        d = target_centroids_arr.shape[1]
        n_anchors = len(matching_anchor_ids)
        k_neighbors = min(max(3, n_anchors // 3), n_anchors)

        # Initialize position
        if initial_position is not None:
            position = initial_position.copy()
        else:
            # Use weighted centroid of anchors as initial guess
            position = np.average(target_centroids_arr, axis=0, weights=weights)

        # Gradient descent to minimize stress
        best_position = position.copy()
        best_stress = float("inf")

        for iteration in range(self.config.max_iterations):
            # Build point matrix: [position, anchor_0, anchor_1, ...]
            all_points = np.vstack([position.reshape(1, -1), target_centroids_arr])
            points_arr = backend.array(all_points.astype(np.float32))

            # Compute geodesic distances
            geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
            backend.eval(geo_dist)
            geo_dist_np = backend.to_numpy(geo_dist)

            # Extract distances from position (row 0) to each anchor
            current_distances = np.array([float(geo_dist_np[0, i + 1]) for i in range(n_anchors)])

            # Compute stress
            residuals = current_distances - source_distances
            stress = np.sum(weights * residuals**2)

            if stress < best_stress:
                best_stress = stress
                best_position = position.copy()

            # Check convergence
            if stress < self.config.convergence_tolerance:
                break

            # Compute gradient in tangent space (local linear approximation)
            gradient = np.zeros(d)
            for i in range(n_anchors):
                diff = position - target_centroids_arr[i]
                dist = current_distances[i]
                if dist > 1e-10:
                    # Tangent-space gradient direction
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 1e-10:
                        gradient += 2 * weights[i] * residuals[i] * diff / diff_norm

            # Update position
            position = position - self.config.learning_rate * gradient

        # Compute final geodesic distances for best position
        all_points = np.vstack([best_position.reshape(1, -1), target_centroids_arr])
        points_arr = backend.array(all_points.astype(np.float32))
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
        backend.eval(geo_dist)
        geo_dist_np = backend.to_numpy(geo_dist)
        final_distances = np.array([float(geo_dist_np[0, i + 1]) for i in range(n_anchors)])

        anchor_stress = {
            anchor_id: float((final_distances[i] - source_distances[i]) ** 2)
            for i, anchor_id in enumerate(matching_anchor_ids)
        }

        # Normalize stress
        normalized_stress = best_stress / (np.sum(source_distances**2) + 1e-10)
        quality = self._assess_quality(normalized_stress)

        # Compute curvature mismatch
        curvature_mismatch = 0.0
        if profile.source_curvature is not None and target_manifold_profile is not None:
            target_curvature = target_manifold_profile.curvature_at_point(best_position)
            if target_curvature is not None:
                curvature_mismatch = abs(
                    profile.source_curvature.mean_sectional - target_curvature.mean_sectional
                )

        # Project volume if available
        projected_volume = None
        if profile.source_volume is not None and self.config.use_curvature_correction:
            target_curvature = (
                target_manifold_profile.curvature_at_point(best_position)
                if target_manifold_profile
                else None
            )
            projected_volume = self._project_volume(
                profile.source_volume,
                best_position,
                profile.source_curvature,
                target_curvature,
            )

        confidence = self._compute_confidence(
            normalized_stress,
            len(matching_anchor_ids),
            curvature_mismatch,
        )

        return TransferPoint(
            concept_id=profile.concept_id,
            source_profile=profile,
            coordinates=best_position,
            projected_volume=projected_volume,
            stress=normalized_stress,
            quality=quality,
            anchor_stress=anchor_stress,
            curvature_mismatch=curvature_mismatch,
            confidence=confidence,
        )

    def transfer_batch(
        self,
        profiles: list[AnchorDistanceProfile],
        target_anchor_activations: dict[str, np.ndarray],
        target_manifold_profile: ManifoldCurvatureProfile | None = None,
        source_model_id: str = "source",
        target_model_id: str = "target",
    ) -> TransferReport:
        """Transfer multiple concepts to target manifold.

        Args:
            profiles: List of anchor distance profiles to transfer.
            target_anchor_activations: Target model anchor activations.
            target_manifold_profile: Curvature profile of target.
            source_model_id: Identifier for source model.
            target_model_id: Identifier for target model.

        Returns:
            TransferReport with all transfer points and statistics.
        """
        transfers = []

        for profile in profiles:
            try:
                transfer = self.project(
                    profile,
                    target_anchor_activations,
                    target_manifold_profile,
                )
                transfers.append(transfer)
            except Exception as e:
                logger.warning(f"Failed to transfer {profile.concept_id}: {e}")

        # Compute statistics
        stresses = [t.stress for t in transfers]
        num_reliable = sum(1 for t in transfers if t.is_reliable)

        source_curvatures = [
            p.source_curvature.mean_sectional for p in profiles if p.source_curvature is not None
        ]
        source_mean_curvature = float(np.mean(source_curvatures)) if source_curvatures else None
        target_mean_curvature = (
            target_manifold_profile.global_mean if target_manifold_profile else None
        )

        return TransferReport(
            transfers=transfers,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            mean_stress=float(np.mean(stresses)) if stresses else 0.0,
            max_stress=float(np.max(stresses)) if stresses else 0.0,
            num_reliable=num_reliable,
            num_unreliable=len(transfers) - num_reliable,
            source_mean_curvature=source_mean_curvature,
            target_mean_curvature=target_mean_curvature,
        )

    def _project_volume(
        self,
        source_volume: ConceptVolume,
        target_position: np.ndarray,
        source_curvature: LocalCurvature | None,
        target_curvature: LocalCurvature | None,
    ) -> ConceptVolume:
        """Project ConceptVolume with curvature correction.

        Adjusts covariance based on curvature difference between manifolds.
        In flatter regions, volumes expand; in more curved regions, they contract.
        """
        projected_covariance = source_volume.covariance.copy()
        projected_radius = source_volume.geodesic_radius

        if source_curvature is not None and target_curvature is not None:
            K_source = source_curvature.mean_sectional
            K_target = target_curvature.mean_sectional

            if abs(K_source) > 1e-10 and abs(K_target) > 1e-10:
                ratio = (1 - K_target / 6) / (1 - K_source / 6)
                ratio = np.clip(ratio, 0.5, 2.0)
                projected_covariance = projected_covariance * ratio
                projected_radius = projected_radius * np.sqrt(ratio)
            elif abs(K_source) > 1e-10:
                expansion = 1 + abs(K_source) * source_volume.geodesic_radius**2 / 6
                projected_covariance = projected_covariance * expansion
                projected_radius = projected_radius * np.sqrt(expansion)
            elif abs(K_target) > 1e-10:
                contraction = 1 / (1 + abs(K_target) * source_volume.geodesic_radius**2 / 6)
                projected_covariance = projected_covariance * contraction
                projected_radius = projected_radius * np.sqrt(contraction)

        return ConceptVolume(
            concept_id=source_volume.concept_id + "_transferred",
            centroid=target_position,
            covariance=projected_covariance,
            geodesic_radius=projected_radius,
            local_curvature=target_curvature,
            num_samples=source_volume.num_samples,
            influence_type=source_volume.influence_type,
        )

    def _assess_quality(self, normalized_stress: float) -> ProjectionQuality:
        """Assess projection quality based on normalized stress."""
        if normalized_stress < 0.1:
            return ProjectionQuality.EXCELLENT
        elif normalized_stress < 0.3:
            return ProjectionQuality.GOOD
        elif normalized_stress < 0.5:
            return ProjectionQuality.MARGINAL
        else:
            return ProjectionQuality.POOR

    def _compute_confidence(
        self,
        normalized_stress: float,
        num_anchors: int,
        curvature_mismatch: float,
    ) -> float:
        """Compute confidence score for projection."""
        stress_factor = np.exp(-normalized_stress * 3)
        anchor_factor = 1 - np.exp(-num_anchors / 20)
        curvature_factor = np.exp(-curvature_mismatch * 2)
        confidence = 0.5 * stress_factor + 0.3 * anchor_factor + 0.2 * curvature_factor
        return float(np.clip(confidence, 0.0, 1.0))


def project_concept(
    concept_activations: np.ndarray,
    concept_id: str,
    source_anchor_activations: dict[str, np.ndarray],
    target_anchor_activations: dict[str, np.ndarray],
    config: CrossManifoldConfig | None = None,
) -> TransferPoint:
    """Convenience function for single concept projection.

    Args:
        concept_activations: Activations for the concept to transfer.
        concept_id: Identifier for the concept.
        source_anchor_activations: Source model anchor activations.
        target_anchor_activations: Target model anchor activations.
        config: Optional configuration.

    Returns:
        TransferPoint with computed position.
    """
    projector = CrossManifoldProjector(config)

    profile = projector.compute_distance_profile(
        concept_activations,
        concept_id,
        source_anchor_activations,
    )

    return projector.project(
        profile,
        target_anchor_activations,
    )
