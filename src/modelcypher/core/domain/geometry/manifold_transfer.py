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
    Chord distance is incorrect in high-dimensional curved manifolds:
    - Positive curvature: chord underestimates true distance
    - Negative curvature: chord overestimates true distance

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
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

from .manifold_curvature import (
    LocalCurvature,
    ManifoldCurvatureProfile,
    SectionalCurvatureEstimator,
)
from .riemannian_density import (
    ConceptVolume,
    RiemannianDensityEstimator,
)

logger = logging.getLogger(__name__)


# Minimum number of anchors for statistical triangulation
MIN_ANCHORS = 3


@dataclass
class AnchorDistanceProfile:
    """Distance profile of a concept relative to landmark anchors.

    Captures the geodesic distances from a concept's centroid to each
    anchor in a fixed set of landmarks. This profile serves as a
    coordinate-free representation that can be used to locate the
    concept in a different manifold via stress minimization.

    This is analogous to the "landmark coordinates" in Landmark MDS
    (de Silva & Tenenbaum, 2004), but uses geodesic rather than
    Chord distances.

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
    distances: "Array"
    weights: "Array"
    source_curvature: LocalCurvature | None
    source_volume: ConceptVolume | None

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_ids)

    @property
    def mean_distance(self) -> float:
        """Weighted mean distance to anchors."""
        backend = get_default_backend()
        # Weighted average: sum(w * d) / sum(w)
        weighted_sum = backend.sum(self.distances * self.weights)
        weight_sum = backend.sum(self.weights)
        backend.eval(weighted_sum, weight_sum)
        weighted_sum_val = float(backend.to_scalar(weighted_sum))
        weight_sum_val = float(backend.to_scalar(weight_sum))
        return weighted_sum_val / weight_sum_val

    @property
    def distance_variance(self) -> float:
        """Variance in anchor distances."""
        backend = get_default_backend()
        var_result = backend.var(self.distances)
        backend.eval(var_result)
        return float(backend.to_scalar(var_result))

    def distance_to(self, anchor_id: str) -> float | None:
        """Get distance to a specific anchor."""
        try:
            idx = self.anchor_ids.index(anchor_id)
            return float(self.distances[idx])
        except ValueError:
            return None


@dataclass
class TransferConfidenceComponents:
    """Raw component factors for transfer confidence.

    Returns individual measurements instead of a weighted composite.
    Consumers decide how to interpret these values.
    """

    stress_factor: float
    """Exponential decay of normalized stress [0, 1]. Higher = lower stress."""

    anchor_factor: float
    """Saturation curve for anchor count [0, 1]. Higher = more anchors."""

    curvature_factor: float
    """Exponential decay of curvature mismatch [0, 1]. Higher = less mismatch."""


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
        stress: Normalized stress of the projection (lower indicates less stress).
        anchor_stress: Per-anchor stress breakdown.
        curvature_mismatch: Difference in local curvature.
        confidence_components: Component factors for transfer confidence.
    """

    concept_id: str
    source_profile: AnchorDistanceProfile
    coordinates: "Array"
    projected_volume: ConceptVolume | None
    stress: float
    anchor_stress: dict[str, float] = field(default_factory=dict)
    curvature_mismatch: float = 0.0
    confidence_components: TransferConfidenceComponents = field(
        default_factory=lambda: TransferConfidenceComponents(
            stress_factor=0.0, anchor_factor=0.0, curvature_factor=0.0
        )
    )


    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        backend = get_default_backend()
        # Use tolist() for O(1) extraction instead of O(n) scalar extractions
        backend.eval(self.coordinates)
        coords_list = backend.tolist(self.coordinates)
        return {
            "conceptId": self.concept_id,
            "coordinates": [float(x) for x in coords_list],
            "stress": self.stress,
            "curvatureMismatch": self.curvature_mismatch,
            "stressFactor": self.confidence_components.stress_factor,
            "anchorFactor": self.confidence_components.anchor_factor,
            "curvatureFactor": self.confidence_components.curvature_factor,
            "numAnchors": self.source_profile.num_anchors,
            "meanSourceDistance": self.source_profile.mean_distance,
        }


@dataclass
class TransferReport:
    """Report on cross-manifold transfer for multiple concepts.

    Contains raw stress distribution metrics. Callers should interpret
    stress values relative to baselines for their specific use case.
    """

    transfers: list[TransferPoint]
    source_model_id: str
    target_model_id: str
    mean_stress: float
    max_stress: float
    min_stress: float
    median_stress: float
    std_stress: float
    source_mean_curvature: float | None
    target_mean_curvature: float | None

    @property
    def transfer_count(self) -> int:
        """Total number of transfer points."""
        return len(self.transfers)


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

    All numerical parameters (convergence tolerance, learning rate, etc.)
    are derived from data. No configuration needed.

    See: de Silva & Tenenbaum (2004) for the landmark MDS framework.
    """

    def __init__(self) -> None:
        self.density_estimator = RiemannianDensityEstimator()
        self.curvature_estimator = SectionalCurvatureEstimator()

    def compute_distance_profile(
        self,
        concept_activations: "Array",
        concept_id: str,
        anchor_activations: dict[str, "Array"],
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

        backend = get_default_backend()

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
                anchor_arr = backend.array(anchor_acts)
                centroid = backend.mean(anchor_arr, axis=0)
                backend.eval(centroid)
                anchor_centroids.append(centroid)

        if len(anchor_ids) < MIN_ANCHORS:
            logger.warning(
                f"Only {len(anchor_ids)} anchors available, "
                f"minimum {MIN_ANCHORS} required for triangulation"
            )

        # Build combined point matrix: [concept_centroid, anchor_0, anchor_1, ...]
        concept_arr = backend.array(concept_centroid)
        concept_reshaped = backend.reshape(concept_arr, (1, -1))
        anchor_reshaped = [backend.reshape(a, (1, -1)) for a in anchor_centroids]
        all_points = backend.concatenate([concept_reshaped] + anchor_reshaped, axis=0)

        # Compute geodesic distances via k-NN graph
        points_arr = backend.astype(all_points, "float32")
        n_points = len(anchor_ids) + 1
        k_neighbors = min(max(3, n_points // 3), n_points - 1)

        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
        backend.eval(geo_dist)
        row0 = backend.take(geo_dist, backend.array([0]), axis=0)
        row0 = backend.squeeze(row0, axis=0)
        anchor_indices = backend.arange(1, len(anchor_ids) + 1)
        distances = backend.take(row0, anchor_indices, axis=0)

        # Derive distance weight decay from data (prevents division by zero)
        dist_weight_decay = division_epsilon(backend, distances)

        # Weight by inverse distance (closer anchors more important)
        weights = 1.0 / (distances + dist_weight_decay)
        weight_sum = backend.sum(weights)
        backend.eval(weight_sum)
        weights = weights / weight_sum  # Normalize

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
        target_anchor_activations: dict[str, "Array"],
        target_manifold_profile: ManifoldCurvatureProfile | None = None,
        initial_position: "Array | None" = None,
        use_curvature_correction: bool = True,
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
            use_curvature_correction: Whether to apply curvature correction to projected volumes.

        Returns:
            TransferPoint with computed position and quality metrics.
        """
        from .riemannian_utils import geodesic_distance_matrix

        backend = get_default_backend()

        # Get target anchor centroids for matching anchors
        matching_anchor_ids = []
        target_centroids = []
        source_distances_list = []
        weights_list = []

        distances_seq = profile.distances
        weights_seq = profile.weights
        if hasattr(distances_seq, "shape"):
            distances_seq = backend.tolist(distances_seq)
        if hasattr(weights_seq, "shape"):
            weights_seq = backend.tolist(weights_seq)

        for i, anchor_id in enumerate(profile.anchor_ids):
            if anchor_id in target_anchor_activations:
                target_acts = target_anchor_activations[anchor_id]
                if len(target_acts) > 0:
                    matching_anchor_ids.append(anchor_id)
                    target_arr = backend.array(target_acts)
                    centroid = backend.mean(target_arr, axis=0)
                    backend.eval(centroid)
                    target_centroids.append(centroid)
                    # Extract scalar from backend array
                    dist_val = distances_seq[i]
                    weight_val = weights_seq[i]
                    source_distances_list.append(float(dist_val))
                    weights_list.append(float(weight_val))

        if len(matching_anchor_ids) < MIN_ANCHORS:
            logger.warning(
                f"Only {len(matching_anchor_ids)} matching anchors, projection may be unreliable"
            )

        # Stack target centroids
        target_centroids_reshaped = [backend.reshape(c, (1, -1)) for c in target_centroids]
        target_centroids_arr = backend.concatenate(target_centroids_reshaped, axis=0)
        source_distances = backend.array(source_distances_list)
        weights = backend.array(weights_list)
        weight_sum = backend.sum(weights)
        backend.eval(weight_sum)
        weights = weights / weight_sum

        backend.eval(target_centroids_arr)
        d = int(target_centroids_arr.shape[1])
        eps = division_epsilon(backend, target_centroids_arr)
        # Convergence tolerance derived from data
        convergence_tolerance = regularization_epsilon(backend, target_centroids_arr)
        n_anchors = len(matching_anchor_ids)
        k_neighbors = min(max(3, n_anchors // 3), n_anchors)

        # Initialize position
        if initial_position is not None:
            position = backend.array(initial_position)
        else:
            # Use weighted centroid of anchors as initial guess
            # Weighted average: sum(w_i * x_i) / sum(w_i)
            weights_expanded = backend.reshape(weights, (-1, 1))
            weighted_centroids = target_centroids_arr * weights_expanded
            position = backend.sum(weighted_centroids, axis=0)
            backend.eval(position)

        # Gradient descent to minimize stress
        best_position = position
        best_stress = float("inf")

        # Learning rate derived from anchor count (adaptive scaling)
        learning_rate = 1.0 / max(1, n_anchors)

        # Max iterations derived from anchor count and logarithmic scaling
        # Base is 10 * ceil(log2(n_anchors + 1)), scaled with problem size
        log_base = 10 * int(math.ceil(math.log2(max(2, n_anchors) + 1)))
        max_iterations = max(log_base, 5 * n_anchors)

        for iteration in range(max_iterations):
            # Build point matrix: [position, anchor_0, anchor_1, ...]
            position_reshaped = backend.reshape(position, (1, -1))
            all_points = backend.concatenate([position_reshaped, target_centroids_arr], axis=0)
            points_arr = backend.astype(all_points, "float32")

            # Compute geodesic distances
            geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
            backend.eval(geo_dist)

            # Extract distances from position (row 0) to each anchor
            row0 = backend.take(geo_dist, backend.array([0]), axis=0)
            row0 = backend.squeeze(row0, axis=0)
            anchor_indices = backend.arange(1, n_anchors + 1)
            current_distances = backend.take(row0, anchor_indices, axis=0)

            # Compute stress
            residuals = current_distances - source_distances
            stress_arr = backend.sum(weights * residuals * residuals)
            backend.eval(stress_arr)
            stress = float(backend.to_scalar(stress_arr))

            if stress < best_stress:
                best_stress = stress
                best_position = position

            # Check convergence
            if stress < convergence_tolerance:
                break

            # Compute gradient in tangent space (local linear approximation)
            diffs = position - target_centroids_arr
            diff_norms = geodesic_norms(diffs, backend)
            eps_vec = backend.full(diff_norms.shape, eps)
            safe_norms = backend.maximum(diff_norms, eps_vec)
            valid_mask = backend.astype(current_distances > eps, "float32") * backend.astype(
                diff_norms > eps, "float32"
            )
            coeffs = (2.0 * weights * residuals) / safe_norms
            coeffs = coeffs * valid_mask
            gradient = backend.sum(diffs * backend.reshape(coeffs, (-1, 1)), axis=0)

            # Update position
            position = position - learning_rate * gradient
            backend.eval(position)

        # Compute final geodesic distances for minimum-stress position
        best_position_reshaped = backend.reshape(best_position, (1, -1))
        all_points = backend.concatenate([best_position_reshaped, target_centroids_arr], axis=0)
        points_arr = backend.astype(all_points, "float32")
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=backend)
        backend.eval(geo_dist)
        row0 = backend.take(geo_dist, backend.array([0]), axis=0)
        row0 = backend.squeeze(row0, axis=0)
        anchor_indices = backend.arange(1, n_anchors + 1)
        final_distances = backend.take(row0, anchor_indices, axis=0)

        final_list = backend.tolist(final_distances)
        source_list = backend.tolist(source_distances)
        anchor_stress = {
            anchor_id: float((float(final_list[i]) - float(source_list[i])) ** 2)
            for i, anchor_id in enumerate(matching_anchor_ids)
        }

        # Normalize stress
        src_dist_sq_sum = backend.sum(source_distances * source_distances)
        backend.eval(src_dist_sq_sum)
        stress_eps = division_epsilon(backend, source_distances)
        src_dist_sq_sum_val = float(backend.to_scalar(src_dist_sq_sum))
        normalized_stress = best_stress / (src_dist_sq_sum_val + stress_eps)

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
        if profile.source_volume is not None and use_curvature_correction:
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

        confidence_components = self._compute_confidence_components(
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
            anchor_stress=anchor_stress,
            curvature_mismatch=curvature_mismatch,
            confidence_components=confidence_components,
        )

    def transfer_batch(
        self,
        profiles: list[AnchorDistanceProfile],
        target_anchor_activations: dict[str, "Array"],
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
        backend = get_default_backend()
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

        # Compute stress distribution statistics (raw measurements)
        stresses = [t.stress for t in transfers]

        source_curvatures = [
            p.source_curvature.mean_sectional for p in profiles if p.source_curvature is not None
        ]
        if source_curvatures:
            source_curvatures_arr = backend.array(source_curvatures)
            source_mean_arr = backend.mean(source_curvatures_arr)
            backend.eval(source_mean_arr)
            source_mean_curvature = float(backend.to_scalar(source_mean_arr))
        else:
            source_mean_curvature = None
        target_mean_curvature = (
            target_manifold_profile.global_mean if target_manifold_profile else None
        )

        if stresses:
            stresses_arr = backend.array(stresses)
            mean_stress_arr = backend.mean(stresses_arr)
            max_stress_arr = backend.max(stresses_arr)
            min_stress_arr = backend.min(stresses_arr)
            std_stress_arr = backend.std(stresses_arr)
            backend.eval(mean_stress_arr, max_stress_arr, min_stress_arr, std_stress_arr)
            mean_stress = float(backend.to_scalar(mean_stress_arr))
            max_stress = float(backend.to_scalar(max_stress_arr))
            min_stress = float(backend.to_scalar(min_stress_arr))
            std_stress = float(backend.to_scalar(std_stress_arr))
            # Compute median via sorting
            sorted_stresses = sorted(stresses)
            n = len(sorted_stresses)
            if n % 2 == 1:
                median_stress = sorted_stresses[n // 2]
            else:
                median_stress = (sorted_stresses[n // 2 - 1] + sorted_stresses[n // 2]) / 2.0
        else:
            mean_stress = 0.0
            max_stress = 0.0
            min_stress = 0.0
            median_stress = 0.0
            std_stress = 0.0

        return TransferReport(
            transfers=transfers,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            mean_stress=mean_stress,
            max_stress=max_stress,
            min_stress=min_stress,
            median_stress=median_stress,
            std_stress=std_stress,
            source_mean_curvature=source_mean_curvature,
            target_mean_curvature=target_mean_curvature,
        )

    def _project_volume(
        self,
        source_volume: ConceptVolume,
        target_position: "Array",
        source_curvature: LocalCurvature | None,
        target_curvature: LocalCurvature | None,
    ) -> ConceptVolume:
        """Project ConceptVolume with curvature correction.

        Adjusts covariance based on curvature difference between manifolds.
        In flatter regions, volumes expand; in more curved regions, they contract.
        """
        backend = get_default_backend()

        # Copy covariance using backend
        projected_covariance = backend.array(source_volume.covariance)
        projected_radius = source_volume.geodesic_radius
        eps = division_epsilon(backend, projected_covariance)

        if source_curvature is not None and target_curvature is not None:
            K_source = source_curvature.mean_sectional
            K_target = target_curvature.mean_sectional

            if abs(K_source) > eps and abs(K_target) > eps:
                ratio = (1 - K_target / 6) / (1 - K_source / 6)
                ratio = max(0.5, min(2.0, ratio))  # Clip to [0.5, 2.0]
                projected_covariance = projected_covariance * ratio
                projected_radius = projected_radius * sqrt_scalar(ratio, backend)
            elif abs(K_source) > eps:
                expansion = 1 + abs(K_source) * source_volume.geodesic_radius**2 / 6
                projected_covariance = projected_covariance * expansion
                projected_radius = projected_radius * sqrt_scalar(expansion, backend)
            elif abs(K_target) > eps:
                contraction = 1 / (1 + abs(K_target) * source_volume.geodesic_radius**2 / 6)
                projected_covariance = projected_covariance * contraction
                projected_radius = projected_radius * sqrt_scalar(contraction, backend)

        return ConceptVolume(
            concept_id=source_volume.concept_id + "_transferred",
            centroid=target_position,
            covariance=projected_covariance,
            geodesic_radius=projected_radius,
            local_curvature=target_curvature,
            num_samples=source_volume.num_samples,
            influence_type=source_volume.influence_type,
            student_t_df=source_volume.student_t_df,
        )

    def _compute_confidence_components(
        self,
        normalized_stress: float,
        num_anchors: int,
        curvature_mismatch: float,
    ) -> TransferConfidenceComponents:
        """Compute raw confidence components for projection.

        Returns individual factors instead of a weighted composite.
        """
        _b = get_default_backend()

        stress_factor = exp_scalar(-normalized_stress * 3, _b)
        anchor_factor = 1 - exp_scalar(-num_anchors / 20, _b)
        curvature_factor = exp_scalar(-curvature_mismatch * 2, _b)
        return TransferConfidenceComponents(
            stress_factor=stress_factor,
            anchor_factor=anchor_factor,
            curvature_factor=curvature_factor,
        )


def project_concept(
    concept_activations: "Array",
    concept_id: str,
    source_anchor_activations: dict[str, "Array"],
    target_anchor_activations: dict[str, "Array"],
) -> TransferPoint:
    """Convenience function for single concept projection.

    All numerical parameters (convergence tolerance, learning rate, etc.)
    are derived from data. No configuration needed.

    Args:
        concept_activations: Activations for the concept to transfer.
        concept_id: Identifier for the concept.
        source_anchor_activations: Source model anchor activations.
        target_anchor_activations: Target model anchor activations.

    Returns:
        TransferPoint with computed position.
    """
    projector = CrossManifoldProjector()

    profile = projector.compute_distance_profile(
        concept_activations,
        concept_id,
        source_anchor_activations,
    )

    return projector.project(
        profile,
        target_anchor_activations,
    )
