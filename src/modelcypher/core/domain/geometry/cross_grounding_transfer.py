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

"""Cross-Grounding Transfer: Density Re-mapping for Coordinate-Invariant Knowledge Transfer.

Transfers knowledge between models with different coordinate systems by preserving
relational stress patterns rather than absolute coordinates.

Notes
-----
Relational stress is the pattern of distances to universal anchors. This pattern
is coordinate-invariant and survives rotation between different model geometries.

The algorithm finds positions in the target model that preserve the source model's
distance relationships to anchor concepts.

R(c) = [d(c, a₁), d(c, a₂), ..., d(c, aₙ)]  (Relational Stress Profile)
θ = arccos(alignment(source_axes, target_axes))  (Grounding Rotation)
argmin_p ||R_source(c) - R_target(p)||²  (Cross-Grounding Synthesis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    is_nan,
    machine_epsilon,
    pi_value,
    sqrt_scalar,
    ulp_scalar,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class RelationalStressProfile:
    """
    Coordinate-invariant fingerprint of a concept's position in the manifold.

    This is the "DNA" of a concept's location - it captures WHERE the concept
    sits relative to universal anchors, without depending on any specific
    coordinate system.

    The key insight: distances are rotation-invariant. If you know the distance
    from a point to 10 reference points, you can reconstruct its position
    (up to reflection) regardless of how the axes are oriented.
    """

    # Core invariants
    anchor_distances: dict[str, float]  # Distance to each universal anchor
    normalized_distances: dict[str, float]  # Distances normalized by anchor spread

    # Local geometry
    local_density: float  # Neighborhood crowding (inverse of mean neighbor distance)
    curvature_signature: tuple[float, ...]  # Eigenvalues of local Hessian approximation
    activation_magnitude: float  # L2 norm of the activation vector

    # Relational structure
    nearest_anchors: tuple[str, ...]  # Top-k nearest anchors (ordered)
    stress_vector: tuple[float, ...]  # Flattened distance vector for optimization

    def distance_to(self, other: "RelationalStressProfile") -> float:
        """Compute stress distance between two profiles."""
        backend = get_default_backend()
        if len(self.stress_vector) != len(other.stress_vector):
            # Different anchor sets - use common anchors
            common = set(self.anchor_distances.keys()) & set(other.anchor_distances.keys())
            self_dists = [self.anchor_distances[a] for a in sorted(common)]
            other_dists = [other.anchor_distances[a] for a in sorted(common)]
            return sqrt_scalar(sum((a - b) ** 2 for a, b in zip(self_dists, other_dists)), backend)
        diffs = [a - b for a, b in zip(self.stress_vector, other.stress_vector)]
        return sqrt_scalar(sum(d * d for d in diffs), backend)


@dataclass(frozen=True)
class GroundingRotation:
    """
    The estimated rotation between two models' coordinate systems.

    A rotation of 0° means the models have aligned axes (both High Visual Grounding).
    A rotation of 90° means orthogonal axes (one visual, one linguistic).

    This isn't a literal SO(n) rotation - it's a measure of how much
    the "principal axes" of spatial encoding differ between models.
    """

    angle_degrees: float  # Estimated rotation angle
    distance_correlation: float  # Correlation between geodesic distance matrices
    aligned: bool
    axis_correspondence: dict[str, str]  # source_axis -> target_axis mapping
    confidence: float  # How confident we are in the rotation estimate


@dataclass
class GhostAnchor:
    """
    A synthetic anchor for a concept that exists in Source but not in Target.

    The Ghost Anchor represents WHERE a concept WOULD live in the target model's
    latent space if it had been trained on the same data. We compute this by
    finding the position that preserves the concept's Relational Stress pattern.
    """

    concept_id: str
    source_position: "Array"  # Original position in source model
    target_position: "Array"  # Synthesized position in target model

    # Transfer quality
    stress_preservation: float  # How well the stress pattern was preserved (0-1)
    grounding_rotation: GroundingRotation  # The rotation applied
    common_anchor_count: int  # Number of shared anchors used
    source_anchor_count: int  # Total anchors in source
    target_anchor_count: int  # Total anchors in target

    # Relational structure
    source_stress: RelationalStressProfile
    target_stress: RelationalStressProfile

    # Confidence
    synthesis_confidence: float  # How confident we are in this Ghost Anchor


@dataclass
class CrossGroundingTransferResult:
    """Result of a cross-grounding knowledge transfer."""

    source_model_grounding: str  # "high_visual" | "moderate" | "alternative"
    target_model_grounding: str
    grounding_rotation: GroundingRotation

    # Transferred concepts
    ghost_anchors: list[GhostAnchor]

    # Quality metrics
    mean_stress_preservation: float
    min_stress_preservation: float

    # Transfer geometry
    interpretability_gap: float  # How much "rotation" was needed


# =============================================================================
# Core Computation
# =============================================================================


class RelationalStressComputer:
    """Computes coordinate-invariant Relational Stress Profiles."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_profile(
        self,
        concept_activation: "Array",
        anchor_activations: dict[str, "Array"],
        k_nearest: int | None = None,
    ) -> RelationalStressProfile:
        """
        Compute the Relational Stress Profile for a concept.

        Args:
            concept_activation: The activation vector for the concept
            anchor_activations: Dict mapping anchor names to their activations
            k_nearest: Number of nearest anchors to track (None = all anchors)

        Returns:
            Coordinate-invariant RelationalStressProfile
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        b = self._backend

        # Build combined matrix: [concept, anchor_0, anchor_1, ...]
        anchor_names = list(anchor_activations.keys())
        concept_2d = b.reshape(concept_activation, (1, -1))
        anchor_list = [b.reshape(anchor_activations[n], (1, -1)) for n in anchor_names]
        all_points = b.concatenate([concept_2d] + anchor_list, axis=0)
        points_arr = b.astype(all_points, "float32")
        b.eval(points_arr)

        # Compute geodesic distances (curvature-aware)
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=None, backend=b)
        b.eval(geo_dist)

        # Extract distances from concept (row 0) to each anchor
        row0 = b.take(geo_dist, b.array([0]), axis=0)
        row0 = b.squeeze(row0, axis=0)
        anchor_indices = b.arange(1, len(anchor_names) + 1)
        anchor_dists = b.take(row0, anchor_indices, axis=0)
        b.eval(anchor_dists)
        sorted_idx = b.argsort(anchor_dists)
        sorted_dists = b.take(anchor_dists, sorted_idx, axis=0)
        b.eval(sorted_idx, sorted_dists)
        sorted_idx_list = [int(x) for x in b.tolist(sorted_idx)]
        # Use tolist() for O(1) extraction instead of O(n) scalar extractions
        anchor_dists_list = b.tolist(anchor_dists)
        distances = {
            name: float(anchor_dists_list[i])
            for i, name in enumerate(anchor_names)
        }

        # Normalize distances by the spread of anchor positions
        anchor_matrix = b.concatenate(anchor_list, axis=0)
        anchor_std = b.std(anchor_matrix)
        b.eval(anchor_std)
        anchor_spread = float(b.to_scalar(anchor_std))
        if anchor_spread > 0:
            normalized = {k: v / anchor_spread for k, v in distances.items()}
        else:
            normalized = distances.copy()

        # Find k nearest anchors using backend sort order
        total_anchors = len(anchor_names)
        effective_k = total_anchors if k_nearest is None else min(k_nearest, total_anchors)
        nearest = tuple(anchor_names[i] for i in sorted_idx_list[:effective_k])

        # Compute local density (inverse of mean distance to k nearest)
        if effective_k > 0:
            prefix_idx = b.arange(effective_k)
            k_vals = b.take(sorted_dists, prefix_idx, axis=0)
            k_mean = b.mean(k_vals)
            b.eval(k_mean)
            eps = division_epsilon(b, anchor_dists)
            local_density = 1.0 / (float(b.to_scalar(k_mean)) + eps)
        else:
            local_density = 0.0

        # Compute curvature signature (eigenvalues of local covariance)
        curvature = self._estimate_local_curvature(concept_activation, anchor_activations)

        # Activation magnitude
        concept_norms = geodesic_norms(b.reshape(concept_activation, (1, -1)), b)
        b.eval(concept_norms)
        magnitude = float(b.to_scalar(concept_norms[0]))

        # Create stress vector (sorted for consistency)
        stress_vector = tuple(distances[k] for k in sorted(distances.keys()))

        return RelationalStressProfile(
            anchor_distances=distances,
            normalized_distances=normalized,
            local_density=float(local_density),
            curvature_signature=curvature,
            activation_magnitude=magnitude,
            nearest_anchors=nearest,
            stress_vector=stress_vector,
        )

    def _estimate_local_curvature(
        self,
        point: "Array",
        neighbors: dict[str, "Array"],
    ) -> tuple[float, ...]:
        """Estimate local manifold curvature using neighbor structure."""
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        b = self._backend

        if len(neighbors) < 3:
            return (0.0,)

        # Build point matrix for geodesic distance computation
        neighbor_names = list(neighbors.keys())
        point_2d = b.reshape(point, (1, -1))
        neighbor_list = [b.reshape(neighbors[n], (1, -1)) for n in neighbor_names]
        all_points = b.concatenate([point_2d] + neighbor_list, axis=0)
        points_arr = b.astype(all_points, "float32")
        b.eval(points_arr)

        # Compute geodesic distances
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=None, backend=b)
        b.eval(geo_dist)

        # Extract distances from point (row 0) to neighbors
        row0 = b.take(geo_dist, b.array([0]), axis=0)
        row0 = b.squeeze(row0, axis=0)
        neighbor_indices = b.arange(1, len(neighbor_names) + 1)
        neighbor_dists = b.take(row0, neighbor_indices, axis=0)
        b.eval(neighbor_dists)
        sorted_idx = b.argsort(neighbor_dists)
        b.eval(sorted_idx)
        sorted_idx_list = [int(x) for x in b.tolist(sorted_idx)]
        k = len(sorted_idx_list)

        # Build local covariance from neighbor directions
        directions = []
        eps = division_epsilon(b, point)
        for idx in sorted_idx_list:
            direction = neighbors[neighbor_names[idx]] - point
            norm_arr = geodesic_norms(b.reshape(direction, (1, -1)), b)
            b.eval(norm_arr)
            norm = float(b.to_scalar(norm_arr[0]))
            if norm > eps:
                normalized = direction / norm
                b.eval(normalized)
                directions.append(normalized)

        if len(directions) < 2:
            return (0.0,)

        # Stack directions and compute covariance
        dir_reshaped = [b.reshape(d, (1, -1)) for d in directions]
        directions_matrix = b.concatenate(dir_reshaped, axis=0)

        # Compute covariance: (X - mean)^T @ (X - mean) / (n-1)
        mean_dir = b.mean(directions_matrix, axis=0, keepdims=True)
        centered = directions_matrix - mean_dir
        cov = b.matmul(b.transpose(centered), centered) / (len(directions) - 1)
        b.eval(cov)

        # Eigenvalues as curvature signature
        try:
            eigenvalues, _ = b.eigh(cov)
            b.eval(eigenvalues)
            # Sort descending and keep top 3
            sorted_idx = b.argsort(-eigenvalues)
            top_idx = sorted_idx[:3]
            top_vals = b.take(eigenvalues, top_idx, axis=0)
            b.eval(top_vals)
            # Use tolist() for O(1) extraction
            eig_sorted = [float(x) for x in b.tolist(top_vals)]
            return tuple(eig_sorted)
        except Exception:
            return (0.0,)


class GroundingRotationEstimator:
    """Estimates the rotation between two models' grounding coordinate systems."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._stress_computer = RelationalStressComputer(backend)

    def estimate_rotation(
        self,
        source_anchors: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
    ) -> GroundingRotation:
        """
        Estimate the rotation between source and target coordinate systems.

        We do this by comparing how the SAME concepts are positioned relative
        to universal anchors in both models. If the relative distances are
        similar, the axes are aligned. If they're different, there's rotation.

        Uses geodesic distances - Euclidean is incorrect in curved manifolds.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        b = self._backend

        # Find common anchors
        common_anchors = set(source_anchors.keys()) & set(target_anchors.keys())
        if len(common_anchors) < 5:
            return GroundingRotation(
                angle_degrees=90.0,
                distance_correlation=0.0,
                aligned=False,
                axis_correspondence={},
                confidence=0.0,
            )

        # Build distance matrices for both models using geodesic distances
        common_list = sorted(common_anchors)
        n = len(common_list)

        # Build source position matrix
        source_list = [b.reshape(source_anchors[a], (1, -1)) for a in common_list]
        source_matrix = b.concatenate(source_list, axis=0)
        source_arr = b.astype(source_matrix, "float32")
        source_geo = geodesic_distance_matrix(source_arr, k_neighbors=None, backend=b)
        b.eval(source_geo)

        # Build target position matrix
        target_list = [b.reshape(target_anchors[a], (1, -1)) for a in common_list]
        target_matrix = b.concatenate(target_list, axis=0)
        target_arr = b.astype(target_matrix, "float32")
        target_geo = geodesic_distance_matrix(target_arr, k_neighbors=None, backend=b)
        b.eval(target_geo)

        # Normalize distance matrices
        source_max_arr = b.max(source_geo)
        target_max_arr = b.max(target_geo)
        b.eval(source_max_arr, target_max_arr)
        source_max = float(b.to_scalar(source_max_arr))
        target_max = float(b.to_scalar(target_max_arr))
        eps = division_epsilon(b, source_arr)
        source_dists = source_geo / (source_max + eps)
        target_dists = target_geo / (target_max + eps)

        # Compute alignment as correlation between distance matrices
        source_flat = b.reshape(source_dists, (-1,))
        target_flat = b.reshape(target_dists, (-1,))
        s_mean = b.mean(source_flat)
        t_mean = b.mean(target_flat)
        s_centered = source_flat - s_mean
        t_centered = target_flat - t_mean
        numerator_arr = b.sum(s_centered * t_centered)
        s_std_arr = b.sqrt(b.sum(s_centered * s_centered))
        t_std_arr = b.sqrt(b.sum(t_centered * t_centered))
        b.eval(numerator_arr, s_std_arr, t_std_arr)
        numerator_val = float(b.to_scalar(numerator_arr))
        s_std = float(b.to_scalar(s_std_arr))
        t_std = float(b.to_scalar(t_std_arr))
        if s_std > 0 and t_std > 0:
            correlation = numerator_val / (s_std * t_std)
        else:
            correlation = 0.0

        if is_nan(correlation, b):
            correlation = 0.0

        # Convert correlation to angle
        corr_clamped = max(-1.0, min(1.0, correlation))
        angle_degrees = acos_scalar(corr_clamped, b) * 180.0 / pi_value(b)
        eps = float(machine_epsilon(b, source_arr))
        aligned = abs(correlation - 1.0) <= eps

        # Estimate axis correspondence using Procrustes-like analysis
        axis_correspondence = self._estimate_axis_correspondence(
            source_anchors, target_anchors, common_list
        )

        # Confidence based on number of common anchors and variance
        diffs = b.abs(source_dists - target_dists)
        off_diag = b.ones((n, n)) - b.eye(n)
        diff_masked = diffs * off_diag
        count = max(1, n * (n - 1))
        mean_diff_arr = b.sum(diff_masked) / float(count)
        var_arr = b.sum((diff_masked - mean_diff_arr) ** 2 * off_diag) / float(count)
        b.eval(mean_diff_arr, var_arr)
        std_diff = sqrt_scalar(float(b.to_scalar(var_arr)), b)
        anchor_total = len(source_anchors) + len(target_anchors)
        overlap_fraction = (2.0 * len(common_anchors) / anchor_total) if anchor_total > 0 else 0.0
        confidence = max(0.0, min(1.0, overlap_fraction)) * (1.0 - std_diff)

        return GroundingRotation(
            angle_degrees=angle_degrees,
            distance_correlation=float(correlation),
            aligned=aligned,
            axis_correspondence=axis_correspondence,
            confidence=float(max(0.0, confidence)),
        )

    def _estimate_axis_correspondence(
        self,
        source_anchors: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
        common_anchors: list[str],
    ) -> dict[str, str]:
        """Estimate which target axis corresponds to which source axis."""
        b = self._backend

        # Build position matrices
        source_list = [b.reshape(source_anchors[a], (1, -1)) for a in common_anchors]
        source_positions = b.concatenate(source_list, axis=0)
        target_list = [b.reshape(target_anchors[a], (1, -1)) for a in common_anchors]
        target_positions = b.concatenate(target_list, axis=0)

        # Center the data
        source_mean = b.mean(source_positions, axis=0, keepdims=True)
        target_mean = b.mean(target_positions, axis=0, keepdims=True)
        source_centered = source_positions - source_mean
        target_centered = target_positions - target_mean

        # Compute principal components via SVD
        try:
            _, _, source_vh = b.svd(source_centered)
            _, _, target_vh = b.svd(target_centered)
            b.eval(source_vh, target_vh)

            # Match axes by correlation
            correspondence = {}
            n_axes = min(3, int(source_vh.shape[0]), int(target_vh.shape[0]))
            if n_axes <= 0:
                return correspondence

            source_axes = source_vh[:n_axes]
            target_axes = target_vh[:n_axes]
            corr = b.abs(b.matmul(source_axes, b.transpose(target_axes)))
            b.eval(corr)

            # Vectorized argmax: find highest-correlation match for each source axis
            best_matches = b.argmax(corr, axis=1)
            b.eval(best_matches)
            # Use tolist() for O(1) extraction instead of O(n) scalar extractions
            match_list = b.tolist(best_matches)
            for i in range(n_axes):
                correspondence[f"source_axis_{i}"] = f"target_axis_{int(match_list[i])}"

            return correspondence
        except Exception:
            return {}


class CrossGroundingSynthesizer:
    """
    Synthesizes Ghost Anchors for cross-grounding knowledge transfer.

    This is the core "3D Printer" - it takes a concept from a source model
    and finds/creates the equivalent position in a target model by preserving
    Relational Stress rather than absolute coordinates.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._stress_computer = RelationalStressComputer(backend)
        self._rotation_estimator = GroundingRotationEstimator(backend)

    def synthesize_ghost_anchor(
        self,
        concept_id: str,
        source_activation: "Array",
        source_anchors: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
        grounding_rotation: GroundingRotation | None = None,
    ) -> GhostAnchor:
        """
        Synthesize a Ghost Anchor for a concept in the target model's space.

        Args:
            concept_id: Identifier for the concept being transferred
            source_activation: The concept's activation in source model
            source_anchors: Universal anchors in source model
            target_anchors: Universal anchors in target model
            grounding_rotation: Pre-computed rotation (or None to compute)

        Returns:
            GhostAnchor with the synthesized position in target space
        """
        # Compute rotation if not provided
        if grounding_rotation is None:
            grounding_rotation = self._rotation_estimator.estimate_rotation(
                source_anchors, target_anchors
            )

        # Compute source stress profile
        source_stress = self._stress_computer.compute_profile(source_activation, source_anchors)

        source_pos = self._backend.array(source_activation)

        # Find common anchors
        source_anchor_count = len(source_anchors)
        target_anchor_count = len(target_anchors)
        common = set(source_stress.anchor_distances.keys()) & set(target_anchors.keys())
        common_anchor_count = len(common)
        if common_anchor_count < 3:
            return GhostAnchor(
                concept_id=concept_id,
                source_position=source_pos,
                target_position=source_pos,  # Fallback: use source position
                stress_preservation=0.0,
                grounding_rotation=grounding_rotation,
                common_anchor_count=common_anchor_count,
                source_anchor_count=source_anchor_count,
                target_anchor_count=target_anchor_count,
                source_stress=source_stress,
                target_stress=source_stress,
                synthesis_confidence=0.0,
            )

        # Solve for target position that preserves relational stress
        target_pos = self._solve_stress_preserving_position(
            source_stress,
            target_anchors,
            common,
        )

        # Compute target stress profile for the synthesized position
        # Create a temporary array for the target position
        target_activation = self._backend.array(target_pos)
        target_stress = self._stress_computer.compute_profile(target_activation, target_anchors)

        # Compute stress preservation score
        stress_preservation = self._compute_stress_preservation(
            source_stress, target_stress, common
        )

        # Compute synthesis confidence
        confidence = stress_preservation * grounding_rotation.confidence

        return GhostAnchor(
            concept_id=concept_id,
            source_position=source_pos,
            target_position=target_pos,
            stress_preservation=stress_preservation,
            grounding_rotation=grounding_rotation,
            common_anchor_count=common_anchor_count,
            source_anchor_count=source_anchor_count,
            target_anchor_count=target_anchor_count,
            source_stress=source_stress,
            target_stress=target_stress,
            synthesis_confidence=confidence,
        )

    def _solve_stress_preserving_position(
        self,
        source_stress: RelationalStressProfile,
        target_anchors: dict[str, "Array"],
        common_anchors: set[str],
    ) -> "Array":
        """
        Solve for the position in target space that minimizes relational stress residual.

        This is a multilateration problem: given geodesic distances to known points,
        find the position. We use iterative optimization with geodesic distance
        measurement and tangent-space gradient approximation.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        b = self._backend

        # Get target anchor positions
        anchor_list = sorted(common_anchors)
        n_anchors = len(anchor_list)

        # Target distances (from source stress profile - these are already geodesic)
        target_distances = {name: source_stress.anchor_distances[name] for name in common_anchors}

        # Build anchor matrix
        anchor_arrays = [b.reshape(target_anchors[a], (1, -1)) for a in anchor_list]
        anchor_matrix = b.concatenate(anchor_arrays, axis=0)
        anchor_arr = b.astype(anchor_matrix, "float32")
        b.eval(anchor_arr)

        anchor_geo = geodesic_distance_matrix(anchor_arr, k_neighbors=None, backend=b)
        b.eval(anchor_geo)
        eps = division_epsilon(b, anchor_arr)

        # Scale target distances by the ratio of geodesic spreads
        source_vals = list(source_stress.anchor_distances.values())
        source_mean = sum(source_vals) / len(source_vals)
        source_variance = sum((v - source_mean) ** 2 for v in source_vals) / len(source_vals)
        source_spread = sqrt_scalar(source_variance, b)

        # Target spread from non-zero geodesic distances
        anchor_flat = b.reshape(anchor_geo, (-1,))
        positive_mask = anchor_flat > 0
        mask_f = b.astype(positive_mask, "float32")
        count_arr = b.sum(mask_f)
        b.eval(count_arr)
        count = float(b.to_scalar(count_arr))
        if count > 0:
            values = anchor_flat * mask_f
            mean_arr = b.sum(values) / count
            var_arr = b.sum((values - mean_arr) ** 2 * mask_f) / count
            b.eval(mean_arr, var_arr)
            target_spread = sqrt_scalar(float(b.to_scalar(var_arr)), b)
        else:
            target_spread = 0.0

        if source_spread > 0 and target_spread > 0:
            scale_factor = target_spread / source_spread
        else:
            scale_factor = 1.0

        scaled_distances = {k: v * scale_factor for k, v in target_distances.items()}

        # Initialize position as weighted centroid of target anchors
        weights = {name: 1.0 / (d + eps) for name, d in scaled_distances.items()}
        total_weight = sum(weights.values())

        # Get dimensionality from first anchor
        first_anchor = target_anchors[anchor_list[0]]
        shape = b.shape(first_anchor)
        d = int(shape[-1]) if len(shape) > 0 else int(shape[0])

        # Initialize position using weighted sum
        position = b.zeros((d,))
        for name in anchor_list:
            pos = target_anchors[name]
            position = position + (weights[name] / total_weight) * pos
        b.eval(position)

        distance_scale = max(scaled_distances.values()) if scaled_distances else 0.0
        learning_rate = 1.0 / (distance_scale + eps)
        error_scale = sum(abs(v) for v in scaled_distances.values())
        error_threshold = eps * (error_scale + eps)

        target_dists_arr = b.array([scaled_distances[name] for name in anchor_list])

        for iteration in range(100):
            # Compute current geodesic distances from position to anchors
            pos_2d = b.reshape(position, (1, -1))
            all_points = b.concatenate([pos_2d, anchor_matrix], axis=0)
            points_arr = b.astype(all_points, "float32")
            b.eval(points_arr)
            geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=None, backend=b)
            b.eval(geo_dist)

            row0 = b.take(geo_dist, b.array([0]), axis=0)
            row0 = b.squeeze(row0, axis=0)
            anchor_indices = b.arange(1, n_anchors + 1)
            current_dists = b.take(row0, anchor_indices, axis=0)

            diffs = position - anchor_matrix
            diff_norms = geodesic_norms(diffs, b)
            valid_mask = b.astype(current_dists > eps, "float32") * b.astype(
                diff_norms > eps, "float32"
            )
            safe_norms = b.maximum(diff_norms, b.full(diff_norms.shape, eps))
            errors = current_dists - target_dists_arr
            coeffs = (2.0 * errors) / safe_norms
            coeffs = coeffs * valid_mask
            gradient = b.sum(diffs * b.reshape(coeffs, (-1, 1)), axis=0)

            total_error_arr = b.sum((errors * valid_mask) ** 2)
            b.eval(gradient, total_error_arr)
            total_error = float(b.to_scalar(total_error_arr))
            position = position - learning_rate * gradient
            b.eval(position)

            if total_error < error_threshold:
                break

            learning_rate *= max(0.0, 1.0 - eps)

        return position

    def _compute_stress_preservation(
        self,
        source_stress: RelationalStressProfile,
        target_stress: RelationalStressProfile,
        common_anchors: set[str],
    ) -> float:
        """Compute how well the stress pattern was preserved."""
        if not common_anchors:
            return 0.0

        # Compare normalized distances
        source_dists = [source_stress.normalized_distances.get(a, 0.0) for a in sorted(common_anchors)]
        target_dists = [target_stress.normalized_distances.get(a, 0.0) for a in sorted(common_anchors)]

        # Compute std for both
        n = len(source_dists)
        s_mean = sum(source_dists) / n
        t_mean = sum(target_dists) / n
        backend = get_default_backend()
        s_var = sum((v - s_mean) ** 2 for v in source_dists) / n
        t_var = sum((v - t_mean) ** 2 for v in target_dists) / n
        s_std = sqrt_scalar(s_var, backend)
        t_std = sqrt_scalar(t_var, backend)

        # Correlation between distance patterns
        if s_std > 0 and t_std > 0:
            numerator = sum((s - s_mean) * (t - t_mean) for s, t in zip(source_dists, target_dists))
            correlation = numerator / (s_std * t_std * n)
            if is_nan(correlation, backend):
                correlation = 0.0
        else:
            correlation = 0.0

        # Also consider absolute distance matching
        eps = ulp_scalar(1.0, backend)
        max_dist = max(max(source_dists), max(target_dists), eps)
        abs_diffs = [abs(s - t) for s, t in zip(source_dists, target_dists)]
        relative_error = sum(abs_diffs) / len(abs_diffs) / max_dist
        distance_match = max(0.0, 1.0 - relative_error)

        # Combine correlation and distance match with data-derived weighting
        weight_denom = s_std + t_std
        eps = ulp_scalar(1.0, backend)
        corr_weight = (s_std / weight_denom) if weight_denom > eps else 0.0
        distance_weight = 1.0 - corr_weight
        return float(corr_weight * max(0.0, correlation) + distance_weight * distance_match)


class CrossGroundingTransferEngine:
    """
    High-level engine for cross-grounding knowledge transfer.

    This orchestrates the full transfer pipeline:
    1. Analyze source and target grounding types
    2. Estimate the rotation between coordinate systems
    3. Synthesize Ghost Anchors for each concept to transfer
    4. Validate the transfer quality
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._synthesizer = CrossGroundingSynthesizer(backend)
        self._rotation_estimator = GroundingRotationEstimator(backend)

    def transfer_concepts(
        self,
        concepts: dict[str, "Array"],  # concept_id -> activation
        source_anchors: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
        source_grounding: str = "unknown",
        target_grounding: str = "unknown",
    ) -> CrossGroundingTransferResult:
        """
        Transfer multiple concepts from source to target model.

        Args:
            concepts: Dict of concept_id -> activation to transfer
            source_anchors: Universal anchors in source model
            target_anchors: Universal anchors in target model
            source_grounding: Source model's grounding type
            target_grounding: Target model's grounding type

        Returns:
            CrossGroundingTransferResult with all Ghost Anchors
        """
        # Estimate grounding rotation
        rotation = self._rotation_estimator.estimate_rotation(source_anchors, target_anchors)

        # Transfer each concept
        ghost_anchors = []

        for concept_id, activation in concepts.items():
            try:
                ghost = self._synthesizer.synthesize_ghost_anchor(
                    concept_id=concept_id,
                    source_activation=activation,
                    source_anchors=source_anchors,
                    target_anchors=target_anchors,
                    grounding_rotation=rotation,
                )
                ghost_anchors.append(ghost)

            except Exception as e:
                logger.warning(f"Failed to transfer concept {concept_id}: {e}")

        # Compute aggregate metrics
        if ghost_anchors:
            preservations = [g.stress_preservation for g in ghost_anchors]
            mean_preservation = sum(preservations) / len(preservations)
            min_preservation = min(preservations)
        else:
            mean_preservation = 0.0
            min_preservation = 0.0

        interpretability_gap = rotation.angle_degrees / 90.0  # Normalized 0-1

        return CrossGroundingTransferResult(
            source_model_grounding=source_grounding,
            target_model_grounding=target_grounding,
            grounding_rotation=rotation,
            ghost_anchors=ghost_anchors,
            mean_stress_preservation=mean_preservation,
            min_stress_preservation=min_preservation,
            interpretability_gap=interpretability_gap,
        )

    def estimate_transfer_feasibility(
        self,
        source_anchors: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
    ) -> dict:
        """
        Estimate how feasible a cross-grounding transfer would be.

        Returns a feasibility report without actually doing the transfer.
        """
        rotation = self._rotation_estimator.estimate_rotation(source_anchors, target_anchors)

        common_anchors = set(source_anchors.keys()) & set(target_anchors.keys())

        feasibility = {
            "common_anchors": len(common_anchors),
            "grounding_rotation_degrees": rotation.angle_degrees,
            "distance_correlation": rotation.distance_correlation,
            "aligned": rotation.aligned,
            "confidence": rotation.confidence,
        }

        return feasibility
