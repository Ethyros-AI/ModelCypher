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

"""
Cross-Grounding Transfer: Density Re-mapping for Coordinate-Invariant Knowledge Transfer.

The Problem:
  Model A (VL, High Visual Grounding) encodes "chair" at position P_A
  Model B (Text, Alternative Grounding) uses a rotated coordinate system
  Naive transfer: Force P_A into Model B → FAILS (wrong axes)

The Solution:
  Instead of transferring coordinates, transfer RELATIONAL STRESS.
  Relational Stress = the pattern of distances to universal anchors.
  This pattern is coordinate-invariant - it survives rotation.

Key Insight:
  If "chair" in Model A has distances {floor: 0.3, ceiling: 0.8, table: 0.2},
  we find/synthesize a position in Model B with the SAME stress pattern,
  regardless of how Model B's axes are oriented.

Mathematical Foundation:
  - Relational Stress Profile: R(c) = [d(c, a₁), d(c, a₂), ..., d(c, aₙ)]
  - Grounding Rotation: θ = arccos(alignment(source_axes, target_axes))
  - Cross-Grounding Synthesis: argmin_p ||R_source(c) - R_target(p)||²
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

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
        if len(self.stress_vector) != len(other.stress_vector):
            # Different anchor sets - use common anchors
            common = set(self.anchor_distances.keys()) & set(other.anchor_distances.keys())
            self_dists = [self.anchor_distances[a] for a in sorted(common)]
            other_dists = [other.anchor_distances[a] for a in sorted(common)]
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(self_dists, other_dists)))
        diffs = [a - b for a, b in zip(self.stress_vector, other.stress_vector)]
        return math.sqrt(sum(d * d for d in diffs))


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
    alignment_score: float  # 1.0 = perfectly aligned, 0.0 = orthogonal
    axis_correspondence: dict[str, str]  # source_axis -> target_axis mapping
    confidence: float  # How confident we are in the rotation estimate

    @property
    def is_aligned(self) -> bool:
        """Are the models' grounding axes well-aligned?"""
        return self.alignment_score > 0.7


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

    # Relational structure
    source_stress: RelationalStressProfile
    target_stress: RelationalStressProfile

    # Confidence
    synthesis_confidence: float  # How confident we are in this Ghost Anchor
    warning: str | None = None  # Any warnings about the synthesis


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
    successful_transfers: int
    failed_transfers: int

    # Recommendations
    interpretability_gap: float  # How much "rotation" was needed
    recommendation: str


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
        k_nearest: int = 5,
    ) -> RelationalStressProfile:
        """
        Compute the Relational Stress Profile for a concept.

        Args:
            concept_activation: The activation vector for the concept
            anchor_activations: Dict mapping anchor names to their activations
            k_nearest: Number of nearest anchors to track

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
        k_neighbors = min(max(3, len(anchor_names) // 2), len(anchor_names))
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        geo_dist_np = b.to_numpy(geo_dist)

        # Extract distances from concept (row 0) to each anchor
        distances = {}
        for i, name in enumerate(anchor_names):
            distances[name] = float(geo_dist_np[0, i + 1])

        # Normalize distances by the spread of anchor positions
        anchor_matrix = b.concatenate(anchor_list, axis=0)
        anchor_std = b.std(anchor_matrix)
        b.eval(anchor_std)
        anchor_spread = float(b.to_numpy(anchor_std))
        if anchor_spread > 0:
            normalized = {k: v / anchor_spread for k, v in distances.items()}
        else:
            normalized = distances.copy()

        # Find k nearest anchors
        sorted_anchors = sorted(distances.items(), key=lambda x: x[1])
        nearest = tuple(name for name, _ in sorted_anchors[:k_nearest])

        # Compute local density (inverse of mean distance to k nearest)
        k_distances = [d for _, d in sorted_anchors[:k_nearest]]
        local_density = 1.0 / (sum(k_distances) / len(k_distances) + 1e-8) if k_distances else 0.0

        # Compute curvature signature (eigenvalues of local covariance)
        curvature = self._estimate_local_curvature(concept_activation, anchor_activations)

        # Activation magnitude
        concept_norm = b.norm(concept_activation)
        b.eval(concept_norm)
        magnitude = float(b.to_numpy(concept_norm))

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
        k_geo = min(max(3, len(neighbor_names) // 2), len(neighbor_names))
        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_geo, backend=b)
        b.eval(geo_dist)
        geo_dist_np = b.to_numpy(geo_dist)

        # Extract distances from point (row 0) to neighbors
        dists = [(neighbor_names[i], float(geo_dist_np[0, i + 1])) for i in range(len(neighbor_names))]
        dists.sort(key=lambda x: x[1])
        k = min(10, len(dists))

        # Build local covariance from neighbor directions
        directions = []
        for name, _ in dists[:k]:
            direction = neighbors[name] - point
            norm_val = b.norm(direction)
            b.eval(norm_val)
            norm = float(b.to_numpy(norm_val))
            if norm > 1e-8:
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
            eig_np = b.to_numpy(eigenvalues)
            # Sort descending and keep top 3
            eig_sorted = sorted([float(e) for e in eig_np], reverse=True)[:3]
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
                alignment_score=0.0,
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
        k_neighbors = min(max(3, n // 3), n - 1)
        source_geo = geodesic_distance_matrix(source_arr, k_neighbors=k_neighbors, backend=b)
        b.eval(source_geo)
        source_dists = b.to_numpy(source_geo)

        # Build target position matrix
        target_list = [b.reshape(target_anchors[a], (1, -1)) for a in common_list]
        target_matrix = b.concatenate(target_list, axis=0)
        target_arr = b.astype(target_matrix, "float32")
        target_geo = geodesic_distance_matrix(target_arr, k_neighbors=k_neighbors, backend=b)
        b.eval(target_geo)
        target_dists = b.to_numpy(target_geo)

        # Normalize distance matrices
        source_max = max(float(v) for row in source_dists for v in row)
        target_max = max(float(v) for row in target_dists for v in row)
        source_dists = [[float(v) / (source_max + 1e-8) for v in row] for row in source_dists]
        target_dists = [[float(v) / (target_max + 1e-8) for v in row] for row in target_dists]

        # Compute alignment as correlation between distance matrices
        source_flat = [v for row in source_dists for v in row]
        target_flat = [v for row in target_dists for v in row]

        # Manual Pearson correlation
        n_pairs = len(source_flat)
        s_mean = sum(source_flat) / n_pairs
        t_mean = sum(target_flat) / n_pairs
        numerator = sum((s - s_mean) * (t - t_mean) for s, t in zip(source_flat, target_flat))
        s_std = math.sqrt(sum((s - s_mean) ** 2 for s in source_flat))
        t_std = math.sqrt(sum((t - t_mean) ** 2 for t in target_flat))
        if s_std > 0 and t_std > 0:
            correlation = numerator / (s_std * t_std)
        else:
            correlation = 0.0

        if math.isnan(correlation):
            correlation = 0.0

        # Convert correlation to angle
        alignment_score = max(0.0, correlation)
        angle_degrees = math.degrees(math.acos(min(1.0, alignment_score)))

        # Estimate axis correspondence using Procrustes-like analysis
        axis_correspondence = self._estimate_axis_correspondence(
            source_anchors, target_anchors, common_list
        )

        # Confidence based on number of common anchors and variance
        diffs = [abs(source_dists[i][j] - target_dists[i][j]) for i in range(n) for j in range(n) if i != j]
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            std_diff = math.sqrt(variance)
        else:
            std_diff = 0.0
        confidence = min(1.0, len(common_anchors) / 20.0) * (1.0 - std_diff)

        return GroundingRotation(
            angle_degrees=angle_degrees,
            alignment_score=float(alignment_score),
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

            source_vh_np = b.to_numpy(source_vh)
            target_vh_np = b.to_numpy(target_vh)

            # Match axes by correlation
            correspondence = {}
            n_axes = min(3, source_vh_np.shape[0], target_vh_np.shape[0])

            for i in range(n_axes):
                best_match = -1
                best_corr = -1.0
                for j in range(n_axes):
                    # Dot product of rows
                    corr = abs(sum(float(source_vh_np[i, k]) * float(target_vh_np[j, k]) for k in range(source_vh_np.shape[1])))
                    if corr > best_corr:
                        best_corr = corr
                        best_match = j
                correspondence[f"source_axis_{i}"] = f"target_axis_{best_match}"

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

        # Convert source position to numpy
        source_pos = self._to_numpy(source_activation)

        # Find common anchors
        common = set(source_stress.anchor_distances.keys()) & set(target_anchors.keys())
        if len(common) < 3:
            return GhostAnchor(
                concept_id=concept_id,
                source_position=source_pos,
                target_position=source_pos,  # Fallback: use source position
                stress_preservation=0.0,
                grounding_rotation=grounding_rotation,
                source_stress=source_stress,
                target_stress=source_stress,
                synthesis_confidence=0.0,
                warning="Insufficient common anchors for cross-grounding transfer",
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

        # Generate warning if stress preservation is low
        warning = None
        if stress_preservation < 0.5:
            warning = f"Low stress preservation ({stress_preservation:.2f}). The target position may not accurately represent the source concept."

        return GhostAnchor(
            concept_id=concept_id,
            source_position=source_pos,
            target_position=target_pos,
            stress_preservation=stress_preservation,
            grounding_rotation=grounding_rotation,
            source_stress=source_stress,
            target_stress=target_stress,
            synthesis_confidence=confidence,
            warning=warning,
        )

    def _solve_stress_preserving_position(
        self,
        source_stress: RelationalStressProfile,
        target_anchors: dict[str, "Array"],
        common_anchors: set[str],
    ) -> "Array":
        """
        Solve for the position in target space that best preserves relational stress.

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

        k_neighbors = min(max(3, n_anchors // 3), n_anchors - 1)
        anchor_geo = geodesic_distance_matrix(anchor_arr, k_neighbors=k_neighbors, backend=b)
        b.eval(anchor_geo)
        anchor_geo_np = b.to_numpy(anchor_geo)

        # Scale target distances by the ratio of geodesic spreads
        source_vals = list(source_stress.anchor_distances.values())
        source_mean = sum(source_vals) / len(source_vals)
        source_variance = sum((v - source_mean) ** 2 for v in source_vals) / len(source_vals)
        source_spread = math.sqrt(source_variance)

        # Target spread from non-zero geodesic distances
        target_vals = [float(v) for row in anchor_geo_np for v in row if v > 0]
        if target_vals:
            target_mean = sum(target_vals) / len(target_vals)
            target_variance = sum((v - target_mean) ** 2 for v in target_vals) / len(target_vals)
            target_spread = math.sqrt(target_variance)
        else:
            target_spread = 0.0

        if source_spread > 0 and target_spread > 0:
            scale_factor = target_spread / source_spread
        else:
            scale_factor = 1.0

        scaled_distances = {k: v * scale_factor for k, v in target_distances.items()}

        # Initialize position as weighted centroid of target anchors
        weights = {name: 1.0 / (d + 0.1) for name, d in scaled_distances.items()}
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

        learning_rate = 0.1

        for iteration in range(100):
            # Compute current geodesic distances from position to anchors
            pos_2d = b.reshape(position, (1, -1))
            all_points = b.concatenate([pos_2d, anchor_matrix], axis=0)
            points_arr = b.astype(all_points, "float32")
            b.eval(points_arr)
            geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=k_neighbors, backend=b)
            b.eval(geo_dist)
            geo_dist_np = b.to_numpy(geo_dist)

            gradient = b.zeros((d,))
            total_error = 0.0

            for i, name in enumerate(anchor_list):
                anchor_pos = target_anchors[name]
                target_dist = scaled_distances[name]
                current_dist = float(geo_dist_np[0, i + 1])

                current_diff = position - anchor_pos

                if current_dist > 1e-8:
                    error = current_dist - target_dist
                    total_error += error**2
                    diff_norm = b.norm(current_diff)
                    b.eval(diff_norm)
                    diff_norm_val = float(b.to_numpy(diff_norm))
                    if diff_norm_val > 1e-8:
                        gradient = gradient + 2 * error * current_diff / diff_norm_val

            b.eval(gradient)
            position = position - learning_rate * gradient
            b.eval(position)

            if total_error < 1e-6:
                break

            learning_rate *= 0.99

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
        s_var = sum((v - s_mean) ** 2 for v in source_dists) / n
        t_var = sum((v - t_mean) ** 2 for v in target_dists) / n
        s_std = math.sqrt(s_var)
        t_std = math.sqrt(t_var)

        # Correlation between distance patterns
        if s_std > 0 and t_std > 0:
            numerator = sum((s - s_mean) * (t - t_mean) for s, t in zip(source_dists, target_dists))
            correlation = numerator / (s_std * t_std * n)
            if math.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Also consider absolute distance matching
        max_dist = max(max(source_dists), max(target_dists), 1e-8)
        abs_diffs = [abs(s - t) for s, t in zip(source_dists, target_dists)]
        relative_error = sum(abs_diffs) / len(abs_diffs) / max_dist
        distance_match = max(0.0, 1.0 - relative_error)

        # Combine correlation and distance match
        return float(0.6 * max(0.0, correlation) + 0.4 * distance_match)


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
        successful = 0
        failed = 0

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

                if ghost.stress_preservation >= 0.5:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.warning(f"Failed to transfer concept {concept_id}: {e}")
                failed += 1

        # Compute aggregate metrics
        if ghost_anchors:
            preservations = [g.stress_preservation for g in ghost_anchors]
            mean_preservation = sum(preservations) / len(preservations)
            min_preservation = min(preservations)
        else:
            mean_preservation = 0.0
            min_preservation = 0.0

        # Generate recommendation
        interpretability_gap = rotation.angle_degrees / 90.0  # Normalized 0-1

        if rotation.is_aligned and mean_preservation > 0.8:
            recommendation = (
                "Excellent transfer quality. Source and target have aligned grounding - "
                "direct coordinate mapping is also viable."
            )
        elif mean_preservation > 0.6:
            recommendation = (
                f"Good transfer quality despite {rotation.angle_degrees:.1f}° grounding rotation. "
                "Ghost Anchors successfully preserved relational structure."
            )
        elif mean_preservation > 0.4:
            recommendation = (
                f"Moderate transfer quality. {rotation.angle_degrees:.1f}° rotation required. "
                "The conceptual geometry is invariant but current anchors capture it with "
                "limited precision. Consider additional probes for validation."
            )
        else:
            recommendation = (
                f"Low transfer quality. {rotation.angle_degrees:.1f}° rotation with stress distortion. "
                "Conceptual geometry is invariant but current anchor set is insufficient to "
                "capture the correspondence. Use more diverse anchors or verify model loading."
            )

        return CrossGroundingTransferResult(
            source_model_grounding=source_grounding,
            target_model_grounding=target_grounding,
            grounding_rotation=rotation,
            ghost_anchors=ghost_anchors,
            mean_stress_preservation=mean_preservation,
            min_stress_preservation=min_preservation,
            successful_transfers=successful,
            failed_transfers=failed,
            interpretability_gap=interpretability_gap,
            recommendation=recommendation,
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
            "alignment_score": rotation.alignment_score,
            "confidence": rotation.confidence,
            "is_aligned": rotation.is_aligned,
        }

        # Feasibility assessment
        if rotation.is_aligned:
            feasibility["feasibility"] = "HIGH"
            feasibility["recommendation"] = (
                "Models have aligned grounding. Direct transfer should work well."
            )
        elif rotation.alignment_score > 0.5:
            feasibility["feasibility"] = "MODERATE"
            feasibility["recommendation"] = (
                "Models have partially aligned grounding. "
                "Cross-grounding transfer recommended over direct mapping."
            )
        else:
            feasibility["feasibility"] = "LOW"
            feasibility["recommendation"] = (
                "Models have significantly different grounding axes. "
                "Transfer will require substantial re-mapping. "
                "Consider using more universal anchors."
            )

        return feasibility
