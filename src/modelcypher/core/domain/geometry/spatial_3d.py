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
Spatial 3D Metrology: Visual-Spatial Grounding Density.

Measures how concentrated a model's probability mass is along human-perceptual
3D axes. All models encode physics geometrically—the formulas, relationships,
and structure are geometric representations. This module measures alignment
with visual experience, not presence/absence of physics understanding.

Key Concepts:
- Spatial Prime Atlas: 3D basis vectors (X=lateral, Y=vertical, Z=depth)
- Euclidean Consistency: Do latent distances obey 3D Pythagorean theorem?
- Stereoscopy: Parallax shift between different viewpoint prompts
- Occlusion: Does "in front of" create measurable Z-axis shifts?
- Gravity Gradient: Does "down" act as a geometric sink?

Interpretation:
- VL Models: Visual grounding concentrates probability on specific 3D axes
  matching human visual experience (HIGH VISUAL GROUNDING).
- Text Models: Same geometric physics knowledge, but probability distributed
  along different axes—linguistic, formula-based, or higher-dimensional
  (ALTERNATIVE GROUNDING). Neither is "abstract"; both are geometric.

Analogy: A blind physicist and a sighted physicist both understand gravity
geometrically. The difference is in their probability distribution over
spatial concepts—one shaped by tactile/auditory experience, one by visual.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _safe_to_list(backend: "Backend", arr: "Array") -> list[float]:
    """Convert array to Python list, handling bfloat16 dtype."""
    try:
        raw = backend.to_numpy(arr)
    except (RuntimeError, TypeError):
        arr_f32 = backend.astype(arr, "float32")
        raw = backend.to_numpy(arr_f32)
    return [float(x) for x in raw.flatten()]


# =============================================================================
# Spatial Prime Atlas: 3D Basis Vectors
# =============================================================================


class SpatialAxis(str, Enum):
    """The three primitive axes of 3D space."""

    X_LATERAL = "x_lateral"  # Left <-> Right
    Y_VERTICAL = "y_vertical"  # Up <-> Down (Gravity)
    Z_DEPTH = "z_depth"  # Forward <-> Backward (Perspective)


@dataclass(frozen=True)
class SpatialAnchor:
    """A spatial concept with expected 3D coordinates."""

    name: str
    prompt: str
    expected_x: float  # -1 (left) to +1 (right)
    expected_y: float  # -1 (down) to +1 (up)
    expected_z: float  # -1 (far) to +1 (near)
    category: str = "general"


# The Spatial Prime Atlas: Concepts with known 3D positions
SPATIAL_PRIME_ATLAS: list[SpatialAnchor] = [
    # Vertical axis (Y) - Gravity gradient
    SpatialAnchor("ceiling", "The ceiling is above.", 0.0, 1.0, 0.0, "vertical"),
    SpatialAnchor("floor", "The floor is below.", 0.0, -1.0, 0.0, "vertical"),
    SpatialAnchor("sky", "The sky stretches overhead.", 0.0, 1.0, 0.5, "vertical"),
    SpatialAnchor("ground", "The ground beneath our feet.", 0.0, -1.0, 0.0, "vertical"),
    SpatialAnchor("cloud", "A cloud floats high above.", 0.0, 0.8, 0.3, "vertical"),
    SpatialAnchor("basement", "The basement is underground.", 0.0, -0.9, 0.0, "vertical"),
    # Lateral axis (X) - Sidedness
    SpatialAnchor("left_hand", "My left hand is on my left side.", -1.0, 0.0, 0.5, "lateral"),
    SpatialAnchor("right_hand", "My right hand is on my right side.", 1.0, 0.0, 0.5, "lateral"),
    SpatialAnchor("west", "The sun sets in the west.", -0.8, 0.0, 0.0, "lateral"),
    SpatialAnchor("east", "The sun rises in the east.", 0.8, 0.0, 0.0, "lateral"),
    # Depth axis (Z) - Perspective
    SpatialAnchor("foreground", "The object in the foreground is close.", 0.0, 0.0, 1.0, "depth"),
    SpatialAnchor(
        "background", "The mountains in the background are distant.", 0.0, 0.0, -1.0, "depth"
    ),
    SpatialAnchor("horizon", "The horizon line marks the far distance.", 0.0, 0.0, -0.9, "depth"),
    SpatialAnchor("here", "I am standing right here.", 0.0, 0.0, 1.0, "depth"),
    SpatialAnchor("there", "The building is over there.", 0.0, 0.0, -0.5, "depth"),
    # Physical objects with mass (Gravity test)
    SpatialAnchor("balloon", "A helium balloon floats upward.", 0.0, 0.7, 0.5, "mass"),
    SpatialAnchor("stone", "A heavy stone falls downward.", 0.0, -0.7, 0.5, "mass"),
    SpatialAnchor("feather", "A light feather drifts slowly.", 0.0, 0.3, 0.5, "mass"),
    SpatialAnchor("anvil", "The anvil sinks like a rock.", 0.0, -0.9, 0.5, "mass"),
    # Furniture (Virtual room test)
    SpatialAnchor("chair", "A chair sits on the floor.", 0.0, -0.5, 0.5, "furniture"),
    SpatialAnchor("table", "A table stands in the room.", 0.0, -0.3, 0.5, "furniture"),
    SpatialAnchor("lamp", "A lamp hangs from the ceiling.", 0.0, 0.7, 0.5, "furniture"),
    SpatialAnchor("rug", "A rug lies flat on the floor.", 0.0, -0.9, 0.5, "furniture"),
]


def get_spatial_anchors_by_axis(axis: SpatialAxis) -> list[SpatialAnchor]:
    """Get anchors that primarily vary along a given axis."""
    if axis == SpatialAxis.Y_VERTICAL:
        return [a for a in SPATIAL_PRIME_ATLAS if a.category in ("vertical", "mass")]
    elif axis == SpatialAxis.X_LATERAL:
        return [a for a in SPATIAL_PRIME_ATLAS if a.category == "lateral"]
    elif axis == SpatialAxis.Z_DEPTH:
        return [a for a in SPATIAL_PRIME_ATLAS if a.category == "depth"]
    return SPATIAL_PRIME_ATLAS


# =============================================================================
# Euclidean Consistency Score
# =============================================================================


@dataclass
class EuclideanConsistencyResult:
    """Result of Euclidean consistency check."""

    consistency_score: float
    pythagorean_error: float
    triangle_inequality_violations: int
    dimensionality_estimate: float
    axis_orthogonality: dict[str, float]

    @property
    def is_euclidean(self) -> bool:
        """Backward compat."""
        return self.consistency_score > 0.6 and self.triangle_inequality_violations == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consistency_score": self.consistency_score,
            "pythagorean_error": self.pythagorean_error,
            "triangle_inequality_violations": self.triangle_inequality_violations,
            "dimensionality_estimate": self.dimensionality_estimate,
            "axis_orthogonality": self.axis_orthogonality,
        }


class EuclideanConsistencyAnalyzer:
    """
    Tests whether latent distances form a valid 3D Euclidean space.

    Core Test: Given three points A, B, C forming a right angle at B,
    check if dist(A,C)² ≈ dist(A,B)² + dist(B,C)² (Pythagorean theorem).

    If this holds consistently, the manifold encodes Euclidean 3-space.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        anchor_activations: dict[str, "Array"],
        anchors: list[SpatialAnchor] | None = None,
    ) -> EuclideanConsistencyResult:
        """
        Analyze Euclidean consistency of spatial anchor representations.

        Args:
            anchor_activations: Map from anchor name to activation vector
            anchors: Spatial anchors (uses SPATIAL_PRIME_ATLAS if None)

        Returns:
            EuclideanConsistencyResult with consistency metrics
        """
        b = self._backend
        anchors = anchors or SPATIAL_PRIME_ATLAS

        # Filter to anchors we have activations for
        available = [a for a in anchors if a.name in anchor_activations]
        if len(available) < 4:
            return EuclideanConsistencyResult(
                consistency_score=0.0,
                pythagorean_error=float("inf"),
                triangle_inequality_violations=0,
                dimensionality_estimate=0.0,
                axis_orthogonality={},
            )

        # Build activation matrix and expected position matrix
        names = [a.name for a in available]
        # Convert to float32 before numpy (handles bfloat16)
        act_list = [_safe_to_list(b, anchor_activations[name]) for name in names]
        activations = b.stack([b.array(act) for act in act_list])
        expected_3d_list = [[a.expected_x, a.expected_y, a.expected_z] for a in available]

        # Compute pairwise latent distances
        n = len(available)
        latent_dists = self._compute_distance_matrix(activations)

        # Compute expected 3D distances (not used currently, kept for future reference)
        # This would compute euclidean distances in expected 3D space

        # Test 1: Pythagorean theorem on right-angle triplets
        pyth_errors = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i == j or j == k or i == k:
                        continue

                    # Check if j forms ~90° angle
                    vec_ij = [expected_3d_list[j][d] - expected_3d_list[i][d] for d in range(3)]
                    vec_jk = [expected_3d_list[k][d] - expected_3d_list[j][d] for d in range(3)]
                    dot = sum(vec_ij[d] * vec_jk[d] for d in range(3))
                    norm_ij = math.sqrt(sum(v * v for v in vec_ij))
                    norm_jk = math.sqrt(sum(v * v for v in vec_jk))
                    norm_prod = norm_ij * norm_jk

                    if norm_prod > 0.1:  # Non-degenerate
                        cos_angle = dot / norm_prod
                        if abs(cos_angle) < 0.2:  # ~90° angle
                            # Test Pythagorean: dist(i,k)² ≈ dist(i,j)² + dist(j,k)²
                            lhs = latent_dists[i, k] ** 2
                            rhs = latent_dists[i, j] ** 2 + latent_dists[j, k] ** 2

                            # Skip invalid values
                            if rhs > 0 and not math.isnan(rhs) and not math.isinf(rhs):
                                if not math.isnan(lhs) and not math.isinf(lhs):
                                    error = abs(lhs - rhs) / rhs
                                    if not math.isnan(error) and not math.isinf(error):
                                        pyth_errors.append(error)

        pythagorean_error = sum(pyth_errors) / len(pyth_errors) if pyth_errors else float("inf")

        # Test 2: Triangle inequality violations
        violations = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i < j < k:
                        d_ij = latent_dists[i, j]
                        d_jk = latent_dists[j, k]
                        d_ik = latent_dists[i, k]
                        if d_ij + d_jk < d_ik - 1e-6:
                            violations += 1
                        if d_ij + d_ik < d_jk - 1e-6:
                            violations += 1
                        if d_jk + d_ik < d_ij - 1e-6:
                            violations += 1

        # Test 3: Intrinsic dimensionality via MDS stress
        # If 3D, MDS with 3 components should have low stress
        dim_estimate = self._estimate_intrinsic_dimension(latent_dists)

        # Test 4: Axis orthogonality
        axis_ortho = self._compute_axis_orthogonality(activations, available)

        # Compute overall consistency score
        pyth_score = max(0, 1.0 - pythagorean_error) if pythagorean_error < float("inf") else 0.0
        triangle_score = 1.0 - min(1.0, violations / max(1, n * (n - 1) * (n - 2) / 6))
        dim_score = max(0, 1.0 - abs(dim_estimate - 3.0) / 3.0)

        consistency_score = 0.4 * pyth_score + 0.3 * triangle_score + 0.3 * dim_score

        return EuclideanConsistencyResult(
            consistency_score=consistency_score,
            pythagorean_error=pythagorean_error,
            triangle_inequality_violations=violations,
            dimensionality_estimate=dim_estimate,
            axis_orthogonality=axis_ortho,
        )

    def _compute_distance_matrix(self, activations: "Array") -> "Array":
        """Compute pairwise geodesic distances.

        Uses k-NN graph shortest paths to estimate true manifold distances.
        In high-dimensional curved spaces, Euclidean distance is incorrect.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        import numpy as np  # Local import for scipy boundary

        b = self._backend
        n = activations.shape[0] if hasattr(activations, "shape") else len(activations)

        # Use geodesic distance matrix for true manifold distances
        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(activations, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        geo_dist_np = b.to_numpy(geo_dist).astype(np.float64)

        # Handle any NaN/inf from geodesic computation
        geo_dist_np = np.nan_to_num(geo_dist_np, nan=0.0, posinf=1e10, neginf=0.0)

        return geo_dist_np

    def _estimate_intrinsic_dimension(self, dist_matrix: "Array") -> float:
        """Estimate intrinsic dimension using MDS eigenvalue decay."""
        import numpy as np  # Local import for scipy boundary

        n = dist_matrix.shape[0]

        # Handle NaN values in distance matrix
        if np.any(np.isnan(dist_matrix)) or np.any(np.isinf(dist_matrix)):
            dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1e10, neginf=-1e10)

        # Double centering for MDS
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (dist_matrix**2) @ H

        # Handle numerical issues in B matrix
        if np.any(np.isnan(B)) or np.any(np.isinf(B)):
            B = np.nan_to_num(B, nan=0.0, posinf=1e10, neginf=-1e10)

        # Eigendecomposition with error handling
        try:
            eigenvalues = np.linalg.eigvalsh(B)
            eigenvalues = np.sort(eigenvalues)[::-1]
        except np.linalg.LinAlgError:
            # Fallback: return a reasonable default
            return float(min(n, 10))

        # Filter out NaN/negative eigenvalues
        eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) == 0:
            return float(min(n, 3))

        # Count significant eigenvalues (> 1% of largest)
        threshold = 0.01 * eigenvalues[0] if len(eigenvalues) > 0 else 0
        significant = sum(e > threshold for e in eigenvalues)

        return float(significant)

    def _compute_axis_orthogonality(
        self,
        activations: "Array",
        anchors: list[SpatialAnchor],
    ) -> dict[str, float]:
        """Compute orthogonality between inferred X, Y, Z axes."""
        import numpy as np  # Local import for scipy boundary

        b = self._backend

        # Find axis-defining anchor pairs
        def find_axis_vector(pos_anchor: str, neg_anchor: str) -> "Array | None":
            pos_idx = next((i for i, a in enumerate(anchors) if a.name == pos_anchor), None)
            neg_idx = next((i for i, a in enumerate(anchors) if a.name == neg_anchor), None)
            if pos_idx is None or neg_idx is None:
                return None
            act_np = b.to_numpy(activations)
            return act_np[pos_idx] - act_np[neg_idx]

        # Infer axis directions
        x_axis = find_axis_vector("right_hand", "left_hand")
        y_axis = find_axis_vector("ceiling", "floor")
        z_axis = find_axis_vector("foreground", "background")

        results = {}

        # Compute pairwise orthogonality (cosine should be ~0)
        def orthogonality(v1: "Array | None", v2: "Array | None", name: str) -> None:
            if v1 is None or v2 is None:
                results[name] = 0.0
                return
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                results[name] = 0.0
                return
            cos = np.dot(v1, v2) / (norm1 * norm2)
            # Orthogonality score: 1 = orthogonal, 0 = parallel
            results[name] = 1.0 - abs(cos)

        orthogonality(x_axis, y_axis, "x_y_orthogonality")
        orthogonality(y_axis, z_axis, "y_z_orthogonality")
        orthogonality(x_axis, z_axis, "x_z_orthogonality")

        return results


# =============================================================================
# Spatial Stereoscopy
# =============================================================================


@dataclass(frozen=True)
class ViewpointPrompt:
    """A prompt describing a scene from a specific viewpoint."""

    scene_id: str
    viewpoint: str  # "front", "left", "right", "above", "behind"
    prompt: str
    expected_parallax_x: float  # Expected X shift relative to front view
    expected_parallax_y: float  # Expected Y shift
    expected_parallax_z: float  # Expected Z shift


# Stereoscopic probe pairs
STEREOSCOPIC_SCENES: list[ViewpointPrompt] = [
    # Scene 1: A cube on a table
    ViewpointPrompt(
        "cube", "front", "A red cube sits on a wooden table, viewed from the front.", 0, 0, 0
    ),
    ViewpointPrompt(
        "cube",
        "left",
        "A red cube sits on a wooden table, viewed from the left side.",
        -0.5,
        0,
        0.2,
    ),
    ViewpointPrompt(
        "cube",
        "right",
        "A red cube sits on a wooden table, viewed from the right side.",
        0.5,
        0,
        0.2,
    ),
    ViewpointPrompt(
        "cube", "above", "A red cube sits on a wooden table, viewed from above.", 0, 0.5, 0.3
    ),
    # Scene 2: A person standing
    ViewpointPrompt("person", "front", "A person stands facing me directly.", 0, 0, 0),
    ViewpointPrompt("person", "left", "A person stands, I see their left profile.", -0.5, 0, 0.1),
    ViewpointPrompt("person", "behind", "A person stands with their back to me.", 0, 0, -0.5),
    # Scene 3: A car on a road
    ViewpointPrompt("car", "front", "A car approaches from the front.", 0, 0, 0.8),
    ViewpointPrompt("car", "side", "A car passes by on the side.", 0.5, 0, 0),
    ViewpointPrompt("car", "behind", "A car drives away into the distance.", 0, 0, -0.8),
]


@dataclass
class StereoscopyResult:
    """Result of stereoscopic parallax analysis."""

    scene_id: str
    parallax_correlation: float  # Correlation between expected and measured parallax
    measured_parallax: dict[str, tuple[float, float, float]]  # viewpoint -> (dx, dy, dz)
    expected_parallax: dict[str, tuple[float, float, float]]
    depth_axis_detected: bool  # Is there a consistent Z-axis?
    perspective_consistency: float  # 0-1 score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "parallax_correlation": self.parallax_correlation,
            "measured_parallax": self.measured_parallax,
            "expected_parallax": self.expected_parallax,
            "depth_axis_detected": self.depth_axis_detected,
            "perspective_consistency": self.perspective_consistency,
        }


class SpatialStereoscopy:
    """
    Measures parallax shift between different viewpoint prompts.

    If the model has a 3D world model, viewing the same scene from
    different angles should produce predictable shifts in latent space
    that match real-world parallax.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze_scene(
        self,
        viewpoint_activations: dict[str, "Array"],
        scene_prompts: list[ViewpointPrompt],
    ) -> StereoscopyResult:
        """
        Analyze parallax for a single scene across viewpoints.

        Args:
            viewpoint_activations: Map from viewpoint to activation
            scene_prompts: ViewpointPrompts for this scene

        Returns:
            StereoscopyResult with parallax analysis
        """
        b = self._backend

        if len(viewpoint_activations) < 2:
            return StereoscopyResult(
                scene_id=scene_prompts[0].scene_id if scene_prompts else "unknown",
                parallax_correlation=0.0,
                measured_parallax={},
                expected_parallax={},
                depth_axis_detected=False,
                perspective_consistency=0.0,
            )

        scene_id = scene_prompts[0].scene_id

        # Find the front view as reference
        front_prompt = next((p for p in scene_prompts if p.viewpoint == "front"), scene_prompts[0])
        if front_prompt.viewpoint not in viewpoint_activations:
            front_prompt = scene_prompts[0]

        front_act = _safe_to_list(b, viewpoint_activations[front_prompt.viewpoint])

        # Compute parallax (difference from front view) for each viewpoint
        measured = {}
        expected = {}

        for prompt in scene_prompts:
            if prompt.viewpoint not in viewpoint_activations:
                continue

            act = _safe_to_list(b, viewpoint_activations[prompt.viewpoint])
            diff = [act[i] - front_act[i] for i in range(len(act))]

            # Project onto principal axes to get (dx, dy, dz) approximation
            # For now, use first 3 principal components as proxy for x, y, z
            if len(diff) >= 3:
                # Simple: use first 3 dimensions as spatial proxy
                dx, dy, dz = float(diff[0]), float(diff[1]), float(diff[2])
            else:
                dx, dy, dz = 0.0, 0.0, 0.0

            measured[prompt.viewpoint] = (dx, dy, dz)
            expected[prompt.viewpoint] = (
                prompt.expected_parallax_x,
                prompt.expected_parallax_y,
                prompt.expected_parallax_z,
            )

        # Compute correlation between measured and expected parallax
        if len(measured) >= 2:
            import numpy as np  # Local import for corrcoef

            meas_flat = []
            exp_flat = []
            for vp in measured:
                meas_flat.extend(measured[vp])
                exp_flat.extend(expected[vp])

            meas_arr = np.array(meas_flat)
            exp_arr = np.array(exp_flat)

            if np.std(meas_arr) > 1e-6 and np.std(exp_arr) > 1e-6:
                correlation = float(np.corrcoef(meas_arr, exp_arr)[0, 1])
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        # Check if there's a consistent Z-axis (depth)
        z_values = [m[2] for m in measured.values()]
        if z_values and len(z_values) > 1:
            z_mean = sum(z_values) / len(z_values)
            z_variance = sum((z - z_mean) ** 2 for z in z_values) / len(z_values)
            z_std = math.sqrt(z_variance)
            depth_detected = z_std > 0.01
        else:
            depth_detected = False

        # Perspective consistency: how well parallax scales with expected depth
        consistency = max(0.0, min(1.0, (correlation + 1) / 2))

        return StereoscopyResult(
            scene_id=scene_id,
            parallax_correlation=correlation,
            measured_parallax=measured,
            expected_parallax=expected,
            depth_axis_detected=depth_detected,
            perspective_consistency=consistency,
        )


# =============================================================================
# Gravity Gradient Analyzer
# =============================================================================


@dataclass
class GravityGradientResult:
    """Result of gravity gradient analysis."""

    gravity_axis_detected: bool
    gravity_direction: "Array | None"  # Unit vector pointing "down"
    mass_correlation: float  # Correlation between object mass and position along gravity axis
    layer_gravity_strengths: dict[int, float]  # Per-layer gravity effect
    sink_anchors: list[str]  # Anchors that act as gravitational sinks
    float_anchors: list[str]  # Anchors that resist gravity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        import numpy as np  # Local import for linalg.norm

        # Truncate gravity_direction for display (full vector is 2048+ dims)
        grav_dir_summary = None
        if self.gravity_direction is not None:
            grav_norm = float(np.linalg.norm(self.gravity_direction))
            grav_dir_summary = {
                "norm": grav_norm,
                "top_5_dims": self.gravity_direction[:5].tolist()
                if len(self.gravity_direction) > 5
                else self.gravity_direction.tolist(),
            }
        return {
            "gravity_axis_detected": self.gravity_axis_detected,
            "gravity_direction_summary": grav_dir_summary,
            "mass_correlation": self.mass_correlation,
            "layer_gravity_strengths": self.layer_gravity_strengths,
            "sink_anchors": self.sink_anchors,
            "float_anchors": self.float_anchors,
        }


class GravityGradientAnalyzer:
    """
    Analyzes whether the model has a "gravity gradient" in its latent space.

    Hypothesis: If a model understands physics, heavy objects should be
    pulled toward a "down" direction, creating a metric distortion where
    "Floor" acts as a geometric sink.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        anchor_activations: dict[str, "Array"],
        layer_activations: dict[int, dict[str, "Array"]] | None = None,
    ) -> GravityGradientResult:
        """
        Analyze gravity gradient in latent representations.

        Args:
            anchor_activations: Map from anchor name to activation
            layer_activations: Optional per-layer activations for depth analysis

        Returns:
            GravityGradientResult with gravity analysis
        """
        b = self._backend

        # Get vertical anchors
        vertical_anchors = get_spatial_anchors_by_axis(SpatialAxis.Y_VERTICAL)
        available = [a for a in vertical_anchors if a.name in anchor_activations]

        if len(available) < 3:
            return GravityGradientResult(
                gravity_axis_detected=False,
                gravity_direction=None,
                mass_correlation=0.0,
                layer_gravity_strengths={},
                sink_anchors=[],
                float_anchors=[],
            )

        # Find gravity direction: vector from "ceiling" to "floor"
        import numpy as np  # Local import for numpy operations

        ceiling_act = None
        floor_act = None
        for anchor in available:
            if anchor.name in ("ceiling", "sky"):
                act = b.to_numpy(anchor_activations[anchor.name])
                ceiling_act = act.astype(np.float64)
            if anchor.name in ("floor", "ground"):
                act = b.to_numpy(anchor_activations[anchor.name])
                floor_act = act.astype(np.float64)

        gravity_dir = None
        if ceiling_act is not None and floor_act is not None:
            # Clip to prevent overflow
            ceiling_act = np.clip(ceiling_act, -1e10, 1e10)
            floor_act = np.clip(floor_act, -1e10, 1e10)
            gravity_dir = floor_act - ceiling_act
            norm = np.linalg.norm(gravity_dir)
            if norm > 1e-6 and not np.isnan(norm) and not np.isinf(norm):
                gravity_dir = gravity_dir / norm
            else:
                gravity_dir = None

        # Analyze mass-position correlation
        mass_positions = []
        for anchor in available:
            act = b.to_numpy(anchor_activations[anchor.name])
            act = act.astype(np.float64)
            act = np.clip(act, -1e10, 1e10)

            # Project onto gravity axis
            if gravity_dir is not None:
                position = np.dot(act, gravity_dir)
            else:
                position = act[1] if len(act) > 1 else 0  # Use 2nd dim as proxy

            # Skip invalid positions
            if np.isnan(position) or np.isinf(position):
                continue

            # Get expected Y position (negative = down = heavy)
            expected_mass = -anchor.expected_y  # Lower Y = heavier

            mass_positions.append((expected_mass, float(position)))

        # Compute correlation
        if len(mass_positions) >= 3:
            masses = np.array([mp[0] for mp in mass_positions])
            positions = np.array([mp[1] for mp in mass_positions])
            std_m = np.std(masses)
            std_p = np.std(positions)
            if std_m > 1e-6 and std_p > 1e-6 and not np.isnan(std_p) and not np.isinf(std_p):
                corr_matrix = np.corrcoef(masses, positions)
                mass_correlation = float(corr_matrix[0, 1])
                if np.isnan(mass_correlation):
                    mass_correlation = 0.0
            else:
                mass_correlation = 0.0
        else:
            mass_correlation = 0.0

        # Per-layer gravity analysis
        layer_strengths = {}
        if layer_activations:
            for layer_idx, layer_acts in layer_activations.items():
                layer_corr = self._compute_layer_gravity(layer_acts, available)
                layer_strengths[layer_idx] = layer_corr

        # Identify sink (heavy) and float (light) anchors
        sink_anchors = [a.name for a in available if a.expected_y < -0.3]
        float_anchors = [a.name for a in available if a.expected_y > 0.3]

        return GravityGradientResult(
            gravity_axis_detected=gravity_dir is not None and abs(mass_correlation) > 0.3,
            gravity_direction=gravity_dir,
            mass_correlation=mass_correlation,
            layer_gravity_strengths=layer_strengths,
            sink_anchors=sink_anchors,
            float_anchors=float_anchors,
        )

    def _compute_layer_gravity(
        self,
        layer_acts: dict[str, "Array"],
        anchors: list[SpatialAnchor],
    ) -> float:
        """Compute gravity correlation for a single layer."""
        import numpy as np  # Local import for corrcoef

        b = self._backend

        pairs = []
        for anchor in anchors:
            if anchor.name not in layer_acts:
                continue
            act_list = _safe_to_list(b, layer_acts[anchor.name])
            # Use mean activation as proxy for "position"
            position = float(sum(act_list) / len(act_list)) if act_list else 0.0
            expected_mass = -anchor.expected_y
            pairs.append((expected_mass, position))

        if len(pairs) < 3:
            return 0.0

        masses = np.array([p[0] for p in pairs])
        positions = np.array([p[1] for p in pairs])

        if np.std(masses) > 1e-6 and np.std(positions) > 1e-6:
            return float(np.corrcoef(masses, positions)[0, 1])
        return 0.0


# =============================================================================
# Volumetric Density Prober
# =============================================================================


@dataclass
class VolumetricDensityResult:
    """Result of volumetric density analysis."""

    anchor_densities: dict[str, float]  # Anchor -> representational density
    density_mass_correlation: float  # Correlation between density and physical mass
    perspective_attenuation: float  # Does density decrease with distance?
    inverse_square_compliance: float  # Does density follow 1/r² law?

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anchor_densities": self.anchor_densities,
            "density_mass_correlation": self.density_mass_correlation,
            "perspective_attenuation": self.perspective_attenuation,
            "inverse_square_compliance": self.inverse_square_compliance,
        }


class VolumetricDensityProber:
    """
    Probes whether physical objects have representational densities
    that match their real-world properties.

    Heavy, dense objects should have "denser" representations (higher
    activation variance/concentration) than light objects.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        anchor_activations: dict[str, "Array"],
        anchors: list[SpatialAnchor] | None = None,
    ) -> VolumetricDensityResult:
        """
        Analyze volumetric density of anchor representations.

        Args:
            anchor_activations: Map from anchor name to activation
            anchors: Spatial anchors (uses mass-related from atlas if None)

        Returns:
            VolumetricDensityResult with density analysis
        """
        b = self._backend

        if anchors is None:
            anchors = [a for a in SPATIAL_PRIME_ATLAS if a.category in ("mass", "furniture")]

        available = [a for a in anchors if a.name in anchor_activations]

        if len(available) < 2:
            return VolumetricDensityResult(
                anchor_densities={},
                density_mass_correlation=0.0,
                perspective_attenuation=0.0,
                inverse_square_compliance=0.0,
            )

        # Compute "density" as activation concentration (L2 norm / variance)
        import numpy as np  # Local import for numpy operations

        densities = {}
        for anchor in available:
            act = b.to_numpy(anchor_activations[anchor.name])
            # Convert to float64 for numerical stability
            act_f64 = act.astype(np.float64)
            act_f64 = np.clip(act_f64, -1e10, 1e10)  # Prevent overflow

            norm = np.linalg.norm(act_f64)
            var = np.var(act_f64)

            # Density metric: higher norm and lower variance = more concentrated
            if var > 1e-6 and not np.isnan(var) and not np.isinf(var):
                density = norm / np.sqrt(var)
            else:
                density = norm

            # Handle NaN/inf
            if np.isnan(density) or np.isinf(density):
                density = 0.0

            densities[anchor.name] = float(density)

        # Correlate density with expected "mass" (negative Y position)
        density_mass_pairs = []
        for anchor in available:
            if anchor.name in densities:
                expected_mass = -anchor.expected_y  # Lower Y = heavier
                density_mass_pairs.append((expected_mass, densities[anchor.name]))

        if len(density_mass_pairs) >= 3:
            masses = np.array([p[0] for p in density_mass_pairs])
            dens = np.array([p[1] for p in density_mass_pairs])
            if np.std(masses) > 1e-6 and np.std(dens) > 1e-6:
                density_mass_corr = float(np.corrcoef(masses, dens)[0, 1])
            else:
                density_mass_corr = 0.0
        else:
            density_mass_corr = 0.0

        # Analyze perspective attenuation (density vs depth)
        depth_density_pairs = []
        for anchor in available:
            if anchor.name in densities:
                depth = anchor.expected_z
                depth_density_pairs.append((depth, densities[anchor.name]))

        if len(depth_density_pairs) >= 3:
            depths = np.array([p[0] for p in depth_density_pairs])
            dens = np.array([p[1] for p in depth_density_pairs])
            if np.std(depths) > 1e-6 and np.std(dens) > 1e-6:
                perspective_atten = float(np.corrcoef(depths, dens)[0, 1])
            else:
                perspective_atten = 0.0
        else:
            perspective_atten = 0.0

        # Check inverse-square law: density ∝ 1/(1 + |z|)²
        inverse_sq_errors = []
        for anchor in available:
            if anchor.name in densities:
                depth = abs(anchor.expected_z)
                expected_attenuation = 1.0 / (1.0 + depth) ** 2
                # Normalize measured density to [0, 1]
                all_dens = list(densities.values())
                if max(all_dens) > min(all_dens):
                    normalized = (densities[anchor.name] - min(all_dens)) / (
                        max(all_dens) - min(all_dens)
                    )
                else:
                    normalized = 0.5
                error = abs(normalized - expected_attenuation)
                inverse_sq_errors.append(error)

        inverse_sq_compliance = (
            1.0 - sum(inverse_sq_errors) / len(inverse_sq_errors) if inverse_sq_errors else 0.0
        )

        return VolumetricDensityResult(
            anchor_densities=densities,
            density_mass_correlation=density_mass_corr,
            perspective_attenuation=perspective_atten,
            inverse_square_compliance=max(0.0, inverse_sq_compliance),
        )


# =============================================================================
# Occlusion Prober
# =============================================================================


@dataclass(frozen=True)
class OcclusionPrompt:
    """A prompt pair testing occlusion understanding."""

    scene_id: str
    object_a: str
    object_b: str
    a_in_front_prompt: str
    b_in_front_prompt: str


OCCLUSION_PROBES: list[OcclusionPrompt] = [
    OcclusionPrompt(
        "box_ball",
        "box",
        "ball",
        "A box is in front of a ball, hiding the ball from view.",
        "A ball is in front of a box, partially blocking the box.",
    ),
    OcclusionPrompt(
        "person_tree",
        "person",
        "tree",
        "A person stands in front of the tree, blocking it.",
        "A tree is in front of the person, obscuring them from view.",
    ),
    OcclusionPrompt(
        "car_building",
        "car",
        "building",
        "A car is parked in front of the building's entrance.",
        "The building looms in front of the parked car.",
    ),
]


@dataclass
class OcclusionResult:
    """Result of occlusion analysis."""

    scene_id: str
    z_shift_detected: bool
    a_front_z_position: float
    b_front_z_position: float
    z_swap_magnitude: float  # How much Z changed when swapping front/back
    occlusion_understood: bool  # Does swap cause appropriate Z shift?

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "z_shift_detected": self.z_shift_detected,
            "a_front_z_position": self.a_front_z_position,
            "b_front_z_position": self.b_front_z_position,
            "z_swap_magnitude": self.z_swap_magnitude,
            "occlusion_understood": self.occlusion_understood,
        }


class OcclusionProber:
    """
    Tests whether the model understands spatial occlusion.

    When "A is in front of B" vs "B is in front of A", there should
    be a measurable Z-axis shift in the representations.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        a_front_activation: "Array",
        b_front_activation: "Array",
        probe: OcclusionPrompt,
    ) -> OcclusionResult:
        """
        Analyze occlusion understanding for a single probe.

        Args:
            a_front_activation: Activation when A is in front
            b_front_activation: Activation when B is in front
            probe: The occlusion probe being tested

        Returns:
            OcclusionResult with occlusion analysis
        """
        b = self._backend

        a_act = _safe_to_list(b, a_front_activation)
        b_act = _safe_to_list(b, b_front_activation)

        diff = [b_act[i] - a_act[i] for i in range(len(a_act))]

        # Measure Z-shift (use 3rd principal component or specific dim)
        # For now, use the dimension with largest absolute change
        abs_diff = [abs(d) for d in diff]
        max_change_idx = abs_diff.index(max(abs_diff))
        z_shift = diff[max_change_idx]

        a_z = float(a_act[max_change_idx])
        b_z = float(b_act[max_change_idx])

        # Z shift should be significant and consistent (one object moves "forward")
        z_shift_magnitude = abs(z_shift)
        # Compute std manually
        a_mean = sum(a_act) / len(a_act)
        a_variance = sum((x - a_mean) ** 2 for x in a_act) / len(a_act)
        a_std = math.sqrt(a_variance)
        z_shift_detected = z_shift_magnitude > 0.1 * a_std

        # Occlusion understood if swapping causes consistent Z movement
        occlusion_understood = z_shift_detected and z_shift_magnitude > 0.05

        return OcclusionResult(
            scene_id=probe.scene_id,
            z_shift_detected=z_shift_detected,
            a_front_z_position=a_z,
            b_front_z_position=b_z,
            z_swap_magnitude=float(z_shift_magnitude),
            occlusion_understood=occlusion_understood,
        )


# =============================================================================
# Unified 3D World Model Analyzer
# =============================================================================


@dataclass
class Spatial3DReport:
    """Comprehensive 3D world model analysis."""

    euclidean_consistency: EuclideanConsistencyResult
    gravity_gradient: GravityGradientResult
    volumetric_density: VolumetricDensityResult
    stereoscopy_results: list[StereoscopyResult]
    occlusion_results: list[OcclusionResult]

    # Summary scores
    has_3d_world_model: bool
    world_model_score: float  # 0-1 composite score
    physics_engine_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "euclidean_consistency": self.euclidean_consistency.to_dict(),
            "gravity_gradient": self.gravity_gradient.to_dict(),
            "volumetric_density": self.volumetric_density.to_dict(),
            "stereoscopy_results": [s.to_dict() for s in self.stereoscopy_results],
            "occlusion_results": [o.to_dict() for o in self.occlusion_results],
            "has_3d_world_model": self.has_3d_world_model,
            "world_model_score": self.world_model_score,
            "physics_engine_detected": self.physics_engine_detected,
        }


class Spatial3DAnalyzer:
    """
    Unified analyzer for 3D world model detection.

    Combines all spatial probes to determine if a model has internalized
    a geometrically consistent 3D physics engine.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._euclidean = EuclideanConsistencyAnalyzer(backend)
        self._gravity = GravityGradientAnalyzer(backend)
        self._density = VolumetricDensityProber(backend)
        self._stereoscopy = SpatialStereoscopy(backend)
        self._occlusion = OcclusionProber(backend)

    def full_analysis(
        self,
        anchor_activations: dict[str, "Array"],
        stereoscopy_activations: dict[str, dict[str, "Array"]] | None = None,
        occlusion_activations: dict[str, tuple["Array", "Array"]] | None = None,
    ) -> Spatial3DReport:
        """
        Run complete 3D world model analysis.

        Args:
            anchor_activations: Spatial anchor activations
            stereoscopy_activations: Scene -> viewpoint -> activation
            occlusion_activations: Scene -> (a_front_act, b_front_act)

        Returns:
            Spatial3DReport with comprehensive analysis
        """
        # Run component analyses
        euclidean = self._euclidean.analyze(anchor_activations)
        gravity = self._gravity.analyze(anchor_activations)
        density = self._density.analyze(anchor_activations)

        # Stereoscopy analysis
        stereo_results = []
        if stereoscopy_activations:
            for scene_id, viewpoint_acts in stereoscopy_activations.items():
                scene_prompts = [p for p in STEREOSCOPIC_SCENES if p.scene_id == scene_id]
                if scene_prompts:
                    result = self._stereoscopy.analyze_scene(viewpoint_acts, scene_prompts)
                    stereo_results.append(result)

        # Occlusion analysis
        occlusion_results = []
        if occlusion_activations:
            for scene_id, (a_act, b_act) in occlusion_activations.items():
                probe = next((p for p in OCCLUSION_PROBES if p.scene_id == scene_id), None)
                if probe:
                    result = self._occlusion.analyze(a_act, b_act, probe)
                    occlusion_results.append(result)

        # Compute summary scores
        euclidean_score = euclidean.consistency_score
        gravity_score = abs(gravity.mass_correlation) if gravity.gravity_axis_detected else 0.0
        density_score = max(0, density.inverse_square_compliance)

        if stereo_results:
            stereo_vals = [s.perspective_consistency for s in stereo_results]
            stereo_score = sum(stereo_vals) / len(stereo_vals)
        else:
            stereo_score = 0.0

        if occlusion_results:
            occ_vals = [1.0 if o.occlusion_understood else 0.0 for o in occlusion_results]
            occlusion_score = sum(occ_vals) / len(occ_vals)
        else:
            occlusion_score = 0.0

        # Composite world model score
        world_model_score = (
            0.25 * euclidean_score
            + 0.20 * gravity_score
            + 0.20 * density_score
            + 0.20 * stereo_score
            + 0.15 * occlusion_score
        )

        has_3d = world_model_score > 0.5 and euclidean.is_euclidean
        physics_detected = gravity.gravity_axis_detected and density.density_mass_correlation > 0.3

        return Spatial3DReport(
            euclidean_consistency=euclidean,
            gravity_gradient=gravity,
            volumetric_density=density,
            stereoscopy_results=stereo_results,
            occlusion_results=occlusion_results,
            has_3d_world_model=has_3d,
            world_model_score=world_model_score,
            physics_engine_detected=physics_detected,
        )


__all__ = [
    # Data structures
    "SpatialAxis",
    "SpatialAnchor",
    "SPATIAL_PRIME_ATLAS",
    "get_spatial_anchors_by_axis",
    # Euclidean consistency
    "EuclideanConsistencyResult",
    "EuclideanConsistencyAnalyzer",
    # Stereoscopy
    "ViewpointPrompt",
    "STEREOSCOPIC_SCENES",
    "StereoscopyResult",
    "SpatialStereoscopy",
    # Gravity
    "GravityGradientResult",
    "GravityGradientAnalyzer",
    # Density
    "VolumetricDensityResult",
    "VolumetricDensityProber",
    # Occlusion
    "OcclusionPrompt",
    "OCCLUSION_PROBES",
    "OcclusionResult",
    "OcclusionProber",
    # Unified
    "Spatial3DReport",
    "Spatial3DAnalyzer",
]
