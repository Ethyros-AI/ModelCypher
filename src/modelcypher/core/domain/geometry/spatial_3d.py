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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.spatial_atlas import (
    SpatialAxis,
    SpatialCategory,
    SpatialConcept,
    SpatialConceptInventory,
)

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
# Backend-Compatible Numerical Helpers (NO NUMPY)
# =============================================================================


def _backend_isnan(backend: "Backend", arr: "Array") -> "Array":
    """Check for NaN values using backend ops. NaN != NaN by IEEE 754."""
    return arr != arr  # NaN is the only value not equal to itself


def _backend_isinf(backend: "Backend", arr: "Array") -> "Array":
    """Check for infinite values using backend ops."""
    # Inf is greater than any finite value
    max_finite = 1e38  # Below float32 max
    return backend.abs(arr) > max_finite


def _backend_nan_to_num(
    backend: "Backend",
    arr: "Array",
    nan_val: float = 0.0,
    posinf_val: float = 1e10,
    neginf_val: float = -1e10,
) -> "Array":
    """Replace NaN/Inf with finite values using backend ops."""
    b = backend
    result = arr

    # Replace NaN
    is_nan = _backend_isnan(b, arr)
    result = b.where(is_nan, b.full(arr.shape, nan_val), result)

    # Replace positive infinity
    is_posinf = arr > 1e38
    result = b.where(is_posinf, b.full(arr.shape, posinf_val), result)

    # Replace negative infinity
    is_neginf = arr < -1e38
    result = b.where(is_neginf, b.full(arr.shape, neginf_val), result)

    b.eval(result)
    return result


def _backend_corrcoef(backend: "Backend", x: "Array", y: "Array") -> float:
    """Compute Pearson correlation coefficient using backend ops.

    r = Σ((x - μ_x)(y - μ_y)) / (n * σ_x * σ_y)
    """
    b = backend

    # Ensure 1D
    x_flat = b.reshape(x, (-1,))
    y_flat = b.reshape(y, (-1,))

    n = x_flat.shape[0]
    if n < 2:
        return 0.0

    # Means
    mean_x = b.mean(x_flat)
    mean_y = b.mean(y_flat)
    b.eval(mean_x, mean_y)

    # Centered
    x_centered = x_flat - mean_x
    y_centered = y_flat - mean_y

    # Standard deviations
    std_x = b.sqrt(b.mean(x_centered * x_centered))
    std_y = b.sqrt(b.mean(y_centered * y_centered))
    b.eval(std_x, std_y)

    std_x_val = float(b.to_numpy(std_x))
    std_y_val = float(b.to_numpy(std_y))

    if std_x_val < 1e-10 or std_y_val < 1e-10:
        return 0.0

    # Covariance
    cov = b.mean(x_centered * y_centered)
    b.eval(cov)

    # Correlation
    corr = cov / (std_x * std_y)
    b.eval(corr)

    result = float(b.to_numpy(corr))

    # Handle NaN result
    if result != result:  # NaN check
        return 0.0

    return result


def _backend_vector_norm(backend: "Backend", v: "Array") -> float:
    """Compute L2 norm of vector using backend ops."""
    b = backend
    v_flat = b.reshape(v, (-1,))
    norm_sq = b.sum(v_flat * v_flat)
    norm = b.sqrt(norm_sq)
    b.eval(norm)
    return float(b.to_numpy(norm))


def _backend_vector_dot(backend: "Backend", v1: "Array", v2: "Array") -> float:
    """Compute dot product of vectors using backend ops."""
    b = backend
    v1_flat = b.reshape(v1, (-1,))
    v2_flat = b.reshape(v2, (-1,))
    dot = b.sum(v1_flat * v2_flat)
    b.eval(dot)
    return float(b.to_numpy(dot))


def _backend_var(backend: "Backend", arr: "Array") -> float:
    """Compute variance using backend ops."""
    b = backend
    arr_flat = b.reshape(arr, (-1,))
    mean_val = b.mean(arr_flat)
    centered = arr_flat - mean_val
    var = b.mean(centered * centered)
    b.eval(var)
    return float(b.to_numpy(var))


def _backend_std(backend: "Backend", arr: "Array") -> float:
    """Compute standard deviation using backend ops."""
    return math.sqrt(_backend_var(backend, arr))


def _backend_clip(backend: "Backend", arr: "Array", min_val: float, max_val: float) -> "Array":
    """Clip array values to [min_val, max_val] using backend ops."""
    b = backend
    result = b.maximum(arr, b.full(arr.shape, min_val))
    result = b.minimum(result, b.full(result.shape, max_val))
    b.eval(result)
    return result


def _scalar_isnan(x: float) -> bool:
    """Check if a scalar Python float is NaN."""
    return x != x


def _scalar_isinf(x: float) -> bool:
    """Check if a scalar Python float is infinite."""
    return abs(x) > 1e38


# =============================================================================
def get_spatial_anchors_by_axis(axis: SpatialAxis) -> list[SpatialConcept]:
    """Get anchors that primarily vary along a given axis."""
    anchors = SpatialConceptInventory.all_concepts()
    if axis == SpatialAxis.Y_VERTICAL:
        return [
            a
            for a in anchors
            if a.category in (SpatialCategory.VERTICAL, SpatialCategory.MASS)
        ]
    if axis == SpatialAxis.X_LATERAL:
        return [a for a in anchors if a.category == SpatialCategory.LATERAL]
    if axis == SpatialAxis.Z_DEPTH:
        return [a for a in anchors if a.category == SpatialCategory.DEPTH]
    return anchors


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
        anchors: list[SpatialConcept] | None = None,
    ) -> EuclideanConsistencyResult:
        """
        Analyze Euclidean consistency of spatial anchor representations.

        Args:
            anchor_activations: Map from anchor name to activation vector
            anchors: Spatial anchors (uses SpatialConceptInventory if None)

        Returns:
            EuclideanConsistencyResult with consistency metrics
        """
        b = self._backend
        anchors = anchors or SpatialConceptInventory.all_concepts()

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

        Uses k-NN graph shortest paths to compute true manifold distances.
        In high-dimensional curved spaces, Euclidean distance is incorrect.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        b = self._backend
        n = activations.shape[0] if hasattr(activations, "shape") else len(activations)

        # Use geodesic distance matrix for true manifold distances
        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(activations, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)

        # Ensure float32 for numerical stability
        geo_dist = b.astype(geo_dist, "float32")

        # Handle any NaN/inf from geodesic computation using backend ops
        geo_dist = _backend_nan_to_num(b, geo_dist, nan_val=0.0, posinf_val=1e10, neginf_val=0.0)

        return geo_dist

    def _estimate_intrinsic_dimension(self, dist_matrix: "Array") -> float:
        """Estimate intrinsic dimension using MDS eigenvalue decay.

        Uses backend ops for all computation - no numpy.
        """
        b = self._backend
        n = dist_matrix.shape[0]

        # Ensure backend array
        dist_matrix = b.array(dist_matrix)
        dist_matrix = b.astype(dist_matrix, "float32")

        # Handle NaN/inf values using backend ops
        dist_matrix = _backend_nan_to_num(b, dist_matrix, nan_val=0.0, posinf_val=1e10, neginf_val=-1e10)

        # Double centering for MDS: B = -0.5 * H @ D² @ H
        # where H = I - (1/n) * 11^T (centering matrix)
        ones = b.ones((n, n))
        eye = b.eye(n)
        H = eye - ones / float(n)
        b.eval(H)

        dist_sq = dist_matrix * dist_matrix
        B = b.matmul(H, b.matmul(dist_sq, H)) * -0.5
        b.eval(B)

        # Handle numerical issues in B matrix
        B = _backend_nan_to_num(b, B, nan_val=0.0, posinf_val=1e10, neginf_val=-1e10)

        # Eigendecomposition using backend eigh (for symmetric matrices)
        try:
            eigenvalues, _ = b.eigh(B)
            b.eval(eigenvalues)

            # Sort descending (eigh returns ascending)
            eigenvalues = b.sort(eigenvalues, axis=-1)
            b.eval(eigenvalues)
            # Reverse to get descending
            eig_list = _safe_to_list(b, eigenvalues)
            eig_list = sorted(eig_list, reverse=True)
        except Exception:
            # Eigendecomposition failed
            return float(min(n, 10))

        # Filter out NaN/negative eigenvalues
        positive_eigs = [e for e in eig_list if e > 0 and e == e]  # e == e filters NaN

        if len(positive_eigs) == 0:
            return float(min(n, 3))

        # Count significant eigenvalues (> 1% of largest)
        threshold = 0.01 * positive_eigs[0] if positive_eigs else 0
        significant = sum(1 for e in positive_eigs if e > threshold)

        return float(significant)

    def _compute_axis_orthogonality(
        self,
        activations: "Array",
        anchors: list[SpatialConcept],
    ) -> dict[str, float]:
        """Compute orthogonality between inferred X, Y, Z axes.

        Uses backend ops for all computation - no numpy.
        """
        b = self._backend

        # Find axis-defining anchor pairs
        def find_axis_vector(pos_anchor: str, neg_anchor: str) -> "Array | None":
            pos_idx = next((i for i, a in enumerate(anchors) if a.name == pos_anchor), None)
            neg_idx = next((i for i, a in enumerate(anchors) if a.name == neg_anchor), None)
            if pos_idx is None or neg_idx is None:
                return None
            # Keep as backend array
            pos_act = activations[pos_idx]
            neg_act = activations[neg_idx]
            diff = pos_act - neg_act
            b.eval(diff)
            return diff

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

            # Use backend helpers for norm and dot
            norm1 = _backend_vector_norm(b, v1)
            norm2 = _backend_vector_norm(b, v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                results[name] = 0.0
                return

            dot = _backend_vector_dot(b, v1, v2)
            cos = dot / (norm1 * norm2)

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
            meas_flat = []
            exp_flat = []
            for vp in measured:
                meas_flat.extend(measured[vp])
                exp_flat.extend(expected[vp])

            # Use backend for correlation
            meas_arr = b.array(meas_flat)
            exp_arr = b.array(exp_flat)
            b.eval(meas_arr, exp_arr)

            correlation = _backend_corrcoef(b, meas_arr, exp_arr)
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
        # Truncate gravity_direction for display (full vector is 2048+ dims)
        grav_dir_summary = None
        if self.gravity_direction is not None:
            grav_vec = self.gravity_direction
            # Handle both list and array types
            if hasattr(grav_vec, "flatten"):
                flat_vec = grav_vec.flatten()
            else:
                flat_vec = grav_vec  # Already a list
            # Compute norm using sum of squares
            norm_sq = sum(float(x) ** 2 for x in flat_vec)
            grav_norm = math.sqrt(norm_sq)
            top_5 = [float(x) for x in flat_vec[:5]]
            grav_dir_summary = {
                "norm": grav_norm,
                "top_5_dims": top_5,
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
        ceiling_act = None
        floor_act = None
        for anchor in available:
            if anchor.name in ("ceiling", "sky"):
                raw_act = anchor_activations[anchor.name]
                # Clip to prevent overflow using backend
                ceiling_act = _backend_clip(b, raw_act, -1e10, 1e10)
            if anchor.name in ("floor", "ground"):
                raw_act = anchor_activations[anchor.name]
                floor_act = _backend_clip(b, raw_act, -1e10, 1e10)

        gravity_dir = None
        gravity_dir_array = None  # Keep backend array for dot products
        if ceiling_act is not None and floor_act is not None:
            gravity_dir_array = floor_act - ceiling_act
            norm = _backend_vector_norm(b, gravity_dir_array)
            if norm > 1e-6 and not _scalar_isnan(norm) and not _scalar_isinf(norm):
                # Normalize on backend
                gravity_dir_array = gravity_dir_array / norm
                b.eval(gravity_dir_array)
                # Also keep a list version for the result
                gravity_dir = _safe_to_list(b, gravity_dir_array)
            else:
                gravity_dir_array = None

        # Analyze mass-position correlation
        mass_positions = []
        for anchor in available:
            raw_act = anchor_activations[anchor.name]
            act = _backend_clip(b, raw_act, -1e10, 1e10)

            # Project onto gravity axis
            if gravity_dir_array is not None:
                position = _backend_vector_dot(b, act, gravity_dir_array)
            else:
                # Use 2nd dim as proxy
                act_list = _safe_to_list(b, act)
                position = act_list[1] if len(act_list) > 1 else 0.0

            # Skip invalid positions
            if _scalar_isnan(position) or _scalar_isinf(position):
                continue

            # Get expected Y position (negative = down = heavy)
            expected_mass = -anchor.expected_y  # Lower Y = heavier

            mass_positions.append((expected_mass, float(position)))

        # Compute correlation using backend
        if len(mass_positions) >= 3:
            masses_list = [mp[0] for mp in mass_positions]
            positions_list = [mp[1] for mp in mass_positions]
            masses_arr = b.array(masses_list)
            positions_arr = b.array(positions_list)

            std_m = _backend_std(b, masses_arr)
            std_p = _backend_std(b, positions_arr)

            if std_m > 1e-6 and std_p > 1e-6 and not _scalar_isnan(std_p) and not _scalar_isinf(std_p):
                mass_correlation = _backend_corrcoef(b, masses_arr, positions_arr)
                if _scalar_isnan(mass_correlation):
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
        anchors: list[SpatialConcept],
    ) -> float:
        """Compute gravity correlation for a single layer."""
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

        masses_arr = b.array([p[0] for p in pairs])
        positions_arr = b.array([p[1] for p in pairs])

        std_m = _backend_std(b, masses_arr)
        std_p = _backend_std(b, positions_arr)

        if std_m > 1e-6 and std_p > 1e-6:
            return _backend_corrcoef(b, masses_arr, positions_arr)
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
        anchors: list[SpatialConcept] | None = None,
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
            anchors = [
                a
                for a in SpatialConceptInventory.all_concepts()
                if a.category in (SpatialCategory.MASS, SpatialCategory.FURNITURE)
            ]

        available = [a for a in anchors if a.name in anchor_activations]

        if len(available) < 2:
            return VolumetricDensityResult(
                anchor_densities={},
                density_mass_correlation=0.0,
                perspective_attenuation=0.0,
                inverse_square_compliance=0.0,
            )

        # Compute "density" as activation concentration (L2 norm / variance)
        densities = {}
        for anchor in available:
            raw_act = anchor_activations[anchor.name]
            # Clip to prevent overflow using backend
            act = _backend_clip(b, raw_act, -1e10, 1e10)

            norm = _backend_vector_norm(b, act)
            var = _backend_var(b, act)

            # Density metric: higher norm and lower variance = more concentrated
            if var > 1e-6 and not _scalar_isnan(var) and not _scalar_isinf(var):
                density = norm / math.sqrt(var)
            else:
                density = norm

            # Handle NaN/inf
            if _scalar_isnan(density) or _scalar_isinf(density):
                density = 0.0

            densities[anchor.name] = float(density)

        # Correlate density with expected "mass" (negative Y position)
        density_mass_pairs = []
        for anchor in available:
            if anchor.name in densities:
                expected_mass = -anchor.expected_y  # Lower Y = heavier
                density_mass_pairs.append((expected_mass, densities[anchor.name]))

        if len(density_mass_pairs) >= 3:
            masses_arr = b.array([p[0] for p in density_mass_pairs])
            dens_arr = b.array([p[1] for p in density_mass_pairs])
            std_m = _backend_std(b, masses_arr)
            std_d = _backend_std(b, dens_arr)
            if std_m > 1e-6 and std_d > 1e-6:
                density_mass_corr = _backend_corrcoef(b, masses_arr, dens_arr)
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
            depths_arr = b.array([p[0] for p in depth_density_pairs])
            dens_arr = b.array([p[1] for p in depth_density_pairs])
            std_depths = _backend_std(b, depths_arr)
            std_dens = _backend_std(b, dens_arr)
            if std_depths > 1e-6 and std_dens > 1e-6:
                perspective_atten = _backend_corrcoef(b, depths_arr, dens_arr)
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
    "SpatialConcept",
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
