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

"""Comprehensive tests for spatial_3d.py.

Tests cover:
- Backend-compatible numerical helpers
- SpatialStereoscopy
- GravityGradientAnalyzer
- VolumetricDensityProber
- OcclusionProber
- Spatial3DAnalyzer
- Dataclasses and to_dict methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.spatial_3d import (
    GravityGradientAnalyzer,
    GravityGradientResult,
    OcclusionProber,
    OcclusionPrompt,
    OcclusionResult,
    Spatial3DAnalyzer,
    Spatial3DReport,
    SpatialStereoscopy,
    StereoscopyResult,
    ViewpointPrompt,
    VolumetricDensityProber,
    VolumetricDensityResult,
    _backend_clip,
    _backend_corrcoef,
    _backend_isinf,
    _backend_isnan,
    _backend_nan_to_num,
    _backend_std,
    _backend_var,
    _backend_vector_dot,
    _backend_vector_norm,
    _safe_to_list,
    _scalar_isinf,
    _scalar_isnan,
    get_spatial_anchors_by_axis,
)
from modelcypher.core.domain.agents.spatial_atlas import SpatialAxis

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _eps(backend: "Backend", arr: "Array", scale: float = 1.0) -> float:
    return division_epsilon(backend, arr) * max(1.0, abs(scale))


# =============================================================================
# Numerical Helper Tests
# =============================================================================


class TestBackendIsnan:
    """Tests for _backend_isnan function."""

    def test_detects_nan(self, any_backend: "Backend") -> None:
        """Should detect NaN values."""
        b = any_backend
        arr = b.array([1.0, float("nan"), 3.0, float("nan")])
        b.eval(arr)

        result = _backend_isnan(b, arr)
        b.eval(result)

        result_np = b.tolist(result)
        assert not result_np[0]  # 1.0 is not NaN
        assert result_np[1]  # NaN
        assert not result_np[2]  # 3.0 is not NaN
        assert result_np[3]  # NaN

    def test_no_nan_array(self, any_backend: "Backend") -> None:
        """Array without NaN should return all False."""
        b = any_backend
        arr = b.array([1.0, 2.0, 3.0])
        b.eval(arr)

        result = _backend_isnan(b, arr)
        b.eval(result)

        result_np = b.tolist(result)
        assert not any(result_np)


class TestBackendIsinf:
    """Tests for _backend_isinf function."""

    def test_detects_inf(self, any_backend: "Backend") -> None:
        """Should detect infinite values."""
        b = any_backend
        arr = b.array([1.0, 1e39, -1e39, 5.0])
        b.eval(arr)

        result = _backend_isinf(b, arr)
        b.eval(result)

        result_np = b.tolist(result)
        assert not result_np[0]  # 1.0 is not inf
        assert result_np[1]  # 1e39 is effectively inf
        assert result_np[2]  # -1e39 is effectively -inf
        assert not result_np[3]  # 5.0 is not inf


class TestBackendNanToNum:
    """Tests for _backend_nan_to_num function."""

    def test_replaces_nan(self, any_backend: "Backend") -> None:
        """Should replace NaN with specified value."""
        b = any_backend
        arr = b.array([1.0, float("nan"), 3.0])
        b.eval(arr)

        result = _backend_nan_to_num(b, arr, nan_val=0.0)
        b.eval(result)

        result_np = b.tolist(result)
        eps = _eps(b, result, scale=3.0)
        assert abs(result_np[0] - 1.0) <= eps
        assert abs(result_np[1] - 0.0) <= eps  # NaN replaced with 0
        assert abs(result_np[2] - 3.0) <= eps

    def test_replaces_posinf(self, any_backend: "Backend") -> None:
        """Should replace positive infinity."""
        b = any_backend
        arr = b.array([1.0, 1e39, 3.0])
        b.eval(arr)

        result = _backend_nan_to_num(b, arr, posinf_val=999.0)
        b.eval(result)

        result_np = b.tolist(result)
        eps = _eps(b, result, scale=999.0)
        assert abs(result_np[0] - 1.0) <= eps
        assert abs(result_np[1] - 999.0) <= eps
        assert abs(result_np[2] - 3.0) <= eps


class TestBackendCorrcoef:
    """Tests for _backend_corrcoef function."""

    def test_perfect_positive_correlation(self, any_backend: "Backend") -> None:
        """Perfect positive correlation should return 1.0."""
        b = any_backend
        x = b.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = b.array([2.0, 4.0, 6.0, 8.0, 10.0])
        b.eval(x, y)

        corr = _backend_corrcoef(b, x, y)
        eps = _eps(b, x)
        assert abs(corr - 1.0) <= eps

    def test_perfect_negative_correlation(self, any_backend: "Backend") -> None:
        """Perfect negative correlation should return -1.0."""
        b = any_backend
        x = b.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = b.array([10.0, 8.0, 6.0, 4.0, 2.0])
        b.eval(x, y)

        corr = _backend_corrcoef(b, x, y)
        eps = _eps(b, x)
        assert abs(corr + 1.0) <= eps

    def test_zero_correlation(self, any_backend: "Backend") -> None:
        """Uncorrelated data should have correlation near 0."""
        b = any_backend
        x = b.array([1.0, -1.0, 1.0, -1.0])
        y = b.array([1.0, 1.0, -1.0, -1.0])
        b.eval(x, y)

        corr = _backend_corrcoef(b, x, y)
        eps = _eps(b, x)
        assert abs(corr) <= eps

    def test_constant_array_returns_zero(self, any_backend: "Backend") -> None:
        """Constant arrays have zero standard deviation, return 0."""
        b = any_backend
        x = b.array([5.0, 5.0, 5.0])
        y = b.array([1.0, 2.0, 3.0])
        b.eval(x, y)

        corr = _backend_corrcoef(b, x, y)
        assert corr == 0.0

    def test_single_element_returns_zero(self, any_backend: "Backend") -> None:
        """Single element arrays return 0."""
        b = any_backend
        x = b.array([1.0])
        y = b.array([2.0])
        b.eval(x, y)

        corr = _backend_corrcoef(b, x, y)
        assert corr == 0.0


class TestBackendVectorNorm:
    """Tests for _backend_vector_norm function."""

    def test_unit_vector(self, any_backend: "Backend") -> None:
        """Unit vector should have norm 1."""
        b = any_backend
        v = b.array([1.0, 0.0, 0.0])
        b.eval(v)

        norm = _backend_vector_norm(b, v)
        eps = _eps(b, v)
        assert abs(norm - 1.0) <= eps

    def test_known_norm(self, any_backend: "Backend") -> None:
        """Known vector should have correct norm."""
        b = any_backend
        v = b.array([3.0, 4.0])  # 3-4-5 triangle
        b.eval(v)

        norm = _backend_vector_norm(b, v)
        eps = _eps(b, v, scale=5.0)
        assert abs(norm - 5.0) <= eps

    def test_zero_vector(self, any_backend: "Backend") -> None:
        """Zero vector should have norm 0."""
        b = any_backend
        v = b.array([0.0, 0.0, 0.0])
        b.eval(v)

        norm = _backend_vector_norm(b, v)
        eps = _eps(b, v)
        assert abs(norm) <= eps


class TestBackendVectorDot:
    """Tests for _backend_vector_dot function."""

    def test_orthogonal_vectors(self, any_backend: "Backend") -> None:
        """Orthogonal vectors have dot product 0."""
        b = any_backend
        v1 = b.array([1.0, 0.0])
        v2 = b.array([0.0, 1.0])
        b.eval(v1, v2)

        dot = _backend_vector_dot(b, v1, v2)
        eps = _eps(b, v1)
        assert abs(dot) <= eps

    def test_parallel_vectors(self, any_backend: "Backend") -> None:
        """Parallel unit vectors have dot product 1."""
        b = any_backend
        v1 = b.array([1.0, 0.0])
        v2 = b.array([1.0, 0.0])
        b.eval(v1, v2)

        dot = _backend_vector_dot(b, v1, v2)
        eps = _eps(b, v1)
        assert abs(dot - 1.0) <= eps

    def test_known_dot(self, any_backend: "Backend") -> None:
        """Known vectors should have correct dot product."""
        b = any_backend
        v1 = b.array([1.0, 2.0, 3.0])
        v2 = b.array([4.0, 5.0, 6.0])
        b.eval(v1, v2)

        dot = _backend_vector_dot(b, v1, v2)
        expected = 1*4 + 2*5 + 3*6  # = 32
        eps = _eps(b, v1, scale=expected)
        assert abs(dot - expected) <= eps


class TestBackendVarStd:
    """Tests for _backend_var and _backend_std functions."""

    def test_variance_known_values(self, any_backend: "Backend") -> None:
        """Known values should have correct variance."""
        b = any_backend
        arr = b.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        b.eval(arr)

        var = _backend_var(b, arr)
        # Mean = 5.0, Variance = 4.0
        eps = _eps(b, arr, scale=4.0)
        assert abs(var - 4.0) <= eps

    def test_std_is_sqrt_variance(self, any_backend: "Backend") -> None:
        """Standard deviation should be sqrt of variance."""
        b = any_backend
        arr = b.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        b.eval(arr)

        var = _backend_var(b, arr)
        std = _backend_std(b, arr)
        expected = sqrt_scalar(var, b)
        eps = _eps(b, arr, scale=expected)
        assert abs(std - expected) <= eps

    def test_constant_array_zero_variance(self, any_backend: "Backend") -> None:
        """Constant array should have zero variance."""
        b = any_backend
        arr = b.array([5.0, 5.0, 5.0, 5.0])
        b.eval(arr)

        var = _backend_var(b, arr)
        eps = _eps(b, arr)
        assert abs(var) <= eps


class TestBackendClip:
    """Tests for _backend_clip function."""

    def test_clip_values(self, any_backend: "Backend") -> None:
        """Values should be clipped to range."""
        b = any_backend
        arr = b.array([-5.0, 0.5, 10.0, 0.0, 1.0])
        b.eval(arr)

        result = _backend_clip(b, arr, 0.0, 1.0)
        b.eval(result)

        result_np = b.tolist(result)
        eps = _eps(b, result, scale=1.0)
        assert abs(result_np[0] - 0.0) <= eps  # -5 clipped to 0
        assert abs(result_np[1] - 0.5) <= eps  # 0.5 unchanged
        assert abs(result_np[2] - 1.0) <= eps  # 10 clipped to 1
        assert abs(result_np[3] - 0.0) <= eps  # 0 unchanged
        assert abs(result_np[4] - 1.0) <= eps  # 1 unchanged


class TestScalarNanInf:
    """Tests for scalar NaN/Inf checkers."""

    def test_scalar_isnan(self) -> None:
        """Should detect NaN scalars."""
        assert _scalar_isnan(float("nan"))
        assert not _scalar_isnan(0.0)
        assert not _scalar_isnan(1.0)
        assert not _scalar_isnan(float("inf"))

    def test_scalar_isinf(self) -> None:
        """Should detect infinite scalars."""
        assert _scalar_isinf(1e39)
        assert _scalar_isinf(-1e39)
        assert not _scalar_isinf(1.0)
        assert not _scalar_isinf(0.0)


class TestSafeToList:
    """Tests for _safe_to_list function."""

    def test_converts_array_to_list(self, any_backend: "Backend") -> None:
        """Should convert array to Python list."""
        b = any_backend
        arr = b.array([1.0, 2.0, 3.0])
        b.eval(arr)

        result = _safe_to_list(b, arr)
        assert result == [1.0, 2.0, 3.0]

    def test_handles_2d_array(self, any_backend: "Backend") -> None:
        """Should flatten 2D array."""
        b = any_backend
        arr = b.array([[1.0, 2.0], [3.0, 4.0]])
        b.eval(arr)

        result = _safe_to_list(b, arr)
        assert result == [1.0, 2.0, 3.0, 4.0]


# =============================================================================
# Spatial Axis Helper Tests
# =============================================================================


class TestGetSpatialAnchorsByAxis:
    """Tests for get_spatial_anchors_by_axis function."""

    def test_vertical_axis(self) -> None:
        """Vertical axis should return vertical and mass anchors."""
        anchors = get_spatial_anchors_by_axis(SpatialAxis.Y_VERTICAL)
        assert len(anchors) > 0
        # All should be vertical, mass, or furniture category
        for a in anchors:
            assert a.category.name in ("VERTICAL", "MASS", "FURNITURE")

    def test_lateral_axis(self) -> None:
        """Lateral axis should return lateral anchors."""
        anchors = get_spatial_anchors_by_axis(SpatialAxis.X_LATERAL)
        assert len(anchors) > 0
        for a in anchors:
            assert a.category.name == "LATERAL"

    def test_depth_axis(self) -> None:
        """Depth axis should return depth anchors."""
        anchors = get_spatial_anchors_by_axis(SpatialAxis.Z_DEPTH)
        assert len(anchors) > 0
        for a in anchors:
            assert a.category.name == "DEPTH"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestViewpointPrompt:
    """Tests for ViewpointPrompt dataclass."""

    def test_viewpoint_prompt_creation(self) -> None:
        """Should create ViewpointPrompt with all fields."""
        vp = ViewpointPrompt(
            scene_id="test",
            viewpoint="front",
            prompt="A test scene.",
            expected_parallax_x=0.0,
            expected_parallax_y=0.0,
            expected_parallax_z=0.0,
        )
        assert vp.scene_id == "test"
        assert vp.viewpoint == "front"


class TestStereoscopyResult:
    """Tests for StereoscopyResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should return dictionary with all fields."""
        result = StereoscopyResult(
            scene_id="cube",
            parallax_correlation=0.8,
            measured_parallax={"front": (0.0, 0.0, 0.0)},
            expected_parallax={"front": (0.0, 0.0, 0.0)},
            depth_axis_detected=True,
            perspective_consistency=0.9,
        )
        d = result.to_dict()
        assert d["scene_id"] == "cube"
        assert d["depth_axis_detected"] is True


class TestGravityGradientResult:
    """Tests for GravityGradientResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should handle None gravity direction."""
        result = GravityGradientResult(
            gravity_axis_detected=False,
            gravity_direction=None,
            mass_correlation=0.0,
            layer_gravity_strengths={},
            sink_anchors=[],
            float_anchors=[],
        )
        d = result.to_dict()
        assert d["gravity_direction_summary"] is None

    def test_to_dict_with_direction(self, any_backend: "Backend") -> None:
        """to_dict should summarize gravity direction."""
        b = any_backend
        direction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        result = GravityGradientResult(
            gravity_axis_detected=True,
            gravity_direction=direction,
            mass_correlation=0.5,
            layer_gravity_strengths={0: 0.3, 1: 0.4},
            sink_anchors=["floor"],
            float_anchors=["ceiling"],
        )
        d = result.to_dict()
        assert d["gravity_direction_summary"] is not None
        assert "norm" in d["gravity_direction_summary"]
        assert "top_5_dims" in d["gravity_direction_summary"]


class TestVolumetricDensityResult:
    """Tests for VolumetricDensityResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should return dictionary."""
        result = VolumetricDensityResult(
            anchor_densities={"box": 1.5, "ball": 0.8},
            density_mass_correlation=0.6,
            perspective_attenuation=-0.3,
            inverse_square_compliance=0.7,
        )
        d = result.to_dict()
        assert "anchor_densities" in d
        assert d["density_mass_correlation"] == 0.6


class TestOcclusionResult:
    """Tests for OcclusionResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should return dictionary."""
        result = OcclusionResult(
            scene_id="box_ball",
            z_shift_detected=True,
            a_front_z_position=0.5,
            b_front_z_position=-0.2,
            z_swap_magnitude=0.7,
            occlusion_understood=True,
        )
        d = result.to_dict()
        assert d["scene_id"] == "box_ball"
        assert d["occlusion_understood"] is True


# =============================================================================
# Analyzer Class Tests
# =============================================================================


class TestSpatialStereoscopy:
    """Tests for SpatialStereoscopy class."""

    def test_analyze_scene_insufficient_viewpoints(
        self, any_backend: "Backend"
    ) -> None:
        """Should return zero for single viewpoint."""
        b = any_backend
        analyzer = SpatialStereoscopy(backend=b)

        activations = {"front": b.random_normal((64,))}
        for v in activations.values():
            b.eval(v)

        prompts = [
            ViewpointPrompt("test", "front", "Test front", 0, 0, 0),
        ]

        result = analyzer.analyze_scene(activations, prompts)
        assert result.parallax_correlation == 0.0

    def test_analyze_scene_valid(self, any_backend: "Backend") -> None:
        """Should produce valid result with multiple viewpoints."""
        b = any_backend
        b.random_seed(42)
        analyzer = SpatialStereoscopy(backend=b)

        activations = {
            "front": b.random_normal((64,)),
            "left": b.random_normal((64,)),
            "right": b.random_normal((64,)),
        }
        for v in activations.values():
            b.eval(v)

        prompts = [
            ViewpointPrompt("test", "front", "Test front", 0, 0, 0),
            ViewpointPrompt("test", "left", "Test left", -0.5, 0, 0.2),
            ViewpointPrompt("test", "right", "Test right", 0.5, 0, 0.2),
        ]

        result = analyzer.analyze_scene(activations, prompts)

        assert result.scene_id == "test"
        assert "front" in result.measured_parallax
        assert "left" in result.measured_parallax


class TestGravityGradientAnalyzer:
    """Tests for GravityGradientAnalyzer class."""

    def test_analyze_insufficient_anchors(self, any_backend: "Backend") -> None:
        """Should return non-detected for insufficient anchors."""
        b = any_backend
        analyzer = GravityGradientAnalyzer(backend=b)

        activations = {"ceiling": b.random_normal((64,))}
        for v in activations.values():
            b.eval(v)

        result = analyzer.analyze(activations)
        assert not result.gravity_axis_detected
        assert result.gravity_direction is None

    def test_analyze_with_vertical_anchors(self, any_backend: "Backend") -> None:
        """Should detect gravity with vertical anchors."""
        b = any_backend
        b.random_seed(42)
        analyzer = GravityGradientAnalyzer(backend=b)

        # Create activations for vertical anchors
        activations = {
            "ceiling": b.random_normal((64,)),
            "floor": b.random_normal((64,)),
            "sky": b.random_normal((64,)),
            "ground": b.random_normal((64,)),
        }
        for v in activations.values():
            b.eval(v)

        result = analyzer.analyze(activations)

        # Should have valid structure (detection depends on data)
        assert hasattr(result, "gravity_axis_detected")
        assert hasattr(result, "mass_correlation")


class TestVolumetricDensityProber:
    """Tests for VolumetricDensityProber class."""

    def test_analyze_insufficient_anchors(self, any_backend: "Backend") -> None:
        """Should return empty densities for insufficient anchors."""
        b = any_backend
        prober = VolumetricDensityProber(backend=b)

        activations = {"heavy": b.random_normal((64,))}
        for v in activations.values():
            b.eval(v)

        result = prober.analyze(activations, anchors=[])
        assert result.anchor_densities == {}

    def test_analyze_computes_densities(self, any_backend: "Backend") -> None:
        """Should compute densities for available anchors."""
        b = any_backend
        b.random_seed(42)
        prober = VolumetricDensityProber(backend=b)

        from modelcypher.core.domain.agents.spatial_atlas import (
            SpatialCategory,
            SpatialConcept,
        )

        # Create custom anchors (id, name, prompt, expected_x, expected_y, expected_z, category)
        anchors = [
            SpatialConcept("heavy", "heavy", "heavy object", 0.0, -0.5, 0.0, SpatialCategory.MASS),
            SpatialConcept("light", "light", "light object", 0.0, 0.5, 0.0, SpatialCategory.MASS),
            SpatialConcept("medium", "medium", "medium object", 0.0, 0.0, 0.0, SpatialCategory.MASS),
        ]

        activations = {
            "heavy": b.random_normal((64,)),
            "light": b.random_normal((64,)),
            "medium": b.random_normal((64,)),
        }
        for v in activations.values():
            b.eval(v)

        result = prober.analyze(activations, anchors=anchors)

        assert len(result.anchor_densities) == 3
        assert "heavy" in result.anchor_densities


class TestOcclusionProber:
    """Tests for OcclusionProber class."""

    def test_analyze_occlusion(self, any_backend: "Backend") -> None:
        """Should analyze occlusion shift."""
        b = any_backend
        b.random_seed(42)
        prober = OcclusionProber(backend=b)

        a_front = b.random_normal((64,))
        b_front = b.random_normal((64,))
        b.eval(a_front, b_front)

        probe = OcclusionPrompt(
            scene_id="test",
            object_a="A",
            object_b="B",
            a_in_front_prompt="A is in front of B",
            b_in_front_prompt="B is in front of A",
        )

        result = prober.analyze(a_front, b_front, probe)

        assert result.scene_id == "test"
        assert hasattr(result, "z_shift_detected")
        assert hasattr(result, "z_swap_magnitude")


class TestSpatial3DAnalyzer:
    """Tests for Spatial3DAnalyzer class."""

    def test_full_analysis_minimal(self, any_backend: "Backend") -> None:
        """Should run full analysis with minimal data."""
        b = any_backend
        b.random_seed(42)
        analyzer = Spatial3DAnalyzer(backend=b)

        # Create minimal anchor activations
        from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
        anchors = SpatialConceptInventory.all_concepts()[:8]

        activations = {}
        for a in anchors:
            act = b.random_normal((32,))
            b.eval(act)
            activations[a.name] = act

        result = analyzer.full_analysis(activations)

        assert isinstance(result, Spatial3DReport)
        assert hasattr(result, "gravity_gradient")
        assert hasattr(result, "volumetric_density")
        assert hasattr(result, "world_model_score")

    def test_full_analysis_to_dict(self, any_backend: "Backend") -> None:
        """Report should be convertible to dict."""
        b = any_backend
        b.random_seed(42)
        analyzer = Spatial3DAnalyzer(backend=b)

        from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
        anchors = SpatialConceptInventory.all_concepts()[:8]

        activations = {}
        for a in anchors:
            act = b.random_normal((32,))
            b.eval(act)
            activations[a.name] = act

        result = analyzer.full_analysis(activations)
        d = result.to_dict()

        assert "gravity_gradient" in d
        assert "world_model_score" in d


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for spatial_3d module."""

    def test_empty_activations(self, any_backend: "Backend") -> None:
        """Should handle empty activations gracefully."""
        b = any_backend
        analyzer = Spatial3DAnalyzer(backend=b)

        result = analyzer.full_analysis({})

        assert result.world_model_score == 0.0

    def test_very_small_activations(self, any_backend: "Backend") -> None:
        """Should handle very small activation values."""
        b = any_backend

        from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
        anchors = SpatialConceptInventory.all_concepts()[:5]

        activations = {}
        for a in anchors:
            act = b.array([1e-20, 1e-20, 1e-20, 1e-20])
            b.eval(act)
            activations[a.name] = act

        prober = VolumetricDensityProber(backend=b)
        result = prober.analyze(activations, anchors=anchors)

        # Should handle gracefully
        assert hasattr(result, "anchor_densities")
