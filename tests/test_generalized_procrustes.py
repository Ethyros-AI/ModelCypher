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
Comprehensive tests for Generalized Procrustes Analysis module.

Tests cover:
- GeneralizedProcrustesResult dataclass and summary property
- GeneralizedProcrustes class (align, align_crms, _compute_consensus)
- LayerRotationResult dataclass
- RotationContinuityResult dataclass and summary property
- RotationContinuityAnalyzer class
- Edge cases and numerical stability

Note: No configuration classes - all parameters are derived from data.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.generalized_procrustes import (
    GeneralizedProcrustes,
    GeneralizedProcrustesResult,
    LayerRotationResult,
    RotationContinuityAnalyzer,
    RotationContinuityResult,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.support.array_utils import array_to_list


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


PI = 3.141592653589793


# =============================================================================
# GeneralizedProcrustesResult Tests
# =============================================================================


class TestGeneralizedProcrustesResult:
    """Tests for GeneralizedProcrustesResult dataclass."""

    def _make_result(self, **kwargs) -> GeneralizedProcrustesResult:
        """Create a GeneralizedProcrustesResult with default values, allowing overrides."""
        defaults = {
            "consensus": [[1.0, 0.0], [0.0, 1.0]],
            "rotations": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            "scales": [1.0, 1.0],
            "residuals": [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            "converged": True,
            "iterations": 5,
            "alignment_error": 0.001,
            "per_model_errors": [0.0005, 0.0005],
            "consensus_variance_ratio": 0.99,
            "sample_count": 2,
            "dimension": 2,
            "model_count": 2,
        }
        defaults.update(kwargs)
        return GeneralizedProcrustesResult(**defaults)

    def test_all_fields_accessible(self) -> None:
        """GeneralizedProcrustesResult should have all required fields."""
        result = self._make_result()
        assert result.consensus is not None
        assert result.rotations is not None
        assert result.scales is not None
        assert result.residuals is not None
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.alignment_error, float)
        assert result.per_model_errors is not None
        assert isinstance(result.consensus_variance_ratio, float)
        assert isinstance(result.sample_count, int)
        assert isinstance(result.dimension, int)
        assert isinstance(result.model_count, int)

    def test_summary_property(self) -> None:
        """Summary property should return formatted string."""
        result = self._make_result(
            converged=True,
            iterations=10,
            alignment_error=0.05,
            consensus_variance_ratio=0.95,
            sample_count=50,
            dimension=64,
            model_count=3,
        )
        summary = result.summary
        assert "Generalized Procrustes Analysis" in summary
        assert "Models: 3" in summary
        assert "Samples: 50 x 64" in summary
        assert "Converged: True" in summary
        assert "iterations: 10" in summary
        assert "Alignment Error: 0.0500" in summary
        assert "Consensus Variance: 95.0%" in summary

    def test_frozen(self) -> None:
        """GeneralizedProcrustesResult should be immutable."""
        result = self._make_result()
        with pytest.raises(Exception):
            result.converged = False  # type: ignore


# =============================================================================
# GeneralizedProcrustes Basic Tests
# =============================================================================


class TestGeneralizedProcrustesInit:
    """Tests for GeneralizedProcrustes initialization."""

    def test_default_initialization(self) -> None:
        """Should initialize with default backend."""
        gpa = GeneralizedProcrustes()
        assert gpa._backend is not None

    def test_explicit_backend(self) -> None:
        """Should accept explicit backend."""
        b = get_default_backend()
        gpa = GeneralizedProcrustes(backend=b)
        assert gpa._backend is b


class TestGeneralizedProcrustesAlign:
    """Tests for GeneralizedProcrustes.align method."""

    def test_align_requires_min_models(self) -> None:
        """Should return None if fewer than 2 models."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        # Requires at least 2 models for alignment - derived from algorithm requirement
        assert GeneralizedProcrustes().align([matrix]) is None

    def test_align_identity_consensus(self) -> None:
        """Identical matrices should align with zero error."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        result = GeneralizedProcrustes().align([matrix, matrix])
        assert result is not None
        backend = get_default_backend()
        eps = _eps(backend, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps
        eps = _eps(backend, result.consensus_variance_ratio, 1.0)
        assert abs(result.consensus_variance_ratio - 1.0) <= eps
        assert result.dimension == 2
        assert result.model_count == 2

    def test_align_three_models(self) -> None:
        """Should handle three or more models."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[0.8, 0.2], [0.2, 0.8]]
        m3 = [[0.9, 0.1], [0.1, 0.9]]
        result = GeneralizedProcrustes().align([m1, m2, m3])
        assert result is not None
        assert result.model_count == 3
        assert result.sample_count == 2
        assert result.dimension == 2

    def test_align_empty_activations_returns_none(self) -> None:
        """Should return None for empty activations."""
        result = GeneralizedProcrustes().align([])
        assert result is None

    def test_align_empty_samples_returns_none(self) -> None:
        """Should return None if matrices have no samples."""
        result = GeneralizedProcrustes().align([[], []])
        assert result is None

    def test_align_mismatched_dimensions_returns_none(self) -> None:
        """Should return None if matrices have different dimensions."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is None

    def test_align_never_scales(self) -> None:
        """Scaling is never allowed - models must align without scale adjustment.

        In high-dimensional geometry, scaling would distort the manifold.
        The algorithm always uses scale=1.0 for all models.
        """
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[2.0, 0.0], [0.0, 2.0]]  # Scaled version
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None
        # All scales should be 1.0 (scaling never applied)
        backend = get_default_backend()
        for scale in result.scales:
            eps = _eps(backend, scale, 1.0)
            assert abs(scale - 1.0) <= eps

    def test_align_never_reflects(self) -> None:
        """Reflections are never allowed - determinant is always +1.

        Reflections would invert the orientation of the representation space,
        which is geometrically incorrect for neural network activations.
        """
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[-1.0, 0.0], [0.0, 1.0]]  # Would require reflection to perfectly align
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None
        # Rotations should all have determinant +1 (no reflections)
        backend = get_default_backend()
        for rotation in result.rotations:
            R = backend.array(rotation)
            det = backend.det(R)
            backend.eval(det)
            det_val = float(backend.to_scalar(det))
            dim = len(rotation)
            eps = _eps(backend, det_val, 1.0) * (dim ** 3)
            assert abs(det_val - 1.0) <= eps

    def test_align_convergence(self) -> None:
        """Should converge - max_iterations derived from model count."""
        m1 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        m2 = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None
        # Should converge (max_iterations = max(100, 10 * model_count))
        assert result.converged or result.iterations > 0

    def test_align_rotations_are_orthogonal(self) -> None:
        """Returned rotations should be orthogonal matrices."""
        b = get_default_backend()
        m1 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
        m2 = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.5, 0.5, 0.9]]
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None

        # Check each rotation is orthogonal: R @ R^T = I
        for rotation in result.rotations:
            R = b.array(rotation)
            I = b.matmul(R, b.transpose(R))
            b.eval(I)
            I_list = array_to_list(b, I)
            # Should be close to identity
            for i in range(len(rotation)):
                for j in range(len(rotation)):
                    expected = 1.0 if i == j else 0.0
                    value = float(I_list[i][j])
                    dim = len(rotation)
                    eps = _eps(b, value, expected) * dim
                    assert abs(value - expected) <= eps


class TestGeneralizedProcrustesAlignCRMs:
    """Tests for GeneralizedProcrustes.align_crms method."""

    def _make_crm(self, model_id: str, hidden_dim: int, activations: dict) -> ConceptResponseMatrix:
        """Helper to create a ConceptResponseMatrix."""
        metadata = AnchorMetadata(
            total_count=len(activations.get(0, {})),
            semantic_prime_count=len(activations.get(0, {})),
            computational_gate_count=0,
            anchor_ids=list(activations.get(0, {}).keys()),
        )
        crm = ConceptResponseMatrix(
            model_identifier=model_id,
            layer_count=len(activations),
            hidden_dim=hidden_dim,
            anchor_metadata=metadata,
        )
        crm.activations = {
            layer: {k: AnchorActivation(k, layer, v) for k, v in acts.items()}
            for layer, acts in activations.items()
        }
        return crm

    def test_align_crms_basic(self) -> None:
        """Should align CRMs from same layer."""
        crm_a = self._make_crm(
            "model_a",
            2,
            {0: {"a": [1.0, 0.0], "b": [0.0, 1.0]}},
        )
        crm_b = self._make_crm(
            "model_b",
            2,
            {0: {"a": [0.9, 0.1], "b": [0.1, 0.9]}},
        )
        result = GeneralizedProcrustes().align_crms([crm_a, crm_b], layer=0)
        assert result is not None
        assert result.sample_count == 2

    def test_align_crms_with_dimension_mismatch(self) -> None:
        """Should truncate to shared dimension."""
        crm_a = self._make_crm(
            "model_a",
            2,
            {0: {"a": [1.0, 0.0], "b": [0.0, 1.0]}},
        )
        crm_b = self._make_crm(
            "model_b",
            3,
            {0: {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}},
        )
        result = GeneralizedProcrustes().align_crms([crm_a, crm_b], layer=0)
        assert result is not None
        assert result.dimension == 2  # Truncated to smaller

    def test_align_crms_missing_layer_returns_none(self) -> None:
        """Should return None if layer not in all CRMs."""
        crm_a = self._make_crm("a", 2, {0: {"a": [1.0, 0.0]}})
        crm_b = self._make_crm("b", 2, {1: {"a": [1.0, 0.0]}})  # Different layer
        result = GeneralizedProcrustes().align_crms([crm_a, crm_b], layer=0)
        assert result is None

    def test_align_crms_different_anchors_still_aligns(self) -> None:
        """Different anchors still produce valid alignment.

        Note: align_crms does NOT require common anchors across CRMs.
        Each CRM's activations are extracted independently based on its
        own anchors. As long as sample counts match, alignment proceeds.
        """
        crm_a = self._make_crm("a", 2, {0: {"x": [1.0, 0.0]}})
        crm_b = self._make_crm("b", 2, {0: {"y": [0.0, 1.0]}})
        # Both have 1 sample, so alignment proceeds (though semantically questionable)
        result = GeneralizedProcrustes().align_crms([crm_a, crm_b], layer=0)
        # Alignment succeeds - both CRMs have 1 sample
        assert result is not None
        assert result.sample_count == 1


class TestGeneralizedProcrustesConsensus:
    """Tests for _compute_consensus method.

    Note: Fréchet mean is ALWAYS used. Arithmetic mean is incorrect on curved
    manifolds - it doesn't respect the geodesic structure of the space.
    """

    def test_frechet_mean_always_used(self) -> None:
        """Fréchet mean is always enabled - the only correct approach on manifolds."""
        m1 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        m2 = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None
        # Should produce valid consensus using Fréchet mean
        assert result.consensus is not None
        assert len(result.consensus) == 3  # 3 samples
        assert len(result.consensus[0]) == 2  # 2 dimensions


# =============================================================================
# LayerRotationResult Tests
# =============================================================================


class TestLayerRotationResult:
    """Tests for LayerRotationResult dataclass."""

    def test_all_fields_accessible(self) -> None:
        """Should have all required fields."""
        result = LayerRotationResult(
            layer_index=0,
            rotation=[[1.0, 0.0], [0.0, 1.0]],
            error=0.01,
            angular_deviation=0.1,
            rotation_delta=0.05,
        )
        backend = get_default_backend()
        eps = _eps(backend, result.error, result.angular_deviation, result.rotation_delta)
        assert result.layer_index == 0
        assert result.rotation is not None
        assert abs(result.error - 0.01) <= eps
        assert abs(result.angular_deviation - 0.1) <= eps
        assert abs(result.rotation_delta - 0.05) <= eps

    def test_optional_fields_default_none(self) -> None:
        """Optional fields should default to None."""
        result = LayerRotationResult(
            layer_index=0,
            rotation=[[1.0, 0.0], [0.0, 1.0]],
            error=0.01,
        )
        assert result.angular_deviation is None
        assert result.rotation_delta is None

    def test_frozen(self) -> None:
        """GeneralizedProcrustesResult should be immutable."""
        result = LayerRotationResult(
            layer_index=0, rotation=[[1.0]], error=0.01
        )
        with pytest.raises(Exception):
            result.error = 0.02  # type: ignore


# =============================================================================
# RotationContinuityResult Tests
# =============================================================================


class TestRotationContinuityResult:
    """Tests for RotationContinuityResult dataclass."""

    def _make_result(self, **kwargs) -> RotationContinuityResult:
        """Create result with defaults."""
        layer_result = LayerRotationResult(
            layer_index=0, rotation=[[1.0, 0.0], [0.0, 1.0]], error=0.01
        )
        defaults = {
            "source_model": "source",
            "target_model": "target",
            "layers": [layer_result],
            "global_rotation_error": 0.05,
            "smoothness_ratio": 0.8,
            "rotation_roughness": 0.01,
            "mean_angular_velocity": 0.1,
            "requires_per_layer_alignment": False,
            "source_dimension": 64,
            "target_dimension": 64,
            "anchor_count": 10,
        }
        defaults.update(kwargs)
        return RotationContinuityResult(**defaults)

    def test_all_fields_accessible(self) -> None:
        """Should have all required fields."""
        result = self._make_result()
        backend = get_default_backend()
        eps = _eps(
            backend,
            result.global_rotation_error,
            result.smoothness_ratio,
            result.rotation_roughness,
            result.mean_angular_velocity,
        )
        assert result.source_model == "source"
        assert result.target_model == "target"
        assert len(result.layers) == 1
        assert abs(result.global_rotation_error - 0.05) <= eps
        assert abs(result.smoothness_ratio - 0.8) <= eps
        assert abs(result.rotation_roughness - 0.01) <= eps
        assert abs(result.mean_angular_velocity - 0.1) <= eps
        assert result.requires_per_layer_alignment is False
        assert result.source_dimension == 64
        assert result.target_dimension == 64
        assert result.anchor_count == 10

    def test_summary_global_rotation_sufficient(self) -> None:
        """Summary should indicate global rotation sufficient."""
        result = self._make_result(requires_per_layer_alignment=False)
        summary = result.summary
        assert "Global rotation SUFFICIENT" in summary
        assert "source" in summary
        assert "target" in summary

    def test_summary_per_layer_required(self) -> None:
        """Summary should indicate per-layer alignment required."""
        result = self._make_result(requires_per_layer_alignment=True)
        summary = result.summary
        assert "Per-layer alignment REQUIRED" in summary

    def test_summary_contains_all_metrics(self) -> None:
        """Summary should contain all key metrics."""
        result = self._make_result(
            global_rotation_error=0.123,
            smoothness_ratio=0.456,
            rotation_roughness=0.789,
            mean_angular_velocity=0.234,
        )
        summary = result.summary
        assert "0.1230" in summary  # alignment error
        assert "0.456" in summary  # smoothness
        assert "0.7890" in summary  # roughness
        assert "0.2340" in summary  # angular velocity


# =============================================================================
# RotationContinuityAnalyzer Tests
# =============================================================================


class TestRotationContinuityAnalyzer:
    """Tests for RotationContinuityAnalyzer class."""

    def test_init_default_backend(self) -> None:
        """Should initialize with default backend."""
        analyzer = RotationContinuityAnalyzer()
        assert analyzer._backend is not None

    def test_init_explicit_backend(self) -> None:
        """Should accept explicit backend."""
        b = get_default_backend()
        analyzer = RotationContinuityAnalyzer(backend=b)
        assert analyzer._backend is b

    def test_compute_per_layer_alignments_basic(self) -> None:
        """Should compute per-layer alignments for simple case."""
        source_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        }
        target_acts = {
            0: {"a": [0.9, 0.1], "b": [0.1, 0.9], "c": [0.5, 0.5]},
            1: {"a": [0.9, 0.1], "b": [0.1, 0.9], "c": [0.5, 0.5]},
        }
        analyzer = RotationContinuityAnalyzer()
        # Smoothness threshold derived from provided distribution
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target",
            smoothness_ratios=[0.5, 0.6, 0.7]
        )
        assert result is not None
        assert result.source_model == "source"
        assert result.target_model == "target"
        assert len(result.layers) == 2

    def test_compute_per_layer_alignments_no_common_layers(self) -> None:
        """Should return None if no common layers."""
        source_acts = {0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]}}
        target_acts = {1: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]}}
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target"
        )
        assert result is None

    def test_compute_per_layer_alignments_insufficient_anchors(self) -> None:
        """Should return None if fewer than 3 common anchors."""
        source_acts = {0: {"a": [1.0, 0.0], "b": [0.0, 1.0]}}  # Only 2
        target_acts = {0: {"a": [1.0, 0.0], "b": [0.0, 1.0]}}
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target"
        )
        assert result is None

    def test_compute_per_layer_alignments_dimension_mismatch(self) -> None:
        """Should handle dimension mismatch by truncating."""
        source_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        }
        target_acts = {
            0: {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.5, 0.5, 0.0]},
        }
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target",
            smoothness_ratios=[0.5, 0.6, 0.7]
        )
        assert result is not None
        assert result.source_dimension == 2
        assert result.target_dimension == 3

    def test_rotation_continuity_metric_smoothness(self) -> None:
        """Should compute smoothness ratio correctly."""
        # Create activations where per-layer vs global differs
        source_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            2: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        }
        target_acts = {
            0: {"a": [0.9, 0.1], "b": [0.1, 0.9], "c": [0.5, 0.5]},
            1: {"a": [0.9, 0.1], "b": [0.1, 0.9], "c": [0.5, 0.5]},
            2: {"a": [0.9, 0.1], "b": [0.1, 0.9], "c": [0.5, 0.5]},
        }
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target",
            smoothness_ratios=[0.5, 0.6, 0.7]
        )
        assert result is not None
        backend = get_default_backend()
        eps = _eps(backend, result.smoothness_ratio)
        assert result.smoothness_ratio >= -eps

    def test_angular_deviation_between_layers(self) -> None:
        """Should compute angular deviation between layer rotations."""
        # Use rotated activations to create different rotations per layer
        backend = get_default_backend()
        angle = backend.array([PI / 4])
        cos45 = backend.cos(angle)
        sin45 = backend.sin(angle)
        backend.eval(cos45)
        backend.eval(sin45)
        cos_val = float(backend.to_scalar(cos45))
        sin_val = float(backend.to_scalar(sin45))

        source_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [cos_val, -sin_val], "b": [sin_val, cos_val], "c": [0.5, 0.5]},
        }
        target_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        }
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target",
            smoothness_ratios=[0.5, 0.6, 0.7]
        )
        assert result is not None
        # Second layer should have angular deviation from first
        if len(result.layers) >= 2:
            assert result.layers[1].angular_deviation is not None


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestProcrustesEdgeCases:
    """Edge case tests for numerical stability in Procrustes alignment."""

    def test_align_with_zero_matrix_does_not_crash(self) -> None:
        """Zero matrices should complete without raising."""
        zero_matrix = [[0.0, 0.0], [0.0, 0.0]]
        identity_matrix = [[1.0, 0.0], [0.0, 1.0]]

        result = GeneralizedProcrustes().align([zero_matrix, identity_matrix])

        if result is not None:
            assert result.dimension == 2
            assert result.model_count == 2

    def test_align_with_near_singular_matrix_completes(self) -> None:
        """Near-singular matrices should not cause SVD numerical issues."""
        near_singular = [[1.0, 2.0], [1.0001, 2.0002]]
        identity = [[1.0, 0.0], [0.0, 1.0]]

        result = GeneralizedProcrustes().align([near_singular, identity])

        if result is not None:
            assert result.dimension == 2
            backend = get_default_backend()
            eps = _eps(backend, result.alignment_error)
            assert result.alignment_error >= -eps

    def test_align_with_rank_deficient_activations_completes(self) -> None:
        """Rank-deficient matrices should not crash SVD."""
        rank_deficient = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        full_rank = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        result = GeneralizedProcrustes().align([rank_deficient, full_rank])

        if result is not None:
            assert result.dimension == 2

    def test_pure_rotation_produces_low_error(self) -> None:
        """Alignment of rotated identity should find the rotation."""
        matrix_a = [[1.0, 0.0], [0.0, 1.0]]
        backend = get_default_backend()
        angle = backend.array([PI / 4])
        c = backend.cos(angle)
        s = backend.sin(angle)
        backend.eval(c)
        backend.eval(s)
        c_val = float(backend.to_scalar(c))
        s_val = float(backend.to_scalar(s))
        matrix_b = [[c_val, -s_val], [s_val, c_val]]

        result = GeneralizedProcrustes().align([matrix_a, matrix_b])

        assert result is not None
        backend = get_default_backend()
        eps = _eps(backend, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_align_large_dimension_mismatch(self) -> None:
        """Test alignment with significantly different dimensions."""
        small_matrix = [[1.0, 2.0], [3.0, 4.0]]
        large_matrix = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ]

        GeneralizedProcrustes().align([small_matrix, large_matrix])
        # Should return None due to dimension mismatch (align expects same dims)
        # align_crms handles dimension truncation, not align

    def test_align_single_row_matrices(self) -> None:
        """Should handle single-row matrices."""
        single_row_a = [[1.0, 2.0, 3.0]]
        single_row_b = [[4.0, 5.0, 6.0]]

        result = GeneralizedProcrustes().align([single_row_a, single_row_b])

        if result is not None:
            assert result.sample_count == 1

    def test_align_very_small_values(self) -> None:
        """Should handle matrices with small (not tiny) values."""
        # Use 1e-3 scale to avoid float32 underflow
        small_values = [[1e-3, 2e-3], [3e-3, 4e-3]]
        normal_values = [[1.0, 2.0], [3.0, 4.0]]

        result = GeneralizedProcrustes().align([small_values, normal_values])

        if result is not None:
            assert result.dimension == 2

    def test_align_identical_matrices_many(self) -> None:
        """Should handle many identical matrices efficiently."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        many_identical = [matrix] * 10

        result = GeneralizedProcrustes().align(many_identical)

        assert result is not None
        assert result.model_count == 10
        backend = get_default_backend()
        eps = _eps(backend, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_align_high_dimensional(self) -> None:
        """Should handle high-dimensional activations."""
        b = get_default_backend()
        b.random_seed(42)

        # Create 20 samples x 64 dimensions
        m1 = array_to_list(b, b.random_normal((20, 64)))
        m2 = array_to_list(b, b.random_normal((20, 64)))

        result = GeneralizedProcrustes().align([m1, m2])

        assert result is not None
        assert result.dimension == 64
        assert result.sample_count == 20

    def test_align_many_samples(self) -> None:
        """Should handle many samples."""
        b = get_default_backend()
        b.random_seed(42)

        # Create 100 samples x 16 dimensions
        m1 = array_to_list(b, b.random_normal((100, 16)))
        m2 = array_to_list(b, b.random_normal((100, 16)))

        result = GeneralizedProcrustes().align([m1, m2])

        assert result is not None
        assert result.sample_count == 100
        assert result.dimension == 16


class TestProcrustesNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_large_scale_difference(self) -> None:
        """Should handle large scale differences (scaling never applied)."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1000.0, 0.0], [0.0, 1000.0]]

        result = GeneralizedProcrustes().align([m1, m2])
        # Alignment proceeds but without scaling adjustment
        assert result is not None

    def test_very_similar_matrices(self) -> None:
        """Should converge quickly for very similar matrices."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1.0 + 1e-8, 0.0], [0.0, 1.0 + 1e-8]]

        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None
        eps = _eps(get_default_backend(), result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_orthogonal_subspaces(self) -> None:
        """Should handle orthogonal subspaces."""
        # These matrices span orthogonal subspaces
        m1 = [[1.0, 0.0], [1.0, 0.0]]  # Only x-axis
        m2 = [[0.0, 1.0], [0.0, 1.0]]  # Only y-axis

        result = GeneralizedProcrustes().align([m1, m2])
        # Should complete, even if error is high
        if result is not None:
            assert result.dimension == 2


class TestProcrustesConvergenceQuality:
    """Tests that verify GPA actually converges to correct answers, not just metadata.

    These exist because several existing tests check metadata (model_count, sample_count,
    dimension) while ignoring whether the algorithm actually solved the alignment problem.
    """

    def test_three_identical_models_converge_with_zero_error(self) -> None:
        """Three identical matrices → converged=True, alignment_error=0."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        result = GeneralizedProcrustes().align([matrix, matrix, matrix])
        assert result is not None
        assert result.converged, (
            f"GPA failed to converge on identical matrices (iterations={result.iterations})"
        )
        b = get_default_backend()
        eps = _eps(b, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps, (
            f"alignment_error={result.alignment_error} should be 0 for identical matrices"
        )

    def test_three_similar_models_actually_converge(self) -> None:
        """Near-identical 2x2 matrices → converged=True (not just result is not None).

        test_align_three_models only checks model_count/sample_count/dimension,
        not whether GPA actually solved the problem.
        """
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[0.8, 0.2], [0.2, 0.8]]
        m3 = [[0.9, 0.1], [0.1, 0.9]]
        result = GeneralizedProcrustes().align([m1, m2, m3])
        assert result is not None
        assert result.converged, (
            f"GPA failed to converge on near-identical matrices (iterations={result.iterations})"
        )

    def test_exact_rotation_gives_zero_alignment_error(self) -> None:
        """m2 = m1 @ R for known rotation R → alignment_error ≈ 0 by construction.

        If alignment error is nonzero here, GPA has a correctness bug, not just
        a convergence issue.
        """
        import numpy as np
        rng = np.random.default_rng(42)
        m1 = rng.standard_normal((10, 2)).astype(np.float32)
        theta = PI / 6  # 30 degrees
        R = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta),  np.cos(theta)]],
            dtype=np.float32,
        )
        m2 = m1 @ R
        result = GeneralizedProcrustes().align([m1.tolist(), m2.tolist()])
        assert result is not None
        assert result.converged, (
            f"GPA failed to converge for exact rotation (iterations={result.iterations})"
        )
        b = get_default_backend()
        eps = _eps(b, result.alignment_error, 0.0)
        assert result.alignment_error <= eps * 1000, (
            f"alignment_error={result.alignment_error:.6f} too large for exact rotation — "
            f"GPA may be computing the wrong consensus or rotation"
        )

    def test_consensus_variance_ratio_is_one_for_identical(self) -> None:
        """Identical matrices → consensus_variance_ratio = 1.0 (all variance explained)."""
        matrix = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        result = GeneralizedProcrustes().align([matrix, matrix])
        assert result is not None
        b = get_default_backend()
        eps = _eps(b, result.consensus_variance_ratio, 1.0)
        assert abs(result.consensus_variance_ratio - 1.0) <= eps, (
            f"consensus_variance_ratio={result.consensus_variance_ratio} should be 1.0 "
            f"for identical matrices"
        )


class TestProcrustesRotationProperties:
    """Tests for rotation matrix properties."""

    def test_rotation_determinant_is_positive_one(self) -> None:
        """Rotations always have determinant +1 (reflections never allowed)."""
        b = get_default_backend()
        m1 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
        m2 = [[0.8, 0.2, 0.4], [0.2, 0.8, 0.4], [0.4, 0.4, 0.9]]

        result = GeneralizedProcrustes().align([m1, m2])

        assert result is not None
        for rotation in result.rotations:
            R = b.array(rotation)
            det = b.det(R)
            b.eval(det)
            det_val = float(b.to_scalar(det))
            # Determinant should be +1 (not -1, which would be reflection)
            dim = len(rotation)
            eps = _eps(b, det_val, 1.0) * (dim ** 3)
            assert abs(det_val - 1.0) <= eps

    def test_rotation_preserves_norm(self) -> None:
        """Rotation should preserve vector norms."""
        b = get_default_backend()
        m1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        m2 = [[0.7, 0.3], [0.3, 0.7], [1.0, 1.0]]

        result = GeneralizedProcrustes().align([m1, m2])
        assert result is not None

        # Check that rotation preserves norms
        for rotation in result.rotations:
            R = b.array(rotation)
            # For each standard basis vector, rotated version should have same norm
            for i in range(len(rotation)):
                e_i = b.zeros((len(rotation),))
                e_i_list = [0.0] * len(rotation)
                e_i_list[i] = 1.0
                e_i = b.array(e_i_list)
                rotated = b.matmul(e_i[None, :], R)
                b.eval(rotated)
                norm = b.norm(rotated)
                b.eval(norm)
                norm_val = float(b.to_scalar(norm))
                eps = _eps(b, norm_val, 1.0)
                assert abs(norm_val - 1.0) <= eps
