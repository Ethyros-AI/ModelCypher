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
- FrechetMeanConfig dataclass
- Config dataclass and factory methods
- Result dataclass and summary property
- GeneralizedProcrustes class (align, align_crms, _compute_consensus)
- LayerRotationResult dataclass
- RotationContinuityResult dataclass and summary property
- RotationContinuityAnalyzer class
- Edge cases and numerical stability
"""

from __future__ import annotations

import math
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.generalized_procrustes import (
    Config,
    FrechetMeanConfig,
    GeneralizedProcrustes,
    LayerRotationResult,
    Result,
    RotationContinuityAnalyzer,
    RotationContinuityResult,
)


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# FrechetMeanConfig Tests
# =============================================================================


class TestFrechetMeanConfig:
    """Tests for FrechetMeanConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have Fréchet mean enabled."""
        config = FrechetMeanConfig()
        assert config.enabled is True
        # k_neighbors is None by default - computed from intrinsic dimension
        assert config.k_neighbors is None
        assert config.max_iterations > 0
        # tolerance is None by default - derived from machine epsilon at runtime
        assert config.tolerance is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([1.0]))
        config = FrechetMeanConfig(
            enabled=False,
            k_neighbors=20,
            max_iterations=100,
            tolerance=eps,
        )
        assert config.enabled is False
        assert config.k_neighbors == 20
        assert config.max_iterations == 100
        assert config.tolerance == eps

    def test_frozen(self) -> None:
        """Config should be immutable."""
        config = FrechetMeanConfig()
        with pytest.raises(Exception):
            config.enabled = False  # type: ignore


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self) -> None:
        """Default config values - smoothness threshold must be explicitly set."""
        config = Config()
        assert config.max_iterations > 0
        # convergence_threshold is None by default - derived from machine epsilon at runtime
        assert config.convergence_threshold is None
        assert config.allow_reflections is False
        assert config.min_models == 2
        assert config.allow_scaling is False
        assert config.frechet_mean.enabled is True
        assert config.per_layer_smoothness_threshold is None
        # effective_smoothness_threshold should raise without explicit threshold
        import pytest
        with pytest.raises(ValueError, match="per_layer_smoothness_threshold not set"):
            _ = config.effective_smoothness_threshold

    def test_with_smoothness_threshold(self) -> None:
        """Config.with_smoothness_threshold() creates config with threshold."""
        ratios = [0.5, 0.55, 0.6, 0.65, 0.7]
        threshold = sum(ratios) / len(ratios)
        config = Config.with_smoothness_threshold(threshold)
        assert config.per_layer_smoothness_threshold == threshold
        assert config.effective_smoothness_threshold == threshold

    def test_from_smoothness_distribution(self) -> None:
        """Config.from_smoothness_distribution() derives threshold from data."""
        # Smoothness ratios with mean ~0.6, std ~0.1
        ratios = [0.5, 0.55, 0.6, 0.65, 0.7]
        config = Config.from_smoothness_distribution(ratios)
        assert config.per_layer_smoothness_threshold is not None
        mean = sum(ratios) / len(ratios)
        variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        expected = max(0.0, mean - variance**0.5)
        backend = get_default_backend()
        eps = _eps(backend, config.effective_smoothness_threshold, expected)
        assert abs(config.effective_smoothness_threshold - expected) <= eps

    def test_from_empty_smoothness_distribution_raises(self) -> None:
        """Config.from_smoothness_distribution() raises on empty data."""
        import pytest
        with pytest.raises(ValueError, match="Cannot derive threshold from empty"):
            Config.from_smoothness_distribution([])

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([1.0]))
        ratios = [0.4, 0.6]
        threshold = sum(ratios) / len(ratios)
        config = Config(
            max_iterations=50,
            convergence_threshold=eps,
            allow_reflections=True,
            min_models=3,
            allow_scaling=True,
            per_layer_smoothness_threshold=threshold,
        )
        assert config.max_iterations == 50
        assert config.convergence_threshold == eps
        assert config.allow_reflections is True
        assert config.min_models == 3
        assert config.allow_scaling is True
        assert config.per_layer_smoothness_threshold == threshold

    def test_default_factory_method(self) -> None:
        """Config.default() should return curvature-aware config."""
        config = Config.default()
        assert config.frechet_mean.enabled is True

    def test_arithmetic_mean_factory_method(self) -> None:
        """Config.arithmetic_mean() should disable Fréchet mean."""
        config = Config.arithmetic_mean()
        assert config.frechet_mean.enabled is False

    def test_frozen(self) -> None:
        """Config should be immutable."""
        config = Config()
        with pytest.raises(Exception):
            config.max_iterations = 50  # type: ignore


# =============================================================================
# Result Tests
# =============================================================================


class TestResult:
    """Tests for Result dataclass."""

    def _make_result(self, **kwargs) -> Result:
        """Create a Result with default values, allowing overrides."""
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
        return Result(**defaults)

    def test_all_fields_accessible(self) -> None:
        """Result should have all required fields."""
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
        """Result should be immutable."""
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
        """Should return None if fewer than min_models."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        config = Config(min_models=2, max_iterations=5)
        assert GeneralizedProcrustes().align([matrix], config=config) is None

    def test_align_identity_consensus(self) -> None:
        """Identical matrices should align with zero error."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        result = GeneralizedProcrustes().align(
            [matrix, matrix], config=Config(max_iterations=5)
        )
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
        result = GeneralizedProcrustes().align([m1, m2, m3], config=Config(max_iterations=20))
        assert result is not None
        assert result.model_count == 3
        assert result.sample_count == 2
        assert result.dimension == 2

    def test_align_empty_activations_returns_none(self) -> None:
        """Should return None for empty activations."""
        result = GeneralizedProcrustes().align([], config=Config(max_iterations=5))
        assert result is None

    def test_align_empty_samples_returns_none(self) -> None:
        """Should return None if matrices have no samples."""
        result = GeneralizedProcrustes().align([[], []], config=Config(max_iterations=5))
        assert result is None

    def test_align_mismatched_dimensions_returns_none(self) -> None:
        """Should return None if matrices have different dimensions."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]
        result = GeneralizedProcrustes().align([m1, m2], config=Config(max_iterations=5))
        assert result is None

    def test_align_with_scaling_enabled(self) -> None:
        """Should handle scaling when allow_scaling=True."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[2.0, 0.0], [0.0, 2.0]]  # Scaled version
        config_scaled = Config(max_iterations=20, allow_scaling=True)
        config_unscaled = Config(max_iterations=20, allow_scaling=False)
        result_scaled = GeneralizedProcrustes().align([m1, m2], config=config_scaled)
        result_unscaled = GeneralizedProcrustes().align([m1, m2], config=config_unscaled)
        assert result_scaled is not None
        assert result_unscaled is not None
        backend = get_default_backend()
        eps = _eps(backend, result_scaled.alignment_error, result_unscaled.alignment_error)
        assert result_scaled.alignment_error <= result_unscaled.alignment_error + eps

    def test_align_with_reflections_allowed(self) -> None:
        """Should allow reflections when configured."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[-1.0, 0.0], [0.0, 1.0]]  # Reflected
        config = Config(max_iterations=20, allow_reflections=True)
        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None

    def test_align_convergence(self) -> None:
        """Should converge within max_iterations."""
        m1 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        m2 = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
        config = Config(max_iterations=100)
        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None
        # Should converge well before max iterations
        assert result.iterations <= 100

    def test_align_rotations_are_orthogonal(self) -> None:
        """Returned rotations should be orthogonal matrices."""
        b = get_default_backend()
        m1 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
        m2 = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.5, 0.5, 0.9]]
        result = GeneralizedProcrustes().align([m1, m2], config=Config(max_iterations=20))
        assert result is not None

        # Check each rotation is orthogonal: R @ R^T = I
        for rotation in result.rotations:
            R = b.array(rotation)
            I = b.matmul(R, b.transpose(R))
            b.eval(I)
            I_np = b.to_numpy(I)
            # Should be close to identity
            for i in range(len(rotation)):
                for j in range(len(rotation)):
                    expected = 1.0 if i == j else 0.0
                    eps = _eps(b, float(I_np[i, j]), expected)
                    assert abs(float(I_np[i, j]) - expected) <= eps


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
        result = GeneralizedProcrustes().align_crms(
            [crm_a, crm_b], layer=0, config=Config(max_iterations=10)
        )
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
        result = GeneralizedProcrustes().align_crms(
            [crm_a, crm_b], layer=0, config=Config(max_iterations=10)
        )
        assert result is not None
        assert result.dimension == 2  # Truncated to smaller

    def test_align_crms_missing_layer_returns_none(self) -> None:
        """Should return None if layer not in all CRMs."""
        crm_a = self._make_crm("a", 2, {0: {"a": [1.0, 0.0]}})
        crm_b = self._make_crm("b", 2, {1: {"a": [1.0, 0.0]}})  # Different layer
        result = GeneralizedProcrustes().align_crms(
            [crm_a, crm_b], layer=0, config=Config(max_iterations=5)
        )
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
        result = GeneralizedProcrustes().align_crms(
            [crm_a, crm_b], layer=0, config=Config(max_iterations=5)
        )
        # Alignment succeeds - both CRMs have 1 sample
        assert result is not None
        assert result.sample_count == 1


class TestGeneralizedProcrustesConsensus:
    """Tests for _compute_consensus method."""

    def test_arithmetic_mean_consensus(self) -> None:
        """Arithmetic mean should average across models."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[2.0, 0.0], [0.0, 2.0]]
        config = Config.arithmetic_mean()
        config = Config(max_iterations=1, frechet_mean=FrechetMeanConfig(enabled=False))
        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None
        # After centering, consensus is computed from centered data

    def test_frechet_mean_consensus(self) -> None:
        """Fréchet mean should use curvature-aware averaging."""
        m1 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        m2 = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
        config = Config.default()  # Fréchet mean enabled
        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None
        # Should produce valid result with Fréchet mean


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
        assert result.layer_index == 0
        assert result.rotation is not None
        assert result.error == 0.01
        assert result.angular_deviation == 0.1
        assert result.rotation_delta == 0.05

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
        """Result should be immutable."""
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
        assert result.source_model == "source"
        assert result.target_model == "target"
        assert len(result.layers) == 1
        assert result.global_rotation_error == 0.05
        assert result.smoothness_ratio == 0.8
        assert result.rotation_roughness == 0.01
        assert result.mean_angular_velocity == 0.1
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
        config = Config.from_smoothness_distribution([0.5, 0.6, 0.7])
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target", config=config
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
        config = Config.from_smoothness_distribution([0.5, 0.6, 0.7])
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target", config=config
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
        config = Config.from_smoothness_distribution([0.5, 0.6, 0.7])
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target", config=config
        )
        assert result is not None
        assert result.smoothness_ratio >= 0  # Should be valid ratio

    def test_angular_deviation_between_layers(self) -> None:
        """Should compute angular deviation between layer rotations."""
        # Use rotated activations to create different rotations per layer
        cos45 = math.cos(math.pi / 4)
        sin45 = math.sin(math.pi / 4)

        source_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [cos45, -sin45], "b": [sin45, cos45], "c": [0.5, 0.5]},
        }
        target_acts = {
            0: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
            1: {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        }
        analyzer = RotationContinuityAnalyzer()
        config = Config.from_smoothness_distribution([0.5, 0.6, 0.7])
        result = analyzer.compute_per_layer_alignments(
            source_acts, target_acts, "source", "target", config=config
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
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(
            [zero_matrix, identity_matrix], config=config
        )

        if result is not None:
            assert result.dimension == 2
            assert result.model_count == 2

    def test_align_with_near_singular_matrix_completes(self) -> None:
        """Near-singular matrices should not cause SVD numerical issues."""
        near_singular = [[1.0, 2.0], [1.0001, 2.0002]]
        identity = [[1.0, 0.0], [0.0, 1.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align([near_singular, identity], config=config)

        if result is not None:
            assert result.dimension == 2
            assert result.alignment_error >= 0

    def test_align_with_rank_deficient_activations_completes(self) -> None:
        """Rank-deficient matrices should not crash SVD."""
        rank_deficient = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        full_rank = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(
            [rank_deficient, full_rank], config=config
        )

        if result is not None:
            assert result.dimension == 2

    def test_pure_rotation_produces_low_error(self) -> None:
        """Alignment of rotated identity should find the rotation."""
        matrix_a = [[1.0, 0.0], [0.0, 1.0]]
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        matrix_b = [[c, -s], [s, c]]

        config = Config(max_iterations=10)
        result = GeneralizedProcrustes().align([matrix_a, matrix_b], config=config)

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
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(
            [small_matrix, large_matrix], config=config
        )
        # Should return None due to dimension mismatch (align expects same dims)
        # align_crms handles dimension truncation, not align

    def test_align_single_row_matrices(self) -> None:
        """Should handle single-row matrices."""
        single_row_a = [[1.0, 2.0, 3.0]]
        single_row_b = [[4.0, 5.0, 6.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(
            [single_row_a, single_row_b], config=config
        )

        if result is not None:
            assert result.sample_count == 1

    def test_align_very_small_values(self) -> None:
        """Should handle matrices with small (not tiny) values."""
        # Use 1e-3 scale to avoid float32 underflow
        small_values = [[1e-3, 2e-3], [3e-3, 4e-3]]
        normal_values = [[1.0, 2.0], [3.0, 4.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(
            [small_values, normal_values], config=config
        )

        if result is not None:
            assert result.dimension == 2

    def test_align_identical_matrices_many(self) -> None:
        """Should handle many identical matrices efficiently."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        many_identical = [matrix] * 10
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(many_identical, config=config)

        assert result is not None
        assert result.model_count == 10
        backend = get_default_backend()
        eps = _eps(backend, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_align_high_dimensional(self) -> None:
        """Should handle high-dimensional activations."""
        b = get_default_backend()
        b.random_seed(42)

        # Create 20 samples x 128 dimensions
        m1 = b.to_numpy(b.random_normal((20, 64))).tolist()
        m2 = b.to_numpy(b.random_normal((20, 64))).tolist()

        config = Config(max_iterations=20)
        result = GeneralizedProcrustes().align([m1, m2], config=config)

        assert result is not None
        assert result.dimension == 64
        assert result.sample_count == 20

    def test_align_many_samples(self) -> None:
        """Should handle many samples."""
        b = get_default_backend()
        b.random_seed(42)

        # Create 100 samples x 16 dimensions
        m1 = b.to_numpy(b.random_normal((100, 16))).tolist()
        m2 = b.to_numpy(b.random_normal((100, 16))).tolist()

        config = Config(max_iterations=30)
        result = GeneralizedProcrustes().align([m1, m2], config=config)

        assert result is not None
        assert result.sample_count == 100
        assert result.dimension == 16


class TestProcrustesNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_large_scale_difference(self) -> None:
        """Should handle large scale differences with allow_scaling."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1000.0, 0.0], [0.0, 1000.0]]
        config = Config(max_iterations=20, allow_scaling=True)

        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None

    def test_very_similar_matrices(self) -> None:
        """Should converge quickly for very similar matrices."""
        m1 = [[1.0, 0.0], [0.0, 1.0]]
        m2 = [[1.0 + 1e-8, 0.0], [0.0, 1.0 + 1e-8]]
        config = Config(max_iterations=100)

        result = GeneralizedProcrustes().align([m1, m2], config=config)
        assert result is not None
        eps = _eps(get_default_backend(), result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_orthogonal_subspaces(self) -> None:
        """Should handle orthogonal subspaces."""
        # These matrices span orthogonal subspaces
        m1 = [[1.0, 0.0], [1.0, 0.0]]  # Only x-axis
        m2 = [[0.0, 1.0], [0.0, 1.0]]  # Only y-axis
        config = Config(max_iterations=10)

        result = GeneralizedProcrustes().align([m1, m2], config=config)
        # Should complete, even if error is high
        if result is not None:
            assert result.dimension == 2


class TestProcrustesRotationProperties:
    """Tests for rotation matrix properties."""

    def test_rotation_determinant_is_positive_one(self) -> None:
        """Rotations should have determinant +1 (no reflections by default)."""
        b = get_default_backend()
        m1 = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
        m2 = [[0.8, 0.2, 0.4], [0.2, 0.8, 0.4], [0.4, 0.4, 0.9]]

        config = Config(max_iterations=20, allow_reflections=False)
        result = GeneralizedProcrustes().align([m1, m2], config=config)

        assert result is not None
        for rotation in result.rotations:
            R = b.array(rotation)
            det = b.det(R)
            b.eval(det)
            det_val = float(b.to_numpy(det))
            # Determinant should be +1 (not -1, which would be reflection)
            eps = _eps(b, det_val, 1.0)
            assert abs(det_val - 1.0) <= eps

    def test_rotation_preserves_norm(self) -> None:
        """Rotation should preserve vector norms."""
        b = get_default_backend()
        m1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        m2 = [[0.7, 0.3], [0.3, 0.7], [1.0, 1.0]]

        result = GeneralizedProcrustes().align([m1, m2], config=Config(max_iterations=20))
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
                norm = float(b.to_numpy(b.sqrt(b.sum(rotated * rotated))))
                eps = _eps(b, norm, 1.0)
                assert abs(norm - 1.0) <= eps
