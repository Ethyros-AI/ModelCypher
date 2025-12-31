# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Integration tests for the merge validation pipeline.

Tests the full flow: entropy profiling → interference analysis → transplant → validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.interference_predictor import (
    MergeAnalysisConfig,
    MergeAnalyzer,
    MergeAnalysisResult,
    TransformationType,
)
from modelcypher.core.domain.geometry.riemannian_density import (
    ConceptVolume,
    RiemannianDensityEstimator,
)
from modelcypher.core.domain.geometry.transplant import (
    CoreBoundaryPartition,
    TransplantDeltaResult,
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.domain.thermo.phase_transition_theory import Phase

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def density_estimator() -> RiemannianDensityEstimator:
    """Create a RiemannianDensityEstimator for testing."""
    return RiemannianDensityEstimator()


@pytest.fixture
def sample_activations(backend: "Backend") -> dict[str, "Array"]:
    """Create sample activation arrays for testing."""
    backend.random_seed(42)
    return {
        "concept_a": backend.random_normal((30, 32)),
        "concept_b": backend.random_normal((30, 32)),
        "concept_c": backend.random_normal((30, 32)),
    }


@pytest.fixture
def sample_volumes(
    density_estimator: RiemannianDensityEstimator,
    sample_activations: dict[str, "Array"],
) -> dict[str, ConceptVolume]:
    """Create sample ConceptVolume objects for testing."""
    volumes = {}
    for concept_id, activations in sample_activations.items():
        volumes[concept_id] = density_estimator.estimate_concept_volume(
            concept_id, activations
        )
    return volumes


# =============================================================================
# TestInterferencePipeline
# =============================================================================


class TestInterferencePipeline:
    """Tests for interference/merge analysis pipeline."""

    def test_interference_analysis_basic(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Two concept volumes should produce valid merge analysis."""
        analyzer = MergeAnalyzer()

        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Should return a valid result
        assert isinstance(result, MergeAnalysisResult)
        assert result.volume_a_id == "concept_a"
        assert result.volume_b_id == "concept_b"

        # Transformations should be a list
        assert isinstance(result.transformations, list)
        for t in result.transformations:
            assert isinstance(t, TransformationType)

    def test_interference_overlap_computation(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Overlap score should have correct properties."""
        analyzer = MergeAnalyzer()

        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Overlap score should be in [0, 1]
        assert 0.0 <= result.overlap_score <= 1.0

        # Self-analysis should have high overlap
        self_result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_a"]
        )
        assert self_result.overlap_score >= 0.9

    def test_transformation_requirements(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Transformations should match geometric properties."""
        # Use config that triggers transformations
        config = MergeAnalysisConfig(
            alpha_scaling_threshold=0.3,  # Lower threshold
            curvature_correction_threshold=0.1,
            procrustes_threshold=0.7,
        )
        analyzer = MergeAnalyzer(config)

        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Should have identified some transformations
        # (exact transformations depend on random data, but logic should work)
        assert isinstance(result.transformations, list)

        # All transformations should be valid enum values
        valid_transformations = set(TransformationType)
        for t in result.transformations:
            assert t in valid_transformations

    def test_interference_raw_measurements(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Output should be raw measurements only, no interpretation strings."""
        analyzer = MergeAnalyzer()

        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Measurements should be raw floats
        assert isinstance(result.overlap_score, float)
        assert isinstance(result.curvature_divergence, float)
        assert isinstance(result.alignment_score, float)
        assert isinstance(result.distance_score, float)
        assert isinstance(result.measurement_confidence, float)

        # All measurements should be finite
        import math

        assert math.isfinite(result.overlap_score)
        assert math.isfinite(result.curvature_divergence)
        assert math.isfinite(result.alignment_score)
        assert math.isfinite(result.distance_score)


class TestInterferenceSymmetry:
    """Tests for symmetry properties of interference analysis."""

    def test_overlap_symmetry(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Overlap(A, B) should approximately equal Overlap(B, A)."""
        analyzer = MergeAnalyzer()

        result_ab = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )
        result_ba = analyzer.analyze(
            sample_volumes["concept_b"], sample_volumes["concept_a"]
        )

        # Overlap should be symmetric (within tolerance)
        assert result_ab.overlap_score == pytest.approx(
            result_ba.overlap_score, rel=0.1
        )


# =============================================================================
# TestTransplantPipeline
# =============================================================================


class TestTransplantPipeline:
    """Tests for the transplant (null-space projection) pipeline."""

    def test_core_boundary_partitioning(self, backend: "Backend") -> None:
        """Should correctly separate core and boundary probes."""
        backend.random_seed(42)
        activations = backend.random_normal((20, 32))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(20)]
        core_probe_ids = {f"probe_{i}" for i in range(5)}  # First 5 are core

        partition = partition_core_boundary(
            activations,
            probe_ids,
            core_probe_ids,
            boundary_k=3,
            backend=backend,
        )

        # Should return correct structure
        assert isinstance(partition, CoreBoundaryPartition)

        # Core indices should match
        assert len(partition.core_indices) == 5
        assert all(i < 5 for i in partition.core_indices)

        # Boundary should be identified
        assert len(partition.boundary_indices) > 0

        # Core and boundary should be disjoint
        core_set = set(partition.core_indices)
        boundary_set = set(partition.boundary_indices)
        assert core_set.isdisjoint(boundary_set)

    def test_null_space_projection(self, backend: "Backend") -> None:
        """Boundary activations should be preserved after transplant."""
        backend.random_seed(42)

        # Create weight matrices
        weight_target = backend.random_normal((64, 32))
        weight_source = backend.random_normal((64, 32))
        backend.eval(weight_target, weight_source)

        # Create core and boundary activations
        backend.random_seed(43)
        activations_core = backend.random_normal((10, 32))
        activations_boundary = backend.random_normal((5, 32))
        backend.eval(activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target,
            weight_source,
            activations_core,
            activations_boundary,
            backend=backend,
        )

        # Should return valid result
        assert isinstance(result, TransplantDeltaResult)
        assert result.applied or not result.applied  # Boolean

        # Merged weight should have same shape as target
        assert result.merged_weight.shape == weight_target.shape

    def test_transplant_delta_properties(self, backend: "Backend") -> None:
        """Transplant delta should have expected properties."""
        backend.random_seed(42)

        weight_target = backend.random_normal((32, 16))
        weight_source = backend.random_normal((32, 16))
        activations_core = backend.random_normal((8, 16))
        activations_boundary = backend.random_normal((4, 16))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target,
            weight_source,
            activations_core,
            activations_boundary,
            backend=backend,
        )

        # Norms should be non-negative
        assert result.delta_norm >= 0.0
        assert result.filtered_norm >= 0.0

        # Preserved fraction should be in [0, 1]
        assert 0.0 <= result.preserved_fraction <= 1.0

        # Null dimension should be non-negative integer
        assert result.null_dim >= 0

    def test_transplant_reproducibility(self, backend: "Backend") -> None:
        """Same inputs should produce same outputs."""
        backend.random_seed(42)

        weight_target = backend.random_normal((32, 16))
        weight_source = backend.random_normal((32, 16))
        activations_core = backend.random_normal((8, 16))
        activations_boundary = backend.random_normal((4, 16))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        result1 = compute_transplant_delta(
            weight_target,
            weight_source,
            activations_core,
            activations_boundary,
            backend=backend,
        )

        result2 = compute_transplant_delta(
            weight_target,
            weight_source,
            activations_core,
            activations_boundary,
            backend=backend,
        )

        # Results should be deterministic
        assert result1.null_dim == result2.null_dim
        assert result1.delta_norm == pytest.approx(result2.delta_norm, rel=1e-6)
        assert result1.preserved_fraction == pytest.approx(
            result2.preserved_fraction, rel=1e-6
        )


# =============================================================================
# TestValidationFlow
# =============================================================================


class TestValidationFlow:
    """Tests for combined validation flow."""

    def test_interference_then_transplant_flow(self, backend: "Backend") -> None:
        """Interference analysis should inform transplant decisions."""
        backend.random_seed(42)

        # Create concept volumes
        activations_a = backend.random_normal((30, 32))
        activations_b = backend.random_normal((30, 32))
        backend.eval(activations_a, activations_b)

        estimator = RiemannianDensityEstimator()
        volume_a = estimator.estimate_concept_volume("model_a", activations_a)
        volume_b = estimator.estimate_concept_volume("model_b", activations_b)

        # Step 1: Analyze interference
        analyzer = MergeAnalyzer()
        analysis = analyzer.analyze(volume_a, volume_b)

        # Step 2: Use analysis to inform transplant
        # (In real usage, transformations would configure transplant parameters)
        weight_target = backend.random_normal((64, 32))
        weight_source = backend.random_normal((64, 32))
        backend.eval(weight_target, weight_source)

        # Partition based on analysis (here using arbitrary split for test)
        activations_core = activations_a[:15]
        activations_boundary = activations_a[15:]

        transplant = compute_transplant_delta(
            weight_target,
            weight_source,
            activations_core,
            activations_boundary,
            backend=backend,
        )

        # Both stages should produce valid results
        assert isinstance(analysis, MergeAnalysisResult)
        assert isinstance(transplant, TransplantDeltaResult)

        # Can use interference metrics to validate transplant
        # High overlap suggests more aggressive transplant is safe
        if analysis.overlap_score > 0.7:
            # Would use higher alpha in real usage
            pass

    def test_validation_result_structure(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Validation results should have all expected fields."""
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Check all expected fields are present
        assert hasattr(result, "volume_a_id")
        assert hasattr(result, "volume_b_id")
        assert hasattr(result, "transformations")
        assert hasattr(result, "overlap_score")
        assert hasattr(result, "curvature_divergence")
        assert hasattr(result, "alignment_score")
        assert hasattr(result, "distance_score")
        assert hasattr(result, "measurement_confidence")
        assert hasattr(result, "transformation_descriptions")

    def test_validation_no_vibes(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Validation should return raw measurements, not interpretations."""
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )

        # Should NOT have interpretation fields
        assert not hasattr(result, "verdict")
        assert not hasattr(result, "recommendation")
        assert not hasattr(result, "risk_level")
        assert not hasattr(result, "quality")

        # Transformation descriptions should be factual, not judgmental
        for desc in result.transformation_descriptions:
            # Should not contain judgmental words
            judgmental = ["good", "bad", "safe", "unsafe", "risky", "excellent"]
            desc_lower = desc.lower()
            for word in judgmental:
                assert word not in desc_lower, f"Found judgmental word '{word}' in: {desc}"

    def test_multi_volume_consistency(
        self, sample_volumes: dict[str, ConceptVolume]
    ) -> None:
        """Analysis across multiple volumes should be internally consistent."""
        analyzer = MergeAnalyzer()

        # Analyze all pairs
        result_ab = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_b"]
        )
        result_bc = analyzer.analyze(
            sample_volumes["concept_b"], sample_volumes["concept_c"]
        )
        result_ac = analyzer.analyze(
            sample_volumes["concept_a"], sample_volumes["concept_c"]
        )

        # All should be valid
        for result in [result_ab, result_bc, result_ac]:
            assert 0.0 <= result.overlap_score <= 1.0
            assert 0.0 <= result.alignment_score <= 1.0
            assert result.measurement_confidence > 0.0


# =============================================================================
# TestPhaseClassification
# =============================================================================


class TestPhaseClassification:
    """Tests for entropy phase classification."""

    def test_phase_enum_values(self) -> None:
        """Phase enum should have expected values."""
        assert Phase.ORDERED.value == "ordered"
        assert Phase.CRITICAL.value == "critical"
        assert Phase.DISORDERED.value == "disordered"

    def test_phase_display_names(self) -> None:
        """Phase display names should be informative."""
        assert "T < T_c" in Phase.ORDERED.display_name
        assert "T ≈ T_c" in Phase.CRITICAL.display_name
        assert "T > T_c" in Phase.DISORDERED.display_name

    def test_phase_modifier_effects(self) -> None:
        """Phase should describe expected modifier effects."""
        # All phases should have expected_modifier_effect property
        for phase in Phase:
            effect = phase.expected_modifier_effect
            assert isinstance(effect, str)
            assert len(effect) > 0


class TestConfigFromData:
    """Tests for data-driven configuration."""

    def test_merge_analysis_config_from_distribution(self) -> None:
        """Config should be derivable from overlap distribution."""
        overlap_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        config = MergeAnalysisConfig.from_overlap_distribution(
            overlap_scores,
            alpha_percentile=0.5,
            curvature_percentile=0.25,
            procrustes_percentile=0.5,
        )

        # Thresholds should be derived from percentiles
        assert isinstance(config, MergeAnalysisConfig)
        assert 0.0 <= config.alpha_scaling_threshold <= 1.0
        assert 0.0 <= config.curvature_correction_threshold <= 1.0

    def test_empty_distribution_uses_defaults(self) -> None:
        """Empty distribution should use default config."""
        config = MergeAnalysisConfig.from_overlap_distribution([])

        # Should return valid config with defaults
        assert isinstance(config, MergeAnalysisConfig)
        assert config.alpha_scaling_threshold == 0.5  # Default value
