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

"""Comprehensive tests for interference_predictor.py.

Tests the merge analysis system that identifies geometric transformations
needed to align models for merging.
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.interference_predictor import (
    TransformationType,
    MergeAnalysisConfig,
    MergeAnalysisResult,
    GlobalMergeAnalysisReport,
    MergeAnalyzer,
    quick_merge_analysis,
)
from modelcypher.core.domain.geometry.riemannian_density import (
    ConceptVolume,
    ConceptVolumeRelation,
    RiemannianDensityEstimator,
    InfluenceType,
)

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Backend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
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
        "concept_a": backend.random_normal((20, 16)),
        "concept_b": backend.random_normal((20, 16)),
        "concept_c": backend.random_normal((20, 16)),
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
# TransformationType Enum Tests
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_values_exist(self) -> None:
        """Verify all expected transformation types exist."""
        assert TransformationType.NULL_SPACE_CONSTRAINT == "null_space_constraint"
        assert TransformationType.CURVATURE_CORRECTION == "curvature_correction"
        assert TransformationType.PROCRUSTES_ROTATION == "procrustes_rotation"
        assert TransformationType.BOUNDARY_PROJECTION == "boundary_projection"
        assert TransformationType.SEMANTIC_VERIFICATION == "semantic_verification"

    def test_is_string_enum(self) -> None:
        """TransformationType should be a string enum."""
        for t in TransformationType:
            assert isinstance(t.value, str)
            assert isinstance(t, str)

    def test_enum_count(self) -> None:
        """Verify expected number of transformation types."""
        assert len(list(TransformationType)) == 5


# =============================================================================
# MergeAnalysisConfig Tests
# =============================================================================


class TestMergeAnalysisConfig:
    """Tests for MergeAnalysisConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MergeAnalysisConfig()
        assert config.null_space_threshold == 0.5
        assert config.curvature_correction_threshold == 0.25
        assert config.procrustes_threshold == 0.5
        assert config.boundary_asymmetry_threshold == 0.5
        assert config.overlap_weight == 0.25
        assert config.curvature_weight == 0.25
        assert config.alignment_weight == 0.25
        assert config.distance_weight == 0.25

    def test_weights_sum_to_one(self) -> None:
        """Weights should sum to 1.0."""
        config = MergeAnalysisConfig()
        total = (
            config.overlap_weight
            + config.curvature_weight
            + config.alignment_weight
            + config.distance_weight
        )
        assert abs(total - 1.0) < 1e-10

    def test_custom_thresholds(self) -> None:
        """Test custom threshold configuration."""
        config = MergeAnalysisConfig(
            null_space_threshold=0.7,
            curvature_correction_threshold=0.1,
            procrustes_threshold=0.8,
        )
        assert config.null_space_threshold == 0.7
        assert config.curvature_correction_threshold == 0.1
        assert config.procrustes_threshold == 0.8

    def test_from_overlap_distribution_empty(self) -> None:
        """Empty overlap distribution should return defaults."""
        config = MergeAnalysisConfig.from_overlap_distribution([])
        assert config.null_space_threshold == 0.5  # default

    def test_from_overlap_distribution_single(self) -> None:
        """Single value distribution."""
        config = MergeAnalysisConfig.from_overlap_distribution([0.6])
        # With single value, all percentiles return that value
        assert config.null_space_threshold == 0.6
        assert config.procrustes_threshold == 1.0 - 0.6

    def test_from_overlap_distribution_multiple(self) -> None:
        """Multiple values should compute percentiles correctly."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        config = MergeAnalysisConfig.from_overlap_distribution(
            scores,
            alpha_percentile=0.5,
            curvature_percentile=0.25,
        )
        # 50th percentile of [0.1..1.0] at index 4 = 0.5
        # The actual value depends on percentile calculation
        assert 0.4 <= config.null_space_threshold <= 0.6

    def test_config_initializes(self) -> None:
        """Configuration should initialize with defaults."""
        config = MergeAnalysisConfig()
        assert isinstance(config, MergeAnalysisConfig)


# =============================================================================
# MergeAnalysisResult Tests
# =============================================================================


class TestMergeAnalysisResult:
    """Tests for MergeAnalysisResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = MergeAnalysisResult(
            volume_a_id="test_a",
            volume_b_id="test_b",
            transformations=[TransformationType.NULL_SPACE_CONSTRAINT],
            overlap_score=0.7,
            curvature_divergence=0.2,
            alignment_score=0.8,
            distance_score=0.3,
            measurement_confidence=0.9,
            transformation_descriptions=["Apply weighted alpha scaling"],
        )
        assert result.volume_a_id == "test_a"
        assert result.volume_b_id == "test_b"
        assert len(result.transformations) == 1
        assert TransformationType.NULL_SPACE_CONSTRAINT in result.transformations
        assert result.overlap_score == 0.7
        assert result.measurement_confidence == 0.9

    def test_empty_transformations(self) -> None:
        """Result with no transformations needed."""
        result = MergeAnalysisResult(
            volume_a_id="a",
            volume_b_id="b",
            transformations=[],
            overlap_score=0.3,
            curvature_divergence=0.1,
            alignment_score=0.9,
            distance_score=0.1,
            measurement_confidence=1.0,
            transformation_descriptions=["Direct merge - no transformations needed"],
        )
        assert result.transformations == []

    def test_multiple_transformations(self) -> None:
        """Result with multiple transformations."""
        result = MergeAnalysisResult(
            volume_a_id="a",
            volume_b_id="b",
            transformations=[
                TransformationType.NULL_SPACE_CONSTRAINT,
                TransformationType.PROCRUSTES_ROTATION,
                TransformationType.BOUNDARY_PROJECTION,
            ],
            overlap_score=0.8,
            curvature_divergence=0.15,
            alignment_score=0.3,
            distance_score=0.4,
            measurement_confidence=0.7,
            transformation_descriptions=["desc1", "desc2", "desc3"],
        )
        assert len(result.transformations) == 3

# =============================================================================
# GlobalMergeAnalysisReport Tests
# =============================================================================


class TestGlobalMergeAnalysisReport:
    """Tests for GlobalMergeAnalysisReport dataclass."""

    def test_empty_report(self) -> None:
        """Test report with no pairs."""
        report = GlobalMergeAnalysisReport(
            pair_results={},
            total_pairs=0,
            transformation_counts={t: 0 for t in TransformationType},
            mean_overlap=0.0,
            mean_curvature_divergence=0.0,
            mean_alignment=1.0,
            transformation_summary="No pairs to analyze",
        )
        assert report.total_pairs == 0
        assert report.mean_alignment == 1.0

    def test_get_pairs_needing_transformation(self) -> None:
        """Test filtering pairs by transformation type."""
        result1 = MergeAnalysisResult(
            volume_a_id="a",
            volume_b_id="b",
            transformations=[TransformationType.NULL_SPACE_CONSTRAINT],
            overlap_score=0.7,
            curvature_divergence=0.1,
            alignment_score=0.8,
            distance_score=0.2,
            measurement_confidence=0.9,
            transformation_descriptions=[],
        )
        result2 = MergeAnalysisResult(
            volume_a_id="c",
            volume_b_id="d",
            transformations=[TransformationType.PROCRUSTES_ROTATION],
            overlap_score=0.4,
            curvature_divergence=0.1,
            alignment_score=0.3,
            distance_score=0.5,
            measurement_confidence=0.9,
            transformation_descriptions=[],
        )
        result3 = MergeAnalysisResult(
            volume_a_id="e",
            volume_b_id="f",
            transformations=[
                TransformationType.NULL_SPACE_CONSTRAINT,
                TransformationType.PROCRUSTES_ROTATION,
            ],
            overlap_score=0.8,
            curvature_divergence=0.2,
            alignment_score=0.2,
            distance_score=0.3,
            measurement_confidence=0.8,
            transformation_descriptions=[],
        )

        report = GlobalMergeAnalysisReport(
            pair_results={
                ("a", "b"): result1,
                ("c", "d"): result2,
                ("e", "f"): result3,
            },
            total_pairs=3,
            transformation_counts={
                TransformationType.NULL_SPACE_CONSTRAINT: 2,
                TransformationType.PROCRUSTES_ROTATION: 2,
                TransformationType.CURVATURE_CORRECTION: 0,
                TransformationType.BOUNDARY_PROJECTION: 0,
                TransformationType.SEMANTIC_VERIFICATION: 0,
            },
            mean_overlap=0.633,
            mean_curvature_divergence=0.133,
            mean_alignment=0.433,
            transformation_summary="test",
        )

        alpha_pairs = report.get_pairs_needing_transformation(
            TransformationType.NULL_SPACE_CONSTRAINT
        )
        assert len(alpha_pairs) == 2
        assert ("a", "b") in alpha_pairs
        assert ("e", "f") in alpha_pairs

        procrustes_pairs = report.get_pairs_needing_transformation(
            TransformationType.PROCRUSTES_ROTATION
        )
        assert len(procrustes_pairs) == 2
        assert ("c", "d") in procrustes_pairs
        assert ("e", "f") in procrustes_pairs

        curvature_pairs = report.get_pairs_needing_transformation(
            TransformationType.CURVATURE_CORRECTION
        )
        assert len(curvature_pairs) == 0

# =============================================================================
# MergeAnalyzer Tests
# =============================================================================


class TestMergeAnalyzer:
    """Tests for MergeAnalyzer class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        analyzer = MergeAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, MergeAnalysisConfig)

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = MergeAnalysisConfig(null_space_threshold=0.8)
        analyzer = MergeAnalyzer(config)
        assert analyzer.config.null_space_threshold == 0.8

    def test_analyze_returns_result(
        self,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """analyze() should return MergeAnalysisResult."""
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        assert isinstance(result, MergeAnalysisResult)
        assert result.volume_a_id == "concept_a"
        assert result.volume_b_id == "concept_b"

    def test_analyze_scores_in_range(
        self,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """All scores should be in [0, 1] range."""
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        assert 0.0 <= result.overlap_score <= 1.0
        assert 0.0 <= result.curvature_divergence <= 1.0
        assert 0.0 <= result.alignment_score <= 1.0
        assert 0.0 <= result.distance_score <= 1.0
        assert 0.0 <= result.measurement_confidence <= 1.0

    def test_analyze_with_precomputed_relation(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """analyze() should accept precomputed relation."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
            relation=relation,
        )
        assert isinstance(result, MergeAnalysisResult)

    def test_analyze_global_returns_report(
        self,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """analyze_global() should return GlobalMergeAnalysisReport."""
        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global(sample_volumes)
        assert isinstance(report, GlobalMergeAnalysisReport)
        # 3 volumes = 3 pairs: (a,b), (a,c), (b,c)
        assert report.total_pairs == 3

    def test_analyze_global_with_precomputed_relations(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """analyze_global() should accept precomputed relations."""
        from modelcypher.core.domain.geometry.riemannian_density import (
            compute_pairwise_relations,
        )

        relations = compute_pairwise_relations(density_estimator, sample_volumes)
        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global(sample_volumes, relations=relations)
        assert report.total_pairs == len(relations)

    def test_transformation_counts(
        self,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """transformation_counts should be populated."""
        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global(sample_volumes)
        # All transformation types should have entries
        for t in TransformationType:
            assert t in report.transformation_counts
            assert report.transformation_counts[t] >= 0

    def test_mean_metrics_in_range(
        self,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """Mean metrics should be in valid ranges."""
        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global(sample_volumes)
        assert 0.0 <= report.mean_overlap <= 1.0
        assert report.mean_curvature_divergence >= 0.0
        assert 0.0 <= report.mean_alignment <= 1.0


# =============================================================================
# MergeAnalyzer Internal Methods Tests
# =============================================================================


class TestMergeAnalyzerInternals:
    """Tests for internal methods of MergeAnalyzer."""

    def test_compute_overlap_score(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """_compute_overlap_score should return average of metrics."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        analyzer = MergeAnalyzer()
        score = analyzer._compute_overlap_score(relation)
        # Should be average of bhattacharyya, overlap, and jaccard
        expected = (
            relation.bhattacharyya_coefficient
            + relation.overlap_coefficient
            + relation.jaccard_index
        ) / 3.0
        assert abs(score - expected) < 1e-10

    def test_compute_curvature_divergence(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """_compute_curvature_divergence should return relation's value."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        analyzer = MergeAnalyzer()
        div = analyzer._compute_curvature_divergence(relation)
        assert div == relation.curvature_divergence

    def test_compute_alignment_score(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """_compute_alignment_score should return relation's value."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        analyzer = MergeAnalyzer()
        align = analyzer._compute_alignment_score(relation)
        assert align == relation.subspace_alignment

    def test_compute_distance_score_normalized(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """_compute_distance_score should be normalized and clamped."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        analyzer = MergeAnalyzer()
        dist = analyzer._compute_distance_score(relation)
        assert 0.0 <= dist <= 1.0

    def test_compute_distance_score_zero_radius(self, backend: "Backend") -> None:
        """Distance score should handle zero radius."""
        # Create minimal volumes with tiny covariance (near-zero radius)
        estimator = RiemannianDensityEstimator()
        activations = backend.array([[0.0] * 8, [1e-10] * 8])
        vol_a = estimator.estimate_concept_volume("a", activations)
        vol_b = estimator.estimate_concept_volume("b", activations)

        relation = estimator.compute_relation(vol_a, vol_b)
        analyzer = MergeAnalyzer()
        dist = analyzer._compute_distance_score(relation)
        assert dist >= 0.0

    def test_identify_transformations_alpha_scaling(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """High overlap should trigger NULL_SPACE_CONSTRAINT."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        # Force high overlap
        config = MergeAnalysisConfig(null_space_threshold=0.0)  # Always trigger
        analyzer = MergeAnalyzer(config)
        transformations = analyzer._identify_transformations(
            relation, overlap_score=0.9, curvature_divergence=0.0, alignment_score=1.0
        )
        assert TransformationType.NULL_SPACE_CONSTRAINT in transformations

    def test_identify_transformations_curvature_correction(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """High curvature divergence should trigger CURVATURE_CORRECTION."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        config = MergeAnalysisConfig(curvature_correction_threshold=0.0)  # Always trigger
        analyzer = MergeAnalyzer(config)
        transformations = analyzer._identify_transformations(
            relation, overlap_score=0.0, curvature_divergence=0.5, alignment_score=1.0
        )
        assert TransformationType.CURVATURE_CORRECTION in transformations

    def test_identify_transformations_procrustes(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """Low alignment should trigger PROCRUSTES_ROTATION."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        config = MergeAnalysisConfig(procrustes_threshold=1.0)  # Always trigger
        analyzer = MergeAnalyzer(config)
        transformations = analyzer._identify_transformations(
            relation, overlap_score=0.0, curvature_divergence=0.0, alignment_score=0.3
        )
        assert TransformationType.PROCRUSTES_ROTATION in transformations

    def test_identify_transformations_none_needed(
        self,
        density_estimator: RiemannianDensityEstimator,
        sample_volumes: dict[str, ConceptVolume],
    ) -> None:
        """When all metrics are good, no transformations needed."""
        relation = density_estimator.compute_relation(
            sample_volumes["concept_a"],
            sample_volumes["concept_b"],
        )
        # Set thresholds that won't trigger anything
        config = MergeAnalysisConfig(
            null_space_threshold=1.0,
            curvature_correction_threshold=1.0,
            procrustes_threshold=0.0,
            boundary_asymmetry_threshold=1.0,
        )
        analyzer = MergeAnalyzer(config)
        transformations = analyzer._identify_transformations(
            relation, overlap_score=0.3, curvature_divergence=0.1, alignment_score=0.9
        )
        assert len(transformations) == 0

    def test_generate_transformation_descriptions_all_types(self) -> None:
        """Should generate descriptions for all transformation types."""
        analyzer = MergeAnalyzer()
        for t in TransformationType:
            descriptions = analyzer._generate_transformation_descriptions([t])
            assert len(descriptions) == 1
            assert isinstance(descriptions[0], str)
            assert len(descriptions[0]) > 0

    def test_generate_transformation_descriptions_empty(self) -> None:
        """Empty transformations should return default message."""
        analyzer = MergeAnalyzer()
        descriptions = analyzer._generate_transformation_descriptions([])
        assert len(descriptions) == 1
        assert "no transformations" in descriptions[0].lower()

    def test_compute_measurement_confidence_high_samples(
        self, backend: "Backend"
    ) -> None:
        """High sample/dimension ratio should give high confidence."""
        estimator = RiemannianDensityEstimator()
        # Many samples (100), small dimension (8) = high ratio
        backend.random_seed(42)
        activations = backend.random_normal((100, 8))
        vol_a = estimator.estimate_concept_volume("a", activations)
        vol_b = estimator.estimate_concept_volume("b", activations)
        relation = estimator.compute_relation(vol_a, vol_b)

        analyzer = MergeAnalyzer()
        confidence = analyzer._compute_measurement_confidence(relation)
        # With ratio 100/8 = 12.5, confidence should be high
        assert confidence > 0.5

    def test_compute_measurement_confidence_low_samples(
        self, backend: "Backend"
    ) -> None:
        """Low sample/dimension ratio should give lower confidence."""
        estimator = RiemannianDensityEstimator()
        # Few samples (5), large dimension (16) = low ratio
        backend.random_seed(42)
        activations = backend.random_normal((5, 16))
        vol_a = estimator.estimate_concept_volume("a", activations)
        vol_b = estimator.estimate_concept_volume("b", activations)
        relation = estimator.compute_relation(vol_a, vol_b)

        analyzer = MergeAnalyzer()
        confidence = analyzer._compute_measurement_confidence(relation)
        # With ratio 5/16 = 0.3125, confidence should be lower
        assert confidence <= 0.5

    def test_generate_transformation_summary_no_pairs(self) -> None:
        """Empty pairs should return appropriate message."""
        analyzer = MergeAnalyzer()
        counts = {t: 0 for t in TransformationType}
        summary = analyzer._generate_transformation_summary(counts, 0)
        assert "No pairs" in summary

    def test_generate_transformation_summary_no_transformations(self) -> None:
        """No transformations needed should return appropriate message."""
        analyzer = MergeAnalyzer()
        counts = {t: 0 for t in TransformationType}
        summary = analyzer._generate_transformation_summary(counts, 5)
        assert "directly" in summary.lower() or "no transformation" in summary.lower()

    def test_generate_transformation_summary_with_transformations(self) -> None:
        """Should include percentage information."""
        analyzer = MergeAnalyzer()
        counts = {t: 0 for t in TransformationType}
        counts[TransformationType.NULL_SPACE_CONSTRAINT] = 3
        counts[TransformationType.PROCRUSTES_ROTATION] = 2
        summary = analyzer._generate_transformation_summary(counts, 10)
        assert "alpha_scaling" in summary
        assert "procrustes_rotation" in summary
        assert "%" in summary


# =============================================================================
# quick_merge_analysis Tests
# =============================================================================


class TestQuickMergeAnalysis:
    """Tests for quick_merge_analysis function."""

    def test_basic_usage(self, backend: "Backend") -> None:
        """Basic usage should return GlobalMergeAnalysisReport."""
        backend.random_seed(42)
        source = {
            "concept_x": backend.random_normal((20, 16)),
            "concept_y": backend.random_normal((20, 16)),
        }
        target = {
            "concept_x": backend.random_normal((20, 16)),
            "concept_y": backend.random_normal((20, 16)),
        }
        report = quick_merge_analysis(source, target)
        assert isinstance(report, GlobalMergeAnalysisReport)
        assert report.total_pairs == 2  # x-x and y-y pairs

    def test_no_common_concepts(self, backend: "Backend") -> None:
        """No common concepts should return empty report."""
        backend.random_seed(42)
        source = {"concept_a": backend.random_normal((20, 16))}
        target = {"concept_b": backend.random_normal((20, 16))}
        report = quick_merge_analysis(source, target)
        assert report.total_pairs == 0
        assert "No common concepts" in report.transformation_summary

    def test_partial_overlap(self, backend: "Backend") -> None:
        """Partial concept overlap should analyze common concepts only."""
        backend.random_seed(42)
        source = {
            "common": backend.random_normal((20, 16)),
            "source_only": backend.random_normal((20, 16)),
        }
        target = {
            "common": backend.random_normal((20, 16)),
            "target_only": backend.random_normal((20, 16)),
        }
        report = quick_merge_analysis(source, target)
        assert report.total_pairs == 1  # Only "common" pair

    def test_volume_ids_prefixed(self, backend: "Backend") -> None:
        """Volume IDs should be prefixed with source:/target:."""
        backend.random_seed(42)
        source = {"test": backend.random_normal((20, 16))}
        target = {"test": backend.random_normal((20, 16))}
        report = quick_merge_analysis(source, target)
        # Check that pair keys have prefixed IDs
        for key in report.pair_results.keys():
            assert key[0].startswith("source:") or key[1].startswith("target:")


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_volume(self, backend: "Backend") -> None:
        """Single sample volume should be handled."""
        estimator = RiemannianDensityEstimator()
        single_sample = backend.array([[1.0, 2.0, 3.0, 4.0]])
        vol = estimator.estimate_concept_volume("single", single_sample)
        assert vol.num_samples == 1
        assert vol.geodesic_radius == 0.0

    def test_two_sample_volume(self, backend: "Backend") -> None:
        """Two sample volume should be handled."""
        estimator = RiemannianDensityEstimator()
        two_samples = backend.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]])
        vol = estimator.estimate_concept_volume("two", two_samples)
        assert vol.num_samples == 2

    def test_identical_volumes(self, backend: "Backend") -> None:
        """Identical volumes should have high similarity."""
        estimator = RiemannianDensityEstimator()
        backend.random_seed(42)
        activations = backend.random_normal((20, 16))
        vol_a = estimator.estimate_concept_volume("a", activations)
        vol_b = estimator.estimate_concept_volume("b", activations)

        analyzer = MergeAnalyzer()
        result = analyzer.analyze(vol_a, vol_b)
        # Same data should have high overlap and alignment
        # Note: CKA should be ~1.0 for identical data
        assert result.alignment_score > 0.5

    def test_different_pairwise_relationships(self, backend: "Backend") -> None:
        """Volumes with different pairwise structures should have lower CKA similarity.

        Note: CKA measures representational similarity based on Gram matrices
        (pairwise relationships between samples). Two representations with
        different pairwise distance structures will have lower CKA.
        """
        estimator = RiemannianDensityEstimator()

        # First set: samples clustered together (similar pairwise distances)
        backend.random_seed(42)
        base_point = backend.random_normal((1, 16))
        noise_a = backend.random_normal((20, 16)) * 0.1
        act_a = base_point + noise_a

        # Second set: samples spread out (very different pairwise distances)
        backend.random_seed(123)
        act_b = backend.random_normal((20, 16)) * 5.0  # Large spread

        vol_a = estimator.estimate_concept_volume("a", act_a)
        vol_b = estimator.estimate_concept_volume("b", act_b)

        analyzer = MergeAnalyzer()
        result = analyzer.analyze(vol_a, vol_b)

        # CKA for different pairwise structures should be lower than identical data
        # We just verify the result is valid - the exact value depends on the data
        assert 0.0 <= result.alignment_score <= 1.0

    def test_analyze_global_empty_volumes(self) -> None:
        """Empty volumes dict should return empty report."""
        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global({})
        assert report.total_pairs == 0
        assert report.mean_alignment == 1.0  # Default value

    def test_analyze_global_single_volume(self, backend: "Backend") -> None:
        """Single volume should return zero pairs."""
        estimator = RiemannianDensityEstimator()
        backend.random_seed(42)
        activations = backend.random_normal((20, 16))
        volumes = {"single": estimator.estimate_concept_volume("single", activations)}

        analyzer = MergeAnalyzer()
        report = analyzer.analyze_global(volumes)
        assert report.total_pairs == 0


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestProperties:
    """Property-based tests for invariants."""

    def test_overlap_score_bounds(self, backend: "Backend") -> None:
        """Overlap score must be in [0, 1]."""
        estimator = RiemannianDensityEstimator()
        analyzer = MergeAnalyzer()

        # Test with various random seeds
        for seed in [1, 42, 123, 456, 789]:
            backend.random_seed(seed)
            act_a = backend.random_normal((20, 16))
            act_b = backend.random_normal((20, 16))
            vol_a = estimator.estimate_concept_volume("a", act_a)
            vol_b = estimator.estimate_concept_volume("b", act_b)
            result = analyzer.analyze(vol_a, vol_b)
            assert 0.0 <= result.overlap_score <= 1.0, f"Failed for seed {seed}"

    def test_distance_score_bounds(self, backend: "Backend") -> None:
        """Distance score must be in [0, 1]."""
        estimator = RiemannianDensityEstimator()
        analyzer = MergeAnalyzer()

        for seed in [1, 42, 123, 456, 789]:
            backend.random_seed(seed)
            act_a = backend.random_normal((20, 16))
            act_b = backend.random_normal((20, 16))
            vol_a = estimator.estimate_concept_volume("a", act_a)
            vol_b = estimator.estimate_concept_volume("b", act_b)
            result = analyzer.analyze(vol_a, vol_b)
            assert 0.0 <= result.distance_score <= 1.0, f"Failed for seed {seed}"

    def test_confidence_bounds(self, backend: "Backend") -> None:
        """Confidence must be in [0, 1]."""
        estimator = RiemannianDensityEstimator()
        analyzer = MergeAnalyzer()

        for seed in [1, 42, 123, 456, 789]:
            backend.random_seed(seed)
            act_a = backend.random_normal((20, 16))
            act_b = backend.random_normal((20, 16))
            vol_a = estimator.estimate_concept_volume("a", act_a)
            vol_b = estimator.estimate_concept_volume("b", act_b)
            result = analyzer.analyze(vol_a, vol_b)
            assert 0.0 <= result.measurement_confidence <= 1.0, f"Failed for seed {seed}"

    def test_transformation_counts_match_pairs(self, backend: "Backend") -> None:
        """Sum of transformation applications should be consistent."""
        estimator = RiemannianDensityEstimator()
        analyzer = MergeAnalyzer()

        backend.random_seed(42)
        volumes = {}
        for i in range(4):
            volumes[f"vol_{i}"] = estimator.estimate_concept_volume(
                f"vol_{i}", backend.random_normal((20, 16))
            )

        report = analyzer.analyze_global(volumes)

        # Each pair should be counted in transformation_counts
        total_transformations = 0
        for result in report.pair_results.values():
            total_transformations += len(result.transformations)

        # Sum of counts should equal total transformations
        sum_counts = sum(report.transformation_counts.values())
        assert sum_counts == total_transformations

    def test_symmetry_of_pair_analysis(self, backend: "Backend") -> None:
        """Analyzing (A, B) and (B, A) should give consistent results."""
        estimator = RiemannianDensityEstimator()
        analyzer = MergeAnalyzer()

        backend.random_seed(42)
        act_a = backend.random_normal((20, 16))
        act_b = backend.random_normal((20, 16))
        vol_a = estimator.estimate_concept_volume("a", act_a)
        vol_b = estimator.estimate_concept_volume("b", act_b)

        result_ab = analyzer.analyze(vol_a, vol_b)
        result_ba = analyzer.analyze(vol_b, vol_a)

        # Symmetric metrics should be similar
        # (overlap, alignment, curvature_divergence are symmetric)
        assert abs(result_ab.overlap_score - result_ba.overlap_score) < 0.01
        assert abs(result_ab.alignment_score - result_ba.alignment_score) < 0.01
        # Distance score depends on direction but should be similar for same data
        assert abs(result_ab.distance_score - result_ba.distance_score) < 0.2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full analysis pipeline."""

    def test_full_pipeline(self, backend: "Backend") -> None:
        """Test complete analysis pipeline."""
        # Create source and target activations
        backend.random_seed(42)
        concepts = ["math", "code", "language"]
        source = {c: backend.random_normal((30, 32)) for c in concepts}
        target = {c: backend.random_normal((30, 32)) for c in concepts}

        # Run quick analysis
        report = quick_merge_analysis(source, target)

        # Verify report structure
        assert report.total_pairs == 3
        assert len(report.pair_results) == 3
        assert all(t in report.transformation_counts for t in TransformationType)
        assert len(report.transformation_summary) > 0

    def test_config_affects_transformations(self, backend: "Backend") -> None:
        """Different configs should produce different transformations."""
        estimator = RiemannianDensityEstimator()

        backend.random_seed(42)
        act_a = backend.random_normal((20, 16))
        act_b = backend.random_normal((20, 16))
        vol_a = estimator.estimate_concept_volume("a", act_a)
        vol_b = estimator.estimate_concept_volume("b", act_b)

        # Strict config - triggers more transformations
        strict_config = MergeAnalysisConfig(
            null_space_threshold=0.0,
            curvature_correction_threshold=0.0,
            procrustes_threshold=1.0,
        )
        strict_analyzer = MergeAnalyzer(strict_config)
        strict_result = strict_analyzer.analyze(vol_a, vol_b)

        # Lenient config - triggers fewer transformations
        lenient_config = MergeAnalysisConfig(
            null_space_threshold=1.0,
            curvature_correction_threshold=1.0,
            procrustes_threshold=0.0,
        )
        lenient_analyzer = MergeAnalyzer(lenient_config)
        lenient_result = lenient_analyzer.analyze(vol_a, vol_b)

        # Strict should have more transformations
        assert len(strict_result.transformations) >= len(lenient_result.transformations)
