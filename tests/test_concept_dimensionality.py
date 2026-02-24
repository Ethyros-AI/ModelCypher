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

"""Tests for concept dimensionality analysis."""

from __future__ import annotations

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityReport,
    ConceptDimensionalityResult,
    ConceptDimensionalityStudy,
    ConceptDimensionalityStudyReport,
    DomainRankCorrelation,
    DomainSummary,
    LayerDimensionalitySummary,
    SkippedProbe,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _scalar_tol(value: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([value]))


class TestConceptDimensionalityResult:
    """Tests for ConceptDimensionalityResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        result = ConceptDimensionalityResult(
            probe_id="test-probe",
            name="Test Probe",
            source="safety",
            domain="refusal",
            category="explicit",
            layer=10,
            support_text_count=5,
            sample_count=5,
            usable_count=4,
            intrinsic_dimension=2.3,
            calibration_weight=0.8,
            ci_lower=2.1,
            ci_upper=2.5,
        )

        assert result.probe_id == "test-probe"
        assert result.name == "Test Probe"
        assert result.source == "safety"
        assert result.domain == "refusal"
        assert result.layer == 10
        assert result.intrinsic_dimension == 2.3
        assert result.calibration_weight == 0.8
        assert result.ci_lower == 2.1
        assert result.ci_upper == 2.5

    def test_result_with_no_calibration(self):
        """Test result without calibration weight."""
        result = ConceptDimensionalityResult(
            probe_id="test",
            name="Test",
            source="safety",
            domain="refusal",
            category="explicit",
            layer=5,
            support_text_count=3,
            sample_count=3,
            usable_count=3,
            intrinsic_dimension=1.5,
            calibration_weight=None,
            ci_lower=None,
            ci_upper=None,
        )

        assert result.calibration_weight is None
        assert result.ci_lower is None
        assert result.ci_upper is None


class TestSkippedProbe:
    """Tests for SkippedProbe dataclass."""

    def test_skipped_probe_creation(self):
        """Test creating a skipped probe record."""
        skipped = SkippedProbe(
            probe_id="skipped-probe",
            name="Skipped Probe",
            reason="insufficient_support_texts",
            support_text_count=2,
            calibration_weight=0.5,
        )

        assert skipped.probe_id == "skipped-probe"
        assert skipped.name == "Skipped Probe"
        assert skipped.reason == "insufficient_support_texts"
        assert skipped.support_text_count == 2
        assert skipped.calibration_weight == 0.5
        assert skipped.activation_count is None
        assert skipped.invalid_counts is None

    def test_skipped_with_invalid_counts(self):
        """Test skipped probe with invalid count details."""
        skipped = SkippedProbe(
            probe_id="invalid-vectors",
            name="Invalid Vectors",
            reason="insufficient_valid_vectors",
            support_text_count=5,
            calibration_weight=0.9,
            activation_count=5,
            invalid_counts={"empty": 1, "non_finite": 2, "length_mismatch": 0},
        )

        assert skipped.activation_count == 5
        assert skipped.invalid_counts["empty"] == 1
        assert skipped.invalid_counts["non_finite"] == 2


class TestDomainSummary:
    """Tests for DomainSummary dataclass."""

    def test_domain_summary_creation(self):
        """Test creating a domain summary."""
        summary = DomainSummary(
            domain="refusal",
            probe_count=10,
            mean_dimension=2.5,
            dimension_histogram={"1": 2, "2": 5, "3": 3, "4+": 0},
        )

        assert summary.domain == "refusal"
        assert summary.probe_count == 10
        assert summary.mean_dimension == 2.5
        assert summary.dimension_histogram["2"] == 5


class TestFilterVectors:
    """Tests for _filter_vectors static method."""

    def test_valid_vectors_pass_through(self):
        """Test that valid vectors pass through unchanged."""
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        cleaned, invalid = ConceptDimensionalityAnalyzer._filter_vectors(vectors)

        assert len(cleaned) == 3
        assert all(count == 0 for count in invalid.values())

    def test_empty_vector_filtered(self):
        """Test that empty vectors are filtered out."""
        vectors = [[1.0, 2.0], [], [3.0, 4.0]]

        cleaned, invalid = ConceptDimensionalityAnalyzer._filter_vectors(vectors)

        assert len(cleaned) == 2
        assert invalid["empty"] == 1

    def test_length_mismatch_filtered(self):
        """Test that mismatched length vectors are filtered."""
        vectors = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]]

        cleaned, invalid = ConceptDimensionalityAnalyzer._filter_vectors(vectors)

        assert len(cleaned) == 2
        assert invalid["length_mismatch"] == 1

    def test_non_finite_filtered(self):
        """Test that non-finite values are filtered."""
        vectors = [
            [1.0, 2.0],
            [float("inf"), 3.0],
            [4.0, float("nan")],
            [5.0, 6.0],
        ]

        cleaned, invalid = ConceptDimensionalityAnalyzer._filter_vectors(vectors)

        assert len(cleaned) == 2
        assert invalid["non_finite"] == 2

    def test_mixed_invalid_vectors(self):
        """Test filtering multiple types of invalid vectors."""
        vectors = [
            [1.0, 2.0],
            [],  # empty
            [3.0, 4.0, 5.0],  # length mismatch
            [float("inf"), 2.0],  # non-finite
            [7.0, 8.0],  # valid
        ]

        cleaned, invalid = ConceptDimensionalityAnalyzer._filter_vectors(vectors)

        assert len(cleaned) == 2
        assert invalid["empty"] == 1
        assert invalid["length_mismatch"] == 1
        assert invalid["non_finite"] == 1


class TestDimensionHistogram:
    """Tests for _dimension_histogram static method."""

    def test_empty_results(self):
        """Test histogram with no results."""
        histogram = ConceptDimensionalityAnalyzer._dimension_histogram([])

        # Empty results produce empty histogram (no buckets created)
        assert histogram == {}

    def test_histogram_counts(self):
        """Test histogram correctly counts dimension buckets."""
        results = [
            _make_result(intrinsic_dimension=1.0),  # floor(1.0) = 1 -> "1"
            _make_result(intrinsic_dimension=1.2),  # floor(1.2) = 1 -> "1"
            _make_result(intrinsic_dimension=2.0),  # floor(2.0) = 2 -> "2"
            _make_result(intrinsic_dimension=2.8),  # floor(2.8) = 2 -> "2"
            _make_result(intrinsic_dimension=3.0),  # floor(3.0) = 3 -> "3"
            _make_result(intrinsic_dimension=3.2),  # floor(3.2) = 3 -> "3"
            _make_result(intrinsic_dimension=4.5),  # floor(4.5) = 4 >= 4 -> "4+"
        ]

        histogram = ConceptDimensionalityAnalyzer._dimension_histogram(results)

        assert histogram["1"] == 2
        assert histogram["2"] == 2
        assert histogram["3"] == 2
        assert histogram["4+"] == 1


class TestMeanDimension:
    """Tests for _mean_dimension static method."""

    def test_empty_results_returns_none(self):
        """Test mean dimension with no results."""
        mean = ConceptDimensionalityAnalyzer._mean_dimension([])
        assert mean is None

    def test_single_result(self):
        """Test mean dimension with single result."""
        results = [_make_result(intrinsic_dimension=2.5)]

        mean = ConceptDimensionalityAnalyzer._mean_dimension(results)

        assert abs(mean - 2.5) <= _scalar_tol(mean)

    def test_multiple_results(self):
        """Test mean dimension with multiple results."""
        results = [
            _make_result(intrinsic_dimension=1.0),
            _make_result(intrinsic_dimension=2.0),
            _make_result(intrinsic_dimension=3.0),
        ]

        mean = ConceptDimensionalityAnalyzer._mean_dimension(results)

        assert abs(mean - 2.0) <= _scalar_tol(mean)


class TestWeightedMeanDimension:
    """Tests for _weighted_mean_dimension static method."""

    def test_empty_results_returns_none(self):
        """Test weighted mean with no results."""
        mean = ConceptDimensionalityAnalyzer._weighted_mean_dimension([])
        assert mean is None

    def test_no_calibration_weights_returns_none(self):
        """Test weighted mean returns None when no weights."""
        results = [_make_result(intrinsic_dimension=2.0, calibration_weight=None)]

        mean = ConceptDimensionalityAnalyzer._weighted_mean_dimension(results)

        assert mean is None

    def test_weighted_mean_calculation(self):
        """Test weighted mean calculation."""
        results = [
            _make_result(intrinsic_dimension=1.0, calibration_weight=1.0),
            _make_result(intrinsic_dimension=3.0, calibration_weight=1.0),
        ]

        mean = ConceptDimensionalityAnalyzer._weighted_mean_dimension(results)

        # (1.0*1.0 + 3.0*1.0) / (1.0 + 1.0) = 2.0
        assert abs(mean - 2.0) <= _scalar_tol(mean)

    def test_weighted_mean_with_unequal_weights(self):
        """Test weighted mean with unequal weights."""
        results = [
            _make_result(intrinsic_dimension=1.0, calibration_weight=3.0),
            _make_result(intrinsic_dimension=5.0, calibration_weight=1.0),
        ]

        mean = ConceptDimensionalityAnalyzer._weighted_mean_dimension(results)

        # (1.0*3.0 + 5.0*1.0) / (3.0 + 1.0) = 8.0 / 4.0 = 2.0
        assert abs(mean - 2.0) <= _scalar_tol(mean)


class TestConceptDimensionalityStudy:
    """Tests for ConceptDimensionalityStudy.summarize."""

    def test_empty_reports(self):
        """Test summarizing empty reports list."""
        summary = ConceptDimensionalityStudy.summarize([])

        assert summary.layers == []
        assert summary.layer_summaries == []
        assert summary.bottleneck_layer is None
        assert summary.bottleneck_mean_dimension is None
        assert summary.collapse_ratio is None

    def test_single_layer_report(self):
        """Test summarizing single layer report."""
        reports = [_make_report(layer=5, mean_dimension=2.5)]

        summary = ConceptDimensionalityStudy.summarize(reports)

        assert summary.layers == [5]
        assert len(summary.layer_summaries) == 1
        assert summary.layer_summaries[0].layer == 5
        assert summary.layer_summaries[0].mean_dimension == 2.5
        assert summary.bottleneck_layer == 5
        assert summary.bottleneck_mean_dimension == 2.5

    def test_multiple_layers_finds_bottleneck(self):
        """Test that bottleneck is found as minimum mean dimension."""
        reports = [
            _make_report(layer=0, mean_dimension=3.0),
            _make_report(layer=5, mean_dimension=2.0),  # Bottleneck
            _make_report(layer=10, mean_dimension=2.5),
        ]

        summary = ConceptDimensionalityStudy.summarize(reports)

        assert summary.bottleneck_layer == 5
        assert summary.bottleneck_mean_dimension == 2.0

    def test_collapse_ratio_calculation(self):
        """Test collapse ratio = bottleneck / endpoint."""
        reports = [
            _make_report(layer=0, mean_dimension=4.0),
            _make_report(layer=5, mean_dimension=2.0),  # Bottleneck
            _make_report(layer=10, mean_dimension=4.0),
        ]

        summary = ConceptDimensionalityStudy.summarize(reports)

        # endpoint_mean = (4.0 + 4.0) / 2 = 4.0
        # collapse_ratio = 2.0 / 4.0 = 0.5
        assert abs(summary.endpoint_mean_dimension - 4.0) <= _scalar_tol(
            summary.endpoint_mean_dimension
        )
        assert abs(summary.collapse_ratio - 0.5) <= _scalar_tol(summary.collapse_ratio)

    def test_layers_sorted_in_summary(self):
        """Test that layers are sorted in summary."""
        reports = [
            _make_report(layer=10, mean_dimension=2.0),
            _make_report(layer=0, mean_dimension=3.0),
            _make_report(layer=5, mean_dimension=2.5),
        ]

        summary = ConceptDimensionalityStudy.summarize(reports)

        assert summary.layers == [0, 5, 10]


class TestLayerDimensionalitySummary:
    """Tests for LayerDimensionalitySummary dataclass."""

    def test_summary_creation(self):
        """Test creating a layer summary."""
        summary = LayerDimensionalitySummary(
            layer=5,
            mean_dimension=2.5,
            dimension_histogram={"1": 1, "2": 3, "3": 1, "4+": 0},
            domain_mean_dimensions={"refusal": 2.0, "compliance": 3.0},
            domain_rank=["refusal", "compliance"],
        )

        assert summary.layer == 5
        assert summary.mean_dimension == 2.5
        assert summary.domain_rank == ["refusal", "compliance"]


class TestDomainRankCorrelation:
    """Tests for DomainRankCorrelation dataclass."""

    def test_correlation_creation(self):
        """Test creating a correlation record."""
        corr = DomainRankCorrelation(
            layer_a=0,
            layer_b=10,
            domain_count=5,
            spearman=0.9,
        )

        assert corr.layer_a == 0
        assert corr.layer_b == 10
        assert corr.domain_count == 5
        assert corr.spearman == 0.9

    def test_correlation_with_none_spearman(self):
        """Test correlation with insufficient data."""
        corr = DomainRankCorrelation(
            layer_a=0,
            layer_b=5,
            domain_count=1,  # Too few for correlation
            spearman=None,
        )

        assert corr.spearman is None


# Helper functions to create test objects


def _make_result(
    probe_id: str = "test",
    name: str = "Test",
    source: str = "safety",
    domain: str = "refusal",
    category: str = "explicit",
    layer: int = 0,
    support_text_count: int = 5,
    sample_count: int = 5,
    usable_count: int = 5,
    intrinsic_dimension: float = 2.0,
    calibration_weight: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> ConceptDimensionalityResult:
    """Create a test result with defaults."""
    return ConceptDimensionalityResult(
        probe_id=probe_id,
        name=name,
        source=source,
        domain=domain,
        category=category,
        layer=layer,
        support_text_count=support_text_count,
        sample_count=sample_count,
        usable_count=usable_count,
        intrinsic_dimension=intrinsic_dimension,
        calibration_weight=calibration_weight,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _make_report(
    layer: int = 0,
    total_probes: int = 10,
    analyzed_count: int = 8,
    skipped_count: int = 2,
    mean_dimension: float | None = 2.5,
    weighted_mean_dimension: float | None = None,
    dimension_histogram: dict[str, int] | None = None,
    domain_summaries: list[DomainSummary] | None = None,
    results: list[ConceptDimensionalityResult] | None = None,
    skipped: list[SkippedProbe] | None = None,
) -> ConceptDimensionalityReport:
    """Create a test report with defaults."""
    if dimension_histogram is None:
        dimension_histogram = {"1": 2, "2": 3, "3": 2, "4+": 1}
    if domain_summaries is None:
        domain_summaries = []
    if results is None:
        results = []
    if skipped is None:
        skipped = []

    return ConceptDimensionalityReport(
        layer=layer,
        total_probes=total_probes,
        analyzed_count=analyzed_count,
        skipped_count=skipped_count,
        mean_dimension=mean_dimension,
        weighted_mean_dimension=weighted_mean_dimension,
        dimension_histogram=dimension_histogram,
        domain_summaries=domain_summaries,
        results=results,
        skipped=skipped,
    )
