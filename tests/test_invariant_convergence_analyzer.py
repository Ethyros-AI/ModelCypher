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
Tests for InvariantConvergenceAnalyzer.

Tests the full implementation with FamilyResult, AlignMode, and CSV export.
"""
import pytest
import math
from dataclasses import dataclass, field
from typing import Dict, List

from modelcypher.core.domain.geometry.invariant_convergence_analyzer import (
    InvariantConvergenceAnalyzer,
    AlignMode,
    AlignmentPair,
    FamilyResult,
    Report,
    Summary,
    ConvergenceMetric,
    ConvergenceReport,
)


# =============================================================================
# Mock Fingerprint Classes
# =============================================================================


@dataclass
class MockFingerprint:
    """Mock fingerprint for testing."""
    prime_id: str
    activation_vectors: Dict[int, Dict[int, float]]  # layer -> dim -> value


@dataclass
class MockModelFingerprints:
    """Mock model fingerprints for testing."""
    model_id: str
    fingerprints: List[MockFingerprint]
    layer_count: int = 32


# =============================================================================
# Tests
# =============================================================================


class TestAlignMode:
    """Tests for alignment mode enum."""
    
    def test_layer_mode(self):
        """LAYER mode should be string."""
        assert AlignMode.LAYER.value == "layer"
    
    def test_normalized_mode(self):
        """NORMALIZED mode should be string."""
        assert AlignMode.NORMALIZED.value == "normalized"


class TestAlignmentPair:
    """Tests for layer alignment pairs."""
    
    def test_create_pair(self):
        """Should create alignment pair."""
        pair = AlignmentPair(
            index=5,
            source_layer=10,
            target_layer=8,
            normalized_depth=0.5,
        )
        
        assert pair.index == 5
        assert pair.source_layer == 10
        assert pair.target_layer == 8
        assert pair.normalized_depth == 0.5


class TestFamilyResult:
    """Tests for family convergence result."""
    
    def test_create_result(self):
        """Should create family result."""
        result = FamilyResult(
            anchor_ids=["fibonacci_1", "fibonacci_2"],
            layers={"5": 0.95, "10": 0.88},
            mean_cosine=0.915,
        )
        
        assert len(result.anchor_ids) == 2
        assert result.layers["5"] == 0.95
        assert result.mean_cosine == 0.915
    
    def test_no_mean_cosine(self):
        """Should allow None mean cosine."""
        result = FamilyResult(
            anchor_ids=[],
            layers={},
            mean_cosine=None,
        )
        
        assert result.mean_cosine is None


class TestInvariantConvergenceAnalyzer:
    """Tests for the main analyzer class."""
    
    def test_default_thresholds(self):
        """Should have default thresholds for known families."""
        analyzer = InvariantConvergenceAnalyzer()
        
        assert "fibonacci" in analyzer.thresholds
        assert "lucas" in analyzer.thresholds
        assert "primes" in analyzer.thresholds
        assert "logic" in analyzer.thresholds
    
    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        custom = {"custom_family": 0.9}
        analyzer = InvariantConvergenceAnalyzer(thresholds=custom)
        
        assert analyzer.thresholds == custom
    
    def test_scaled_index_middle(self):
        """Scaled index should interpolate correctly."""
        analyzer = InvariantConvergenceAnalyzer()
        
        # Middle position in 10-aligned to 20-total
        result = analyzer._scaled_index(5, 10, 20)
        
        # fraction = 5/9 ≈ 0.556, scaled ≈ 10.5, rounded = 11
        assert 9 <= result <= 12
    
    def test_scaled_index_start(self):
        """Scaled index at start should be 0."""
        analyzer = InvariantConvergenceAnalyzer()
        
        result = analyzer._scaled_index(0, 10, 20)
        assert result == 0
    
    def test_scaled_index_end(self):
        """Scaled index at end should be last."""
        analyzer = InvariantConvergenceAnalyzer()
        
        result = analyzer._scaled_index(9, 10, 20)
        assert result == 19  # 20-1
    
    def test_cosine_sparse_identical(self):
        """Identical sparse vectors should give cosine = 1."""
        analyzer = InvariantConvergenceAnalyzer()
        
        vec = {0: 1.0, 1: 2.0, 2: 3.0}
        result = analyzer._cosine_sparse(vec, vec)
        
        assert result is not None
        assert abs(result - 1.0) < 0.001
    
    def test_cosine_sparse_orthogonal(self):
        """Orthogonal sparse vectors should give cosine = 0."""
        analyzer = InvariantConvergenceAnalyzer()
        
        a = {0: 1.0}
        b = {1: 1.0}
        result = analyzer._cosine_sparse(a, b)
        
        assert result is not None
        assert abs(result) < 0.001
    
    def test_cosine_sparse_empty(self):
        """Empty vectors should return None."""
        analyzer = InvariantConvergenceAnalyzer()
        
        assert analyzer._cosine_sparse({}, {0: 1.0}) is None
        assert analyzer._cosine_sparse({0: 1.0}, {}) is None
    
    def test_average_sparse_single(self):
        """Averaging single vector should return itself."""
        analyzer = InvariantConvergenceAnalyzer()
        
        vec = [{0: 2.0, 1: 4.0}]
        result = analyzer._average_sparse(vec)
        
        assert result == {0: 2.0, 1: 4.0}
    
    def test_average_sparse_multiple(self):
        """Averaging multiple vectors should give mean."""
        analyzer = InvariantConvergenceAnalyzer()
        
        vecs = [
            {0: 1.0, 1: 2.0},
            {0: 3.0, 1: 4.0},
        ]
        result = analyzer._average_sparse(vecs)
        
        assert result[0] == 2.0
        assert result[1] == 3.0
    
    def test_format_layer_label_layer_mode(self):
        """LAYER mode should give integer label."""
        analyzer = InvariantConvergenceAnalyzer()
        
        label = analyzer._format_layer_label(5, AlignMode.LAYER, {})
        assert label == "5"
    
    def test_format_layer_label_normalized_mode(self):
        """NORMALIZED mode should give float label."""
        analyzer = InvariantConvergenceAnalyzer()
        
        axis = {0: 0.5}
        label = analyzer._format_layer_label(0, AlignMode.NORMALIZED, axis)
        assert label == "0.5000"


class TestCSVExport:
    """Tests for CSV export functionality."""
    
    def test_csv_header(self):
        """CSV should start with header."""
        report = Report(
            models=Report.Models(model_a="a", model_b="b"),
            align_mode=AlignMode.LAYER,
            dimension_alignment=None,
            layers=[],
            families={},
            summary=Summary({}, {}),
            layer_count=0,
            source_layer_count=0,
            target_layer_count=0,
            aligned_layers=[],
        )
        
        lines = InvariantConvergenceAnalyzer.csv_lines(report)
        assert lines[0] == "family,layer,cosine"
    
    def test_csv_format(self):
        """CSV should format family results correctly."""
        report = Report(
            models=Report.Models(model_a="a", model_b="b"),
            align_mode=AlignMode.LAYER,
            dimension_alignment=None,
            layers=[5.0],
            families={
                "fibonacci": FamilyResult(
                    anchor_ids=["fibonacci_1"],
                    layers={"5": 0.95},
                    mean_cosine=0.95,
                ),
            },
            summary=Summary({"fibonacci": 0.95}, {"5": 0.95}),
            layer_count=32,
            source_layer_count=32,
            target_layer_count=32,
            aligned_layers=[],
        )
        
        lines = InvariantConvergenceAnalyzer.csv_lines(report)
        
        assert len(lines) == 2  # header + 1 data row
        assert "fibonacci,5,0.950000" in lines[1]


class TestLegacyInterface:
    """Tests for backward compatibility."""
    
    def test_convergence_metric_fields(self):
        """ConvergenceMetric should have expected fields."""
        metric = ConvergenceMetric(
            sequence_family="fibonacci",
            step=100,
            alignment_score=0.85,
            variance=0.01,
            is_converged=True,
        )
        
        assert metric.sequence_family == "fibonacci"
        assert metric.step == 100
        assert metric.is_converged is True
    
    def test_convergence_report_fields(self):
        """ConvergenceReport should have expected fields."""
        report = ConvergenceReport(
            model_id="test-model",
            metrics=[],
            overall_convergence=0.9,
            stable_families=["fibonacci"],
        )
        
        assert report.model_id == "test-model"
        assert report.stable_families == ["fibonacci"]
