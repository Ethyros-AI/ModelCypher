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

"""Tests for geometric confidence computation.

Key principles verified:
1. No vibes - only raw measurements returned
2. Confidence IS geometry - derived from geometric signals
3. No interpretation strings - all values are floats
"""

import pytest

from modelcypher.core.use_cases.merge.confidence import (
    compute_geometric_confidence_from_transplant,
    compute_mean_confidence,
)

# NOTE: compute_safety_verdict was REMOVED.
# Categorical verdicts ("healthy", "degenerate", "collapsed") violate the
# "no vibes" principle. Use raw measurements from geometry_metrics instead.


class TestComputeGeometricConfidenceFromTransplant:
    """Tests for compute_geometric_confidence_from_transplant."""

    def test_empty_metrics_returns_zeros(self):
        """Empty transplant metrics should return zero values."""
        result = compute_geometric_confidence_from_transplant({})

        assert result["mean_preserved_fraction"] == 0.0
        assert result["mean_cka_after"] == 0.0
        assert result["mean_projection_loss"] == 0.0
        assert result["transplant_ratio"] == 0.0

    def test_full_metrics_returns_correct_averages(self):
        """Full metrics should compute correct averages."""
        metrics = {
            "preserved_fractions": [0.8, 0.9, 0.7],
            "cka_after": [0.95, 0.90, 0.85],
            "projection_losses": [0.1, 0.2, 0.15],
            "null_dims": [10, 20, 15],
            "shared_subspace_dimensions": [50, 60, 55],
            "weights_transplanted": 30,
            "weights_considered": 50,
            "layers_transplanted": 3,
            "layers_considered": 5,
        }

        result = compute_geometric_confidence_from_transplant(metrics)

        assert result["mean_preserved_fraction"] == pytest.approx(0.8, rel=0.01)
        assert result["mean_cka_after"] == pytest.approx(0.9, rel=0.01)
        assert result["mean_projection_loss"] == pytest.approx(0.15, rel=0.01)
        assert result["transplant_ratio"] == pytest.approx(0.6, rel=0.01)
        assert result["mean_null_dim"] == pytest.approx(15.0, rel=0.01)
        assert result["mean_shared_subspace_dim"] == pytest.approx(55.0, rel=0.01)
        assert result["layers_transplanted"] == 3
        assert result["layers_considered"] == 5

    def test_returns_only_numeric_values(self):
        """All returned values should be numeric, no interpretation strings."""
        metrics = {
            "preserved_fractions": [0.5],
            "cka_after": [0.8],
            "weights_transplanted": 10,
            "weights_considered": 20,
        }

        result = compute_geometric_confidence_from_transplant(metrics)

        # All values should be int or float, not strings like "good" or "poor"
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"

    def test_handles_single_value_lists(self):
        """Should handle single-element lists correctly."""
        metrics = {
            "preserved_fractions": [0.75],
            "cka_after": [0.92],
        }

        result = compute_geometric_confidence_from_transplant(metrics)

        assert result["mean_preserved_fraction"] == 0.75
        assert result["mean_cka_after"] == 0.92

    def test_division_by_zero_protection(self):
        """Should not crash when weights_considered is 0."""
        metrics = {
            "weights_transplanted": 10,
            "weights_considered": 0,
        }

        result = compute_geometric_confidence_from_transplant(metrics)

        # Should use max(0, 1) = 1 to avoid division by zero
        assert result["transplant_ratio"] == 10.0


class TestComputeMeanConfidence:
    """Tests for compute_mean_confidence.

    mean_confidence IS mean_preserved_fraction - the geometric reality
    of how much manifold structure survived the null-space projection.
    No weighted combinations, no interpretation layers.
    """

    def test_returns_preserved_fraction_directly(self):
        """Confidence IS the preserved fraction - no interpretation."""
        geometry_metrics = {
            "mean_preserved_fraction": 0.85,
            "mean_cka_after": 0.92,  # Ignored
            "transplant_ratio": 0.75,  # Ignored
        }

        confidence = compute_mean_confidence(geometry_metrics)

        # Returns preserved_fraction directly, not a weighted combination
        assert confidence == 0.85

    def test_perfect_preservation_gives_one(self):
        """100% preservation = 1.0 confidence."""
        geometry_metrics = {"mean_preserved_fraction": 1.0}
        assert compute_mean_confidence(geometry_metrics) == 1.0

    def test_zero_preservation_gives_zero(self):
        """0% preservation = 0.0 confidence."""
        geometry_metrics = {"mean_preserved_fraction": 0.0}
        assert compute_mean_confidence(geometry_metrics) == 0.0

    def test_handles_missing_key(self):
        """Missing preserved_fraction defaults to 0."""
        geometry_metrics = {}
        assert compute_mean_confidence(geometry_metrics) == 0.0

    def test_ignores_other_signals(self):
        """Only uses preserved_fraction, ignores other metrics."""
        # High CKA and transplant ratio, but low preservation
        geometry_metrics = {
            "mean_preserved_fraction": 0.2,
            "mean_cka_after": 0.99,
            "transplant_ratio": 0.99,
        }

        # Returns the geometric truth: only 20% was preserved
        assert compute_mean_confidence(geometry_metrics) == 0.2


class TestIntegration:
    """Integration tests for the full geometric confidence flow."""

    def test_full_flow_high_preservation_merge(self):
        """Test full flow for a merge with high preservation."""
        transplant_metrics = {
            "preserved_fractions": [0.85, 0.90, 0.88],
            "cka_after": [0.95, 0.92, 0.94],
            "projection_losses": [0.05, 0.08, 0.06],
            "weights_transplanted": 45,
            "weights_considered": 50,
            "layers_transplanted": 3,
            "layers_considered": 3,
        }

        geometry = compute_geometric_confidence_from_transplant(transplant_metrics)
        confidence = compute_mean_confidence(geometry)

        # Confidence IS preserved_fraction (mean of 0.85, 0.90, 0.88)
        assert confidence == pytest.approx(0.8767, rel=0.01)
        # Raw measurements available for caller interpretation
        assert geometry["mean_preserved_fraction"] == pytest.approx(0.8767, rel=0.01)
        assert geometry["transplant_ratio"] == pytest.approx(0.9, rel=0.01)

    def test_full_flow_low_preservation_merge(self):
        """Test full flow for a merge with low preservation."""
        transplant_metrics = {
            "preserved_fractions": [0.3, 0.4, 0.35],
            "cka_after": [0.6, 0.55, 0.58],
            "weights_transplanted": 30,
            "weights_considered": 50,
        }

        geometry = compute_geometric_confidence_from_transplant(transplant_metrics)
        confidence = compute_mean_confidence(geometry)

        # Confidence IS preserved_fraction (mean of 0.3, 0.4, 0.35)
        assert confidence == pytest.approx(0.35, rel=0.01)
        # Raw measurements - caller interprets what this means
        assert geometry["mean_preserved_fraction"] == pytest.approx(0.35, rel=0.01)

    def test_full_flow_failed_merge(self):
        """Test full flow for a failed merge (nothing transplanted)."""
        transplant_metrics = {
            "preserved_fractions": [],
            "cka_after": [],
            "weights_transplanted": 0,
            "weights_considered": 50,
        }

        geometry = compute_geometric_confidence_from_transplant(transplant_metrics)
        confidence = compute_mean_confidence(geometry)

        assert confidence == 0.0
        assert geometry["transplant_ratio"] == 0.0
