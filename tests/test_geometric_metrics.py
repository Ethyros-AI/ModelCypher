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

"""Tests for geometric metric aggregation.

Key principles verified:
1. No interpretation strings - all values are floats
2. Raw measurements only - no qualitative labels
"""

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.use_cases.merge.metrics import (
    compute_geometric_metrics_from_transplant,
)


class TestComputeGeometricMetricsFromTransplant:
    """Tests for compute_geometric_metrics_from_transplant."""

    def test_empty_metrics_returns_zeros(self):
        """Empty transplant metrics should return zero values."""
        result = compute_geometric_metrics_from_transplant({})

        assert result["mean_preserved_fraction"] == 0.0
        assert result["mean_cka_after"] == 0.0
        assert result["mean_projection_loss"] == 0.0
        assert result["transplant_ratio"] == 0.0

    def test_full_metrics_returns_correct_averages(self):
        """Full metrics should compute correct averages."""
        backend = get_default_backend()
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

        result = compute_geometric_metrics_from_transplant(metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))

        assert abs(result["mean_preserved_fraction"] - 0.8) <= eps
        assert abs(result["mean_cka_after"] - 0.9) <= eps
        assert abs(result["mean_projection_loss"] - 0.15) <= eps
        assert abs(result["transplant_ratio"] - 0.6) <= eps
        assert abs(result["mean_null_dim"] - 15.0) <= eps
        assert abs(result["mean_shared_subspace_dim"] - 55.0) <= eps
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

        result = compute_geometric_metrics_from_transplant(metrics)

        # All values should be int or float, not strings like "good" or "poor"
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"

    def test_handles_single_value_lists(self):
        """Should handle single-element lists correctly."""
        backend = get_default_backend()
        metrics = {
            "preserved_fractions": [0.75],
            "cka_after": [0.92],
        }

        result = compute_geometric_metrics_from_transplant(metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))

        assert abs(result["mean_preserved_fraction"] - 0.75) <= eps
        assert abs(result["mean_cka_after"] - 0.92) <= eps

    def test_division_by_zero_protection(self):
        """Should not crash when weights_considered is 0."""
        backend = get_default_backend()
        metrics = {
            "weights_transplanted": 10,
            "weights_considered": 0,
        }

        result = compute_geometric_metrics_from_transplant(metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))

        # Should use max(0, 1) = 1 to avoid division by zero
        assert abs(result["transplant_ratio"] - 10.0) <= eps


class TestIntegration:
    """Integration tests for the full geometric metric flow."""

    def test_full_flow_high_preservation_merge(self):
        """Test full flow for a merge with high preservation."""
        backend = get_default_backend()
        transplant_metrics = {
            "preserved_fractions": [0.85, 0.90, 0.88],
            "cka_after": [0.95, 0.92, 0.94],
            "projection_losses": [0.05, 0.08, 0.06],
            "weights_transplanted": 45,
            "weights_considered": 50,
            "layers_transplanted": 3,
            "layers_considered": 3,
        }

        geometry = compute_geometric_metrics_from_transplant(transplant_metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))
        expected_preserved = sum(transplant_metrics["preserved_fractions"]) / len(
            transplant_metrics["preserved_fractions"]
        )

        # Raw measurements available for caller interpretation
        assert abs(geometry["mean_preserved_fraction"] - expected_preserved) <= eps
        assert abs(geometry["transplant_ratio"] - 0.9) <= eps

    def test_full_flow_low_preservation_merge(self):
        """Test full flow for a merge with low preservation."""
        backend = get_default_backend()
        transplant_metrics = {
            "preserved_fractions": [0.3, 0.4, 0.35],
            "cka_after": [0.6, 0.55, 0.58],
            "weights_transplanted": 30,
            "weights_considered": 50,
        }

        geometry = compute_geometric_metrics_from_transplant(transplant_metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))
        expected_preserved = sum(transplant_metrics["preserved_fractions"]) / len(
            transplant_metrics["preserved_fractions"]
        )

        # Raw measurements - caller interprets what this means
        assert abs(geometry["mean_preserved_fraction"] - expected_preserved) <= eps

    def test_full_flow_failed_merge(self):
        """Test full flow for a failed merge (nothing transplanted)."""
        backend = get_default_backend()
        transplant_metrics = {
            "preserved_fractions": [],
            "cka_after": [],
            "weights_transplanted": 0,
            "weights_considered": 50,
        }

        geometry = compute_geometric_metrics_from_transplant(transplant_metrics)
        eps = machine_epsilon(backend, backend.array([0.0]))

        assert abs(geometry["transplant_ratio"] - 0.0) <= eps
