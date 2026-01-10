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

"""Deviation budget tracking tests."""

import unittest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain.geometry.deviation_budget import (
    DeviationBudget,
    BudgetStatus,
    ScaleRecommendation,
    MERGE_BUDGET_THRESHOLD,
    MERGE_BUDGET_WARNING,
    INJECTION_SCALE_SAFE,
    INJECTION_SCALE_MAX,
)


class TestDeviationBudgetMerge(unittest.TestCase):
    """Tests for merge budget tracking."""

    def setUp(self):
        self.budget = DeviationBudget()
        # Create baseline weights
        self.baseline_weights = {
            "layer1.weight": mx.ones((10, 10)),
            "layer2.weight": mx.zeros((10, 10)),
        }
        self.budget.record_baseline(self.baseline_weights)

    def test_baseline_recording(self):
        """Test that baseline is recorded correctly."""
        self.assertIn("default", self.budget._baseline_weights)
        self.assertEqual(self.budget._cumulative_deviation, 0.0)

    def test_no_deviation_from_baseline(self):
        """Test that identical weights have zero deviation."""
        deviation = self.budget.compute_deviation(self.baseline_weights)
        self.assertAlmostEqual(deviation, 0.0, places=5)

    def test_deviation_computation(self):
        """Test deviation is computed correctly."""
        # Modify weights
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 1.0,  # Add 1.0 to each element
            "layer2.weight": mx.zeros((10, 10)),
        }
        deviation = self.budget.compute_deviation(modified_weights)
        # 10*10 = 100 elements, each changed by 1.0
        # sqrt(100 * 1^2) = 10.0
        self.assertAlmostEqual(deviation, 10.0, places=4)

    def test_safe_merge_budget(self):
        """Test budget check for safe merge."""
        # Small deviation - should be safe
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 0.1,
            "layer2.weight": mx.zeros((10, 10)),
        }
        status = self.budget.check_merge_budget(modified_weights)

        self.assertIsInstance(status, BudgetStatus)
        self.assertTrue(status.is_safe)
        self.assertLess(status.budget_used_percent, 100.0)

    def test_warning_threshold(self):
        """Test that warning threshold is detected."""
        # Create deviation close to warning threshold
        # Need sqrt(n) * delta = 35, so for 100 elements, delta = 3.5
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 3.6,
            "layer2.weight": mx.zeros((10, 10)),
        }
        status = self.budget.check_merge_budget(modified_weights)

        self.assertTrue(status.is_safe)
        self.assertIn("Approaching budget", status.recommendation)

    def test_exceeded_budget(self):
        """Test budget exceeded detection."""
        # Create large deviation > 50
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 6.0,  # sqrt(100) * 6 = 60
            "layer2.weight": mx.zeros((10, 10)),
        }
        status = self.budget.check_merge_budget(modified_weights)

        self.assertFalse(status.is_safe)
        self.assertIn("Budget exceeded", status.recommendation)
        self.assertGreater(status.budget_used_percent, 100.0)

    def test_suggest_delta_scale(self):
        """Test delta scale suggestion."""
        # Propose a delta that would exceed budget
        delta = {
            "layer1.weight": mx.ones((10, 10)) * 10.0,  # Would add 100 L2
        }
        suggested_scale = self.budget.suggest_delta_scale(delta)

        # Should suggest a scale < 1.0 to stay within budget
        self.assertLess(suggested_scale, 1.0)
        self.assertGreater(suggested_scale, 0.0)


class TestDeviationBudgetInjection(unittest.TestCase):
    """Tests for injection scale tracking."""

    def setUp(self):
        self.budget = DeviationBudget()

    def test_safe_injection_scale(self):
        """Test that small injection scale is safe."""
        embedding = mx.ones((1, 1024)) * 0.1
        layer_activations = mx.ones((1, 1024)) * 1.0

        status = self.budget.check_injection_scale(
            embedding, layer_activations, scale=1.0
        )

        self.assertIsInstance(status, BudgetStatus)
        self.assertTrue(status.is_safe)

    def test_unsafe_injection_scale(self):
        """Test that large injection scale is detected as unsafe."""
        embedding = mx.ones((1, 1024)) * 1.0
        layer_activations = mx.ones((1, 1024)) * 0.1

        status = self.budget.check_injection_scale(
            embedding, layer_activations, scale=100.0
        )

        self.assertFalse(status.is_safe)

    def test_null_space_allows_higher_scale(self):
        """Test that null-space projection allows higher scales."""
        embedding = mx.ones((1, 1024)) * 1.0
        layer_activations = mx.ones((1, 1024)) * 1.0

        # Without null-space
        status_full = self.budget.check_injection_scale(
            embedding, layer_activations, scale=8.0, use_null_space=False
        )

        # With null-space
        status_null = self.budget.check_injection_scale(
            embedding, layer_activations, scale=8.0, use_null_space=True
        )

        # Null-space should have lower budget usage (2x threshold)
        self.assertLess(
            status_null.budget_used_percent,
            status_full.budget_used_percent
        )

    def test_recommend_scale(self):
        """Test scale recommendation."""
        embedding = mx.ones((1, 1024)) * 1.0
        layer_activations = mx.ones((1, 1024)) * 0.5

        recommendation = self.budget.recommend_scale(
            embedding, layer_activations, target_budget_percent=50.0
        )

        self.assertIsInstance(recommendation, ScaleRecommendation)
        self.assertGreater(recommendation.scale, 0.0)
        self.assertGreater(recommendation.max_safe_scale, recommendation.scale)


class TestDeviationBudgetConstants(unittest.TestCase):
    """Test that constants are set to empirically derived values."""

    def test_merge_threshold(self):
        """Test merge threshold matches empirical findings."""
        self.assertEqual(MERGE_BUDGET_THRESHOLD, 50.0)

    def test_warning_threshold(self):
        """Test warning threshold is below merge threshold."""
        self.assertLess(MERGE_BUDGET_WARNING, MERGE_BUDGET_THRESHOLD)
        self.assertEqual(MERGE_BUDGET_WARNING, 35.0)

    def test_injection_safe_scale(self):
        """Test injection safe scale matches findings."""
        self.assertEqual(INJECTION_SCALE_SAFE, 5.0)

    def test_injection_max_scale(self):
        """Test max injection scale before degeneration."""
        self.assertGreater(INJECTION_SCALE_MAX, INJECTION_SCALE_SAFE)
        self.assertEqual(INJECTION_SCALE_MAX, 10.0)


class TestDeviationBudgetNamedBaselines(unittest.TestCase):
    """Test multiple named baselines."""

    def setUp(self):
        self.budget = DeviationBudget()

    def test_multiple_baselines(self):
        """Test recording and using multiple baselines."""
        baseline_a = {"weight": mx.ones((5, 5))}
        baseline_b = {"weight": mx.zeros((5, 5))}

        self.budget.record_baseline(baseline_a, name="model_a")
        self.budget.record_baseline(baseline_b, name="model_b")

        # Test weights closer to baseline_a (ones)
        test_weights = {"weight": mx.ones((5, 5)) * 0.9}

        deviation_a = self.budget.compute_deviation(test_weights, baseline_name="model_a")
        deviation_b = self.budget.compute_deviation(test_weights, baseline_name="model_b")

        # Deviation from model_a (ones) should be smaller since test is closer to ones
        # sqrt(25 * 0.1^2) = 0.5 vs sqrt(25 * 0.9^2) = 4.5
        self.assertLess(deviation_a, deviation_b)

    def test_missing_baseline_warning(self):
        """Test warning when baseline doesn't exist."""
        deviation = self.budget.compute_deviation({}, baseline_name="nonexistent")
        self.assertEqual(deviation, 0.0)


if __name__ == "__main__":
    unittest.main()
