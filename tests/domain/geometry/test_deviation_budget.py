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
    MERGE_BUDGET_TOLERANCE,
    MERGE_WARNING_TOLERANCE,
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
        # weight_norm = sqrt(100 * 1^2) = 10.0
        # threshold = 10.0 * 0.01 = 0.1
        # warning = 10.0 * 0.007 = 0.07
        # Small deviation - should be safe (< warning)
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 0.005,  # sqrt(100) * 0.005 = 0.05 < 0.07
            "layer2.weight": mx.zeros((10, 10)),
        }
        status = self.budget.check_merge_budget(modified_weights)

        self.assertIsInstance(status, BudgetStatus)
        self.assertTrue(status.is_safe)
        self.assertLess(status.budget_used_percent, 100.0)

    def test_warning_threshold(self):
        """Test that warning threshold is detected."""
        # Thresholds are now derived from weight norm:
        # weight_norm = sqrt(100 * 1^2 + 100 * 0^2) = 10.0
        # threshold = 10.0 * 0.01 = 0.1
        # warning = 10.0 * 0.007 = 0.07
        # Create deviation > warning but < threshold
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 0.008,  # sqrt(100) * 0.008 = 0.08
            "layer2.weight": mx.zeros((10, 10)),
        }
        status = self.budget.check_merge_budget(modified_weights)

        self.assertTrue(status.is_safe)
        self.assertIn("Approaching budget", status.recommendation)

    def test_exceeded_budget(self):
        """Test budget exceeded detection."""
        # Thresholds are derived from weight norm:
        # weight_norm = sqrt(100 * 1^2) = 10.0
        # threshold = 10.0 * 0.01 = 0.1
        # Create deviation > 0.1
        modified_weights = {
            "layer1.weight": mx.ones((10, 10)) + 0.02,  # sqrt(100) * 0.02 = 0.2 > 0.1
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
    """Test that constants define geometric tolerance fractions."""

    def test_merge_tolerance_fraction(self):
        """Test merge tolerance is 1% of weight norm."""
        self.assertEqual(MERGE_BUDGET_TOLERANCE, 0.01)

    def test_warning_tolerance_fraction(self):
        """Test warning tolerance is below merge tolerance."""
        self.assertLess(MERGE_WARNING_TOLERANCE, MERGE_BUDGET_TOLERANCE)
        self.assertEqual(MERGE_WARNING_TOLERANCE, 0.007)

    def test_injection_safe_scale(self):
        """Test injection safe scale matches findings."""
        self.assertEqual(INJECTION_SCALE_SAFE, 5.0)

    def test_injection_max_scale(self):
        """Test max injection scale before degeneration."""
        self.assertGreater(INJECTION_SCALE_MAX, INJECTION_SCALE_SAFE)
        self.assertEqual(INJECTION_SCALE_MAX, 10.0)


class TestGeometricThresholdDerivation(unittest.TestCase):
    """Test that thresholds are geometrically derived from weight norm."""

    def test_threshold_scales_with_weight_norm(self):
        """Test that threshold scales with model size."""
        # Small model
        small_weights = {"weight": mx.ones((10, 10))}  # norm = 10
        budget_small = DeviationBudget()
        budget_small.record_baseline(small_weights)

        # Large model (10x more weights)
        large_weights = {"weight": mx.ones((100, 100))}  # norm = 100
        budget_large = DeviationBudget()
        budget_large.record_baseline(large_weights)

        # Threshold should scale proportionally
        threshold_small = budget_small.get_threshold()
        threshold_large = budget_large.get_threshold()

        # Large model threshold should be ~10x small model threshold
        ratio = threshold_large / threshold_small
        self.assertAlmostEqual(ratio, 10.0, places=1)

    def test_threshold_is_percentage_of_norm(self):
        """Test that threshold is 1% of weight norm."""
        weights = {"weight": mx.ones((100, 100))}  # norm = 100
        budget = DeviationBudget()
        budget.record_baseline(weights)

        threshold = budget.get_threshold()
        expected = 100.0 * MERGE_BUDGET_TOLERANCE  # 100 * 0.01 = 1.0

        self.assertAlmostEqual(threshold, expected, places=4)

    def test_weight_norm_computed_correctly(self):
        """Test that weight norm is Frobenius norm."""
        # sqrt(100 * 2^2) = sqrt(400) = 20
        weights = {"weight": mx.ones((10, 10)) * 2.0}
        budget = DeviationBudget()

        norm = budget._compute_weight_norm(weights)
        self.assertAlmostEqual(norm, 20.0, places=4)

    def test_auto_compute_scale(self):
        """Test that auto_compute_scale uses measured delta."""
        budget = DeviationBudget()
        target_weights = {"weight": mx.ones((10, 10))}  # norm = 10
        budget.record_baseline(target_weights)

        # Source is different by 0.5 per element
        source_weights = {"weight": mx.ones((10, 10)) + 0.5}  # delta norm = 5

        scale = budget.auto_compute_scale(
            source_weights=source_weights,
            target_weights=target_weights,
            remaining_sources=1,
        )

        # Scale should be based on measured delta (5.0) and threshold (0.1)
        # Expected: 0.1 * 0.7 / 5.0 = 0.014, but clamped to min 0.1
        self.assertGreaterEqual(scale, 0.1)
        self.assertLessEqual(scale, 1.0)

    def test_compute_delta_magnitude(self):
        """Test that delta magnitude is computed correctly."""
        budget = DeviationBudget()
        target_weights = {"weight": mx.zeros((10, 10))}
        source_weights = {"weight": mx.ones((10, 10))}  # delta = 1 per element

        delta = budget.compute_delta_magnitude(source_weights, target_weights)
        # sqrt(100 * 1^2) = 10.0
        self.assertAlmostEqual(delta, 10.0, places=4)


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


class TestMemoryTokenScaleTracking(unittest.TestCase):
    """Test memory token scale tracking methods."""

    def setUp(self):
        self.budget = DeviationBudget()

    def test_memory_token_safe_scale(self):
        """Test that safe memory token scale is accepted."""
        # Small memory content relative to activations
        memory_content = mx.ones((1, 128)) * 0.5
        layer_activations = mx.ones((10, 128))

        status = self.budget.check_memory_token_scale(
            memory_content, layer_activations, scale=5.0, use_null_space=True
        )

        self.assertTrue(status.is_safe)
        self.assertIn("Safe", status.recommendation)

    def test_memory_token_allows_higher_scale(self):
        """Test that memory tokens allow higher scale than direct injection."""
        memory_content = mx.ones((1, 128)) * 10.0
        layer_activations = mx.ones((10, 128))

        # With null-space, memory token threshold is 20.0
        # Relative magnitude = 10 * sqrt(128) / sqrt(128) = 10.0
        status_memory = self.budget.check_memory_token_scale(
            memory_content, layer_activations, scale=10.0, use_null_space=True
        )

        # Direct injection would fail at same scale
        status_direct = self.budget.check_injection_scale(
            memory_content, layer_activations, scale=10.0, use_null_space=True
        )

        # Memory token should be safer at same scale
        self.assertLessEqual(
            status_memory.budget_used_percent,
            status_direct.budget_used_percent,
        )

    def test_memory_token_unsafe_detection(self):
        """Test detection of unsafe memory token scale."""
        # Very large memory content (30x layer norm)
        memory_content = mx.ones((1, 128)) * 30.0
        layer_activations = mx.ones((10, 128))

        status = self.budget.check_memory_token_scale(
            memory_content, layer_activations, scale=30.0, use_null_space=True
        )

        self.assertFalse(status.is_safe)
        self.assertIn("Reduce", status.recommendation)

    def test_recommend_memory_scale(self):
        """Test memory scale recommendation."""
        direction = mx.ones((1, 128))
        layer_activations = mx.ones((10, 128))

        recommendation = self.budget.recommend_memory_scale(
            direction, layer_activations, target_budget_percent=50.0
        )

        self.assertIsInstance(recommendation, ScaleRecommendation)
        self.assertGreater(recommendation.scale, 0.0)
        # Memory tokens always recommend null-space
        self.assertTrue(recommendation.use_null_space)


class TestMemoryTokenConstants(unittest.TestCase):
    """Test memory token threshold constants."""

    def test_memory_token_safe_scale_constant(self):
        """Test memory token safe scale is 10.0."""
        from modelcypher.core.domain.geometry.deviation_budget import (
            MEMORY_TOKEN_SCALE_SAFE,
        )
        self.assertEqual(MEMORY_TOKEN_SCALE_SAFE, 10.0)

    def test_memory_token_max_scale_constant(self):
        """Test memory token max scale is 20.0."""
        from modelcypher.core.domain.geometry.deviation_budget import (
            MEMORY_TOKEN_SCALE_MAX,
        )
        self.assertEqual(MEMORY_TOKEN_SCALE_MAX, 20.0)

    def test_memory_token_is_4x_injection(self):
        """Test memory token max is 4x direct injection safe."""
        from modelcypher.core.domain.geometry.deviation_budget import (
            INJECTION_SCALE_SAFE,
            MEMORY_TOKEN_SCALE_MAX,
        )
        # Memory: 20.0, Direct: 5.0 -> 4x
        self.assertEqual(MEMORY_TOKEN_SCALE_MAX / INJECTION_SCALE_SAFE, 4.0)


if __name__ == "__main__":
    unittest.main()
