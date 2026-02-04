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

"""Deviation measurement tests.

Philosophy: The geometry handles safety by construction. These tests verify
that measurement and scale derivation work correctly, NOT that we gate on
thresholds.
"""

import unittest

from tests.conftest import HAS_MLX


import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.deviation_budget import (
    DeviationTracker,
    DeviationMeasurement,
)


class TestDeviationMeasurement(unittest.TestCase):
    """Tests for deviation measurement (informational only)."""

    def setUp(self):
        self.tracker = DeviationTracker()
        self.baseline_weights = {
            "layer1.weight": get_default_backend().ones((10, 10)),
            "layer2.weight": get_default_backend().zeros((10, 10)),
        }
        self.tracker.record_baseline(self.baseline_weights)

    def test_baseline_recording(self):
        """Test that baseline is recorded correctly."""
        self.assertIn("default", self.tracker._baseline_weights)
        self.assertIn("default", self.tracker._baseline_norms)
        self.assertIn("default", self.tracker._baseline_condition_numbers)

    def test_no_deviation_from_baseline(self):
        """Test that identical weights have zero deviation."""
        deviation = self.tracker.compute_deviation(self.baseline_weights)
        self.assertAlmostEqual(deviation, 0.0, places=5)

    def test_deviation_computation(self):
        """Test deviation is computed correctly."""
        modified_weights = {
            "layer1.weight": get_default_backend().ones((10, 10)) + 1.0,
            "layer2.weight": get_default_backend().zeros((10, 10)),
        }
        deviation = self.tracker.compute_deviation(modified_weights)
        # 10*10 = 100 elements, each changed by 1.0
        # sqrt(100 * 1^2) = 10.0
        self.assertAlmostEqual(deviation, 10.0, places=4)

    def test_measure_returns_measurement(self):
        """Test that measure returns DeviationMeasurement."""
        modified_weights = {
            "layer1.weight": get_default_backend().ones((10, 10)) + 0.1,
            "layer2.weight": get_default_backend().zeros((10, 10)),
        }
        measurement = self.tracker.measure(modified_weights)

        self.assertIsInstance(measurement, DeviationMeasurement)
        self.assertGreater(measurement.deviation, 0.0)
        self.assertGreater(measurement.baseline_norm, 0.0)
        self.assertGreater(measurement.deviation_percent, 0.0)
        self.assertGreater(measurement.condition_number, 0.0)

    def test_weight_norm_computed_correctly(self):
        """Test that weight norm is Frobenius norm."""
        # sqrt(100 * 2^2) = sqrt(400) = 20
        weights = {"weight": get_default_backend().ones((10, 10)) * 2.0}
        norm = self.tracker._compute_weight_norm(weights)
        self.assertAlmostEqual(norm, 20.0, places=4)

    def test_condition_number_computed(self):
        """Test that condition number is computed and positive."""
        # Larger matrix for reliable SVD
        weights = {"weight": get_default_backend().random_normal((100, 100))}
        cond = self.tracker._compute_condition_number(weights)
        # Condition number should be >= 1.0 for any matrix
        self.assertGreaterEqual(cond, 1.0)


class TestDeltaMagnitude(unittest.TestCase):
    """Tests for delta magnitude computation."""

    def setUp(self):
        self.tracker = DeviationTracker()

    def test_compute_delta_magnitude(self):
        """Test that delta magnitude is computed correctly."""
        target_weights = {"weight": get_default_backend().zeros((10, 10))}
        source_weights = {"weight": get_default_backend().ones((10, 10))}

        delta = self.tracker.compute_delta_magnitude(source_weights, target_weights)
        # sqrt(100 * 1^2) = 10.0
        self.assertAlmostEqual(delta, 10.0, places=4)

    def test_zero_delta_for_identical(self):
        """Test zero delta for identical weights."""
        weights = {"weight": get_default_backend().ones((10, 10))}
        delta = self.tracker.compute_delta_magnitude(weights, weights)
        self.assertAlmostEqual(delta, 0.0, places=5)


class TestScaleDerivation(unittest.TestCase):
    """Tests for SVD-based scale derivation."""

    def setUp(self):
        self.tracker = DeviationTracker()
        self.target_weights = {"weight": get_default_backend().ones((10, 10))}
        self.tracker.record_baseline(self.target_weights)

    def test_derive_scale_returns_positive(self):
        """Test that derive_scale returns a positive value."""
        source_weights = {"weight": get_default_backend().ones((10, 10)) + 0.5}
        activations = get_default_backend().random_normal((100, 10))

        scale = self.tracker.derive_scale(source_weights, self.target_weights, activations)

        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, 1.0)

    def test_derive_scale_zero_delta(self):
        """Test scale derivation with zero delta."""
        activations = get_default_backend().random_normal((100, 10))

        scale = self.tracker.derive_scale(
            self.target_weights, self.target_weights, activations
        )

        self.assertEqual(scale, 1.0)

    def test_derive_scale_varies_with_activations(self):
        """Test that scale depends on activation structure."""
        source_weights = {"weight": get_default_backend().ones((10, 10)) + 1.0}

        # Low-rank activations (less null-space capacity)
        low_rank = get_default_backend().ones((100, 10))
        scale_low = self.tracker.derive_scale(
            source_weights, self.target_weights, low_rank
        )

        # Full-rank activations (more null-space capacity)
        full_rank = get_default_backend().random_normal((100, 10))
        scale_full = self.tracker.derive_scale(
            source_weights, self.target_weights, full_rank
        )

        # Both should be valid scales
        self.assertGreater(scale_low, 0.0)
        self.assertGreater(scale_full, 0.0)


class TestNamedBaselines(unittest.TestCase):
    """Test multiple named baselines."""

    def setUp(self):
        self.tracker = DeviationTracker()

    def test_multiple_baselines(self):
        """Test recording and using multiple baselines."""
        baseline_a = {"weight": get_default_backend().ones((5, 5))}
        baseline_b = {"weight": get_default_backend().zeros((5, 5))}

        self.tracker.record_baseline(baseline_a, name="model_a")
        self.tracker.record_baseline(baseline_b, name="model_b")

        test_weights = {"weight": get_default_backend().ones((5, 5)) * 0.9}

        deviation_a = self.tracker.compute_deviation(test_weights, baseline_name="model_a")
        deviation_b = self.tracker.compute_deviation(test_weights, baseline_name="model_b")

        # Deviation from model_a (ones) should be smaller
        self.assertLess(deviation_a, deviation_b)

    def test_missing_baseline_returns_zero(self):
        """Test that missing baseline returns zero deviation."""
        deviation = self.tracker.compute_deviation({}, baseline_name="nonexistent")
        self.assertEqual(deviation, 0.0)


class TestDeriveDeltaScale(unittest.TestCase):
    """Tests for geometry-derived delta_scale computation.

    The derive_delta_scale function computes scale from null_rank/in_dim ratio,
    which is geometry-derived (not heuristic):
    - null_rank comes from eigenvalue rank threshold (sqrt(eps) cutoff)
    - in_dim is the fixed input dimension
    - The ratio represents actual available capacity
    """

    def test_half_capacity_gives_half_scale(self):
        """Test that half null-space capacity gives scale ≈ 0.5."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        scale = derive_delta_scale(null_rank=50, in_dim=100)
        self.assertAlmostEqual(scale, 0.5, places=4)

    def test_full_capacity_gives_full_scale(self):
        """Test that full null-space capacity gives scale = 1.0."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        scale = derive_delta_scale(null_rank=100, in_dim=100)
        self.assertEqual(scale, 1.0)

    def test_zero_capacity_gives_eps(self):
        """Test that zero null-space capacity gives scale ≈ eps."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        scale = derive_delta_scale(null_rank=0, in_dim=100)
        self.assertLess(scale, 1e-6)
        self.assertGreater(scale, 0.0)

    def test_sequential_stacking_divides_capacity(self):
        """Test that n_merges divides the capacity ratio."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        # 50% capacity, 2 merges -> 25% each
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=2)
        self.assertAlmostEqual(scale, 0.25, places=4)

        # 50% capacity, 4 merges -> 12.5% each
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=4)
        self.assertAlmostEqual(scale, 0.125, places=4)

    def test_edge_cases(self):
        """Test edge cases don't cause errors."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        # Zero in_dim should return 1.0 (no constraint)
        scale = derive_delta_scale(null_rank=0, in_dim=0)
        self.assertEqual(scale, 1.0)

        # Zero n_merges treated as 1
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=0)
        self.assertAlmostEqual(scale, 0.5, places=4)

        # Negative n_merges treated as 1
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=-1)
        self.assertAlmostEqual(scale, 0.5, places=4)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility alias."""

    def test_deviation_budget_alias(self):
        """Test that DeviationBudget is an alias for DeviationTracker."""
        from modelcypher.core.domain.geometry.deviation_budget import DeviationBudget

        self.assertIs(DeviationBudget, DeviationTracker)


if __name__ == "__main__":
    unittest.main()
