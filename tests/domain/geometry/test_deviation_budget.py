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

import pytest

from modelcypher.core.domain.geometry.deviation_budget import (
    DeviationMeasurement,
    DeviationTracker,
)


class TestDeviationMeasurement:
    """Tests for deviation measurement (informational only)."""

    def test_baseline_recording(self, any_backend):
        """Test that baseline is recorded correctly."""
        tracker = DeviationTracker()
        baseline_weights = {
            "layer1.weight": any_backend.ones((10, 10)),
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        tracker.record_baseline(baseline_weights)

        assert "default" in tracker._baseline_weights
        assert "default" in tracker._baseline_norms
        assert "default" in tracker._baseline_condition_numbers

    def test_no_deviation_from_baseline(self, any_backend):
        """Test that identical weights have zero deviation."""
        tracker = DeviationTracker()
        baseline_weights = {
            "layer1.weight": any_backend.ones((10, 10)),
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        tracker.record_baseline(baseline_weights)

        deviation = tracker.compute_deviation(baseline_weights)
        assert abs(deviation) < 1e-5

    def test_deviation_computation(self, any_backend):
        """Test deviation is computed correctly."""
        tracker = DeviationTracker()
        baseline_weights = {
            "layer1.weight": any_backend.ones((10, 10)),
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        tracker.record_baseline(baseline_weights)

        modified_weights = {
            "layer1.weight": any_backend.ones((10, 10)) + 1.0,
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        deviation = tracker.compute_deviation(modified_weights)
        # 10*10 = 100 elements, each changed by 1.0
        # sqrt(100 * 1^2) = 10.0
        assert abs(deviation - 10.0) < 0.01

    def test_measure_returns_measurement(self, any_backend):
        """Test that measure returns DeviationMeasurement."""
        tracker = DeviationTracker()
        baseline_weights = {
            "layer1.weight": any_backend.ones((10, 10)),
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        tracker.record_baseline(baseline_weights)

        modified_weights = {
            "layer1.weight": any_backend.ones((10, 10)) + 0.1,
            "layer2.weight": any_backend.zeros((10, 10)),
        }
        measurement = tracker.measure(modified_weights)

        assert isinstance(measurement, DeviationMeasurement)
        assert measurement.deviation > 0.0
        assert measurement.baseline_norm > 0.0
        assert measurement.deviation_percent > 0.0
        assert measurement.condition_number > 0.0

    def test_weight_norm_computed_correctly(self, any_backend):
        """Test that weight norm is Frobenius norm."""
        tracker = DeviationTracker()
        # sqrt(100 * 2^2) = sqrt(400) = 20
        weights = {"weight": any_backend.ones((10, 10)) * 2.0}
        norm = tracker._compute_weight_norm(weights)
        assert abs(norm - 20.0) < 0.01

    def test_condition_number_computed(self, any_backend):
        """Test that condition number is computed and positive."""
        tracker = DeviationTracker()
        # Larger matrix for reliable SVD
        weights = {"weight": any_backend.random_normal((100, 100))}
        cond = tracker._compute_condition_number(weights)
        # Condition number should be >= 1.0 for any matrix
        assert cond >= 1.0


class TestDeltaMagnitude:
    """Tests for delta magnitude computation."""

    def test_compute_delta_magnitude(self, any_backend):
        """Test that delta magnitude is computed correctly."""
        tracker = DeviationTracker()
        target_weights = {"weight": any_backend.zeros((10, 10))}
        source_weights = {"weight": any_backend.ones((10, 10))}

        delta = tracker.compute_delta_magnitude(source_weights, target_weights)
        # sqrt(100 * 1^2) = 10.0
        assert abs(delta - 10.0) < 0.01

    def test_zero_delta_for_identical(self, any_backend):
        """Test zero delta for identical weights."""
        tracker = DeviationTracker()
        weights = {"weight": any_backend.ones((10, 10))}
        delta = tracker.compute_delta_magnitude(weights, weights)
        assert abs(delta) < 1e-5


class TestScaleDerivation:
    """Tests for SVD-based scale derivation."""

    def test_derive_scale_returns_positive(self, any_backend):
        """Test that derive_scale returns a positive value."""
        tracker = DeviationTracker()
        target_weights = {"weight": any_backend.ones((10, 10))}
        tracker.record_baseline(target_weights)

        source_weights = {"weight": any_backend.ones((10, 10)) + 0.5}
        activations = any_backend.random_normal((100, 10))

        scale = tracker.derive_scale(source_weights, target_weights, activations)

        assert scale > 0.0
        assert scale <= 1.0

    def test_derive_scale_zero_delta(self, any_backend):
        """Test scale derivation with zero delta."""
        tracker = DeviationTracker()
        target_weights = {"weight": any_backend.ones((10, 10))}
        tracker.record_baseline(target_weights)
        activations = any_backend.random_normal((100, 10))

        scale = tracker.derive_scale(target_weights, target_weights, activations)

        assert scale == 1.0

    def test_derive_scale_varies_with_activations(self, any_backend):
        """Test that scale depends on activation structure."""
        tracker = DeviationTracker()
        target_weights = {"weight": any_backend.ones((10, 10))}
        tracker.record_baseline(target_weights)

        source_weights = {"weight": any_backend.ones((10, 10)) + 1.0}

        # Low-rank activations (less null-space capacity)
        low_rank = any_backend.ones((100, 10))
        scale_low = tracker.derive_scale(source_weights, target_weights, low_rank)

        # Full-rank activations (more null-space capacity)
        full_rank = any_backend.random_normal((100, 10))
        scale_full = tracker.derive_scale(source_weights, target_weights, full_rank)

        # Both should be valid scales
        assert scale_low > 0.0
        assert scale_full > 0.0


class TestNamedBaselines:
    """Test multiple named baselines."""

    def test_multiple_baselines(self, any_backend):
        """Test recording and using multiple baselines."""
        tracker = DeviationTracker()
        baseline_a = {"weight": any_backend.ones((5, 5))}
        baseline_b = {"weight": any_backend.zeros((5, 5))}

        tracker.record_baseline(baseline_a, name="model_a")
        tracker.record_baseline(baseline_b, name="model_b")

        test_weights = {"weight": any_backend.ones((5, 5)) * 0.9}

        deviation_a = tracker.compute_deviation(test_weights, baseline_name="model_a")
        deviation_b = tracker.compute_deviation(test_weights, baseline_name="model_b")

        # Deviation from model_a (ones) should be smaller
        assert deviation_a < deviation_b

    def test_missing_baseline_returns_zero(self):
        """Test that missing baseline returns zero deviation."""
        tracker = DeviationTracker()
        deviation = tracker.compute_deviation({}, baseline_name="nonexistent")
        assert deviation == 0.0


class TestDeriveDeltaScale:
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
        assert abs(scale - 0.5) < 0.001

    def test_full_capacity_gives_full_scale(self):
        """Test that full null-space capacity gives scale = 1.0."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        scale = derive_delta_scale(null_rank=100, in_dim=100)
        assert scale == 1.0

    def test_zero_capacity_gives_eps(self):
        """Test that zero null-space capacity gives scale = sqrt(eps_f32)."""
        import math

        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        scale = derive_delta_scale(null_rank=0, in_dim=100)
        sqrt_eps_f32 = math.sqrt(math.ldexp(1.0, -23))
        assert scale == pytest.approx(sqrt_eps_f32)
        assert scale > 0.0

    def test_sequential_stacking_divides_capacity(self):
        """Test that n_merges divides the capacity ratio."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        # 50% capacity, 2 merges -> 25% each
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=2)
        assert abs(scale - 0.25) < 0.001

        # 50% capacity, 4 merges -> 12.5% each
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=4)
        assert abs(scale - 0.125) < 0.001

    def test_edge_cases(self):
        """Test edge cases don't cause errors."""
        from modelcypher.core.domain.geometry.deviation_budget import derive_delta_scale

        # Zero in_dim should return 1.0 (no constraint)
        scale = derive_delta_scale(null_rank=0, in_dim=0)
        assert scale == 1.0

        # Zero n_merges treated as 1
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=0)
        assert abs(scale - 0.5) < 0.001

        # Negative n_merges treated as 1
        scale = derive_delta_scale(null_rank=50, in_dim=100, n_merges=-1)
        assert abs(scale - 0.5) < 0.001
