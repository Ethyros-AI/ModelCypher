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

"""Tests for differentiable expansion loss.

Tests the differentiable proxy for geometric alignment training.
"""

from __future__ import annotations

import pytest
import mlx.core as mx

from modelcypher.adapters.geometry.mlx.differentiable_expansion import (
    ExpansionTrajectory,
    ExpansionLossTracker,
    soft_argmax,
    differentiable_expansion_loss,
    compute_expansion_metrics,
)


class TestSoftArgmax:
    """Tests for soft_argmax function."""

    def test_finds_clear_peak(self):
        """soft_argmax finds obvious peak."""
        values = mx.array([1.0, 2.0, 5.0, 2.0, 1.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)
        mx.eval(soft_idx, soft_val)

        # Should be close to index 2 (the peak)
        assert abs(float(soft_idx) - 2.0) < 0.5
        # Value should be close to 5.0
        assert abs(float(soft_val) - 5.0) < 1.0

    def test_uniform_values_returns_midpoint(self):
        """Uniform values give index near middle."""
        values = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mx.eval(values)

        soft_idx, _ = soft_argmax(values)
        mx.eval(soft_idx)

        # Should be near middle (index 2)
        assert abs(float(soft_idx) - 2.0) < 0.5

    def test_returns_mlx_arrays(self):
        """soft_argmax returns MLX arrays for gradient flow."""
        values = mx.array([1.0, 3.0, 2.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)

        assert isinstance(soft_idx, mx.array)
        assert isinstance(soft_val, mx.array)

    def test_single_value(self):
        """Single value returns index 0."""
        values = mx.array([5.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)
        mx.eval(soft_idx, soft_val)

        assert float(soft_idx) == 0.0
        assert float(soft_val) == 5.0


class TestDifferentiableExpansionLoss:
    """Tests for differentiable_expansion_loss function."""

    def test_balanced_trajectory_low_loss(self):
        """Trajectory with balanced expansion/compression has low loss.

        For expansion_ratio = 1.0, we need compression_rate = expansion_rate.
        """
        # With peak at layer 5 of 10:
        # expansion_rate = (peak - initial) / 5
        # compression_rate = (peak - final) / 4
        # For expansion_ratio = 1.0: compression_rate = expansion_rate
        # expansion_rate = (30-10)/5 = 4
        # compression_rate = 4
        # final = peak - compression_rate * 4 = 30 - 16 = 14

        initial = 10.0
        peak = 30.0
        expansion_rate = (peak - initial) / 5.0
        target_compression_rate = expansion_rate  # For expansion_ratio = 1.0
        final = peak - target_compression_rate * 4.0

        trajectory = []
        for i in range(11):
            if i <= 5:
                val = initial + (peak - initial) * (i / 5.0)
            else:
                val = peak - (peak - final) * ((i - 5) / 5.0)
            trajectory.append(val)

        traj = mx.array(trajectory)
        mx.eval(traj)

        loss, expansion_ratio = differentiable_expansion_loss(traj)
        mx.eval(loss, expansion_ratio)

        # Loss should be reasonably low (soft argmax introduces some smoothing)
        assert float(loss) < 0.5
        # Expansion ratio should be close to 1.0
        assert abs(float(expansion_ratio) - 1.0) < 0.5

    def test_high_compression_different_from_low(self):
        """High vs low compression trajectories give different losses."""
        # High compression: peak=100, final=10
        high_comp = mx.array([10.0, 50.0, 100.0, 50.0, 10.0])
        mx.eval(high_comp)

        # Low compression: peak=100, final=80
        low_comp = mx.array([10.0, 50.0, 100.0, 90.0, 80.0])
        mx.eval(low_comp)

        loss_high, ratio_high = differentiable_expansion_loss(high_comp)
        loss_low, ratio_low = differentiable_expansion_loss(low_comp)
        mx.eval(loss_high, ratio_high, loss_low, ratio_low)

        # Ratios should be different
        assert float(ratio_high) != float(ratio_low)

    def test_low_compression_high_loss(self):
        """Trajectory with very different expansion/compression has higher loss."""
        # Very asymmetric: big rise, tiny fall
        trajectory = mx.array([10.0, 30.0, 50.0, 48.0, 46.0])
        mx.eval(trajectory)

        loss, ratio = differentiable_expansion_loss(trajectory)
        mx.eval(loss, ratio)

        # Loss should be non-trivial for asymmetric trajectory
        assert float(loss) > 0.1

    def test_numerical_stability_flat_trajectory(self):
        """Flat trajectory doesn't cause numerical issues."""
        trajectory = mx.array([10.0, 10.0, 10.0, 10.0, 10.0])
        mx.eval(trajectory)

        loss, ratio = differentiable_expansion_loss(trajectory)
        mx.eval(loss, ratio)

        # Should not be NaN or inf
        assert not mx.isnan(loss).item()
        assert not mx.isinf(loss).item()
        assert not mx.isnan(ratio).item()
        assert not mx.isinf(ratio).item()

    def test_numerical_stability_peak_at_start(self):
        """Peak at start doesn't cause division by zero."""
        trajectory = mx.array([50.0, 40.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        loss, ratio = differentiable_expansion_loss(trajectory)
        mx.eval(loss, ratio)

        assert not mx.isnan(loss).item()
        assert not mx.isinf(loss).item()

    def test_numerical_stability_peak_at_end(self):
        """Peak at end doesn't cause division by zero."""
        trajectory = mx.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mx.eval(trajectory)

        loss, ratio = differentiable_expansion_loss(trajectory)
        mx.eval(loss, ratio)

        assert not mx.isnan(loss).item()
        assert not mx.isinf(loss).item()

    def test_returns_mlx_arrays(self):
        """Returns MLX arrays for gradient computation."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        loss, ratio = differentiable_expansion_loss(trajectory)

        assert isinstance(loss, mx.array)
        assert isinstance(ratio, mx.array)


class TestComputeExpansionMetrics:
    """Tests for compute_expansion_metrics function."""

    def test_returns_all_expected_keys(self):
        """compute_expansion_metrics returns all expected keys."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        metrics = compute_expansion_metrics(trajectory)

        expected_keys = [
            "expansion_ratio",
            "peak_layer",
            "peak_norm",
            "expansion_rate",
            "compression_rate",
            "initial_norm",
            "final_norm",
            "n_layers",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_returns_python_floats(self):
        """compute_expansion_metrics returns Python floats for logging."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        metrics = compute_expansion_metrics(trajectory)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"

    def test_boundary_values_correct(self):
        """Initial and final norms match trajectory boundaries."""
        trajectory = mx.array([5.0, 15.0, 25.0, 20.0, 8.0])
        mx.eval(trajectory)

        metrics = compute_expansion_metrics(trajectory)

        assert metrics["initial_norm"] == 5.0
        assert metrics["final_norm"] == 8.0
        assert metrics["n_layers"] == 4  # 5 values = 4 layers + embedding


class TestExpansionLossTracker:
    """Tests for ExpansionLossTracker class."""

    def test_record_and_summary(self):
        """Tracker records metrics and provides summary."""
        tracker = ExpansionLossTracker()

        tracker.record({"expansion_ratio": 1.0}, epoch=0, step=0)
        tracker.record({"expansion_ratio": 1.2}, epoch=0, step=1)
        tracker.record({"expansion_ratio": 0.8}, epoch=1, step=0)

        summary = tracker.get_summary()

        assert "expansion_ratio_mean" in summary
        assert abs(summary["expansion_ratio_mean"] - 1.0) < 0.001
        assert summary["n_samples"] == 3

    def test_empty_summary(self):
        """Empty tracker returns empty summary."""
        tracker = ExpansionLossTracker()
        summary = tracker.get_summary()
        assert summary == {}


class TestExpansionTrajectory:
    """Tests for ExpansionTrajectory dataclass."""

    def test_stores_all_fields(self):
        """ExpansionTrajectory stores all required fields."""
        norms = mx.array([1.0, 2.0, 3.0])
        peak_val = mx.array(3.0)
        peak_idx = mx.array(2.0)
        initial = mx.array(1.0)
        final = mx.array(3.0)

        traj = ExpansionTrajectory(
            norms=norms,
            soft_peak_val=peak_val,
            soft_peak_idx=peak_idx,
            initial_norm=initial,
            final_norm=final,
        )

        assert traj.norms is norms
        assert traj.soft_peak_val is peak_val
        assert traj.soft_peak_idx is peak_idx
        assert traj.initial_norm is initial
        assert traj.final_norm is final
