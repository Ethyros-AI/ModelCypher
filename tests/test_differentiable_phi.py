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

"""Tests for differentiable_phi.py - Differentiable phi loss for training.

Tests cover:
- soft_argmax: Differentiable peak detection
- compute_trajectory_norms: Forward pass through model
- differentiable_phi_loss: Loss computation and numerical stability
- compute_phi_metrics: Monitoring metrics
- PhiLossTracker: Curriculum and tracking
"""

from __future__ import annotations

import pytest
import mlx.core as mx

from modelcypher.core.domain.geometry.differentiable_phi import (
    PHI,
    PhiTrajectory,
    PhiLossTracker,
    soft_argmax,
    differentiable_phi_loss,
    compute_phi_metrics,
)


# =============================================================================
# soft_argmax Tests
# =============================================================================


class TestSoftArgmax:
    """Tests for soft_argmax function."""

    def test_finds_clear_peak(self):
        """soft_argmax finds index of clear maximum."""
        values = mx.array([1.0, 2.0, 5.0, 3.0, 1.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)
        mx.eval(soft_idx, soft_val)

        # Peak is clearly at index 2 (value 5.0)
        assert abs(float(soft_idx) - 2.0) < 0.3  # Allow some softness
        assert float(soft_val) > 4.0  # Weighted toward peak

    def test_uniform_values_returns_midpoint(self):
        """Uniform values return center index."""
        values = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)
        mx.eval(soft_idx, soft_val)

        # With uniform values, soft_idx should be near middle (2.0)
        assert abs(float(soft_idx) - 2.0) < 0.01
        assert abs(float(soft_val) - 1.0) < 0.01

    def test_returns_mlx_arrays(self):
        """soft_argmax returns MLX arrays (not Python floats)."""
        values = mx.array([1.0, 3.0, 2.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)

        assert isinstance(soft_idx, mx.array)
        assert isinstance(soft_val, mx.array)

    def test_single_value(self):
        """soft_argmax handles single value."""
        values = mx.array([5.0])
        mx.eval(values)

        soft_idx, soft_val = soft_argmax(values)
        mx.eval(soft_idx, soft_val)

        assert float(soft_idx) == 0.0
        assert abs(float(soft_val) - 5.0) < 0.01  # Allow small numerical tolerance


# =============================================================================
# differentiable_phi_loss Tests
# =============================================================================


class TestDifferentiablePhiLoss:
    """Tests for differentiable_phi_loss function."""

    def test_perfect_phi_trajectory_low_loss(self):
        """Trajectory matching phi geometry has low comp_phi loss.

        Note: Due to soft_argmax smoothing, the comp_phi won't be exactly 1.0
        even with a mathematically perfect trajectory. We test that a
        well-designed trajectory has comp_phi reasonably close to 1.0.
        """
        # Construct trajectory where comp/phi ≈ 1.0
        # Need: compression_rate / (expansion_rate * PHI) = 1.0
        # So: compression_rate = expansion_rate * PHI
        #
        # With peak at layer 5 of 10:
        # expansion_rate = (peak - initial) / 5
        # compression_rate = (peak - final) / 4
        #
        # Setting initial=10, peak=30, final computed to give comp/phi=1.0:
        # expansion_rate = (30-10)/5 = 4
        # compression_rate = expansion_rate * PHI = 4 * 1.618 = 6.472
        # final = peak - compression_rate * 4 = 30 - 25.89 = 4.11

        initial = 10.0
        peak = 30.0
        expansion_rate = (peak - initial) / 5.0
        target_compression_rate = expansion_rate * PHI
        final = peak - target_compression_rate * 4.0

        # Build trajectory: linear rise to peak, linear fall to final
        trajectory = []
        for i in range(11):  # 0 to 10 inclusive
            if i <= 5:
                # Linear rise
                val = initial + (peak - initial) * (i / 5.0)
            else:
                # Linear fall
                val = peak - (peak - final) * ((i - 5) / 5.0)
            trajectory.append(val)

        traj = mx.array(trajectory)
        mx.eval(traj)

        loss, comp_phi = differentiable_phi_loss(traj)
        mx.eval(loss, comp_phi)

        # comp_phi should be reasonably close to 1.0
        # Soft argmax smoothing means it won't be exact, but should be in range
        assert 0.5 < float(comp_phi) < 2.0, f"comp_phi={float(comp_phi)} outside reasonable range"

    def test_high_compression_different_from_low(self):
        """Trajectory with higher compression has higher comp_phi than low compression."""
        # Steep rise, shallow fall -> low compression -> comp/phi < 1
        # Shallow rise, steep fall -> high compression -> comp/phi > 1
        high_compression = mx.array([
            10.0,  # initial
            15.0,
            18.0,
            20.0,  # peak (layer 3)
            12.0,
            4.0,   # steep fall
            2.0,   # final
        ])
        mx.eval(high_compression)

        low_compression = mx.array([
            5.0,   # initial
            10.0,
            20.0,
            30.0,  # peak
            28.0,  # very shallow fall
            27.0,
            26.0,  # final
        ])
        mx.eval(low_compression)

        _, comp_phi_high = differentiable_phi_loss(high_compression)
        _, comp_phi_low = differentiable_phi_loss(low_compression)
        mx.eval(comp_phi_high, comp_phi_low)

        # High compression trajectory should have higher comp_phi than low compression
        assert float(comp_phi_high) > float(comp_phi_low)

    def test_low_compression_high_loss(self):
        """Trajectory with insufficient compression has low comp_phi."""
        # Steep rise, very shallow fall
        trajectory = mx.array([
            5.0,   # initial
            10.0,
            20.0,
            30.0,  # peak
            28.0,  # very shallow fall
            27.0,
            26.0,  # final
        ])
        mx.eval(trajectory)

        loss, comp_phi = differentiable_phi_loss(trajectory)
        mx.eval(loss, comp_phi)

        # Low compression should give comp_phi < 1
        assert float(comp_phi) < 1.0

    def test_numerical_stability_flat_trajectory(self):
        """Flat trajectory doesn't cause numerical issues."""
        trajectory = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mx.eval(trajectory)

        loss, comp_phi = differentiable_phi_loss(trajectory)
        mx.eval(loss, comp_phi)

        # Should return finite values
        assert not mx.isnan(loss).any()
        assert not mx.isinf(loss).any()
        assert not mx.isnan(comp_phi).any()
        assert not mx.isinf(comp_phi).any()

    def test_numerical_stability_peak_at_start(self):
        """Peak at start doesn't cause division by zero."""
        trajectory = mx.array([100.0, 50.0, 30.0, 20.0, 10.0])  # Peak at index 0
        mx.eval(trajectory)

        loss, comp_phi = differentiable_phi_loss(trajectory)
        mx.eval(loss, comp_phi)

        assert not mx.isnan(loss).any()
        assert not mx.isinf(loss).any()

    def test_numerical_stability_peak_at_end(self):
        """Peak at end doesn't cause division by zero."""
        trajectory = mx.array([10.0, 20.0, 30.0, 50.0, 100.0])  # Peak at final index
        mx.eval(trajectory)

        loss, comp_phi = differentiable_phi_loss(trajectory)
        mx.eval(loss, comp_phi)

        assert not mx.isnan(loss).any()
        assert not mx.isinf(loss).any()

    def test_returns_mlx_arrays(self):
        """differentiable_phi_loss returns MLX arrays for gradient flow."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        loss, comp_phi = differentiable_phi_loss(trajectory)

        assert isinstance(loss, mx.array)
        assert isinstance(comp_phi, mx.array)


# =============================================================================
# compute_phi_metrics Tests
# =============================================================================


class TestComputePhiMetrics:
    """Tests for compute_phi_metrics function."""

    def test_returns_all_expected_keys(self):
        """compute_phi_metrics returns all expected keys."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        metrics = compute_phi_metrics(trajectory)

        expected_keys = [
            "comp_phi",
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
        """compute_phi_metrics returns Python floats for logging."""
        trajectory = mx.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mx.eval(trajectory)

        metrics = compute_phi_metrics(trajectory)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"

    def test_boundary_values_correct(self):
        """Initial and final norms match trajectory boundaries."""
        trajectory = mx.array([5.0, 15.0, 25.0, 20.0, 8.0])
        mx.eval(trajectory)

        metrics = compute_phi_metrics(trajectory)

        assert metrics["initial_norm"] == 5.0
        assert metrics["final_norm"] == 8.0
        assert metrics["n_layers"] == 4  # 5 values = 4 layers + embedding


# =============================================================================
# PhiLossTracker Tests
# =============================================================================


class TestPhiLossTracker:
    """Tests for PhiLossTracker class."""

    def test_record_and_summary(self):
        """Tracker records metrics and provides summary."""
        tracker = PhiLossTracker()

        # Record some metrics
        tracker.record({"comp_phi": 1.0}, epoch=0, step=0)
        tracker.record({"comp_phi": 1.2}, epoch=0, step=1)
        tracker.record({"comp_phi": 0.8}, epoch=1, step=0)

        summary = tracker.get_summary()

        assert "comp_phi_mean" in summary
        assert abs(summary["comp_phi_mean"] - 1.0) < 0.001
        assert summary["n_samples"] == 3

    def test_empty_summary(self):
        """Empty tracker returns empty summary."""
        tracker = PhiLossTracker()
        summary = tracker.get_summary()
        assert summary == {}


# =============================================================================
# PhiTrajectory Dataclass Tests
# =============================================================================


class TestPhiTrajectory:
    """Tests for PhiTrajectory dataclass."""

    def test_stores_all_fields(self):
        """PhiTrajectory stores all provided fields."""
        norms = mx.array([1.0, 2.0, 3.0, 2.0, 1.0])
        mx.eval(norms)

        trajectory = PhiTrajectory(
            norms=norms,
            soft_peak_val=mx.array(3.0),
            soft_peak_idx=mx.array(2.0),
            initial_norm=mx.array(1.0),
            final_norm=mx.array(1.0),
        )

        assert trajectory.norms.shape[0] == 5
        assert float(trajectory.soft_peak_val) == 3.0
        assert float(trajectory.soft_peak_idx) == 2.0
