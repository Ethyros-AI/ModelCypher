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

"""Tests for loop preservation loss (loop_preservation.py).

Validates the implementation of spectral entropy trajectory tracking
and loop preservation loss computation.
"""

from __future__ import annotations

import pytest
import numpy as np

from modelcypher.core.domain.training.loop_preservation import (
    LoopPreservationConfig,
    loop_preservation_loss,
    _compute_spectral_entropy,
)


class TestLoopPreservationConfig:
    """Tests for LoopPreservationConfig dataclass."""

    def test_config_creation(self):
        """Config can be created with all required fields."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,
            lambda_scale=0.01,
        )

        assert config.highway_layer == 4
        assert config.base_delta_entropy == 0.5
        assert config.lambda_scale == 0.01

    def test_config_to_dict(self):
        """Config can be serialized to dict."""
        config = LoopPreservationConfig(
            highway_layer=8,
            base_delta_entropy=-0.2,
            lambda_scale=0.005,
        )

        d = config.to_dict()
        assert d["highway_layer"] == 8
        assert d["base_delta_entropy"] == -0.2
        assert d["lambda_scale"] == 0.005


class TestLoopPreservationLoss:
    """Tests for loop_preservation_loss function."""

    def test_loss_zero_when_current_better_than_base(self):
        """Loss is 0 when current entropy trajectory is better than base."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,  # Base: H_exit - H_highway = 0.5
            lambda_scale=1.0,
        )

        # Current: entropy grows MORE than base (0.7 > 0.5)
        current_trajectory = {
            4: 1.0,   # H_highway = 1.0
            12: 1.7,  # H_exit = 1.7, so ΔH = 0.7
        }

        loss, delta = loop_preservation_loss(current_trajectory, config)

        assert loss == 0.0, "Loss should be 0 when current ΔH > base ΔH"
        assert delta == pytest.approx(0.7, abs=1e-6)

    def test_loss_positive_when_entropy_collapses(self):
        """Loss is positive when entropy collapses (worse than base)."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,  # Base has healthy growth
            lambda_scale=1.0,
        )

        # Current: entropy DECREASES (collapse)
        current_trajectory = {
            4: 2.0,   # H_highway = 2.0
            12: 1.8,  # H_exit = 1.8, so ΔH = -0.2 (collapse!)
        }

        loss, delta = loop_preservation_loss(current_trajectory, config)

        # Expected: λ * max(0, 0.5 - (-0.2)) = 1.0 * 0.7 = 0.7
        assert loss == pytest.approx(0.7, abs=1e-6)
        assert delta == pytest.approx(-0.2, abs=1e-6)

    def test_loss_scales_with_lambda(self):
        """Loss scales linearly with lambda_scale."""
        config1 = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,
            lambda_scale=1.0,
        )

        config2 = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,
            lambda_scale=0.1,
        )

        current_trajectory = {
            4: 2.0,
            12: 1.8,  # ΔH = -0.2
        }

        loss1, _ = loop_preservation_loss(current_trajectory, config1)
        loss2, _ = loop_preservation_loss(current_trajectory, config2)

        assert loss1 == pytest.approx(loss2 * 10, abs=1e-6)

    def test_empty_trajectory_returns_zero(self):
        """Empty trajectory returns zero loss."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.5,
            lambda_scale=1.0,
        )

        loss, delta = loop_preservation_loss({}, config)

        assert loss == 0.0
        assert delta == 0.0

    def test_uses_closest_layer_when_exact_highway_missing(self):
        """Uses closest available layer when exact highway layer not in trajectory."""
        config = LoopPreservationConfig(
            highway_layer=5,  # Highway at layer 5
            base_delta_entropy=0.5,
            lambda_scale=1.0,
        )

        # Trajectory has layer 4 and 6, but not 5
        current_trajectory = {
            4: 1.0,   # Closest to highway
            6: 1.2,
            12: 1.6,  # Exit
        }

        # Should use layer 4 (closest to 5) as highway
        # ΔH = 1.6 - 1.0 = 0.6 > 0.5, so loss = 0
        loss, delta = loop_preservation_loss(current_trajectory, config)

        assert loss == 0.0
        assert delta == pytest.approx(0.6, abs=1e-6)


class TestSpectralEntropyComputation:
    """Tests for _compute_spectral_entropy helper."""

    def test_uniform_singular_values_high_entropy(self):
        """Uniform singular values produce high entropy."""
        import mlx.core as mx

        # Create data with near-uniform singular values using an orthogonal matrix
        # A random orthogonal matrix has all singular values = 1 (uniform)
        n = 10
        q, _ = np.linalg.qr(np.random.randn(n, 8))
        hidden = mx.array(q.astype(np.float32))

        mx.eval(hidden)
        entropy = _compute_spectral_entropy(hidden)

        # Entropy should be positive and reasonable for uniform SVs
        assert entropy > 0.0
        assert entropy < 10.0  # Reasonable upper bound

    def test_single_dominant_direction_low_entropy(self):
        """Single dominant direction produces low entropy."""
        import mlx.core as mx

        # Create rank-1 data
        u = mx.random.normal((10, 1))
        v = mx.random.normal((1, 8))
        hidden = u @ v

        mx.eval(hidden)
        entropy = _compute_spectral_entropy(hidden)

        # Entropy should be very low for rank-1 data
        assert entropy < 1.0

    def test_handles_3d_input(self):
        """Correctly handles 3D input [batch, seq, hidden]."""
        import mlx.core as mx

        hidden = mx.random.normal((2, 5, 8))
        mx.eval(hidden)

        entropy = _compute_spectral_entropy(hidden)

        assert entropy >= 0.0
        assert entropy < 10.0

    def test_empty_input_returns_zero(self):
        """Empty input returns zero entropy."""
        import mlx.core as mx

        hidden = mx.zeros((0, 8))
        mx.eval(hidden)

        entropy = _compute_spectral_entropy(hidden)

        assert entropy == 0.0

    def test_small_input_returns_zero(self):
        """Very small input returns zero entropy."""
        import mlx.core as mx

        hidden = mx.zeros((1, 8))
        mx.eval(hidden)

        entropy = _compute_spectral_entropy(hidden)

        assert entropy == 0.0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_healthy_reasoning_trajectory(self):
        """Healthy reasoning has growing entropy (no loss)."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.3,
            lambda_scale=0.01,
        )

        # Healthy trajectory: entropy grows through layers
        trajectory = {
            0: 0.5,
            4: 1.0,   # Highway
            8: 1.5,
            12: 2.0,  # Exit, ΔH = 1.0 > 0.3
        }

        loss, delta = loop_preservation_loss(trajectory, config)

        assert loss == 0.0, "Healthy trajectory should have no loss"
        assert delta > config.base_delta_entropy

    def test_collapsed_trajectory_penalized(self):
        """Collapsed trajectory (entropy drops) is penalized."""
        config = LoopPreservationConfig(
            highway_layer=4,
            base_delta_entropy=0.3,
            lambda_scale=0.01,
        )

        # Collapsed: entropy drops after highway
        trajectory = {
            0: 0.5,
            4: 2.0,   # Highway peak
            8: 1.5,
            12: 1.0,  # Exit, ΔH = -1.0 (collapse!)
        }

        loss, delta = loop_preservation_loss(trajectory, config)

        # Loss = 0.01 * max(0, 0.3 - (-1.0)) = 0.01 * 1.3 = 0.013
        assert loss > 0, "Collapsed trajectory should be penalized"
        assert loss == pytest.approx(0.013, abs=1e-6)
        assert delta < 0, "Delta should be negative for collapse"
