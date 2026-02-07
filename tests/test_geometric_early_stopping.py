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

"""Tests for data-derived early stopping.

Pure Python — no backend needed. Validates that the stopping criterion
adapts to data noise:

1. Decreasing losses -> not stable (still improving)
2. Flat losses -> stable (SE threshold dominates)
3. Noisy-but-flat losses -> stable (SE captures noise)
4. Insufficient data -> not stable (< 2*window)
5. Homogeneous vs diverse data -> threshold adapts
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.training.geometric_early_stopping import (
    _SQRT_EPS,
    check_loss_stable,
)


class TestCheckLossStable:
    """Tests for check_loss_stable convergence detection."""

    def test_insufficient_data_not_stable(self):
        """Less than 2*window entries -> not stable."""
        losses = [(i, 1.0, 100.0) for i in range(5)]
        stable, threshold = check_loss_stable(losses, window=10)
        assert not stable
        assert threshold == 0.0

    def test_exactly_2x_window_boundary(self):
        """Exactly 2*window entries should be evaluated."""
        window = 5
        losses = [(i, 1.0, 100.0) for i in range(2 * window)]
        stable, threshold = check_loss_stable(losses, window=window)
        # All identical -> delta = 0, which is < threshold
        assert stable
        assert threshold >= _SQRT_EPS

    def test_decreasing_losses_not_stable(self):
        """Steadily decreasing losses -> still improving -> not stable."""
        window = 10
        # Linear decrease: earlier window has higher mean than recent window
        losses = [(i, 10.0 - i * 0.1, 100.0) for i in range(40)]
        stable, threshold = check_loss_stable(losses, window=window)
        assert not stable

    def test_flat_losses_stable(self):
        """Perfectly flat losses -> stable."""
        window = 10
        losses = [(i, 2.5, 100.0) for i in range(30)]
        stable, threshold = check_loss_stable(losses, window=window)
        assert stable
        # With zero variance, threshold falls back to _SQRT_EPS
        assert threshold == _SQRT_EPS

    def test_noisy_but_flat_stable(self):
        """Noisy losses with same mean across windows -> stable.

        SE captures the noise. We construct both windows from the same
        distribution with matched means by interleaving, so the delta
        is guaranteed small relative to within-window variance.
        """
        window = 20
        # Deterministic noise pattern: alternating +0.05, -0.05
        # Both windows get the same pattern -> identical means,
        # but nonzero variance -> SE > _SQRT_EPS.
        noise = [0.05 if i % 2 == 0 else -0.05 for i in range(2 * window)]
        losses = [(i, 2.0 + noise[i], 100.0) for i in range(2 * window)]
        stable, threshold = check_loss_stable(losses, window=window)
        assert stable
        assert threshold > _SQRT_EPS  # SE should be above machine epsilon

    def test_homogeneous_tight_threshold(self):
        """Homogeneous data (low variance) -> tight threshold."""
        window = 10
        # Very low variance: all losses are 1.0 +/- 1e-6
        losses = [(i, 1.0 + (i % 2) * 1e-6, 100.0) for i in range(30)]
        _, threshold_tight = check_loss_stable(losses, window=window)

        # High variance: losses oscillate between 1.0 and 1.5
        losses_diverse = [(i, 1.0 + (i % 2) * 0.5, 100.0) for i in range(30)]
        _, threshold_wide = check_loss_stable(losses_diverse, window=window)

        assert threshold_wide > threshold_tight

    def test_threshold_is_se_not_fixed(self):
        """Threshold should scale with data variance, not be a fixed constant."""
        window = 10
        # Low noise
        losses_low = [(i, 1.0 + (i % 3) * 0.001, 100.0) for i in range(30)]
        _, thresh_low = check_loss_stable(losses_low, window=window)

        # High noise
        losses_high = [(i, 1.0 + (i % 3) * 0.1, 100.0) for i in range(30)]
        _, thresh_high = check_loss_stable(losses_high, window=window)

        assert thresh_high > thresh_low

    def test_sqrt_eps_constant(self):
        """_SQRT_EPS should be sqrt of float32 machine epsilon."""
        expected = math.sqrt(math.ldexp(1.0, -23))
        assert abs(_SQRT_EPS - expected) < 1e-15

    def test_default_window(self):
        """Default window=10 should work."""
        losses = [(i, 1.0, 100.0) for i in range(25)]
        stable, threshold = check_loss_stable(losses)
        assert stable  # All flat at 1.0

    def test_large_step_improvement_not_stable(self):
        """A sudden improvement in the recent window -> not stable."""
        window = 5
        # Earlier window: loss ~ 5.0
        earlier = [(i, 5.0, 100.0) for i in range(window)]
        # Recent window: loss ~ 2.0
        recent = [(i + window, 2.0, 100.0) for i in range(window)]
        losses = earlier + recent
        stable, _ = check_loss_stable(losses, window=window)
        assert not stable
