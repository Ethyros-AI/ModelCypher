# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.geometric_early_stopping import (
    _SQRT_EPS,
    check_loss_stable,
)


class TestCheckLossStable:
    """Tests for check_loss_stable() — data-derived early stopping."""

    def test_constant_loss_is_stable(self):
        """20 identical losses → converged."""
        losses = [(i, 1.0, 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is True
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_diverging_loss_not_stable(self):
        """Recent mean >> earlier mean → not converged."""
        losses = [(i, 1.0 if i < 10 else 5.0, 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is False

    def test_auto_window_derivation(self):
        """10 losses → auto window = 5, splits into two windows of 5."""
        losses = [(i, 1.0, 100.0) for i in range(10)]
        is_stable, threshold = check_loss_stable(losses)

        # 10 // 2 = 5, both windows all 1.0 → stable
        assert is_stable is True

    def test_insufficient_history(self):
        """3 losses with window=5 → insufficient (need 2*5=10)."""
        losses = [(i, 1.0, 100.0) for i in range(3)]
        is_stable, threshold = check_loss_stable(losses, window=5)

        assert is_stable is False
        assert threshold == 0.0

    def test_window_less_than_two(self):
        """window=1 → early return (False, 0.0)."""
        losses = [(i, 1.0, 100.0) for i in range(10)]
        is_stable, threshold = check_loss_stable(losses, window=1)

        assert is_stable is False
        assert threshold == 0.0

    def test_numeric_floor_triggers(self):
        """Low-variance data → threshold clamped to _SQRT_EPS."""
        # All losses within 1e-8 of 1.0 → SE_diff ~ 0, clamped to _SQRT_EPS
        losses = [(i, 1.0 + 1e-10 * (i % 2), 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is True
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_high_variance_data_driven_threshold(self):
        """Noisy losses → threshold > _SQRT_EPS."""
        # Alternating pattern creates measurable variance
        losses = [(i, 1.0 + 0.1 * (-1) ** i, 100.0) for i in range(20)]
        _, threshold = check_loss_stable(losses)

        assert threshold > _SQRT_EPS
