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

"""Tests for spectral budget monitoring.

Backend-agnostic. Uses synthetic data, no model loading.

1. Known spectral norm (diagonal matrix) -> correct ratio
2. Below threshold -> not exhausted
3. Above threshold -> exhausted
4. Empty list -> not exhausted
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.spectral_budget import (
    compute_budget_ratios,
    is_budget_exhausted,
)


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


class TestComputeBudgetRatios:
    """Tests for compute_budget_ratios."""

    def test_diagonal_known_spectral_norm(self, backend):
        """Diagonal LoRA product -> spectral norm = largest diagonal entry."""
        b = backend
        import numpy as np

        # lora_a @ lora_b will be a diagonal matrix with known spectral norm
        # A = I[4x2], B = diag([3, 1]) extended to [2x4]
        # Product = A @ B: [4x4] with singular values [3, 1, 0, 0]
        A = b.array(np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32))
        B = b.array(np.array([[3, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
        b.eval(A, B)

        scale = 1.0
        sigma_k = 5.0

        ratios = compute_budget_ratios([(scale, A, B, sigma_k)], b)

        assert len(ratios) == 1
        # spectral_norm = 3.0, ratio = 3.0 / 5.0 = 0.6
        assert abs(ratios[0] - 0.6) < 0.01, f"Expected ratio ~0.6, got {ratios[0]}"

    def test_scale_factor_applied(self, backend):
        """Scale factor should multiply the product before SVD."""
        b = backend
        import numpy as np

        A = b.array(np.eye(2, dtype=np.float32))
        B = b.array(np.eye(2, dtype=np.float32))
        b.eval(A, B)

        sigma_k = 1.0

        # scale=0.5: product = 0.5*I, spectral_norm = 0.5, ratio = 0.5
        ratios = compute_budget_ratios([(0.5, A, B, sigma_k)], b)
        assert len(ratios) == 1
        assert abs(ratios[0] - 0.5) < 0.01

    def test_zero_sigma_k_skipped(self, backend):
        """Entries with sigma_k <= 0 should be skipped."""
        b = backend
        import numpy as np

        A = b.array(np.eye(2, dtype=np.float32))
        B = b.array(np.eye(2, dtype=np.float32))
        b.eval(A, B)

        ratios = compute_budget_ratios([(1.0, A, B, 0.0)], b)
        assert len(ratios) == 0

    def test_multiple_layers(self, backend):
        """Multiple LoRA products should each get a ratio."""
        b = backend
        import numpy as np

        A1 = b.array(np.eye(2, dtype=np.float32))
        B1 = b.array(np.eye(2, dtype=np.float32))
        A2 = b.array(2.0 * np.eye(2, dtype=np.float32))
        B2 = b.array(np.eye(2, dtype=np.float32))
        b.eval(A1, B1, A2, B2)

        ratios = compute_budget_ratios([
            (1.0, A1, B1, 2.0),  # ratio = 1.0/2.0 = 0.5
            (1.0, A2, B2, 4.0),  # ratio = 2.0/4.0 = 0.5
        ], b)

        assert len(ratios) == 2
        for r in ratios:
            assert abs(r - 0.5) < 0.01

    def test_empty_input(self, backend):
        """Empty input should return empty list."""
        ratios = compute_budget_ratios([], backend)
        assert ratios == []


class TestIsBudgetExhausted:
    """Tests for is_budget_exhausted (pure Python)."""

    def test_empty_ratios_not_exhausted(self):
        """Empty ratios -> not exhausted."""
        exhausted, median = is_budget_exhausted([])
        assert not exhausted
        assert median == 0.0

    def test_below_threshold_not_exhausted(self):
        """All ratios below threshold -> not exhausted."""
        exhausted, median = is_budget_exhausted([0.3, 0.5, 0.7])
        assert not exhausted
        assert abs(median - 0.5) < 0.01

    def test_above_threshold_exhausted(self):
        """Median above threshold -> exhausted."""
        exhausted, median = is_budget_exhausted([0.8, 0.95, 0.99])
        assert exhausted
        assert median > 0.9

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        # With default 0.9: not exhausted
        exhausted_default, _ = is_budget_exhausted([0.5, 0.6, 0.7])
        assert not exhausted_default

        # With threshold 0.5: exhausted (median = 0.6)
        exhausted_custom, median = is_budget_exhausted([0.5, 0.6, 0.7], threshold=0.5)
        assert exhausted_custom
        assert abs(median - 0.6) < 0.01

    def test_single_ratio(self):
        """Single ratio should work (median = that ratio)."""
        exhausted, median = is_budget_exhausted([0.95])
        assert exhausted
        assert abs(median - 0.95) < 0.01

    def test_even_number_of_ratios(self):
        """Even count uses floor-median (len//2 index)."""
        # [0.3, 0.5, 0.7, 0.8] -> sorted index 2 = 0.7
        exhausted, median = is_budget_exhausted([0.8, 0.3, 0.7, 0.5])
        assert not exhausted  # 0.7 < 0.9
        assert abs(median - 0.7) < 0.01
