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

"""Tests for core.domain.geometry.jacobian_bounds."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.jacobian_bounds import (
    submultiplicative_jacobian_bound,
)


class TestSubmultiplicativeJacobianBound:
    def test_all_zero_norms(self):
        """All norms zero → bound is 1.0 (identity residual)."""
        norms = {"o": 0.0, "v": 0.0, "down": 0.0, "gate": 0.0, "up": 0.0}
        assert submultiplicative_jacobian_bound(norms) == pytest.approx(1.0)

    def test_known_values(self):
        """Verify formula: 1 + σ(O)·σ(V) + σ(down)·(σ(gate) + σ(up))."""
        norms = {"o": 2.0, "v": 3.0, "down": 4.0, "gate": 1.0, "up": 0.5}
        expected = 1.0 + 2.0 * 3.0 + 4.0 * (1.0 + 0.5)  # 1 + 6 + 6 = 13
        assert submultiplicative_jacobian_bound(norms) == pytest.approx(expected)

    def test_missing_key(self):
        """Missing required key → KeyError."""
        norms = {"o": 1.0, "v": 1.0}  # missing down, gate, up
        with pytest.raises(KeyError):
            submultiplicative_jacobian_bound(norms)
