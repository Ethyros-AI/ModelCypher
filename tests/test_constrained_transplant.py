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

"""Tests for constrained_transplant.py."""

from __future__ import annotations

import math
import pytest
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.constrained_transplant import verify_boundary_invariance
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

class TestConstrainedTransplant:
    """Tests for boundary invariance verification."""

    def test_verify_boundary_invariance_exact(self):
        """Standard verification should pass if weights are identical."""
        backend = get_default_backend()
        
        # Random weights
        w_target = backend.eye(4)
        w_trans = w_target # Identical
        
        # Boundary activations
        boundary = backend.eye(4)
        
        res = verify_boundary_invariance(w_trans, w_target, boundary, backend=backend)
        assert res["passed"] is True
        eps = division_epsilon(backend, w_target)
        assert res["max_relative_diff"] < eps

    def test_verify_boundary_invariance_fail(self):
        """Should fail if weights differ significantly on boundary subspace."""
        backend = get_default_backend()
        
        w_target = backend.zeros((2, 2))
        w_trans = backend.ones((2, 2)) # Huge difference
        
        boundary = backend.eye(2) # Identity activates everything
        
        res = verify_boundary_invariance(w_trans, w_target, boundary, backend=backend)
        
        assert res["passed"] is False
        assert math.isinf(res["max_relative_diff"])
        assert math.isinf(res["mean_relative_diff"])

    def test_verify_empty_boundary(self):
        """Should succeed trivially on empty boundary."""
        backend = get_default_backend()
        w = backend.zeros((2,2))
        boundary = backend.zeros((0,2))
        
        res = verify_boundary_invariance(w, w, boundary, backend=backend)
        assert res["passed"] is True
        assert res["boundary_samples"] == 0
