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

"""Tests for principal subspace angle computation.

Verifies Björck-Golub (1973) principal angle measurement on synthetic
subspaces with known geometry.
"""
from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.spectral_analysis import (
    principal_subspace_angle,
)


def _get_backend():
    """Get default backend for tests."""
    from modelcypher.core.domain._backend import get_default_backend

    return get_default_backend()


class TestPrincipalSubspaceAngle:
    """Tests for principal_subspace_angle."""

    def test_identical_subspaces_angle_zero(self):
        """Same subspace → angle = 0."""
        b = _get_backend()
        # Two identical orthonormal row vectors
        V = b.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        angle = principal_subspace_angle(V, V, backend=b)
        assert abs(angle) < 1e-6

    def test_orthogonal_subspaces_angle_pi_over_2(self):
        """Orthogonal 1-d subspaces → angle = π/2."""
        b = _get_backend()
        V_base = b.array([[1.0, 0.0]])
        V_adapted = b.array([[0.0, 1.0]])
        angle = principal_subspace_angle(V_base, V_adapted, backend=b)
        assert abs(angle - math.pi / 2) < 1e-6

    def test_known_rotation_angle(self):
        """45-degree rotation of a 1-d subspace → angle = π/4."""
        b = _get_backend()
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        V_base = b.array([[1.0, 0.0]])
        V_adapted = b.array([[c, s]])
        angle = principal_subspace_angle(V_base, V_adapted, backend=b)
        assert abs(angle - math.pi / 4) < 1e-5

    def test_2d_subspace_partial_overlap(self):
        """2-d subspace with one shared and one rotated direction."""
        b = _get_backend()
        # Base: span{e1, e2}
        V_base = b.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Adapted: span{e1, e3} — shares e1, differs on e2 vs e3
        V_adapted = b.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        angle = principal_subspace_angle(V_base, V_adapted, backend=b)
        # Largest principal angle = π/2 (between e2 and e3)
        assert abs(angle - math.pi / 2) < 1e-5

    def test_2d_subspace_identical(self):
        """Two identical 2-d subspaces → angle = 0."""
        b = _get_backend()
        V = b.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        angle = principal_subspace_angle(V, V, backend=b)
        assert abs(angle) < 1e-6

    def test_small_perturbation_small_angle(self):
        """A small perturbation produces a small but nonzero angle.

        Resolution floor: sqrt(2 * eps_f32) ≈ 4.9e-4 rad.  Use 1e-3
        to stay above it while still testing the small-angle regime.
        """
        b = _get_backend()
        theta = 1e-3
        V_base = b.array([[1.0, 0.0]])
        V_adapted = b.array([[math.cos(theta), math.sin(theta)]])
        angle = principal_subspace_angle(V_base, V_adapted, backend=b)
        assert angle < 0.01
        assert angle > 0.0
