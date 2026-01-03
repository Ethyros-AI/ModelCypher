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

"""Neighborhood stability tests (requires MLX)."""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_fidelity_sweep import (
    ManifoldFidelitySweep,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _build_sweep(x):
    """Build sweep for testing - all parameters derived from data."""
    backend = get_default_backend()
    eps = division_epsilon(backend, x)
    return ManifoldFidelitySweep(backend=backend), eps


def test_neighborhood_stability_identical():
    """Identical point sets should have 1.0 neighborhood overlap."""
    mx.random.seed(0)
    x = mx.random.normal((50, 128))
    sweep, eps = _build_sweep(x)

    # Using private method for testing the neighborhood overlap logic
    overlap = sweep._compute_knn_overlap(x, x, k=5)

    assert abs(float(overlap) - 1.0) <= eps


def test_neighborhood_stability_orthogonal():
    """Orthogonal subspaces should have low neighborhood overlap."""
    mx.random.seed(1)
    n = 20
    d = 10
    # Create two orthogonal bases - QR requires CPU stream on MLX
    q1, _ = mx.linalg.qr(mx.random.normal((d, d)), stream=mx.cpu)
    q2, _ = mx.linalg.qr(mx.random.normal((d, d)), stream=mx.cpu)

    # Points in different subspaces
    x = mx.random.normal((n, 2)) @ q1[:2, :]
    y = mx.random.normal((n, 2)) @ q2[:2, :]

    sweep, eps = _build_sweep(x)
    overlap = sweep._compute_knn_overlap(x, y, k=3)
    overlap_identical = sweep._compute_knn_overlap(x, x, k=3)

    assert float(overlap) <= float(overlap_identical) - eps


def test_neighborhood_stability_scaling():
    """Neighborhood overlap should be invariant to global scaling."""
    mx.random.seed(2)
    x = mx.random.normal((30, 64))
    y = x * 10.0

    sweep, eps = _build_sweep(x)
    overlap = sweep._compute_knn_overlap(x, y, k=5)

    assert abs(float(overlap) - 1.0) <= eps


def test_neighborhood_stability_k_sensitivity():
    """Test sensitivity of k-NN overlap to k value."""
    mx.random.seed(3)
    x = mx.random.normal((40, 64))
    y = x + mx.random.normal((40, 64)) * 0.1  # Add noise

    sweep, eps = _build_sweep(x)
    overlap_k3 = sweep._compute_knn_overlap(x, y, k=3)
    overlap_k10 = sweep._compute_knn_overlap(x, y, k=10)

    assert float(overlap_k10) * 10 + eps >= float(overlap_k3) * 3


def test_neighborhood_stability_small_n():
    """Test neighborhood stability with small number of points."""
    x = mx.array([[1.0, 0.0], [0.0, 1.0]])
    y = mx.array([[0.0, 1.0], [1.0, 0.0]])  # Swapped

    sweep, eps = _build_sweep(x)
    overlap = sweep._compute_knn_overlap(x, y, k=1)

    # Point 0's neighbor in X is Point 1.
    # Point 0's neighbor in Y is Point 1.
    # So overlap should still be 1.0 because the relative neighborhoods are preserved.
    # Wait, distance between P0 and P1 is sqrt(2) in both.
    # KNN(P0, X) = {P1}
    # KNN(P0, Y) = {P1}
    # Overlap = 1.0
    assert abs(float(overlap) - 1.0) <= eps
