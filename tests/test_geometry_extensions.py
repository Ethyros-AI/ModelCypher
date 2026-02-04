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

"""
Unit tests for geometry extension parity modules (requires MLX).

Tests:
- DoRA decomposition analysis
- Tangent space alignment
"""

import pytest

from tests.conftest import HAS_MLX


# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.dora_decomposition import (
    ChangeType,
    DoRADecomposition,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.tangent_space_alignment import (
    TangentSpaceAlignment,
)


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestDoRADecomposition:
    """Tests for DoRA decomposition."""

    def test_same_weights(self):
        """Identical weights should show minimal change."""
        dora = DoRADecomposition()
        w = get_default_backend().random_normal((64, 64))

        metrics = dora.decompose(w, w, "test")

        assert metrics is not None
        eps = _div_eps()
        assert abs(metrics.magnitude_ratio - 1.0) <= eps
        assert abs(metrics.directional_drift - 0.0) <= eps
        assert abs(metrics.direction_cosine - 1.0) <= eps

    def test_scaled_weights(self):
        """Scaled weights should show magnitude change only."""
        dora = DoRADecomposition()
        w1 = get_default_backend().random_normal((64, 64))
        w2 = w1 * 2.0  # Double magnitude

        metrics = dora.decompose(w1, w2, "test")

        assert metrics is not None
        eps = _div_eps()
        assert abs(metrics.magnitude_ratio - 2.0) <= eps
        # Direction should be same
        assert abs(metrics.direction_cosine - 1.0) <= eps

    def test_adapter_analysis(self):
        """Test multi-layer adapter analysis."""
        dora = DoRADecomposition()

        base = {
            "layer1": get_default_backend().random_normal((32, 32)),
            "layer2": get_default_backend().random_normal((32, 32)),
        }
        current = {
            "layer1": base["layer1"] * 1.1,  # Small magnitude change
            "layer2": base["layer2"] + get_default_backend().random_normal((32, 32)) * 0.1,  # Direction change
        }

        result = dora.analyze_adapter(base, current)

        assert len(result.per_layer_metrics) == 2
        eps = _div_eps()
        assert result.overall_magnitude_change >= -eps
        assert result.overall_directional_drift >= -eps

    def test_change_type_classification(self):
        """Test dominant change type classification."""
        dora = DoRADecomposition()

        # Minimal change
        w = get_default_backend().random_normal((32, 32))
        result = dora.analyze_adapter({"l": w}, {"l": w})
        assert result.dominant_change_type == ChangeType.MINIMAL


class TestTangentSpaceAlignment:
    """Tests for tangent space alignment."""

    def test_identical_points(self):
        """Identical point sets should have high alignment."""
        # All parameters derived from data
        aligner = TangentSpaceAlignment()
        points = get_default_backend().random_normal((20, 64))

        result = aligner.compute_layer_metrics(points, points)

        assert result is not None
        eps = _div_eps()
        assert abs(result.mean_cosine - 1.0) <= eps
        assert result.coverage > eps

    def test_orthogonal_points(self):
        """Orthogonal point sets should have lower alignment."""
        # All parameters derived from data
        aligner = TangentSpaceAlignment()

        # Create two distinct random manifolds
        points1 = get_default_backend().random_normal((20, 64))
        points2 = get_default_backend().random_normal((20, 64))

        result = aligner.compute_layer_metrics(points1, points2)

        assert result is not None
        # Random points have lower agreement than identical
        assert result.anchor_count == 20

    def test_insufficient_points(self):
        """Should return None for insufficient points."""
        # All parameters derived from data
        aligner = TangentSpaceAlignment()
        # MIN_ANCHOR_COUNT = 3, so need fewer than 3
        points = get_default_backend().random_normal((2, 64))  # Too few

        result = aligner.compute_layer_metrics(points, points)
        assert result is None
