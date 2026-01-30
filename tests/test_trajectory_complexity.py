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

"""Unit tests for trajectory complexity metrics.

Tests the TrajectoryComplexity class which measures intra-layer dynamics
to detect iterative refinement patterns ("looping") vs. direct feedforward flow.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.trajectory_complexity import (
    TrajectoryComplexity,
    TrajectoryComplexityResult,
    compute_trajectory_complexity,
)


class TestTrajectoryComplexityResult:
    """Tests for TrajectoryComplexityResult dataclass."""

    def test_frozen_dataclass(self):
        """TrajectoryComplexityResult should be immutable."""
        result = TrajectoryComplexityResult(
            path_length=10.0,
            direct_distance=5.0,
            path_length_ratio=2.0,
            mean_curvature=0.5,
            max_curvature=1.0,
            curvature_variance=0.1,
            mean_return_cka=0.3,
            max_return_cka=0.5,
            return_visit_count=2,
            trajectory_effective_rank=2.5,
            trajectory_spectral_entropy=0.9,
            layer_count=5,
            layer_indices=(0, 4, 8, 12, 16),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.path_length_ratio = 3.0

    def test_as_dict(self):
        """as_dict should include all fields."""
        result = TrajectoryComplexityResult(
            path_length=10.0,
            direct_distance=5.0,
            path_length_ratio=2.0,
            mean_curvature=0.5,
            max_curvature=1.0,
            curvature_variance=0.1,
            mean_return_cka=0.3,
            max_return_cka=0.5,
            return_visit_count=2,
            trajectory_effective_rank=2.5,
            trajectory_spectral_entropy=0.9,
            layer_count=5,
            layer_indices=(0, 4, 8, 12, 16),
        )

        d = result.as_dict()

        assert d["path_length"] == 10.0
        assert d["direct_distance"] == 5.0
        assert d["path_length_ratio"] == 2.0
        assert d["mean_curvature"] == 0.5
        assert d["max_curvature"] == 1.0
        assert d["curvature_variance"] == 0.1
        assert d["mean_return_cka"] == 0.3
        assert d["max_return_cka"] == 0.5
        assert d["return_visit_count"] == 2
        assert d["trajectory_effective_rank"] == 2.5
        assert d["trajectory_spectral_entropy"] == 0.9
        assert d["layer_count"] == 5
        assert d["layer_indices"] == [0, 4, 8, 12, 16]

    def test_nan_values_preserved(self):
        """NaN values should be preserved."""
        result = TrajectoryComplexityResult(
            path_length=0.0,
            direct_distance=0.0,
            path_length_ratio=float("nan"),
            mean_curvature=float("nan"),
            max_curvature=float("nan"),
            curvature_variance=float("nan"),
            mean_return_cka=float("nan"),
            max_return_cka=float("nan"),
            return_visit_count=0,
            trajectory_effective_rank=0.0,
            trajectory_spectral_entropy=0.0,
            layer_count=1,
            layer_indices=(0,),
        )

        assert math.isnan(result.path_length_ratio)
        d = result.as_dict()
        assert math.isnan(d["path_length_ratio"])


class TestTrajectoryComplexity:
    """Tests for TrajectoryComplexity class."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def tc(self, backend):
        """Create TrajectoryComplexity instance."""
        return TrajectoryComplexity(backend)

    def test_empty_input(self, tc):
        """Empty input should return degenerate result."""
        result = tc.compute({})

        assert result.layer_count == 0
        assert math.isnan(result.path_length_ratio)
        assert result.trajectory_effective_rank == 0.0

    def test_single_layer(self, tc, backend):
        """Single layer should return degenerate result."""
        act = backend.random_normal((64,))
        backend.eval(act)

        result = tc.compute({0: act})

        assert result.layer_count == 1
        assert math.isnan(result.path_length_ratio)

    def test_two_layers(self, tc, backend):
        """Two layers should compute path but not curvature."""
        act1 = backend.random_normal((64,))
        act2 = backend.random_normal((64,))
        backend.eval(act1, act2)

        result = tc.compute({0: act1, 4: act2})

        assert result.layer_count == 2
        # Path ratio can be computed with 2 points
        assert result.path_length > 0
        assert result.direct_distance > 0
        # Curvature needs 3 points
        assert math.isnan(result.mean_curvature)

    def test_straight_line_trajectory(self, tc, backend):
        """Straight line trajectory should have path_ratio ~= 1.0."""
        # Create activations along a straight line
        base = backend.random_normal((64,))
        direction = backend.random_normal((64,))
        backend.eval(base, direction)

        acts = {}
        for i in range(5):
            acts[i] = base + direction * float(i)
            backend.eval(acts[i])

        result = tc.compute(acts)

        # Path ratio for straight line should be ~1.0
        assert 0.99 < result.path_length_ratio < 1.01

        # Curvature should be ~0 for straight line
        assert result.mean_curvature < 0.01

    def test_zigzag_trajectory(self, tc, backend):
        """Zigzag trajectory should have high path_ratio and curvature."""
        # Create activations that zigzag
        base = backend.zeros((64,))
        backend.eval(base)

        acts = {}
        for i in range(5):
            offset = backend.zeros((64,))
            # Alternate direction
            if i % 2 == 0:
                offset = backend.array([1.0 if j < 32 else 0.0 for j in range(64)])
            else:
                offset = backend.array([0.0 if j < 32 else 1.0 for j in range(64)])
            acts[i] = base + offset * float(i + 1)
            backend.eval(acts[i])

        result = tc.compute(acts)

        # Path ratio should be > 1 for non-straight path
        assert result.path_length_ratio > 1.0

        # Should have measurable curvature
        assert result.mean_curvature > 0

    def test_return_visits_computation(self, tc, backend):
        """Return visits CKA is computed for non-adjacent layer pairs."""
        # Note: CKA with single-sample (1, dim) arrays typically returns NaN
        # because the centering step zeros out single vectors.
        # This test verifies the computation runs without error.
        acts = {}
        for i in range(5):
            acts[i] = backend.random_normal((64,))
            backend.eval(acts[i])

        result = tc.compute(acts)

        # Return visit count is always computed (may be 0 if CKA returns NaN)
        assert result.return_visit_count >= 0
        # Structure should be correct
        assert result.layer_count == 5

    def test_spectral_rank_increases_with_diversity(self, tc, backend):
        """More diverse trajectory directions should increase effective rank."""
        # Simple trajectory (low rank)
        base = backend.random_normal((64,))
        direction = backend.random_normal((64,))
        backend.eval(base, direction)

        simple_acts = {i: base + direction * float(i) for i in range(5)}
        for k in simple_acts:
            backend.eval(simple_acts[k])

        simple_result = tc.compute(simple_acts)

        # Diverse trajectory (higher rank)
        diverse_acts = {}
        for i in range(5):
            # Each layer has independent direction
            diverse_acts[i] = backend.random_normal((64,))
            backend.eval(diverse_acts[i])

        diverse_result = tc.compute(diverse_acts)

        # Diverse should have higher effective rank
        assert diverse_result.trajectory_effective_rank > simple_result.trajectory_effective_rank

    def test_2d_input_mean_pooled(self, tc, backend):
        """2D input [samples, features] should be mean-pooled."""
        # Multiple samples per layer
        acts = {}
        for i in range(5):
            samples = backend.random_normal((10, 64))  # 10 samples
            backend.eval(samples)
            acts[i] = samples

        result = tc.compute(acts)

        # Should work with 2D input
        assert result.layer_count == 5
        assert result.path_length > 0


class TestConvenienceFunction:
    """Tests for compute_trajectory_complexity convenience function."""

    def test_compute_trajectory_complexity_function(self):
        """Convenience function should work without explicit backend."""
        backend = get_default_backend()

        acts = {}
        for i in range(5):
            acts[i] = backend.random_normal((64,))
            backend.eval(acts[i])

        result = compute_trajectory_complexity(acts)

        assert isinstance(result, TrajectoryComplexityResult)
        assert result.layer_count == 5


class TestTrajectoryComplexityProperties:
    """Property-based tests for trajectory complexity metrics."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    def test_path_ratio_always_geq_one(self, backend):
        """Path ratio should always be >= 1 (triangle inequality)."""
        tc = TrajectoryComplexity(backend)

        # Random trajectory
        acts = {}
        for i in range(10):
            acts[i] = backend.random_normal((32,))
            backend.eval(acts[i])

        result = tc.compute(acts)

        if not math.isnan(result.path_length_ratio):
            assert result.path_length_ratio >= 0.99  # Allow small numerical error

    def test_curvature_bounded(self, backend):
        """Curvature should be bounded [0, pi]."""
        tc = TrajectoryComplexity(backend)

        # Random trajectory
        acts = {}
        for i in range(10):
            acts[i] = backend.random_normal((32,))
            backend.eval(acts[i])

        result = tc.compute(acts)

        if not math.isnan(result.mean_curvature):
            assert 0 <= result.mean_curvature <= math.pi
        if not math.isnan(result.max_curvature):
            assert 0 <= result.max_curvature <= math.pi

    def test_effective_rank_positive(self, backend):
        """Effective rank should always be positive."""
        tc = TrajectoryComplexity(backend)

        acts = {}
        for i in range(5):
            acts[i] = backend.random_normal((32,))
            backend.eval(acts[i])

        result = tc.compute(acts)

        assert result.trajectory_effective_rank >= 0

    def test_cka_bounded(self, backend):
        """CKA values should be bounded [0, 1]."""
        tc = TrajectoryComplexity(backend)

        acts = {}
        for i in range(5):
            acts[i] = backend.random_normal((32,))
            backend.eval(acts[i])

        result = tc.compute(acts)

        if not math.isnan(result.mean_return_cka):
            assert 0 <= result.mean_return_cka <= 1.0
        if not math.isnan(result.max_return_cka):
            assert 0 <= result.max_return_cka <= 1.0
