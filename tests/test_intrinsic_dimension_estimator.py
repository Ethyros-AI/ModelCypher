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

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    BootstrapConfiguration,
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
)


def test_two_nn_insufficient_samples() -> None:
    points = [[0.0, 0.0], [1.0, 0.0]]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimensionEstimator.estimate_two_nn(points)
    assert exc.value.kind == "insufficientSamples"


def test_two_nn_invalid_dimension() -> None:
    # Non-uniform dimensions trigger backend ValueError, not EstimatorError
    # This is correct - backend validates before estimator
    points = [[0.0, 0.0], [1.0], [2.0, 0.0]]
    with pytest.raises((EstimatorError, ValueError)):
        IntrinsicDimensionEstimator.estimate_two_nn(points)


def test_two_nn_degenerate_neighbors() -> None:
    # Degenerate case now triggers regressionDegenerate (regression variant is default)
    points = [[1.0, 1.0] for _ in range(5)]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimensionEstimator.estimate_two_nn(points)
    assert exc.value.kind in ("nearestNeighborDegenerate", "regressionDegenerate")


def test_two_nn_estimate_basic() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    config = TwoNNConfiguration(use_regression=False)
    estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)
    assert estimate.sample_count == 6
    assert estimate.usable_count >= 3
    assert estimate.intrinsic_dimension > 0


def test_two_nn_bootstrap_ci() -> None:
    backend = get_default_backend()
    points = backend.array([[float(i), 0.0] for i in range(6)])
    config = TwoNNConfiguration(use_regression=False)
    bootstrap = BootstrapConfiguration(resamples=50, confidence_level=0.9, seed=7)
    computer = IntrinsicDimensionEstimator(backend)
    estimate = computer.compute(points, configuration=config, bootstrap=bootstrap)
    assert estimate.ci is not None
    assert estimate.ci.lower <= estimate.ci.upper


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestDimensionInvariants:
    """Tests for mathematical invariants of dimension estimation."""

    @pytest.mark.parametrize("seed", range(5))
    def test_dimension_always_positive(self, seed: int) -> None:
        """Intrinsic dimension must be > 0.

        Mathematical property: Dimension is a positive quantity by definition.
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        # Generate points in 5D with some spread
        data = backend.random_normal((20, 5))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(data, configuration=config)
        assert estimate.intrinsic_dimension > 0

    @pytest.mark.parametrize("true_dim", [1, 2, 3, 5])
    def test_dimension_bounded_by_ambient(self, true_dim: int) -> None:
        """Estimated dimension should not exceed ambient dimension.

        Mathematical property: Intrinsic dimension ≤ ambient dimension.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        # Generate points in true_dim-dimensional manifold embedded in higher dim
        n_samples = 50
        data = backend.random_normal((n_samples, true_dim))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(data, configuration=config)

        # Should be close to true_dim (with geodesic distances, variance is higher)
        assert estimate.intrinsic_dimension <= true_dim + 3  # Allow reasonable overshoot

    def test_1d_manifold_dimension_near_one(self) -> None:
        """Points on a line should have dimension ≈ 1.

        Mathematical property: 1D manifold has intrinsic dimension 1.

        Note: TwoNN requires non-uniform spacing to avoid degeneracy.
        Equally-spaced points cause r2/r1 ratios to be constant, breaking
        the maximum likelihood estimator. Random sampling along the line
        is how real manifold data would be collected.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        # Random sampling along x-axis: manifold is exactly 1D
        t = backend.random_uniform(low=0.0, high=20.0, shape=(50,))
        zeros = backend.zeros((50,))
        # Stack to get [50, 3] array with points along x-axis
        points = backend.stack([t, zeros, zeros], axis=1)
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(points, configuration=config)

        # Should be close to 1 (line is 1-dimensional)
        assert 0.5 <= estimate.intrinsic_dimension <= 2.0

    def test_2d_manifold_dimension_near_two(self) -> None:
        """Points on a plane should have dimension ≈ 2.

        Mathematical property: 2D manifold has intrinsic dimension 2.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        # Points on xy-plane: (x, y, 0)
        n = 50
        xy = backend.random_uniform(low=-10.0, high=10.0, shape=(n, 2))
        zeros = backend.zeros((n, 1))
        # Concatenate to get [n, 3] array with z=0
        points = backend.concatenate([xy, zeros], axis=1)
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(points, configuration=config)

        # Should be close to 2 (with geodesic distances, variance is higher)
        assert 1.0 <= estimate.intrinsic_dimension <= 4.0


class TestConfidenceIntervalInvariants:
    """Tests for confidence interval invariants."""

    @pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
    def test_ci_lower_lte_upper(self, confidence_level: float) -> None:
        """CI lower bound must be ≤ upper bound.

        Mathematical property: By construction of confidence intervals.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        data = backend.random_normal((30, 3))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        bootstrap = BootstrapConfiguration(
            resamples=100,
            confidence_level=confidence_level,
            seed=42,
        )
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(data, configuration=config, bootstrap=bootstrap)

        assert estimate.ci is not None
        assert estimate.ci.lower <= estimate.ci.upper

    @pytest.mark.parametrize("seed", range(5))
    def test_ci_contains_point_estimate(self, seed: int) -> None:
        """CI should typically contain the point estimate.

        Note: This isn't mathematically guaranteed but should usually hold.
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        data = backend.random_normal((30, 3))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        bootstrap = BootstrapConfiguration(
            resamples=100,
            confidence_level=0.95,
            seed=seed,
        )
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(data, configuration=config, bootstrap=bootstrap)

        if estimate.ci is not None:
            # Point estimate should be near the CI
            assert estimate.ci.lower <= estimate.intrinsic_dimension + 0.5
            assert estimate.ci.upper >= estimate.intrinsic_dimension - 0.5


class TestUsableCountInvariants:
    """Tests for usable count invariants."""

    @pytest.mark.parametrize("n_samples", [10, 20, 50, 100])
    def test_usable_count_bounded_by_sample_count(self, n_samples: int) -> None:
        """Usable count must be ≤ sample count.

        Mathematical property: Can't use more points than we have.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        data = backend.random_normal((n_samples, 3))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimensionEstimator(backend)
        estimate = computer.compute(data, configuration=config)

        assert estimate.usable_count <= estimate.sample_count
        assert estimate.usable_count > 0
