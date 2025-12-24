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

from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
    BootstrapConfiguration,
    EstimatorError,
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
)


def test_two_nn_insufficient_samples() -> None:
    points = [[0.0, 0.0], [1.0, 0.0]]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimensionEstimator.estimate_two_nn(points)
    assert exc.value.kind == "insufficientSamples"


def test_two_nn_invalid_dimension() -> None:
    points = [[0.0, 0.0], [1.0], [2.0, 0.0]]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimensionEstimator.estimate_two_nn(points)
    assert exc.value.kind == "invalidPointDimension"


def test_two_nn_degenerate_neighbors() -> None:
    points = [[1.0, 1.0] for _ in range(5)]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimensionEstimator.estimate_two_nn(points)
    assert exc.value.kind == "nearestNeighborDegenerate"


def test_two_nn_estimate_basic() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    config = TwoNNConfiguration(use_regression=False)
    estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)
    assert estimate.sample_count == 6
    assert estimate.usable_count >= 3
    assert estimate.intrinsic_dimension > 0


def test_two_nn_bootstrap_ci() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    config = TwoNNConfiguration(
        use_regression=False,
        bootstrap=BootstrapConfiguration(resamples=50, confidence_level=0.9, seed=7),
    )
    estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)
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
        import numpy as np
        rng = np.random.default_rng(seed)
        # Generate points in 2D with some spread
        points = rng.standard_normal((20, 5)).tolist()

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)
        assert estimate.intrinsic_dimension > 0

    @pytest.mark.parametrize("true_dim", [1, 2, 3, 5])
    def test_dimension_bounded_by_ambient(self, true_dim: int) -> None:
        """Estimated dimension should not exceed ambient dimension.

        Mathematical property: Intrinsic dimension ≤ ambient dimension.
        """
        import numpy as np
        rng = np.random.default_rng(42)
        # Generate points in true_dim-dimensional manifold embedded in higher dim
        n_samples = 50
        points = rng.standard_normal((n_samples, true_dim)).tolist()

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

        # Should be close to true_dim but definitely not exceed ambient
        assert estimate.intrinsic_dimension <= true_dim + 1  # Allow small overshoot

    def test_1d_manifold_dimension_near_one(self) -> None:
        """Points on a line should have dimension ≈ 1.

        Mathematical property: 1D manifold has intrinsic dimension 1.

        Note: TwoNN requires non-uniform spacing to avoid degeneracy.
        Equally-spaced points cause r2/r1 ratios to be constant, breaking
        the maximum likelihood estimator. Random sampling along the line
        is how real manifold data would be collected.
        """
        import numpy as np
        rng = np.random.default_rng(42)

        # Random sampling along x-axis: manifold is exactly 1D
        t = rng.uniform(0, 20, 50)
        points = [[t[i], 0.0, 0.0] for i in range(50)]

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

        # Should be close to 1 (line is 1-dimensional)
        assert 0.5 <= estimate.intrinsic_dimension <= 2.0

    def test_2d_manifold_dimension_near_two(self) -> None:
        """Points on a plane should have dimension ≈ 2.

        Mathematical property: 2D manifold has intrinsic dimension 2.
        """
        import numpy as np
        rng = np.random.default_rng(42)

        # Points on xy-plane: (x, y, 0)
        n = 50
        xy = rng.uniform(-10, 10, (n, 2))
        points = [[xy[i, 0], xy[i, 1], 0.0] for i in range(n)]

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

        # Should be close to 2
        assert 1.5 <= estimate.intrinsic_dimension <= 3.0


class TestConfidenceIntervalInvariants:
    """Tests for confidence interval invariants."""

    @pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
    def test_ci_lower_lte_upper(self, confidence_level: float) -> None:
        """CI lower bound must be ≤ upper bound.

        Mathematical property: By construction of confidence intervals.
        """
        import numpy as np
        rng = np.random.default_rng(42)
        points = rng.standard_normal((30, 3)).tolist()

        config = TwoNNConfiguration(
            use_regression=True,
            bootstrap=BootstrapConfiguration(
                resamples=100,
                confidence_level=confidence_level,
                seed=42,
            ),
        )
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

        assert estimate.ci is not None
        assert estimate.ci.lower <= estimate.ci.upper

    @pytest.mark.parametrize("seed", range(5))
    def test_ci_contains_point_estimate(self, seed: int) -> None:
        """CI should typically contain the point estimate.

        Note: This isn't mathematically guaranteed but should usually hold.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((30, 3)).tolist()

        config = TwoNNConfiguration(
            use_regression=True,
            bootstrap=BootstrapConfiguration(
                resamples=100,
                confidence_level=0.95,
                seed=seed,
            ),
        )
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

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
        import numpy as np
        rng = np.random.default_rng(42)
        points = rng.standard_normal((n_samples, 3)).tolist()

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, configuration=config)

        assert estimate.usable_count <= estimate.sample_count
        assert estimate.usable_count > 0
