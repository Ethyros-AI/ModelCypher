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
    IntrinsicDimension,
    TwoNNConfiguration,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def test_two_nn_insufficient_samples() -> None:
    points = [[0.0, 0.0], [1.0, 0.0]]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimension.compute_two_nn(points)
    assert exc.value.kind == "insufficientSamples"


def test_two_nn_invalid_dimension() -> None:
    # Non-uniform dimensions trigger backend ValueError, not EstimatorError
    # This is correct - backend validates before estimator
    points = [[0.0, 0.0], [1.0], [2.0, 0.0]]
    with pytest.raises((EstimatorError, ValueError)):
        IntrinsicDimension.compute_two_nn(points)


def test_two_nn_degenerate_neighbors() -> None:
    # With machine epsilon precision, identical points are correctly detected
    # as degenerate. All distances are below machine epsilon, so no valid
    # neighbor ratios can be computed.
    points = [[1.0, 1.0] for _ in range(5)]
    with pytest.raises(EstimatorError) as exc:
        IntrinsicDimension.compute_two_nn(points)
    # Error kind can vary: two_nn (insufficient samples), regressionDegenerate, etc.
    assert exc.value.kind in ("two_nn", "insufficientSamples", "nearestNeighborDegenerate", "regressionDegenerate")


def test_two_nn_estimate_basic() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    config = TwoNNConfiguration(use_regression=False)
    estimate = IntrinsicDimension.compute_two_nn(points, configuration=config)
    assert estimate.sample_count == 6
    assert estimate.usable_count >= 3
    backend = get_default_backend()
    eps = _eps(backend, estimate.intrinsic_dimension)
    assert estimate.intrinsic_dimension > eps


def test_two_nn_bootstrap_ci() -> None:
    backend = get_default_backend()
    points = backend.array([[float(i), 0.0] for i in range(6)])
    config = TwoNNConfiguration(use_regression=False)
    bootstrap = BootstrapConfiguration(resamples=50, confidence_level=0.9, seed=7)
    computer = IntrinsicDimension(backend)
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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(data, configuration=config)
        eps = _eps(backend, estimate.intrinsic_dimension)
        assert estimate.intrinsic_dimension > eps

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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(data, configuration=config)

        eps = _eps(backend, estimate.intrinsic_dimension, float(true_dim))
        assert estimate.intrinsic_dimension <= true_dim + eps

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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, 1.0)
        assert estimate.ci.lower - eps <= 1.0 <= estimate.ci.upper + eps

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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, 2.0)
        assert estimate.ci.lower - eps <= 2.0 <= estimate.ci.upper + eps


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
        computer = IntrinsicDimension(backend)
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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(data, configuration=config, bootstrap=bootstrap)

        if estimate.ci is not None:
            # Point estimate should be near the CI
            eps = _eps(
                backend,
                estimate.ci.lower,
                estimate.ci.upper,
                estimate.intrinsic_dimension,
            )
            assert estimate.ci.lower <= estimate.intrinsic_dimension + eps
            assert estimate.ci.upper >= estimate.intrinsic_dimension - eps


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
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(data, configuration=config)

        assert estimate.usable_count <= estimate.sample_count
        assert estimate.usable_count > 0


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestIntrinsicDimensionHypothesis:
    """Hypothesis-based property tests for intrinsic dimension estimation."""

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_dim=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dimension_always_positive_hypothesis(
        self, n_samples: int, n_dim: int, seed: int
    ):
        """Intrinsic dimension is always positive (Hypothesis)."""
        backend = get_default_backend()
        backend.random_seed(seed)
        data = backend.random_normal((n_samples, n_dim))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        try:
            estimate = computer.compute(data, configuration=config)
            eps = _eps(backend, estimate.intrinsic_dimension)
            assert estimate.intrinsic_dimension > eps
        except EstimatorError:
            # Some degenerate configurations may fail
            assume(False)

    @given(
        n_samples=st.integers(min_value=15, max_value=40),
        n_dim=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_usable_count_bounded_hypothesis(
        self, n_samples: int, n_dim: int, seed: int
    ):
        """Usable count <= sample count (Hypothesis)."""
        backend = get_default_backend()
        backend.random_seed(seed)
        data = backend.random_normal((n_samples, n_dim))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        try:
            estimate = computer.compute(data, configuration=config)
            assert estimate.usable_count <= estimate.sample_count
            assert estimate.sample_count == n_samples
        except EstimatorError:
            assume(False)

    @given(
        n_samples=st.integers(min_value=40, max_value=80),
        ambient_dim=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dimension_bounded_by_ambient_hypothesis(
        self, n_samples: int, ambient_dim: int, seed: int
    ):
        """Intrinsic dimension should be bounded by ambient dimension (Hypothesis).

        Note: Uses n_samples >= 40 for more stable estimates. TwoNN with small
        samples and geodesic distances can have higher variance.
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        data = backend.random_normal((n_samples, ambient_dim))
        backend.eval(data)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        try:
            estimate = computer.compute(data, configuration=config)
            # Allow wiggle room - geodesic distances + small samples = variance
            eps = _eps(backend, estimate.intrinsic_dimension, float(ambient_dim))
            assert estimate.intrinsic_dimension <= ambient_dim + eps
        except EstimatorError:
            assume(False)


# =============================================================================
# Synthetic Manifold Ground Truth Tests
# =============================================================================


class TestSyntheticManifoldDimension:
    """Ground truth tests on synthetic manifolds with known dimension."""

    def test_sphere_dimension(self) -> None:
        """n-sphere embedded in R^{n+1} should have dimension n.

        Mathematical property: S^n is an n-dimensional manifold.
        Testing S^2 (surface of 3D sphere) which should have ID ≈ 2.
        """
        import math

        backend = get_default_backend()
        backend.random_seed(42)
        n_samples = 100

        # Sample uniformly on unit sphere S^2 using rejection sampling
        # Generate random 3D points and normalize
        points_list = []
        backend.random_seed(42)
        for i in range(n_samples * 3):  # Generate extra to ensure enough valid points
            backend.random_seed(42 + i)
            point = backend.random_normal((3,))
            backend.eval(point)
            point_np = backend.to_numpy(point)
            norm = math.sqrt(sum(x * x for x in point_np))
            eps = _eps(backend, norm)
            if norm > eps:
                normalized = [x / norm for x in point_np]
                points_list.append(normalized)
            if len(points_list) >= n_samples:
                break

        points = backend.array(points_list[:n_samples])
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, 2.0)
        assert estimate.ci.lower - eps <= 2.0 <= estimate.ci.upper + eps

    def test_swiss_roll_dimension(self) -> None:
        """Swiss roll is a 2D manifold in 3D space.

        Mathematical property: Swiss roll has intrinsic dimension 2.
        """
        import math

        backend = get_default_backend()
        backend.random_seed(42)
        n_samples = 150

        # Generate swiss roll: (t*cos(t), height, t*sin(t))
        # t ranges from ~1.5π to 4.5π for a nice roll
        points_list = []
        for i in range(n_samples):
            # Uniform in (t, height) space
            t = 1.5 * math.pi + (3.0 * math.pi) * (i / n_samples)
            # Add noise to t to break regularity
            backend.random_seed(42 + i)
            noise = backend.random_uniform(low=-0.2, high=0.2, shape=())
            backend.eval(noise)
            t += float(backend.to_numpy(noise).item())

            # Random height
            backend.random_seed(1000 + i)
            height = backend.random_uniform(low=0.0, high=10.0, shape=())
            backend.eval(height)
            h = float(backend.to_numpy(height).item())

            x = t * math.cos(t)
            y = h
            z = t * math.sin(t)
            points_list.append([x, y, z])

        points = backend.array(points_list)
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, 2.0)
        assert estimate.ci.lower - eps <= 2.0 <= estimate.ci.upper + eps

    def test_linear_subspace_dimension(self) -> None:
        """k-dimensional linear subspace in R^n should have dimension k.

        Mathematical property: Linear subspace of dimension k has ID = k.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        # Create 3D subspace in 10D ambient space
        true_dim = 3
        ambient_dim = 10
        n_samples = 100

        # Generate random basis for 3D subspace
        backend.random_seed(100)
        basis = backend.random_normal((true_dim, ambient_dim))
        backend.eval(basis)

        # Generate random coefficients
        backend.random_seed(42)
        coeffs = backend.random_normal((n_samples, true_dim))
        backend.eval(coeffs)

        # Project onto subspace: points = coeffs @ basis
        points = backend.matmul(coeffs, basis)
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, float(true_dim))
        assert estimate.ci.lower - eps <= true_dim <= estimate.ci.upper + eps

    def test_product_manifold_dimension(self) -> None:
        """Product of S^1 × S^1 (torus) should have dimension 2.

        Mathematical property: dim(M × N) = dim(M) + dim(N).
        S^1 × S^1 = T^2 (2-torus) has dimension 1 + 1 = 2.
        """
        import math

        backend = get_default_backend()
        backend.random_seed(42)
        n_samples = 150

        # Generate points on flat torus embedded in R^4:
        # (cos(θ), sin(θ), cos(φ), sin(φ))
        points_list = []
        for i in range(n_samples):
            theta = 2 * math.pi * (i / n_samples)
            # Add noise to theta
            backend.random_seed(42 + i)
            noise_theta = backend.random_uniform(low=-0.1, high=0.1, shape=())
            backend.eval(noise_theta)
            theta += float(backend.to_numpy(noise_theta).item())

            # Random phi
            backend.random_seed(1000 + i)
            phi_rand = backend.random_uniform(low=0.0, high=2 * math.pi, shape=())
            backend.eval(phi_rand)
            phi = float(backend.to_numpy(phi_rand).item())

            points_list.append([
                math.cos(theta),
                math.sin(theta),
                math.cos(phi),
                math.sin(phi),
            ])

        points = backend.array(points_list)
        backend.eval(points)

        config = TwoNNConfiguration(use_regression=True)
        computer = IntrinsicDimension(backend)
        estimate = computer.compute(
            points,
            configuration=config,
            bootstrap=BootstrapConfiguration(),
        )
        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, 2.0)
        assert estimate.ci.lower - eps <= 2.0 <= estimate.ci.upper + eps
