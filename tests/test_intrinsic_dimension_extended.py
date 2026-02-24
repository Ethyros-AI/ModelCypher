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

"""Extended tests for intrinsic dimension estimation.

Tests critical APIs:
- IntrinsicDimension.compute(): TwoNN-based intrinsic dimension
- IntrinsicDimension.compute_two_nn(): Static method variant
- IntrinsicDimension.batch_compute(): GPU-batched computation
- IntrinsicDimension.local_dimension_map(): Per-point ID estimates
- IntrinsicDimension.detect_dimension_deficiency(): Find collapsed regions
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    LocalDimensionMap,
    TwoNNEstimate,
)
from modelcypher.core.domain.geometry.numerical_stability import all_finite, division_epsilon


@pytest.fixture
def backend():
    return get_default_backend()


class TestIntrinsicDimensionCompute:
    """Tests for IntrinsicDimension.compute()."""

    def test_basic_computation(self, backend):
        """Basic intrinsic dimension computation should work."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert isinstance(result, TwoNNEstimate)
        assert result.intrinsic_dimension > 0
        assert result.sample_count == 32
        assert result.usable_count > 0

    def test_minimum_samples_required(self, backend):
        """Fewer than 3 samples should raise error."""
        points = backend.random_normal((2, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        with pytest.raises(Exception):  # EstimatorError
            estimator.compute(points)

    def test_three_samples_minimum(self, backend):
        """Exactly 3 samples should work (minimum valid)."""
        points = backend.random_normal((3, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert result.sample_count == 3

    def test_dimension_bounded_by_ambient(self, backend):
        """Intrinsic dimension should not exceed ambient dimension."""
        ambient_dim = 8
        points = backend.random_normal((64, ambient_dim))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        eps = division_epsilon(backend, points)
        assert result.intrinsic_dimension <= ambient_dim + eps

    def test_with_confidence_interval(self, backend):
        """Computing with CI should return valid intervals."""
        points = backend.random_normal((64, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points, with_ci=True)

        assert result.ci is not None
        assert result.ci.lower <= result.intrinsic_dimension
        assert result.ci.upper >= result.intrinsic_dimension
        assert result.ci.resamples > 0

    def test_small_sample_ci(self, backend):
        """Small samples still produce CI via bootstrap resampling."""
        points = backend.random_normal((8, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points, with_ci=True)

        # Bootstrap works with any resample count >= 2
        # No arbitrary minimum - the math handles small samples correctly
        assert result.ci is not None
        assert result.ci.resamples == 8  # resamples = sample_count

    def test_usable_count_valid(self, backend):
        """Usable count should be <= sample count."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert result.usable_count <= result.sample_count


class TestComputeTwoNNStatic:
    """Tests for IntrinsicDimension.compute_two_nn() static method."""

    def test_static_method_works(self, backend):
        """Static method should work without instance."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        result = IntrinsicDimension.compute_two_nn(points, backend=backend)

        assert isinstance(result, TwoNNEstimate)
        assert result.intrinsic_dimension > 0

    def test_accepts_list_input(self, backend):
        """Should accept list of lists as input."""
        points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        result = IntrinsicDimension.compute_two_nn(points, backend=backend)

        assert result.sample_count == 3

    def test_with_ci_flag(self, backend):
        """with_ci flag should be passed through."""
        points = backend.random_normal((50, 16))  # Sufficient for bootstrap quantiles
        backend.eval(points)

        result = IntrinsicDimension.compute_two_nn(points, backend=backend, with_ci=True)

        # Should have CI (sample size supports quantile resolution)
        assert result.ci is not None


class TestBatchCompute:
    """Tests for IntrinsicDimension.batch_compute()."""

    def test_empty_list(self, backend):
        """Empty list should return empty results."""
        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute([])

        assert results == []

    def test_single_point_cloud(self, backend):
        """Single point cloud should work."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute([points])

        assert len(results) == 1
        assert results[0] is not None
        assert results[0].intrinsic_dimension > 0

    def test_multiple_point_clouds(self, backend):
        """Multiple point clouds should all be processed."""
        clouds = [
            backend.random_normal((20, 8)),
            backend.random_normal((25, 12)),
            backend.random_normal((30, 16)),
        ]
        for c in clouds:
            backend.eval(c)

        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute(clouds)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert result.intrinsic_dimension > 0

    def test_too_few_samples_returns_none(self, backend):
        """Point clouds with < 3 samples should return None."""
        clouds = [
            backend.random_normal((32, 16)),  # Valid
            backend.random_normal((2, 16)),   # Too few
            backend.random_normal((16, 16)),  # Valid
        ]
        for c in clouds:
            backend.eval(c)

        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute(clouds)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None  # Too few samples
        assert results[2] is not None

    def test_varying_dimensions(self, backend):
        """Point clouds with different dimensions should work."""
        clouds = [
            backend.random_normal((20, 8)),
            backend.random_normal((20, 16)),
            backend.random_normal((20, 32)),
        ]
        for c in clouds:
            backend.eval(c)

        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute(clouds)

        assert len(results) == 3
        # All should succeed
        for result in results:
            assert result is not None


class TestLocalDimensionMap:
    """Tests for IntrinsicDimension.local_dimension_map()."""

    def test_basic_local_dimension(self, backend):
        """Basic local dimension map should work."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        assert isinstance(result, LocalDimensionMap)
        assert backend.shape(result.dimensions)[0] == 32
        assert result.modal_dimension > 0
        assert result.mean_dimension > 0
        assert result.k_neighbors > 0

    def test_dimensions_shape(self, backend):
        """Local dimensions should have same length as input."""
        n_points = 50
        points = backend.random_normal((n_points, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        assert backend.shape(result.dimensions)[0] == n_points

    def test_deficient_indices_valid(self, backend):
        """Deficient indices should be valid point indices."""
        points = backend.random_normal((64, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        for idx in result.deficient_indices:
            assert 0 <= idx < 64

    def test_statistics_reasonable(self, backend):
        """Mean and std should be reasonable values."""
        points = backend.random_normal((64, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        # Mean should be positive
        assert result.mean_dimension >= 0
        # Std should be non-negative
        assert result.std_dimension >= 0

    def test_small_sample_handles_gracefully(self, backend):
        """Small samples should not crash."""
        points = backend.random_normal((5, 16))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        # Should return a result (may have limited statistics)
        assert result is not None


class TestDetectDimensionDeficiency:
    """Tests for IntrinsicDimension.detect_dimension_deficiency()."""

    def test_returns_list(self, backend):
        """Should return a list of indices."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        deficient = IntrinsicDimension.detect_dimension_deficiency(points, backend=backend)

        assert isinstance(deficient, list)
        for idx in deficient:
            assert isinstance(idx, int)
            assert 0 <= idx < 32

    def test_empty_result_valid(self, backend):
        """No deficient points is a valid result."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        deficient = IntrinsicDimension.detect_dimension_deficiency(points, backend=backend)

        # May be empty (no statistical outliers)
        assert isinstance(deficient, list)


class TestIntrinsicDimensionMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_points=st.integers(min_value=10, max_value=64),
        d=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_id_positive(self, n_points, d):
        """Intrinsic dimension should always be positive."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert result.intrinsic_dimension > 0

    @given(
        n_points=st.integers(min_value=10, max_value=64),
        d=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_usable_count_bounded(self, n_points, d):
        """Usable count should be bounded by sample count."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert 0 < result.usable_count <= result.sample_count

    @given(
        n_clouds=st.integers(min_value=1, max_value=5),
        n_points=st.integers(min_value=10, max_value=32),
        d=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_batch_result_length(self, n_clouds, n_points, d):
        """Batch compute should return same number of results as inputs."""
        backend = get_default_backend()
        clouds = [backend.random_normal((n_points, d)) for _ in range(n_clouds)]
        for c in clouds:
            backend.eval(c)

        estimator = IntrinsicDimension(backend)
        results = estimator.batch_compute(clouds)

        assert len(results) == n_clouds

    @given(
        n_points=st.integers(min_value=20, max_value=64),
        d=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_local_dimensions_shape(self, n_points, d):
        """Local dimension map should have correct shape."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        assert backend.shape(result.dimensions)[0] == n_points

    @given(
        n_points=st.integers(min_value=20, max_value=64),
        d=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_deficient_indices_bounded(self, n_points, d):
        """Deficient indices should be valid point indices."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        deficient = IntrinsicDimension.detect_dimension_deficiency(points, backend=backend)

        assert all(0 <= idx < n_points for idx in deficient)
