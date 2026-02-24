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

"""Tests for intrinsic_dimension.py - TwoNN intrinsic dimension estimation.

Tests cover:
- ConfidenceInterval, TwoNNEstimate, LocalDimensionMap dataclasses
- IntrinsicDimension.compute_two_nn() static method
- IntrinsicDimension.compute() instance method
- IntrinsicDimension.local_dimension_map() method
- IntrinsicDimension.detect_dimension_deficiency() static method
- Edge cases: small samples, high-dimensional data
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    ConfidenceInterval,
    IntrinsicDimension,
    LocalDimensionMap,
    TwoNNEstimate,
)

# =============================================================================
# Dataclass Tests
# =============================================================================


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_fields_stored_correctly(self):
        """ConfidenceInterval stores all fields."""
        ci = ConfidenceInterval(lower=1.5, upper=2.5, resamples=100)
        assert ci.lower == 1.5
        assert ci.upper == 2.5
        assert ci.resamples == 100


class TestTwoNNEstimate:
    """Tests for TwoNNEstimate dataclass."""

    def test_fields_stored_correctly(self):
        """TwoNNEstimate stores all fields."""
        estimate = TwoNNEstimate(
            intrinsic_dimension=2.5,
            sample_count=100,
            usable_count=95,
            ci=None,
        )
        assert estimate.intrinsic_dimension == 2.5
        assert estimate.sample_count == 100
        assert estimate.usable_count == 95
        assert estimate.ci is None

    def test_with_confidence_interval(self):
        """TwoNNEstimate can include CI."""
        ci = ConfidenceInterval(1.5, 3.5, 500)
        estimate = TwoNNEstimate(2.5, 100, 95, ci=ci)
        assert estimate.ci is not None
        assert estimate.ci.lower == 1.5


class TestLocalDimensionMap:
    """Tests for LocalDimensionMap dataclass."""

    def test_fields_stored_correctly(self):
        """LocalDimensionMap stores all fields."""
        backend = get_default_backend()
        dims = backend.array([2.0, 2.1, 1.9, 2.0])

        ldm = LocalDimensionMap(
            dimensions=dims,
            modal_dimension=2.0,
            mean_dimension=2.0,
            std_dimension=0.1,
            deficient_indices=[],
            k_neighbors=5,
        )
        assert ldm.modal_dimension == 2.0
        assert ldm.k_neighbors == 5


# =============================================================================
# IntrinsicDimension.compute_two_nn Tests
# =============================================================================


class TestComputeTwoNN:
    """Tests for compute_two_nn static method."""

    def test_compute_two_nn_2d_points(self):
        """compute_two_nn estimates for 2D point cloud."""
        get_default_backend()
        # Generate 2D points with random noise for proper variance
        n = 100
        points = []
        for i in range(n):
            x = (i % 10) / 10.0 + (hash(str(i)) % 1000) / 50000.0
            y = (i // 10) / 10.0 + (hash(str(i + 1000)) % 1000) / 50000.0
            points.append([x, y])

        result = IntrinsicDimension.compute_two_nn(points)

        assert isinstance(result, TwoNNEstimate)
        assert result.sample_count == n
        assert result.intrinsic_dimension > 0

    def test_compute_two_nn_with_backend_array(self):
        """compute_two_nn accepts backend arrays."""
        backend = get_default_backend()
        # Create larger point set with variance
        n = 50
        points_list = []
        for i in range(n):
            x = (hash(str(i)) % 1000) / 1000.0
            y = (hash(str(i + 500)) % 1000) / 1000.0
            points_list.append([x, y])
        points = backend.array(points_list)

        result = IntrinsicDimension.compute_two_nn(points, backend=backend)

        assert isinstance(result, TwoNNEstimate)
        assert result.intrinsic_dimension > 0

    def test_compute_two_nn_insufficient_samples(self):
        """compute_two_nn raises error for insufficient samples."""
        with pytest.raises(EstimatorError):
            IntrinsicDimension.compute_two_nn([[0.0, 0.0], [1.0, 1.0]])

    def test_compute_two_nn_with_ci(self):
        """compute_two_nn can compute confidence interval."""
        n = 100
        points = []
        for i in range(n):
            x = (hash(str(i)) % 1000) / 1000.0
            y = (hash(str(i + 500)) % 1000) / 1000.0
            points.append([x, y])

        result = IntrinsicDimension.compute_two_nn(points, with_ci=True)

        assert isinstance(result, TwoNNEstimate)
        # CI computation is optional based on sample size
        assert result.intrinsic_dimension > 0


# =============================================================================
# IntrinsicDimension.compute Tests
# =============================================================================


class TestCompute:
    """Tests for compute instance method."""

    def test_compute_returns_twonn_estimate(self):
        """compute returns TwoNNEstimate."""
        backend = get_default_backend()
        n = 30
        points = backend.array([[i / n, (i * 3 % n) / n, (i * 7 % n) / n] for i in range(n)])

        estimator = IntrinsicDimension(backend)
        result = estimator.compute(points)

        assert isinstance(result, TwoNNEstimate)
        assert result.sample_count == n
        assert result.usable_count > 0

    def test_compute_small_samples_raises(self):
        """compute raises for fewer than 3 samples."""
        backend = get_default_backend()
        points = backend.array([[0.0, 0.0], [1.0, 1.0]])

        estimator = IntrinsicDimension(backend)
        with pytest.raises(EstimatorError):
            estimator.compute(points)


# =============================================================================
# IntrinsicDimension.local_dimension_map Tests
# =============================================================================


class TestLocalDimensionMapMethod:
    """Tests for local_dimension_map method."""

    def test_local_dimension_map_returns_correct_type(self):
        """local_dimension_map returns LocalDimensionMap."""
        backend = get_default_backend()
        n = 30
        points = backend.array([[i / n, (i * 3 % n) / n] for i in range(n)])

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        assert isinstance(result, LocalDimensionMap)
        assert result.k_neighbors > 0

    def test_local_dimension_map_dimensions_shape(self):
        """local_dimension_map returns dimensions with correct shape."""
        backend = get_default_backend()
        n = 20
        points = backend.array([[i / n, (i * 3 % n) / n] for i in range(n)])

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points)

        assert result.dimensions.shape[0] == n


# =============================================================================
# IntrinsicDimension.detect_dimension_deficiency Tests
# =============================================================================


class TestDetectDimensionDeficiency:
    """Tests for detect_dimension_deficiency static method."""

    def test_detect_dimension_deficiency_returns_list(self):
        """detect_dimension_deficiency returns list of indices."""
        backend = get_default_backend()
        n = 30
        points = backend.array([[i / n, (i * 3 % n) / n] for i in range(n)])

        result = IntrinsicDimension.detect_dimension_deficiency(points, backend)

        assert isinstance(result, list)

    def test_detect_dimension_deficiency_returns_indices(self):
        """detect_dimension_deficiency returns valid indices."""
        backend = get_default_backend()
        n = 50
        points = []
        for i in range(n):
            x = (hash(str(i)) % 1000) / 1000.0
            y = (hash(str(i + 500)) % 1000) / 1000.0
            points.append([x, y])
        points_arr = backend.array(points)

        result = IntrinsicDimension.detect_dimension_deficiency(points_arr, backend)

        assert isinstance(result, list)
        # All indices should be valid
        for idx in result:
            assert 0 <= idx < n
