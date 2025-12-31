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

"""Tests for geometry exceptions (EstimatorError, ProjectionError)."""

import pytest

from modelcypher.core.domain.geometry.exceptions import (
    EstimatorError,
    ProjectionError,
)


class TestEstimatorError:
    """Tests for EstimatorError exception."""

    def test_init_basic(self):
        err = EstimatorError("testKind", "Test message")
        assert err.kind == "testKind"
        assert str(err) == "Test message"
        assert err.count is None

    def test_init_with_count(self):
        err = EstimatorError("testKind", "Test message", count=5)
        assert err.kind == "testKind"
        assert str(err) == "Test message"
        assert err.count == 5

    def test_is_exception(self):
        err = EstimatorError("testKind", "Test message")
        assert isinstance(err, Exception)

    def test_can_be_raised(self):
        with pytest.raises(EstimatorError) as exc_info:
            raise EstimatorError("testKind", "Test message", count=10)
        assert exc_info.value.kind == "testKind"
        assert exc_info.value.count == 10


class TestEstimatorErrorFactoryMethods:
    """Tests for EstimatorError static factory methods."""

    def test_insufficient_samples(self):
        err = EstimatorError.insufficient_samples(2)
        assert err.kind == "insufficientSamples"
        assert err.count == 2
        assert "at least 3 samples" in str(err)
        assert "got 2" in str(err)

    def test_insufficient_samples_various_counts(self):
        for count in [0, 1, 2]:
            err = EstimatorError.insufficient_samples(count)
            assert err.count == count
            assert str(count) in str(err)

    def test_invalid_point_dimension(self):
        err = EstimatorError.invalid_point_dimension(expected=64, found=32)
        assert err.kind == "invalidPointDimension"
        assert err.count is None
        assert "expected 64" in str(err)
        assert "found 32" in str(err)

    def test_non_finite_point_value(self):
        err = EstimatorError.non_finite_point_value()
        assert err.kind == "nonFinitePointValue"
        assert err.count is None
        assert "NaN" in str(err) or "Inf" in str(err)

    def test_nearest_neighbor_degenerate(self):
        err = EstimatorError.nearest_neighbor_degenerate()
        assert err.kind == "nearestNeighborDegenerate"
        assert err.count is None
        assert "degenerate" in str(err).lower()

    def test_regression_degenerate(self):
        err = EstimatorError.regression_degenerate()
        assert err.kind == "regressionDegenerate"
        assert err.count is None
        assert "degenerate" in str(err).lower()


class TestEstimatorErrorKinds:
    """Tests for EstimatorError kind values."""

    def test_all_kinds_unique(self):
        kinds = [
            EstimatorError.insufficient_samples(1).kind,
            EstimatorError.invalid_point_dimension(1, 2).kind,
            EstimatorError.non_finite_point_value().kind,
            EstimatorError.nearest_neighbor_degenerate().kind,
            EstimatorError.regression_degenerate().kind,
        ]
        assert len(kinds) == len(set(kinds))

    def test_kinds_are_camel_case(self):
        kinds = [
            EstimatorError.insufficient_samples(1).kind,
            EstimatorError.invalid_point_dimension(1, 2).kind,
            EstimatorError.non_finite_point_value().kind,
            EstimatorError.nearest_neighbor_degenerate().kind,
            EstimatorError.regression_degenerate().kind,
        ]
        for kind in kinds:
            # Check starts with lowercase (camelCase)
            assert kind[0].islower()
            # Check no underscores
            assert "_" not in kind


class TestProjectionError:
    """Tests for ProjectionError exception."""

    def test_is_exception(self):
        err = ProjectionError("Test message")
        assert isinstance(err, Exception)

    def test_message(self):
        err = ProjectionError("Projection failed")
        assert str(err) == "Projection failed"

    def test_can_be_raised(self):
        with pytest.raises(ProjectionError) as exc_info:
            raise ProjectionError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_empty_message(self):
        err = ProjectionError()
        assert str(err) == ""

    def test_not_estimator_error(self):
        err = ProjectionError("Test")
        assert not isinstance(err, EstimatorError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and behavior."""

    def test_estimator_error_inheritance(self):
        err = EstimatorError("test", "message")
        assert isinstance(err, Exception)
        assert isinstance(err, BaseException)

    def test_projection_error_inheritance(self):
        err = ProjectionError("message")
        assert isinstance(err, Exception)
        assert isinstance(err, BaseException)

    def test_catch_as_exception(self):
        caught = False
        try:
            raise EstimatorError("test", "message")
        except Exception:
            caught = True
        assert caught

    def test_estimator_error_attributes_preserved_in_except(self):
        try:
            raise EstimatorError.insufficient_samples(2)
        except EstimatorError as e:
            assert e.kind == "insufficientSamples"
            assert e.count == 2
