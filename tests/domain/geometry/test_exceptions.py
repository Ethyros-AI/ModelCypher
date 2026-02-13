# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Comprehensive tests for geometry domain exception classes."""

from __future__ import annotations

import pytest

import modelcypher.core.domain.geometry.exceptions as mod


# =============================================================================
# EstimatorError
# =============================================================================


class TestEstimatorError:
    """Tests for EstimatorError construction, attributes, and factory methods."""

    # --- Direct construction ---

    def test_construction_with_all_args(self) -> None:
        err = mod.EstimatorError(
            kind="testKind",
            message="Something went wrong",
            count=42,
        )
        assert err.kind == "testKind"
        assert str(err) == "Something went wrong"
        assert err.count == 42

    def test_construction_without_count(self) -> None:
        err = mod.EstimatorError(kind="simple", message="oops")
        assert err.kind == "simple"
        assert str(err) == "oops"
        assert err.count is None

    def test_inherits_from_exception(self) -> None:
        err = mod.EstimatorError(kind="test", message="msg")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(mod.EstimatorError) as exc_info:
            raise mod.EstimatorError(kind="raised", message="catch me")
        assert exc_info.value.kind == "raised"
        assert str(exc_info.value) == "catch me"

    def test_can_be_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise mod.EstimatorError(kind="base", message="generic catch")

    # --- Factory: insufficient_samples ---

    def test_insufficient_samples(self) -> None:
        err = mod.EstimatorError.insufficient_samples(5)
        assert err.kind == "insufficientSamples"
        assert err.count == 5
        assert "5" in str(err)
        assert "at least 3" in str(err)

    def test_insufficient_samples_count_zero(self) -> None:
        err = mod.EstimatorError.insufficient_samples(0)
        assert err.kind == "insufficientSamples"
        assert err.count == 0
        assert "0" in str(err)

    def test_insufficient_samples_count_one(self) -> None:
        err = mod.EstimatorError.insufficient_samples(1)
        assert err.count == 1

    # --- Factory: invalid_point_dimension ---

    def test_invalid_point_dimension(self) -> None:
        err = mod.EstimatorError.invalid_point_dimension(3, 4)
        assert err.kind == "invalidPointDimension"
        assert "expected 3" in str(err)
        assert "found 4" in str(err)
        assert err.count is None

    def test_invalid_point_dimension_large_values(self) -> None:
        err = mod.EstimatorError.invalid_point_dimension(128, 256)
        assert "expected 128" in str(err)
        assert "found 256" in str(err)

    # --- Factory: non_finite_point_value ---

    def test_non_finite_point_value(self) -> None:
        err = mod.EstimatorError.non_finite_point_value()
        assert err.kind == "nonFinitePointValue"
        assert "non-finite" in str(err).lower() or "NaN" in str(err) or "Inf" in str(err)
        assert err.count is None

    # --- Factory: nearest_neighbor_degenerate ---

    def test_nearest_neighbor_degenerate(self) -> None:
        err = mod.EstimatorError.nearest_neighbor_degenerate()
        assert err.kind == "nearestNeighborDegenerate"
        assert "degenerate" in str(err).lower()
        assert err.count is None

    # --- Factory: regression_degenerate ---

    def test_regression_degenerate(self) -> None:
        err = mod.EstimatorError.regression_degenerate()
        assert err.kind == "regressionDegenerate"
        assert "degenerate" in str(err).lower()
        assert err.count is None

    # --- All factory methods produce EstimatorError instances ---

    @pytest.mark.parametrize(
        "factory_call",
        [
            lambda: mod.EstimatorError.insufficient_samples(2),
            lambda: mod.EstimatorError.invalid_point_dimension(3, 4),
            lambda: mod.EstimatorError.non_finite_point_value(),
            lambda: mod.EstimatorError.nearest_neighbor_degenerate(),
            lambda: mod.EstimatorError.regression_degenerate(),
        ],
        ids=[
            "insufficient_samples",
            "invalid_point_dimension",
            "non_finite_point_value",
            "nearest_neighbor_degenerate",
            "regression_degenerate",
        ],
    )
    def test_factory_produces_estimator_error_instance(self, factory_call) -> None:
        err = factory_call()
        assert isinstance(err, mod.EstimatorError)
        assert isinstance(err, Exception)

    @pytest.mark.parametrize(
        "factory_call",
        [
            lambda: mod.EstimatorError.insufficient_samples(2),
            lambda: mod.EstimatorError.invalid_point_dimension(3, 4),
            lambda: mod.EstimatorError.non_finite_point_value(),
            lambda: mod.EstimatorError.nearest_neighbor_degenerate(),
            lambda: mod.EstimatorError.regression_degenerate(),
        ],
        ids=[
            "insufficient_samples",
            "invalid_point_dimension",
            "non_finite_point_value",
            "nearest_neighbor_degenerate",
            "regression_degenerate",
        ],
    )
    def test_factory_errors_have_non_empty_message(self, factory_call) -> None:
        err = factory_call()
        assert len(str(err)) > 0

    @pytest.mark.parametrize(
        "factory_call",
        [
            lambda: mod.EstimatorError.insufficient_samples(2),
            lambda: mod.EstimatorError.invalid_point_dimension(3, 4),
            lambda: mod.EstimatorError.non_finite_point_value(),
            lambda: mod.EstimatorError.nearest_neighbor_degenerate(),
            lambda: mod.EstimatorError.regression_degenerate(),
        ],
        ids=[
            "insufficient_samples",
            "invalid_point_dimension",
            "non_finite_point_value",
            "nearest_neighbor_degenerate",
            "regression_degenerate",
        ],
    )
    def test_factory_errors_are_raisable(self, factory_call) -> None:
        with pytest.raises(mod.EstimatorError):
            raise factory_call()

    # --- Unique kinds for all factories ---

    def test_all_factory_kinds_are_distinct(self) -> None:
        kinds = {
            mod.EstimatorError.insufficient_samples(1).kind,
            mod.EstimatorError.invalid_point_dimension(1, 2).kind,
            mod.EstimatorError.non_finite_point_value().kind,
            mod.EstimatorError.nearest_neighbor_degenerate().kind,
            mod.EstimatorError.regression_degenerate().kind,
        }
        assert len(kinds) == 5


# =============================================================================
# ProjectionError
# =============================================================================


class TestProjectionError:
    def test_construction(self) -> None:
        err = mod.ProjectionError("projection failed")
        assert str(err) == "projection failed"

    def test_inherits_from_exception(self) -> None:
        err = mod.ProjectionError("msg")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(mod.ProjectionError) as exc_info:
            raise mod.ProjectionError("bad projection")
        assert "bad projection" in str(exc_info.value)

    def test_can_be_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise mod.ProjectionError("generic catch")

    def test_empty_message(self) -> None:
        err = mod.ProjectionError()
        assert isinstance(err, mod.ProjectionError)

    def test_is_not_estimator_error(self) -> None:
        err = mod.ProjectionError("msg")
        assert not isinstance(err, mod.EstimatorError)
