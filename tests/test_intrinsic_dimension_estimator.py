from __future__ import annotations

import pytest

from modelcypher.core.domain.intrinsic_dimension_estimator import (
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
