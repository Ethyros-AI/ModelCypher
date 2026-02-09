# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.acquisition_composite import (
    CompositeAcquisition,
    create_composite_acquisition,
)
from modelcypher.core.domain.geometry.acquisition_protocols import (
    AcquisitionResult,
    AcquisitionScore,
)


class _StubAcquisition:
    def __init__(self, result: AcquisitionResult) -> None:
        self._result = result

    def score(self, _candidates, _corpus, backend=None) -> AcquisitionResult:
        return self._result


def test_compute_weights_edge_cases(any_backend) -> None:
    acquisition = CompositeAcquisition(any_backend)

    dense = acquisition.compute_weights(coverage_radius=0.0, mean_local_id=2.0)
    flat = acquisition.compute_weights(coverage_radius=2.0, mean_local_id=0.0)
    mixed = acquisition.compute_weights(coverage_radius=2.0, mean_local_id=1.0)

    assert dense.coreset_weight == 0.0
    assert dense.coverage_weight == pytest.approx(0.5)
    assert dense.density_weight == pytest.approx(0.5)

    assert flat.coreset_weight == 1.0
    assert flat.coverage_weight == 0.0
    assert flat.density_weight == 0.0

    assert mixed.coreset_weight == pytest.approx(2.0 / 3.0)
    assert mixed.coverage_weight == pytest.approx(1.0 / 6.0)
    assert mixed.density_weight == pytest.approx(1.0 / 6.0)


def test_score_edge_cases(any_backend) -> None:
    b = any_backend
    acquisition = CompositeAcquisition(b)

    empty_candidates = b.zeros((0, 2))
    corpus = b.array([[1.0, 0.0], [0.0, 1.0]])
    result_empty_candidates = acquisition.score(empty_candidates, corpus)
    assert result_empty_candidates.scores == []

    candidates = b.array([[1.0, 0.0], [0.0, 1.0]])
    empty_corpus = b.zeros((0, 2))
    result_empty_corpus = acquisition.score(candidates, empty_corpus)
    assert len(result_empty_corpus.scores) == 2
    assert all(score.score == 1.0 for score in result_empty_corpus.scores)
    assert math.isinf(result_empty_corpus.coverage_radius)


def test_score_combines_components_and_handles_missing_scores(any_backend) -> None:
    b = any_backend
    acquisition = CompositeAcquisition(b)

    acquisition._coreset = _StubAcquisition(
        AcquisitionResult(
            scores=[
                AcquisitionScore(
                    probe_idx=0,
                    score=0.6,
                    coreset_contribution=0.5,
                    coverage_contribution=0.0,
                    density_contribution=0.0,
                ),
                AcquisitionScore(
                    probe_idx=1,
                    score=0.4,
                    coreset_contribution=0.4,
                    coverage_contribution=0.0,
                    density_contribution=0.0,
                ),
            ],
            coverage_radius=2.0,
            mean_local_id=0.0,
            sparse_fraction=0.0,
        )
    )
    acquisition._manifold = _StubAcquisition(
        AcquisitionResult(
            scores=[
                AcquisitionScore(
                    probe_idx=0,
                    score=0.0,
                    coreset_contribution=0.0,
                    coverage_contribution=0.3,
                    density_contribution=0.9,
                )
            ],
            coverage_radius=0.0,
            mean_local_id=1.0,
            sparse_fraction=0.25,
        )
    )

    result = acquisition.score(
        b.array([[1.0, 0.0], [0.0, 1.0]]),
        b.array([[0.5, 0.5], [0.2, 0.8]]),
    )

    assert result.coverage_radius == pytest.approx(2.0)
    assert result.mean_local_id == pytest.approx(1.0)
    assert result.sparse_fraction == pytest.approx(0.25)
    # Probe 1 has missing manifold score and should use uniform fallback score=1.0.
    assert result.scores[0].probe_idx == 1
    assert result.scores[0].score == pytest.approx(1.0)
    # Probe 0 composite score: 2/3*0.6 + 1/6*0.3 + 1/6*0.9 = 0.6
    assert result.scores[1].probe_idx == 0
    assert result.scores[1].score == pytest.approx(0.6)


def test_get_weights(any_backend) -> None:
    b = any_backend
    acquisition = CompositeAcquisition(b)

    empty = acquisition.get_weights(b.zeros((0, 2)))
    assert empty.coreset_weight == 1.0
    assert empty.coverage_weight == 0.0
    assert empty.density_weight == 0.0
    assert math.isinf(empty.coverage_radius)

    acquisition._coreset = _StubAcquisition(
        AcquisitionResult(
            scores=[],
            coverage_radius=4.0,
            mean_local_id=0.0,
            sparse_fraction=0.0,
        )
    )
    acquisition._manifold = _StubAcquisition(
        AcquisitionResult(
            scores=[],
            coverage_radius=0.0,
            mean_local_id=2.0,
            sparse_fraction=0.0,
        )
    )
    weights = acquisition.get_weights(b.array([[1.0, 0.0], [0.0, 1.0]]))

    assert weights.coreset_weight == pytest.approx(2.0 / 3.0)
    assert weights.coverage_weight == pytest.approx(1.0 / 6.0)
    assert weights.density_weight == pytest.approx(1.0 / 6.0)


def test_create_composite_acquisition(any_backend) -> None:
    acquisition = create_composite_acquisition(
        backend=any_backend,
        k_neighbors=9,
        refine_iterations=3,
    )

    assert isinstance(acquisition, CompositeAcquisition)
    assert acquisition._config.k_neighbors == 9
    assert acquisition._config.refine_iterations == 3

