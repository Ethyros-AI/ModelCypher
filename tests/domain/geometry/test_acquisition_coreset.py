# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelcypher.core.domain.geometry.acquisition_coreset import (
    CoreSetAcquisition,
    create_coreset_acquisition,
)


class _StubGeometry:
    def __init__(self, backend, distance_matrix, fps_indices=None) -> None:
        self._backend = backend
        self._distance_matrix = backend.array(distance_matrix)
        self._fps_indices = fps_indices or []

    def geodesic_distances(self, _points, k_neighbors=None, refine_iterations=1):
        return SimpleNamespace(distances=self._distance_matrix)

    def farthest_point_sampling(self, _points, batch_size: int):
        return SimpleNamespace(selected_indices=self._fps_indices[:batch_size])


def _make_stub_distance_matrix() -> list[list[float]]:
    return [
        [0.0, 10.0, 1.0, 2.0, 0.5],
        [10.0, 0.0, 4.0, 3.0, 5.0],
        [1.0, 4.0, 0.0, 0.8, 1.2],
        [2.0, 3.0, 0.8, 0.0, 0.2],
        [0.5, 5.0, 1.2, 0.2, 0.0],
    ]


def test_coreset_score_edge_cases(any_backend) -> None:
    b = any_backend
    acq = CoreSetAcquisition(backend=b)

    empty = acq.score(b.zeros((0, 2)), b.zeros((2, 2)))
    assert empty.scores == []

    no_corpus = acq.score(b.zeros((3, 2)), b.zeros((0, 2)))
    assert len(no_corpus.scores) == 3
    assert all(score.score == 1.0 for score in no_corpus.scores)


def test_coreset_score_and_select_batch_with_stub_geometry(any_backend) -> None:
    b = any_backend
    acq = CoreSetAcquisition(backend=b)
    acq._geometry = _StubGeometry(b, _make_stub_distance_matrix(), fps_indices=[2, 0, 1])

    candidates = b.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    corpus = b.array([[1.0, 1.0], [2.0, 2.0]])

    result = acq.score(candidates, corpus)
    assert result.coverage_radius == pytest.approx(10.0)
    assert result.top_indices == [1, 0, 2]
    assert result.scores[0].score == pytest.approx(0.2)
    assert result.scores[1].score == pytest.approx(0.1)
    assert result.scores[2].score == pytest.approx(0.05)

    selected = acq.select_batch(candidates, corpus, batch_size=2)
    assert selected == [1, 0]


def test_coreset_select_batch_without_corpus_uses_fps(any_backend) -> None:
    b = any_backend
    acq = CoreSetAcquisition(backend=b)
    acq._geometry = _StubGeometry(b, _make_stub_distance_matrix(), fps_indices=[2, 0, 1])

    selected = acq.select_batch(
        b.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        b.zeros((0, 2)),
        batch_size=2,
    )
    assert selected == [2, 0]


def test_coreset_coverage_radius_singleton_is_infinite(any_backend) -> None:
    b = any_backend
    acq = CoreSetAcquisition(backend=b)
    single = b.array([[1.0, 2.0]])
    dist = b.array([[0.0]])

    radius = acq._compute_coverage_radius(single, dist)
    assert radius == float("inf")


def test_create_coreset_acquisition(any_backend) -> None:
    acq = create_coreset_acquisition(backend=any_backend, k_neighbors=7, refine_iterations=2)
    assert isinstance(acq, CoreSetAcquisition)
    assert acq._config.k_neighbors == 7
    assert acq._config.refine_iterations == 2

