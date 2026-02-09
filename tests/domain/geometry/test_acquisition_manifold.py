# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from math import pi
from types import SimpleNamespace

import pytest

from modelcypher.core.domain.geometry.acquisition_manifold import (
    DirectionalGapScore,
    ManifoldCoverageAcquisition,
    create_manifold_coverage_acquisition,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import LocalDimensionMap


class _StubGeometry:
    def __init__(self, backend, distance_matrix) -> None:
        self._backend = backend
        self._distance_matrix = backend.array(distance_matrix)

    def geodesic_distances(self, _points, k_neighbors=None, refine_iterations=1):
        return SimpleNamespace(distances=self._distance_matrix)


def test_manifold_coverage_score_edge_cases(any_backend) -> None:
    b = any_backend
    acq = ManifoldCoverageAcquisition(backend=b)

    empty = acq.score(b.zeros((0, 2)), b.zeros((3, 2)))
    assert empty.scores == []

    no_corpus = acq.score(b.zeros((2, 2)), b.zeros((0, 2)))
    assert len(no_corpus.scores) == 2
    assert all(s.score == 1.0 for s in no_corpus.scores)

    tiny_corpus = acq.score(b.zeros((2, 2)), b.zeros((2, 2)))
    assert len(tiny_corpus.scores) == 2
    assert all(s.score == 1.0 for s in tiny_corpus.scores)


def test_manifold_coverage_score_with_stubbed_geometry_and_dimensions(any_backend) -> None:
    b = any_backend
    acq = ManifoldCoverageAcquisition(backend=b)

    distance_matrix = [
        [0.0, 1.0, 1.5, 3.0, 2.0],
        [1.0, 0.0, 1.0, 0.1, 3.0],
        [1.5, 1.0, 0.0, 2.0, 0.2],
        [3.0, 0.1, 2.0, 0.0, 1.0],
        [2.0, 3.0, 0.2, 1.0, 0.0],
    ]
    acq._geometry = _StubGeometry(b, distance_matrix)
    acq._id_estimator = SimpleNamespace(
        local_dimension_map=lambda _corpus: LocalDimensionMap(
            dimensions=b.array([2.0, 6.0, 2.0]),
            modal_dimension=2.0,
            mean_dimension=10.0 / 3.0,
            std_dimension=1.0,
            deficient_indices=[],
            k_neighbors=2,
        )
    )
    acq._compute_directional_gaps = lambda _corpus: {
        1: DirectionalGapScore(
            point_idx=1,
            max_gap_angle=pi / 2.0,
            sparse_direction=(1.0, 0.0),
            normalized_gap=0.5,
        )
    }

    corpus = b.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    candidates = b.array([[2.0, 0.0], [0.0, 3.0]])

    result = acq.score(candidates, corpus)

    assert result.mean_local_id == pytest.approx(10.0 / 3.0)
    assert result.sparse_fraction == pytest.approx(1.0 / 3.0)
    assert result.top_indices == [0, 1]
    assert result.scores[0].score == pytest.approx(1.5)
    assert result.scores[0].coverage_contribution == pytest.approx(1.5)
    assert result.scores[0].density_contribution == pytest.approx(3.0)
    # Missing directional gap falls back to uniform score.
    assert result.scores[1].score == pytest.approx(1.0)


def test_manifold_get_sparse_directions_sorted(any_backend) -> None:
    b = any_backend
    acq = ManifoldCoverageAcquisition(backend=b)
    acq._compute_directional_gaps = lambda _corpus: {
        0: DirectionalGapScore(0, 0.5, (1.0, 0.0), 0.5 / pi),
        1: DirectionalGapScore(1, 1.2, (0.0, 1.0), 1.2 / pi),
        2: DirectionalGapScore(2, 0.7, (0.0, -1.0), 0.7 / pi),
    }

    sorted_gaps = acq.get_sparse_directions(b.zeros((3, 2)))
    assert [gap.point_idx for gap in sorted_gaps] == [1, 2, 0]


def test_create_manifold_coverage_acquisition(any_backend) -> None:
    acq = create_manifold_coverage_acquisition(
        backend=any_backend,
        k_neighbors=9,
        refine_iterations=2,
    )
    assert isinstance(acq, ManifoldCoverageAcquisition)
    assert acq._config.k_neighbors == 9
    assert acq._config.refine_iterations == 2

