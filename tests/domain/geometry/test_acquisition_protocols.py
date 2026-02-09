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

from modelcypher.core.domain.geometry.acquisition_protocols import (
    AcquisitionResult,
    AcquisitionScore,
    empty_acquisition_result,
    uniform_acquisition_result,
)


def test_acquisition_score_to_dict() -> None:
    score = AcquisitionScore(
        probe_idx=3,
        score=0.75,
        coreset_contribution=0.5,
        coverage_contribution=0.2,
        density_contribution=0.05,
    )

    assert score.to_dict() == {
        "probe_idx": 3,
        "score": 0.75,
        "coreset_contribution": 0.5,
        "coverage_contribution": 0.2,
        "density_contribution": 0.05,
    }


def test_acquisition_result_helpers_and_serialization() -> None:
    s0 = AcquisitionScore(
        probe_idx=1,
        score=0.9,
        coreset_contribution=0.6,
        coverage_contribution=0.2,
        density_contribution=0.1,
    )
    s1 = AcquisitionScore(
        probe_idx=0,
        score=0.7,
        coreset_contribution=0.4,
        coverage_contribution=0.2,
        density_contribution=0.1,
    )
    result = AcquisitionResult(
        scores=[s0, s1],
        coverage_radius=2.0,
        mean_local_id=1.5,
        sparse_fraction=0.25,
    )

    assert result.top_score == s0
    assert result.top_indices == [1, 0]
    assert result.select_top_k(1) == [s0]
    assert result.to_dict()["scores"][0]["probe_idx"] == 1
    assert result.to_dict()["coverage_radius"] == 2.0


def test_empty_acquisition_result_defaults() -> None:
    result = empty_acquisition_result()

    assert result.scores == []
    assert result.coverage_radius == 0.0
    assert result.mean_local_id == 0.0
    assert result.sparse_fraction == 0.0
    assert result.top_score is None
    assert result.top_indices == []


def test_uniform_acquisition_result_has_equal_scores() -> None:
    result = uniform_acquisition_result(4)

    assert len(result.scores) == 4
    assert all(s.score == 1.0 for s in result.scores)
    assert all(s.coreset_contribution == 1.0 for s in result.scores)
    assert all(s.coverage_contribution == 0.0 for s in result.scores)
    assert all(s.density_contribution == 0.0 for s in result.scores)
    assert math.isinf(result.coverage_radius)
    assert result.mean_local_id == 0.0
    assert result.sparse_fraction == 1.0
    assert result.top_indices == [0, 1, 2, 3]

