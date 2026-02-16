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

"""Tests for GeodesicTrajectoryService."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from modelcypher.core.use_cases.geodesic_trajectory_service import (
    GeodesicTrajectoryResult,
    GeodesicTrajectoryService,
)


@dataclass(frozen=True)
class _FakeTrajectory:
    """Minimal TrajectoryActivations-compatible object."""

    positions: dict[int, Any]
    velocities: dict[int, Any]
    intermediate_positions: dict[int, Any]
    embedding_positions: Any
    q_positions: dict[int, Any]
    k_positions: dict[int, Any]
    v_positions: dict[int, Any]
    gate_positions: dict[int, Any]
    text_lengths: list[int]
    total_tokens: int
    n_texts: int


class _MockActivationProvider:
    """Mock that returns a pre-built trajectory."""

    def __init__(self, trajectory: _FakeTrajectory):
        self._trajectory = trajectory

    def collect_trajectory_batch(
        self, model: Any, tokenizer: Any, texts: list[str]
    ) -> _FakeTrajectory:
        return self._trajectory


def _make_straight_trajectory(backend, n_tokens: int = 10, dim: int = 8):
    """Points on a straight line in R^dim: t * e_0 for t in 0..n_tokens-1."""
    rows = []
    for t in range(n_tokens):
        row = [float(t)] + [0.0] * (dim - 1)
        rows.append(row)
    positions = backend.array(rows)
    backend.eval(positions)
    return positions


def _make_curved_trajectory(backend, n_tokens: int = 10, dim: int = 8):
    """Points on a quarter-circle in the first two dims of R^dim."""
    rows = []
    for t in range(n_tokens):
        angle = (math.pi / 2) * t / (n_tokens - 1)
        row = [5.0 * math.cos(angle), 5.0 * math.sin(angle)] + [0.0] * (dim - 2)
        rows.append(row)
    positions = backend.array(rows)
    backend.eval(positions)
    return positions


def _build_service(backend, positions):
    """Build GeodesicTrajectoryService with mock trajectory."""
    n_tokens = len(positions.tolist())
    empty = backend.array([[0.0]])
    backend.eval(empty)

    trajectory = _FakeTrajectory(
        positions={0: positions},
        velocities={},
        intermediate_positions={},
        embedding_positions=empty,
        q_positions={},
        k_positions={},
        v_positions={},
        gate_positions={},
        text_lengths=[n_tokens],
        total_tokens=n_tokens,
        n_texts=1,
    )
    provider = _MockActivationProvider(trajectory)
    return GeodesicTrajectoryService(backend=backend, activation_provider=provider)


def test_straight_trajectory_low_deviation(any_backend):
    """Straight-line trajectory should have near-zero deviation."""
    b = any_backend
    positions = _make_straight_trajectory(b, n_tokens=10, dim=8)
    service = _build_service(b, positions)

    result = service.measure(model=None, tokenizer=None, text="dummy")

    assert isinstance(result, GeodesicTrajectoryResult)
    assert result.token_count == 10
    assert result.layer_analyzed == 0
    assert len(result.step_deviations) == 9
    # On a straight line, geodesic == euclidean, so deviation ~ 0
    assert result.mean_deviation < 0.1
    assert result.path_length_ratio < 1.1


def test_curved_trajectory_has_deviation(any_backend):
    """Curved (quarter-circle) trajectory should show path_length_ratio > 1."""
    b = any_backend
    positions = _make_curved_trajectory(b, n_tokens=15, dim=8)
    service = _build_service(b, positions)

    result = service.measure(model=None, tokenizer=None, text="dummy")

    assert result.token_count == 15
    # Quarter circle: arc length = pi*r/2 ~ 7.85, chord = sqrt(50) ~ 7.07
    # path_length_ratio should be > 1.0
    assert result.path_length_ratio >= 1.0


def test_to_dict_has_expected_keys(any_backend):
    """to_dict() returns all expected fields."""
    b = any_backend
    positions = _make_straight_trajectory(b, n_tokens=5, dim=4)
    service = _build_service(b, positions)

    result = service.measure(model=None, tokenizer=None, text="dummy")
    d = result.to_dict()

    expected_keys = {
        "token_count",
        "layer_analyzed",
        "step_deviations",
        "mean_deviation",
        "max_deviation",
        "intrinsic_dimension",
        "path_length_ratio",
    }
    assert set(d.keys()) == expected_keys


def test_intrinsic_dimension_positive(any_backend):
    """Intrinsic dimension should be a positive number."""
    b = any_backend
    positions = _make_curved_trajectory(b, n_tokens=20, dim=8)
    service = _build_service(b, positions)

    result = service.measure(model=None, tokenizer=None, text="dummy")
    assert result.intrinsic_dimension > 0


def test_too_few_tokens_raises(any_backend):
    """Fewer than 3 tokens should raise ValueError."""
    b = any_backend
    positions = b.array([[1.0, 0.0], [2.0, 0.0]])
    b.eval(positions)
    service = _build_service(b, positions)

    with pytest.raises(ValueError, match="Need >= 3 tokens"):
        service.measure(model=None, tokenizer=None, text="dummy")


def test_invalid_layer_raises(any_backend):
    """Requesting a non-existent layer should raise ValueError."""
    b = any_backend
    positions = _make_straight_trajectory(b, n_tokens=5, dim=4)
    service = _build_service(b, positions)

    with pytest.raises(ValueError, match="Layer 99 not in trajectory"):
        service.measure(model=None, tokenizer=None, text="dummy", target_layer=99)
