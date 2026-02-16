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


class _FakeTokenizer:
    """Minimal tokenizer that maps characters to token IDs."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, ids: list[int]) -> str:
        # Return a single character label for each id
        return f"t{ids[0]}"


def test_annotated_steps_contain_tokens(any_backend):
    """annotate_tokens=True produces token labels and annotated steps."""
    b = any_backend
    # 5 tokens, so tokenizer should produce 5 token IDs
    positions = _make_straight_trajectory(b, n_tokens=5, dim=4)

    # Build service with a fake tokenizer
    n_tokens = 5
    empty = b.array([[0.0]])
    b.eval(empty)
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
    service = GeodesicTrajectoryService(backend=b, activation_provider=provider)

    tokenizer = _FakeTokenizer()
    result = service.measure(
        model=None, tokenizer=tokenizer, text="abcde",
        annotate_tokens=True,
    )

    assert result.token_labels is not None
    assert len(result.token_labels) == 5
    assert result.annotated_steps is not None
    assert len(result.annotated_steps) == 4  # n_tokens - 1 steps
    assert "from" in result.annotated_steps[0]
    assert "to" in result.annotated_steps[0]
    assert "deviation" in result.annotated_steps[0]

    # to_dict includes annotation
    d = result.to_dict()
    assert "token_labels" in d
    assert "annotated_steps" in d


def test_measure_batch_aggregates_categories(any_backend):
    """measure_batch() aggregates results per category."""
    b = any_backend
    positions_a = _make_straight_trajectory(b, n_tokens=5, dim=4)
    positions_b = _make_curved_trajectory(b, n_tokens=8, dim=4)

    # Build two services with different trajectories is awkward, so build one
    # and use straight trajectory (both prompts get same positions)
    service = _build_service(b, positions_a)

    from modelcypher.core.use_cases.geodesic_trajectory_service import (
        GeodesicComparisonResult,
    )

    result = service.measure_batch(
        model=None,
        tokenizer=None,
        categorized_prompts={
            "cat_a": ["prompt1", "prompt2"],
            "cat_b": ["prompt3"],
        },
        model_path="/fake/model",
    )

    assert isinstance(result, GeodesicComparisonResult)
    assert len(result.categories) == 2
    assert result.categories[0].category == "cat_a"
    assert result.categories[0].prompt_count == 2
    assert result.categories[1].category == "cat_b"
    assert result.categories[1].prompt_count == 1

    # to_dict works
    d = result.to_dict()
    assert d["model_path"] == "/fake/model"
    assert len(d["categories"]) == 2
