"""Synthetic tests for the published contextual-curvature operator."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.contextual_curvature import (
    compute_contextual_curvature,
)


def test_straight_trajectory_has_zero_contextual_curvature(any_backend) -> None:
    positions = any_backend.array([[float(index), 0.0] for index in range(8)])
    profile = compute_contextual_curvature(
        positions,
        backend=any_backend,
        window_size=3,
    )

    values = any_backend.tolist(profile.contextual_curvature_radians)
    assert profile.token_positions == (4, 5, 6, 7)
    assert values == pytest.approx([0.0] * len(values), abs=any_backend.finfo().eps)


def test_contextual_curvature_is_rotation_and_scale_invariant(any_backend) -> None:
    positions = any_backend.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [0.0, 3.0],
            [-1.0, 2.0],
        ]
    )
    angle = math.pi / 3.0
    rotation = any_backend.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    transformed = (positions @ rotation) * math.sqrt(2.0)

    original = compute_contextual_curvature(
        positions,
        backend=any_backend,
        window_size=3,
    )
    rotated = compute_contextual_curvature(
        transformed,
        backend=any_backend,
        window_size=3,
    )

    assert any_backend.tolist(rotated.contextual_curvature_radians) == pytest.approx(
        any_backend.tolist(original.contextual_curvature_radians),
        rel=math.sqrt(any_backend.finfo().eps),
        abs=math.sqrt(any_backend.finfo().eps),
    )
