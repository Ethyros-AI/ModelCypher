# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry import alignment


class _FakeIntrinsicDimension:
    def __init__(self, _backend, intrinsic_dim: float = 3.0) -> None:
        self._intrinsic_dim = intrinsic_dim

    def compute(self, _target):
        return type("_Result", (), {"intrinsic_dimension": self._intrinsic_dim})()


def test_invariant_alignment_records_underdetermined_stats(any_backend) -> None:
    b = any_backend

    source = b.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    target = b.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    stats: dict[str, float] = {}
    F = alignment.invariant_alignment(b, source, target, stats=stats)

    assert tuple(b.shape(F)) == (5, 3)
    assert stats["underdetermined"] == 1.0
    assert stats["n_samples"] == 3.0
    assert stats["d_source"] == 5.0
    assert stats["underdetermination_ratio"] == pytest.approx(5.0 / 3.0)


def test_invariant_alignment_without_underdetermined_branch(any_backend) -> None:
    b = any_backend

    source = b.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    target = b.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    stats: dict[str, float] = {}
    F = alignment.invariant_alignment(b, source, target, stats=stats)

    assert tuple(b.shape(F)) == (3, 2)
    assert "underdetermined" not in stats


def test_geodesic_invariant_alignment_stats_and_transform(monkeypatch, any_backend) -> None:
    b = any_backend

    source = b.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, -1.0, 1.5],
            [0.0, 1.0, -0.5],
            [1.5, 0.5, 0.5],
        ]
    )
    target = b.array(
        [
            [0.2, 1.0],
            [0.4, -0.5],
            [1.2, 0.0],
            [0.8, 0.7],
        ]
    )

    # Deterministic relative geometry pair with reflected target geometry.
    G_source = b.array(
        [
            [1.7640524, 0.4001572, 0.9787380, 2.2408931],
            [1.8675580, -0.9772779, 0.9500884, -0.1513572],
            [-0.1032189, 0.4105985, 0.1440436, 1.4542735],
            [0.7610377, 0.1216750, 0.4438632, 0.3336743],
        ]
    )
    reflection = b.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    G_target = b.matmul(G_source, reflection)
    b.eval(G_target)

    calls = {"n": 0}

    def _fake_geodesic_cosine_matrix(_arr, _backend):
        calls["n"] += 1
        return G_source if calls["n"] == 1 else G_target

    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.riemannian_utils.geodesic_cosine_matrix",
        _fake_geodesic_cosine_matrix,
    )

    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.intrinsic_dimension.IntrinsicDimension",
        lambda _backend: _FakeIntrinsicDimension(_backend, intrinsic_dim=3.0),
    )

    stats: dict[str, float] = {}
    F = alignment.geodesic_invariant_alignment(b, source, target, stats=stats)

    assert tuple(b.shape(F)) == (3, 2)
    assert calls["n"] == 2
    assert stats["geodesic_alignment"] == 1.0
    assert stats["n_samples"] == 4.0
    assert stats["d_source"] == 3.0
    assert stats["d_target"] == 2.0
    assert "geodesic_time_sec" in stats
    assert "relative_space_alignment_error" in stats
    assert stats["relative_space_alignment_error"] >= 0.0
