# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from modelcypher.core.domain.geometry.rank_selection import (
    select_full_rank_indices,
    select_shared_full_rank_indices,
)


def test_select_full_rank_indices_handles_edge_inputs(any_backend) -> None:
    b = any_backend

    assert select_full_rank_indices(b.zeros((0, 3)), max_count=3, backend=b) == []
    assert select_full_rank_indices(b.array([[1.0, 2.0]]), max_count=0, backend=b) == []
    assert select_full_rank_indices(
        b.array([[1.0, 0.0], [0.0, 1.0]]),
        max_count=4,
        backend=b,
    ) == [0, 1]


def test_select_full_rank_indices_respects_max_count_and_uniqueness(any_backend) -> None:
    b = any_backend
    points = b.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 2.0],  # Dependent on [1, 1]
        ]
    )

    selected = select_full_rank_indices(points, max_count=2, backend=b, center=False)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected).issubset({0, 1, 2, 3})


def test_select_shared_full_rank_indices_requires_both_spaces(any_backend) -> None:
    b = any_backend
    source = b.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],  # Dependent in source
        ]
    )
    target = b.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],  # Also dependent in target
        ]
    )

    selected = select_shared_full_rank_indices(
        source_points=source,
        target_points=target,
        max_count=2,
        backend=b,
        center=False,
    )

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected).issubset({0, 1, 2, 3})
