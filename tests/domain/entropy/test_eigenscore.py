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

from modelcypher.core.domain.entropy.eigenscore import EigenScoreCalculator


def test_eigenscore_sequence_requires_min_tokens(any_backend) -> None:
    b = any_backend
    calculator = EigenScoreCalculator(b)

    with pytest.raises(ValueError):
        calculator.compute_from_sequence(b.array([[1.0, 0.0], [0.0, 1.0]]), min_tokens=3)


def test_eigenscore_sequence_returns_reasonable_values(any_backend) -> None:
    b = any_backend
    calculator = EigenScoreCalculator(b)
    hidden_states = b.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )

    result = calculator.compute_from_sequence(hidden_states, min_tokens=3)

    assert 0.0 <= result.eigenscore <= 1.0
    assert 1.0 <= result.effective_rank <= 2.0
    assert result.eigenvalue_entropy >= 0.0
    assert result.condition_number >= 1.0
    assert 0.0 <= result.top_eigenvalue_ratio <= 1.0
    assert result.n_samples == 4
    assert result.n_dimensions == 2


def test_eigenscore_from_layers_handles_1d_2d_3d(any_backend) -> None:
    b = any_backend
    calculator = EigenScoreCalculator(b)
    layer_hidden = [
        b.array([1.0, 0.0]),
        b.array([[0.0, 1.0], [1.0, 0.0]]),          # last token -> [1, 0]
        b.array([[[0.0, 1.0], [0.0, 1.0]]]),        # batch0 last token -> [0, 1]
    ]

    result = calculator.compute_from_layers(layer_hidden)

    assert result.n_samples == 3
    assert result.n_dimensions == 2
    assert 0.0 <= result.eigenscore <= 1.0


def test_eigenscore_from_covariance_degenerate_case(any_backend) -> None:
    b = any_backend
    calculator = EigenScoreCalculator(b)
    covariance = b.zeros((3, 3))

    result = calculator.compute_from_covariance(covariance, n_samples=5)

    assert result.eigenscore == 0.0
    assert result.effective_rank == 1.0
    assert result.eigenvalue_entropy == 0.0
    assert result.condition_number == 1.0
    assert result.top_eigenvalue_ratio == 1.0
    assert result.n_samples == 5
    assert result.n_dimensions == 3


def test_streaming_eigenscore_update_compute_and_reset(any_backend) -> None:
    b = any_backend
    calculator = EigenScoreCalculator(b)
    streamer = calculator.create_streamer()

    streamer.update(b.array([1.0, 0.0]))
    assert streamer.n_samples == 1
    assert streamer.hidden_dim == 2
    with pytest.raises(ValueError):
        streamer.compute()

    streamer.update(b.array([[0.0, 1.0], [1.0, 1.0]]))  # 2D path -> last token
    streamer.update(b.array([[[1.0, 0.0], [0.0, 1.0]]]))  # 3D path -> batch0 last token
    assert streamer.n_samples == 3

    result = streamer.compute()
    assert result.n_samples == 3
    assert result.n_dimensions == 2
    assert result.condition_number >= 1.0

    streamer.reset()
    assert streamer.n_samples == 0
    assert streamer.hidden_dim == 0
    assert streamer.mean is None
    assert streamer.m2 is None

