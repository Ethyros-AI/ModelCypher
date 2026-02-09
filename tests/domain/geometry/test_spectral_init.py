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

from modelcypher.core.domain.geometry.spectral_init import (
    _power_iteration_spectral_norm,
    spectral_normalized_init,
    spectral_normalized_lora_init,
)


def test_power_iteration_spectral_norm_for_known_matrix(any_backend) -> None:
    b = any_backend
    matrix = b.array([[3.0, 0.0], [0.0, 1.0]])

    spectral = _power_iteration_spectral_norm(matrix, b, n_iters=20)

    assert spectral == pytest.approx(3.0, rel=1e-3, abs=1e-3)


def test_power_iteration_spectral_norm_zero_matrix(any_backend) -> None:
    b = any_backend

    spectral = _power_iteration_spectral_norm(b.zeros((3, 3)), b, n_iters=10)

    assert spectral == 0.0


def test_spectral_normalized_init_targets_requested_norm(any_backend) -> None:
    b = any_backend
    target = 2.5

    weights = spectral_normalized_init(
        shape=(5, 4),
        backend=b,
        target_spectral=target,
        n_power_iters=20,
    )
    measured = _power_iteration_spectral_norm(weights, b, n_iters=30)

    assert tuple(weights.shape) == (5, 4)
    assert measured == pytest.approx(target, rel=8e-2, abs=8e-2)


def test_spectral_normalized_lora_init_shapes_and_budget(any_backend) -> None:
    b = any_backend
    sigma_k = 1.44

    A, B = spectral_normalized_lora_init(
        in_features=3,
        out_features=2,
        rank=1,
        sigma_k=sigma_k,
        backend=b,
    )

    assert tuple(A.shape) == (1, 3)
    assert tuple(B.shape) == (2, 1)

    product = b.matmul(B, A)
    measured = _power_iteration_spectral_norm(product, b, n_iters=30)
    assert measured == pytest.approx(sigma_k, rel=2e-1, abs=2e-1)

