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

from modelcypher.core.domain.geometry.variance_concentration import (
    VarianceConcentrationResult,
    compute_variance_concentration,
    identify_bottleneck_layers,
)


def test_compute_variance_concentration_rank1_matrix(any_backend) -> None:
    b = any_backend
    activations = b.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    result = compute_variance_concentration(activations, backend=b)

    assert result.var_top1 > 0.99
    assert result.effective_rank == pytest.approx(1.0, rel=1e-2, abs=1e-2)
    assert result.n_singular_values == 3
    assert result.total_variance > 0.0
    assert result.var_top_k[1] == pytest.approx(result.var_top1)
    assert result.var_top_k[10] == pytest.approx(1.0)


def test_compute_variance_concentration_zero_matrix(any_backend) -> None:
    b = any_backend
    activations = b.zeros((4, 3))

    result = compute_variance_concentration(activations, backend=b, top_k=[1, 2, 4])

    # Regularization epsilon creates a uniform tiny spectrum.
    assert result.var_top1 == pytest.approx(1.0 / 3.0)
    assert result.effective_rank == pytest.approx(3.0)
    assert result.total_variance > 0.0
    assert result.var_top_k[1] == pytest.approx(1.0 / 3.0)
    assert result.var_top_k[2] == pytest.approx(2.0 / 3.0)
    assert result.var_top_k[4] == pytest.approx(1.0)


def test_identify_bottleneck_layers_by_variance_and_rank() -> None:
    layer_metrics = {
        1: VarianceConcentrationResult(
            var_top1=0.95,
            var_top_k={1: 0.95},
            effective_rank=2.0,
            n_singular_values=8,
            total_variance=1.0,
        ),
        2: VarianceConcentrationResult(
            var_top1=0.30,
            var_top_k={1: 0.30},
            effective_rank=1.5,
            n_singular_values=8,
            total_variance=1.0,
        ),
        3: VarianceConcentrationResult(
            var_top1=0.20,
            var_top_k={1: 0.20},
            effective_rank=8.0,
            n_singular_values=8,
            total_variance=1.0,
        ),
    }

    detected_default = identify_bottleneck_layers(layer_metrics)
    detected_custom = identify_bottleneck_layers(
        layer_metrics,
        var_threshold=0.90,
        effective_rank_threshold=1.7,
    )

    assert detected_default[0] == 1
    assert 2 in detected_default
    assert 3 not in detected_default

    assert detected_custom == [1, 2]
