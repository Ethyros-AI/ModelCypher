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
    assert result.n_samples == 4
    assert result.hidden_dim == 3
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
    assert result.n_samples == 4
    assert result.hidden_dim == 3
    assert result.total_variance > 0.0
    assert result.var_top_k[1] == pytest.approx(1.0 / 3.0)
    assert result.var_top_k[2] == pytest.approx(2.0 / 3.0)
    assert result.var_top_k[4] == pytest.approx(1.0)


# --- Bottleneck detection (changepoint-based, Bai & Perron 1998) ---


def _make_result(var_top1: float) -> VarianceConcentrationResult:
    return VarianceConcentrationResult(
        var_top1=var_top1,
        var_top_k={1: var_top1},
        effective_rank=10.0,
        n_singular_values=64,
        n_samples=128,
        hidden_dim=64,
        total_variance=1.0,
    )


def test_identify_bottleneck_clear_bimodal() -> None:
    """Clear bimodal distribution -> changepoint finds bottleneck layers."""
    # 5 low-concentration layers + 5 high-concentration layers
    layer_metrics = {}
    for i in range(5):
        layer_metrics[i] = _make_result(0.05 + i * 0.02)  # 0.05..0.13
    for i in range(5, 10):
        layer_metrics[i] = _make_result(0.70 + (i - 5) * 0.02)  # 0.70..0.78

    bottlenecks, cp = identify_bottleneck_layers(layer_metrics, seed=42)

    assert cp is not None
    assert cp.rss_reduction > 0.5  # Strong changepoint
    assert len(bottlenecks) == 5  # Layers 5-9 are bottlenecks
    assert all(idx >= 5 for idx in bottlenecks)


def test_identify_bottleneck_uniform_spacing() -> None:
    """Uniformly spaced values -> no stable changepoint, no bottlenecks."""
    layer_metrics = {
        i: _make_result(0.1 + i * 0.05) for i in range(8)
    }

    bottlenecks, cp = identify_bottleneck_layers(layer_metrics, seed=42)

    # Linear data -> changepoint unstable or weak
    assert cp is not None
    assert bottlenecks == []


def test_identify_bottleneck_too_few_layers() -> None:
    """Fewer than 5 layers -> returns empty with None changepoint."""
    layer_metrics = {
        i: _make_result(0.1 * (i + 1)) for i in range(4)
    }

    bottlenecks, cp = identify_bottleneck_layers(layer_metrics, seed=42)

    assert bottlenecks == []
    assert cp is None


def test_identify_bottleneck_empty_raises() -> None:
    with pytest.raises(ValueError, match="No variance measurements available"):
        identify_bottleneck_layers({})
