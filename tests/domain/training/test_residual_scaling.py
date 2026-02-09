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

from modelcypher.core.domain.training.residual_scaling import (
    ResidualScaleStats,
    ResidualScalingConfig,
    ResidualScalingState,
    compute_residual_scale,
    log_residual_stats,
    spectral_norm_power_iteration,
)


def test_spectral_norm_power_iteration_handles_shapes(any_backend) -> None:
    b = any_backend

    vec = b.array([3.0, 4.0])
    mat = b.array([[3.0, 0.0], [0.0, 4.0]])
    batch = b.array([[[3.0, 0.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]])
    empty = b.array([]).reshape((0, 2))

    assert spectral_norm_power_iteration(vec, b) == pytest.approx(5.0, rel=1e-3, abs=1e-3)
    assert spectral_norm_power_iteration(mat, b) == pytest.approx(4.0, rel=1e-2, abs=1e-2)
    assert spectral_norm_power_iteration(batch, b) == pytest.approx(4.0, rel=1e-2, abs=1e-2)
    assert spectral_norm_power_iteration(empty, b) == 0.0


def test_compute_residual_scale_clamps_to_precision_bounds() -> None:
    sqrt_eps = 1e-4

    low_alpha, low_stats = compute_residual_scale(
        input_spectral=1.0,
        residual_spectral=1e9,
        layer_idx=0,
        sqrt_eps=sqrt_eps,
    )
    high_alpha, high_stats = compute_residual_scale(
        input_spectral=1e9,
        residual_spectral=1.0,
        layer_idx=1,
        sqrt_eps=sqrt_eps,
    )
    default_alpha, default_stats = compute_residual_scale(
        input_spectral=1.0,
        residual_spectral=0.0,
        layer_idx=2,
        sqrt_eps=sqrt_eps,
    )

    assert low_alpha == pytest.approx(sqrt_eps)
    assert low_stats.min_alpha == sqrt_eps
    assert low_stats.max_alpha == pytest.approx(1.0 / sqrt_eps)

    assert high_alpha == pytest.approx(1.0 / sqrt_eps)
    assert high_stats.alpha == pytest.approx(1.0 / sqrt_eps)

    assert default_alpha == 1.0
    assert default_stats.alpha == 1.0
    assert default_stats.is_valid is True


def test_residual_scaling_state_summary_and_reset() -> None:
    state = ResidualScalingState()
    state.add_layer_stats(
        ResidualScaleStats(
            layer_idx=0,
            input_spectral=1.0,
            residual_spectral=2.0,
            alpha=0.5,
            min_alpha=0.1,
            max_alpha=10.0,
        )
    )
    state.add_layer_stats(
        ResidualScaleStats(
            layer_idx=1,
            input_spectral=1.0,
            residual_spectral=0.01,
            alpha=50.0,
            min_alpha=0.1,
            max_alpha=10.0,
        )
    )

    summary = state.get_alpha_summary()
    assert summary["mean"] == pytest.approx(25.25)
    assert summary["min"] == pytest.approx(0.5)
    assert summary["max"] == pytest.approx(50.0)
    assert summary["n_invalid"] == 1

    state.reset_step()
    assert state.step == 1
    assert state.layer_stats == []


def test_residual_scaling_config_round_trip_and_logging(caplog) -> None:
    config = ResidualScalingConfig(enabled=False)
    payload = config.to_dict()
    restored = ResidualScalingConfig.from_dict(payload)
    assert restored.enabled is False

    state = ResidualScalingState()
    state.add_layer_stats(
        ResidualScaleStats(
            layer_idx=0,
            input_spectral=1.0,
            residual_spectral=1.0,
            alpha=1.0,
            min_alpha=0.1,
            max_alpha=10.0,
        )
    )
    with caplog.at_level("INFO"):
        log_residual_stats(state, prefix="[test] ")
    assert "Residual scaling" in caplog.text

