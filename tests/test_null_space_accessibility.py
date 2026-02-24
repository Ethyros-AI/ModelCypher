# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.null_space_accessibility import (
    aggregate_layer_accessibility,
    analyze_layer_null_observability,
    analyze_module_null_accessibility,
)


def _activation_cloud():
    backend = get_default_backend()
    activations = backend.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    backend.eval(activations)
    return backend, activations


def test_analyze_layer_null_observability_reports_rank_and_coverage():
    backend, activations = _activation_cloud()
    result = analyze_layer_null_observability(activations, backend)

    assert result["n_samples"] == 4
    assert result["hidden_dim"] == 3
    assert result["coverage_ratio"] == pytest.approx(4.0 / 3.0)
    assert result["available_rank"] == 2
    assert result["used_rank"] == 1
    assert math.isfinite(float(result["condition_number"]))


def test_module_accessibility_delta_in_available_direction_is_preserved():
    backend, activations = _activation_cloud()
    delta_w = backend.array([[0.0, 1.0, 0.0]], dtype="float32")
    backend.eval(delta_w)

    result = analyze_module_null_accessibility(delta_w, activations, backend)
    assert result["dimension_compatible"] == 1
    assert float(result["behavioral_preserved_fraction"]) > 0.99
    assert int(result["principal_angle_count"]) >= 1
    assert math.isfinite(float(result["principal_angle_mean"]))


def test_module_accessibility_delta_in_used_direction_is_removed():
    backend, activations = _activation_cloud()
    delta_w = backend.array([[1.0, 0.0, 0.0]], dtype="float32")
    backend.eval(delta_w)

    result = analyze_module_null_accessibility(delta_w, activations, backend)
    assert result["dimension_compatible"] == 1
    assert float(result["behavioral_preserved_fraction"]) < 1e-3
    assert math.isfinite(float(result["grassmann_geodesic_distance"]))


def test_aggregate_layer_accessibility_groups_modules_by_layer():
    module_metrics = {
        "model.layers.2.self_attn.q_proj.weight": {
            "dimension_compatible": 1,
            "behavioral_delta_norm": 2.0,
            "behavioral_preserved_fraction": 0.4,
            "principal_angle_mean": 0.2,
            "available_rank": 100,
            "coverage_ratio": 1.5,
            "condition_number": 8.0,
        },
        "model.layers.2.self_attn.k_proj.weight": {
            "dimension_compatible": 1,
            "behavioral_delta_norm": 1.0,
            "behavioral_preserved_fraction": 0.1,
            "principal_angle_mean": 0.5,
            "available_rank": 90,
            "coverage_ratio": 1.5,
            "condition_number": 12.0,
        },
        "model.layers.3.self_attn.q_proj.weight": {
            "dimension_compatible": 1,
            "behavioral_delta_norm": 1.0,
            "behavioral_preserved_fraction": 0.8,
            "principal_angle_mean": 0.1,
            "available_rank": 80,
            "coverage_ratio": 1.2,
            "condition_number": 5.0,
        },
    }

    aggregated = aggregate_layer_accessibility(module_metrics)
    assert set(aggregated.keys()) == {2, 3}
    assert aggregated[2]["module_count"] == 2
    assert aggregated[3]["module_count"] == 1
    # Weighted by behavioral delta norm: (0.4*2 + 0.1*1) / 3 = 0.3
    assert float(aggregated[2]["behavioral_preserved_fraction"]) == pytest.approx(0.3)
