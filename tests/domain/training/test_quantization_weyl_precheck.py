# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.quantization_weyl_precheck import (
    run_quantization_weyl_precheck,
)


def test_quantization_weyl_precheck_passes_for_non_crossing_error() -> None:
    backend = get_default_backend()

    fp_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[10.0, 0.0], [0.0, 1.0]],
            dtype="float32",
        ),
    }
    q_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[9.0, 0.0], [0.0, 1.0]],
            dtype="float32",
        ),
    }

    payload = run_quantization_weyl_precheck(
        fp_weights=fp_weights,
        quantized_weights=q_weights,
        backend=backend,
    )

    assert payload["n_layers"] == 1
    assert payload["n_crossing"] == 0
    assert payload["all_non_crossing"] is True
    layer = payload["per_layer"][0]
    assert layer["crossing"] is False
    assert layer["error_over_gap_half"] < 1.0


def test_quantization_weyl_precheck_detects_crossing_error() -> None:
    backend = get_default_backend()

    fp_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[10.0, 0.0], [0.0, 1.0]],
            dtype="float32",
        ),
    }
    q_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[4.0, 0.0], [0.0, 1.0]],
            dtype="float32",
        ),
    }

    payload = run_quantization_weyl_precheck(
        fp_weights=fp_weights,
        quantized_weights=q_weights,
        backend=backend,
    )

    assert payload["n_layers"] == 1
    assert payload["n_crossing"] == 1
    assert payload["all_non_crossing"] is False
    layer = payload["per_layer"][0]
    assert layer["crossing"] is True
    assert layer["error_over_gap_half"] >= 1.0


def test_quantization_weyl_precheck_flags_zero_gap_with_nonzero_error() -> None:
    backend = get_default_backend()

    # Repeated singular values at the structural boundary (gap_half == 0).
    fp_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        ),
    }
    q_weights = {
        "model.layers.0.self_attn.q_proj.weight": backend.array(
            [[2.1, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        ),
    }

    payload = run_quantization_weyl_precheck(
        fp_weights=fp_weights,
        quantized_weights=q_weights,
        backend=backend,
    )

    assert payload["n_layers"] == 1
    assert payload["n_crossing"] == 1
    assert payload["all_non_crossing"] is False
    layer = payload["per_layer"][0]
    assert layer["gap_half"] == 0.0
    assert layer["crossing"] is True
    assert math.isinf(layer["error_over_gap_half"])
