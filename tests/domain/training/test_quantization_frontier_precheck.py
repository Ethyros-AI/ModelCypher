# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.quantization_frontier_precheck import (
    make_quantization_frontier_precheck_payload_v1,
    run_quantization_frontier_precheck_v1,
)


def test_quantization_frontier_payload_helper_builds_canonical_invalid_schema() -> None:
    payload = make_quantization_frontier_precheck_payload_v1(
        n_probes=3,
        raw_weyl={"n_crossing": 1},
        failure_modes=["activation_collection_failed"],
    )

    assert payload["operator"] == "quantization_frontier_precheck_v1"
    assert payload["valid"] is False
    assert payload["failure_modes"] == ["activation_collection_failed"]
    assert payload["n_probes"] == 3
    assert payload["subspace_source"] == "hidden_probe_output"
    assert payload["n_layers"] == 0
    assert payload["n_overlapping_layers"] == 0
    assert payload["per_layer_cka"] == {}
    assert payload["per_layer_hidden_probe_rho_out"] == {}
    assert payload["raw_weyl"] == {"n_crossing": 1}


def test_quantization_frontier_precheck_valid_nominal_case() -> None:
    backend = get_default_backend()
    fp_acts = {
        0: backend.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
            dtype="float32",
        ),
    }
    q_acts = {
        0: backend.array(
            [[1.1, 0.0], [-1.1, 0.0], [0.0, 0.0]],
            dtype="float32",
        ),
    }
    raw_weyl = {"max_error_over_gap_half": 42.0}

    payload = run_quantization_frontier_precheck_v1(
        fp_activations=fp_acts,
        quantized_activations=q_acts,
        n_probes=3,
        backend=backend,
        raw_weyl=raw_weyl,
    )

    assert payload["operator"] == "quantization_frontier_precheck_v1"
    assert payload["valid"] is True
    assert payload["failure_modes"] == []
    assert payload["subspace_source"] == "hidden_probe_output"
    assert payload["n_layers"] == 1
    assert payload["raw_weyl"] == raw_weyl
    assert payload["min_cka"] == payload["mean_cka"]
    assert payload["per_layer_cka"][0] >= 0.999
    assert payload["per_layer_hidden_probe_d_eff"][0] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert payload["per_layer_hidden_probe_k_eff"][0] == 1
    assert payload["per_layer_hidden_probe_gap_eff"][0] == pytest.approx(math.sqrt(2.0))
    assert payload["per_layer_hidden_probe_rho_out"][0] == pytest.approx(0.1)


def test_quantization_frontier_precheck_flags_insufficient_probes() -> None:
    backend = get_default_backend()

    payload = run_quantization_frontier_precheck_v1(
        fp_activations={},
        quantized_activations={},
        n_probes=1,
        backend=backend,
        raw_weyl={"n_layers": 0},
    )

    assert payload["valid"] is False
    assert payload["failure_modes"] == ["insufficient_probes"]
    assert payload["raw_weyl"] == {"n_layers": 0}


def test_quantization_frontier_precheck_flags_no_overlapping_layers() -> None:
    backend = get_default_backend()
    fp_acts = {0: backend.array([[1.0], [0.0], [-1.0]], dtype="float32")}
    q_acts = {1: backend.array([[1.0], [0.0], [-1.0]], dtype="float32")}

    payload = run_quantization_frontier_precheck_v1(
        fp_activations=fp_acts,
        quantized_activations=q_acts,
        n_probes=3,
        backend=backend,
    )

    assert payload["valid"] is False
    assert payload["failure_modes"] == ["no_overlapping_layers"]


def test_quantization_frontier_precheck_flags_degenerate_centered_gram() -> None:
    backend = get_default_backend()
    fp_acts = {0: backend.array([[1.0], [1.0], [1.0]], dtype="float32")}
    q_acts = {0: backend.array([[1.0], [1.0], [1.0]], dtype="float32")}

    payload = run_quantization_frontier_precheck_v1(
        fp_activations=fp_acts,
        quantized_activations=q_acts,
        n_probes=3,
        backend=backend,
    )

    assert payload["valid"] is False
    assert payload["failure_modes"] == ["degenerate_centered_gram"]


def test_quantization_frontier_precheck_flags_nonfinite_metric() -> None:
    backend = get_default_backend()
    fp_acts = {0: backend.array([[1.0], [0.0], [-1.0]], dtype="float32")}
    q_acts = {0: backend.array([[1.0], [float("nan")], [-1.0]], dtype="float32")}

    payload = run_quantization_frontier_precheck_v1(
        fp_activations=fp_acts,
        quantized_activations=q_acts,
        n_probes=3,
        backend=backend,
    )

    assert payload["valid"] is False
    assert payload["failure_modes"] == ["nonfinite_metric"]
