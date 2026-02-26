# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for Weyl quantization validation script outputs and arguments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import weyl_quantization_validation as weyl_validation


def _sample_result(
    *,
    fp_model_id: str,
    q_model_id: str,
    n_layers: int,
    n_tail_dims_match: int,
    all_weyl_safe: bool,
    per_layer: list[dict[str, float | int | bool | str | list[int]]],
) -> dict[str, object]:
    return {
        "fp_model": f"/tmp/{fp_model_id}",
        "q_model": f"/tmp/{q_model_id}",
        "fp_model_id": fp_model_id,
        "q_model_id": q_model_id,
        "n_layers": n_layers,
        "n_weyl_safe": 0 if not all_weyl_safe else n_layers,
        "n_tail_dims_match": n_tail_dims_match,
        "all_weyl_safe": all_weyl_safe,
        "max_error_norm": max(float(layer["error_norm"]) for layer in per_layer),
        "max_error_over_gap_ratio": max(
            float(layer["error_over_gap_ratio"]) for layer in per_layer
        ),
        "verdict": "SAFE" if all_weyl_safe else "VIOLATION",
        "per_layer": per_layer,
    }


def test_parse_args_defaults_to_randomized_mode():
    args = weyl_validation._parse_args([])

    assert args.geometry_mode == "randomized"
    assert args.geometry_seed == weyl_validation.DEFAULT_GEOMETRY_SEED


def test_parse_args_accepts_exact_mode():
    args = weyl_validation._parse_args(["--geometry-mode", "exact"])

    assert args.geometry_mode == "exact"
    assert args.geometry_seed == weyl_validation.DEFAULT_GEOMETRY_SEED


def test_compute_aggregate_reduces_per_layer_metrics():
    result_a = _sample_result(
        fp_model_id="fp-a",
        q_model_id="q-a",
        n_layers=2,
        n_tail_dims_match=1,
        all_weyl_safe=False,
        per_layer=[
            {
                "layer_key": "a.0",
                "shape": [4, 4],
                "fp_sigma_max": 10.0,
                "q_sigma_max": 9.9,
                "sigma_max_diff": 0.1,
                "fp_sigma_k": 2.0,
                "q_sigma_k": 1.95,
                "sigma_k_diff": 0.05,
                "fp_tail_dims": 3,
                "q_tail_dims": 3,
                "tail_dims_match": True,
                "fp_spectral_gap": 0.02,
                "error_norm": 0.03,
                "weyl_threshold": 0.01,
                "error_over_gap_ratio": 3.0,
                "weyl_safe": False,
            },
            {
                "layer_key": "a.1",
                "shape": [4, 4],
                "fp_sigma_max": 8.0,
                "q_sigma_max": 7.92,
                "sigma_max_diff": 0.08,
                "fp_sigma_k": 1.5,
                "q_sigma_k": 1.47,
                "sigma_k_diff": 0.03,
                "fp_tail_dims": 2,
                "q_tail_dims": 1,
                "tail_dims_match": False,
                "fp_spectral_gap": 0.01,
                "error_norm": 0.05,
                "weyl_threshold": 0.005,
                "error_over_gap_ratio": 10.0,
                "weyl_safe": False,
            },
        ],
    )
    result_b = _sample_result(
        fp_model_id="fp-b",
        q_model_id="q-b",
        n_layers=1,
        n_tail_dims_match=1,
        all_weyl_safe=False,
        per_layer=[
            {
                "layer_key": "b.0",
                "shape": [4, 4],
                "fp_sigma_max": 12.0,
                "q_sigma_max": 11.994,
                "sigma_max_diff": 0.006,
                "fp_sigma_k": 3.0,
                "q_sigma_k": 2.997,
                "sigma_k_diff": 0.003,
                "fp_tail_dims": 3,
                "q_tail_dims": 3,
                "tail_dims_match": True,
                "fp_spectral_gap": 0.003,
                "error_norm": 0.06,
                "weyl_threshold": 0.0015,
                "error_over_gap_ratio": 40.0,
                "weyl_safe": False,
            },
        ],
    )

    aggregate = weyl_validation._compute_aggregate([result_a, result_b])

    assert aggregate["n_pairs"] == 2
    assert aggregate["n_layers_total"] == 3
    assert aggregate["n_tail_dims_match_total"] == 2
    assert aggregate["tail_match_pct"] == 100.0 * 2.0 / 3.0
    assert aggregate["max_error_norm"] == 0.06
    assert aggregate["max_error_over_gap_ratio"] == 40.0
    assert aggregate["max_sigma_max_rel_pct"] == 1.0
    assert aggregate["max_sigma_k_rel_pct"] == 2.5
    assert len(aggregate["per_model"]) == 2


def test_main_emits_analysis_config_and_aggregate(monkeypatch, tmp_path):
    fp_model = tmp_path / "fp-model"
    q_model = tmp_path / "q-model"
    fp_model.mkdir()
    q_model.mkdir()

    output_dir = tmp_path / "weyl-output"
    args = argparse.Namespace(
        pairs=[str(fp_model), str(q_model)],
        output_dir=output_dir,
        geometry_mode="exact",
        geometry_seed=123,
    )

    fake_result = _sample_result(
        fp_model_id="fp-model",
        q_model_id="q-model",
        n_layers=1,
        n_tail_dims_match=1,
        all_weyl_safe=False,
        per_layer=[
            {
                "layer_key": "model.layers.0.mlp.down_proj.weight",
                "shape": [8, 8],
                "fp_sigma_max": 10.0,
                "q_sigma_max": 9.999,
                "sigma_max_diff": 0.001,
                "fp_sigma_k": 2.0,
                "q_sigma_k": 1.999,
                "sigma_k_diff": 0.001,
                "fp_tail_dims": 4,
                "q_tail_dims": 4,
                "tail_dims_match": True,
                "fp_spectral_gap": 0.01,
                "error_norm": 0.02,
                "weyl_threshold": 0.005,
                "error_over_gap_ratio": 4.0,
                "weyl_safe": False,
            },
        ],
    )

    monkeypatch.setattr(weyl_validation, "_parse_args", lambda: args)
    monkeypatch.setattr(weyl_validation, "initialize_default_backend", lambda: object())
    monkeypatch.setattr(weyl_validation, "MLXTrainingAdapter", lambda backend: object())
    monkeypatch.setattr(
        weyl_validation,
        "_validate_pair",
        lambda *_args, **_kwargs: fake_result,
    )

    weyl_validation.main()

    payload_files = sorted(output_dir.glob("*/weyl_quantization_validation.json"))
    assert len(payload_files) == 1
    payload = json.loads(payload_files[0].read_text())

    assert payload["analysis_config"]["geometry_mode"] == "exact"
    assert payload["analysis_config"]["geometry_seed"] is None
    assert payload["analysis_config"]["error_norm_mode"] == "exact_svd"
    assert payload["aggregate"]["max_error_over_gap_ratio"] == 4.0
    assert payload["aggregate"]["tail_match_pct"] == 100.0


def test_spectral_norm_power_iter_is_deterministic():
    backend = get_default_backend()
    matrix = backend.array([[3.0, 0.0], [0.0, 1.0]], dtype="float32")
    backend.eval(matrix)

    sigma_1 = weyl_validation._spectral_norm_power_iter(matrix, backend, n_iters=20)
    sigma_2 = weyl_validation._spectral_norm_power_iter(matrix, backend, n_iters=20)
    assert sigma_1 == pytest.approx(sigma_2, rel=0.0, abs=1e-8)


def test_spectral_norm_exact_matches_svd():
    backend = get_default_backend()
    matrix = backend.array([[2.0, 0.0], [0.0, 0.5]], dtype="float32")
    backend.eval(matrix)

    sigma_exact = weyl_validation._spectral_norm_exact(matrix, backend)
    _, s, _ = backend.svd(backend.astype(matrix, "float32"), compute_uv=True)
    backend.eval(s)
    sigma_svd = float(backend.to_scalar(s[0]))
    assert sigma_exact == pytest.approx(sigma_svd, rel=0.0, abs=1e-8)
