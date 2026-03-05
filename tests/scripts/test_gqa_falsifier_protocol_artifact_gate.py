"""Tests for artifact integrity gate in gqa_falsifier_protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gqa_falsifier_protocol as gqa


def test_artifact_gate_raises_on_validator_failure(monkeypatch, tmp_path):
    emitted_dir = tmp_path / "run_out"
    emitted_dir.mkdir(parents=True)

    monkeypatch.setattr(
        gqa.artifact_validator,
        "validate_run_dir",
        lambda _run_dir, schema_mode="v2": {
            "ok": False,
            "errors": ["simulated failure"],
            "warnings": [],
            "run_dir": str(emitted_dir),
            "detected_schema": None,
            "files_checked": [],
        },
    )

    with pytest.raises(RuntimeError, match="Artifact integrity validation failed"):
        gqa._validate_emitted_artifacts_or_raise(emitted_dir)


def test_run_full_raises_after_emit_when_validator_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gqa,
        "MODEL_REGISTRY",
        {
            "TestModel": {
                "path": "/tmp/does-not-matter",
                "L": 4,
                "d": 64,
                "GQA": 2,
                "hybrid": False,
                "family": "test",
                "quant": "bf16",
            }
        },
    )

    monkeypatch.setattr(
        gqa,
        "compute_z_couple",
        lambda _name: {
            "model": "TestModel",
            "z_couple": 0.1,
            "r_pearson": 0.1,
            "n_eff_z": 6.0,
            "mde": 0.5,
            "above_mde": False,
            "commensurable": True,
            "h_logit_resid_range": 1.0,
            "h_logit_saturation": 0.9,
        },
    )
    monkeypatch.setattr(
        gqa,
        "compute_c_cancel_from_cached",
        lambda _name: {
            "model": "TestModel",
            "c_cancel": 0.2,
            "beta_num": 0.1,
            "beta_den": -0.1,
            "n_eff_c": 6.0,
        },
    )
    monkeypatch.setattr(gqa, "design_diagnostics", lambda _records, _response: {"n": 1, "error": "insufficient_data"})
    monkeypatch.setattr(
        gqa,
        "weighted_ols_regression",
        lambda _records, _response, _weights, label: {"label": label, "n": 1, "error": "insufficient_data"},
    )
    monkeypatch.setattr(
        gqa,
        "adjudicate_falsifiers",
        lambda *_args, **_kwargs: {
            "F1": {"status": "UNDERPOWERED", "reason": "x"},
            "F2": {"status": "UNDERPOWERED", "reason": "x"},
            "F3": {"status": "UNDERPOWERED", "reason": "x"},
        },
    )

    emitted = {"called": False}

    def _emit(*_args, **_kwargs):
        emitted["called"] = True
        out_dir = tmp_path / "emitted_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(gqa, "emit_artifacts", _emit)
    monkeypatch.setattr(
        gqa.artifact_validator,
        "validate_run_dir",
        lambda _run_dir, schema_mode="v2": {
            "ok": False,
            "errors": ["forced validator failure"],
            "warnings": [],
            "run_dir": "x",
            "detected_schema": None,
            "files_checked": [],
        },
    )

    with pytest.raises(RuntimeError, match="Artifact integrity validation failed"):
        gqa.run_full(collect_missing=False, include_smollm3=False, run_id="test_gate_failure")

    assert emitted["called"] is True
