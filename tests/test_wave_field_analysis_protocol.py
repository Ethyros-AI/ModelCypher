# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module(
    "wave_field_analysis_script",
    "scripts/wave_field_analysis.py",
)
VALIDATOR = _load_script_module(
    "wave_field_validator_script",
    "scripts/validate_wave_kernel_falsifier_artifacts.py",
)


def _profile_from_values(values: np.ndarray) -> object:
    distances = list(range(values.shape[0]))
    means = [float(v) for v in values]
    counts = [5] * values.shape[0]
    return SCRIPT.DistanceProfile(distances=distances, means=means, counts=counts)


def test_synthetic_exponential_profile_prefers_m1_on_holdout():
    distances = np.arange(12, dtype=np.float64)
    base = SCRIPT.monotonic_decay(distances, 0.9, 0.22)
    calibration = _profile_from_values(base + np.array([
        0.00, 0.01, -0.01, 0.02, -0.02, 0.015, -0.01, 0.005, 0.0, 0.0, 0.0, 0.0,
    ]))
    holdout = _profile_from_values(base)

    fits = SCRIPT.fit_profile_models(calibration, holdout)

    assert fits["m1"]["holdout"]["rmse"] is not None
    assert fits["m2"]["holdout"]["rmse"] is not None
    assert fits["m1"]["holdout"]["rmse"] <= fits["m2"]["holdout"]["rmse"]
    assert fits["m2"]["boundary_equivalent"] is True


def test_synthetic_wave_profile_prefers_m2_on_holdout():
    distances = np.arange(16, dtype=np.float64)
    calibration_values = SCRIPT.damped_oscillation(distances, 0.8, 0.12, 1.0, 0.15)
    holdout_values = SCRIPT.damped_oscillation(distances, 0.8, 0.12, 1.0, 0.15)
    calibration = _profile_from_values(calibration_values)
    holdout = _profile_from_values(holdout_values)

    fits = SCRIPT.fit_profile_models(calibration, holdout)

    assert fits["m2"]["holdout"]["rmse"] is not None
    assert fits["m1"]["holdout"]["rmse"] is not None
    assert fits["m2"]["holdout"]["rmse"] < fits["m1"]["holdout"]["rmse"]
    assert fits["m2"]["boundary_equivalent"] is False


def test_content_dominated_matrix_has_low_distance_r2_and_no_wave_support():
    rng = np.random.default_rng(7)
    seq_len = 14
    matrix = np.zeros((seq_len, seq_len), dtype=np.float64)
    for row in range(seq_len):
        raw = rng.random(row + 1)
        matrix[row, : row + 1] = raw / raw.sum()

    profile, distance_r2 = SCRIPT.compute_prompt_measurements(matrix)
    calibration = SCRIPT.aggregate_profiles([profile, profile])
    holdout_noise = _profile_from_values(rng.random(len(profile.distances)))
    fits = SCRIPT.fit_profile_models(calibration, holdout_noise)
    holdout_delta = fits["m2"]["holdout"]["rmse"] - fits["m1"]["holdout"]["rmse"]

    assert distance_r2 < 0.5
    assert not (holdout_delta < 0.0 and fits["m2"]["boundary_equivalent"] is False)


def test_boundary_equivalent_detects_flat_m2_case():
    distances = np.arange(10, dtype=np.float64)
    params = {
        "a": 1.0,
        "gamma": 0.0,
        "omega": 0.0,
        "phi": -math.pi / 2.0,
    }

    assert SCRIPT._m2_boundary_equivalent(distances, params) is True


def test_validator_rejects_missing_and_malformed_artifacts(tmp_path: Path):
    run_dir = tmp_path / "20260306_120000"
    run_dir.mkdir()

    manifest = {
        "run_id": run_dir.name,
        "protocol": "F-WAVE-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-06T12:00:00+00:00",
        "output_dir": str(run_dir),
        "probe_file": "docs/research/wave_kernel_probe_manifest.json",
        "models_requested": [],
        "probe_counts": {"total": 0, "by_family": {}},
        "claim": "x",
        "claim_form": "y",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_head_fit_table.jsonl").write_text("{bad json\n", encoding="utf-8")
    (run_dir / "model_family_summary.json").write_text(
        json.dumps({
            "run_id": run_dir.name,
            "protocol": "F-WAVE-01",
            "artifact_schema_version": "v1",
            "generated_at": "2026-03-06T12:00:00+00:00",
            "models": [],
            "families": [],
        }),
        encoding="utf-8",
    )
    (run_dir / "falsifier_outcome.json").write_text(
        json.dumps({
            "run_id": run_dir.name,
            "protocol": "F-WAVE-01",
            "artifact_schema_version": "v1",
            "generated_at": "2026-03-06T12:00:00+00:00",
            "claim": "x",
            "claim_form": "y",
            "observable": "z",
            "overall": "insufficient_data",
            "promotion_blocked": True,
            "family_outcomes": [],
            "model_outcomes": [],
        }),
        encoding="utf-8",
    )

    result = VALIDATOR.validate_run_dir(run_dir, include_self=True)

    assert result["ok"] is False
    assert any("per_head_fit_table.jsonl" in error for error in result["errors"])
    assert any("artifact_validation.json" in error for error in result["errors"])


def test_validator_accepts_minimal_valid_run_dir(tmp_path: Path):
    run_dir = tmp_path / "20260306_120001"
    run_dir.mkdir()

    manifest = {
        "run_id": run_dir.name,
        "protocol": "F-WAVE-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-06T12:00:00+00:00",
        "output_dir": str(run_dir),
        "probe_file": "docs/research/wave_kernel_probe_manifest.json",
        "models_requested": [],
        "probe_counts": {"total": 0, "by_family": {}},
        "claim": "x",
        "claim_form": "y",
    }
    summary = {
        "run_id": run_dir.name,
        "protocol": "F-WAVE-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-06T12:00:00+00:00",
        "models": [],
        "families": [],
    }
    outcome = {
        "run_id": run_dir.name,
        "protocol": "F-WAVE-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-06T12:00:00+00:00",
        "claim": "x",
        "claim_form": "y",
        "observable": "z",
        "overall": "insufficient_data",
        "promotion_blocked": True,
        "family_outcomes": [],
        "model_outcomes": [],
    }
    head_row = {
        "record_type": "model_skip",
        "protocol": "F-WAVE-01",
        "artifact_schema_version": "v1",
        "model_path": "/tmp/model",
        "model_name": "Qwen3.5-0.8B-bf16",
        "family": "Qwen",
        "status": "skipped",
    }
    artifact_validation = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "run_dir": str(run_dir),
        "files_checked": [
            "run_manifest.json",
            "per_head_fit_table.jsonl",
            "model_family_summary.json",
            "falsifier_outcome.json",
        ],
        "validated_at": "2026-03-06T12:00:00+00:00",
    }

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_head_fit_table.jsonl").write_text(json.dumps(head_row) + "\n", encoding="utf-8")
    (run_dir / "model_family_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "falsifier_outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    (run_dir / "artifact_validation.json").write_text(
        json.dumps(artifact_validation),
        encoding="utf-8",
    )

    result = VALIDATOR.validate_run_dir(run_dir, include_self=True)

    assert result["ok"] is True
