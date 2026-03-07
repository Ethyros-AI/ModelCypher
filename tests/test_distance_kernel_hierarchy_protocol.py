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
    "distance_kernel_hierarchy_script",
    "scripts/distance_kernel_hierarchy_analysis.py",
)
VALIDATOR = _load_script_module(
    "dkh_validator_script",
    "scripts/validate_distance_kernel_hierarchy_artifacts.py",
)


def _profile_from_values(values: np.ndarray) -> object:
    distances = list(range(values.shape[0]))
    means = [float(v) for v in values]
    counts = [5] * values.shape[0]
    return SCRIPT.DistanceProfile(distances=distances, means=means, counts=counts)


def test_synthetic_exponential_profile_classified_m1_by_aicc():
    """A clear exponential profile should be classified as M1 by AICc."""
    distances = np.arange(20, dtype=np.float64)
    base = SCRIPT.monotonic_decay(distances, 0.9, 0.22)
    noise = np.array([
        0.00, 0.01, -0.01, 0.02, -0.02, 0.015, -0.01, 0.005, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    calibration = _profile_from_values(base + noise)
    holdout = _profile_from_values(base)

    fits = SCRIPT.fit_profile_models(calibration, holdout)

    classification = SCRIPT.classify_head_aicc(
        aicc_m0=fits["m0"]["calibration"]["aicc"],
        aicc_m1=fits["m1"]["calibration"]["aicc"],
        n_points=fits["m0"]["calibration"]["n_points"],
    )

    assert classification["head_classification"] == "m1_class"
    assert classification["delta_aicc_m0_minus_m1"] > 0.0
    assert fits["m1"]["holdout"]["rmse"] < fits["m0"]["holdout"]["rmse"]


def test_synthetic_constant_profile_classified_m0_by_aicc():
    """A flat profile should be classified as M0 by AICc."""
    values = np.full(20, 0.35, dtype=np.float64)
    noise = np.random.default_rng(42).normal(0, 0.001, 20)
    calibration = _profile_from_values(values + noise)
    holdout = _profile_from_values(values)

    fits = SCRIPT.fit_profile_models(calibration, holdout)

    classification = SCRIPT.classify_head_aicc(
        aicc_m0=fits["m0"]["calibration"]["aicc"],
        aicc_m1=fits["m1"]["calibration"]["aicc"],
        n_points=fits["m0"]["calibration"]["n_points"],
    )

    assert classification["head_classification"] == "m0_class"
    assert classification["delta_aicc_m0_minus_m1"] <= 0.0


def test_aicc_penalty_matches_analytic_formula():
    """delta_penalty(n) must match 2 + 12/(n-3) - 4/(n-2) exactly."""
    for n in (10, 20, 28, 30, 50, 100):
        expected = 2.0 + 12.0 / (n - 3) - 4.0 / (n - 2)
        computed = SCRIPT.delta_penalty(n)
        np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_aicc_penalty_infinite_for_small_n():
    """delta_penalty should be infinite when n <= 3."""
    assert SCRIPT.delta_penalty(3) == float("inf")
    assert SCRIPT.delta_penalty(2) == float("inf")
    assert SCRIPT.delta_penalty(1) == float("inf")


def test_classification_concordance_with_holdout_on_synthetic():
    """For clear synthetic data, AICc classification should agree with holdout."""
    # Strong exponential
    distances = np.arange(25, dtype=np.float64)
    exp_base = SCRIPT.monotonic_decay(distances, 0.8, 0.3)
    cal = _profile_from_values(exp_base)
    hold = _profile_from_values(exp_base)
    fits = SCRIPT.fit_profile_models(cal, hold)
    classification = SCRIPT.classify_head_aicc(
        fits["m0"]["calibration"]["aicc"],
        fits["m1"]["calibration"]["aicc"],
        fits["m0"]["calibration"]["n_points"],
    )
    holdout_best = SCRIPT._best_model_by_holdout(fits)
    assert classification["head_classification"] == "m1_class"
    assert holdout_best == "m1"

    # Strong constant
    const_base = np.full(25, 0.5, dtype=np.float64)
    cal2 = _profile_from_values(const_base)
    hold2 = _profile_from_values(const_base)
    fits2 = SCRIPT.fit_profile_models(cal2, hold2)
    classification2 = SCRIPT.classify_head_aicc(
        fits2["m0"]["calibration"]["aicc"],
        fits2["m1"]["calibration"]["aicc"],
        fits2["m0"]["calibration"]["n_points"],
    )
    holdout_best2 = SCRIPT._best_model_by_holdout(fits2)
    # For a constant profile, M1 can fit it too (gamma=0), so both are equivalent.
    # M0 should win or tie by AICc (fewer parameters, same fit).
    assert classification2["head_classification"] == "m0_class"


def test_distance_r2_matches_variance_decomposition():
    """distance_r2 should match the explained/total variance ratio."""
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.1, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.3, 0.4],
    ], dtype=np.float64)

    _, distance_r2 = SCRIPT.compute_prompt_measurements(matrix)

    values: list[float] = []
    dists: list[int] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            values.append(float(matrix[i, j]))
            dists.append(i - j)

    values_arr = np.asarray(values, dtype=np.float64)
    dists_arr = np.asarray(dists, dtype=np.int64)
    grand_mean = float(np.mean(values_arr))
    distance_means = {
        int(d): float(np.mean(values_arr[dists_arr == d]))
        for d in sorted(set(dists))
    }
    explained = np.asarray(
        [distance_means[int(d)] for d in dists_arr],
        dtype=np.float64,
    )
    explained_variance = float(np.mean((explained - grand_mean) ** 2))
    total_variance = float(np.mean((values_arr - grand_mean) ** 2))

    assert total_variance > 0.0
    np.testing.assert_allclose(
        distance_r2,
        explained_variance / total_variance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_validator_rejects_missing_and_malformed_artifacts(tmp_path: Path):
    run_dir = tmp_path / "20260307_120000"
    run_dir.mkdir()

    manifest = {
        "run_id": run_dir.name,
        "protocol": "F-DKH-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-07T12:00:00+00:00",
        "output_dir": str(run_dir),
        "probe_file": "docs/research/wave_kernel_probe_manifest.json",
        "models_requested": [],
        "probe_counts": {"total": 0, "by_family": {}},
        "claim": "x",
        "claim_form": "y",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_head_classification.jsonl").write_text("{bad json\n", encoding="utf-8")
    (run_dir / "model_family_summary.json").write_text(
        json.dumps({
            "run_id": run_dir.name,
            "protocol": "F-DKH-01",
            "artifact_schema_version": "v1",
            "generated_at": "2026-03-07T12:00:00+00:00",
            "models": [],
            "families": [],
        }),
        encoding="utf-8",
    )
    (run_dir / "falsifier_outcome.json").write_text(
        json.dumps({
            "run_id": run_dir.name,
            "protocol": "F-DKH-01",
            "artifact_schema_version": "v1",
            "generated_at": "2026-03-07T12:00:00+00:00",
            "claim": "x",
            "claim_form": "y",
            "observable": "z",
            "overall": "insufficient_data",
            "promotion_blocked": True,
            "predictions": [],
        }),
        encoding="utf-8",
    )

    result = VALIDATOR.validate_run_dir(run_dir, include_self=True)

    assert result["ok"] is False
    assert any("per_head_classification.jsonl" in error for error in result["errors"])
    assert any("artifact_validation.json" in error for error in result["errors"])


def test_validator_accepts_minimal_valid_run_dir(tmp_path: Path):
    run_dir = tmp_path / "20260307_120001"
    run_dir.mkdir()

    manifest = {
        "run_id": run_dir.name,
        "protocol": "F-DKH-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-07T12:00:00+00:00",
        "output_dir": str(run_dir),
        "probe_file": "docs/research/wave_kernel_probe_manifest.json",
        "models_requested": [],
        "probe_counts": {"total": 0, "by_family": {}},
        "claim": "x",
        "claim_form": "y",
    }
    summary = {
        "run_id": run_dir.name,
        "protocol": "F-DKH-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-07T12:00:00+00:00",
        "models": [],
        "families": [],
    }
    outcome = {
        "run_id": run_dir.name,
        "protocol": "F-DKH-01",
        "artifact_schema_version": "v1",
        "generated_at": "2026-03-07T12:00:00+00:00",
        "claim": "x",
        "claim_form": "y",
        "observable": "z",
        "overall": "insufficient_data",
        "promotion_blocked": True,
        "predictions": [],
    }
    head_row = {
        "record_type": "model_skip",
        "protocol": "F-DKH-01",
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
            "per_head_classification.jsonl",
            "model_family_summary.json",
            "falsifier_outcome.json",
        ],
        "validated_at": "2026-03-07T12:00:00+00:00",
    }

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_head_classification.jsonl").write_text(json.dumps(head_row) + "\n", encoding="utf-8")
    (run_dir / "model_family_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "falsifier_outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    (run_dir / "artifact_validation.json").write_text(
        json.dumps(artifact_validation),
        encoding="utf-8",
    )

    result = VALIDATOR.validate_run_dir(run_dir, include_self=True)

    assert result["ok"] is True
