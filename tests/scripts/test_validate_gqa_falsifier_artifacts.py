"""Tests for gqa_falsifier_protocol artifact validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_gqa_falsifier_artifacts as validator


def _model_table_doc() -> dict:
    return {
        "run_id": "r1",
        "timestamp": "2026-03-04T00:00:00",
        "protocol": "F-GQA-01",
        "n_models": 1,
        "models": [],
    }


def _within_family_doc() -> dict:
    return {"run_id": "r1", "families": {}}


def _falsifier_doc() -> dict:
    return {
        "run_id": "r1",
        "protocol": "F-GQA-01",
        "timestamp": "2026-03-04T00:00:00",
        "falsifiers": {},
        "overall": "INCONCLUSIVE",
    }


def _regression_doc_v1() -> dict:
    return {
        "run_id": "r1",
        "z_couple_regression": {},
        "c_cancel_regression": {},
        "z_couple_diagnostics": {},
        "c_cancel_diagnostics": {},
    }


def _regression_doc_v2() -> dict:
    return {
        "run_id": "r1",
        "z_couple_regression_full": {},
        "z_couple_regression_commensurable": {},
        "c_cancel_regression": {},
        "z_couple_diagnostics": {},
        "c_cancel_diagnostics": {},
        "commensurability_note": "x",
    }


def _write_run_dir(run_dir: Path, regression_doc: dict, truncate_regression: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_table.json").write_text(json.dumps(_model_table_doc()), encoding="utf-8")
    if truncate_regression:
        (run_dir / "regression_summary.json").write_text('{"run_id":', encoding="utf-8")
    else:
        (run_dir / "regression_summary.json").write_text(json.dumps(regression_doc), encoding="utf-8")
    (run_dir / "within_family_trends.json").write_text(json.dumps(_within_family_doc()), encoding="utf-8")
    (run_dir / "falsifier_outcome.json").write_text(json.dumps(_falsifier_doc()), encoding="utf-8")


def test_validate_v2_run_dir_passes(tmp_path):
    run_dir = tmp_path / "run_v2"
    _write_run_dir(run_dir, _regression_doc_v2())

    result = validator.validate_run_dir(run_dir, schema_mode="auto")

    assert result["ok"] is True
    assert result["detected_schema"] == "v2"
    assert result["errors"] == []
    assert sorted(result["files_checked"]) == sorted(validator.REQUIRED_FILES)


def test_validate_v1_run_dir_passes_auto(tmp_path):
    run_dir = tmp_path / "run_v1"
    _write_run_dir(run_dir, _regression_doc_v1())

    result = validator.validate_run_dir(run_dir, schema_mode="auto")

    assert result["ok"] is True
    assert result["detected_schema"] == "v1"
    assert result["errors"] == []


def test_truncated_json_fails(tmp_path):
    run_dir = tmp_path / "run_bad_json"
    _write_run_dir(run_dir, _regression_doc_v2(), truncate_regression=True)

    result = validator.validate_run_dir(run_dir, schema_mode="auto")

    assert result["ok"] is False
    assert any("JSON parse error" in e for e in result["errors"])


def test_missing_required_file_fails(tmp_path):
    run_dir = tmp_path / "run_missing"
    _write_run_dir(run_dir, _regression_doc_v2())
    (run_dir / "within_family_trends.json").unlink()

    result = validator.validate_run_dir(run_dir, schema_mode="auto")

    assert result["ok"] is False
    assert any("within_family_trends.json: missing required file" in e for e in result["errors"])


def test_schema_mode_v2_rejects_v1(tmp_path):
    run_dir = tmp_path / "run_v1_reject"
    _write_run_dir(run_dir, _regression_doc_v1())

    result = validator.validate_run_dir(run_dir, schema_mode="v2")

    assert result["ok"] is False
    assert any("missing v2 keys" in e for e in result["errors"])


def test_cli_all_runs_mixed_results_exit_nonzero(tmp_path):
    root = tmp_path / "runs"
    good = root / "good"
    bad = root / "bad"
    _write_run_dir(good, _regression_doc_v2())
    _write_run_dir(bad, _regression_doc_v2(), truncate_regression=True)

    exit_code = validator.main(["--root", str(root), "--all-runs", "--schema", "auto"])
    assert exit_code == 1
