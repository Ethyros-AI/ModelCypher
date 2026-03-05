#!/usr/bin/env python3
"""Validate gqa_falsifier_protocol artifact directories.

Checks:
1. Required files exist
2. Every required file is valid JSON
3. Minimal required keys are present
4. Regression summary matches schema mode (v1, v2, or auto-detected)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "model_table.json",
    "regression_summary.json",
    "within_family_trends.json",
    "falsifier_outcome.json",
)

MODEL_TABLE_REQUIRED = {"run_id", "timestamp", "protocol", "n_models", "models"}
WITHIN_FAMILY_REQUIRED = {"run_id", "families"}
FALSIFIER_OUTCOME_REQUIRED = {"run_id", "protocol", "timestamp", "falsifiers", "overall"}

REGRESSION_V1_REQUIRED = {
    "run_id",
    "z_couple_regression",
    "c_cancel_regression",
    "z_couple_diagnostics",
    "c_cancel_diagnostics",
}

REGRESSION_V2_REQUIRED = {
    "run_id",
    "z_couple_regression_full",
    "z_couple_regression_commensurable",
    "c_cancel_regression",
    "z_couple_diagnostics",
    "c_cancel_diagnostics",
    "commensurability_note",
}


def _missing_keys(doc: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(k for k in required if k not in doc)


def _validate_required_keys(
    file_name: str,
    payload: dict[str, Any],
    required: set[str],
    errors: list[str],
) -> None:
    missing = _missing_keys(payload, required)
    if missing:
        errors.append(f"{file_name}: missing keys: {', '.join(missing)}")


def _detect_regression_schema(
    payload: dict[str, Any],
    schema_mode: str,
) -> tuple[str | None, list[str]]:
    errors: list[str] = []

    if schema_mode == "v1":
        missing = _missing_keys(payload, REGRESSION_V1_REQUIRED)
        if missing:
            errors.append(f"regression_summary.json: missing v1 keys: {', '.join(missing)}")
            return None, errors
        return "v1", errors

    if schema_mode == "v2":
        missing = _missing_keys(payload, REGRESSION_V2_REQUIRED)
        if missing:
            errors.append(f"regression_summary.json: missing v2 keys: {', '.join(missing)}")
            return None, errors
        return "v2", errors

    # auto: v2 preferred if present, else v1, else error
    has_v2 = all(k in payload for k in REGRESSION_V2_REQUIRED)
    has_v1 = all(k in payload for k in REGRESSION_V1_REQUIRED)
    if has_v2:
        return "v2", errors
    if has_v1:
        return "v1", errors

    errors.append(
        "regression_summary.json: cannot detect schema in auto mode "
        "(missing both v1 and v2 required key sets)"
    )
    return None, errors


def validate_run_dir(run_dir: Path, schema_mode: str = "auto") -> dict[str, Any]:
    """Validate one run directory.

    Returns:
      {
        "ok": bool,
        "errors": list[str],
        "warnings": list[str],
        "run_dir": str,
        "detected_schema": str|None,
        "files_checked": list[str],
      }
    """
    errors: list[str] = []
    warnings: list[str] = []
    files_checked: list[str] = []
    detected_schema: str | None = None

    if not run_dir.exists():
        errors.append(f"{run_dir}: run directory does not exist")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "run_dir": str(run_dir),
            "detected_schema": detected_schema,
            "files_checked": files_checked,
        }
    if not run_dir.is_dir():
        errors.append(f"{run_dir}: not a directory")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "run_dir": str(run_dir),
            "detected_schema": detected_schema,
            "files_checked": files_checked,
        }

    parsed: dict[str, dict[str, Any]] = {}
    for file_name in REQUIRED_FILES:
        file_path = run_dir / file_name
        if not file_path.exists():
            errors.append(f"{file_name}: missing required file")
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(
                f"{file_name}: JSON parse error at line {exc.lineno}, col {exc.colno}: {exc.msg}"
            )
            continue
        except OSError as exc:
            errors.append(f"{file_name}: read error: {exc}")
            continue

        if not isinstance(payload, dict):
            errors.append(f"{file_name}: top-level JSON must be an object")
            continue

        parsed[file_name] = payload
        files_checked.append(file_name)

    if "model_table.json" in parsed:
        _validate_required_keys("model_table.json", parsed["model_table.json"], MODEL_TABLE_REQUIRED, errors)
    if "within_family_trends.json" in parsed:
        _validate_required_keys(
            "within_family_trends.json",
            parsed["within_family_trends.json"],
            WITHIN_FAMILY_REQUIRED,
            errors,
        )
    if "falsifier_outcome.json" in parsed:
        _validate_required_keys(
            "falsifier_outcome.json",
            parsed["falsifier_outcome.json"],
            FALSIFIER_OUTCOME_REQUIRED,
            errors,
        )
    if "regression_summary.json" in parsed:
        detected_schema, schema_errors = _detect_regression_schema(
            parsed["regression_summary.json"],
            schema_mode=schema_mode,
        )
        errors.extend(schema_errors)
        if detected_schema == "v1":
            warnings.append("regression_summary.json: legacy v1 schema detected")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "run_dir": str(run_dir),
        "detected_schema": detected_schema,
        "files_checked": files_checked,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gqa_falsifier_protocol artifact directories.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Validate one run directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/gqa_falsifier_protocol"),
        help="Root directory containing run subdirectories (default: results/gqa_falsifier_protocol).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Validate all run subdirectories under --root.",
    )
    parser.add_argument(
        "--schema",
        choices=("auto", "v1", "v2"),
        default="auto",
        help="Schema mode for regression_summary.json (default: auto).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.run_dir is not None and args.all_runs:
        print("ERROR: use either --run-dir or --all-runs, not both.")
        return 1
    if args.run_dir is None and not args.all_runs:
        print("ERROR: provide --run-dir or --all-runs.")
        return 1

    if args.run_dir is not None:
        run_dirs = [args.run_dir]
    else:
        if not args.root.exists() or not args.root.is_dir():
            print(f"ERROR: root directory not found: {args.root}")
            return 1
        run_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
        if not run_dirs:
            print(f"ERROR: no run directories found under {args.root}")
            return 1

    all_ok = True
    for run_dir in run_dirs:
        result = validate_run_dir(run_dir, schema_mode=args.schema)
        status = "PASS" if result["ok"] else "FAIL"
        schema = result["detected_schema"] or "unknown"
        print(f"[{status}] {run_dir} (schema={schema}, files_checked={len(result['files_checked'])})")
        for warning in result["warnings"]:
            print(f"  WARN: {warning}")
        for error in result["errors"]:
            print(f"  ERROR: {error}")
        if not result["ok"]:
            all_ok = False

    print(f"SUMMARY: {'PASS' if all_ok else 'FAIL'} ({len(run_dirs)} run directories)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
