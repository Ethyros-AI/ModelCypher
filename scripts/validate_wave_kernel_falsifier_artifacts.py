#!/usr/bin/env python3
"""Validate wave-kernel falsifier artifact directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "run_manifest.json",
    "per_head_fit_table.jsonl",
    "model_family_summary.json",
    "falsifier_outcome.json",
    "artifact_validation.json",
)

RUN_MANIFEST_REQUIRED = {
    "run_id",
    "protocol",
    "artifact_schema_version",
    "generated_at",
    "output_dir",
    "probe_file",
    "models_requested",
    "probe_counts",
    "claim",
    "claim_form",
}

MODEL_SUMMARY_REQUIRED = {
    "run_id",
    "protocol",
    "artifact_schema_version",
    "generated_at",
    "models",
    "families",
}

FALSIFIER_OUTCOME_REQUIRED = {
    "run_id",
    "protocol",
    "artifact_schema_version",
    "generated_at",
    "claim",
    "claim_form",
    "observable",
    "overall",
    "promotion_blocked",
    "family_outcomes",
    "model_outcomes",
}

ARTIFACT_VALIDATION_REQUIRED = {
    "ok",
    "errors",
    "warnings",
    "run_dir",
    "files_checked",
    "validated_at",
}

PER_HEAD_REQUIRED = {
    "record_type",
    "protocol",
    "artifact_schema_version",
    "model_path",
    "model_name",
    "family",
    "status",
}


def _missing_keys(doc: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(key for key in required if key not in doc)


def _validate_json_file(
    *,
    file_path: Path,
    required_keys: set[str],
    errors: list[str],
    files_checked: list[str],
) -> dict[str, Any] | None:
    if not file_path.exists():
        errors.append(f"{file_path.name}: missing required file")
        return None

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            f"{file_path.name}: JSON parse error at line {exc.lineno}, "
            f"col {exc.colno}: {exc.msg}"
        )
        return None

    if not isinstance(payload, dict):
        errors.append(f"{file_path.name}: top-level JSON must be an object")
        return None

    missing = _missing_keys(payload, required_keys)
    if missing:
        errors.append(f"{file_path.name}: missing keys: {', '.join(missing)}")
    files_checked.append(file_path.name)
    return payload


def _validate_jsonl_file(
    *,
    file_path: Path,
    errors: list[str],
    files_checked: list[str],
) -> list[dict[str, Any]]:
    if not file_path.exists():
        errors.append(f"{file_path.name}: missing required file")
        return []

    rows: list[dict[str, Any]] = []
    for line_idx, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(
                f"{file_path.name}: JSON parse error on line {line_idx}, "
                f"col {exc.colno}: {exc.msg}"
            )
            continue

        if not isinstance(payload, dict):
            errors.append(f"{file_path.name}: line {line_idx} must decode to an object")
            continue

        missing = _missing_keys(payload, PER_HEAD_REQUIRED)
        if missing:
            errors.append(
                f"{file_path.name}: line {line_idx} missing keys: {', '.join(missing)}"
            )
            continue
        rows.append(payload)

    if not rows:
        errors.append(f"{file_path.name}: no valid JSONL rows found")
    files_checked.append(file_path.name)
    return rows


def validate_run_dir(
    run_dir: Path,
    *,
    include_self: bool = True,
) -> dict[str, Any]:
    """Validate one run directory."""
    errors: list[str] = []
    warnings: list[str] = []
    files_checked: list[str] = []

    if not run_dir.exists():
        errors.append(f"{run_dir}: run directory does not exist")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "run_dir": str(run_dir),
            "files_checked": files_checked,
        }
    if not run_dir.is_dir():
        errors.append(f"{run_dir}: not a directory")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "run_dir": str(run_dir),
            "files_checked": files_checked,
        }

    run_manifest = _validate_json_file(
        file_path=run_dir / "run_manifest.json",
        required_keys=RUN_MANIFEST_REQUIRED,
        errors=errors,
        files_checked=files_checked,
    )
    _validate_jsonl_file(
        file_path=run_dir / "per_head_fit_table.jsonl",
        errors=errors,
        files_checked=files_checked,
    )
    model_summary = _validate_json_file(
        file_path=run_dir / "model_family_summary.json",
        required_keys=MODEL_SUMMARY_REQUIRED,
        errors=errors,
        files_checked=files_checked,
    )
    falsifier_outcome = _validate_json_file(
        file_path=run_dir / "falsifier_outcome.json",
        required_keys=FALSIFIER_OUTCOME_REQUIRED,
        errors=errors,
        files_checked=files_checked,
    )

    if include_self:
        artifact_validation = _validate_json_file(
            file_path=run_dir / "artifact_validation.json",
            required_keys=ARTIFACT_VALIDATION_REQUIRED,
            errors=errors,
            files_checked=files_checked,
        )
        if artifact_validation is not None and Path(
            artifact_validation.get("run_dir", "")
        ).resolve() != run_dir.resolve():
            errors.append("artifact_validation.json: run_dir does not match validated directory")

    if run_manifest is not None and falsifier_outcome is not None:
        if run_manifest["run_id"] != falsifier_outcome["run_id"]:
            errors.append("run_id mismatch between run_manifest.json and falsifier_outcome.json")

    if run_manifest is not None and model_summary is not None:
        if run_manifest["run_id"] != model_summary["run_id"]:
            errors.append("run_id mismatch between run_manifest.json and model_family_summary.json")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "run_dir": str(run_dir),
        "files_checked": files_checked,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate wave-kernel falsifier artifact directories.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Validate one run directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/wave_kernel_falsifier"),
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Validate all run subdirectories under --root.",
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
        run_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
        if not run_dirs:
            print(f"ERROR: no run directories found under {args.root}")
            return 1

    all_ok = True
    for run_dir in run_dirs:
        result = validate_run_dir(run_dir, include_self=True)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] {run_dir} (files_checked={len(result['files_checked'])})")
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
