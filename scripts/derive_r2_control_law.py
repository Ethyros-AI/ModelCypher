#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Derive the R2 closed-loop law from retained artifacts.

This is research-only glue for the R2/Q1 blocker. It normalizes the retained
350M counterexamples and non-counterexample references into one state table,
emits a serializable closed-loop law, and writes a pre-registered falsifier
manifest for the single frozen-tuple rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.mass_step_size import (
    CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    DerivedClosedLoopLaw,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    BehavioralStateMeasurement,
    compute_closed_loop_trigger_reasons,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "nblora_vs_standard"
PIPELINE_REFERENCE = REPO_ROOT / "results" / "pipeline_validation" / "350M" / "result.json"
STAGE_A_REPORT = RESULTS_ROOT / "validate_derived_stage_a_seed42.json"
CAYLEY_REPORT = RESULTS_ROOT / "validate_derived_r2_behavioral_probe_cayley_seed42_quick.json"
ADAMW_REPORT = RESULTS_ROOT / "validate_derived_r2_behavioral_probe_adamw_seed42_quick.json"
CAYLEY_LOG = RESULTS_ROOT / "validate_derived_r2_behavioral_probe_cayley_seed42_quick.log"
ADAMW_LOG = RESULTS_ROOT / "validate_derived_r2_behavioral_probe_adamw_seed42_quick.log"

DEFAULT_STATE_TABLE = RESULTS_ROOT / "r2_artifact_state_table.json"
DEFAULT_LAW_PATH = RESULTS_ROOT / "r2_control_law.json"
DEFAULT_MANIFEST_PATH = RESULTS_ROOT / "r2_closed_loop_falsifier_manifest.json"
DEFAULT_VALIDATION_PATH = RESULTS_ROOT / "r2_control_law_validation.json"
DEFAULT_FALSIFIER_REPORT = RESULTS_ROOT / "validate_derived_r2_closed_loop_seed42_quick.json"
DEFAULT_FALSIFIER_ARTIFACT_ROOT = RESULTS_ROOT / "phase5_artifacts_r2_closed_loop_seed42"
DEFAULT_LEDGER_PATH = RESULTS_ROOT / "R1-LEDGER.tsv"

FROZEN_MODEL_PATH = Path("/Users/jasonkempf/Local Models/LFM2-350M-MLX-bf16")
FROZEN_TRAIN_PATH = REPO_ROOT / "data" / "training" / "benchmark_train.jsonl"
FROZEN_EVAL_PATH = REPO_ROOT / "data" / "training" / "benchmark_val.jsonl"

_ONLINE_EVAL_RE = re.compile(
    r"Online eval epoch (?P<epoch>\d+): (?P<correct>\d+)/(?P<total>\d+) correct "
    r"\((?P<acc>[0-9.]+)%\), baseline=(?P<base_correct>\d+)/(?P<base_total>\d+), "
    r"lost=(?P<lost>\d+), gained=(?P<gained>\d+), degraded_raw=(?P<raw>True|False), "
    r"degraded_significant=(?P<sig>True|False)",
)
_MARGIN_RE = re.compile(
    r"Margin: median=(?P<median>-?[0-9.]+) mean=(?P<mean>-?[0-9.]+) "
    r"min=(?P<min>-?[0-9.]+) near_zero=(?P<near_zero>\d+)",
)
_STABLE_RANK_RE = re.compile(
    r"Stable rank: median=(?P<median>-?[0-9.]+) min=(?P<min>-?[0-9.]+) "
    r"\(rank=(?P<rank>\d+)\)",
)
_STOP_EPOCH_RE = re.compile(r"epoch=(?P<epoch>\d+)")


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    report_path: Path
    safe_reference: bool
    log_path: Path | None = None
    select_passed_trial: bool = False


def _default_artifacts() -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            artifact_id="stage_a_seed42",
            report_path=STAGE_A_REPORT,
            safe_reference=True,
        ),
        ArtifactSpec(
            artifact_id="pipeline_validation_safe",
            report_path=PIPELINE_REFERENCE,
            safe_reference=True,
            select_passed_trial=True,
        ),
        ArtifactSpec(
            artifact_id="behavioral_probe_cayley_seed42",
            report_path=CAYLEY_REPORT,
            safe_reference=False,
            log_path=CAYLEY_LOG,
        ),
        ArtifactSpec(
            artifact_id="behavioral_probe_adamw_seed42",
            report_path=ADAMW_REPORT,
            safe_reference=False,
            log_path=ADAMW_LOG,
        ),
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_stop_epoch(stop_reason: str | None) -> int | None:
    if not stop_reason:
        return None
    match = _STOP_EPOCH_RE.search(stop_reason)
    if match is None:
        return None
    return int(match.group("epoch"))


def _select_trial(data: dict[str, Any], *, safe_reference: bool) -> tuple[int | None, dict[str, Any]]:
    trials = data.get("trial_results")
    if not isinstance(trials, list):
        return None, {
            "passed": bool(data.get("all_passed", False)),
            "failure_modes": list((data.get("diagnostics") or {}).get("failure_modes", [])),
            "stop_reason": data.get("detail"),
            "controller_mode": data.get("controller_mode"),
            "optimizer_research_mode": data.get("optimizer_research_mode"),
            "benchmark_suite": data.get("benchmark_suite"),
        }
    if safe_reference:
        for idx, trial in enumerate(trials):
            if trial.get("passed") is True:
                return idx, dict(trial)
    if not trials:
        raise ValueError(f"No trial_results in {data!r}")
    return 0, dict(trials[0])


def _parse_epoch_log(log_path: Path | None) -> dict[int, dict[str, Any]]:
    if log_path is None or not log_path.exists():
        return {}
    epoch_rows: dict[int, dict[str, Any]] = {}
    current_epoch: int | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = payload.get("message")
        if not isinstance(message, str):
            continue
        online_eval = _ONLINE_EVAL_RE.search(message)
        if online_eval is not None:
            current_epoch = int(online_eval.group("epoch"))
            base_correct = int(online_eval.group("base_correct"))
            base_total = int(online_eval.group("base_total"))
            current_correct = int(online_eval.group("correct"))
            current_total = int(online_eval.group("total"))
            row = epoch_rows.setdefault(current_epoch, {"epoch": current_epoch})
            row.update(
                {
                    "online_eval_pre_n_correct": current_correct,
                    "online_eval_pre_n_total": current_total,
                    "online_eval_baseline_n_correct": base_correct,
                    "online_eval_baseline_n_total": base_total,
                    "online_eval_pre_degraded_raw": online_eval.group("raw") == "True",
                    "online_eval_pre_degraded_significant": online_eval.group("sig") == "True",
                    "online_eval_accuracy_delta": (
                        (current_correct / max(current_total, 1))
                        - (base_correct / max(base_total, 1))
                    ),
                }
            )
            continue

        if current_epoch is None:
            continue

        margin_match = _MARGIN_RE.search(message)
        if margin_match is not None:
            epoch_rows.setdefault(current_epoch, {"epoch": current_epoch}).update(
                {
                    "margin_median": float(margin_match.group("median")),
                    "margin_mean": float(margin_match.group("mean")),
                    "margin_min": float(margin_match.group("min")),
                    "margin_n_near_zero": int(margin_match.group("near_zero")),
                }
            )
            continue

        stable_rank_match = _STABLE_RANK_RE.search(message)
        if stable_rank_match is not None:
            epoch_rows.setdefault(current_epoch, {"epoch": current_epoch}).update(
                {
                    "stable_rank_median": float(stable_rank_match.group("median")),
                    "stable_rank_min": float(stable_rank_match.group("min")),
                    "adapter_rank": int(stable_rank_match.group("rank")),
                }
            )
    return epoch_rows


def _trial_summary_row(
    *,
    spec: ArtifactSpec,
    trial_index: int | None,
    trial: dict[str, Any],
) -> dict[str, Any]:
    margin_near_zero_delta = None
    if (
        trial.get("margin_n_near_zero_adapted") is not None
        and trial.get("margin_n_near_zero_baseline") is not None
    ):
        margin_near_zero_delta = (
            int(trial["margin_n_near_zero_adapted"])
            - int(trial["margin_n_near_zero_baseline"])
        )
    row = {
        "artifact_id": spec.artifact_id,
        "source_path": str(spec.report_path),
        "kind": "trial_summary",
        "safe_reference": spec.safe_reference,
        "trial_index": trial_index,
        "passed": bool(trial.get("passed", False)),
        "failure_modes": list(trial.get("failure_modes", [])),
        "stop_reason": trial.get("stop_reason"),
        "stop_epoch": _parse_stop_epoch(trial.get("stop_reason")),
        "benchmark_overall_delta": trial.get("benchmark_overall_delta"),
        "online_eval_delta_correct": trial.get("online_eval_delta_correct"),
        "margin_mean_delta": trial.get("margin_mean_delta"),
        "margin_n_near_zero_delta": margin_near_zero_delta,
        "margin_n_flipped_sign": trial.get("margin_n_flipped_sign"),
        "cka_blindness_ratio": trial.get("cka_blindness_ratio"),
        "eta_ceiling": trial.get("eta_ceiling"),
        "eta_sps": trial.get("eta_sps"),
        "eta_step": trial.get("eta_step"),
        "eta_weyl": trial.get("eta_weyl"),
        "max_effective_gain_ratio": trial.get("max_effective_gain_ratio"),
        "controller_mode": trial.get("controller_mode"),
        "optimizer_research_mode": trial.get("optimizer_research_mode"),
        "benchmark_suite": trial.get("benchmark_suite"),
    }
    return row


def _epoch_rows_for_artifact(
    *,
    spec: ArtifactSpec,
    log_epochs: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch, payload in sorted(log_epochs.items()):
        row = {
            "artifact_id": spec.artifact_id,
            "source_path": str(spec.log_path) if spec.log_path is not None else str(spec.report_path),
            "kind": "epoch",
            "safe_reference": spec.safe_reference,
            "epoch": epoch,
            **payload,
        }
        rows.append(row)
    return rows


def _build_law(specs: list[ArtifactSpec]) -> DerivedClosedLoopLaw:
    return DerivedClosedLoopLaw(
        source_artifacts=tuple(str(spec.report_path) for spec in specs),
        safe_artifacts=tuple(
            str(spec.report_path) for spec in specs if spec.safe_reference
        ),
        counterexample_artifacts=tuple(
            str(spec.report_path) for spec in specs if not spec.safe_reference
        ),
        arm_on_online_eval_accuracy_drop=False,
        arm_on_margin_trend_declining=True,
        arm_on_stable_rank_concentration=True,
        max_interventions=1,
        require_ordering_surface=True,
    )


def _arm_epoch_and_reasons(
    *,
    law: DerivedClosedLoopLaw,
    epoch_rows: list[dict[str, Any]],
) -> tuple[int | None, list[str]]:
    margin_history: list[float] = []
    stable_rank_history: list[float] = []
    for row in sorted(epoch_rows, key=lambda item: int(item["epoch"])):
        if row.get("margin_median") is not None:
            margin_history.append(float(row["margin_median"]))
        if row.get("stable_rank_median") is not None:
            stable_rank_history.append(float(row["stable_rank_median"]))
        behavioral_state = BehavioralStateMeasurement(
            online_eval_accuracy_delta=row.get("online_eval_accuracy_delta"),
            adapter_rank=row.get("adapter_rank"),
        )
        reasons = compute_closed_loop_trigger_reasons(
            law,
            behavioral_state=behavioral_state,
            margin_history=margin_history,
            stable_rank_history=stable_rank_history,
            loss_stability_window_epochs=2,
            adapter_rank=row.get("adapter_rank"),
        )
        row["arm_reasons"] = list(reasons)
        if reasons:
            return int(row["epoch"]), list(reasons)
    return None, []


def derive_control_law(
    specs: list[ArtifactSpec] | None = None,
) -> tuple[DerivedClosedLoopLaw, list[dict[str, Any]], dict[str, Any]]:
    artifact_specs = specs or _default_artifacts()
    law = _build_law(artifact_specs)
    state_table: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    for spec in artifact_specs:
        report = _load_json(spec.report_path)
        trial_index, trial = _select_trial(report, safe_reference=spec.safe_reference)
        summary_row = _trial_summary_row(spec=spec, trial_index=trial_index, trial=trial)
        state_table.append(summary_row)

        epoch_rows = _epoch_rows_for_artifact(
            spec=spec,
            log_epochs=_parse_epoch_log(spec.log_path),
        )
        arm_epoch, arm_reasons = _arm_epoch_and_reasons(
            law=law,
            epoch_rows=epoch_rows,
        )
        summary_row["arm_epoch_candidate"] = arm_epoch
        summary_row["arm_reasons"] = list(arm_reasons)
        if epoch_rows:
            state_table.extend(epoch_rows)

        stop_epoch = summary_row.get("stop_epoch")
        if spec.safe_reference:
            passed = arm_epoch is None
            validations.append(
                {
                    "artifact_id": spec.artifact_id,
                    "expectation": "quiet_on_safe_reference",
                    "passed": passed,
                    "arm_epoch_candidate": arm_epoch,
                }
            )
        elif spec.artifact_id == "behavioral_probe_cayley_seed42":
            passed = arm_epoch is not None and stop_epoch is not None and arm_epoch <= stop_epoch
            validations.append(
                {
                    "artifact_id": spec.artifact_id,
                    "expectation": "arm_at_or_before_geometric_stop",
                    "passed": passed,
                    "arm_epoch_candidate": arm_epoch,
                    "stop_epoch": stop_epoch,
                    "arm_reasons": list(arm_reasons),
                }
            )
        elif spec.artifact_id == "behavioral_probe_adamw_seed42":
            passed = arm_epoch is not None and arm_epoch <= 1
            validations.append(
                {
                    "artifact_id": spec.artifact_id,
                    "expectation": "arm_before_online_eval_degraded_significant",
                    "passed": passed,
                    "arm_epoch_candidate": arm_epoch,
                    "arm_reasons": list(arm_reasons),
                }
            )

    validation_summary = {
        "schema": "r2_control_law_validation_v1",
        "created_at_utc": _utc_now(),
        "law": law.to_dict(),
        "checks": validations,
        "all_passed": all(check["passed"] for check in validations),
    }
    return law, state_table, validation_summary


def build_falsifier_manifest(
    *,
    law_path: Path,
    state_table_path: Path,
    validation_path: Path,
    report_path: Path = DEFAULT_FALSIFIER_REPORT,
    artifact_root: Path = DEFAULT_FALSIFIER_ARTIFACT_ROOT,
) -> dict[str, Any]:
    return {
        "schema": "r2_closed_loop_falsifier_manifest_v1",
        "created_at_utc": _utc_now(),
        "modelcypher_commit": _git_commit(),
        "law_path": str(law_path),
        "state_table_path": str(state_table_path),
        "validation_path": str(validation_path),
        "frozen_tuple": {
            "model_path": str(FROZEN_MODEL_PATH),
            "train_dataset_path": str(FROZEN_TRAIN_PATH),
            "eval_dataset_path": str(FROZEN_EVAL_PATH),
            "seed": 42,
            "trials": 1,
            "benchmark_suite": "quick",
            "controller_mode": CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
            "optimizer_research_mode": OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
        },
        "artifacts": {
            "report_path": str(report_path),
            "artifact_root": str(artifact_root),
            "ledger_path": str(DEFAULT_LEDGER_PATH),
        },
        "acceptance": [
            "controller arms before prior failure point",
            "behavior is preserved or improved versus matched failing arm",
            "structural gates do not regress",
            "otherwise classify mechanism as MECHANISM_UNDERSPECIFIED",
        ],
        "command": (
            "poetry run python scripts/derive_r2_control_law.py "
            "--run-falsifier"
        ),
    }


def _active_gpu_processes() -> list[str]:
    """Return PIDs of Python/MLX processes likely using the GPU.

    Filters out VS Code extensions, uvicorn/web servers, multiprocessing
    resource trackers, and the current process.
    """
    result = subprocess.run(
        ["zsh", "-lc", "pgrep -lf 'python|mlx' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    current_pid = str(os.getpid())
    _SAFE_PATTERNS = (
        "vscode",
        "pylance",
        "pet server",
        "Code Helper",
        "uvicorn",
        "resource_tracker",
        "multiprocessing",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    filtered = []
    for line in lines:
        if line.startswith(f"{current_pid} "):
            continue
        if any(pat in line for pat in _SAFE_PATTERNS):
            continue
        filtered.append(line)
    return filtered


def _append_ledger_row(
    *,
    ledger_path: Path,
    run_id: str,
    status: str,
    claim: str,
    command: str,
    primary_observable: str,
    artifact_dir: str,
    next_falsifier: str,
) -> None:
    row = "\t".join(
        [
            run_id,
            _utc_now(),
            _git_commit(),
            status,
            claim,
            "derived_control_law(local_350M_seed42)",
            (
                f"model={FROZEN_MODEL_PATH};train={FROZEN_TRAIN_PATH};"
                f"val={FROZEN_EVAL_PATH};precision=bf16;seed=42;trials=1;benchmark=quick"
            ),
            command,
            primary_observable,
            artifact_dir,
            next_falsifier,
        ]
    )
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def run_falsifier(
    *,
    law: DerivedClosedLoopLaw,
    manifest: dict[str, Any],
    report_path: Path = DEFAULT_FALSIFIER_REPORT,
    artifact_root: Path = DEFAULT_FALSIFIER_ARTIFACT_ROOT,
) -> dict[str, Any]:
    active = _active_gpu_processes()
    if active:
        raise RuntimeError(
            "Refusing to run the closed-loop falsifier while other python/mlx "
            f"processes are active: {active}"
        )

    from modelcypher.cli.composition import get_backend, get_dataset_training_service
    from modelcypher.core.use_cases.derived_training_validation_service import (
        DerivedTrainingValidationService,
    )

    validator = DerivedTrainingValidationService(
        dataset_training_service=get_dataset_training_service(),
        backend=get_backend(),
    )
    try:
        result = validator.validate(
            model_path=FROZEN_MODEL_PATH,
            dataset_path=FROZEN_TRAIN_PATH,
            eval_dataset_path=FROZEN_EVAL_PATH,
            trials=1,
            base_seed=42,
            benchmark_suite="quick",
            controller_mode=CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
            optimizer_research_mode=OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
            controller_law=law,
            enable_phase5_inference=True,
            phase5_probe_count=20,
            artifact_root=artifact_root,
        )
        payload = result.to_dict()
        status = "completed" if result.all_passed else "counterexample"
    except TrainingDerivationError as exc:
        payload = {
            "all_passed": False,
            "failure_class": exc.failure_class,
            "detail": exc.detail,
            "diagnostics": exc.diagnostics,
            "controller_mode": CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
            "optimizer_research_mode": OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
            "benchmark_suite": "quick",
            "model_path": str(FROZEN_MODEL_PATH),
            "dataset_path": str(FROZEN_TRAIN_PATH),
            "eval_dataset_path": str(FROZEN_EVAL_PATH),
            "trials_requested": 1,
        }
        status = "mechanism_underspecified"

    _write_json(report_path, payload)
    _append_ledger_row(
        ledger_path=DEFAULT_LEDGER_PATH,
        run_id="r2_closed_loop_cayley_seed42",
        status=status,
        claim=(
            "An offline-derived closed-loop layer freeze can move the frozen-tuple "
            "failure boundary later than the matched Cayley counterexample."
        ),
        command=manifest["command"],
        primary_observable=json.dumps(
            {
                "all_passed": payload.get("all_passed"),
                "failure_class": payload.get("failure_class"),
                "detail": payload.get("detail"),
            },
            sort_keys=True,
        ),
        artifact_dir=str(RESULTS_ROOT),
        next_falsifier=(
            "If the closed-loop arm did not move degradation later, classify "
            "MECHANISM_UNDERSPECIFIED and derive the next layer-local operator."
        ),
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the R2 closed-loop law from retained 350M artifacts.",
    )
    parser.add_argument(
        "--state-table-path",
        type=Path,
        default=DEFAULT_STATE_TABLE,
        help="Output path for the normalized artifact state table JSON.",
    )
    parser.add_argument(
        "--law-path",
        type=Path,
        default=DEFAULT_LAW_PATH,
        help="Output path for the derived closed-loop law JSON.",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Output path for the derivation validation JSON.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the falsifier manifest JSON.",
    )
    parser.add_argument(
        "--run-falsifier",
        action="store_true",
        help="After derivation, run the single frozen-tuple closed-loop falsifier.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    law, state_table, validation = derive_control_law()
    manifest = build_falsifier_manifest(
        law_path=args.law_path,
        state_table_path=args.state_table_path,
        validation_path=args.validation_path,
    )
    _write_json(args.state_table_path, state_table)
    _write_json(args.law_path, law.to_dict())
    _write_json(args.validation_path, validation)
    _write_json(args.manifest_path, manifest)

    if args.run_falsifier:
        payload = run_falsifier(law=law, manifest=manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("all_passed") else 4

    print(
        json.dumps(
            {
                "law_path": str(args.law_path),
                "state_table_path": str(args.state_table_path),
                "validation_path": str(args.validation_path),
                "manifest_path": str(args.manifest_path),
                "all_passed": validation["all_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
