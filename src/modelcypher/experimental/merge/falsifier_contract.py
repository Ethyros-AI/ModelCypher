# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Research-only artifact contract for merge portability falsifier runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MERGE_PORTABILITY_SCHEMA = "mc.merge.portability.falsifier.v1"
MERGE_PORTABILITY_SUMMARY_SCHEMA = "mc.merge.portability.summary.v1"
REQUIRED_BUNDLE_FILES = ("REPORT.md", "summary.json", "manifest.json", "ledger.jsonl")
REQUIRED_CLAIM_FORM_FIELDS = (
    "claim_id",
    "status",
    "mechanism",
    "equation_or_theorem",
    "precision_state",
    "measurement_operator",
    "commensurability_proof",
    "directional_prediction",
    "falsifier",
)
DEFAULT_ARM_SPECS = (
    {"arm_id": "target_baseline", "category": "baseline", "applicability": "always"},
    {"arm_id": "source_baseline", "category": "baseline", "applicability": "always"},
    {
        "arm_id": "aligned_direct_delta",
        "category": "ablation",
        "applicability": "always",
    },
    {
        "arm_id": "null_space_projector_mp_tikhonov",
        "category": "projector",
        "applicability": "always",
    },
    {
        "arm_id": "behavior_jacobian_projector",
        "category": "projector",
        "applicability": "always",
    },
    {
        "arm_id": "task_arithmetic",
        "category": "baseline",
        "applicability": "same_family_only",
    },
    {"arm_id": "slerp", "category": "baseline", "applicability": "same_family_only"},
    {"arm_id": "ties", "category": "baseline", "applicability": "same_family_only"},
    {
        "arm_id": "procrustes_only_stitch",
        "category": "ablation",
        "applicability": "always",
    },
    {
        "arm_id": "stitch_plus_projector",
        "category": "ablation",
        "applicability": "always",
    },
)
DEFAULT_MEASUREMENTS = (
    "held_out_behavior_accuracy",
    "degeneration_repetition",
    "held_out_probe_cka",
    "mode_connectivity_barrier",
    "quantized_behavior_delta_vs_bf16",
    "projector_telemetry",
)
DEFAULT_DECISION_RULE = (
    "Promote only if projector-based merge beats applicable baselines on preserved "
    "behavior without violating degeneration or quantized-retention controls. "
    "Otherwise classify failure as coordinate-resolution failure, "
    "projector/operator failure, or measurement invalidity."
)
PROMOTABLE_STATUSES = {"candidate", "validated"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_claim_form() -> dict[str, Any]:
    return {
        "claim_id": "CR-MRG-PORTABILITY-SCAFFOLD",
        "status": "[EXPLORATORY]",
        "mechanism": (
            "Coordinate resolution F maps source structure into target coordinates, "
            "then the preservation projector constrains the additive delta to "
            "directions the target does not already occupy."
        ),
        "equation_or_theorem": (
            "Same-dimensional: W' = W_t + (W_s_aligned - W_t) P_null. "
            "Cross-dimensional: reconstruct aligned target-coordinate behavior "
            "before applying the same preservation operator."
        ),
        "precision_state": (
            "bf16 reference plus matched quantized comparison arm under a frozen "
            "measurement operator."
        ),
        "measurement_operator": (
            "observable = f(geometry_state, architecture_state, scale_state, "
            "precision_state, measurement_operator)"
        ),
        "commensurability_proof": (
            "Held-out behavior, degeneration, CKA, and quantized retention are all "
            "measured under one frozen evaluator bundle and one held-out probe set."
        ),
        "directional_prediction": (
            "If the projector geometry is sufficient, projector arms preserve target "
            "behavior better than direct delta and same-family merge baselines."
        ),
        "falsifier": (
            "If the projector arms do not beat applicable baselines under the frozen "
            "evaluator, the portability claim remains open or is reclassified."
        ),
    }


def build_merge_portability_manifest(
    *,
    run_id: str,
    output_dir: str | Path,
    claim_form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    claim = dict(build_default_claim_form())
    if claim_form:
        claim.update(claim_form)

    return {
        "_schema": MERGE_PORTABILITY_SCHEMA,
        "run_id": run_id,
        "created_at": _utc_now(),
        "status": "exploratory",
        "roadmap_item": "R5",
        "open_question": "Q8",
        "output_dir": str(Path(output_dir)),
        "frozen_contract": {
            "evaluator_bundle": {"name": "merge_portability_evaluator_v1", "frozen": True},
            "held_out_probe_set": {"name": "merge_portability_probes_v1", "frozen": True},
            "comparison_budget": {"name": "merge_portability_budget_v1", "frozen": True},
            "quantization_policy": {
                "name": "bf16_vs_quantized_v1",
                "reference_precision": "bf16",
                "comparison_precision": "quantized",
                "frozen": True,
            },
        },
        "arms": list(DEFAULT_ARM_SPECS),
        "measurements": list(DEFAULT_MEASUREMENTS),
        "decision_rule": DEFAULT_DECISION_RULE,
        "claim_form": claim,
        "artifact_contract": {
            "required_files": list(REQUIRED_BUNDLE_FILES),
            "append_only_ledger": "ledger.jsonl",
        },
    }


def build_merge_portability_summary(
    *,
    run_id: str,
    promotion_status: str = "exploratory",
) -> dict[str, Any]:
    return {
        "_schema": MERGE_PORTABILITY_SUMMARY_SCHEMA,
        "run_id": run_id,
        "created_at": _utc_now(),
        "promotion_status": promotion_status,
        "roadmap_item": "R5",
        "open_question": "Q8",
        "decision_rule": DEFAULT_DECISION_RULE,
        "result": "pending_measurement",
    }


def emit_merge_portability_bundle(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_doc = summary or build_merge_portability_summary(
        run_id=manifest["run_id"],
    )
    report_lines = [
        "# Merge Portability Falsifier Scaffold",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        "- Scope: `R5 / Q8` only",
        "- Status: exploratory scaffold",
        "",
        "## Frozen Contract",
        "",
        "- Evaluator bundle, held-out probe set, comparison budget, and quantization policy are frozen in `manifest.json`.",
        "- This scaffold is not a certificate and does not promote merge claims on its own.",
        "",
        "## Decision Rule",
        "",
        DEFAULT_DECISION_RULE,
        "",
    ]

    (out_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary_doc, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    ledger_entry = {
        "timestamp": _utc_now(),
        "event": "bundle_initialized",
        "run_id": manifest["run_id"],
        "status": summary_doc["promotion_status"],
    }
    (out_dir / "ledger.jsonl").write_text(
        json.dumps(ledger_entry, default=str) + "\n",
        encoding="utf-8",
    )
    return out_dir


def validate_merge_portability_bundle(output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    files_checked: list[str] = []

    for required_name in REQUIRED_BUNDLE_FILES:
        file_path = out_dir / required_name
        files_checked.append(required_name)
        if not file_path.exists():
            errors.append(f"missing required artifact: {required_name}")

    manifest_path = out_dir / "manifest.json"
    summary_path = out_dir / "summary.json"
    if errors:
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "files_checked": files_checked,
            "run_dir": str(out_dir),
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if manifest.get("_schema") != MERGE_PORTABILITY_SCHEMA:
        errors.append("manifest.json: unexpected schema")

    claim_form = manifest.get("claim_form") or {}
    missing_claim_fields = [
        field_name
        for field_name in REQUIRED_CLAIM_FORM_FIELDS
        if not claim_form.get(field_name)
    ]
    if missing_claim_fields:
        errors.append(
            "manifest.json: missing claim-form fields: "
            + ", ".join(sorted(missing_claim_fields))
        )

    frozen_contract = manifest.get("frozen_contract") or {}
    for contract_name in (
        "evaluator_bundle",
        "held_out_probe_set",
        "comparison_budget",
        "quantization_policy",
    ):
        contract_entry = frozen_contract.get(contract_name) or {}
        if contract_entry.get("frozen") is not True:
            errors.append(f"manifest.json: {contract_name} must be frozen")

    arm_ids = [arm.get("arm_id") for arm in manifest.get("arms") or []]
    expected_arm_ids = [arm["arm_id"] for arm in DEFAULT_ARM_SPECS]
    if arm_ids != expected_arm_ids:
        errors.append("manifest.json: arm set does not match required baseline order")

    measurements = manifest.get("measurements") or []
    if measurements != list(DEFAULT_MEASUREMENTS):
        errors.append("manifest.json: measurements do not match required contract")

    promotion_status = str(summary.get("promotion_status", "")).lower()
    if promotion_status in PROMOTABLE_STATUSES and not claim_form.get(
        "commensurability_proof"
    ):
        errors.append(
            "summary.json: promotable result requires a non-empty commensurability proof"
        )

    if summary.get("run_id") != manifest.get("run_id"):
        errors.append("run_id mismatch between manifest.json and summary.json")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "files_checked": files_checked,
        "run_dir": str(out_dir),
    }

