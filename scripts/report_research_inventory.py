#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate a reproducible research inventory for scripts, results, and claims.

This report is intentionally operational, not doctrinal. It tracks repo-observable
signals that help answer:

1. Which scripts/results are currently authoritative (`canonical`)?
2. Which should collapse to retained summaries instead of live code/raw runs (`summary_only`)?
3. Which have no current signal and should be deleted (`delete`)?

Status is derived from active docs, tested entry points, claim registries, and
artifact layout. It is not a scientific evidence label.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "repo_research_inventory"

ACTIVE_DOCS = (
    ROOT / "docs" / "VISION.md",
    ROOT / "docs" / "MISSION.md",
    ROOT / "docs" / "RESEARCH-ROADMAP.md",
    ROOT / "docs" / "research" / "OPEN-MATHEMATICAL-QUESTIONS.md",
    ROOT / "docs" / "research" / "SOTA-AUDIT-2026-03.md",
    ROOT / "docs" / "research" / "PRODUCT-MAINTENANCE-AUDIT-2026-03.md",
)

SCRIPT_REFERENCE_RE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)")
RESULT_REFERENCE_RE = re.compile(r"results/[A-Za-z0-9_./-]+")
RESULT_BINARY_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".npz",
    ".npy",
}
SUMMARY_FILE_NAMES = (
    "REPORT.md",
    "analysis_summary.json",
    "verdict.json",
    "multiseed_gates.json",
    "summary.json",
    "results.json",
)
SUMMARY_JSON_KEYWORDS = (
    "summary",
    "report",
    "verdict",
    "gates",
    "results",
    "comparison",
    "metrics",
    "scorecard",
    "correction",
    "survey",
    "validation",
)
RESULT_STATUS_VALUES = {"canonical", "summary_only", "delete"}

# These are current operational utilities rather than experiment leaves.
SCRIPT_STATUS_OVERRIDES = {
    "scripts/report_doctrine_audit.py": "canonical",
    "scripts/report_token_budget.py": "canonical",
    "scripts/report_research_inventory.py": "canonical",
}

# Some scripts intentionally write into differently named result families.
SCRIPT_ARTIFACT_OVERRIDES = {
    "scripts/estimate_bl_jacobian.py": ("results/bl_estimation",),
    "scripts/g5_build_non_ceiling_eval_set.py": ("results/g5_8b_validation",),
    "scripts/pipeline_validation.py": ("results/pipeline_validation",),
    "scripts/reinforce_revalidation.py": ("results/reinforce_frontier_1p2b",),
    "scripts/validate_gqa_falsifier_artifacts.py": ("results/gqa_falsifier_protocol",),
    "scripts/weyl_quantization_validation.py": ("results/weyl_quantization_validation",),
}


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    track: str
    current_status: str
    classification: str
    statement: str
    operator: str
    architecture_terms: str
    scale_terms: str
    falsifier: str
    artifact_paths: tuple[str, ...]
    result_families: tuple[str, ...]
    recommended_next_step: str
    integration_target_path: str | None


@dataclass(frozen=True)
class ScriptRecord:
    path: str
    status: str
    evidence_status: str
    direct_doc_refs: tuple[str, ...]
    exact_test_matches: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ResultRecord:
    family: str
    status: str
    evidence_status: str
    artifact_path: str | None
    size_bytes: int
    file_count: int
    immediate_subdir_count: int
    binary_file_count: int
    doc_refs: tuple[str, ...]
    claim_ids: tuple[str, ...]
    notes: tuple[str, ...]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _dedupe_ordered(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _collect_active_doc_refs() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    script_refs: dict[str, list[str]] = defaultdict(list)
    result_refs: dict[str, list[str]] = defaultdict(list)
    for doc_path in ACTIVE_DOCS:
        if not doc_path.exists():
            continue
        doc_rel = _rel(doc_path)
        content = _read_text(doc_path)
        for match in SCRIPT_REFERENCE_RE.findall(content):
            script_refs[match].append(doc_rel)
        for match in RESULT_REFERENCE_RE.findall(content):
            family = _extract_result_family(match)
            if family is not None:
                result_refs[family].append(doc_rel)
    return (
        {path: _dedupe_ordered(refs) for path, refs in script_refs.items()},
        {family: _dedupe_ordered(refs) for family, refs in result_refs.items()},
    )


def _extract_result_family(path_str: str) -> str | None:
    parts = Path(path_str).parts
    if len(parts) < 2 or parts[0] != "results":
        return None
    return parts[1]


def _load_claim_registry() -> list[ClaimRecord]:
    internal_path = ROOT / "results" / "sota_audit_2026_03" / "internal_claim_registry.json"
    crosswalk_path = ROOT / "results" / "sota_audit_2026_03" / "claim_crosswalk.json"
    if not internal_path.exists() or not crosswalk_path.exists():
        return []

    internal_claims = json.loads(_read_text(internal_path))["claims"]
    crosswalk_records = {
        record["claim_id"]: record
        for record in json.loads(_read_text(crosswalk_path))["records"]
    }
    claims: list[ClaimRecord] = []
    for internal in internal_claims:
        crosswalk = crosswalk_records.get(internal["claim_id"], {})
        artifact_paths = _dedupe_ordered(
            [
                internal.get("latest_results_path", ""),
                *crosswalk.get("internal_evidence_pointers", []),
            ]
        )
        result_families = _dedupe_ordered(
            [
                family
                for family in (_extract_result_family(path) for path in artifact_paths)
                if family is not None
            ]
        )
        claims.append(
            ClaimRecord(
                claim_id=internal["claim_id"],
                track=internal.get("track", ""),
                current_status=internal.get("current_status", ""),
                classification=crosswalk.get("classification", ""),
                statement=internal.get("statement", ""),
                operator=internal.get("operator", ""),
                architecture_terms=internal.get("architecture_terms", ""),
                scale_terms=internal.get("scale_terms", ""),
                falsifier=internal.get("falsifier", ""),
                artifact_paths=artifact_paths,
                result_families=result_families,
                recommended_next_step=crosswalk.get("recommended_next_step", ""),
                integration_target_path=crosswalk.get("integration_target_path"),
            )
        )
    return claims


def _collect_exact_test_matches(script_paths: list[Path]) -> dict[str, tuple[str, ...]]:
    matches: dict[str, list[str]] = defaultdict(list)
    scripts_by_stem: dict[str, list[Path]] = defaultdict(list)
    for script_path in script_paths:
        scripts_by_stem[script_path.stem].append(script_path)

    for test_path in sorted((ROOT / "tests").rglob("test_*.py")):
        stem = test_path.stem.removeprefix("test_")
        for script_path in scripts_by_stem.get(stem, []):
            matches[_rel(script_path)].append(_rel(test_path))
    return {path: _dedupe_ordered(refs) for path, refs in matches.items()}


def derive_script_status(
    *,
    script_rel: str,
    direct_doc_ref_count: int,
    exact_test_match_count: int,
    artifact_path_count: int,
) -> str:
    override = SCRIPT_STATUS_OVERRIDES.get(script_rel)
    if override is not None:
        return override
    if direct_doc_ref_count > 0 and exact_test_match_count > 0:
        return "canonical"
    if direct_doc_ref_count > 0 or exact_test_match_count > 0 or artifact_path_count > 0:
        return "summary_only"
    return "delete"


def derive_script_evidence_status(
    *,
    direct_doc_ref_count: int,
    exact_test_match_count: int,
    artifact_path_count: int,
) -> str:
    if exact_test_match_count > 0 and artifact_path_count > 0:
        return "tested+artifact"
    if exact_test_match_count > 0 and direct_doc_ref_count > 0:
        return "tested+docs"
    if exact_test_match_count > 0:
        return "tested"
    if artifact_path_count > 0 and direct_doc_ref_count > 0:
        return "artifact+docs"
    if artifact_path_count > 0:
        return "artifact"
    if direct_doc_ref_count > 0:
        return "docs"
    return "unlinked"


def derive_result_status(
    *,
    has_report: bool,
    doc_ref_count: int,
    claim_ref_count: int,
    file_count: int,
    immediate_subdir_count: int,
) -> str:
    if has_report or doc_ref_count > 0 or claim_ref_count > 0:
        return "canonical"
    if file_count > 0 or immediate_subdir_count > 0:
        return "summary_only"
    return "delete"


def derive_result_evidence_status(
    *,
    has_report: bool,
    doc_ref_count: int,
    claim_ref_count: int,
    file_count: int,
) -> str:
    if claim_ref_count > 0 and doc_ref_count > 0:
        return "claims+docs"
    if claim_ref_count > 0:
        return "claims"
    if has_report and doc_ref_count > 0:
        return "report+docs"
    if has_report:
        return "report"
    if doc_ref_count > 0:
        return "docs"
    if file_count > 0:
        return "files_only"
    return "empty"


def _load_summary_status_override(family_path: Path) -> str | None:
    summary_path = family_path / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(_read_text(summary_path))
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    if isinstance(status, str) and status in RESULT_STATUS_VALUES:
        return status
    return None


def _resolve_script_artifact_paths(
    script_path: Path,
    claim_artifacts_by_script: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    script_rel = _rel(script_path)
    candidates: list[str] = []
    overridden = SCRIPT_ARTIFACT_OVERRIDES.get(script_rel, ())
    for candidate in overridden:
        if (ROOT / candidate).exists():
            candidates.append(candidate)

    exact_family = ROOT / "results" / script_path.stem
    if exact_family.is_dir():
        candidates.append(_rel(exact_family))

    candidates.extend(claim_artifacts_by_script.get(script_rel, ()))

    return _dedupe_ordered(candidates)


def _build_script_registry(
    active_doc_refs: dict[str, tuple[str, ...]],
    claims: list[ClaimRecord],
) -> list[ScriptRecord]:
    script_paths = sorted(
        path
        for path in (ROOT / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    exact_test_matches = _collect_exact_test_matches(script_paths)
    claim_artifacts_by_script: dict[str, tuple[str, ...]] = defaultdict(tuple)
    claim_artifact_lists: dict[str, list[str]] = defaultdict(list)
    for claim in claims:
        if claim.integration_target_path is None:
            continue
        if not claim.integration_target_path.startswith("scripts/"):
            continue
        for artifact_path in claim.artifact_paths:
            if artifact_path.startswith("results/"):
                claim_artifact_lists[claim.integration_target_path].append(artifact_path)
    claim_artifacts_by_script = {
        path: _dedupe_ordered(artifact_paths)
        for path, artifact_paths in claim_artifact_lists.items()
    }
    records: list[ScriptRecord] = []

    for script_path in script_paths:
        script_rel = _rel(script_path)
        artifact_paths = _resolve_script_artifact_paths(script_path, claim_artifacts_by_script)
        doc_refs = active_doc_refs.get(script_rel, ())
        tests = exact_test_matches.get(script_rel, ())
        notes: list[str] = []
        if script_rel in SCRIPT_STATUS_OVERRIDES:
            notes.append("status override: operational reporting utility")
        if script_rel in SCRIPT_ARTIFACT_OVERRIDES:
            notes.append("artifact override: script writes into differently named result family")
        records.append(
            ScriptRecord(
                path=script_rel,
                status=derive_script_status(
                    script_rel=script_rel,
                    direct_doc_ref_count=len(doc_refs),
                    exact_test_match_count=len(tests),
                    artifact_path_count=len(artifact_paths),
                ),
                evidence_status=derive_script_evidence_status(
                    direct_doc_ref_count=len(doc_refs),
                    exact_test_match_count=len(tests),
                    artifact_path_count=len(artifact_paths),
                ),
                direct_doc_refs=doc_refs,
                exact_test_matches=tests,
                artifact_paths=artifact_paths,
                notes=tuple(notes),
            )
        )
    return records


def _select_summary_artifact(family_path: Path) -> str | None:
    for file_name in SUMMARY_FILE_NAMES:
        candidate = family_path / file_name
        if candidate.exists():
            return _rel(candidate)

    json_candidates = sorted(
        path
        for path in family_path.rglob("*.json")
        if len(path.relative_to(family_path).parts) <= 3
        and any(keyword in path.stem.lower() for keyword in SUMMARY_JSON_KEYWORDS)
    )
    if json_candidates:
        return _rel(json_candidates[0])

    markdown_candidates = sorted(
        path
        for path in family_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".txt"}
        and len(path.relative_to(family_path).parts) <= 3
    )
    if markdown_candidates:
        return _rel(markdown_candidates[0])
    return None


def _family_stats(family_path: Path) -> tuple[int, int, int, int]:
    size_bytes = 0
    file_count = 0
    binary_file_count = 0
    for root, _, files in os.walk(family_path):
        root_path = Path(root)
        for file_name in files:
            file_path = root_path / file_name
            try:
                size_bytes += file_path.stat().st_size
            except FileNotFoundError:
                continue
            file_count += 1
            if file_path.suffix.lower() in RESULT_BINARY_EXTENSIONS:
                binary_file_count += 1
    immediate_subdir_count = sum(1 for child in family_path.iterdir() if child.is_dir())
    return size_bytes, file_count, immediate_subdir_count, binary_file_count


def _build_result_registry(
    active_result_refs: dict[str, tuple[str, ...]],
    claims: list[ClaimRecord],
    output_dir: Path,
) -> list[ResultRecord]:
    claim_ids_by_family: dict[str, list[str]] = defaultdict(list)
    for claim in claims:
        for family in claim.result_families:
            claim_ids_by_family[family].append(claim.claim_id)

    records: list[ResultRecord] = []
    for family_path in sorted((ROOT / "results").iterdir()):
        if not family_path.is_dir():
            continue
        if family_path == output_dir:
            continue

        family_name = family_path.name
        size_bytes, file_count, immediate_subdir_count, binary_file_count = _family_stats(
            family_path
        )
        claim_ids = _dedupe_ordered(claim_ids_by_family.get(family_name, []))
        doc_refs = active_result_refs.get(family_name, ())
        has_report = (family_path / "REPORT.md").exists()
        summary_status_override = _load_summary_status_override(family_path)
        notes: list[str] = []
        if has_report:
            notes.append("contains family-level REPORT.md")
        if summary_status_override is not None:
            notes.append(f"summary.json status override: {summary_status_override}")
        if immediate_subdir_count > 1:
            notes.append(f"{immediate_subdir_count} immediate subdirectories")
        if binary_file_count > 0:
            notes.append(f"{binary_file_count} binary artifact files")

        status = summary_status_override or derive_result_status(
            has_report=has_report,
            doc_ref_count=len(doc_refs),
            claim_ref_count=len(claim_ids),
            file_count=file_count,
            immediate_subdir_count=immediate_subdir_count,
        )
        records.append(
            ResultRecord(
                family=family_name,
                status=status,
                evidence_status=derive_result_evidence_status(
                    has_report=has_report,
                    doc_ref_count=len(doc_refs),
                    claim_ref_count=len(claim_ids),
                    file_count=file_count,
                ),
                artifact_path=_select_summary_artifact(family_path),
                size_bytes=size_bytes,
                file_count=file_count,
                immediate_subdir_count=immediate_subdir_count,
                binary_file_count=binary_file_count,
                doc_refs=doc_refs,
                claim_ids=claim_ids,
                notes=tuple(notes),
            )
        )
    return records


def _render_scripts_inventory(
    script_records: list[ScriptRecord],
    output_dir_rel: str,
) -> str:
    status_counts = Counter(record.status for record in script_records)
    canonical_records = [record for record in script_records if record.status == "canonical"]
    summary_only_records = [
        record for record in script_records if record.status == "summary_only"
    ]
    delete_records = [record for record in script_records if record.status == "delete"]

    lines = [
        "# Scripts Inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This file is generated by `poetry run python scripts/report_research_inventory.py`.",
        "",
        "Status meanings:",
        "- `canonical`: current operational script with direct doc/test signals",
        "- `summary_only`: retain the summary result or claim pointer, not the live script",
        "- `delete`: no current repo signal beyond the file itself",
        "",
        f"Full registries live in `{output_dir_rel}/`.",
        "",
        f"- `canonical`: {status_counts.get('canonical', 0)}",
        f"- `summary_only`: {status_counts.get('summary_only', 0)}",
        f"- `delete`: {status_counts.get('delete', 0)}",
        "",
        "## Canonical Scripts",
        "",
        "| Script | Evidence | Artifact Path | Tests | Docs |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for record in canonical_records:
        artifact = record.artifact_paths[0] if record.artifact_paths else "—"
        lines.append(
            f"| `{record.path}` | `{record.evidence_status}` | `{artifact}` | "
            f"{len(record.exact_test_matches)} | {len(record.direct_doc_refs)} |"
        )

    lines.extend(
        [
            "",
            "## Summary-Only Scripts",
            "",
            "| Script | Evidence | Artifact Path | Notes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in summary_only_records[:25]:
        artifact = record.artifact_paths[0] if record.artifact_paths else "—"
        note = (
            "; ".join(record.notes)
            if record.notes
            else "linked through docs/tests/artifacts, but retain only the summary result"
        )
        lines.append(
            f"| `{record.path}` | `{record.evidence_status}` | `{artifact}` | {note} |"
        )

    lines.extend(
        [
            "",
            "## Delete Candidates",
            "",
            "| Script | Evidence | Notes |",
            "| --- | --- | --- |",
        ]
    )
    for record in delete_records[:25]:
        note = "; ".join(record.notes) if record.notes else "no active doc/test/artifact signal"
        lines.append(f"| `{record.path}` | `{record.evidence_status}` | {note} |")

    lines.extend(
        [
            "",
            "## Registry Files",
            "",
            f"- `{output_dir_rel}/scripts_registry.json`",
            f"- `{output_dir_rel}/results_registry.json`",
            f"- `{output_dir_rel}/claim_registry.json`",
            f"- `{output_dir_rel}/retention_plan.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f}G"


def _render_readme(
    script_records: list[ScriptRecord],
    result_records: list[ResultRecord],
    claim_records: list[ClaimRecord],
) -> str:
    script_counts = Counter(record.status for record in script_records)
    result_counts = Counter(record.status for record in result_records)
    total_result_bytes = sum(record.size_bytes for record in result_records)

    lines = [
        "# Repo Research Inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This directory contains generated inventories for the current research surface.",
        "",
        "## Counts",
        "",
        f"- scripts: `{len(script_records)}`",
        f"- claims: `{len(claim_records)}`",
        f"- result families: `{len(result_records)}`",
        f"- results size: `{_format_gib(total_result_bytes)}`",
        "",
        "### Script status",
        "",
        f"- `canonical`: `{script_counts.get('canonical', 0)}`",
        f"- `summary_only`: `{script_counts.get('summary_only', 0)}`",
        f"- `delete`: `{script_counts.get('delete', 0)}`",
        "",
        "### Result status",
        "",
        f"- `canonical`: `{result_counts.get('canonical', 0)}`",
        f"- `summary_only`: `{result_counts.get('summary_only', 0)}`",
        f"- `delete`: `{result_counts.get('delete', 0)}`",
        "",
        "## Files",
        "",
        "- `scripts_registry.json`: per-script maintenance recommendation",
        "- `results_registry.json`: top-level result-family maintenance recommendation",
        "- `claim_registry.json`: merged internal claim registry + SOTA crosswalk",
        "- `retention_plan.md`: summary-retention and deletion actions for high-cost result families",
        "",
    ]
    return "\n".join(lines) + "\n"


def _retention_action(record: ResultRecord) -> str:
    if record.status == "delete":
        if record.file_count == 0 and record.immediate_subdir_count == 0:
            return "Delete the empty family from the worktree."
        return "Delete the family after confirming no retained summary bundle is required."
    if record.status == "summary_only":
        if record.immediate_subdir_count > 1:
            return (
                f"Retain one summary bundle, then delete the repeated raw runs from the "
                f"worktree; raw run count is {record.immediate_subdir_count}."
            )
        if record.artifact_path is None:
            return "Extract one summary bundle, then delete the remaining raw files."
        return "Retain the selected summary pointer and delete the remaining raw files."
    if record.status == "canonical" and record.immediate_subdir_count > 1:
        return (
            "Retain the family as the canonical evidence bucket, but delete repeated raw runs "
            "after retaining one summary bundle and any genuinely reusable artifact."
        )
    return "Retain in worktree for now."


def _render_retention_plan(result_records: list[ResultRecord]) -> str:
    total_result_bytes = sum(record.size_bytes for record in result_records)
    sorted_records = sorted(result_records, key=lambda record: record.size_bytes, reverse=True)
    lines = [
        "# Result Retention Plan",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This plan is worktree-focused. `status` is the recommended retention policy for",
        "the top-level result family, not a scientific evidence tag.",
        "",
        "| Family | Status | Size | Share | Subdirs | Binary Files | Summary Pointer |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in sorted_records[:15]:
        share = 0.0 if total_result_bytes == 0 else 100.0 * record.size_bytes / total_result_bytes
        artifact = record.artifact_path or "—"
        lines.append(
            f"| `{record.family}` | `{record.status}` | `{_format_gib(record.size_bytes)}` | "
            f"{share:.1f}% | {record.immediate_subdir_count} | {record.binary_file_count} | "
            f"`{artifact}` |"
        )

    lines.extend(["", "## Immediate Actions", ""])
    for record in sorted_records[:10]:
        share = 0.0 if total_result_bytes == 0 else 100.0 * record.size_bytes / total_result_bytes
        lines.append(
            f"- `{record.family}` (`{_format_gib(record.size_bytes)}`, {share:.1f}% of `results/`): "
            f"{_retention_action(record)}"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def generate_inventory(output_dir: Path = OUTPUT_DIR, *, write_scripts_inventory: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_script_refs, active_result_refs = _collect_active_doc_refs()
    claims = _load_claim_registry()
    scripts = _build_script_registry(active_script_refs, claims)
    results = _build_result_registry(active_result_refs, claims, output_dir)

    _write_json(output_dir / "claim_registry.json", [asdict(record) for record in claims])
    _write_json(output_dir / "scripts_registry.json", [asdict(record) for record in scripts])
    _write_json(output_dir / "results_registry.json", [asdict(record) for record in results])

    (output_dir / "README.md").write_text(
        _render_readme(scripts, results, claims),
        encoding="utf-8",
    )
    legacy_archive_plan = output_dir / "archive_plan.md"
    if legacy_archive_plan.exists():
        legacy_archive_plan.unlink()
    (output_dir / "retention_plan.md").write_text(
        _render_retention_plan(results),
        encoding="utf-8",
    )

    if write_scripts_inventory:
        scripts_inventory_path = ROOT / "scripts" / "INVENTORY.md"
        scripts_inventory_path.write_text(
            _render_scripts_inventory(scripts, _rel(output_dir)),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated inventory artifacts.",
    )
    parser.add_argument(
        "--skip-scripts-inventory",
        action="store_true",
        help="Do not rewrite scripts/INVENTORY.md.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    generate_inventory(output_dir=output_dir, write_scripts_inventory=not args.skip_scripts_inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
