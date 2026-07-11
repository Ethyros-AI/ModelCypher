# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_RESULTS = _ROOT / "results"

_MAINTAINED_DOCS = (
    _ROOT / "AGENTS.md",
    _ROOT / "docs" / "MISSION.md",
    _ROOT / "docs" / "VISION.md",
    _ROOT / "docs" / "RESEARCH-ROADMAP.md",
    _ROOT / "docs" / "ARCHITECTURE.md",
    _ROOT / "docs" / "CURIOSITY-DAEMON.md",
    _ROOT / "docs" / "START-HERE.md",
    _ROOT / "docs" / "research" / "OPEN-MATHEMATICAL-QUESTIONS.md",
    _ROOT / "docs" / "research" / "AUTONOMOUS-RESEARCH-PROTOCOL.md",
)
_REPO_PATH_PATTERN = re.compile(
    r"(AGENTS\.md|CONTRIBUTING\.md|docs/[A-Za-z0-9._/-]+|src/[A-Za-z0-9._/-]+|"
    r"scripts/[A-Za-z0-9._/-]+|tests/[A-Za-z0-9._/-]+|results/[A-Za-z0-9._/-]+|"
    r"papers/[A-Za-z0-9._/-]+|data/[A-Za-z0-9._/-]+)"
)
_GRANDFATHERED_EXPERIMENTAL_IMPORTS = {
    "src/modelcypher/infrastructure/service_factory.py",
    "src/modelcypher/cli/composition.py",
    "src/modelcypher/core/domain/lora_memory_store.py",
    "src/modelcypher/core/use_cases/profile_service.py",
    "src/modelcypher/core/use_cases/adapter_analysis_service.py",
    "src/modelcypher/core/use_cases/model_service.py",
}
_GRANDFATHERED_CANONICAL_RESULTS = {
    "closedform_sequential_correction",
    "continual_learning",
    "entropy_curvature",
    "entropy_curvature_operator_split",
    "f5_sign_law",
    "f5_sign_law_analysis",
    "f5_sign_law_analysis_6models",
    "f5_sign_law_full",
    "f5_sign_law_precheck_20260304T130042",
    "f5_sign_law_precheck_20260304T130042_mistral",
    "g5_8b_validation",
    "g5_8b_validation_multiseed",
    "geometry_sota",
    "gqa_norm_entropy_coupling",
    "information_bridge_linear_cka",
    "layer_invariants",
    "lora_memory_capacity_validation",
    "nblora_vs_standard",
    "pipeline_validation",
    "pipeline_validation_blindness_350M_t20",
    "pipeline_validation_blindness_smoke",
    "pipeline_validation_cert_350m",
    "pipeline_validation_cert_350m_5t",
    "pipeline_validation_cert_350m_scoped",
    "pipeline_validation_cert_350m_v2",
    "quantization_ab_survey",
    "quantization_frontier",
    "quantization_scale_ab_test",
    "sota_audit_2026_03",
    "tangent_subspace_id_mechanism",
    "weight_geometry",
}
_OWNER_LOCAL_RESULT_FAMILIES = _GRANDFATHERED_CANONICAL_RESULTS | {
    "measurement_atlas",
    "repo_research_inventory",
}
_OWNER_LOCAL_DATA_REFERENCES = {
    "data/training/r1_quick_aligned_train.jsonl",
    "data/training/r1_quick_aligned_val.jsonl",
}


def _is_declared_owner_local_reference(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] == "results":
        return parts[1] in _OWNER_LOCAL_RESULT_FAMILIES
    return rel_path in _OWNER_LOCAL_DATA_REFERENCES


def test_owner_local_reference_policy_is_narrow() -> None:
    assert _is_declared_owner_local_reference(
        "results/repo_research_inventory/results_registry.json"
    )
    assert _is_declared_owner_local_reference(
        "data/training/r1_quick_aligned_train.jsonl"
    )
    assert not _is_declared_owner_local_reference("results/unknown_new_family/result.json")
    assert not _is_declared_owner_local_reference("src/modelcypher/missing.py")


def test_deleted_probe_shim_stays_deleted() -> None:
    shim_path = _ROOT / "src/modelcypher/core/domain/geometry/orthogonal_probe_generator.py"
    assert not shim_path.exists()


def test_runtime_source_omits_deleted_loader_aliases_and_probe_shim_imports() -> None:
    banned_fragments = [
        ".load_model_for_training(",
        "def load_model_for_training(",
        "def get_model_loader(",
        "def load_model_weights_only(",
        "orthogonal_probe_generator",
        "layer_agreement_rate",
        "legacy_profile_path",
    ]
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            if fragment in content:
                violations.append(f"{path.relative_to(_ROOT)}: contains '{fragment}'")

    assert not violations, "\n".join(violations)


def test_runtime_source_omits_backward_compatibility_language() -> None:
    banned_fragments = [
        "backward compatibility",
        "backwards compatibility",
        "backward compat",
        "backwards compat",
    ]
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        content = path.read_text(encoding="utf-8").lower()
        for fragment in banned_fragments:
            if fragment in content:
                violations.append(f"{path.relative_to(_ROOT)}: contains '{fragment}'")

    assert not violations, "\n".join(violations)


def test_non_experimental_source_does_not_gain_new_experimental_imports() -> None:
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        if rel.startswith("src/modelcypher/experimental/"):
            continue
        content = path.read_text(encoding="utf-8")
        if "modelcypher.experimental" not in content:
            continue
        if rel not in _GRANDFATHERED_EXPERIMENTAL_IMPORTS:
            violations.append(f"{rel}: imports modelcypher.experimental")

    assert not violations, "\n".join(violations)


def test_maintained_docs_reference_existing_repo_paths() -> None:
    violations: list[str] = []
    for doc_path in _MAINTAINED_DOCS:
        content = doc_path.read_text(encoding="utf-8")
        matches = {
            match.group(0).rstrip("`).,:")
            for match in _REPO_PATH_PATTERN.finditer(content)
        }
        for rel_path in sorted(matches):
            if "*" in rel_path:
                continue
            if not (_ROOT / rel_path).exists():
                if _is_declared_owner_local_reference(rel_path):
                    continue
                violations.append(
                    f"{doc_path.relative_to(_ROOT)}: missing referenced path '{rel_path}'"
                )

    assert not violations, "\n".join(violations)


def test_new_canonical_result_families_require_complete_bundle() -> None:
    registry_path = _RESULTS / "repo_research_inventory" / "results_registry.json"
    if not registry_path.exists():
        pytest.skip(
            "owner-local research inventory is absent; see "
            "docs/research/OWNER-ARTIFACT-POLICY.md"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for record in registry:
        family = record["family"]
        if record["status"] != "canonical":
            continue
        if family in _GRANDFATHERED_CANONICAL_RESULTS:
            continue

        family_dir = _RESULTS / family
        report_path = family_dir / "REPORT.md"
        summary_ok = any(
            path.name
            in {
                "summary.json",
                "verdict.json",
                "analysis_summary.json",
                "multiseed_gates.json",
            }
            or "summary" in path.stem
            or "verdict" in path.stem
            or "analysis" in path.stem
            for path in family_dir.rglob("*.json")
        )
        manifest_ok = any(
            "manifest" in path.name.lower() for path in family_dir.rglob("*")
        )
        ledger_ok = any("ledger" in path.name.lower() for path in family_dir.rglob("*"))

        if not report_path.exists():
            violations.append(f"{family}: missing REPORT.md")
        if not summary_ok:
            violations.append(f"{family}: missing machine-readable summary JSON")
        if not manifest_ok:
            violations.append(f"{family}: missing run manifest or charter")
        if not ledger_ok:
            violations.append(f"{family}: missing append-only ledger")

    assert not violations, "\n".join(violations)
