"""Mechanical contracts for the owner-run SOTA replication packet."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs" / "research" / "replication" / "ws4_2"
MANIFESTS = {
    "contextual": PACKET / "contextual_curvature.manifest.json",
    "intrinsic_dimension": PACKET / "intrinsic_dimension_profile.manifest.json",
    "fixed_basis": PACKET / "fixed_basis_feature_survival.manifest.json",
}
REQUIRED_CHARTER_FIELDS = {
    "linkedBlocker",
    "runId",
    "claimContract",
    "primaryObservable",
    "explicitFalsifier",
    "mutableSurface",
    "frozenSurfaces",
    "baselineCommand",
    "comparisonBudget",
    "artifactDirectory",
    "ledgerPath",
}


def _load(name: str) -> dict:
    return json.loads(MANIFESTS[name].read_text(encoding="utf-8"))


def test_all_manifests_are_owner_gated_complete_run_charters() -> None:
    for path in MANIFESTS.values():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "owner_run_required"
        assert REQUIRED_CHARTER_FIELDS <= manifest.keys()
        assert manifest["claimContract"] == (
            "observable = f(geometry_state, architecture_state, scale_state, "
            "precision_state, measurement_operator)"
        )
        assert manifest["ownerGate"].strip()


def test_each_manifest_links_exactly_one_active_blocker() -> None:
    assert _load("contextual")["linkedBlocker"] == "A1"
    assert _load("intrinsic_dimension")["linkedBlocker"] == "A1"
    assert _load("fixed_basis")["linkedBlocker"] == "R4"


def test_contextual_manifest_freezes_only_cited_paper_parameters() -> None:
    parameters = _load("contextual")["paperParameters"]
    assert {name: setting["value"] for name, setting in parameters.items()} == {
        "windowSize": 3,
        "minimumTokenPosition": 7,
        "crossValidationFolds": 10,
        "confidenceLevel": 0.95,
        "perturbationScale": 0.2,
        "subspaceDimension": 2,
        "perturbationsPerToken": 300,
        "importanceBins": 100,
        "importanceEpsilon": 1e-12,
        "importanceWeightCap": 10,
        "bootstrapReplicates": 2000,
    }
    assert all(setting["source"] for setting in parameters.values())


def test_commands_resolve_to_tracked_executable_surfaces() -> None:
    contextual_command = _load("contextual")["baselineCommand"]
    fixed_basis_command = _load("fixed_basis")["baselineCommand"]
    dimension_command = _load("intrinsic_dimension")["baselineCommand"]
    assert "scripts/run_contextual_curvature_replication.py" in contextual_command
    assert "scripts/run_fixed_basis_feature_survival.py" in fixed_basis_command
    assert "mc --json analyze dimension-profile" in dimension_command
    assert "--local" in dimension_command
    assert "--with-mle" in dimension_command
    assert "--with-ci" in dimension_command
    assert (ROOT / "scripts" / "run_contextual_curvature_replication.py").is_file()
    assert (ROOT / "scripts" / "run_fixed_basis_feature_survival.py").is_file()


def test_real_model_handoffs_remain_explicit_in_code() -> None:
    contextual = (ROOT / "scripts" / "run_contextual_curvature_replication.py").read_text(
        encoding="utf-8"
    )
    fixed_basis = (ROOT / "scripts" / "run_fixed_basis_feature_survival.py").read_text(
        encoding="utf-8"
    )
    assert (
        "# TODO(owner): run the MLX real-model replication and review discrepancies per WS4.2."
        in contextual
    )
    assert (
        "# TODO(owner): run the fixed-basis comparison on real precision states per R4."
        in fixed_basis
    )
