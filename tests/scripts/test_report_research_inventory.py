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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import report_research_inventory as inventory


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_generate_inventory_classifies_scripts_results_and_claims(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path

    for doc_rel in (
        "docs/VISION.md",
        "docs/MISSION.md",
        "docs/RESEARCH-ROADMAP.md",
        "docs/research/OPEN-MATHEMATICAL-QUESTIONS.md",
        "docs/research/SOTA-AUDIT-2026-03.md",
        "docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md",
    ):
        _write(repo_root / doc_rel, "")

    _write(
        repo_root / "docs/VISION.md",
        "\n".join(
            [
                "Use `scripts/canonical_script.py`.",
                "Canonical evidence: `results/canonical_family/REPORT.md`.",
            ]
        ),
    )
    _write(
        repo_root / "docs/research/OPEN-MATHEMATICAL-QUESTIONS.md",
        "Blocking artifact: `results/claim_family/run.json`.",
    )
    _write(
        repo_root / "docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md",
        "Scanner: `poetry run python scripts/report_doctrine_audit.py`.",
    )

    _write(repo_root / "scripts/canonical_script.py", "print('canonical')\n")
    _write(repo_root / "scripts/artifact_only.py", "print('artifact')\n")
    _write(repo_root / "scripts/unlinked.py", "print('delete')\n")
    _write(repo_root / "scripts/report_doctrine_audit.py", "print('doctrine')\n")

    _write(repo_root / "tests/scripts/test_canonical_script.py", "def test_placeholder():\n    pass\n")

    _write(repo_root / "results/canonical_family/REPORT.md", "# report\n")
    _write(repo_root / "results/artifact_only/summary.json", "{}\n")
    _write(repo_root / "results/report_summary_only/REPORT.md", "# report\n")
    _write(
        repo_root / "results/report_summary_only/summary.json",
        json.dumps({"status": "summary_only"}),
    )
    _write(repo_root / "results/summary_family/run1/model.safetensors", "a")
    _write(repo_root / "results/summary_family/run2/model.safetensors", "b")
    _write(repo_root / "results/claim_family/run.json", "{}\n")

    _write(
        repo_root / "results/sota_audit_2026_03/internal_claim_registry.json",
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "CR-TEST-001",
                        "track": "test",
                        "statement": "Test claim",
                        "current_status": "[EMPIRICAL]",
                        "operator": "test operator",
                        "architecture_terms": "test architecture",
                        "scale_terms": "test scale",
                        "falsifier": "test falsifier",
                        "latest_results_path": "results/claim_family/run.json",
                    }
                ]
            }
        ),
    )
    _write(
        repo_root / "results/sota_audit_2026_03/claim_crosswalk.json",
        json.dumps(
            {
                "records": [
                    {
                        "claim_id": "CR-TEST-001",
                        "internal_status": "[EMPIRICAL]",
                        "internal_evidence_pointers": ["results/claim_family/run.json"],
                        "classification": "PUSH_FURTHER",
                        "recommended_next_step": "Run the next falsifier.",
                        "integration_target_path": "scripts/canonical_script.py",
                    }
                ]
            }
        ),
    )

    monkeypatch.setattr(inventory, "ROOT", repo_root)
    monkeypatch.setattr(
        inventory,
        "ACTIVE_DOCS",
        tuple(
            repo_root / rel
            for rel in (
                "docs/VISION.md",
                "docs/MISSION.md",
                "docs/RESEARCH-ROADMAP.md",
                "docs/research/OPEN-MATHEMATICAL-QUESTIONS.md",
                "docs/research/SOTA-AUDIT-2026-03.md",
                "docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md",
            )
        ),
    )
    output_dir = repo_root / "results" / "repo_research_inventory"
    monkeypatch.setattr(inventory, "OUTPUT_DIR", output_dir)

    inventory.generate_inventory(output_dir=output_dir, write_scripts_inventory=True)

    scripts_registry = json.loads((output_dir / "scripts_registry.json").read_text(encoding="utf-8"))
    results_registry = json.loads((output_dir / "results_registry.json").read_text(encoding="utf-8"))
    claim_registry = json.loads((output_dir / "claim_registry.json").read_text(encoding="utf-8"))

    scripts_by_path = {record["path"]: record for record in scripts_registry}
    assert scripts_by_path["scripts/canonical_script.py"]["status"] == "canonical"
    assert scripts_by_path["scripts/canonical_script.py"]["evidence_status"] == "tested+artifact"
    assert scripts_by_path["scripts/artifact_only.py"]["status"] == "summary_only"
    assert scripts_by_path["scripts/artifact_only.py"]["artifact_paths"] == ["results/artifact_only"]
    assert scripts_by_path["scripts/unlinked.py"]["status"] == "delete"
    assert scripts_by_path["scripts/report_doctrine_audit.py"]["status"] == "canonical"

    results_by_family = {record["family"]: record for record in results_registry}
    assert results_by_family["canonical_family"]["status"] == "canonical"
    assert results_by_family["claim_family"]["status"] == "canonical"
    assert results_by_family["report_summary_only"]["status"] == "summary_only"
    assert results_by_family["summary_family"]["status"] == "summary_only"
    assert results_by_family["summary_family"]["immediate_subdir_count"] == 2

    assert claim_registry[0]["claim_id"] == "CR-TEST-001"
    assert claim_registry[0]["classification"] == "PUSH_FURTHER"
    assert claim_registry[0]["result_families"] == ["claim_family"]

    scripts_inventory = (repo_root / "scripts" / "INVENTORY.md").read_text(encoding="utf-8")
    assert "This file is generated" in scripts_inventory
    assert "`scripts/canonical_script.py`" in scripts_inventory

    retention_plan = (output_dir / "retention_plan.md").read_text(encoding="utf-8")
    assert "`summary_family`" in retention_plan
