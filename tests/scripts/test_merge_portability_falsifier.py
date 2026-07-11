from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import merge_portability_falsifier as merge_falsifier

from modelcypher.experimental.merge.falsifier_contract import (
    build_merge_portability_manifest,
    build_merge_portability_summary,
    emit_merge_portability_bundle,
    validate_merge_portability_bundle,
)


def test_emit_scaffold_writes_required_bundle(tmp_path) -> None:
    out_dir = merge_falsifier.emit_scaffold(
        output_dir=tmp_path / "merge_portability",
        run_id="q8_scaffold",
    )

    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "ledger.jsonl").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["roadmap_item"] == "R5"
    assert manifest["open_question"] == "Q8"
    assert manifest["frozen_contract"]["evaluator_bundle"]["frozen"] is True


def test_validator_rejects_promotable_result_without_commensurability_proof(
    tmp_path,
) -> None:
    out_dir = tmp_path / "invalid_promotable_bundle"
    manifest = build_merge_portability_manifest(
        run_id="bad_q8_bundle",
        output_dir=out_dir,
        claim_form={"commensurability_proof": ""},
    )
    summary = build_merge_portability_summary(
        run_id="bad_q8_bundle",
        promotion_status="candidate",
    )
    emit_merge_portability_bundle(out_dir, manifest=manifest, summary=summary)

    validation = validate_merge_portability_bundle(out_dir)
    assert validation["ok"] is False
    assert any("commensurability proof" in error for error in validation["errors"])
