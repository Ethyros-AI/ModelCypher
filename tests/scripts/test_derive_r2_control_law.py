# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import derive_r2_control_law as derive_r2


def test_derive_control_law_passes_retained_artifact_checks() -> None:
    law, state_table, validation = derive_r2.derive_control_law()

    assert validation["all_passed"] is True
    assert law.arm_on_online_eval_accuracy_drop is True
    assert law.arm_on_margin_trend_declining is True
    assert law.arm_on_stable_rank_concentration is True
    assert state_table


def test_state_table_contains_expected_arm_epochs_for_counterexamples() -> None:
    _law, state_table, _validation = derive_r2.derive_control_law()
    summary_rows = {
        row["artifact_id"]: row
        for row in state_table
        if row.get("kind") == "trial_summary"
    }

    assert summary_rows["pipeline_validation_safe"]["arm_epoch_candidate"] is None
    assert summary_rows["stage_a_seed42"]["arm_epoch_candidate"] is None
    assert summary_rows["behavioral_probe_cayley_seed42"]["arm_epoch_candidate"] == 2
    assert summary_rows["behavioral_probe_adamw_seed42"]["arm_epoch_candidate"] == 1


def test_build_falsifier_manifest_uses_closed_loop_mode(tmp_path: Path) -> None:
    manifest = derive_r2.build_falsifier_manifest(
        law_path=tmp_path / "law.json",
        state_table_path=tmp_path / "state.json",
        validation_path=tmp_path / "validation.json",
        report_path=tmp_path / "report.json",
        artifact_root=tmp_path / "artifacts",
    )

    assert manifest["frozen_tuple"]["controller_mode"] == "mass_behavioral_closed_loop"
    assert manifest["frozen_tuple"]["optimizer_research_mode"] == "cayley_stiefel_mass"
    assert manifest["frozen_tuple"]["seed"] == 42
