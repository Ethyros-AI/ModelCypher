# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import postmortem_r2_closed_loop as postmortem_r2

RETAINED_CLOSED_LOOP = (
    REPO_ROOT
    / "results"
    / "nblora_vs_standard"
    / "validate_derived_r2_closed_loop_seed42_quick.json"
)
pytestmark = pytest.mark.skipif(
    not RETAINED_CLOSED_LOOP.exists(),
    reason=(
        "owner-local retained R2 artifact is absent; see "
        "docs/research/OWNER-ARTIFACT-POLICY.md"
    ),
)


def test_build_postmortem_reports_unarmed_retained_artifact() -> None:
    postmortem = postmortem_r2.build_postmortem()

    assert postmortem["classification"]["status"] == "MECHANISM_NOT_ENGAGED"
    assert postmortem["classification"]["counterexample_confirmed"] is False
    assert postmortem["classification"]["target_selection_fallback"] == "not_armed"

    arm_event = postmortem["arm_event"]
    assert arm_event["armed"] is False
    assert arm_event["target_layer"] is None
    assert arm_event["ordering_metrics_all_null"] is True
    assert arm_event["available_ordering_metric_counts"] == {
        "behavioral_transport_over_remaining_budget": 0,
        "spectral_budget_ratio": 0,
        "stable_rank_concentration": 0,
    }

    transport = postmortem["transport_at_arm"]
    assert transport["target_layer_transport_rank"] is None
    assert transport["top_layers"] == []


def test_build_postmortem_marks_no_freeze_or_off_surface_claim_for_unarmed_artifact() -> None:
    postmortem = postmortem_r2.build_postmortem()

    assert postmortem["freeze_effectiveness"]["freeze_applied"] is False
    assert postmortem["freeze_effectiveness"]["target_layer_parameter_update_norm_after_arm"] is None

    divergence = postmortem["divergence"]
    assert divergence["inference_probe_min_cka_layer_on_adaptation_surface"] is None


def test_render_markdown_mentions_unarmed_status() -> None:
    markdown = postmortem_r2.render_markdown(postmortem_r2.build_postmortem())

    assert "MECHANISM_NOT_ENGAGED" in markdown
    assert "not_armed" in markdown
    assert "no closed-loop arm event was recorded" in markdown
