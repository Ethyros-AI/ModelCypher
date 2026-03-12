# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import postmortem_r2_closed_loop as postmortem_r2


def test_build_postmortem_detects_lexicographic_fallback_and_misaligned_target() -> None:
    postmortem = postmortem_r2.build_postmortem()

    assert postmortem["classification"]["status"] == "MECHANISM_UNDERSPECIFIED"
    assert postmortem["classification"]["temporal_blind_spot"] is True
    assert postmortem["classification"]["target_selection_fallback"] == "lexicographic_tie_break"
    assert postmortem["classification"]["target_misaligned_with_transport"] is True

    arm_event = postmortem["arm_event"]
    assert arm_event["target_layer"] == "model.layers.8.self_attn.v_proj.weight"
    assert arm_event["ordering_metrics_all_null"] is True
    assert arm_event["available_ordering_metric_counts"] == {
        "behavioral_transport_over_remaining_budget": 0,
        "spectral_budget_ratio": 0,
        "stable_rank_concentration": 0,
    }

    transport = postmortem["transport_at_arm"]
    assert transport["target_layer_transport_rank"] == 13
    assert transport["top_layers"][0]["layer"] == "model.layers.14.self_attn.q_proj.weight"


def test_build_postmortem_detects_freeze_effectiveness_and_off_surface_divergence() -> None:
    postmortem = postmortem_r2.build_postmortem()

    assert postmortem["freeze_effectiveness"]["freeze_applied"] is True
    assert postmortem["freeze_effectiveness"]["target_layer_parameter_update_norm_after_arm"] == 0.0

    divergence = postmortem["divergence"]
    assert divergence["inference_probe_min_cka_layer"] == 4
    assert divergence["inference_probe_min_cka_layer_on_adaptation_surface"] is False


def test_render_markdown_mentions_counterexample_and_fallback() -> None:
    markdown = postmortem_r2.render_markdown(postmortem_r2.build_postmortem())

    assert "MECHANISM_UNDERSPECIFIED" in markdown
    assert "lexicographic_tie_break" in markdown
    assert "model.layers.8.self_attn.v_proj.weight" in markdown
