# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.experimental.merge.models import build_transmission_layer_scores
from modelcypher.experimental.merge.pipeline import (
    _summarize_layer_coupling,
    _summarize_probe_rank_coverage,
)
from modelcypher.experimental.merge.stages.transplant_stage import (
    compute_density_dominance_margin,
)
from modelcypher.experimental.merge.stages.transplant_weight_processor import (
    _summarize_mlp_scale_observations,
)


def test_density_dominance_margin_is_signed_midpoint_difference() -> None:
    assert compute_density_dominance_margin(0.5) == 0.0
    assert compute_density_dominance_margin(0.75) == 0.5
    assert compute_density_dominance_margin(0.25) == -0.5


def test_transmission_layer_scores_report_raw_ranking_components() -> None:
    scores = build_transmission_layer_scores(
        variance_concentrations={0: 0.9, 1: 0.2, 2: 0.3},
        effective_ranks={0: 1.0, 1: 9.0, 2: 7.0},
    )

    assert [row["layer_idx"] for row in scores] == [1, 2, 0]
    best = scores[0]
    assert best["low_variance_advantage"] >= best["high_rank_advantage"]
    assert "transmission_score" in best
    assert "distance_to_nearest_edge" in best


def test_mlp_scale_observations_report_raw_divergence_not_reversion() -> None:
    summary = _summarize_mlp_scale_observations(
        {"gate": 0.125, "up": 1.0, "down": 8.0}
    )

    assert summary["scale_divergence"] == pytest.approx(64.0)
    assert summary["observed_scales"]["gate"] == 0.125
    assert "joint_reference_corrections" in summary


def test_probe_rank_coverage_summary_preserves_per_layer_ratios() -> None:
    summary = _summarize_probe_rank_coverage(
        {
            "3": {
                "source_rank": 12,
                "source_dim": 16,
                "target_rank": 10,
                "target_dim": 16,
                "alignment_rank": 10,
                "coverage_ratio": 0.625,
                "deficit": 6,
                "trajectory_rank": 11,
            }
        }
    )

    assert summary["layers_measured"] == 1
    assert summary["per_layer"]["3"]["coverage_ratio"] == 0.625
    assert summary["mean_coverage_ratio"] == pytest.approx(0.625)


def test_layer_coupling_summary_exposes_mass_by_target_layer() -> None:
    summary = _summarize_layer_coupling(
        [[0.7, 0.3], [0.2, 0.8]],
        source_layers=[4, 9],
        target_layers=[1, 2],
    )

    assert summary["target_layers_measured"] == 2
    assert summary["by_target_layer"]["1"]["incoming_mass"] == pytest.approx(0.9)
    assert summary["by_target_layer"]["2"]["argmax_source_layer"] == 9
