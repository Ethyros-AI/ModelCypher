# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the falsifier verdict logic."""

from __future__ import annotations

import pytest

from scripts.lkm.evaluate_falsifier import compute_verdict


def _make_arms(
    em_r4_s1500: float = 0.889,
    em_r4_s4500: float = 0.89,
    em_r16_s1500: float = 0.95,
    em_r16_s4500: float = 0.96,
    rf_passed_r4: float = 0.9,
    rf_failed_r4: float = 0.6,
    rf_passed_r16: float = 0.95,
    rf_failed_r16: float = 0.9,
    mean_interference_r4: float = 0.1,
    n_high_interference_r4: int = 0,
) -> dict:
    """Build mock arm results for testing verdict logic."""
    def _geom(
        rf_passed, rf_failed, mean_interf=0.1, n_high=0
    ):
        return {
            "summary": {
                "mean_rf": (rf_passed + rf_failed) / 2,
                "mean_rf_passed": rf_passed,
                "mean_rf_failed": rf_failed,
                "mean_interference": mean_interf,
                "max_interference": mean_interf + 0.1,
                "n_high_interference_pairs": n_high,
            },
            "retained_fractions": [],
            "interference_matrix": [],
        }

    return {
        "B0-r4-s1500": {
            "em": em_r4_s1500,
            "geometry": _geom(
                rf_passed_r4,
                rf_failed_r4,
                mean_interference_r4,
                n_high_interference_r4,
            ),
        },
        "B0-r4-s4500": {
            "em": em_r4_s4500,
            "geometry": _geom(rf_passed_r4, rf_failed_r4),
        },
        "B0-r16-s1500": {
            "em": em_r16_s1500,
            "geometry": _geom(rf_passed_r16, rf_failed_r16),
        },
        "B0-r16-s4500": {
            "em": em_r16_s4500,
            "geometry": _geom(rf_passed_r16, rf_failed_r16),
        },
    }


class TestVerdictRankBottleneck:
    """H-RANK: Large rank gain, no opt gain, RF separation."""

    def test_pure_rank_bottleneck(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.89,   # +0.001 < 5%
            em_r16_s1500=0.96,  # +0.071 > 5%
            em_r16_s4500=0.97,
            rf_passed_r4=0.91,
            rf_failed_r4=0.62,  # separation = 0.29
        )
        verdict = compute_verdict(arms)
        assert verdict["verdict"] == "RANK_BOTTLENECK"
        assert verdict["hypothesis_tests"]["H-RANK"]["pass"] is True
        assert verdict["hypothesis_tests"]["H-OPT"]["pass"] is False

    def test_rank_gains(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.89,
            em_r16_s1500=0.96,
            em_r16_s4500=0.97,
        )
        verdict = compute_verdict(arms)
        assert verdict["hypothesis_tests"]["H-RANK"]["rank_em_gain"] == pytest.approx(
            0.071, abs=0.001
        )
        assert verdict["hypothesis_tests"]["H-RANK"]["opt_em_gain"] == pytest.approx(
            0.001, abs=0.001
        )


class TestVerdictOptimizationCeiling:
    """H-OPT: Large opt gain, no rank gain."""

    def test_pure_optimization_ceiling(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.96,   # +0.071 > 5%
            em_r16_s1500=0.90,  # +0.011 < 5%
            em_r16_s4500=0.97,
        )
        verdict = compute_verdict(arms)
        assert verdict["verdict"] == "OPTIMIZATION_CEILING"
        assert verdict["hypothesis_tests"]["H-OPT"]["pass"] is True
        assert verdict["hypothesis_tests"]["H-RANK"]["pass"] is False


class TestVerdictInterferenceClustering:
    """H-INTRF: Large rank gain + high interference pairs."""

    def test_interference_clustering(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.91,   # partial gain
            em_r16_s1500=0.96,  # large rank gain
            em_r16_s4500=0.97,
            rf_passed_r4=0.91,
            rf_failed_r4=0.62,
            mean_interference_r4=0.4,
            n_high_interference_r4=5,
        )
        verdict = compute_verdict(arms)
        # INTRF takes precedence over RANK when interference is present
        assert verdict["verdict"] == "INTERFERENCE_CLUSTERING"
        assert verdict["hypothesis_tests"]["H-INTRF"]["pass"] is True


class TestVerdictConfound:
    """H-CONFOUND: Both gains large, low RF separation."""

    def test_confound(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.95,   # +0.061 > 5%
            em_r16_s1500=0.95,  # +0.061 > 5%
            em_r16_s4500=0.98,
            rf_passed_r4=0.85,
            rf_failed_r4=0.83,  # separation = 0.02 < 0.05
        )
        verdict = compute_verdict(arms)
        assert verdict["verdict"] == "CONFOUND_RANK_HELPS_OPT"
        assert verdict["hypothesis_tests"]["H-CONFOUND"]["pass"] is True


class TestVerdictEdgeCases:
    """Edge cases and neither/mixed outcomes."""

    def test_neither_gain(self):
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.89,
            em_r16_s1500=0.89,
            em_r16_s4500=0.89,
        )
        verdict = compute_verdict(arms)
        assert verdict["verdict"] == "NEITHER"

    def test_mixed_gains_with_rf_separation(self):
        """Both gains large + RF separation -> not pure CONFOUND."""
        arms = _make_arms(
            em_r4_s1500=0.889,
            em_r4_s4500=0.95,
            em_r16_s1500=0.96,
            em_r16_s4500=0.98,
            rf_passed_r4=0.91,
            rf_failed_r4=0.62,  # large separation
        )
        verdict = compute_verdict(arms)
        # Both gains large, RF separation exists but confound is false
        assert verdict["verdict"] == "MIXED"

    def test_all_arms_present(self):
        arms = _make_arms()
        verdict = compute_verdict(arms)
        assert "B0-r4-s1500" in verdict["arms"]
        assert "B0-r4-s4500" in verdict["arms"]
        assert "B0-r16-s1500" in verdict["arms"]
        assert "B0-r16-s4500" in verdict["arms"]
        assert "H-RANK" in verdict["hypothesis_tests"]
        assert "H-OPT" in verdict["hypothesis_tests"]
        assert "H-INTRF" in verdict["hypothesis_tests"]
        assert "H-CONFOUND" in verdict["hypothesis_tests"]
        assert verdict["verdict"] is not None
