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

from modelcypher.core.use_cases.geometry_safety_service import (
    DriftThresholds,
    VulnerabilityThresholds,
)


class TestDriftThresholdsGeometric:
    def test_well_separated_distributions(self):
        """Safe drift [0,1] vs attack drift [3,5] → stable crossing between them."""
        safe = [float(i) * 0.1 for i in range(20)]  # 0.0..1.9
        attack = [3.0 + float(i) * 0.1 for i in range(20)]  # 3.0..4.9
        thresholds, result = DriftThresholds.from_calibration_geometric(
            safe, attack, seed=42
        )

        assert result.is_stable is True
        assert result.auroc > 0.9
        # Minimal < moderate < significant (CI ordering)
        assert thresholds.minimal <= thresholds.moderate
        assert thresholds.moderate <= thresholds.significant
        # Boundary should be between the two groups
        assert thresholds.moderate >= 1.9
        assert thresholds.moderate <= 3.0

    def test_overlapping_distributions(self):
        """Overlapping groups → boundary exists but may not be stable."""
        safe = [1.0, 2.0, 3.0, 4.0, 5.0]
        attack = [3.0, 4.0, 5.0, 6.0, 7.0]
        thresholds, result = DriftThresholds.from_calibration_geometric(
            safe, attack, seed=42
        )

        assert isinstance(thresholds.minimal, float)
        assert isinstance(thresholds.moderate, float)
        assert isinstance(thresholds.significant, float)
        # AUROC should be moderate (not perfect separation)
        assert result.auroc < 1.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="Both groups need >= 2"):
            DriftThresholds.from_calibration_geometric([1.0], [2.0, 3.0])

    def test_returns_crossing_result(self):
        """Result includes full CrossingResult for promotion decision."""
        safe = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        attack = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
        _, result = DriftThresholds.from_calibration_geometric(safe, attack, seed=42)

        assert result.n_a == 10
        assert result.n_b == 10
        assert result.ci_lower <= result.boundary
        assert result.ci_upper >= result.boundary


class TestVulnerabilityThresholdsGeometric:
    def test_clear_spike_boundary(self):
        """Safe delta-H ~ 0, attack delta-H ~ 2 → clear spike boundary."""
        safe_dh = [0.0 + i * 0.05 for i in range(20)]  # 0..0.95
        attack_dh = [2.0 + i * 0.05 for i in range(20)]  # 2..2.95
        safe_ent = [1.0 + i * 0.05 for i in range(20)]  # 1..1.95
        attack_ent = [4.0 + i * 0.05 for i in range(20)]  # 4..4.95

        thresholds, results = VulnerabilityThresholds.from_calibration_geometric(
            safe_dh, attack_dh, safe_ent, attack_ent, seed=42
        )

        # Spike boundary between safe and attack delta-H
        assert thresholds.entropy_spike >= 0.95
        assert thresholds.entropy_spike <= 2.0
        assert results["entropy_spike"].is_stable is True

        # Bypass boundary between safe and attack entropy
        assert thresholds.boundary_bypass >= 1.95
        assert thresholds.boundary_bypass <= 4.0
        assert results["boundary_bypass"].is_stable is True

    def test_suppression_boundary_negative(self):
        """Refusal suppression uses negative delta-H boundary."""
        # Safe: delta-H near 0
        safe_dh = [0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.2, -0.2]
        # Attack: strongly negative delta-H (suppresses entropy)
        attack_dh = [-2.0, -1.8, -2.2, -1.9, -2.1, -2.0, -1.8, -2.2, -1.9, -2.1]
        safe_ent = [1.0] * 10
        attack_ent = [1.0] * 10  # entropy not relevant for suppression

        thresholds, results = VulnerabilityThresholds.from_calibration_geometric(
            safe_dh, attack_dh, safe_ent, attack_ent, seed=42
        )

        # Suppression threshold should be negative (between -0.2 and -1.8)
        assert thresholds.refusal_suppression < 0.0
        assert thresholds.refusal_suppression > -2.0

    def test_returns_all_three_results(self):
        """Result dict has entries for all three vulnerability types."""
        safe_dh = [0.1, 0.2, 0.3, 0.4, 0.5]
        attack_dh = [1.0, 1.1, 1.2, 1.3, 1.4]
        safe_ent = [1.0, 1.1, 1.2, 1.3, 1.4]
        attack_ent = [3.0, 3.1, 3.2, 3.3, 3.4]

        _, results = VulnerabilityThresholds.from_calibration_geometric(
            safe_dh, attack_dh, safe_ent, attack_ent, seed=42
        )

        assert "entropy_spike" in results
        assert "boundary_bypass" in results
        assert "refusal_suppression" in results

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="Both groups need >= 2"):
            VulnerabilityThresholds.from_calibration_geometric(
                [0.1], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]
            )
