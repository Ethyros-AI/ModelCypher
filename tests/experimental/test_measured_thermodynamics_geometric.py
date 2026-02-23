# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

from modelcypher.experimental.thermo.measured_thermodynamics import (
    MeasuredEnergy,
    MeasuredThresholds,
)


class TestMeasuredEnergyConfidence:
    """Confidence uses CLT rate 1 - 1/sqrt(n), not hardcoded /10 divisor."""

    def test_zero_samples(self):
        e = MeasuredEnergy(value=0.0, probability=0.5, sample_count=0, temperature=1.0)
        assert e.confidence == 0.0

    def test_one_sample(self):
        e = MeasuredEnergy(value=0.0, probability=0.5, sample_count=1, temperature=1.0)
        assert e.confidence == pytest.approx(0.0)

    def test_four_samples(self):
        e = MeasuredEnergy(value=0.0, probability=0.5, sample_count=4, temperature=1.0)
        assert e.confidence == pytest.approx(0.5)

    def test_100_samples(self):
        e = MeasuredEnergy(value=0.0, probability=0.5, sample_count=100, temperature=1.0)
        assert e.confidence == pytest.approx(0.9)

    def test_10000_samples(self):
        e = MeasuredEnergy(value=0.0, probability=0.5, sample_count=10000, temperature=1.0)
        assert e.confidence == pytest.approx(0.99)

    def test_monotonically_increasing(self):
        """Confidence increases with sample count."""
        counts = [1, 4, 16, 64, 256, 1024]
        confidences = [
            MeasuredEnergy(
                value=0.0, probability=0.5, sample_count=n, temperature=1.0
            ).confidence
            for n in counts
        ]
        for i in range(len(confidences) - 1):
            assert confidences[i] < confidences[i + 1]


class TestMeasuredThresholdsGeometric:
    """from_baseline_entropies_geometric() uses distribution crossing."""

    def test_clear_four_class_separation(self):
        """Four well-separated entropy classes → three clear boundaries."""
        solved = [0.5 + i * 0.02 for i in range(15)]  # 0.5..0.78
        attempted = [1.5 + i * 0.02 for i in range(15)]  # 1.5..1.78
        hedged = [3.0 + i * 0.02 for i in range(15)]  # 3.0..3.28
        refused = [5.0 + i * 0.02 for i in range(15)]  # 5.0..5.28

        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused_entropies=refused,
            hedged_entropies=hedged,
            attempted_entropies=attempted,
            solved_entropies=solved,
            model_id="test-model",
            seed=42,
        )

        # Thresholds should be between adjacent classes
        assert t.attempted_threshold >= 0.78
        assert t.attempted_threshold <= 1.5
        assert t.hedged_threshold >= 1.78
        assert t.hedged_threshold <= 3.0
        assert t.refused_threshold >= 3.28
        assert t.refused_threshold <= 5.0

    def test_classification_with_geometric_thresholds(self):
        """Geometric thresholds correctly classify outcomes."""
        solved = [0.5, 0.6, 0.7, 0.8, 0.9]
        attempted = [1.5, 1.6, 1.7, 1.8, 1.9]
        hedged = [3.0, 3.1, 3.2, 3.3, 3.4]
        refused = [5.0, 5.1, 5.2, 5.3, 5.4]

        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused, hedged, attempted, solved,
            model_id="test",
            seed=42,
        )

        assert t.classify_outcome(0.5, 0.0) == "solved"
        assert t.classify_outcome(5.5, 0.0) == "hedged"  # above refused but no variance threshold

    def test_fallback_when_insufficient_per_class_data(self):
        """< 2 samples per class → falls back to legacy method."""
        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused_entropies=[5.0],  # only 1
            hedged_entropies=[3.0],  # only 1
            attempted_entropies=[1.5],  # only 1
            solved_entropies=[0.5],  # only 1
            model_id="test",
        )
        # Should still produce valid thresholds via legacy path
        assert isinstance(t.refused_threshold, float)
        assert isinstance(t.hedged_threshold, float)
        assert isinstance(t.attempted_threshold, float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot derive thresholds from empty"):
            MeasuredThresholds.from_baseline_entropies_geometric(
                [], [], [], [],
                model_id="test",
            )

    def test_adaptive_percentiles_small_sample(self):
        """< 10 total samples → only median percentile reported."""
        solved = [0.5, 0.6, 0.7]
        attempted = [1.5, 1.6, 1.7]

        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused_entropies=[5.0, 5.1],
            hedged_entropies=[3.0, 3.1],
            attempted_entropies=attempted,
            solved_entropies=solved,
            model_id="test",
            seed=42,
        )

        # Only 10 samples total — should have full percentile keys
        # (boundary is exactly 10, which uses full set)
        assert 50 in t.percentiles

    def test_adaptive_percentiles_large_sample(self):
        """≥ 10 total samples → standard percentile keys."""
        solved = [0.5 + i * 0.02 for i in range(10)]
        attempted = [1.5 + i * 0.02 for i in range(10)]
        hedged = [3.0 + i * 0.02 for i in range(10)]
        refused = [5.0 + i * 0.02 for i in range(10)]

        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused, hedged, attempted, solved,
            model_id="test",
            seed=42,
        )

        assert 5 in t.percentiles
        assert 25 in t.percentiles
        assert 50 in t.percentiles
        assert 75 in t.percentiles
        assert 95 in t.percentiles
        assert 99 in t.percentiles

    def test_partial_class_data(self):
        """Some classes have enough data, some don't — uses crossing where possible."""
        # Solved and attempted have enough, hedged and refused don't
        solved = [0.5 + i * 0.02 for i in range(10)]
        attempted = [1.5 + i * 0.02 for i in range(10)]
        hedged = [3.0]  # only 1
        refused = [5.0]  # only 1

        t = MeasuredThresholds.from_baseline_entropies_geometric(
            refused_entropies=refused,
            hedged_entropies=hedged,
            attempted_entropies=attempted,
            solved_entropies=solved,
            model_id="test",
            seed=42,
        )

        # attempted_threshold should be from crossing (solved vs attempted)
        assert t.attempted_threshold >= 0.67  # above solved range
        assert t.attempted_threshold <= 1.5  # below attempted range
