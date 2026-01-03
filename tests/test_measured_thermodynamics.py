# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for measured thermodynamics components."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    log_scalar,
)
from modelcypher.core.domain.thermo.measured_thermodynamics import (
    MeasuredBasinTopology,
    MeasuredEnergy,
)


def _eps(*values: float) -> float:
    b = get_default_backend()
    scale = max(1.0, max(abs(value) for value in values)) if values else 1.0
    arr = b.array([scale])
    b.eval(arr)
    return division_epsilon(b, arr) * scale


class TestMeasuredEnergy:
    """Tests for MeasuredEnergy dataclass."""

    def test_from_probability_same_probability(self):
        """Test energy is 0 when p = p_ref."""
        energy = MeasuredEnergy.from_probability(
            probability=0.3,
            reference_probability=0.3,
            temperature=1.0,
            sample_count=100,
        )

        eps = _eps(energy.value, 0.0)
        assert abs(energy.value) <= eps
        assert energy.probability == 0.3
        assert energy.sample_count == 100

    def test_from_probability_lower_probability(self):
        """Test energy is positive when p < p_ref (less likely = higher energy)."""
        energy = MeasuredEnergy.from_probability(
            probability=0.1,
            reference_probability=0.3,
            temperature=1.0,
            sample_count=50,
        )

        # E = -T * log(p/p_ref) = -1 * log(0.1/0.3) = -log(1/3) = log(3) > 0
        assert energy.value > 0
        expected = -log_scalar(0.1 / 0.3, get_default_backend())
        eps = _eps(energy.value, expected)
        assert abs(energy.value - expected) <= eps

    def test_from_probability_higher_probability(self):
        """Test energy is negative when p > p_ref (more likely = lower energy)."""
        energy = MeasuredEnergy.from_probability(
            probability=0.5,
            reference_probability=0.2,
            temperature=1.0,
            sample_count=100,
        )

        # E = -T * log(p/p_ref) = -1 * log(0.5/0.2) = -log(2.5) < 0
        assert energy.value < 0

    def test_from_probability_temperature_scaling(self):
        """Test that energy scales with temperature."""
        energy_t1 = MeasuredEnergy.from_probability(
            probability=0.1,
            reference_probability=0.3,
            temperature=1.0,
            sample_count=100,
        )

        energy_t2 = MeasuredEnergy.from_probability(
            probability=0.1,
            reference_probability=0.3,
            temperature=2.0,
            sample_count=100,
        )

        # Energy should scale with temperature
        expected = 2.0 * energy_t1.value
        eps = _eps(energy_t2.value, expected)
        assert abs(energy_t2.value - expected) <= eps

    def test_confidence_low_samples(self):
        """Test confidence is low with few samples."""
        energy = MeasuredEnergy(
            value=1.0,
            probability=0.2,
            sample_count=5,
            temperature=1.0,
        )

        # With 5 samples: 1 - 1/(1 + 5/10) = 1 - 1/1.5 = 0.333
        expected = 1.0 - 1.0 / (1.0 + 5 / 10.0)
        eps = _eps(energy.confidence, expected)
        assert abs(energy.confidence - expected) <= eps

    def test_confidence_high_samples(self):
        """Test confidence is high with many samples."""
        energy = MeasuredEnergy(
            value=1.0,
            probability=0.2,
            sample_count=1000,
            temperature=1.0,
        )

        # With 1000 samples: 1 - 1/(1 + 100) = 1 - 1/101 ≈ 0.99
        expected = 1.0 - 1.0 / (1.0 + 1000 / 10.0)
        eps = _eps(energy.confidence, expected)
        assert abs(energy.confidence - expected) <= eps

    def test_numerical_stability_zero_probability(self):
        """Test handling of zero probability (clamped to 1e-10)."""
        energy = MeasuredEnergy.from_probability(
            probability=0.0,
            reference_probability=0.5,
            temperature=1.0,
            sample_count=10,
        )

        # Should not raise, energy should be very high (low probability)
        b = get_default_backend()
        expected = -log_scalar(b.finfo().tiny / 0.5, b)
        eps = _eps(energy.value, expected)
        assert abs(energy.value - expected) <= eps


class TestMeasuredBasinTopology:
    """Tests for MeasuredBasinTopology dataclass."""

    @pytest.fixture
    def balanced_topology(self):
        """Create topology with balanced outcome counts."""
        return MeasuredBasinTopology.from_outcome_counts(
            refused_count=25,
            hedged_count=25,
            attempted_count=25,
            solved_count=25,
            temperature=1.0,
            model_id="test-model",
        )

    @pytest.fixture
    def refusal_heavy_topology(self):
        """Create topology with high refusal rate."""
        return MeasuredBasinTopology.from_outcome_counts(
            refused_count=80,
            hedged_count=15,
            attempted_count=3,
            solved_count=2,
            temperature=1.0,
            model_id="test-model",
        )

    def test_refusal_energy_is_zero(self, balanced_topology):
        """Test refusal basin is the reference (E=0)."""
        assert balanced_topology.refusal_energy.value == 0.0

    def test_from_outcome_counts_zero_total_raises(self):
        """Test that zero total observations raises error."""
        with pytest.raises(ValueError, match="zero observations"):
            MeasuredBasinTopology.from_outcome_counts(
                refused_count=0,
                hedged_count=0,
                attempted_count=0,
                solved_count=0,
                temperature=1.0,
                model_id="test-model",
            )

    def test_balanced_counts_similar_energies(self, balanced_topology):
        """Test that balanced counts give similar energies."""
        # All outcomes equally likely = similar energies
        # (relative to refusal which is 0)
        eps = _eps(
            balanced_topology.caution_energy.value,
            balanced_topology.solution_energy.value,
        )
        assert abs(balanced_topology.caution_energy.value) <= eps
        assert abs(balanced_topology.solution_energy.value) <= eps

    def test_refusal_heavy_high_solution_energy(self, refusal_heavy_topology):
        """Test that high refusal rate makes solution high energy."""
        # Solution is rare = high energy relative to refusal
        assert refusal_heavy_topology.solution_energy.value > 0

    def test_escape_probability_at_measurement_temperature(self, balanced_topology):
        """Test escape probability at measurement temperature."""
        p_escape = balanced_topology.escape_probability(1.0)

        # Should be between 0 and 1
        eps = _eps(p_escape, 0.0, 1.0)
        assert -eps <= p_escape <= 1.0 + eps

    def test_escape_probability_increases_with_temperature(self, refusal_heavy_topology):
        """Test that escape probability increases with temperature."""
        p_low = refusal_heavy_topology.escape_probability(0.5)
        p_high = refusal_heavy_topology.escape_probability(2.0)

        # Higher temperature = easier to escape
        assert p_high >= p_low

    def test_escape_probability_zero_temperature(self, balanced_topology):
        """Test escape probability is 0 at T=0."""
        p_escape = balanced_topology.escape_probability(0.0)
        eps = _eps(p_escape, 0.0)
        assert abs(p_escape) <= eps

    def test_basin_weights_sum_to_one(self, balanced_topology):
        """Test that basin weights sum to 1."""
        weights = balanced_topology.basin_weights(1.0)

        assert len(weights) == 3
        total = sum(weights)
        eps = _eps(total, 1.0)
        assert abs(total - 1.0) <= eps

    def test_basin_weights_length(self, balanced_topology):
        """Test basin weight list has 3 elements."""
        weights = balanced_topology.basin_weights(1.0)
        assert len(weights) == 3

    def test_basin_weights_refusal_heavy(self, refusal_heavy_topology):
        """Test basin weights favor first basin when refusal rate is high."""
        weights = refusal_heavy_topology.basin_weights(1.0)

        # First basin (index 0) should have higher weight (lower energy = higher probability)
        assert weights[0] > weights[2]

    def test_model_id_preserved(self, balanced_topology):
        """Test model_id is preserved."""
        assert balanced_topology.model_id == "test-model"

    def test_temperature_preserved(self, balanced_topology):
        """Test temperature is preserved."""
        assert balanced_topology.temperature == 1.0

    def test_custom_escape_rate(self):
        """Test custom escape rate override."""
        topology = MeasuredBasinTopology.from_outcome_counts(
            refused_count=50,
            hedged_count=30,
            attempted_count=15,
            solved_count=5,
            temperature=1.0,
            model_id="test-model",
            escape_rate=0.5,  # Override
        )

        # Should use custom escape rate, not derived
        # Ridge energy depends on escape rate
        assert topology.transition_ridge.probability == 0.5


class TestThermodynamicProperties:
    """Tests for thermodynamic properties and relationships."""

    def test_energy_ordering_by_probability(self):
        """Test that energy ordering matches probability ordering."""
        topology = MeasuredBasinTopology.from_outcome_counts(
            refused_count=50,  # Most common
            hedged_count=30,
            attempted_count=15,
            solved_count=5,  # Least common
            temperature=1.0,
            model_id="test",
        )

        # Lower probability = higher energy (relative to refusal at 0)
        # refusal(50) < caution(30) < solution(5)
        assert topology.caution_energy.value > topology.refusal_energy.value
        assert topology.solution_energy.value > topology.caution_energy.value

    def test_boltzmann_relationship(self):
        """Test Boltzmann relationship: p_i/p_j = exp(-(E_i - E_j)/T)."""
        topology = MeasuredBasinTopology.from_outcome_counts(
            refused_count=40,
            hedged_count=30,
            attempted_count=20,
            solved_count=10,
            temperature=1.0,
            model_id="test",
        )

        # Check Boltzmann ratio for caution vs refusal
        p_caution = 30 / 100
        p_refusal = 40 / 100
        measured_ratio = p_caution / p_refusal

        # From E = -T * log(p/p_ref):
        # p/p_ref = exp(-E/T) when refusal is reference
        delta_e = topology.caution_energy.value - topology.refusal_energy.value
        predicted_ratio = exp_scalar(-delta_e / 1.0, get_default_backend())

        eps = _eps(measured_ratio, predicted_ratio)
        assert abs(measured_ratio - predicted_ratio) <= eps

    def test_temperature_affects_weights(self):
        """Test that temperature affects basin weight distribution."""
        topology = MeasuredBasinTopology.from_outcome_counts(
            refused_count=60,
            hedged_count=20,
            attempted_count=15,
            solved_count=5,
            temperature=1.0,
            model_id="test",
        )

        weights_cold = topology.basin_weights(0.5)
        weights_hot = topology.basin_weights(2.0)

        # At low temperature, weight concentrates in lowest energy (basin 0)
        # At high temperature, weights become more uniform
        # So basin 0 weight should be higher at low temperature
        assert weights_cold[0] > weights_hot[0]

    def test_confidence_increases_with_samples(self):
        """Test that confidence in energy measurements increases with samples."""
        few_samples = MeasuredBasinTopology.from_outcome_counts(
            refused_count=5,
            hedged_count=3,
            attempted_count=1,
            solved_count=1,
            temperature=1.0,
            model_id="test",
        )

        many_samples = MeasuredBasinTopology.from_outcome_counts(
            refused_count=500,
            hedged_count=300,
            attempted_count=150,
            solved_count=50,
            temperature=1.0,
            model_id="test",
        )

        # More samples = higher confidence
        assert many_samples.refusal_energy.confidence > few_samples.refusal_energy.confidence
