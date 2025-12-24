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

"""
Tests for per-neuron sparsity analysis.
"""
import pytest
import math
from hypothesis import given, strategies as st, settings

from modelcypher.core.domain.geometry.neuron_sparsity_analyzer import (
    NeuronStats,
    NeuronSparsityMap,
    NeuronSparsityConfig,
    NeuronActivationCollector,
    compute_neuron_sparsity_map,
    compare_neuron_sparsity,
    identify_domain_specific_neurons,
)


class TestNeuronStats:
    """Tests for NeuronStats dataclass."""

    def test_sparsity_score_calculation(self):
        """Sparsity score should be 1 - active_fraction."""
        stats = NeuronStats(
            layer=0,
            neuron_idx=0,
            mean_activation=0.5,
            max_activation=1.0,
            min_activation=0.0,
            activation_variance=0.1,
            active_fraction=0.3,
            prompt_count=100,
        )

        assert stats.sparsity_score == 0.7

    def test_is_dead_detection(self):
        """Dead neurons should have near-zero max activation."""
        dead_neuron = NeuronStats(
            layer=0,
            neuron_idx=0,
            mean_activation=0.0,
            max_activation=1e-12,
            min_activation=0.0,
            activation_variance=0.0,
            active_fraction=0.0,
            prompt_count=100,
        )

        active_neuron = NeuronStats(
            layer=0,
            neuron_idx=1,
            mean_activation=0.5,
            max_activation=0.5,
            min_activation=0.1,
            activation_variance=0.01,
            active_fraction=0.9,
            prompt_count=100,
        )

        assert dead_neuron.is_dead
        assert not active_neuron.is_dead

    def test_coefficient_of_variation(self):
        """CV should be std / mean."""
        stats = NeuronStats(
            layer=0,
            neuron_idx=0,
            mean_activation=1.0,
            max_activation=2.0,
            min_activation=0.0,
            activation_variance=0.25,  # std = 0.5
            active_fraction=1.0,
            prompt_count=100,
        )

        # CV = 0.5 / 1.0 = 0.5
        assert abs(stats.coefficient_of_variation - 0.5) < 0.01


class TestNeuronSparsityMap:
    """Tests for NeuronSparsityMap."""

    @pytest.fixture
    def sample_map(self):
        """Create sample sparsity map."""
        config = NeuronSparsityConfig(sparsity_threshold=0.8, dead_neuron_threshold=0.99)
        stats = {
            0: [
                NeuronStats(0, 0, 0.5, 1.0, 0.0, 0.1, 0.1, 100),  # sparse (0.9)
                NeuronStats(0, 1, 0.5, 1.0, 0.0, 0.1, 0.9, 100),  # active (0.1)
                NeuronStats(0, 2, 0.0, 1e-12, 0.0, 0.0, 0.0, 100),  # dead (1.0)
            ],
            1: [
                NeuronStats(1, 0, 0.3, 0.8, 0.0, 0.05, 0.5, 100),  # medium (0.5)
                NeuronStats(1, 1, 0.1, 0.2, 0.0, 0.01, 0.15, 100),  # sparse (0.85)
            ],
        }
        return NeuronSparsityMap(stats=stats, config=config, total_prompts=100)

    def test_sparse_neurons_detection(self, sample_map):
        """Should identify neurons with sparsity >= threshold."""
        sparse = sample_map.sparse_neurons

        assert 0 in sparse
        assert 0 in sparse[0]  # neuron 0 in layer 0 is sparse
        assert 2 in sparse[0]  # neuron 2 in layer 0 is dead (also sparse)
        assert 1 not in sparse[0]  # neuron 1 in layer 0 is active

        assert 1 in sparse
        assert 1 in sparse[1]  # neuron 1 in layer 1 is sparse

    def test_dead_neurons_detection(self, sample_map):
        """Should identify neurons that never activate."""
        dead = sample_map.dead_neurons

        assert 0 in dead
        assert 2 in dead[0]  # only neuron 2 in layer 0 is dead
        assert 0 not in dead[0]  # neuron 0 is sparse but not dead

    def test_graft_candidates(self, sample_map):
        """Should return neurons sparse enough for grafting."""
        # Default threshold from config (0.8)
        candidates = sample_map.get_graft_candidates()
        assert 0 in candidates
        assert 1 in candidates

        # Higher threshold
        strict_candidates = sample_map.get_graft_candidates(threshold=0.95)
        assert 0 in strict_candidates
        assert 2 in strict_candidates[0]  # dead neuron has sparsity 1.0

    def test_layer_summary(self, sample_map):
        """Should compute correct layer statistics."""
        summary = sample_map.get_layer_summary(0)

        assert summary["total_neurons"] == 3
        assert summary["sparse_count"] == 2  # neurons 0 and 2
        assert summary["dead_count"] == 1  # only neuron 2

    def test_overall_summary(self, sample_map):
        """Should compute correct overall statistics."""
        summary = sample_map.summary()

        assert summary["num_layers"] == 2
        assert summary["total_neurons"] == 5
        assert summary["total_prompts"] == 100


class TestNeuronActivationCollector:
    """Tests for NeuronActivationCollector."""

    def test_add_sample(self):
        """Should accumulate activations correctly."""
        collector = NeuronActivationCollector()

        # Add two samples
        collector.add_sample({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
        collector.add_sample({0: [0.5, 1.5, 2.5], 1: [3.5, 4.5, 5.5]})

        assert collector._sample_count == 2

    def test_compute_sparsity_map(self):
        """Should compute correct statistics from collected activations."""
        collector = NeuronActivationCollector(
            NeuronSparsityConfig(activation_threshold=0.1)
        )

        # Add samples where neuron 0 is always active, neuron 1 sometimes
        for _ in range(10):
            collector.add_sample({0: [1.0, 0.05]})  # neuron 0 active, neuron 1 not

        for _ in range(10):
            collector.add_sample({0: [1.0, 0.5]})  # both active

        sparsity_map = collector.compute_sparsity_map()

        # Layer 0 should have 2 neurons
        assert 0 in sparsity_map.stats
        assert len(sparsity_map.stats[0]) == 2

        # Neuron 0 should have 100% active fraction
        neuron_0 = sparsity_map.stats[0][0]
        assert neuron_0.active_fraction == 1.0

        # Neuron 1 should have 50% active fraction
        neuron_1 = sparsity_map.stats[0][1]
        assert neuron_1.active_fraction == 0.5

    def test_clear(self):
        """Should clear collected data."""
        collector = NeuronActivationCollector()
        collector.add_sample({0: [1.0, 2.0]})

        collector.clear()

        assert collector._sample_count == 0
        assert len(collector._activations) == 0


class TestComputeNeuronSparsityMap:
    """Tests for compute_neuron_sparsity_map function."""

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        result = compute_neuron_sparsity_map({})

        assert result.total_prompts == 0
        assert len(result.stats) == 0

    def test_single_layer_single_prompt(self):
        """Should handle minimal input."""
        activations = {0: [[1.0, 0.0, 0.5]]}  # 1 prompt, 3 neurons

        result = compute_neuron_sparsity_map(activations)

        assert result.total_prompts == 1
        assert 0 in result.stats
        assert len(result.stats[0]) == 3


class TestCompareNeuronSparsity:
    """Tests for compare_neuron_sparsity function."""

    def test_graft_candidate_identification(self):
        """Should identify neurons sparse in target but not source."""
        config = NeuronSparsityConfig(sparsity_threshold=0.8)

        # Source: neuron 0 active, neuron 1 sparse
        source_stats = {
            0: [
                NeuronStats(0, 0, 0.5, 1.0, 0.0, 0.1, 0.9, 100),  # active
                NeuronStats(0, 1, 0.1, 0.2, 0.0, 0.01, 0.1, 100),  # sparse
            ]
        }
        source_map = NeuronSparsityMap(stats=source_stats, config=config, total_prompts=100)

        # Target: neuron 0 sparse, neuron 1 active
        target_stats = {
            0: [
                NeuronStats(0, 0, 0.1, 0.2, 0.0, 0.01, 0.1, 100),  # sparse
                NeuronStats(0, 1, 0.5, 1.0, 0.0, 0.1, 0.9, 100),  # active
            ]
        }
        target_map = NeuronSparsityMap(stats=target_stats, config=config, total_prompts=100)

        comparison = compare_neuron_sparsity(source_map, target_map)

        # Neuron 0 is sparse in target, active in source -> graft candidate
        assert 0 in comparison["graft_candidates"]
        assert 0 in comparison["graft_candidates"][0]


class TestPropertyBasedTests:
    """Property-based tests for mathematical invariants."""

    @given(active_fraction=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_sparsity_score_bounded(self, active_fraction):
        """Sparsity score must be in [0, 1]."""
        stats = NeuronStats(
            layer=0,
            neuron_idx=0,
            mean_activation=0.5,
            max_activation=1.0,
            min_activation=0.0,
            activation_variance=0.1,
            active_fraction=active_fraction,
            prompt_count=100,
        )

        assert 0.0 <= stats.sparsity_score <= 1.0

    @given(active_fraction=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_sparsity_plus_active_equals_one(self, active_fraction):
        """Sparsity + active_fraction should equal 1."""
        stats = NeuronStats(
            layer=0,
            neuron_idx=0,
            mean_activation=0.5,
            max_activation=1.0,
            min_activation=0.0,
            activation_variance=0.1,
            active_fraction=active_fraction,
            prompt_count=100,
        )

        assert abs(stats.sparsity_score + stats.active_fraction - 1.0) < 1e-10
