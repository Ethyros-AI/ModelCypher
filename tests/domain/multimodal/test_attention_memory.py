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

"""Tests for attention-based memory token injection."""

import pytest

from modelcypher.core.domain.multimodal.attention_memory import (
    AttentionMemoryInjector,
    LayerType,
    LayerTypeConfig,
    MemoryTokenContent,
    get_architecture_config,
    register_architecture,
    KNOWN_ARCHITECTURES,
)


class TestLayerTypeDetection:
    """Tests for hybrid architecture layer type detection."""

    def test_lfm2_architecture_known(self):
        """Test that LFM2 architecture is in known configurations."""
        config = get_architecture_config("LFM2")
        assert config is not None
        assert config.n_layers == 16
        assert config.attention_layers == [2, 5, 8, 10, 12, 14]

    def test_detect_known_architecture(self):
        """Test layer type detection for known architecture."""
        injector = AttentionMemoryInjector()
        layer_types = injector.detect_layer_types(architecture_name="LFM2")

        assert layer_types[8] == LayerType.ATTENTION
        assert layer_types[10] == LayerType.ATTENTION
        assert layer_types[7] == LayerType.CONV
        assert layer_types[9] == LayerType.CONV

    def test_detect_from_config(self):
        """Test layer type detection from model config dict."""
        injector = AttentionMemoryInjector()
        config = {
            "num_hidden_layers": 12,
            "attention_layers": [3, 6, 9],
            "conv_layers": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        }
        layer_types = injector.detect_layer_types(model_config=config)

        assert layer_types[3] == LayerType.ATTENTION
        assert layer_types[6] == LayerType.ATTENTION
        assert layer_types[0] == LayerType.CONV

    def test_fallback_all_attention(self):
        """Test fallback to all attention when no hybrid info."""
        injector = AttentionMemoryInjector()
        config = {"num_hidden_layers": 24}
        layer_types = injector.detect_layer_types(model_config=config)

        for i in range(24):
            assert layer_types[i] == LayerType.ATTENTION

    def test_register_new_architecture(self):
        """Test registering a new architecture."""
        new_config = LayerTypeConfig(
            n_layers=8,
            attention_layers=[2, 4, 6],
            conv_layers=[0, 1, 3, 5, 7],
            semantic_highway=(3, 4, 5),
            hidden_dim=512,
            n_heads=8,
        )
        register_architecture("TestArch", new_config)

        retrieved = get_architecture_config("TestArch")
        assert retrieved.n_layers == 8
        assert retrieved.attention_layers == [2, 4, 6]

        # Cleanup
        del KNOWN_ARCHITECTURES["TestArch"]


class TestOptimalMemoryLayers:
    """Tests for optimal memory layer selection."""

    def test_optimal_layers_lfm2(self):
        """Test optimal layer selection for LFM2."""
        injector = AttentionMemoryInjector()
        layer_types = injector.detect_layer_types(architecture_name="LFM2")
        optimal = injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(7, 8, 9)
        )

        assert 8 in optimal
        assert 7 not in optimal
        assert 9 not in optimal

    def test_optimal_layers_all_attention(self):
        """Test optimal layers when all are attention."""
        injector = AttentionMemoryInjector()
        layer_types = {i: LayerType.ATTENTION for i in range(24)}
        optimal = injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(7, 8, 9)
        )

        assert set(optimal) == {7, 8, 9}

    def test_fallback_when_no_highway_attention(self):
        """Test fallback when no attention in highway."""
        injector = AttentionMemoryInjector()
        layer_types = {
            0: LayerType.ATTENTION,
            1: LayerType.CONV,
            2: LayerType.CONV,
            3: LayerType.CONV,
            4: LayerType.CONV,
            5: LayerType.ATTENTION,
        }
        optimal = injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(2, 3, 4)
        )

        assert len(optimal) > 0
        assert 5 in optimal


class TestNullBasisComputation:
    """Tests for null-space basis computation."""

    def test_null_basis_shape(self, any_backend):
        """Test that null basis has correct shape."""
        injector = AttentionMemoryInjector()
        activations = any_backend.random_normal((5, 128))
        any_backend.eval(activations)

        null_basis = injector.compute_null_basis(activations, null_rank=64)

        assert null_basis.shape == (64, 128)

    def test_null_basis_caching(self, any_backend):
        """Test that null basis is cached."""
        injector = AttentionMemoryInjector()
        activations = any_backend.random_normal((5, 128))
        any_backend.eval(activations)

        basis1 = injector.compute_null_basis(
            activations, null_rank=32, cache_key="test"
        )

        other_activations = any_backend.random_normal((5, 128))
        any_backend.eval(other_activations)
        basis2 = injector.compute_null_basis(
            other_activations, null_rank=32, cache_key="test"
        )

        # Cached basis should be identical - compare via sum difference
        diff = any_backend.sum(any_backend.abs(basis1 - basis2))
        any_backend.eval(diff)
        assert float(diff) < 1e-10

        injector._null_basis_cache.clear()


class TestMemoryContentComputation:
    """Tests for memory token content computation."""

    def test_direction_steering(self, any_backend):
        """Test direction steering (source - neutral)."""
        injector = AttentionMemoryInjector()
        source = any_backend.ones((1, 128)) * 2.0
        neutral = any_backend.ones((1, 128))
        any_backend.eval(source, neutral)

        result = injector.compute_memory_content(
            source, neutral, null_basis=None, scale=1.0, use_null_space=False
        )

        assert isinstance(result, MemoryTokenContent)
        assert abs(result.direction_norm - 128 ** 0.5) < 0.01

    def test_scale_applied(self, any_backend):
        """Test that scale is applied correctly."""
        injector = AttentionMemoryInjector()
        source = any_backend.ones((1, 128))
        neutral = any_backend.zeros((1, 128))
        any_backend.eval(source, neutral)

        result_scale_1 = injector.compute_memory_content(
            source, neutral, scale=1.0, use_null_space=False
        )
        result_scale_10 = injector.compute_memory_content(
            source, neutral, scale=10.0, use_null_space=False
        )

        norm_1 = float(any_backend.sqrt(any_backend.sum(result_scale_1.content ** 2)))
        norm_10 = float(any_backend.sqrt(any_backend.sum(result_scale_10.content ** 2)))

        assert abs(norm_10 / norm_1 - 10.0) < 0.01

    def test_null_space_projection(self, any_backend):
        """Test that null-space projection works."""
        injector = AttentionMemoryInjector()
        source = any_backend.random_normal((1, 128))
        neutral = any_backend.zeros((1, 128))
        any_backend.eval(source, neutral)

        activations = any_backend.random_normal((10, 128))
        any_backend.eval(activations)
        null_basis = injector.compute_null_basis(activations, null_rank=64)

        result = injector.compute_memory_content(
            source, neutral, null_basis=null_basis, scale=5.0, use_null_space=True
        )

        assert result.null_space_projected
        content_norm = float(any_backend.sqrt(any_backend.sum(result.content ** 2)))
        assert content_norm > 0


class TestMemoryTokenValidation:
    """Tests for memory token scale measurement (informational only)."""

    def test_measurement_returns_info(self, any_backend):
        """Test that validation returns informational measurement."""
        injector = AttentionMemoryInjector()
        memory = MemoryTokenContent(
            content=any_backend.ones((1, 128)) * 0.1,
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=True,
            direction_norm=1.0,
        )
        layer_activations = any_backend.ones((10, 128))
        any_backend.eval(layer_activations)

        is_valid, msg = injector.validate_memory_scale(memory, layer_activations)

        assert is_valid
        assert "magnitude" in msg.lower()

    def test_measurement_reports_projection_status(self, any_backend):
        """Test that measurement reports whether null-space projected."""
        injector = AttentionMemoryInjector()
        memory = MemoryTokenContent(
            content=any_backend.ones((1, 128)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=True,
            direction_norm=1.0,
        )
        layer_activations = any_backend.ones((10, 128))
        any_backend.eval(layer_activations)

        is_valid, msg = injector.validate_memory_scale(memory, layer_activations)

        assert is_valid
        assert "null-space projected" in msg


class TestApplyMemoryToHiddenStates:
    """Tests for applying memory token to hidden states."""

    def test_apply_at_position_0(self, any_backend):
        """Test applying memory at position 0 (prepended)."""
        injector = AttentionMemoryInjector()
        hidden = any_backend.ones((2, 10, 128))
        memory = MemoryTokenContent(
            content=any_backend.zeros((1, 128)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        any_backend.eval(hidden)

        result = injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=0
        )

        assert result.shape == hidden.shape
        pos0_sum = float(any_backend.sum(result[:, 0, :]))
        assert abs(pos0_sum) < 1e-5
        pos1_sum = float(any_backend.sum(result[:, 1, :]))
        assert pos1_sum > 0

    def test_apply_at_last_position(self, any_backend):
        """Test applying memory at last position."""
        injector = AttentionMemoryInjector()
        hidden = any_backend.ones((1, 5, 64))
        memory = MemoryTokenContent(
            content=any_backend.zeros((1, 64)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        any_backend.eval(hidden)

        result = injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=4
        )

        last_sum = float(any_backend.sum(result[:, -1, :]))
        assert abs(last_sum) < 1e-5

    def test_apply_at_middle_position(self, any_backend):
        """Test applying memory at middle position."""
        injector = AttentionMemoryInjector()
        hidden = any_backend.ones((1, 5, 64))
        memory = MemoryTokenContent(
            content=any_backend.zeros((1, 64)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        any_backend.eval(hidden)

        result = injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=2
        )

        pos2_sum = float(any_backend.sum(result[:, 2, :]))
        assert abs(pos2_sum) < 1e-5
        pos0_sum = float(any_backend.sum(result[:, 0, :]))
        assert pos0_sum > 0


class TestDeviationTrackerAPI:
    """Test deviation tracker API is available."""

    def test_deviation_tracker_exists(self):
        """Test DeviationTracker class is importable."""
        from modelcypher.core.domain.geometry.deviation_budget import DeviationTracker

        tracker = DeviationTracker()
        assert tracker is not None

    def test_deviation_measurement_exists(self):
        """Test DeviationMeasurement dataclass is importable."""
        from modelcypher.core.domain.geometry.deviation_budget import DeviationMeasurement

        measurement = DeviationMeasurement(
            deviation=1.0,
            baseline_norm=100.0,
            deviation_percent=1.0,
            condition_number=10.0,
        )
        assert measurement.deviation == 1.0
