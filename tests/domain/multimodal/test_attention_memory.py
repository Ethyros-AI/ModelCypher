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

import unittest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain.multimodal.attention_memory import (
    AttentionMemoryInjector,
    LayerType,
    LayerTypeConfig,
    MemoryTokenContent,
    get_architecture_config,
    register_architecture,
    KNOWN_ARCHITECTURES,
)


class TestLayerTypeDetection(unittest.TestCase):
    """Tests for hybrid architecture layer type detection."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_lfm2_architecture_known(self):
        """Test that LFM2 architecture is in known configurations."""
        config = get_architecture_config("LFM2")
        self.assertIsNotNone(config)
        self.assertEqual(config.n_layers, 16)
        self.assertEqual(config.attention_layers, [2, 5, 8, 10, 12, 14])

    def test_detect_known_architecture(self):
        """Test layer type detection for known architecture."""
        layer_types = self.injector.detect_layer_types(architecture_name="LFM2")

        # Check attention layers
        self.assertEqual(layer_types[8], LayerType.ATTENTION)
        self.assertEqual(layer_types[10], LayerType.ATTENTION)

        # Check conv layers
        self.assertEqual(layer_types[7], LayerType.CONV)
        self.assertEqual(layer_types[9], LayerType.CONV)

    def test_detect_from_config(self):
        """Test layer type detection from model config dict."""
        config = {
            "num_hidden_layers": 12,
            "attention_layers": [3, 6, 9],
            "conv_layers": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        }
        layer_types = self.injector.detect_layer_types(model_config=config)

        self.assertEqual(layer_types[3], LayerType.ATTENTION)
        self.assertEqual(layer_types[6], LayerType.ATTENTION)
        self.assertEqual(layer_types[0], LayerType.CONV)

    def test_fallback_all_attention(self):
        """Test fallback to all attention when no hybrid info."""
        config = {"num_hidden_layers": 24}
        layer_types = self.injector.detect_layer_types(model_config=config)

        for i in range(24):
            self.assertEqual(layer_types[i], LayerType.ATTENTION)

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
        self.assertEqual(retrieved.n_layers, 8)
        self.assertEqual(retrieved.attention_layers, [2, 4, 6])

        # Cleanup
        del KNOWN_ARCHITECTURES["TestArch"]


class TestOptimalMemoryLayers(unittest.TestCase):
    """Tests for optimal memory layer selection."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_optimal_layers_lfm2(self):
        """Test optimal layer selection for LFM2."""
        layer_types = self.injector.detect_layer_types(architecture_name="LFM2")
        optimal = self.injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(7, 8, 9)
        )

        # Layer 8 is the only attention layer in highway 7-9
        self.assertIn(8, optimal)
        # Layer 7 and 9 are conv, should not be included
        self.assertNotIn(7, optimal)
        self.assertNotIn(9, optimal)

    def test_optimal_layers_all_attention(self):
        """Test optimal layers when all are attention."""
        layer_types = {i: LayerType.ATTENTION for i in range(24)}
        optimal = self.injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(7, 8, 9)
        )

        # All highway layers should be included
        self.assertEqual(set(optimal), {7, 8, 9})

    def test_fallback_when_no_highway_attention(self):
        """Test fallback when no attention in highway."""
        layer_types = {
            0: LayerType.ATTENTION,
            1: LayerType.CONV,
            2: LayerType.CONV,
            3: LayerType.CONV,
            4: LayerType.CONV,
            5: LayerType.ATTENTION,
        }
        optimal = self.injector.get_optimal_memory_layers(
            layer_types, semantic_highway=(2, 3, 4)  # All conv
        )

        # Should pick nearest attention layer(s)
        self.assertTrue(len(optimal) > 0)
        # Should include layer 5 (nearest to mid=3)
        self.assertIn(5, optimal)


class TestNullBasisComputation(unittest.TestCase):
    """Tests for null-space basis computation."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_null_basis_shape(self):
        """Test that null basis has correct shape."""
        # Simulate activations (5 samples, 128 dim)
        activations = mx.random.normal((5, 128))
        mx.eval(activations)

        null_basis = self.injector.compute_null_basis(activations, null_rank=64)

        self.assertEqual(null_basis.shape, (64, 128))

    def test_null_basis_caching(self):
        """Test that null basis is cached."""
        activations = mx.random.normal((5, 128))
        mx.eval(activations)

        # Compute with cache key
        basis1 = self.injector.compute_null_basis(
            activations, null_rank=32, cache_key="test"
        )

        # Different activations, same cache key should return cached
        other_activations = mx.random.normal((5, 128))
        mx.eval(other_activations)
        basis2 = self.injector.compute_null_basis(
            other_activations, null_rank=32, cache_key="test"
        )

        # Should be the same object (cached)
        self.assertTrue(mx.array_equal(basis1, basis2))

        # Cleanup cache
        self.injector._null_basis_cache.clear()


class TestMemoryContentComputation(unittest.TestCase):
    """Tests for memory token content computation."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_direction_steering(self):
        """Test direction steering (source - neutral)."""
        source = mx.ones((1, 128)) * 2.0
        neutral = mx.ones((1, 128))
        mx.eval(source, neutral)

        result = self.injector.compute_memory_content(
            source, neutral, null_basis=None, scale=1.0, use_null_space=False
        )

        self.assertIsInstance(result, MemoryTokenContent)
        # Direction should be (2-1) * 1.0 = 1.0 per element
        # Norm = sqrt(128 * 1^2) = sqrt(128)
        self.assertAlmostEqual(result.direction_norm, 128 ** 0.5, places=3)

    def test_scale_applied(self):
        """Test that scale is applied correctly."""
        source = mx.ones((1, 128))
        neutral = mx.zeros((1, 128))
        mx.eval(source, neutral)

        result_scale_1 = self.injector.compute_memory_content(
            source, neutral, scale=1.0, use_null_space=False
        )
        result_scale_10 = self.injector.compute_memory_content(
            source, neutral, scale=10.0, use_null_space=False
        )

        # Content norm should be 10x
        norm_1 = float(mx.sqrt(mx.sum(result_scale_1.content ** 2)))
        norm_10 = float(mx.sqrt(mx.sum(result_scale_10.content ** 2)))

        self.assertAlmostEqual(norm_10 / norm_1, 10.0, places=3)

    def test_null_space_projection(self):
        """Test that null-space projection works."""
        source = mx.random.normal((1, 128))
        neutral = mx.zeros((1, 128))
        mx.eval(source, neutral)

        # Create simple null basis
        activations = mx.random.normal((10, 128))
        mx.eval(activations)
        null_basis = self.injector.compute_null_basis(activations, null_rank=64)

        result = self.injector.compute_memory_content(
            source, neutral, null_basis=null_basis, scale=5.0, use_null_space=True
        )

        self.assertTrue(result.null_space_projected)
        # Content should be in null-space (lower rank)
        # Just verify it's not zero and has reasonable norm
        content_norm = float(mx.sqrt(mx.sum(result.content ** 2)))
        self.assertGreater(content_norm, 0)


class TestMemoryTokenValidation(unittest.TestCase):
    """Tests for memory token scale measurement (informational only)."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_measurement_returns_info(self):
        """Test that validation returns informational measurement."""
        # Geometry handles safety - this is just measurement
        memory = MemoryTokenContent(
            content=mx.ones((1, 128)) * 0.1,
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=True,
            direction_norm=1.0,
        )
        layer_activations = mx.ones((10, 128))
        mx.eval(layer_activations)

        is_valid, msg = self.injector.validate_memory_scale(
            memory, layer_activations
        )

        # Always valid - geometry handles safety by construction
        self.assertTrue(is_valid)
        self.assertIn("magnitude", msg.lower())

    def test_measurement_reports_projection_status(self):
        """Test that measurement reports whether null-space projected."""
        memory = MemoryTokenContent(
            content=mx.ones((1, 128)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=True,
            direction_norm=1.0,
        )
        layer_activations = mx.ones((10, 128))
        mx.eval(layer_activations)

        is_valid, msg = self.injector.validate_memory_scale(
            memory, layer_activations
        )

        self.assertTrue(is_valid)
        self.assertIn("null-space projected", msg)


class TestApplyMemoryToHiddenStates(unittest.TestCase):
    """Tests for applying memory token to hidden states."""

    def setUp(self):
        self.injector = AttentionMemoryInjector()

    def test_apply_at_position_0(self):
        """Test applying memory at position 0 (prepended)."""
        hidden = mx.ones((2, 10, 128))  # batch=2, seq=10, dim=128
        memory = MemoryTokenContent(
            content=mx.zeros((1, 128)),  # Distinct from ones
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        mx.eval(hidden)

        result = self.injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=0
        )

        # Shape should be unchanged
        self.assertEqual(result.shape, hidden.shape)

        # Position 0 should be zeros (memory)
        pos0_sum = float(mx.sum(result[:, 0, :]))
        self.assertAlmostEqual(pos0_sum, 0.0, places=5)

        # Other positions should be ones
        pos1_sum = float(mx.sum(result[:, 1, :]))
        self.assertGreater(pos1_sum, 0)

    def test_apply_at_last_position(self):
        """Test applying memory at last position."""
        hidden = mx.ones((1, 5, 64))
        memory = MemoryTokenContent(
            content=mx.zeros((1, 64)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        mx.eval(hidden)

        result = self.injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=4
        )

        # Last position should be zeros
        last_sum = float(mx.sum(result[:, -1, :]))
        self.assertAlmostEqual(last_sum, 0.0, places=5)

    def test_apply_at_middle_position(self):
        """Test applying memory at middle position."""
        hidden = mx.ones((1, 5, 64))
        memory = MemoryTokenContent(
            content=mx.zeros((1, 64)),
            source_concept="test",
            scale_applied=1.0,
            null_space_projected=False,
            direction_norm=1.0,
        )
        mx.eval(hidden)

        result = self.injector.apply_memory_to_hidden_states(
            hidden, memory, memory_position=2
        )

        # Position 2 should be zeros
        pos2_sum = float(mx.sum(result[:, 2, :]))
        self.assertAlmostEqual(pos2_sum, 0.0, places=5)

        # Positions 0, 1, 3, 4 should be ones
        pos0_sum = float(mx.sum(result[:, 0, :]))
        self.assertGreater(pos0_sum, 0)


class TestDeviationTrackerAPI(unittest.TestCase):
    """Test deviation tracker API is available."""

    def test_deviation_tracker_exists(self):
        """Test DeviationTracker class is importable."""
        from modelcypher.core.domain.geometry.deviation_budget import DeviationTracker

        # Should be able to instantiate
        tracker = DeviationTracker()
        self.assertIsNotNone(tracker)

    def test_deviation_measurement_exists(self):
        """Test DeviationMeasurement dataclass is importable."""
        from modelcypher.core.domain.geometry.deviation_budget import DeviationMeasurement

        # Should be able to create measurement
        measurement = DeviationMeasurement(
            deviation=1.0,
            baseline_norm=100.0,
            deviation_percent=1.0,
            condition_number=10.0,
        )
        self.assertEqual(measurement.deviation, 1.0)


if __name__ == "__main__":
    unittest.main()
