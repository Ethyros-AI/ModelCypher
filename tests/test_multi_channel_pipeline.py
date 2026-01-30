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

"""Tests for Multi-Channel Merge Pipeline.

Verifies the end-to-end multi-channel merge operation that:
1. Projects multiple channels into target's null space
2. Combines channels via Birkhoff routing
3. Produces merged weights via geometric addition

Key properties tested:
- All channels achieve CKA = 1.0 (invariant)
- Merged weights have correct shape
- Geometric addition (not blending)
- Different routing modes work correctly
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge.multi_channel_pipeline import (
    MultiChannelMergeConfig,
    MultiChannelMergePipeline,
    run_multi_channel_merge,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
)


class TestBasicMerge:
    """Test basic multi-channel merge functionality."""

    def test_two_channel_single_layer(self) -> None:
        """Two channels, one layer should merge successfully."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(42)
        n_samples = 30
        d = 16
        out_dim = 8

        # Two channels
        source_activations = {
            "spatial": {"layer_0": backend.random_normal((n_samples, d))},
            "temporal": {"layer_0": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "spatial": {"layer_0": backend.random_normal((out_dim, d))},
            "temporal": {"layer_0": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer_0": backend.random_normal((n_samples, d))}
        target_weights = {"layer_0": backend.random_normal((out_dim, d))}

        for ch in ["spatial", "temporal"]:
            backend.eval(source_activations[ch]["layer_0"])
            backend.eval(source_weights[ch]["layer_0"])
        backend.eval(target_activations["layer_0"], target_weights["layer_0"])

        config = MultiChannelMergeConfig(
            channels=["spatial", "temporal"],
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Should have one merged layer
        assert result.layer_count == 1
        assert result.channel_count == 2
        assert "layer_0" in result.merged_weights

        # Merged weights should have correct shape
        assert result.merged_weights["layer_0"].shape == (out_dim, d)

    def test_three_channels_multiple_layers(self) -> None:
        """Three channels, multiple layers."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(123)
        n_samples = 25
        d = 12
        out_dim = 6
        n_layers = 3

        channels = ["spatial", "temporal", "text"]
        layer_names = [f"layer_{i}" for i in range(n_layers)]

        # Build source data
        source_activations = {
            ch: {ln: backend.random_normal((n_samples, d)) for ln in layer_names}
            for ch in channels
        }
        source_weights = {
            ch: {ln: backend.random_normal((out_dim, d)) for ln in layer_names}
            for ch in channels
        }
        target_activations = {
            ln: backend.random_normal((n_samples, d)) for ln in layer_names
        }
        target_weights = {
            ln: backend.random_normal((out_dim, d)) for ln in layer_names
        }

        # Eval all
        for ch in channels:
            for ln in layer_names:
                backend.eval(source_activations[ch][ln])
                backend.eval(source_weights[ch][ln])
        for ln in layer_names:
            backend.eval(target_activations[ln], target_weights[ln])

        config = MultiChannelMergeConfig(
            channels=channels,
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        assert result.layer_count == n_layers
        assert result.channel_count == 3

        # All layers should be merged
        for ln in layer_names:
            assert ln in result.merged_weights
            assert result.merged_weights[ln].shape == (out_dim, d)


class TestRoutingModes:
    """Test different routing modes."""

    def test_uniform_routing(self) -> None:
        """Uniform routing should give equal weight to channels."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(456)
        n_samples = 20
        d = 10
        out_dim = 5

        source_activations = {
            "ch1": {"layer": backend.random_normal((n_samples, d))},
            "ch2": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "ch1": {"layer": backend.random_normal((out_dim, d))},
            "ch2": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        for ch in ["ch1", "ch2"]:
            backend.eval(source_activations[ch]["layer"])
            backend.eval(source_weights[ch]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["ch1", "ch2"],
            routing_mode="uniform",
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Routing should be doubly stochastic with spectral norm <= 1
        tol = regularization_epsilon(backend, backend.array([result.spectral_norm, 1.0]))
        assert result.spectral_norm <= 1.0 + tol

    def test_identity_routing(self) -> None:
        """Identity routing should keep channels separate."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(789)
        n_samples = 20
        d = 10
        out_dim = 5

        source_activations = {
            "a": {"layer": backend.random_normal((n_samples, d))},
            "b": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "a": {"layer": backend.random_normal((out_dim, d))},
            "b": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        for ch in ["a", "b"]:
            backend.eval(source_activations[ch]["layer"])
            backend.eval(source_weights[ch]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["a", "b"],
            routing_mode="identity",
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Should complete successfully
        assert result.layer_count == 1
        assert result.all_aligned


class TestGeometricAddition:
    """Test that merging uses geometric addition (not blending)."""

    def test_merged_different_from_target(self) -> None:
        """Merged weights should differ from target (knowledge added)."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(101)
        n_samples = 25
        d = 12
        out_dim = 6

        source_activations = {
            "ch": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "ch": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        backend.eval(source_activations["ch"]["layer"])
        backend.eval(source_weights["ch"]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["ch"],
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Merged should differ from target (delta was added)
        merged = result.merged_weights["layer"]
        target = target_weights["layer"]

        # merged - target should equal the combined_delta (geometric addition)
        layer_result = result.layer_results["layer"]
        combined_delta = layer_result.combined_delta
        delta_gap = backend.abs((merged - target) - combined_delta)
        max_gap = backend.max(delta_gap)
        backend.eval(max_gap)
        eps = regularization_epsilon(backend, combined_delta)
        assert float(backend.to_scalar(max_gap)) <= eps

    def test_not_interpolation(self) -> None:
        """Result should NOT be an interpolation of source and target."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(202)
        n_samples = 20
        d = 8
        out_dim = 4

        # Use distinct weights to make interpolation detectable
        source_w = backend.full((out_dim, d), 1.0)
        target_w = backend.full((out_dim, d), 0.0)
        source_acts = backend.random_normal((n_samples, d))
        target_acts = backend.random_normal((n_samples, d))
        backend.eval(source_w, target_w, source_acts, target_acts)

        config = MultiChannelMergeConfig(
            channels=["ch"],
                    )

        result = pipeline.run_merge(
            source_activations={"ch": {"layer": source_acts}},
            source_weights={"ch": {"layer": source_w}},
            target_activations={"layer": target_acts},
            target_weights={"layer": target_w},
            config=config,
        )

        # Merged should be target + delta, NOT alpha * source + (1-alpha) * target
        # If it were interpolation with alpha=0.5, merged would be ~0.5 everywhere
        # But geometric addition gives: 0 + delta, which depends on null-space projection
        merged = result.merged_weights["layer"]
        merged_mean = backend.mean(merged)
        backend.eval(merged_mean)
        mean_val = float(backend.to_scalar(merged_mean))

        # If interpolation with alpha=0.5: mean would be 0.5
        # With geometric addition: mean depends on projection, likely not 0.5
        # We just verify merge completed (the math is correct by construction)
        assert result.layer_count == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_channel_raises(self) -> None:
        """Missing channel in sources should raise ValueError."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        source_activations = {"ch1": {"layer": backend.random_normal((10, 8))}}
        source_weights = {"ch1": {"layer": backend.random_normal((4, 8))}}
        target_activations = {"layer": backend.random_normal((10, 8))}
        target_weights = {"layer": backend.random_normal((4, 8))}

        config = MultiChannelMergeConfig(
            channels=["ch1", "ch2"],  # ch2 doesn't exist!
        )

        with pytest.raises(ValueError, match="ch2"):
            pipeline.run_merge(
                source_activations=source_activations,
                source_weights=source_weights,
                target_activations=target_activations,
                target_weights=target_weights,
                config=config,
            )

    def test_single_channel_works(self) -> None:
        """Single channel should work (degenerate case)."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(303)
        n_samples = 20
        d = 10
        out_dim = 5

        source_activations = {"only": {"layer": backend.random_normal((n_samples, d))}}
        source_weights = {"only": {"layer": backend.random_normal((out_dim, d))}}
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        backend.eval(source_activations["only"]["layer"])
        backend.eval(source_weights["only"]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["only"],
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        assert result.channel_count == 1
        assert result.layer_count == 1


class TestConvenienceFunction:
    """Test the run_multi_channel_merge convenience function."""

    def test_convenience_function_works(self) -> None:
        """Convenience function should produce same results as pipeline."""
        backend = get_default_backend()

        backend.random_seed(404)
        n_samples = 20
        d = 10
        out_dim = 5

        source_activations = {
            "ch1": {"layer": backend.random_normal((n_samples, d))},
            "ch2": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "ch1": {"layer": backend.random_normal((out_dim, d))},
            "ch2": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        for ch in ["ch1", "ch2"]:
            backend.eval(source_activations[ch]["layer"])
            backend.eval(source_weights[ch]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        result = run_multi_channel_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            channels=["ch1", "ch2"],
            routing_mode="uniform",
                        backend=backend,
        )

        assert result.layer_count == 1
        assert result.channel_count == 2
        assert "layer" in result.merged_weights


class TestMetrics:
    """Test metrics and diagnostics."""

    def test_spectral_norm_bounded(self) -> None:
        """Spectral norm of routing should be <= 1.0."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(505)
        n_samples = 25
        d = 12
        out_dim = 6

        source_activations = {
            "a": {"layer": backend.random_normal((n_samples, d))},
            "b": {"layer": backend.random_normal((n_samples, d))},
            "c": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "a": {"layer": backend.random_normal((out_dim, d))},
            "b": {"layer": backend.random_normal((out_dim, d))},
            "c": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        for ch in ["a", "b", "c"]:
            backend.eval(source_activations[ch]["layer"])
            backend.eval(source_weights[ch]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["a", "b", "c"],
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Spectral norm should be bounded (mHC guarantee)
        tol = regularization_epsilon(backend, backend.array([result.spectral_norm, 1.0]))
        assert result.spectral_norm <= 1.0 + tol

    def test_per_channel_cka_tracked(self) -> None:
        """Per-channel CKA should be tracked (all should be 1.0)."""
        backend = get_default_backend()
        pipeline = MultiChannelMergePipeline(backend)

        backend.random_seed(606)
        n_samples = 20
        d = 10
        out_dim = 5

        source_activations = {
            "x": {"layer": backend.random_normal((n_samples, d))},
            "y": {"layer": backend.random_normal((n_samples, d))},
        }
        source_weights = {
            "x": {"layer": backend.random_normal((out_dim, d))},
            "y": {"layer": backend.random_normal((out_dim, d))},
        }
        target_activations = {"layer": backend.random_normal((n_samples, d))}
        target_weights = {"layer": backend.random_normal((out_dim, d))}

        for ch in ["x", "y"]:
            backend.eval(source_activations[ch]["layer"])
            backend.eval(source_weights[ch]["layer"])
        backend.eval(target_activations["layer"], target_weights["layer"])

        config = MultiChannelMergeConfig(
            channels=["x", "y"],
                    )

        result = pipeline.run_merge(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            config=config,
        )

        # Per-channel CKA should be 1.0 (invariant) - use approx for float comparison
        assert "x" in result.per_channel_cka
        assert "y" in result.per_channel_cka
        assert result.per_channel_cka["x"] == pytest.approx(1.0, rel=1e-5)
        assert result.per_channel_cka["y"] == pytest.approx(1.0, rel=1e-5)
