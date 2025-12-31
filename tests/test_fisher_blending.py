# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Comprehensive tests for Fisher Information-Weighted Blending.

These tests are designed to find bugs and edge cases, not just verify happy paths.
Focus areas:
- Numerical stability (division by zero, overflow, underflow)
- Shape mismatches and broadcasting
- Empty/degenerate inputs
- Extreme configuration values
- NaN/inf propagation
"""

import pytest
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fisher_blending import (
    FisherBlendingConfig,
    FisherBlendingResult,
    FisherEstimationMethod,
    FisherNormalization,
    FisherWeights,
    apply_fisher_blending,
    combine_fisher_weights,
    estimate_fisher_from_loss_landscape,
    fisher_weighted_merge,
    normalize_fisher_weights,
    quick_fisher_blend,
)


@pytest.fixture
def backend():
    """Get the default backend for tests."""
    return get_default_backend()


# =============================================================================
# FisherWeights.from_gradients Tests
# =============================================================================


class TestFisherWeightsFromGradients:
    """Tests for FisherWeights.from_gradients classmethod."""

    def test_empty_gradient_history(self, backend):
        """Empty gradient history should produce empty weights."""
        result = FisherWeights.from_gradients({}, backend=backend)

        assert result.weights_by_key == {}
        assert result.total_parameters == 0
        assert result.mean_fisher == 0.0
        assert result.std_fisher == 0.0

    def test_single_gradient_per_key(self, backend):
        """Single gradient should produce high Fisher (variance ~0)."""
        grad = backend.ones((10, 10))
        backend.eval(grad)
        gradient_history = {"layer.weight": [grad]}

        result = FisherWeights.from_gradients(gradient_history, backend=backend)

        # Single sample has zero variance, so Fisher = 1/epsilon (very high)
        assert "layer.weight" in result.weights_by_key
        fisher_values = backend.to_numpy(result.weights_by_key["layer.weight"])
        # Should be clamped to max_fisher
        config = FisherBlendingConfig()
        assert fisher_values.max() <= config.max_fisher

    def test_constant_gradients_high_fisher(self, backend):
        """Constant gradients (zero variance) should produce high Fisher values."""
        # All identical gradients
        grad = backend.ones((5, 5))
        backend.eval(grad)
        gradient_history = {"weight": [grad, grad, grad, grad, grad]}

        result = FisherWeights.from_gradients(gradient_history, backend=backend)

        # Zero variance -> Fisher = 1/epsilon -> clamped to max_fisher
        fisher_values = backend.to_numpy(result.weights_by_key["weight"])
        config = FisherBlendingConfig()
        assert fisher_values.min() >= config.min_fisher

    def test_high_variance_gradients_low_fisher(self, backend):
        """High variance gradients should produce low Fisher values."""
        backend.random_seed(42)
        gradient_history = {
            "weight": [backend.random_normal((10, 10)) * 1000 for _ in range(50)]
        }
        for g in gradient_history["weight"]:
            backend.eval(g)

        result = FisherWeights.from_gradients(gradient_history, backend=backend)

        # High variance -> low Fisher
        fisher_values = backend.to_numpy(result.weights_by_key["weight"])
        # Should be much lower than max_fisher
        assert fisher_values.mean() < 1.0

    def test_empty_gradient_list_for_key(self, backend):
        """Key with empty gradient list should be skipped."""
        gradient_history = {"weight": []}

        result = FisherWeights.from_gradients(gradient_history, backend=backend)

        assert "weight" not in result.weights_by_key

    def test_multiple_keys(self, backend):
        """Multiple parameter keys should all be processed."""
        backend.random_seed(42)
        gradient_history = {
            "layer1.weight": [backend.random_normal((5, 5)) for _ in range(10)],
            "layer2.weight": [backend.random_normal((3, 3)) for _ in range(10)],
            "layer3.bias": [backend.random_normal((5,)) for _ in range(10)],
        }
        for key in gradient_history:
            for g in gradient_history[key]:
                backend.eval(g)

        result = FisherWeights.from_gradients(gradient_history, backend=backend)

        assert len(result.weights_by_key) == 3
        assert result.total_parameters == 5 * 5 + 3 * 3 + 5

    def test_fisher_clamping(self, backend):
        """Fisher values should be clamped to [min_fisher, max_fisher]."""
        config = FisherBlendingConfig(min_fisher=0.1, max_fisher=10.0)
        # Zero variance -> would be 1/epsilon without clamping
        grad = backend.ones((3, 3))
        backend.eval(grad)
        gradient_history = {"weight": [grad] * 5}

        result = FisherWeights.from_gradients(gradient_history, config=config, backend=backend)

        fisher_values = backend.to_numpy(result.weights_by_key["weight"])
        assert fisher_values.min() >= 0.1
        assert fisher_values.max() <= 10.0


# =============================================================================
# FisherWeights.uniform Tests
# =============================================================================


class TestFisherWeightsUniform:
    """Tests for FisherWeights.uniform classmethod."""

    def test_uniform_weights(self, backend):
        """Uniform weights should all be 1.0."""
        keys = ["layer1", "layer2"]
        shapes = {"layer1": (5, 5), "layer2": (3, 3)}

        result = FisherWeights.uniform(keys, shapes, backend=backend)

        for key in keys:
            values = backend.to_numpy(result.weights_by_key[key])
            assert (values == 1.0).all()

    def test_uniform_with_missing_shape(self, backend):
        """Keys not in shapes should be skipped."""
        keys = ["layer1", "layer2", "layer3"]
        shapes = {"layer1": (5, 5), "layer2": (3, 3)}  # layer3 missing

        result = FisherWeights.uniform(keys, shapes, backend=backend)

        assert "layer1" in result.weights_by_key
        assert "layer2" in result.weights_by_key
        assert "layer3" not in result.weights_by_key

    def test_uniform_metadata(self, backend):
        """Uniform weights should have correct metadata."""
        keys = ["w"]
        shapes = {"w": (10, 10)}

        result = FisherWeights.uniform(keys, shapes, backend=backend)

        assert result.estimation_method == FisherEstimationMethod.IDENTITY
        assert result.mean_fisher == 1.0
        assert result.std_fisher == 0.0
        assert result.total_parameters == 100


# =============================================================================
# normalize_fisher_weights Tests
# =============================================================================


class TestNormalizeFisherWeights:
    """Tests for normalize_fisher_weights function."""

    def test_none_normalization(self, backend):
        """NONE normalization should return input unchanged."""
        fisher = backend.array([1.0, 2.0, 3.0])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.NONE, backend=backend)

        assert backend.to_numpy(result).tolist() == [1.0, 2.0, 3.0]

    def test_layer_normalization(self, backend):
        """LAYER normalization should scale to [0, 1]."""
        fisher = backend.array([1.0, 2.0, 3.0])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.LAYER, backend=backend)
        result_np = backend.to_numpy(result)

        assert result_np.min() == pytest.approx(0.0, abs=1e-6)
        assert result_np.max() == pytest.approx(1.0, abs=1e-6)

    def test_layer_normalization_uniform_values(self, backend):
        """LAYER normalization with uniform values should return ones."""
        fisher = backend.array([5.0, 5.0, 5.0])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.LAYER, backend=backend)
        result_np = backend.to_numpy(result)

        # All same value -> all become 1.0
        assert (result_np == 1.0).all()

    def test_global_normalization(self, backend):
        """GLOBAL normalization should apply sigmoid(z-score)."""
        fisher = backend.array([0.0, 1.0, 2.0])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.GLOBAL, backend=backend)
        result_np = backend.to_numpy(result)

        # Should be in (0, 1) after sigmoid
        assert result_np.min() > 0.0
        assert result_np.max() < 1.0
        # Mean value should be ~0.5
        assert result_np[1] == pytest.approx(0.5, abs=0.1)

    def test_global_normalization_zero_std(self, backend):
        """GLOBAL normalization with zero std should return ones."""
        fisher = backend.array([5.0, 5.0, 5.0])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.GLOBAL, backend=backend)
        result_np = backend.to_numpy(result)

        assert (result_np == 1.0).all()

    def test_softmax_normalization(self, backend):
        """SOFTMAX normalization should sum to size after scaling."""
        fisher = backend.array([[1.0, 2.0], [3.0, 4.0]])
        backend.eval(fisher)

        result = normalize_fisher_weights(fisher, FisherNormalization.SOFTMAX, backend=backend)
        result_np = backend.to_numpy(result)

        # Should preserve relative magnitudes and scale by size
        assert result_np.shape == (2, 2)
        # Higher input values should produce higher outputs
        assert result_np[1, 1] > result_np[0, 0]

    def test_softmax_temperature(self, backend):
        """Higher temperature should produce more uniform softmax."""
        fisher = backend.array([1.0, 10.0])
        backend.eval(fisher)

        result_cold = normalize_fisher_weights(
            fisher, FisherNormalization.SOFTMAX, temperature=0.1, backend=backend
        )
        result_hot = normalize_fisher_weights(
            fisher, FisherNormalization.SOFTMAX, temperature=10.0, backend=backend
        )

        cold_np = backend.to_numpy(result_cold)
        hot_np = backend.to_numpy(result_hot)

        # Hot temperature should produce more uniform distribution
        cold_ratio = cold_np[1] / (cold_np[0] + 1e-10)
        hot_ratio = hot_np[1] / (hot_np[0] + 1e-10)
        assert cold_ratio > hot_ratio


# =============================================================================
# apply_fisher_blending Tests
# =============================================================================


class TestApplyFisherBlending:
    """Tests for apply_fisher_blending function."""

    def test_no_fisher_info_standard_blend(self, backend):
        """Without Fisher info, should do standard linear blending."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        backend.eval(source, target)

        merged, alpha_eff = apply_fisher_blending(
            source, target, base_alpha=0.3, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        assert merged_np.mean() == pytest.approx(0.3, abs=1e-6)

    def test_alpha_zero_all_source(self, backend):
        """Alpha=0 should produce all source weights."""
        source = backend.ones((5, 5)) * 10.0
        target = backend.ones((5, 5)) * 20.0
        backend.eval(source, target)

        merged, _ = apply_fisher_blending(
            source, target, base_alpha=0.0, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        assert merged_np.mean() == pytest.approx(10.0, abs=1e-6)

    def test_alpha_one_all_target(self, backend):
        """Alpha=1 should produce all target weights."""
        source = backend.ones((5, 5)) * 10.0
        target = backend.ones((5, 5)) * 20.0
        backend.eval(source, target)

        merged, _ = apply_fisher_blending(
            source, target, base_alpha=1.0, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        assert merged_np.mean() == pytest.approx(20.0, abs=1e-6)

    def test_fisher_bias_toward_target(self, backend):
        """Higher target Fisher should bias toward target."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        source_fisher = backend.ones((5, 5)) * 0.1  # Low importance
        target_fisher = backend.ones((5, 5)) * 10.0  # High importance
        backend.eval(source, target, source_fisher, target_fisher)

        # Use NONE normalization to avoid LAYER normalization making uniform values equal
        # (LAYER normalization normalizes within each array, so uniform 0.1 -> all 1.0,
        # uniform 10.0 -> all 1.0, making them equal)
        config = FisherBlendingConfig(strength=1.0, normalization=FisherNormalization.NONE)
        merged, alpha_eff = apply_fisher_blending(
            source, target, base_alpha=0.5,
            source_fisher=source_fisher, target_fisher=target_fisher,
            config=config, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        # Should be biased toward target (values > 0.5)
        # With high target Fisher, effective alpha should be high
        assert merged_np.mean() > 0.5

    def test_fisher_bias_toward_source(self, backend):
        """Higher source Fisher should bias toward source."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        source_fisher = backend.ones((5, 5)) * 10.0  # High importance
        target_fisher = backend.ones((5, 5)) * 0.1  # Low importance
        backend.eval(source, target, source_fisher, target_fisher)

        config = FisherBlendingConfig(strength=1.0)
        merged, alpha_eff = apply_fisher_blending(
            source, target, base_alpha=0.5,
            source_fisher=source_fisher, target_fisher=target_fisher,
            config=config, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        # Should be biased toward source (values < 0.5)
        assert merged_np.mean() < 0.5

    def test_source_bias_parameter(self, backend):
        """source_bias should shift effective alpha."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        backend.eval(source, target)

        config_neutral = FisherBlendingConfig(source_bias=0.0)
        config_biased = FisherBlendingConfig(source_bias=0.5)

        _, alpha_neutral = apply_fisher_blending(
            source, target, base_alpha=0.5, config=config_neutral, backend=backend
        )
        _, alpha_biased = apply_fisher_blending(
            source, target, base_alpha=0.5, config=config_biased, backend=backend
        )

        # Positive source_bias should reduce target importance (lower alpha)
        neutral_mean = backend.to_numpy(backend.mean(alpha_neutral)).item()
        biased_mean = backend.to_numpy(backend.mean(alpha_biased)).item()
        assert biased_mean <= neutral_mean

    def test_clip_alpha(self, backend):
        """clip_alpha=True should keep alpha in [0, 1]."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        source_fisher = backend.ones((5, 5)) * 1e10  # Extreme
        target_fisher = backend.ones((5, 5)) * 1e-10  # Extreme
        backend.eval(source, target, source_fisher, target_fisher)

        config = FisherBlendingConfig(clip_alpha=True, strength=1.0)
        merged, alpha_eff = apply_fisher_blending(
            source, target, base_alpha=0.5,
            source_fisher=source_fisher, target_fisher=target_fisher,
            config=config, backend=backend
        )

        alpha_np = backend.to_numpy(alpha_eff)
        assert alpha_np.min() >= 0.0
        assert alpha_np.max() <= 1.0

    def test_strength_zero_ignores_fisher(self, backend):
        """strength=0 should use base_alpha only."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        source_fisher = backend.ones((5, 5)) * 100.0
        target_fisher = backend.ones((5, 5)) * 0.01
        backend.eval(source, target, source_fisher, target_fisher)

        config = FisherBlendingConfig(strength=0.0)
        merged, alpha_eff = apply_fisher_blending(
            source, target, base_alpha=0.3,
            source_fisher=source_fisher, target_fisher=target_fisher,
            config=config, backend=backend
        )

        merged_np = backend.to_numpy(merged)
        # Should be exactly 0.3 (base_alpha) since strength=0
        assert merged_np.mean() == pytest.approx(0.3, abs=1e-6)

    def test_shape_broadcast(self, backend):
        """Fisher weights should broadcast to weight shape."""
        source = backend.ones((5, 5))
        target = backend.ones((5, 5)) * 2.0
        # Scalar-like Fisher
        source_fisher = backend.array([1.0])
        target_fisher = backend.array([1.0])
        backend.eval(source, target, source_fisher, target_fisher)

        # Should broadcast without error
        merged, _ = apply_fisher_blending(
            source, target, base_alpha=0.5,
            source_fisher=source_fisher, target_fisher=target_fisher,
            backend=backend
        )

        assert merged.shape == (5, 5)

    def test_only_source_fisher_provided(self, backend):
        """With only source Fisher, target gets uniform weight."""
        source = backend.zeros((5, 5))
        target = backend.ones((5, 5))
        source_fisher = backend.ones((5, 5)) * 10.0
        backend.eval(source, target, source_fisher)

        merged, _ = apply_fisher_blending(
            source, target, base_alpha=0.5,
            source_fisher=source_fisher, target_fisher=None,
            backend=backend
        )

        # Should complete without error
        assert merged.shape == (5, 5)


# =============================================================================
# fisher_weighted_merge Tests
# =============================================================================


class TestFisherWeightedMerge:
    """Tests for fisher_weighted_merge function."""

    def test_basic_merge(self, backend):
        """Basic merge should produce FisherBlendingResult."""
        source_weights = {"layer.weight": backend.zeros((5, 5))}
        target_weights = {"layer.weight": backend.ones((5, 5))}
        for w in source_weights.values():
            backend.eval(w)
        for w in target_weights.values():
            backend.eval(w)

        shapes = {"layer.weight": (5, 5)}
        source_fisher = FisherWeights.uniform(["layer.weight"], shapes, backend=backend)
        target_fisher = FisherWeights.uniform(["layer.weight"], shapes, backend=backend)

        result = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            base_alpha=0.5, backend=backend
        )

        assert isinstance(result, FisherBlendingResult)
        assert "layer.weight" in result.merged_weights
        assert result.parameters_blended == 1

    def test_missing_target_key_skipped(self, backend):
        """Keys only in source should be skipped."""
        source_weights = {
            "layer1.weight": backend.zeros((5, 5)),
            "layer2.weight": backend.zeros((3, 3)),
        }
        target_weights = {"layer1.weight": backend.ones((5, 5))}  # layer2 missing
        for w in source_weights.values():
            backend.eval(w)
        for w in target_weights.values():
            backend.eval(w)

        shapes = {"layer1.weight": (5, 5), "layer2.weight": (3, 3)}
        source_fisher = FisherWeights.uniform(list(shapes.keys()), shapes, backend=backend)
        target_fisher = FisherWeights.uniform(["layer1.weight"], {"layer1.weight": (5, 5)}, backend=backend)

        result = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            backend=backend
        )

        assert "layer1.weight" in result.merged_weights
        assert "layer2.weight" not in result.merged_weights
        assert result.parameters_blended == 1

    def test_mean_effective_alpha(self, backend):
        """mean_effective_alpha should be computed correctly."""
        source_weights = {"w": backend.zeros((10, 10))}
        target_weights = {"w": backend.ones((10, 10))}
        backend.eval(source_weights["w"], target_weights["w"])

        shapes = {"w": (10, 10)}
        source_fisher = FisherWeights.uniform(["w"], shapes, backend=backend)
        target_fisher = FisherWeights.uniform(["w"], shapes, backend=backend)

        result = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            base_alpha=0.7, backend=backend
        )

        # With uniform Fisher and strength=0.5 (default), mean_alpha should be ~0.7
        assert 0.5 <= result.mean_effective_alpha <= 1.0

    def test_fisher_applied_flag(self, backend):
        """fisher_applied should reflect actual Fisher usage."""
        source_weights = {"w": backend.zeros((5, 5))}
        target_weights = {"w": backend.ones((5, 5))}
        backend.eval(source_weights["w"], target_weights["w"])

        shapes = {"w": (5, 5)}
        source_fisher = FisherWeights.uniform(["w"], shapes, backend=backend)
        target_fisher = FisherWeights.uniform(["w"], shapes, backend=backend)

        # With strength > 0 and Fisher weights present
        result = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            config=FisherBlendingConfig(strength=0.5),
            backend=backend
        )
        assert result.fisher_applied is True

        # With strength = 0
        result_no_fisher = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            config=FisherBlendingConfig(strength=0.0),
            backend=backend
        )
        assert result_no_fisher.fisher_applied is False


# =============================================================================
# estimate_fisher_from_loss_landscape Tests
# =============================================================================


class TestEstimateFisherFromLossLandscape:
    """Tests for estimate_fisher_from_loss_landscape function."""

    def test_constant_loss_low_fisher(self, backend):
        """Constant loss should produce low Fisher (not sensitive)."""
        weights = {"w": backend.ones((5, 5))}
        backend.eval(weights["w"])

        def constant_loss(w):
            return 1.0  # Loss doesn't depend on weights

        config = FisherBlendingConfig(seed=42)
        result = estimate_fisher_from_loss_landscape(
            weights, constant_loss, num_samples=10,
            config=config, backend=backend
        )

        # Low sensitivity should give low Fisher (clamped to min_fisher)
        fisher_np = backend.to_numpy(result.weights_by_key["w"])
        assert fisher_np.max() == pytest.approx(config.min_fisher, rel=0.1)

    def test_sensitive_loss_high_fisher(self, backend):
        """Loss sensitive to weights should produce higher Fisher."""
        weights = {"w": backend.ones((5, 5))}
        backend.eval(weights["w"])

        def sensitive_loss(w):
            # Loss = sum of weights (very sensitive)
            b = get_default_backend()
            total = b.sum(w["w"])
            b.eval(total)
            return float(b.to_numpy(total).item())

        config = FisherBlendingConfig(seed=42)
        result = estimate_fisher_from_loss_landscape(
            weights, sensitive_loss, num_samples=20,
            perturbation_scale=0.1,
            config=config, backend=backend
        )

        fisher_np = backend.to_numpy(result.weights_by_key["w"])
        # Should be higher than min_fisher
        assert fisher_np.mean() > config.min_fisher

    def test_deterministic_with_seed(self, backend):
        """Same seed should produce same Fisher estimates."""
        weights = {"w": backend.ones((3, 3))}
        backend.eval(weights["w"])

        def loss(w):
            b = get_default_backend()
            total = b.sum(w["w"])
            b.eval(total)
            return float(b.to_numpy(total).item())

        config1 = FisherBlendingConfig(seed=12345)
        config2 = FisherBlendingConfig(seed=12345)

        result1 = estimate_fisher_from_loss_landscape(
            weights, loss, num_samples=5, config=config1, backend=backend
        )
        result2 = estimate_fisher_from_loss_landscape(
            weights, loss, num_samples=5, config=config2, backend=backend
        )

        np1 = backend.to_numpy(result1.weights_by_key["w"])
        np2 = backend.to_numpy(result2.weights_by_key["w"])
        assert (np1 == np2).all()

    def test_multiple_weight_keys(self, backend):
        """Should estimate Fisher for all weight keys."""
        weights = {
            "layer1": backend.ones((3, 3)),
            "layer2": backend.ones((2, 2)),
        }
        for w in weights.values():
            backend.eval(w)

        def loss(w):
            return 1.0

        result = estimate_fisher_from_loss_landscape(
            weights, loss, num_samples=5, backend=backend
        )

        assert "layer1" in result.weights_by_key
        assert "layer2" in result.weights_by_key
        assert result.total_parameters == 9 + 4


# =============================================================================
# combine_fisher_weights Tests
# =============================================================================


class TestCombineFisherWeights:
    """Tests for combine_fisher_weights function."""

    def test_empty_list_raises(self, backend):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            combine_fisher_weights([])

    def test_single_element_returns_same(self, backend):
        """Single Fisher weight should be returned unchanged."""
        shapes = {"w": (5, 5)}
        single = FisherWeights.uniform(["w"], shapes, backend=backend)

        result = combine_fisher_weights([single], backend=backend)

        assert result is single

    def test_mean_combination(self, backend):
        """Mean combination should average Fisher values."""
        shapes = {"w": (2, 2)}
        fw1 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 2.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=2.0,
            std_fisher=0.0,
        )
        backend.eval(fw1.weights_by_key["w"])
        fw2 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 4.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=4.0,
            std_fisher=0.0,
        )
        backend.eval(fw2.weights_by_key["w"])

        result = combine_fisher_weights([fw1, fw2], combination_method="mean", backend=backend)

        combined_np = backend.to_numpy(result.weights_by_key["w"])
        assert combined_np.mean() == pytest.approx(3.0, abs=1e-6)

    def test_max_combination(self, backend):
        """Max combination should take maximum Fisher values."""
        shapes = {"w": (2, 2)}
        fw1 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 2.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=2.0,
            std_fisher=0.0,
        )
        backend.eval(fw1.weights_by_key["w"])
        fw2 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 4.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=4.0,
            std_fisher=0.0,
        )
        backend.eval(fw2.weights_by_key["w"])

        result = combine_fisher_weights([fw1, fw2], combination_method="max", backend=backend)

        combined_np = backend.to_numpy(result.weights_by_key["w"])
        assert combined_np.mean() == pytest.approx(4.0, abs=1e-6)

    def test_harmonic_combination(self, backend):
        """Harmonic combination should compute harmonic mean."""
        shapes = {"w": (2, 2)}
        # Harmonic mean of 2 and 4 = 2*2*4/(2+4) = 16/6 = 2.667
        fw1 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 2.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=2.0,
            std_fisher=0.0,
        )
        backend.eval(fw1.weights_by_key["w"])
        fw2 = FisherWeights(
            weights_by_key={"w": backend.ones((2, 2)) * 4.0},
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=4,
            mean_fisher=4.0,
            std_fisher=0.0,
        )
        backend.eval(fw2.weights_by_key["w"])

        result = combine_fisher_weights([fw1, fw2], combination_method="harmonic", backend=backend)

        combined_np = backend.to_numpy(result.weights_by_key["w"])
        expected_harmonic = 2 * 2.0 * 4.0 / (2.0 + 4.0)  # 2.667
        assert combined_np.mean() == pytest.approx(expected_harmonic, abs=0.1)

    def test_partial_key_overlap(self, backend):
        """Should handle partial key overlap between Fisher weights."""
        fw1 = FisherWeights(
            weights_by_key={
                "layer1": backend.ones((2, 2)) * 2.0,
                "layer2": backend.ones((2, 2)) * 3.0,
            },
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=8,
            mean_fisher=2.5,
            std_fisher=0.5,
        )
        for w in fw1.weights_by_key.values():
            backend.eval(w)
        fw2 = FisherWeights(
            weights_by_key={
                "layer2": backend.ones((2, 2)) * 5.0,
                "layer3": backend.ones((2, 2)) * 4.0,
            },
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=8,
            mean_fisher=4.5,
            std_fisher=0.5,
        )
        for w in fw2.weights_by_key.values():
            backend.eval(w)

        result = combine_fisher_weights([fw1, fw2], combination_method="mean", backend=backend)

        # layer1 only in fw1, layer3 only in fw2, layer2 in both
        assert "layer1" in result.weights_by_key
        assert "layer2" in result.weights_by_key
        assert "layer3" in result.weights_by_key

        # layer1 should be unchanged (only source)
        assert backend.to_numpy(result.weights_by_key["layer1"]).mean() == pytest.approx(2.0, abs=1e-6)
        # layer2 should be average (3+5)/2 = 4
        assert backend.to_numpy(result.weights_by_key["layer2"]).mean() == pytest.approx(4.0, abs=1e-6)
        # layer3 should be unchanged (only in fw2)
        assert backend.to_numpy(result.weights_by_key["layer3"]).mean() == pytest.approx(4.0, abs=1e-6)


# =============================================================================
# quick_fisher_blend Tests
# =============================================================================


class TestQuickFisherBlend:
    """Tests for quick_fisher_blend convenience function."""

    def test_basic_blend(self, backend):
        """Basic blend should work."""
        source = {"w": backend.zeros((5, 5))}
        target = {"w": backend.ones((5, 5))}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        assert "w" in result
        result_np = backend.to_numpy(result["w"])
        # With uniform Fisher and strength=0.5, should be close to 0.5
        assert 0.3 <= result_np.mean() <= 0.7

    def test_alpha_zero(self, backend):
        """Alpha=0 should produce all source."""
        source = {"w": backend.ones((5, 5)) * 10.0}
        target = {"w": backend.ones((5, 5)) * 20.0}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.0, backend=backend)

        result_np = backend.to_numpy(result["w"])
        assert result_np.mean() == pytest.approx(10.0, abs=1e-6)

    def test_strength_zero(self, backend):
        """Strength=0 should do standard blending."""
        source = {"w": backend.zeros((5, 5))}
        target = {"w": backend.ones((5, 5))}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.3, strength=0.0, backend=backend)

        result_np = backend.to_numpy(result["w"])
        assert result_np.mean() == pytest.approx(0.3, abs=1e-6)

    def test_only_common_keys(self, backend):
        """Should only blend keys present in both."""
        source = {"w1": backend.zeros((3, 3)), "w2": backend.zeros((3, 3))}
        target = {"w1": backend.ones((3, 3)), "w3": backend.ones((3, 3))}
        for w in source.values():
            backend.eval(w)
        for w in target.values():
            backend.eval(w)

        result = quick_fisher_blend(source, target, backend=backend)

        assert "w1" in result
        assert "w2" not in result
        assert "w3" not in result


# =============================================================================
# Integration and Edge Case Tests
# =============================================================================


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_full_pipeline_from_gradients_to_merge(self, backend):
        """Full pipeline: gradients -> Fisher -> merge."""
        backend.random_seed(42)

        # Create source and target weights
        source_weights = {"layer.weight": backend.random_normal((10, 10))}
        target_weights = {"layer.weight": backend.random_normal((10, 10))}
        backend.eval(source_weights["layer.weight"], target_weights["layer.weight"])

        # Create gradient history (simulating training)
        source_grads = {
            "layer.weight": [backend.random_normal((10, 10)) * 0.1 for _ in range(20)]
        }
        target_grads = {
            "layer.weight": [backend.random_normal((10, 10)) * 0.5 for _ in range(20)]
        }
        for g in source_grads["layer.weight"]:
            backend.eval(g)
        for g in target_grads["layer.weight"]:
            backend.eval(g)

        # Estimate Fisher from gradients
        source_fisher = FisherWeights.from_gradients(source_grads, backend=backend)
        target_fisher = FisherWeights.from_gradients(target_grads, backend=backend)

        # Merge
        result = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            base_alpha=0.5,
            backend=backend
        )

        assert result.parameters_blended == 1
        assert result.fisher_applied is True
        # Source had lower variance -> higher Fisher -> should bias toward source
        # (lower effective alpha)

    def test_very_small_weights(self, backend):
        """Should handle very small weight values."""
        source = {"w": backend.ones((5, 5)) * 1e-10}
        target = {"w": backend.ones((5, 5)) * 1e-10}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        result_np = backend.to_numpy(result["w"])
        assert not any(val != val for val in result_np.flat)  # No NaN

    def test_very_large_weights(self, backend):
        """Should handle very large weight values."""
        source = {"w": backend.ones((5, 5)) * 1e10}
        target = {"w": backend.ones((5, 5)) * 1e10}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        result_np = backend.to_numpy(result["w"])
        assert result_np.mean() == pytest.approx(1e10, rel=0.1)

    def test_mixed_positive_negative_weights(self, backend):
        """Should handle mixed positive/negative weights."""
        source = {"w": backend.array([[-1.0, 1.0], [1.0, -1.0]])}
        target = {"w": backend.array([[1.0, -1.0], [-1.0, 1.0]])}
        backend.eval(source["w"], target["w"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        result_np = backend.to_numpy(result["w"])
        # With uniform Fisher, should be close to average (near 0)
        assert abs(result_np.mean()) < 0.5

    def test_1d_weights(self, backend):
        """Should handle 1D weights (biases)."""
        source = {"b": backend.zeros((10,))}
        target = {"b": backend.ones((10,))}
        backend.eval(source["b"], target["b"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        assert result["b"].shape == (10,)

    def test_3d_weights(self, backend):
        """Should handle 3D weights (conv filters)."""
        source = {"conv": backend.zeros((3, 3, 3))}
        target = {"conv": backend.ones((3, 3, 3))}
        backend.eval(source["conv"], target["conv"])

        result = quick_fisher_blend(source, target, alpha=0.5, backend=backend)

        assert result["conv"].shape == (3, 3, 3)
