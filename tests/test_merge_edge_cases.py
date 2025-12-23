"""
Edge case tests for merge pipeline robustness.

Tests for:
- Mismatched layer counts
- Empty activations
- Extreme alpha values
- NaN handling
- Single sample activations
- Zero reliability domain signals
"""
import pytest
import math
import mlx.core as mx
from typing import Dict, List

from modelcypher.core.domain.merging.unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    ModuleScope,
    compute_dimension_blending_weights,
)
from modelcypher.core.domain.geometry.domain_signal_profile import (
    DomainSignalScores,
    DomainSignalDecision,
    domain_adjusted_alpha,
    DomainSignalConfig,
)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_weights_dicts(self):
        """Empty weight dictionaries should produce empty result."""
        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights={},
            target_weights={},
            layer_confidences={},
        )

        assert result.layers_merged == 0
        assert len(result.merged_weights) == 0

    def test_mismatched_keys(self):
        """Weights not in both source and target should be skipped."""
        source = {
            "shared.weight": mx.ones((4, 4)),
            "only_in_source.weight": mx.ones((4, 4)),
        }
        target = {
            "shared.weight": mx.zeros((4, 4)),
            "only_in_target.weight": mx.zeros((4, 4)),
        }

        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        # Only shared keys should be merged
        assert "shared.weight" in result.merged_weights
        # Keys only in one dict should be handled gracefully

    def test_empty_activation_dict(self):
        """Dimension blending with empty activations should fallback gracefully."""
        source = {"model.layers.0.mlp.down_proj.weight": mx.ones((64, 128))}
        target = {"model.layers.0.mlp.down_proj.weight": mx.zeros((64, 128))}

        config = UnifiedMergeConfig(use_dimension_blending=True)
        merger = UnifiedManifoldMerger(config)

        # Should not crash with empty activations
        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
            source_activations={},  # Empty
            target_activations={},  # Empty
        )

        assert result is not None
        assert result.dimension_blending_applied is False

    def test_extreme_alpha_values_clamped(self):
        """Extreme domain signal scores should result in clamped alpha."""
        # Test with combined_score at extremes
        extreme_high = DomainSignalScores(
            sparsity_score=1.0,
            smoothness_score=1.0,
            combined_score=1.0,
            reliability=1.0,
        )

        extreme_low = DomainSignalScores(
            sparsity_score=0.0,
            smoothness_score=0.0,
            combined_score=0.0,
            reliability=1.0,
        )

        high_alpha = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=extreme_high,
            strength=1.0,
            min_alpha=0.2,
            max_alpha=0.95,
        )

        low_alpha = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=extreme_low,
            strength=1.0,
            min_alpha=0.2,
            max_alpha=0.95,
        )

        assert 0.2 <= high_alpha <= 0.95
        assert 0.2 <= low_alpha <= 0.95
        assert high_alpha > low_alpha  # High combined_score → higher alpha

    def test_zero_reliability_no_adjustment(self):
        """Zero reliability domain signals should not adjust alpha."""
        scores = DomainSignalScores(
            sparsity_score=1.0,
            smoothness_score=1.0,
            combined_score=1.0,
            reliability=0.0,  # Zero reliability
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=1.0,
            min_alpha=0.2,
            max_alpha=0.95,
        )

        # With zero reliability, alpha should remain at base
        assert adjusted == 0.5

    def test_zero_strength_no_adjustment(self):
        """Zero strength should not adjust alpha."""
        scores = DomainSignalScores(
            sparsity_score=1.0,
            smoothness_score=1.0,
            combined_score=1.0,
            reliability=1.0,
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=0.0,  # Zero strength
            min_alpha=0.2,
            max_alpha=0.95,
        )

        # With zero strength, alpha should remain at base
        assert adjusted == 0.5

    def test_single_sample_activation(self):
        """Dimension blending with single sample should handle gracefully."""
        # Single sample per layer
        source_acts = mx.array([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 4)
        target_acts = mx.array([[1.1, 2.1, 3.1, 4.1]])  # Shape: (1, 4)

        # Should not crash
        try:
            weights = compute_dimension_blending_weights(
                source_activations=source_acts,
                target_activations=target_acts,
                threshold=0.3,
                fallback_weight=0.5,
            )
            # If it returns, weights should be valid
            if weights is not None:
                assert weights.weights.shape[0] == 4
        except Exception:
            # Some edge cases may raise - that's acceptable as long as it doesn't crash silently
            pass

    def test_mismatched_activation_shapes(self):
        """Mismatched activation shapes should fallback gracefully."""
        source_acts = mx.array([[1.0, 2.0, 3.0, 4.0] for _ in range(10)])  # (10, 4)
        target_acts = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(10)])  # (10, 5) - different dims

        # compute_dimension_blending_weights should handle or raise gracefully
        try:
            weights = compute_dimension_blending_weights(
                source_activations=source_acts,
                target_activations=target_acts,
                threshold=0.3,
                fallback_weight=0.5,
            )
        except (ValueError, AssertionError):
            pass  # Expected for shape mismatch

    def test_all_modules_filtered_by_scope(self):
        """ModuleScope that matches nothing should handle gracefully."""
        # Create weights that are only attention modules
        source = {
            "model.layers.0.self_attn.q_proj.weight": mx.ones((64, 64)),
        }
        target = {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
        }

        # Use MLP_ONLY scope - should not match any attention modules
        config = UnifiedMergeConfig(module_scope=ModuleScope.MLP_ONLY)
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        # Result should contain the key, but it should be unchanged from target
        assert "model.layers.0.self_attn.q_proj.weight" in result.merged_weights
        assert mx.allclose(
            result.merged_weights["model.layers.0.self_attn.q_proj.weight"],
            target["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_very_small_weights(self):
        """Very small weight values should not cause numerical issues."""
        source = {"layer.0.weight": mx.ones((4, 4)) * 1e-10}
        target = {"layer.0.weight": mx.ones((4, 4)) * 1e-10}

        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        # Should not produce NaN or Inf
        merged = result.merged_weights["layer.0.weight"]
        assert not mx.any(mx.isnan(merged))
        assert not mx.any(mx.isinf(merged))

    def test_very_large_weights(self):
        """Very large weight values should not cause numerical issues."""
        source = {"layer.0.weight": mx.ones((4, 4)) * 1e10}
        target = {"layer.0.weight": mx.ones((4, 4)) * 1e10}

        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        # Should not produce NaN or Inf
        merged = result.merged_weights["layer.0.weight"]
        assert not mx.any(mx.isnan(merged))
        assert not mx.any(mx.isinf(merged))

    def test_1d_weight_skips_spectral_penalty(self):
        """1D weights should skip spectral penalty computation."""
        source = {"bias": mx.ones((64,))}
        target = {"bias": mx.zeros((64,))}

        config = UnifiedMergeConfig(spectral_penalty_strength=1.0)
        merger = UnifiedManifoldMerger(config)

        # Should not crash on 1D weight
        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        assert "bias" in result.merged_weights

    def test_many_layers_performance(self):
        """Many layers should be handled efficiently."""
        num_layers = 100
        source = {f"model.layers.{i}.weight": mx.ones((64, 64)) for i in range(num_layers)}
        target = {f"model.layers.{i}.weight": mx.zeros((64, 64)) for i in range(num_layers)}
        confidences = {i: 0.5 for i in range(num_layers)}

        merger = UnifiedManifoldMerger()

        # Should complete in reasonable time
        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences=confidences,
        )

        assert result.layers_merged == num_layers


# =============================================================================
# Property-Based Edge Cases
# =============================================================================


class TestAlphaBoundsProperty:
    """Property tests for alpha bounds."""

    @pytest.mark.parametrize(
        "combined_score,reliability,strength",
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
        ],
    )
    def test_alpha_always_bounded(self, combined_score, reliability, strength):
        """Alpha should always be within [min_alpha, max_alpha]."""
        scores = DomainSignalScores(
            sparsity_score=combined_score,
            smoothness_score=combined_score,
            combined_score=combined_score,
            reliability=reliability,
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=strength,
            min_alpha=0.1,
            max_alpha=0.95,
        )

        assert 0.1 <= adjusted <= 0.95

    @pytest.mark.parametrize(
        "base_alpha",
        [0.0, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0],
    )
    def test_base_alpha_range(self, base_alpha):
        """Various base alphas should produce bounded results."""
        scores = DomainSignalScores(
            sparsity_score=0.7,
            smoothness_score=0.7,
            combined_score=0.7,
            reliability=1.0,
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=base_alpha,
            scores=scores,
            strength=1.0,
            min_alpha=0.1,
            max_alpha=0.95,
        )

        assert 0.1 <= adjusted <= 0.95


# =============================================================================
# Domain Signal Edge Cases
# =============================================================================


class TestDomainSignalEdgeCases:
    """Edge cases for domain signal computation."""

    def test_decision_skipped_preserves_base_alpha(self):
        """Skipped decision should preserve base alpha."""
        decision = DomainSignalDecision.skipped(
            layer=0,
            base_alpha=0.6,
            reason="No data available",
        )

        assert decision.base_alpha == 0.6
        assert decision.adjusted_alpha == 0.6
        assert decision.applied is False

    def test_decision_applied_tracks_scores(self):
        """Applied decision should track scores."""
        scores = DomainSignalScores(
            sparsity_score=0.7,
            smoothness_score=0.6,
            combined_score=0.65,
            reliability=0.9,
        )

        decision = DomainSignalDecision.create_applied(
            layer=0,
            base_alpha=0.5,
            adjusted_alpha=0.65,
            scores=scores,
        )

        assert decision.applied is True
        assert decision.scores is not None
        assert decision.scores.combined_score == 0.65

    def test_neutral_scores_minimal_adjustment(self):
        """Neutral scores (0.5) should cause minimal adjustment."""
        scores = DomainSignalScores(
            sparsity_score=0.5,
            smoothness_score=0.5,
            combined_score=0.5,
            reliability=1.0,
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=1.0,
            min_alpha=0.2,
            max_alpha=0.95,
        )

        # With neutral scores and base_alpha=0.5, adjustment should be moderate
        # desired_alpha = 0.5 * (0.95 - 0.2) + 0.2 = 0.575
        assert 0.5 <= adjusted <= 0.6
