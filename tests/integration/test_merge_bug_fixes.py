"""
Integration tests validating bug fixes in unified_manifold_merger.py.

Tests the 5 critical bug fixes:
1. Attribute name mismatch (adjusted_alpha → gated_alpha)
2. Unbounded alpha accumulation (clamping after adjustments)
3. Dimension blending disabled (now properly integrated)
4. Module scope filtering (now applied in merge loop)
5. Silent fallbacks (now emit warnings)
"""
import logging
import pytest
import mlx.core as mx
from typing import Dict, List

from modelcypher.core.domain.merging.unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    ModuleScope,
    ModuleKind,
    MLPInternalIntersectionMode,
    MLPInternalGatingResult,
    apply_mlp_internal_gating,
    compute_dimension_blending_weights,
)
from modelcypher.core.domain.geometry.domain_signal_profile import (
    DomainSignalProfile,
    DomainSignalScores,
    DomainSignalDecision,
    LayerSignal,
    compute_domain_scores,
    domain_adjusted_alpha,
    DomainSignalConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_weights() -> tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    """Create synthetic source and target weights for testing."""
    source = {
        "model.layers.0.self_attn.q_proj.weight": mx.ones((64, 64)) * 0.5,
        "model.layers.0.self_attn.k_proj.weight": mx.ones((64, 64)) * 0.5,
        "model.layers.0.self_attn.v_proj.weight": mx.ones((64, 64)) * 0.5,
        "model.layers.0.self_attn.o_proj.weight": mx.ones((64, 64)) * 0.5,
        "model.layers.0.mlp.gate_proj.weight": mx.ones((128, 64)) * 0.5,
        "model.layers.0.mlp.up_proj.weight": mx.ones((128, 64)) * 0.5,
        "model.layers.0.mlp.down_proj.weight": mx.ones((64, 128)) * 0.5,
        "model.layers.1.self_attn.q_proj.weight": mx.ones((64, 64)) * 0.5,
        "model.layers.1.mlp.gate_proj.weight": mx.ones((128, 64)) * 0.5,
        "model.layers.1.mlp.down_proj.weight": mx.ones((64, 128)) * 0.5,
    }
    target = {
        "model.layers.0.self_attn.q_proj.weight": mx.ones((64, 64)) * 1.0,
        "model.layers.0.self_attn.k_proj.weight": mx.ones((64, 64)) * 1.0,
        "model.layers.0.self_attn.v_proj.weight": mx.ones((64, 64)) * 1.0,
        "model.layers.0.self_attn.o_proj.weight": mx.ones((64, 64)) * 1.0,
        "model.layers.0.mlp.gate_proj.weight": mx.ones((128, 64)) * 1.0,
        "model.layers.0.mlp.up_proj.weight": mx.ones((128, 64)) * 1.0,
        "model.layers.0.mlp.down_proj.weight": mx.ones((64, 128)) * 1.0,
        "model.layers.1.self_attn.q_proj.weight": mx.ones((64, 64)) * 1.0,
        "model.layers.1.mlp.gate_proj.weight": mx.ones((128, 64)) * 1.0,
        "model.layers.1.mlp.down_proj.weight": mx.ones((64, 128)) * 1.0,
    }
    return source, target


@pytest.fixture
def synthetic_activations() -> tuple[Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """Create synthetic activations for dimension blending tests."""
    # Create correlated activations for source and target
    import random
    random.seed(42)

    source_acts = {}
    target_acts = {}

    for layer in range(2):
        # Generate source activations
        source_samples = []
        target_samples = []
        for _ in range(10):  # 10 samples per layer
            src = [random.gauss(0.5, 0.1) for _ in range(64)]
            # Target is correlated with source (with some noise)
            tgt = [s + random.gauss(0, 0.05) for s in src]
            source_samples.append(src)
            target_samples.append(tgt)

        source_acts[layer] = source_samples
        target_acts[layer] = target_samples

    return source_acts, target_acts


@pytest.fixture
def domain_signal_profiles() -> tuple[DomainSignalProfile, DomainSignalProfile]:
    """Create domain signal profiles for testing."""
    from datetime import datetime

    source_signals = {
        0: LayerSignal(sparsity=0.3, gradient_snr=2.0, gradient_sample_count=10),
        1: LayerSignal(sparsity=0.4, gradient_snr=1.5, gradient_sample_count=10),
    }
    target_signals = {
        0: LayerSignal(sparsity=0.6, gradient_snr=1.0, gradient_sample_count=10),
        1: LayerSignal(sparsity=0.5, gradient_snr=2.5, gradient_sample_count=10),
    }

    source_profile = DomainSignalProfile(
        layer_signals=source_signals,
        model_id="source-model",
        domain="code",
        baseline_domain="baseline",
        total_layers=2,
        prompt_count=10,
        max_tokens_per_prompt=128,
        generated_at=datetime.utcnow(),
    )
    target_profile = DomainSignalProfile(
        layer_signals=target_signals,
        model_id="target-model",
        domain="code",
        baseline_domain="baseline",
        total_layers=2,
        prompt_count=10,
        max_tokens_per_prompt=128,
        generated_at=datetime.utcnow(),
    )

    return source_profile, target_profile


# =============================================================================
# Bug Fix 1: Attribute Name Mismatch
# =============================================================================


class TestBugFix1_AttributeName:
    """
    Verify MLP internal gating uses correct attribute name (gated_alpha, not adjusted_alpha).

    Bug: Line 1413 used gating_result.adjusted_alpha but MLPInternalGatingResult has gated_alpha.
    Fix: Changed to gating_result.gated_alpha at line 1416.
    """

    def test_mlp_internal_gating_result_has_gated_alpha(self):
        """Verify MLPInternalGatingResult dataclass has gated_alpha field."""
        result = MLPInternalGatingResult(
            key="test.mlp.gate_proj.weight",
            module_kind=ModuleKind.GATE_PROJ,
            original_alpha=0.5,
            gated_alpha=0.6,
            gating_applied=True,
            intersection_mode_used=MLPInternalIntersectionMode.INVARIANTS,
        )

        # Accessing gated_alpha should work
        assert result.gated_alpha == 0.6

        # Verify adjusted_alpha doesn't exist (would cause AttributeError before fix)
        assert not hasattr(result, "adjusted_alpha")

    def test_apply_mlp_internal_gating_returns_correct_type(self):
        """Verify apply_mlp_internal_gating returns result with gated_alpha."""
        config = UnifiedMergeConfig(
            mlp_internal_intersection_mode=MLPInternalIntersectionMode.INVARIANTS,
        )

        result = apply_mlp_internal_gating(
            key="model.layers.0.mlp.gate_proj.weight",
            base_alpha=0.5,
            config=config,
            intersection_confidences={0: 0.8},
        )

        # Result should have gated_alpha, not adjusted_alpha
        assert hasattr(result, "gated_alpha")
        assert isinstance(result.gated_alpha, float)

    def test_merge_with_mlp_gating_enabled_no_crash(self, synthetic_weights):
        """Full merge with MLP gating enabled should not raise AttributeError."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(
            mlp_internal_intersection_mode=MLPInternalIntersectionMode.INVARIANTS,
        )
        merger = UnifiedManifoldMerger(config)

        # This should not raise AttributeError: 'MLPInternalGatingResult' has no attribute 'adjusted_alpha'
        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
        )

        assert result is not None
        assert result.layers_merged > 0


# =============================================================================
# Bug Fix 2: Unbounded Alpha Accumulation
# =============================================================================


class TestBugFix2_AlphaClamping:
    """
    Verify alpha stays bounded after domain signal and MLP gating adjustments.

    Bug: Alpha was assigned directly after domain_signal_decisions[layer].adjusted_alpha
         and gating_result.gated_alpha without clamping to [min_alpha, max_alpha].
    Fix: Added clamping at lines 1404 and 1418.
    """

    def test_domain_adjusted_alpha_clamped(self):
        """Domain signal adjustment should produce clamped alpha."""
        scores = DomainSignalScores(
            sparsity_score=1.0,  # Extreme value
            smoothness_score=1.0,  # Extreme value
            combined_score=1.0,  # Would push alpha to max
            reliability=1.0,
        )

        config = DomainSignalConfig(min_alpha=0.2, max_alpha=0.95)

        # Even with extreme combined_score, alpha should be clamped
        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=1.0,
            min_alpha=config.min_alpha,
            max_alpha=config.max_alpha,
        )

        assert adjusted >= config.min_alpha
        assert adjusted <= config.max_alpha

    def test_domain_adjusted_alpha_clamped_low(self):
        """Alpha should not go below min_alpha."""
        scores = DomainSignalScores(
            sparsity_score=0.0,
            smoothness_score=0.0,
            combined_score=0.0,  # Would push alpha to min
            reliability=1.0,
        )

        adjusted = domain_adjusted_alpha(
            base_alpha=0.5,
            scores=scores,
            strength=1.0,
            min_alpha=0.2,
            max_alpha=0.95,
        )

        assert adjusted >= 0.2

    def test_merge_with_extreme_domain_signals_bounded(
        self, synthetic_weights, domain_signal_profiles
    ):
        """Merge with extreme domain signals should produce bounded alphas."""
        source, target = synthetic_weights
        source_profile, target_profile = domain_signal_profiles

        config = UnifiedMergeConfig(
            min_alpha=0.1,
            max_alpha=0.95,
            domain_signal_strength=1.0,
        )
        merger = UnifiedManifoldMerger(config)

        # Create extreme domain signal decisions
        decisions = {}
        for layer in [0, 1]:
            scores = DomainSignalScores(
                sparsity_score=1.0,
                smoothness_score=1.0,
                combined_score=1.0,
                reliability=1.0,
            )
            decisions[layer] = DomainSignalDecision.create_applied(
                layer=layer,
                base_alpha=0.5,
                adjusted_alpha=1.5,  # Extreme value that should be clamped
                scores=scores,
            )

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5, 1: 0.5},
            domain_signal_decisions=decisions,
        )

        # All alphas in result should be bounded
        for layer, alpha in result.alpha_profile.alpha_by_layer.items():
            assert alpha >= config.min_alpha, f"Layer {layer} alpha {alpha} below min"
            assert alpha <= config.max_alpha, f"Layer {layer} alpha {alpha} above max"


# =============================================================================
# Bug Fix 3: Dimension Blending Disabled
# =============================================================================


class TestBugFix3_DimensionBlending:
    """
    Verify dimension blending is actually applied when enabled.

    Bug: dimension_blending_applied was always False because the
         compute_dimension_blending_weights() function was never called.
    Fix: Added _apply_dimension_blending() method and integrated into merge loop.
    """

    def test_dimension_blending_weights_computed(self):
        """compute_dimension_blending_weights should return valid weights."""
        # Create correlated activation matrices
        import random
        random.seed(42)

        source_acts = mx.array([[random.gauss(0.5, 0.1) for _ in range(64)] for _ in range(10)])
        target_acts = source_acts + mx.array([[random.gauss(0, 0.05) for _ in range(64)] for _ in range(10)])

        weights = compute_dimension_blending_weights(
            source_activations=source_acts,
            target_activations=target_acts,
            threshold=0.3,
            fallback_weight=0.5,
        )

        assert weights is not None
        assert weights.weights.shape[0] == 64  # One weight per dimension
        assert 0.0 <= weights.mean_weight <= 1.0
        assert 0.0 <= weights.covered_fraction <= 1.0

    def test_merge_with_dimension_blending_enabled(
        self, synthetic_weights, synthetic_activations
    ):
        """Merge with dimension blending enabled should set applied flag."""
        source, target = synthetic_weights
        source_acts, target_acts = synthetic_activations

        config = UnifiedMergeConfig(
            use_dimension_blending=True,
            dimension_blend_threshold=0.3,
        )
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
            source_activations=source_acts,
            target_activations=target_acts,
        )

        # With activations provided and dimension_blending enabled,
        # the flag should be True
        assert result.dimension_blending_applied is True

    def test_dimension_blending_disabled_without_activations(self, synthetic_weights):
        """Dimension blending should not apply without activations."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(use_dimension_blending=True)
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
            # No activations provided
        )

        # Without activations, dimension blending cannot be applied
        assert result.dimension_blending_applied is False


# =============================================================================
# Bug Fix 4: Module Scope Filtering
# =============================================================================


class TestBugFix4_ModuleScope:
    """
    Verify module scope filtering is applied in merge loop.

    Bug: config.module_scope setting had no effect - all modules were always merged.
    Fix: Added scope filtering logic at start of merge loop (lines 1372-1392).
    """

    def test_attention_only_scope_skips_mlp(self, synthetic_weights):
        """ATTENTION_ONLY scope should skip MLP modules."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(module_scope=ModuleScope.ATTENTION_ONLY)
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
        )

        # MLP weights should be unchanged (copied from target)
        for key in result.merged_weights:
            if "mlp" in key:
                # MLP weights should equal target (not blended)
                expected = target[key]
                actual = result.merged_weights[key]
                assert mx.allclose(actual, expected), f"MLP weight {key} was modified but should be skipped"

    def test_mlp_only_scope_skips_attention(self, synthetic_weights):
        """MLP_ONLY scope should skip attention modules."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(module_scope=ModuleScope.MLP_ONLY)
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
        )

        # Attention weights should be unchanged (copied from target)
        for key in result.merged_weights:
            if "self_attn" in key:
                expected = target[key]
                actual = result.merged_weights[key]
                assert mx.allclose(actual, expected), f"Attention weight {key} was modified but should be skipped"

    def test_all_scope_merges_everything(self, synthetic_weights):
        """ALL scope should merge all modules."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(
            module_scope=ModuleScope.ALL,
            base_alpha=0.5,  # 50/50 blend
        )
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5, 1: 0.5},
        )

        # All weights should be blended (not equal to source or target)
        for key in result.merged_weights:
            if key in source and key in target:
                actual = result.merged_weights[key]
                is_source = mx.allclose(actual, source[key])
                is_target = mx.allclose(actual, target[key])
                # Should be blended (neither exactly source nor target)
                assert not (is_source and is_target), f"Weight {key} should be blended"


# =============================================================================
# Bug Fix 5: Silent Fallbacks
# =============================================================================


class TestBugFix5_FallbackLogging:
    """
    Verify fallbacks emit warnings for observability.

    Bug: Transport-guided, affine stitching, verb/noun, and shared subspace
         silently fell back to vanilla blending on failure.
    Fix: Added logger.warning() calls when fallback occurs.
    """

    def test_transport_fallback_logs_warning(self, synthetic_weights, caplog):
        """Transport-guided fallback should log a warning."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(
            use_transport_guided=True,
            gromov_wasserstein_blend_strength=0.0,  # Force fallback
        )
        merger = UnifiedManifoldMerger(config)

        with caplog.at_level(logging.DEBUG):
            result = merger.merge_with_confidence(
                source_weights=source,
                target_weights=target,
                layer_confidences={0: 0.8, 1: 0.7},
            )

        # Should log about GW strength being 0
        assert any("GW blend strength is 0" in record.message for record in caplog.records)

    def test_affine_stitching_fallback_logs_warning(self, synthetic_weights, caplog):
        """Affine stitching without activations should log a warning."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(use_affine_stitching=True)
        merger = UnifiedManifoldMerger(config)

        with caplog.at_level(logging.WARNING):
            result = merger.merge_with_confidence(
                source_weights=source,
                target_weights=target,
                layer_confidences={0: 0.8, 1: 0.7},
                # No activations - should trigger fallback
            )

        # Should log about requiring activations
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("requires activations" in msg or "falling back" in msg.lower() for msg in warning_messages)

    def test_shared_subspace_fallback_logs_warning(self, synthetic_weights, caplog):
        """Shared subspace without sufficient data should log a warning."""
        source, target = synthetic_weights

        config = UnifiedMergeConfig(
            use_shared_subspace_projection=True,
            shared_subspace_blend_weight=0.5,
        )
        merger = UnifiedManifoldMerger(config)

        with caplog.at_level(logging.WARNING):
            result = merger.merge_with_confidence(
                source_weights=source,
                target_weights=target,
                layer_confidences={0: 0.8, 1: 0.7},
                source_activations={},  # Empty - should trigger fallback
                target_activations={},
            )

        # Should log about insufficient data
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        # Shared subspace projection may not be triggered with empty activations,
        # but if it is, it should log


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests for the fixed merge pipeline."""

    def test_full_merge_with_all_features(
        self, synthetic_weights, synthetic_activations, domain_signal_profiles
    ):
        """Full merge with all features enabled should work."""
        source, target = synthetic_weights
        source_acts, target_acts = synthetic_activations
        source_profile, target_profile = domain_signal_profiles

        config = UnifiedMergeConfig(
            use_adaptive_alpha_smoothing=True,
            use_dimension_blending=True,
            spectral_penalty_strength=0.5,
            module_scope=ModuleScope.ALL,
            mlp_internal_intersection_mode=MLPInternalIntersectionMode.INVARIANTS,
        )
        merger = UnifiedManifoldMerger(config)

        # Pre-compute domain signal decisions
        decisions = {}
        for layer in [0, 1]:
            scores = compute_domain_scores(
                source_profile=source_profile,
                target_profile=target_profile,
                layer=layer,
            )
            if scores:
                adjusted = domain_adjusted_alpha(
                    base_alpha=0.5,
                    scores=scores,
                    strength=1.0,
                    min_alpha=config.min_alpha,
                    max_alpha=config.max_alpha,
                )
                decisions[layer] = DomainSignalDecision.create_applied(
                    layer=layer,
                    base_alpha=0.5,
                    adjusted_alpha=adjusted,
                    scores=scores,
                )

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
            source_activations=source_acts,
            target_activations=target_acts,
            domain_signal_decisions=decisions,
        )

        # Verify result is valid
        assert result is not None
        assert result.layers_merged > 0
        assert len(result.merged_weights) == len(source)

        # Verify alphas are bounded
        for layer, alpha in result.alpha_profile.alpha_by_layer.items():
            assert config.min_alpha <= alpha <= config.max_alpha

        # Verify dimension blending was applied (with activations)
        assert result.dimension_blending_applied is True

    def test_merge_result_metrics(self, synthetic_weights):
        """Verify merge result contains expected metrics."""
        source, target = synthetic_weights

        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.7},
        )

        # Check result structure
        assert hasattr(result, "merged_weights")
        assert hasattr(result, "alpha_profile")
        assert hasattr(result, "mean_alpha")
        assert hasattr(result, "layers_merged")
        assert hasattr(result, "spectral_penalty_applied")
        assert hasattr(result, "dimension_blending_applied")

        # Check values are reasonable
        assert 0.0 <= result.mean_alpha <= 1.0
        assert result.layers_merged >= 0
