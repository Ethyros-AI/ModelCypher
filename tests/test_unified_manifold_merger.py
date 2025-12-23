"""
Tests for UnifiedManifoldMerger (adaptive alpha, spectral penalty, dimension blending).
"""
import pytest
import math
import mlx.core as mx

from modelcypher.core.domain.merging.unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    UnifiedMergeResult,
    LayerAlphaProfile,
    compute_adaptive_alpha_profile,
    compute_spectral_penalty,
    apply_spectral_penalty_to_alpha,
    compute_dimension_blending_weights,
    BlendMode,
    ModuleBlendPolicy,
)


class TestLayerAlphaProfile:
    """Tests for the LayerAlphaProfile dataclass."""
    
    def test_mean_alpha_single_layer(self):
        """Mean alpha with single layer should equal that layer's alpha."""
        profile = LayerAlphaProfile(
            alpha_by_layer={5: 0.7},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        
        assert profile.mean_alpha == 0.7
    
    def test_mean_alpha_multiple_layers(self):
        """Mean alpha should average across layers."""
        profile = LayerAlphaProfile(
            alpha_by_layer={0: 0.2, 1: 0.4, 2: 0.6},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        
        assert abs(profile.mean_alpha - 0.4) < 0.01
    
    def test_mean_alpha_empty(self):
        """Empty profile should return base alpha."""
        profile = LayerAlphaProfile(
            alpha_by_layer={},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        
        assert profile.mean_alpha == 0.5
    
    def test_alpha_fallback(self):
        """Missing layer should return base alpha."""
        profile = LayerAlphaProfile(
            alpha_by_layer={5: 0.7},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        
        assert profile.alpha(5) == 0.7
        assert profile.alpha(99) == 0.5  # Not in profile
    
    def test_alpha_variance(self):
        """Variance should measure spread of alphas."""
        # All same = 0 variance
        profile1 = LayerAlphaProfile(
            alpha_by_layer={0: 0.5, 1: 0.5, 2: 0.5},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        assert profile1.alpha_variance == 0.0
        
        # Different values = positive variance
        profile2 = LayerAlphaProfile(
            alpha_by_layer={0: 0.2, 1: 0.5, 2: 0.8},
            smoothing_window=2,
            base_alpha=0.5,
            used_procrustes_error=False,
        )
        assert profile2.alpha_variance > 0


class TestComputeAdaptiveAlphaProfile:
    """Tests for the adaptive alpha profile computation."""
    
    def test_high_confidence_lowers_alpha(self):
        """High confidence should result in lower alpha (trust source)."""
        profile = compute_adaptive_alpha_profile(
            layer_confidences={0: 0.9, 1: 0.1},
            base_alpha=0.5,
            smoothing_window=0,  # Disable smoothing for clarity
        )
        
        # Layer 0 (high confidence) should have lower alpha
        # Layer 1 (low confidence) should have higher alpha
        assert profile.alpha(0) < profile.alpha(1)
    
    def test_procrustes_error_increases_alpha(self):
        """Higher Procrustes error should increase alpha (trust target)."""
        profile_without = compute_adaptive_alpha_profile(
            layer_confidences={0: 0.5},
            base_alpha=0.5,
            smoothing_window=0,
        )
        
        profile_with = compute_adaptive_alpha_profile(
            layer_confidences={0: 0.5},
            base_alpha=0.5,
            smoothing_window=0,
            procrustes_error_by_layer={0: 0.5},  # Moderate error
        )
        
        # Error should push alpha higher
        assert profile_with.alpha(0) > profile_without.alpha(0)
    
    def test_smoothing_reduces_variance(self):
        """Gaussian smoothing should reduce alpha variance."""
        layer_confs = {i: (i / 10.0) for i in range(10)}  # Varying confidences
        
        profile_no_smooth = compute_adaptive_alpha_profile(
            layer_confidences=layer_confs,
            smoothing_window=0,
        )
        
        profile_smoothed = compute_adaptive_alpha_profile(
            layer_confidences=layer_confs,
            smoothing_window=2,
        )
        
        # Smoothing should reduce variance
        assert profile_smoothed.alpha_variance < profile_no_smooth.alpha_variance
    
    def test_clamping(self):
        """Alphas should be clamped to [min_alpha, max_alpha]."""
        profile = compute_adaptive_alpha_profile(
            layer_confidences={0: 1.0, 1: 0.0},  # Extreme confidences
            min_alpha=0.2,
            max_alpha=0.8,
            smoothing_window=0,
        )
        
        assert profile.alpha(0) >= 0.2
        assert profile.alpha(0) <= 0.8
        assert profile.alpha(1) >= 0.2
        assert profile.alpha(1) <= 0.8
    
    def test_empty_confidences(self):
        """Empty confidences should return empty profile."""
        profile = compute_adaptive_alpha_profile(
            layer_confidences={},
            base_alpha=0.6,
        )
        
        assert len(profile.alpha_by_layer) == 0
        assert profile.base_alpha == 0.6


class TestSpectralPenalty:
    """Tests for spectral penalty computation."""
    
    def test_identity_matrix_finite_penalty(self):
        """Identity matrix should have a finite penalty."""
        weight = mx.eye(10)
        penalty = compute_spectral_penalty(weight)
        
        # Identity has condition number 1, penalty should be bounded
        assert 0.0 <= penalty <= 1.0
    
    def test_ill_conditioned_bounded_penalty(self):
        """Near-singular matrix should have bounded penalty."""
        # Create a matrix with varying singular values
        weight = mx.array([[1, 0], [0, 0.01]], dtype=mx.float32)  # Condition number = 100
        penalty = compute_spectral_penalty(weight)
        
        # Penalty should be bounded in [0, 1]
        assert 0.0 <= penalty <= 1.0
    
    def test_1d_returns_zero(self):
        """Non-2D input should return 0."""
        weight = mx.array([1.0, 2.0, 3.0])
        penalty = compute_spectral_penalty(weight)
        
        assert penalty == 0.0


class TestApplySpectralPenaltyToAlpha:
    """Tests for spectral penalty application to alpha."""
    
    def test_no_penalty_when_disabled(self):
        """Strength 0 should return unchanged alpha."""
        source = mx.eye(10)
        target = mx.eye(10)
        
        adjusted = apply_spectral_penalty_to_alpha(
            alpha=0.5,
            source_weight=source,
            target_weight=target,
            strength=0.0,
        )
        
        assert adjusted == 0.5
    
    def test_ill_source_bounded_alpha(self):
        """Spectral penalty should keep alpha bounded."""
        source = mx.array([[1, 0], [0, 0.001]], dtype=mx.float32)  # Ill-conditioned
        target = mx.eye(2)  # Well-conditioned
        
        adjusted = apply_spectral_penalty_to_alpha(
            alpha=0.5,
            source_weight=source,
            target_weight=target,
            strength=1.0,
        )
        
        # Alpha should remain bounded
        assert 0.1 <= adjusted <= 0.95
    
    def test_ill_target_bounded_alpha(self):
        """Spectral penalty with ill target should keep alpha bounded."""
        source = mx.eye(2)  # Well-conditioned
        target = mx.array([[1, 0], [0, 0.001]], dtype=mx.float32)  # Ill-conditioned
        
        adjusted = apply_spectral_penalty_to_alpha(
            alpha=0.5,
            source_weight=source,
            target_weight=target,
            strength=1.0,
        )
        
        # Alpha should remain bounded
        assert 0.1 <= adjusted <= 0.95


class TestUnifiedMergeConfig:
    """Tests for merge configuration."""
    
    def test_default_config(self):
        """Default config should have reasonable values."""
        config = UnifiedMergeConfig()
        
        assert config.base_alpha == 0.5
        assert config.alignment_rank == 32
        assert config.use_adaptive_alpha_smoothing == True
    
    def test_conservative_preset(self):
        """Conservative preset should favor target."""
        config = UnifiedMergeConfig.conservative()
        
        assert config.base_alpha > 0.5  # Favor target
        assert config.permutation_confidence_threshold > 0.5
    
    def test_aggressive_preset(self):
        """Aggressive preset should favor source."""
        config = UnifiedMergeConfig.aggressive()
        
        assert config.base_alpha < 0.5  # Favor source
        assert config.permutation_confidence_threshold < 0.5


class TestModuleBlendPolicy:
    """Tests for module blend policy."""
    
    def test_default_policy(self):
        """Default policy should have standard module sets."""
        policy = ModuleBlendPolicy()
        
        assert "q_proj" in policy.soft_blend_kinds
        assert "v_proj" in policy.hard_swap_kinds
        assert "o_proj" in policy.skip_kinds


class TestUnifiedManifoldMerger:
    """Integration tests for the unified merger."""
    
    def test_extract_layer_index(self):
        """Layer index extraction should work."""
        merger = UnifiedManifoldMerger()
        
        assert merger._extract_layer_index("model.layers.5.mlp.up_proj.weight") == 5
        assert merger._extract_layer_index("model.layers.12.self_attn.q_proj.weight") == 12
        assert merger._extract_layer_index("embed_tokens.weight") == 0
    
    def test_compute_blend_mode_skip(self):
        """Skip kinds should return SKIP mode."""
        merger = UnifiedManifoldMerger(UnifiedMergeConfig(use_module_blend_policy=True))
        
        mode = merger.compute_blend_mode("model.layers.5.self_attn.o_proj.weight", 0.5)
        assert mode == BlendMode.SKIP
    
    def test_compute_blend_mode_soft(self):
        """Regular modules should return SOFT mode."""
        merger = UnifiedManifoldMerger(UnifiedMergeConfig(use_module_blend_policy=True))
        
        mode = merger.compute_blend_mode("model.layers.5.self_attn.q_proj.weight", 0.5)
        assert mode == BlendMode.SOFT
    
    def test_merge_with_confidence_basic(self):
        """Basic merge with confidence should work."""
        merger = UnifiedManifoldMerger()
        
        source = {"layer.0.weight": mx.ones((4, 4))}
        target = {"layer.0.weight": mx.zeros((4, 4))}
        confidences = {0: 0.8}  # High confidence
        
        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences=confidences,
        )
        
        assert "layer.0.weight" in result.merged_weights
        assert result.layers_merged == 1
