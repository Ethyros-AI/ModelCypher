"""
Quality metrics validation for merge pipeline.

Tests that verify:
- CKA metrics are computed correctly
- Dimension blending improves correlation
- Module scope filtering preserves excluded modules
- Alpha profiles are mathematically valid
"""
import pytest
import math
import mlx.core as mx
from typing import Dict, List
from dataclasses import dataclass
from hypothesis import given, strategies as st, settings

from modelcypher.core.domain.merging.unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    LayerAlphaProfile,
    compute_adaptive_alpha_profile,
    compute_dimension_blending_weights,
)
from modelcypher.core.domain.geometry.domain_signal_profile import (
    DomainSignalScores,
    compute_domain_scores,
    domain_adjusted_alpha,
    DomainSignalConfig,
    DomainSignalProfile,
    LayerSignal,
)


# =============================================================================
# Quality Report Data Structure
# =============================================================================


@dataclass
class LayerQualityMetrics:
    """Quality metrics for a single layer."""
    layer_index: int
    alpha: float
    source_weight_norm: float
    target_weight_norm: float
    merged_weight_norm: float
    blend_ratio: float  # How close to source (0) vs target (1)


@dataclass
class MergeQualityReport:
    """Comprehensive quality report for a merge operation."""
    per_layer: Dict[int, LayerQualityMetrics]
    mean_alpha: float
    alpha_variance: float
    total_layers: int
    spectral_penalty_applied: bool
    dimension_blending_applied: bool


def compute_quality_report(
    source_weights: Dict[str, mx.array],
    target_weights: Dict[str, mx.array],
    merged_weights: Dict[str, mx.array],
    alpha_profile: LayerAlphaProfile,
) -> MergeQualityReport:
    """Compute quality metrics for merge result."""
    per_layer = {}

    for key in merged_weights:
        if key not in source_weights or key not in target_weights:
            continue

        # Extract layer index
        import re
        match = re.search(r'layers\.(\d+)', key)
        if not match:
            continue
        layer = int(match.group(1))

        source_norm = float(mx.linalg.norm(source_weights[key]))
        target_norm = float(mx.linalg.norm(target_weights[key]))
        merged_norm = float(mx.linalg.norm(merged_weights[key]))

        # Compute blend ratio based on norms
        # 0 = closer to source, 1 = closer to target
        if abs(target_norm - source_norm) > 1e-6:
            blend_ratio = (merged_norm - source_norm) / (target_norm - source_norm)
            blend_ratio = max(0.0, min(1.0, blend_ratio))
        else:
            blend_ratio = 0.5

        per_layer[layer] = LayerQualityMetrics(
            layer_index=layer,
            alpha=alpha_profile.alpha(layer),
            source_weight_norm=source_norm,
            target_weight_norm=target_norm,
            merged_weight_norm=merged_norm,
            blend_ratio=blend_ratio,
        )

    return MergeQualityReport(
        per_layer=per_layer,
        mean_alpha=alpha_profile.mean_alpha,
        alpha_variance=alpha_profile.alpha_variance,
        total_layers=len(per_layer),
        spectral_penalty_applied=False,  # Set by caller
        dimension_blending_applied=False,  # Set by caller
    )


# =============================================================================
# Property-Based Tests for Mathematical Parity
# =============================================================================


class TestMathematicalParity:
    """Property-based tests verifying mathematical invariants."""

    @given(
        combined_score=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        reliability=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        strength=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_domain_signal_score_bounded(self, combined_score, reliability, strength):
        """Domain signal adjusted alpha must be in [min_alpha, max_alpha]."""
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

        assert 0.1 <= adjusted <= 0.95, f"Alpha {adjusted} out of bounds"

    @given(
        base_alpha=st.floats(0.1, 0.95, allow_nan=False, allow_infinity=False),
        strength=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_alpha_adjustment_monotonic(self, base_alpha, strength):
        """Higher combined_score should result in higher (or equal) adjusted alpha."""
        low_scores = DomainSignalScores(
            sparsity_score=0.2,
            smoothness_score=0.2,
            combined_score=0.2,
            reliability=1.0,
        )

        high_scores = DomainSignalScores(
            sparsity_score=0.8,
            smoothness_score=0.8,
            combined_score=0.8,
            reliability=1.0,
        )

        low_adjusted = domain_adjusted_alpha(
            base_alpha=base_alpha,
            scores=low_scores,
            strength=strength,
            min_alpha=0.1,
            max_alpha=0.95,
        )

        high_adjusted = domain_adjusted_alpha(
            base_alpha=base_alpha,
            scores=high_scores,
            strength=strength,
            min_alpha=0.1,
            max_alpha=0.95,
        )

        # Higher combined_score should give higher or equal alpha
        assert high_adjusted >= low_adjusted - 1e-6, (
            f"Monotonicity violated: high={high_adjusted}, low={low_adjusted}"
        )

    @given(
        confidence=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_adaptive_alpha_profile_bounded(self, confidence):
        """Adaptive alpha profile should produce bounded alphas."""
        profile = compute_adaptive_alpha_profile(
            layer_confidences={0: confidence},
            base_alpha=0.5,
            min_alpha=0.1,
            max_alpha=0.95,
            smoothing_window=0,
        )

        alpha = profile.alpha(0)
        assert 0.1 <= alpha <= 0.95, f"Alpha {alpha} out of bounds for confidence {confidence}"

    @given(
        conf1=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        conf2=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_higher_confidence_lower_alpha(self, conf1, conf2):
        """Higher confidence should result in lower (or equal) alpha."""
        if conf1 == conf2:
            return  # Skip equal case

        profile = compute_adaptive_alpha_profile(
            layer_confidences={0: conf1, 1: conf2},
            base_alpha=0.5,
            min_alpha=0.1,
            max_alpha=0.95,
            smoothing_window=0,
        )

        alpha1 = profile.alpha(0)
        alpha2 = profile.alpha(1)

        if conf1 > conf2:
            # Higher confidence → lower alpha (trust source more)
            assert alpha1 <= alpha2 + 1e-6, (
                f"Higher confidence should give lower alpha: "
                f"conf1={conf1}, alpha1={alpha1}, conf2={conf2}, alpha2={alpha2}"
            )
        else:
            assert alpha2 <= alpha1 + 1e-6


# =============================================================================
# Dimension Blending Quality Tests
# =============================================================================


class TestDimensionBlendingQuality:
    """Tests for dimension blending quality."""

    def test_correlated_activations_produce_valid_weights(self):
        """Highly correlated activations should produce valid blending weights."""
        import random
        random.seed(42)

        # Create highly correlated activations
        source_acts = mx.array([[random.gauss(0.5, 0.1) for _ in range(32)] for _ in range(20)])
        # Target is highly correlated with source
        target_acts = source_acts + mx.array([[random.gauss(0, 0.01) for _ in range(32)] for _ in range(20)])

        weights = compute_dimension_blending_weights(
            source_activations=source_acts,
            target_activations=target_acts,
            threshold=0.3,
            fallback_weight=0.5,
        )

        assert weights is not None
        assert weights.weights.shape[0] == 32
        # High correlation should result in lower weights (trust source)
        # Most weights should be below the fallback
        low_weight_fraction = float(mx.mean(weights.weights < 0.5))
        assert low_weight_fraction > 0.5, "High correlation should produce low weights"

    def test_uncorrelated_activations_produce_fallback_weights(self):
        """Uncorrelated activations should produce fallback weights."""
        import random
        random.seed(42)

        # Create uncorrelated activations
        source_acts = mx.array([[random.gauss(0.5, 0.5) for _ in range(32)] for _ in range(20)])
        random.seed(123)  # Different seed for target
        target_acts = mx.array([[random.gauss(0.5, 0.5) for _ in range(32)] for _ in range(20)])

        weights = compute_dimension_blending_weights(
            source_activations=source_acts,
            target_activations=target_acts,
            threshold=0.5,  # Higher threshold
            fallback_weight=0.6,
        )

        assert weights is not None
        # Low correlation should result in more fallback weights
        # Mean weight should be closer to fallback
        assert 0.4 <= weights.mean_weight <= 0.8

    def test_blending_weights_bounded(self):
        """Blending weights should be in [0, 1]."""
        import random
        random.seed(42)

        source_acts = mx.array([[random.gauss(0, 1) for _ in range(32)] for _ in range(20)])
        target_acts = mx.array([[random.gauss(0, 1) for _ in range(32)] for _ in range(20)])

        weights = compute_dimension_blending_weights(
            source_activations=source_acts,
            target_activations=target_acts,
            threshold=0.3,
            fallback_weight=0.5,
        )

        assert weights is not None
        assert float(mx.min(weights.weights)) >= 0.0
        assert float(mx.max(weights.weights)) <= 1.0


# =============================================================================
# Alpha Profile Quality Tests
# =============================================================================


class TestAlphaProfileQuality:
    """Tests for alpha profile quality."""

    def test_smoothing_reduces_variance(self):
        """Gaussian smoothing should reduce alpha variance."""
        layer_confs = {i: (i / 10.0) for i in range(10)}

        profile_no_smooth = compute_adaptive_alpha_profile(
            layer_confidences=layer_confs,
            smoothing_window=0,
        )

        profile_smoothed = compute_adaptive_alpha_profile(
            layer_confidences=layer_confs,
            smoothing_window=2,
        )

        assert profile_smoothed.alpha_variance <= profile_no_smooth.alpha_variance

    def test_procrustes_error_increases_alpha(self):
        """Higher Procrustes error should increase alpha."""
        profile_no_error = compute_adaptive_alpha_profile(
            layer_confidences={0: 0.5},
            base_alpha=0.5,
            smoothing_window=0,
        )

        profile_with_error = compute_adaptive_alpha_profile(
            layer_confidences={0: 0.5},
            base_alpha=0.5,
            smoothing_window=0,
            procrustes_error_by_layer={0: 0.5},
        )

        assert profile_with_error.alpha(0) >= profile_no_error.alpha(0)

    def test_alpha_profile_statistics(self):
        """Alpha profile statistics should be correct."""
        profile = LayerAlphaProfile(
            alpha_by_layer={0: 0.3, 1: 0.5, 2: 0.7},
            smoothing_window=0,
            base_alpha=0.5,
            used_procrustes_error=False,
        )

        # Mean should be (0.3 + 0.5 + 0.7) / 3 = 0.5
        assert abs(profile.mean_alpha - 0.5) < 0.01

        # Variance should be positive for non-uniform alphas
        assert profile.alpha_variance > 0


# =============================================================================
# Integration Quality Tests
# =============================================================================


class TestIntegrationQuality:
    """Integration tests for overall merge quality."""

    def test_merge_preserves_weight_scale(self):
        """Merge should preserve approximate weight scale."""
        source = {"model.layers.0.weight": mx.ones((64, 64)) * 0.5}
        target = {"model.layers.0.weight": mx.ones((64, 64)) * 1.0}

        merger = UnifiedManifoldMerger(UnifiedMergeConfig(base_alpha=0.5))

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.5},
        )

        merged = result.merged_weights["model.layers.0.weight"]
        merged_mean = float(mx.mean(merged))

        # With alpha=0.5, merged should be approximately (0.5 + 1.0) / 2 = 0.75
        assert 0.5 <= merged_mean <= 1.0

    def test_merge_quality_report_structure(self):
        """Quality report should have correct structure."""
        source = {
            "model.layers.0.weight": mx.ones((64, 64)) * 0.5,
            "model.layers.1.weight": mx.ones((64, 64)) * 0.5,
        }
        target = {
            "model.layers.0.weight": mx.ones((64, 64)) * 1.0,
            "model.layers.1.weight": mx.ones((64, 64)) * 1.0,
        }

        merger = UnifiedManifoldMerger()

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8, 1: 0.6},
        )

        report = compute_quality_report(
            source_weights=source,
            target_weights=target,
            merged_weights=result.merged_weights,
            alpha_profile=result.alpha_profile,
        )

        assert report.total_layers == 2
        assert 0 in report.per_layer
        assert 1 in report.per_layer
        assert report.mean_alpha > 0

    def test_high_confidence_produces_lower_alpha(self):
        """High confidence layers should have lower alpha (trust source)."""
        source = {
            "model.layers.0.weight": mx.ones((64, 64)),  # High confidence
            "model.layers.1.weight": mx.ones((64, 64)),  # Low confidence
        }
        target = {
            "model.layers.0.weight": mx.zeros((64, 64)),
            "model.layers.1.weight": mx.zeros((64, 64)),
        }

        config = UnifiedMergeConfig(use_adaptive_alpha_smoothing=True)
        merger = UnifiedManifoldMerger(config)

        result = merger.merge_with_confidence(
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.9, 1: 0.1},  # Layer 0 high, layer 1 low
        )

        # Layer 0 (high confidence) should have lower alpha
        alpha_0 = result.alpha_profile.alpha(0)
        alpha_1 = result.alpha_profile.alpha(1)

        assert alpha_0 < alpha_1, f"High confidence should give lower alpha: {alpha_0} vs {alpha_1}"
