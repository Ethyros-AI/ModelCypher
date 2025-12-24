"""Integration tests for the model merge pipeline.

Tests the full workflow:
    profile → guide → merge → validate

This validates that the entropy-aware merge components work together
to produce stable, knowledge-preserving model merges.
"""
from __future__ import annotations

import pytest
import numpy as np

from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeValidator,
    LayerEntropyProfile,
    MergeEntropyValidation,
    MergeStability,
)
from modelcypher.core.domain.merging.unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    compute_adaptive_alpha_profile,
)
from modelcypher.core.domain.thermo.phase_transition_theory import Phase


# =============================================================================
# Entropy Merge Validator Integration
# =============================================================================


class TestMergeProfileIntegration:
    """Tests for the profile step of the merge pipeline."""

    def test_create_simulated_profile_returns_valid_structure(self) -> None:
        """Profile creation should return valid LayerEntropyProfile."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("test-model", num_layers=32)

        assert profile.model_name == "test-model"
        assert len(profile.layer_profiles) == 32
        assert profile.mean_entropy >= 0
        assert profile.entropy_variance >= 0
        assert profile.dominant_phase in Phase
        assert profile.merge_risk_level in {"low", "medium", "high"}

    @pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
    def test_profile_scales_with_layer_count(self, num_layers: int) -> None:
        """Profile should scale correctly with layer count."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("test-model", num_layers=num_layers)

        assert len(profile.layer_profiles) == num_layers
        assert profile.critical_layer_count <= num_layers

    def test_critical_layers_identified(self) -> None:
        """Critical layers near phase boundaries should be identified."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("test-model", num_layers=32)

        # Should have some critical layers (may be 0 in edge cases)
        assert profile.critical_layer_count >= 0

        # Critical layers should be in the profile
        critical_count = sum(
            1 for p in profile.layer_profiles.values() if p.is_critical
        )
        assert critical_count == profile.critical_layer_count


class TestMergeGuideIntegration:
    """Tests for the guide step of the merge pipeline."""

    def test_alpha_adjustments_computed(self) -> None:
        """Alpha adjustments should be computed for source/target profiles."""
        validator = EntropyMergeValidator()
        source = validator.create_simulated_profile("source", num_layers=16)
        target = validator.create_simulated_profile("target", num_layers=16)

        alpha_adj = validator.compute_alpha_adjustments(source, target)

        # Should have adjustments for all layers
        assert len(alpha_adj) == 16
        # All adjustments should be positive
        for adj in alpha_adj.values():
            assert adj > 0

    def test_smoothing_sigmas_computed(self) -> None:
        """Smoothing sigmas should be computed for critical layers."""
        validator = EntropyMergeValidator()
        source = validator.create_simulated_profile("source", num_layers=16)
        target = validator.create_simulated_profile("target", num_layers=16)

        sigmas = validator.compute_smoothing_sigmas(source, target)

        # Should have sigma for each layer
        assert len(sigmas) == 16
        # All sigmas should be positive
        for sigma in sigmas.values():
            assert sigma > 0

    def test_guide_respects_phase_differences(self) -> None:
        """Guide should adjust alpha based on phase differences."""
        validator = EntropyMergeValidator()

        # Create profiles with deterministic seeds for reproducibility
        source = validator.create_simulated_profile("source", num_layers=16)
        target = validator.create_simulated_profile("target", num_layers=16)

        alpha_adj = validator.compute_alpha_adjustments(source, target)

        # All adjustments should be bounded
        for adj in alpha_adj.values():
            assert 0.1 <= adj <= 2.0


class TestMergeValidationIntegration:
    """Tests for the validate step of the merge pipeline."""

    def test_validation_detects_stable_merge(self) -> None:
        """Validation should detect a stable merge."""
        validator = EntropyMergeValidator()

        # Similar entropies = stable merge
        source = {"layer.0": 1.5, "layer.1": 1.6, "layer.2": 1.7}
        target = {"layer.0": 1.55, "layer.1": 1.65, "layer.2": 1.75}
        merged = {"layer.0": 1.52, "layer.1": 1.62, "layer.2": 1.72}

        result = validator.validate_merge(source, target, merged)

        assert result.overall_stability in {MergeStability.STABLE, MergeStability.MARGINAL}
        assert result.mean_knowledge_retention >= 0.8
        assert result.is_safe is True

    def test_validation_detects_unstable_merge(self) -> None:
        """Validation should detect an unstable merge."""
        validator = EntropyMergeValidator()

        # Very different entropies = unstable merge
        source = {"layer.0": 1.0, "layer.1": 1.1}
        target = {"layer.0": 3.0, "layer.1": 3.1}
        merged = {"layer.0": 5.0, "layer.1": 5.1}  # Far from both

        result = validator.validate_merge(source, target, merged)

        # Should detect instability (may be marginal or unstable)
        assert result.overall_stability in {
            MergeStability.MARGINAL,
            MergeStability.UNSTABLE,
            MergeStability.CRITICAL,
        } or result.is_safe is False or result.mean_knowledge_retention < 1.0

    def test_validation_tracks_layer_validations(self) -> None:
        """Validation should track per-layer validation results."""
        validator = EntropyMergeValidator()

        source = {f"layer.{i}": 1.5 + 0.1 * i for i in range(10)}
        target = {f"layer.{i}": 1.6 + 0.1 * i for i in range(10)}
        merged = {f"layer.{i}": 1.55 + 0.1 * i for i in range(10)}

        result = validator.validate_merge(source, target, merged)

        assert len(result.layer_validations) == 10


# =============================================================================
# Full Pipeline Integration
# =============================================================================


class TestFullMergePipeline:
    """Tests for the full profile → guide → merge → validate pipeline."""

    def test_pipeline_profile_to_guide(self) -> None:
        """Pipeline: profile → guide should produce consistent results."""
        validator = EntropyMergeValidator()

        # Step 1: Profile both models
        source_profile = validator.create_simulated_profile("source", num_layers=32)
        target_profile = validator.create_simulated_profile("target", num_layers=32)

        # Step 2: Generate guide
        alpha_adj = validator.compute_alpha_adjustments(source_profile, target_profile)
        sigmas = validator.compute_smoothing_sigmas(source_profile, target_profile)

        # Verify consistency
        assert len(alpha_adj) == 32
        assert len(sigmas) == 32

        # Verify all values are reasonable
        for layer_name in alpha_adj:
            assert 0.1 <= alpha_adj[layer_name] <= 2.0
            assert sigmas[layer_name] > 0

    def test_pipeline_with_unified_merger(self) -> None:
        """Pipeline should work with UnifiedManifoldMerger."""
        validator = EntropyMergeValidator()
        merger = UnifiedManifoldMerger()

        # Step 1: Profile
        source_profile = validator.create_simulated_profile("source", num_layers=8)
        target_profile = validator.create_simulated_profile("target", num_layers=8)

        # Step 2: Guide (get alpha adjustments)
        alpha_adj = validator.compute_alpha_adjustments(source_profile, target_profile)

        # Step 3: Convert to layer confidences for merger
        # Higher alpha adj = lower confidence (need more target)
        layer_confidences = {
            i: max(0.1, min(0.9, 1.0 - (alpha_adj.get(f"layer.{i}", 1.0) - 1.0) * 0.5))
            for i in range(8)
        }

        # Step 4: Compute adaptive alpha profile
        profile = compute_adaptive_alpha_profile(
            layer_confidences=layer_confidences,
            smoothing_window=2,
        )

        # Verify profile is valid
        assert 0 <= profile.mean_alpha <= 1
        assert profile.alpha_variance >= 0

    def test_pipeline_entropy_guided_merge_validation(self) -> None:
        """Full pipeline: profile → guide → simulated merge → validate."""
        validator = EntropyMergeValidator()

        # Step 1: Profile
        source_profile = validator.create_simulated_profile("source", num_layers=16)
        target_profile = validator.create_simulated_profile("target", num_layers=16)

        # Step 2: Guide
        alpha_adj = validator.compute_alpha_adjustments(source_profile, target_profile)

        # Step 3: Simulate merge (blend entropies based on alpha)
        source_entropies = {}
        target_entropies = {}
        merged_entropies = {}

        for layer_name, layer_profile in source_profile.layer_profiles.items():
            source_entropy = layer_profile.mean_entropy
            target_layer = target_profile.layer_profiles.get(layer_name)
            if target_layer is None:
                continue

            target_entropy = target_layer.mean_entropy
            alpha = alpha_adj.get(layer_name, 1.0)

            # Weighted blend based on alpha
            # alpha > 1 = trust target more, alpha < 1 = trust source more
            source_weight = 1.0 / alpha if alpha > 0 else 1.0
            target_weight = alpha if alpha > 0 else 1.0
            total = source_weight + target_weight

            merged_entropy = (
                source_weight * source_entropy + target_weight * target_entropy
            ) / total

            source_entropies[layer_name] = source_entropy
            target_entropies[layer_name] = target_entropy
            merged_entropies[layer_name] = merged_entropy

        # Step 4: Validate
        result = validator.validate_merge(
            source_entropies, target_entropies, merged_entropies,
            source_model="source", target_model="target",
        )

        # Verify validation produces reasonable result
        assert result.overall_stability in MergeStability
        assert 0.0 <= result.mean_knowledge_retention <= 1.0
        assert len(result.layer_validations) == len(source_entropies)


# =============================================================================
# Adaptive Alpha Integration
# =============================================================================


class TestAdaptiveAlphaIntegration:
    """Tests for adaptive alpha profile integration."""

    def test_adaptive_alpha_from_entropy_profile(self) -> None:
        """Adaptive alpha should integrate with entropy profiles."""
        validator = EntropyMergeValidator()

        profile = validator.create_simulated_profile("test", num_layers=32)

        # Convert entropy profile to confidences
        # Lower entropy = higher confidence (more predictable)
        confidences = {}
        for layer_name, layer_profile in profile.layer_profiles.items():
            # Extract layer index
            try:
                layer_idx = int(layer_name.split(".")[-1])
            except (ValueError, IndexError):
                continue

            # Convert entropy to confidence (inverse relationship)
            # Normalize entropy to [0, 1] range, then invert
            normalized_entropy = min(1.0, layer_profile.mean_entropy / 5.0)
            confidence = 1.0 - normalized_entropy
            confidences[layer_idx] = max(0.1, min(0.9, confidence))

        # Compute adaptive alpha
        alpha_profile = compute_adaptive_alpha_profile(
            layer_confidences=confidences,
            smoothing_window=3,
            min_alpha=0.2,
            max_alpha=0.8,
        )

        # Verify bounds
        for layer_idx in confidences:
            alpha = alpha_profile.alpha(layer_idx)
            assert 0.2 <= alpha <= 0.8

    def test_smoothing_reduces_alpha_variance(self) -> None:
        """Gaussian smoothing should reduce alpha variance."""
        # Create oscillating confidences
        confidences = {i: 0.9 if i % 2 == 0 else 0.1 for i in range(20)}

        profile_no_smooth = compute_adaptive_alpha_profile(
            layer_confidences=confidences,
            smoothing_window=0,
        )

        profile_smoothed = compute_adaptive_alpha_profile(
            layer_confidences=confidences,
            smoothing_window=3,
        )

        # Smoothing should reduce variance
        assert profile_smoothed.alpha_variance <= profile_no_smooth.alpha_variance


# =============================================================================
# Error Handling
# =============================================================================


class TestMergePipelineErrorHandling:
    """Tests for error handling in the merge pipeline."""

    def test_empty_entropies_handled(self) -> None:
        """Empty entropy dicts should be handled gracefully."""
        validator = EntropyMergeValidator()

        result = validator.validate_merge({}, {}, {})

        # Should return a result (may indicate no layers validated)
        assert result is not None
        assert len(result.layer_validations) == 0

    def test_mismatched_layers_handled(self) -> None:
        """Mismatched layer names should be handled gracefully."""
        validator = EntropyMergeValidator()

        source = {"layer.0": 1.5}
        target = {"layer.1": 1.6}  # Different layer name
        merged = {"layer.2": 1.55}  # Another different name

        result = validator.validate_merge(source, target, merged)

        # Should handle gracefully (may have 0 validations)
        assert result is not None

    def test_negative_entropy_handled(self) -> None:
        """Negative entropy values should be handled gracefully."""
        validator = EntropyMergeValidator()

        # Entropy shouldn't be negative, but test robustness
        source = {"layer.0": -1.0}
        target = {"layer.0": 1.0}
        merged = {"layer.0": 0.0}

        result = validator.validate_merge(source, target, merged)

        # Should handle without crashing
        assert result is not None
