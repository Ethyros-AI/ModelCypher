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
Comprehensive tests for the DimensionBlender module.

Tests cover:
- DimensionDomainScores dataclass
- LayerDimensionProfile dataclass
- DimensionBlendConfig dataclass
- CorrelationWeightConfig dataclass
- DimensionCorrelations dataclass
- DimensionBlender class
- Affinity map functions
- Correlation-based weighting functions
- Blending functions
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend


def _test_correlation_config():
    """Create test CorrelationWeightConfig with explicit thresholds."""
    from modelcypher.core.domain.geometry.dimension_blender import (
        CorrelationWeightConfig,
    )

    return CorrelationWeightConfig.with_thresholds(
        min_correlation_for_default=0.8,
        correlation_scale=5.0,
        base_alpha=0.5,
        stability_alpha=0.7,
    )


def _get_atlas_domain():
    """Get AtlasDomain enum lazily."""
    from modelcypher.core.domain.agents.unified_atlas import AtlasDomain
    return AtlasDomain


# =============================================================================
# DimensionBlender Import Tests
# =============================================================================


class TestDimensionBlenderImport:
    """Test that DimensionBlender imports correctly with lazy dependencies."""

    def test_dimension_blender_has_compute_alpha_vector_method(self):
        """DimensionBlender class has required compute_alpha_vector method."""
        from modelcypher.core.domain.geometry import DimensionBlender

        assert hasattr(DimensionBlender, "compute_alpha_vector")
        assert callable(getattr(DimensionBlender, "compute_alpha_vector", None))

    def test_config_classes_have_required_fields(self):
        """Config dataclasses have expected fields with defaults."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
            DimensionBlendConfig,
        )

        # Verify configs can be instantiated with defaults
        blend_config = DimensionBlendConfig()
        assert hasattr(blend_config, "default_alpha")
        assert hasattr(blend_config, "activation_threshold")

        corr_config = CorrelationWeightConfig()
        assert hasattr(corr_config, "correlation_scale")
        assert hasattr(corr_config, "base_alpha")

    def test_lazy_getter_functions_return_dicts(self):
        """Lazy getter functions return dict mappings when called."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_balanced_affinity,
            get_coder_to_instruct_affinity,
            get_instruct_to_coder_affinity,
        )

        # Actually call the functions and verify return types
        i2c = get_instruct_to_coder_affinity()
        balanced = get_balanced_affinity()
        c2i = get_coder_to_instruct_affinity()

        assert isinstance(i2c, dict)
        assert isinstance(balanced, dict)
        assert isinstance(c2i, dict)


# =============================================================================
# DimensionDomainScores Tests
# =============================================================================


class TestDimensionDomainScores:
    """Tests for DimensionDomainScores dataclass."""

    def test_normalize_empty_scores(self):
        """Normalizing empty scores should do nothing."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
        )

        scores = DimensionDomainScores(dimension_index=0)
        scores.normalize()

        assert scores.dominant_domain is None
        assert scores.confidence == 0.0

    def test_normalize_single_domain(self):
        """Single domain should have confidence 1.0."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
        )

        AtlasDomain = _get_atlas_domain()
        scores = DimensionDomainScores(
            dimension_index=0,
            scores={AtlasDomain.COMPUTATIONAL: 5.0},
        )
        scores.normalize()

        assert scores.dominant_domain == AtlasDomain.COMPUTATIONAL
        assert scores.confidence == 1.0
        assert scores.scores[AtlasDomain.COMPUTATIONAL] == 1.0

    def test_normalize_multiple_domains(self):
        """Multiple domains should normalize to sum 1.0."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
        )

        AtlasDomain = _get_atlas_domain()
        scores = DimensionDomainScores(
            dimension_index=0,
            scores={
                AtlasDomain.COMPUTATIONAL: 6.0,
                AtlasDomain.LINGUISTIC: 4.0,
            },
        )
        scores.normalize()

        total = sum(scores.scores.values())
        assert abs(total - 1.0) < 1e-10

        assert scores.dominant_domain == AtlasDomain.COMPUTATIONAL
        assert abs(scores.confidence - 0.6) < 1e-10

    def test_total_activation_field(self):
        """total_activation field should be accessible."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
        )

        scores = DimensionDomainScores(
            dimension_index=5,
            total_activation=10.5,
        )

        assert scores.dimension_index == 5
        assert scores.total_activation == 10.5


# =============================================================================
# LayerDimensionProfile Tests
# =============================================================================


class TestLayerDimensionProfile:
    """Tests for LayerDimensionProfile dataclass."""

    def test_get_domain_distribution_empty(self):
        """Empty profile should return empty distribution."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            LayerDimensionProfile,
        )

        profile = LayerDimensionProfile(
            layer_index=0,
            dimension_count=10,
        )

        dist = profile.get_domain_distribution()
        assert dist == {}

    def test_get_domain_distribution_with_scores(self):
        """Should count dimensions by dominant domain."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
            LayerDimensionProfile,
        )

        AtlasDomain = _get_atlas_domain()

        profile = LayerDimensionProfile(
            layer_index=0,
            dimension_count=4,
        )

        # Create scores with dominant domains
        for i in range(4):
            scores = DimensionDomainScores(dimension_index=i)
            if i < 2:
                scores.dominant_domain = AtlasDomain.COMPUTATIONAL
            else:
                scores.dominant_domain = AtlasDomain.LINGUISTIC
            profile.dimension_scores[i] = scores

        dist = profile.get_domain_distribution()

        assert dist[AtlasDomain.COMPUTATIONAL] == 2
        assert dist[AtlasDomain.LINGUISTIC] == 2

    def test_get_domain_distribution_ignores_none(self):
        """Should ignore dimensions without dominant domain."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionDomainScores,
            LayerDimensionProfile,
        )

        AtlasDomain = _get_atlas_domain()

        profile = LayerDimensionProfile(
            layer_index=0,
            dimension_count=3,
        )

        # One with domain, two without
        scores1 = DimensionDomainScores(dimension_index=0)
        scores1.dominant_domain = AtlasDomain.COMPUTATIONAL
        profile.dimension_scores[0] = scores1

        scores2 = DimensionDomainScores(dimension_index=1)
        profile.dimension_scores[1] = scores2

        scores3 = DimensionDomainScores(dimension_index=2)
        profile.dimension_scores[2] = scores3

        dist = profile.get_domain_distribution()

        assert dist.get(AtlasDomain.COMPUTATIONAL) == 1
        assert len(dist) == 1


# =============================================================================
# DimensionBlendConfig Tests
# =============================================================================


class TestDimensionBlendConfig:
    """Tests for DimensionBlendConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        config = DimensionBlendConfig()

        assert config.default_alpha == 0.5
        assert config.activation_threshold is None
        assert config.confidence_threshold is None
        assert config.smoothing == 0.2
        assert config.domain_alpha_map == {}

    def test_from_activation_distribution(self):
        """Should derive thresholds from distribution."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        activation_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        confidence_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        config = DimensionBlendConfig.from_activation_distribution(
            activation_values=activation_values,
            confidence_values=confidence_values,
        )

        assert config.activation_threshold is not None
        assert config.confidence_threshold is not None
        assert min(activation_values) <= config.activation_threshold <= max(activation_values)
        assert min(confidence_values) <= config.confidence_threshold <= max(confidence_values)

    def test_from_activation_distribution_empty_raises(self):
        """Empty values should raise ValueError."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        with pytest.raises(ValueError):
            DimensionBlendConfig.from_activation_distribution(
                activation_values=[],
                confidence_values=[],
            )

    def test_with_thresholds(self):
        """Should create config with explicit thresholds."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        config = DimensionBlendConfig.with_thresholds(
            activation_threshold=0.1,
            confidence_threshold=0.5,
            default_alpha=0.6,
            smoothing=0.3,
        )

        assert config.activation_threshold == 0.1
        assert config.confidence_threshold == 0.5
        assert config.default_alpha == 0.6
        assert config.smoothing == 0.3

    def test_hashable(self):
        """Config should be hashable (frozen dataclass)."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        config = DimensionBlendConfig()
        # Should not raise
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_with_domain_alpha_map(self):
        """Should accept domain alpha map."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
        )

        AtlasDomain = _get_atlas_domain()
        domain_map = {
            AtlasDomain.COMPUTATIONAL: 0.3,
            AtlasDomain.LINGUISTIC: 0.7,
        }

        config = DimensionBlendConfig.with_thresholds(
            activation_threshold=0.1,
            confidence_threshold=0.5,
            domain_alpha_map=domain_map,
        )

        assert config.domain_alpha_map[AtlasDomain.COMPUTATIONAL] == 0.3
        assert config.domain_alpha_map[AtlasDomain.LINGUISTIC] == 0.7


# =============================================================================
# CorrelationWeightConfig Tests
# =============================================================================


class TestCorrelationWeightConfig:
    """Tests for CorrelationWeightConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
        )

        config = CorrelationWeightConfig()

        assert config.correlation_scale == 5.0
        assert config.base_alpha == 0.5
        assert config.stability_alpha == 0.7
        assert config.min_correlation_for_default is None

    def test_with_thresholds(self):
        """Should create config with explicit thresholds."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
        )

        config = CorrelationWeightConfig.with_thresholds(
            min_correlation_for_default=0.9,
            correlation_scale=10.0,
            base_alpha=0.4,
            stability_alpha=0.8,
        )

        assert config.min_correlation_for_default == 0.9
        assert config.correlation_scale == 10.0
        assert config.base_alpha == 0.4
        assert config.stability_alpha == 0.8

    def test_from_correlation_distribution(self):
        """Should derive threshold from distribution."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
        )

        correlation_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        config = CorrelationWeightConfig.from_correlation_distribution(
            correlation_values=correlation_values,
        )

        assert config.min_correlation_for_default is not None
        assert min(correlation_values) <= config.min_correlation_for_default <= max(
            correlation_values
        )

    def test_from_correlation_distribution_empty_raises(self):
        """Empty values should raise ValueError."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
        )

        with pytest.raises(ValueError):
            CorrelationWeightConfig.from_correlation_distribution(
                correlation_values=[],
            )

    def test_frozen_dataclass(self):
        """Config should be immutable."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            CorrelationWeightConfig,
        )

        config = CorrelationWeightConfig()

        with pytest.raises(Exception):
            config.correlation_scale = 10.0  # type: ignore


# =============================================================================
# Affinity Map Tests
# =============================================================================


class TestAffinityMaps:
    """Test lazy-loaded affinity maps."""

    def test_instruct_to_coder_affinity(self):
        """get_instruct_to_coder_affinity returns valid mapping."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_instruct_to_coder_affinity,
        )

        affinity = get_instruct_to_coder_affinity()

        # Should be a dict mapping AtlasDomain to float
        assert isinstance(affinity, dict)
        assert len(affinity) > 0

        # All values should be in [0, 1]
        for val in affinity.values():
            assert 0 <= val <= 1

    def test_balanced_affinity(self):
        """get_balanced_affinity returns valid mapping."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_balanced_affinity,
        )

        affinity = get_balanced_affinity()

        assert isinstance(affinity, dict)
        assert len(affinity) > 0

    def test_coder_to_instruct_affinity(self):
        """get_coder_to_instruct_affinity is inverse of instruct_to_coder."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_coder_to_instruct_affinity,
            get_instruct_to_coder_affinity,
        )

        i2c = get_instruct_to_coder_affinity()
        c2i = get_coder_to_instruct_affinity()

        # Values should be complementary
        for domain, val in i2c.items():
            if domain in c2i:
                assert i2c[domain] + c2i[domain] == pytest.approx(1.0)

    def test_affinity_maps_have_same_domains(self):
        """All affinity maps should cover the same domains."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_balanced_affinity,
            get_coder_to_instruct_affinity,
            get_instruct_to_coder_affinity,
        )

        i2c_domains = set(get_instruct_to_coder_affinity().keys())
        balanced_domains = set(get_balanced_affinity().keys())
        c2i_domains = set(get_coder_to_instruct_affinity().keys())

        assert i2c_domains == balanced_domains
        assert i2c_domains == c2i_domains

    def test_affinity_maps_cached(self):
        """Repeated calls should return same objects (lazy caching)."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_instruct_to_coder_affinity,
        )

        first = get_instruct_to_coder_affinity()
        second = get_instruct_to_coder_affinity()

        assert first is second


# =============================================================================
# Correlation Computation Tests
# =============================================================================


class TestCorrelationWeights:
    """Test correlation-based dimension weighting."""

    def test_compute_dimension_correlations(self):
        """compute_dimension_correlations returns valid correlations."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        # Create source and target activations using backend
        hidden_dim = 10
        num_probes = 5
        backend.random_seed(42)
        source = backend.random_normal((num_probes, hidden_dim))
        target = source  # Identical activations

        correlations = compute_dimension_correlations(source, target, config)

        # Identical activations should have high correlation
        assert correlations.mean_correlation > 0.9
        assert correlations.high_correlation_count == hidden_dim

    def test_compute_dimension_correlations_orthogonal(self):
        """Orthogonal activations should have low correlation."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        hidden_dim = 4
        num_probes = 4

        # Create orthogonal vectors for each dimension
        source = backend.eye(num_probes)
        # Shift to make orthogonal
        target = backend.zeros((num_probes, hidden_dim))
        for i in range(num_probes):
            target = backend.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ])

        correlations = compute_dimension_correlations(source, target, config)

        # Orthogonal should have low correlation
        assert correlations.mean_correlation < 0.5

    def test_compute_dimension_correlations_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        source = backend.random_normal((5, 10))
        target = backend.random_normal((5, 8))

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_dimension_correlations(source, target, config)

    def test_compute_correlation_weights(self):
        """compute_correlation_weights returns weights in [0, 1]."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_weights,
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        hidden_dim = 10
        num_probes = 5
        backend.random_seed(42)
        source = backend.random_normal((num_probes, hidden_dim))
        backend.random_seed(43)
        target = backend.random_normal((num_probes, hidden_dim))

        correlations = compute_dimension_correlations(source, target, config)
        weights = compute_correlation_weights(correlations, config)

        weights_np = backend.to_numpy(weights)

        assert weights_np.shape == (hidden_dim,)
        assert (weights_np >= 0).all()
        assert (weights_np <= 1).all()

    def test_compute_correlation_based_alpha(self):
        """compute_correlation_based_alpha returns valid alpha vector."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        hidden_dim = 10
        num_probes = 5
        backend.random_seed(42)
        source = backend.random_normal((num_probes, hidden_dim))
        backend.random_seed(43)
        target = backend.random_normal((num_probes, hidden_dim))

        alpha, correlations = compute_correlation_based_alpha(source, target, config, base_alpha=0.5)

        alpha_np = backend.to_numpy(alpha)
        assert alpha_np.shape == (hidden_dim,)
        assert (alpha_np >= 0).all()
        assert (alpha_np <= 1).all()

    def test_compute_correlation_based_alpha_identical(self):
        """Identical activations should result in base alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        hidden_dim = 10
        num_probes = 5
        backend.random_seed(42)
        source = backend.random_normal((num_probes, hidden_dim))
        target = source  # Identical

        alpha, correlations = compute_correlation_based_alpha(source, target, config, base_alpha=0.5)

        alpha_np = backend.to_numpy(alpha)
        # High correlation should result in values close to base_alpha
        mean_alpha = float(alpha_np.mean())
        assert 0.3 < mean_alpha < 0.7


# =============================================================================
# DimensionCorrelations Tests
# =============================================================================


class TestDimensionCorrelations:
    """Tests for DimensionCorrelations dataclass."""

    def test_agreement_ratio_all_high(self):
        """All high correlation should have ratio 1.0."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionCorrelations,
        )

        backend = get_default_backend()
        correlations = DimensionCorrelations(
            correlations=backend.array([0.9, 0.95, 0.85, 0.92]),
            mean_correlation=0.905,
            std_correlation=0.04,
            high_correlation_count=4,
            low_correlation_count=0,
        )

        assert correlations.agreement_ratio == 1.0

    def test_agreement_ratio_none_high(self):
        """No high correlation should have ratio 0.0."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionCorrelations,
        )

        backend = get_default_backend()
        correlations = DimensionCorrelations(
            correlations=backend.array([0.1, 0.2, 0.3, 0.4]),
            mean_correlation=0.25,
            std_correlation=0.1,
            high_correlation_count=0,
            low_correlation_count=4,
        )

        assert correlations.agreement_ratio == 0.0

    def test_agreement_ratio_mixed(self):
        """Mixed correlations should have ratio in (0, 1)."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionCorrelations,
        )

        backend = get_default_backend()
        correlations = DimensionCorrelations(
            correlations=backend.array([0.9, 0.1, 0.85, 0.2]),
            mean_correlation=0.5125,
            std_correlation=0.35,
            high_correlation_count=2,
            low_correlation_count=2,
        )

        assert correlations.agreement_ratio == 0.5


# =============================================================================
# Apply Correlation Weights Tests
# =============================================================================


class TestApplyCorrelationWeights:
    """Tests for apply_correlation_weights_to_alpha."""

    def test_zero_weights_use_base_alpha(self):
        """Zero weights should result in base alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            apply_correlation_weights_to_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        base_alpha = backend.array([0.5, 0.5, 0.5, 0.5])
        weights = backend.array([0.0, 0.0, 0.0, 0.0])

        result = apply_correlation_weights_to_alpha(base_alpha, weights, config)
        result_np = backend.to_numpy(result)

        # All weights 0 → result = base_alpha
        for val in result_np:
            assert abs(val - 0.5) < 1e-5

    def test_full_weights_use_stability_alpha(self):
        """Full weights should result in stability alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            apply_correlation_weights_to_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        base_alpha = backend.array([0.5, 0.5, 0.5, 0.5])
        weights = backend.array([1.0, 1.0, 1.0, 1.0])

        result = apply_correlation_weights_to_alpha(base_alpha, weights, config)
        result_np = backend.to_numpy(result)

        # All weights 1 → result = stability_alpha
        for val in result_np:
            assert abs(val - 0.7) < 1e-5

    def test_mixed_weights(self):
        """Mixed weights should interpolate."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            apply_correlation_weights_to_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        base_alpha = backend.array([0.5])
        weights = backend.array([0.5])

        result = apply_correlation_weights_to_alpha(base_alpha, weights, config)
        result_np = backend.to_numpy(result)

        # weight=0.5 → result = 0.5 * base + 0.5 * stability = 0.5*0.5 + 0.5*0.7 = 0.6
        assert abs(result_np[0] - 0.6) < 1e-5


# =============================================================================
# Blend Functions Tests
# =============================================================================


class TestBlendFunctions:
    """Tests for blending functions."""

    def test_blend_domain_and_correlation_alpha_pure_domain(self):
        """blend_ratio=0 should use pure domain alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            blend_domain_and_correlation_alpha,
        )

        backend = get_default_backend()

        domain_alpha = backend.array([0.3, 0.7, 0.5])
        correlation_alpha = backend.array([0.8, 0.2, 0.5])

        result = blend_domain_and_correlation_alpha(domain_alpha, correlation_alpha, blend_ratio=0.0)
        result_np = backend.to_numpy(result)

        # Pure domain alpha
        assert abs(result_np[0] - 0.3) < 1e-5
        assert abs(result_np[1] - 0.7) < 1e-5
        assert abs(result_np[2] - 0.5) < 1e-5

    def test_blend_domain_and_correlation_alpha_pure_correlation(self):
        """blend_ratio=1 should use pure correlation alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            blend_domain_and_correlation_alpha,
        )

        backend = get_default_backend()

        domain_alpha = backend.array([0.3, 0.7, 0.5])
        correlation_alpha = backend.array([0.8, 0.2, 0.5])

        result = blend_domain_and_correlation_alpha(domain_alpha, correlation_alpha, blend_ratio=1.0)
        result_np = backend.to_numpy(result)

        # Pure correlation alpha
        assert abs(result_np[0] - 0.8) < 1e-5
        assert abs(result_np[1] - 0.2) < 1e-5
        assert abs(result_np[2] - 0.5) < 1e-5

    def test_blend_domain_and_correlation_alpha_half(self):
        """blend_ratio=0.5 should average."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            blend_domain_and_correlation_alpha,
        )

        backend = get_default_backend()

        domain_alpha = backend.array([0.0, 1.0])
        correlation_alpha = backend.array([1.0, 0.0])

        result = blend_domain_and_correlation_alpha(domain_alpha, correlation_alpha, blend_ratio=0.5)
        result_np = backend.to_numpy(result)

        # 0.5 blend should average
        assert abs(result_np[0] - 0.5) < 1e-5
        assert abs(result_np[1] - 0.5) < 1e-5

    def test_blend_clamps_ratio(self):
        """blend_ratio should be clamped to [0, 1]."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            blend_domain_and_correlation_alpha,
        )

        backend = get_default_backend()

        domain_alpha = backend.array([0.3])
        correlation_alpha = backend.array([0.8])

        # Should clamp to 1.0
        result1 = blend_domain_and_correlation_alpha(domain_alpha, correlation_alpha, blend_ratio=2.0)
        result1_np = backend.to_numpy(result1)
        assert abs(result1_np[0] - 0.8) < 1e-5

        # Should clamp to 0.0
        result2 = blend_domain_and_correlation_alpha(domain_alpha, correlation_alpha, blend_ratio=-1.0)
        result2_np = backend.to_numpy(result2)
        assert abs(result2_np[0] - 0.3) < 1e-5


# =============================================================================
# Correlation Summary Tests
# =============================================================================


class TestCorrelationSummary:
    """Tests for correlation_summary function."""

    def test_correlation_summary_fields(self):
        """Summary should have expected fields."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionCorrelations,
            correlation_summary,
        )

        backend = get_default_backend()
        correlations = DimensionCorrelations(
            correlations=backend.array([0.9, 0.1, 0.5, 0.8]),
            mean_correlation=0.575,
            std_correlation=0.3,
            high_correlation_count=2,
            low_correlation_count=2,
        )

        summary = correlation_summary(correlations)

        assert "hidden_dim" in summary
        assert "mean_correlation" in summary
        assert "std_correlation" in summary
        assert "min_correlation" in summary
        assert "max_correlation" in summary
        assert "high_correlation_count" in summary
        assert "low_correlation_count" in summary
        assert "agreement_ratio" in summary

    def test_correlation_summary_values(self):
        """Summary values should match input."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionCorrelations,
            correlation_summary,
        )

        backend = get_default_backend()
        correlations = DimensionCorrelations(
            correlations=backend.array([0.1, 0.5, 0.9]),
            mean_correlation=0.5,
            std_correlation=0.33,
            high_correlation_count=1,
            low_correlation_count=2,
        )

        summary = correlation_summary(correlations)

        assert summary["hidden_dim"] == 3
        assert abs(summary["mean_correlation"] - 0.5) < 1e-5
        assert abs(summary["min_correlation"] - 0.1) < 1e-5
        assert abs(summary["max_correlation"] - 0.9) < 1e-5
        assert summary["high_correlation_count"] == 1
        assert summary["low_correlation_count"] == 2


# =============================================================================
# DimensionBlender Class Tests
# =============================================================================


class TestDimensionBlenderClass:
    """Tests for DimensionBlender class methods."""

    def test_build_probe_domain_map(self):
        """Should build correct mapping from probes."""
        from modelcypher.core.domain.agents.unified_atlas import AtlasDomain, AtlasProbe, AtlasSource
        from modelcypher.core.domain.geometry.dimension_blender import DimensionBlender

        probes = [
            AtlasProbe(
                id="p1",
                source=AtlasSource.SEQUENCE_INVARIANT,
                domain=AtlasDomain.COMPUTATIONAL,
                name="test1",
                description="Test probe 1",
                cross_domain_weight=1.0,
                category_name="test_category",
            ),
            AtlasProbe(
                id="p2",
                source=AtlasSource.SEMANTIC_PRIME,
                domain=AtlasDomain.LINGUISTIC,
                name="test2",
                description="Test probe 2",
                cross_domain_weight=1.0,
                category_name="test_category",
            ),
        ]

        # The build_probe_domain_map uses probe_id property which is f"{source.value}:{id}"
        probe_map = DimensionBlender.build_probe_domain_map(probes)

        assert probe_map["sequence_invariant:p1"] == AtlasDomain.COMPUTATIONAL
        assert probe_map["semantic_prime:p2"] == AtlasDomain.LINGUISTIC
        assert len(probe_map) == 2

    def test_compute_dimension_profiles_empty_fingerprints(self):
        """Empty fingerprints should create profiles with zero scores."""
        from modelcypher.core.domain.geometry.dimension_blender import DimensionBlender

        profiles = DimensionBlender.compute_dimension_profiles(
            fingerprints=[],
            probe_domain_map={},
            layer_indices=[0, 1],
            hidden_dim=4,
        )

        assert len(profiles) == 2
        assert profiles[0].dimension_count == 4
        assert profiles[1].dimension_count == 4

    def test_compute_alpha_vector_all_default(self):
        """Low activation should result in default alpha."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
            DimensionBlender,
            DimensionDomainScores,
            LayerDimensionProfile,
        )

        backend = get_default_backend()

        profile = LayerDimensionProfile(
            layer_index=0,
            dimension_count=4,
        )

        # Add low-activation scores
        for i in range(4):
            profile.dimension_scores[i] = DimensionDomainScores(
                dimension_index=i,
                total_activation=0.01,  # Below threshold
            )

        config = DimensionBlendConfig.with_thresholds(
            activation_threshold=0.1,
            confidence_threshold=0.3,
            default_alpha=0.5,
        )

        alpha = DimensionBlender.compute_alpha_vector(profile, config)
        alpha_np = backend.to_numpy(alpha)

        # All should be default alpha
        for val in alpha_np:
            assert abs(val - 0.5) < 1e-5

    def test_compute_alpha_vectors(self):
        """Should compute alpha vectors for multiple layers."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
            DimensionBlender,
            DimensionDomainScores,
            LayerDimensionProfile,
        )

        backend = get_default_backend()

        profiles = {}
        for layer_idx in [0, 1, 2]:
            profile = LayerDimensionProfile(
                layer_index=layer_idx,
                dimension_count=4,
            )
            for i in range(4):
                profile.dimension_scores[i] = DimensionDomainScores(dimension_index=i)
            profiles[layer_idx] = profile

        config = DimensionBlendConfig()
        alpha_vectors = DimensionBlender.compute_alpha_vectors(profiles, config)

        assert len(alpha_vectors) == 3
        assert 0 in alpha_vectors
        assert 1 in alpha_vectors
        assert 2 in alpha_vectors

        for layer_idx, alpha in alpha_vectors.items():
            assert alpha.shape == (4,)

    def test_summarize_profiles(self):
        """Should generate summary statistics."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlender,
            DimensionDomainScores,
            LayerDimensionProfile,
        )

        AtlasDomain = _get_atlas_domain()

        profiles = {}
        profile = LayerDimensionProfile(
            layer_index=0,
            dimension_count=4,
        )

        for i in range(4):
            scores = DimensionDomainScores(dimension_index=i)
            if i < 2:
                scores.dominant_domain = AtlasDomain.COMPUTATIONAL
            else:
                scores.dominant_domain = AtlasDomain.LINGUISTIC
            profile.dimension_scores[i] = scores

        profiles[0] = profile

        summary = DimensionBlender.summarize_profiles(profiles)

        assert summary["layer_count"] == 1
        assert 0 in summary["layers"]
        assert summary["layers"][0]["dimension_count"] == 4
        assert summary["layers"][0]["classified_count"] == 4
        # Domain distribution uses domain.value which is lowercase
        assert "computational" in summary["layers"][0]["domain_distribution"]
        assert "linguistic" in summary["layers"][0]["domain_distribution"]


# =============================================================================
# Sigmoid Function Tests
# =============================================================================


class TestSigmoid:
    """Tests for _sigmoid helper function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be 0.5."""
        from modelcypher.core.domain.geometry.dimension_blender import _sigmoid

        backend = get_default_backend()
        result = _sigmoid(backend.array([0.0]))
        val = float(backend.to_numpy(result)[0])

        assert abs(val - 0.5) < 1e-6

    def test_sigmoid_large_positive(self):
        """Large positive input should be close to 1."""
        from modelcypher.core.domain.geometry.dimension_blender import _sigmoid

        backend = get_default_backend()
        result = _sigmoid(backend.array([10.0]))
        val = float(backend.to_numpy(result)[0])

        assert val > 0.99

    def test_sigmoid_large_negative(self):
        """Large negative input should be close to 0."""
        from modelcypher.core.domain.geometry.dimension_blender import _sigmoid

        backend = get_default_backend()
        result = _sigmoid(backend.array([-10.0]))
        val = float(backend.to_numpy(result)[0])

        assert val < 0.01

    def test_sigmoid_numerical_stability(self):
        """Should handle extreme values without overflow."""
        from modelcypher.core.domain.geometry.dimension_blender import _sigmoid

        backend = get_default_backend()

        # Very large positive
        result1 = _sigmoid(backend.array([1000.0]))
        val1 = float(backend.to_numpy(result1)[0])
        assert not math.isnan(val1)
        assert not math.isinf(val1)

        # Very large negative
        result2 = _sigmoid(backend.array([-1000.0]))
        val2 = float(backend.to_numpy(result2)[0])
        assert not math.isnan(val2)
        assert not math.isinf(val2)

    def test_sigmoid_array(self):
        """Should work on arrays."""
        from modelcypher.core.domain.geometry.dimension_blender import _sigmoid

        backend = get_default_backend()
        result = _sigmoid(backend.array([-10.0, 0.0, 10.0]))
        vals = backend.to_numpy(result)

        assert vals[0] < 0.01
        assert abs(vals[1] - 0.5) < 1e-6
        assert vals[2] > 0.99


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and stress tests."""

    def test_single_dimension(self):
        """Should handle single dimension."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        source = backend.array([[1.0], [2.0], [3.0]])
        target = backend.array([[1.0], [2.0], [3.0]])

        alpha, correlations = compute_correlation_based_alpha(source, target, config)

        assert alpha.shape == (1,)
        assert correlations.correlations.shape == (1,)

    def test_single_probe(self):
        """Should handle single probe."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        source = backend.array([[1.0, 2.0, 3.0]])
        target = backend.array([[1.0, 2.0, 3.0]])

        alpha, correlations = compute_correlation_based_alpha(source, target, config)

        assert alpha.shape == (3,)
        assert correlations.correlations.shape == (3,)

    def test_zero_activations(self):
        """Zero activations should result in zero correlation."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        source = backend.zeros((5, 10))
        target = backend.zeros((5, 10))

        correlations = compute_dimension_correlations(source, target, config)

        # Zero vectors have undefined correlation → treated as 0
        assert correlations.mean_correlation == 0.0

    def test_large_dimension(self):
        """Should handle large dimensions."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        hidden_dim = 1024
        num_probes = 10
        backend.random_seed(42)
        source = backend.random_normal((num_probes, hidden_dim))
        backend.random_seed(43)
        target = backend.random_normal((num_probes, hidden_dim))

        alpha, correlations = compute_correlation_based_alpha(source, target, config)

        assert alpha.shape == (hidden_dim,)
        assert correlations.correlations.shape == (hidden_dim,)

    def test_negative_correlations(self):
        """Should handle negative correlations."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        backend = get_default_backend()
        config = _test_correlation_config()

        # Opposite vectors → negative correlation
        source = backend.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        target = backend.array([[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0]])

        correlations = compute_dimension_correlations(source, target, config)

        # Should be -1.0
        corr_np = backend.to_numpy(correlations.correlations)
        for c in corr_np:
            assert c < -0.9
