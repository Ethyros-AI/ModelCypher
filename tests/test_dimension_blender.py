"""
Dimension Blender Unit Tests.

Tests for the DimensionBlender module with lazy import handling.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestDimensionBlenderImport:
    """Test that DimensionBlender imports correctly with lazy dependencies."""

    def test_import_dimension_blender_from_geometry(self):
        """DimensionBlender can be imported from geometry package."""
        from modelcypher.core.domain.geometry import DimensionBlender
        assert DimensionBlender is not None

    def test_import_dimension_blender_directly(self):
        """DimensionBlender can be imported directly from module."""
        from modelcypher.core.domain.geometry.dimension_blender import DimensionBlender
        assert DimensionBlender is not None

    def test_import_config_classes(self):
        """Config dataclasses are available."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            DimensionBlendConfig,
            CorrelationWeightConfig,
        )
        assert DimensionBlendConfig is not None
        assert CorrelationWeightConfig is not None

    def test_import_lazy_getter_functions(self):
        """Lazy getter functions are available."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            get_instruct_to_coder_affinity,
            get_balanced_affinity,
            get_coder_to_instruct_affinity,
        )
        # These should work when called (triggers lazy import)
        assert callable(get_instruct_to_coder_affinity)
        assert callable(get_balanced_affinity)
        assert callable(get_coder_to_instruct_affinity)


class TestCorrelationWeights:
    """Test correlation-based dimension weighting."""

    def test_compute_dimension_correlations(self):
        """compute_dimension_correlations returns valid correlations."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )
        
        # Create source and target activations
        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = source.copy()  # Identical activations
        
        correlations = compute_dimension_correlations(source, target)
        
        # Identical activations should have high correlation
        assert correlations.mean_correlation > 0.9
        assert correlations.high_correlation_count == hidden_dim

    def test_compute_correlation_weights(self):
        """compute_correlation_weights returns weights in [0, 1]."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
            compute_correlation_weights,
        )
        
        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        
        correlations = compute_dimension_correlations(source, target)
        weights = compute_correlation_weights(correlations)
        
        assert weights.shape == (hidden_dim,)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

    def test_compute_correlation_based_alpha(self):
        """compute_correlation_based_alpha returns valid alpha vector."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )
        
        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        
        alpha, correlations = compute_correlation_based_alpha(
            source, target, base_alpha=0.5
        )
        
        assert alpha.shape == (hidden_dim,)
        assert np.all(alpha >= 0)
        assert np.all(alpha <= 1)


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
            get_instruct_to_coder_affinity,
            get_coder_to_instruct_affinity,
        )
        
        i2c = get_instruct_to_coder_affinity()
        c2i = get_coder_to_instruct_affinity()
        
        # Values should be complementary
        for domain, val in i2c.items():
            if domain in c2i:
                assert i2c[domain] + c2i[domain] == pytest.approx(1.0)
