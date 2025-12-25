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
Dimension Blender Unit Tests.

Tests for the DimensionBlender module with lazy import handling.
"""

from __future__ import annotations

import numpy as np
import pytest


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


class TestCorrelationWeights:
    """Test correlation-based dimension weighting."""

    def test_compute_dimension_correlations(self):
        """compute_dimension_correlations returns valid correlations."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_dimension_correlations,
        )

        config = _test_correlation_config()

        # Create source and target activations
        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = source.copy()  # Identical activations

        correlations = compute_dimension_correlations(source, target, config)

        # Identical activations should have high correlation
        assert correlations.mean_correlation > 0.9
        assert correlations.high_correlation_count == hidden_dim

    def test_compute_correlation_weights(self):
        """compute_correlation_weights returns weights in [0, 1]."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_weights,
            compute_dimension_correlations,
        )

        config = _test_correlation_config()

        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = np.random.randn(num_probes, hidden_dim).astype(np.float32)

        correlations = compute_dimension_correlations(source, target, config)
        weights = compute_correlation_weights(correlations, config)

        assert weights.shape == (hidden_dim,)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

    def test_compute_correlation_based_alpha(self):
        """compute_correlation_based_alpha returns valid alpha vector."""
        from modelcypher.core.domain.geometry.dimension_blender import (
            compute_correlation_based_alpha,
        )

        config = _test_correlation_config()

        hidden_dim = 10
        num_probes = 5
        source = np.random.randn(num_probes, hidden_dim).astype(np.float32)
        target = np.random.randn(num_probes, hidden_dim).astype(np.float32)

        alpha, correlations = compute_correlation_based_alpha(source, target, config, base_alpha=0.5)

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
            get_coder_to_instruct_affinity,
            get_instruct_to_coder_affinity,
        )

        i2c = get_instruct_to_coder_affinity()
        c2i = get_coder_to_instruct_affinity()

        # Values should be complementary
        for domain, val in i2c.items():
            if domain in c2i:
                assert i2c[domain] + c2i[domain] == pytest.approx(1.0)
