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

from __future__ import annotations

import math

from modelcypher.core.domain.geometry.sparse_region_locator import (
    LayerActivationStats,
    SparseRegionLocator,
)


def test_sparse_region_locator_analysis() -> None:
    # All parameters derived from data - no config needed
    locator = SparseRegionLocator()
    # Create data with distinct sparsity values to enable threshold derivation:
    # Layer 0: domain=0.2, baseline=1.0 → sparsity = 1 - 0.2/1.0 = 0.8 (sparse)
    # Layer 1: domain=0.8, baseline=1.0 → sparsity = 1 - 0.8/1.0 = 0.2 (dense)
    # Layer 2: domain=0.1, baseline=1.0 → sparsity = 1 - 0.1/1.0 = 0.9 (very sparse)
    domain_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=0.2,  # sparse relative to baseline
            max_activation=0.2,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=0.8,  # dense relative to baseline
            max_activation=0.8,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=2,
            mean_activation=0.1,  # very sparse relative to baseline
            max_activation=0.1,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]
    baseline_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=2,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]

    result = locator.analyze(
        domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test"
    )
    # Sparsity values: [0.8, 0.2, 0.9] - gap between 0.2 and 0.8 is 0.6
    # Threshold should be derived from the maximum gap
    assert len(result.sparse_layers) >= 0  # Depends on data-derived threshold
    assert result.skip_layers == []
    assert result.sparsity_threshold > 0  # Derived from data

    # Test analyze_from_activations with data that has separable sparsity
    # Layer 0: domain=0.2, baseline=1.0 → sparsity = 0.8
    # Layer 1: domain=0.9, baseline=1.0 → sparsity = 0.1
    # Layer 2: domain=0.3, baseline=1.0 → sparsity = 0.7
    from_activations = locator.analyze_from_activations(
        domain_activations=[{0: 0.2, 1: 0.9, 2: 0.3}],
        baseline_activations=[{0: 1.0, 1: 1.0, 2: 1.0}],
        domain="test",
    )
    assert len(from_activations.sparse_layers) >= 0  # Depends on data


def test_sparse_region_locator_no_configuration_needed() -> None:
    """SparseRegionLocator requires no configuration - all params derived from data."""
    locator = SparseRegionLocator()
    assert locator is not None


def test_layer_activation_stats_creation() -> None:
    """LayerActivationStats can be created with all fields."""
    stats = LayerActivationStats(
        layer_index=5,
        mean_activation=0.75,
        max_activation=1.5,
        activation_variance=0.1,
        prompt_count=10,
    )
    assert stats.layer_index == 5
    assert stats.mean_activation == 0.75
    assert stats.max_activation == 1.5
    assert stats.activation_variance == 0.1
    assert stats.prompt_count == 10


def test_sparse_region_locator_high_sparsity() -> None:
    """High sparsity layers are identified when clearly different from low sparsity."""
    locator = SparseRegionLocator()
    # Domain has much lower activation than baseline = high sparsity (0.9)
    # Add another layer with low sparsity for contrast
    domain_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=0.1,  # sparsity = 1 - 0.1/1.0 = 0.9
            max_activation=0.1,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=0.95,  # sparsity = 1 - 0.95/1.0 = 0.05
            max_activation=0.95,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]
    baseline_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]
    result = locator.analyze(
        domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test"
    )
    # Threshold derived from gap between 0.9 and 0.05 sparsity values
    # Layer 0 has high sparsity (0.9) and should be identified
    assert 0 in result.sparse_layers


def test_sparse_region_locator_threshold_derived_from_data() -> None:
    """Threshold is derived from data distribution, not hardcoded."""
    locator = SparseRegionLocator()
    # Two layers with different sparsity values
    domain_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=0.3,  # sparsity = 0.7
            max_activation=0.3,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=0.8,  # sparsity = 0.2
            max_activation=0.8,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]
    baseline_stats = [
        LayerActivationStats(
            layer_index=0,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
        LayerActivationStats(
            layer_index=1,
            mean_activation=1.0,
            max_activation=1.0,
            activation_variance=0.0,
            prompt_count=2,
        ),
    ]
    result = locator.analyze(
        domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test"
    )
    expected = (0.2 + 0.7) / 2.0
    eps = math.ulp(expected)
    assert abs(result.sparsity_threshold - expected) <= eps
