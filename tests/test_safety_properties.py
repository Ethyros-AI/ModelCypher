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

"""Property-based tests for safety module."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.safety.delta_feature_set import DeltaFeatureSet


# Strategies for generating valid feature sets
@st.composite
def geodesic_spreads_tuple(draw, min_size=0, max_size=50):
    """Generate a tuple of non-negative geodesic spreads."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return tuple(
        draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        for _ in range(size)
    )


@st.composite
def sparsity_tuple(draw, size):
    """Generate a tuple of sparsity values in [0, 1]."""
    return tuple(
        draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for _ in range(size)
    )


@st.composite
def delta_feature_set(draw):
    """Generate a valid DeltaFeatureSet."""
    geodesic_spreads = draw(geodesic_spreads_tuple())
    size = len(geodesic_spreads)
    sparsity = draw(sparsity_tuple(size)) if size > 0 else ()

    # Outlier layer indices must be valid indices
    if size > 0:
        num_outlier = draw(st.integers(min_value=0, max_value=size))
        outlier_indices = tuple(
            sorted(draw(st.sampled_from(range(size))) for _ in range(num_outlier))
        )
    else:
        outlier_indices = ()

    return DeltaFeatureSet(
        geodesic_spreads=geodesic_spreads,
        sparsity=sparsity,
        outlier_layer_indices=outlier_indices,
    )


class TestDeltaFeatureSetProperties:
    """Property-based tests for DeltaFeatureSet."""

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_layer_count_equals_geodesic_spreads_length(self, features: DeltaFeatureSet):
        """layer_count should equal len(geodesic_spreads)."""
        assert features.layer_count == len(features.geodesic_spreads)

    @given(geodesic_spreads_tuple(min_size=1))
    @settings(max_examples=100)
    def test_mean_geodesic_spread_bounded_by_min_max(self, geodesic_spreads: tuple[float, ...]):
        """mean_geodesic_spread should be between min and max of geodesic_spreads."""
        features = DeltaFeatureSet(geodesic_spreads=geodesic_spreads)
        mean = features.mean_geodesic_spread
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array(list(geodesic_spreads)))
        assert min(geodesic_spreads) - eps <= mean <= max(geodesic_spreads) + eps

    @given(geodesic_spreads_tuple(min_size=1))
    @settings(max_examples=100)
    def test_max_geodesic_spread_is_maximum(self, geodesic_spreads: tuple[float, ...]):
        """max_geodesic_spread should equal max(geodesic_spreads)."""
        features = DeltaFeatureSet(geodesic_spreads=geodesic_spreads)
        assert features.max_geodesic_spread == max(geodesic_spreads)

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_outlier_layer_fraction_bounded(self, features: DeltaFeatureSet):
        """outlier_layer_fraction should be in [0, 1]."""
        fraction = features.outlier_layer_fraction
        assert 0.0 <= fraction <= 1.0

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_serialization_roundtrip(self, features: DeltaFeatureSet):
        """to_dict/from_dict should be a perfect round-trip."""
        restored = DeltaFeatureSet.from_dict(features.to_dict())
        assert restored.geodesic_spreads == features.geodesic_spreads
        assert restored.sparsity == features.sparsity
        assert restored.outlier_layer_indices == features.outlier_layer_indices

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_mean_sparsity_bounded(self, sparsity_list: list[float]):
        """mean_sparsity should be in [0, 1] when sparsity values are in [0, 1]."""
        features = DeltaFeatureSet(sparsity=tuple(sparsity_list))
        mean = features.mean_sparsity
        assert 0.0 <= mean <= 1.0

    def test_empty_feature_set_defaults(self):
        """Empty feature set should have safe default values."""
        features = DeltaFeatureSet()
        assert features.layer_count == 0
        assert features.mean_geodesic_spread == 0.0
        assert features.max_geodesic_spread == 0.0
        assert features.mean_sparsity == 0.0
        assert features.outlier_layer_fraction == 0.0
        assert not features.has_outlier_layers
