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

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.safety.delta_feature_set import DeltaFeatureSet


# Strategies for generating valid feature sets
@st.composite
def l2_norms_tuple(draw, min_size=0, max_size=50):
    """Generate a tuple of non-negative L2 norms."""
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
    l2_norms = draw(l2_norms_tuple())
    size = len(l2_norms)
    sparsity = draw(sparsity_tuple(size)) if size > 0 else ()

    # Suspect layer indices must be valid indices
    if size > 0:
        num_suspect = draw(st.integers(min_value=0, max_value=size))
        suspect_indices = tuple(sorted(draw(st.sampled_from(range(size))) for _ in range(num_suspect)))
    else:
        suspect_indices = ()

    return DeltaFeatureSet(
        l2_norms=l2_norms,
        sparsity=sparsity,
        suspect_layer_indices=suspect_indices,
    )


class TestDeltaFeatureSetProperties:
    """Property-based tests for DeltaFeatureSet."""

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_layer_count_equals_l2_norms_length(self, features: DeltaFeatureSet):
        """layer_count should equal len(l2_norms)."""
        assert features.layer_count == len(features.l2_norms)

    @given(l2_norms_tuple(min_size=1))
    @settings(max_examples=100)
    def test_mean_l2_norm_bounded_by_min_max(self, l2_norms: tuple[float, ...]):
        """mean_l2_norm should be between min and max of l2_norms."""
        features = DeltaFeatureSet(l2_norms=l2_norms)
        mean = features.mean_l2_norm
        assert min(l2_norms) <= mean <= max(l2_norms)

    @given(l2_norms_tuple(min_size=1))
    @settings(max_examples=100)
    def test_max_l2_norm_is_maximum(self, l2_norms: tuple[float, ...]):
        """max_l2_norm should equal max(l2_norms)."""
        features = DeltaFeatureSet(l2_norms=l2_norms)
        assert features.max_l2_norm == max(l2_norms)

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_suspect_layer_fraction_bounded(self, features: DeltaFeatureSet):
        """suspect_layer_fraction should be in [0, 1]."""
        fraction = features.suspect_layer_fraction
        assert 0.0 <= fraction <= 1.0

    @given(delta_feature_set())
    @settings(max_examples=100)
    def test_serialization_roundtrip(self, features: DeltaFeatureSet):
        """to_dict/from_dict should be a perfect round-trip."""
        restored = DeltaFeatureSet.from_dict(features.to_dict())
        assert restored.l2_norms == features.l2_norms
        assert restored.sparsity == features.sparsity
        assert restored.suspect_layer_indices == features.suspect_layer_indices

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
        assert features.mean_l2_norm == 0.0
        assert features.max_l2_norm == 0.0
        assert features.mean_sparsity == 0.0
        assert features.suspect_layer_fraction == 0.0
        assert not features.has_suspect_layers
