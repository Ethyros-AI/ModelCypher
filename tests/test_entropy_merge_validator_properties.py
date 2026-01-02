# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Property-based tests for the entropy merge validator.

Uses Hypothesis to verify mathematical properties and invariants.
"""

from __future__ import annotations

from hypothesis import given, strategies as st, assume, settings

from modelcypher.core.domain.merging.entropy_merge_validator import (
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    ModelEntropyProfile,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _div_eps() -> float:
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


# =============================================================================
# Strategy Definitions
# =============================================================================


entropy_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# Layer Merge Validation Properties
# =============================================================================


class TestLayerMergeValidationProperties:
    """Property tests for layer merge validation."""

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_entropy_ratio_is_non_negative(
        self, source: float, target: float, merged: float
    ) -> None:
        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert validation.entropy_ratio >= 0.0

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_knowledge_retention_in_bounds(
        self, source: float, target: float, merged: float
    ) -> None:
        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert 0.0 <= validation.knowledge_retention_score <= 1.0

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_perfect_merge_has_unit_retention(
        self, source: float, target: float
    ) -> None:
        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=target,
        )

        assert abs(validation.knowledge_retention_score - 1.0) < _div_eps()

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_entropy_delta_is_absolute(
        self, source: float, target: float, merged: float
    ) -> None:
        expected_delta = abs(merged - target)

        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert abs(validation.entropy_delta - expected_delta) < _div_eps()


# =============================================================================
# Merge Entropy Validation Properties
# =============================================================================


class TestMergeEntropyValidationProperties:
    """Property tests for overall merge validation."""

    @given(
        layer_data=st.lists(
            st.tuples(entropy_strategy, entropy_strategy, entropy_strategy),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_max_ratio_gte_mean_ratio(
        self, layer_data: list[tuple[float, float, float]]
    ) -> None:
        layer_validations = {}
        for i, (source, target, merged) in enumerate(layer_data):
            layer_validations[f"layer_{i}"] = LayerMergeValidation.compute(
                layer_name=f"layer_{i}",
                source_entropy=source,
                target_entropy=target,
                merged_entropy=merged,
            )

        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations=layer_validations,
        )

        assert validation.max_entropy_ratio >= validation.mean_entropy_ratio

    @given(
        layer_data=st.lists(
            st.tuples(entropy_strategy, entropy_strategy, entropy_strategy),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_mean_retention_in_bounds(
        self, layer_data: list[tuple[float, float, float]]
    ) -> None:
        layer_validations = {}
        for i, (source, target, merged) in enumerate(layer_data):
            layer_validations[f"layer_{i}"] = LayerMergeValidation.compute(
                layer_name=f"layer_{i}",
                source_entropy=source,
                target_entropy=target,
                merged_entropy=merged,
            )

        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations=layer_validations,
        )

        assert 0.0 <= validation.mean_knowledge_retention <= 1.0

    def test_empty_validations_handles_gracefully(self) -> None:
        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations={},
        )

        assert validation.mean_entropy_ratio == 0.0
        assert validation.max_entropy_ratio == 0.0
        assert validation.mean_knowledge_retention == 1.0
        assert validation.entropy_ratio_std == 0.0


# =============================================================================
# Model Entropy Profile Properties
# =============================================================================


class TestModelEntropyProfileProperties:
    """Property tests for model entropy profiles."""

    @given(
        layer_count=st.integers(min_value=1, max_value=10),
        entropies=st.lists(
            entropy_strategy,
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_from_layer_profiles_computes_mean_correctly(
        self, layer_count: int, entropies: list[float]
    ) -> None:
        assume(len(entropies) >= layer_count)
        entropies = entropies[:layer_count]

        layer_profiles = {}
        for i, entropy in enumerate(entropies):
            layer_profiles[f"layer_{i}"] = LayerEntropyProfile(
                layer_name=f"layer_{i}",
                mean_entropy=entropy,
                entropy_variance=0.1,
            )

        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles=layer_profiles,
        )

        expected_mean = sum(entropies) / len(entropies)
        assert profile.mean_entropy == expected_mean

    @given(
        layer_count=st.integers(min_value=1, max_value=10),
        entropies=st.lists(
            entropy_strategy,
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_variance_is_non_negative(
        self, layer_count: int, entropies: list[float]
    ) -> None:
        assume(len(entropies) >= layer_count)
        entropies = entropies[:layer_count]

        layer_profiles = {}
        for i, entropy in enumerate(entropies):
            layer_profiles[f"layer_{i}"] = LayerEntropyProfile(
                layer_name=f"layer_{i}",
                mean_entropy=entropy,
                entropy_variance=0.1,
            )

        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles=layer_profiles,
        )

        assert profile.entropy_variance >= 0.0

    def test_empty_profiles_handles_gracefully(self) -> None:
        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles={},
        )

        assert profile.mean_entropy == 0.0
        assert profile.entropy_variance == 0.0
