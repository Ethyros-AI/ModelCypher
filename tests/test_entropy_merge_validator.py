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

"""Tests for EntropyMergeValidator."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeValidator,
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    ModelEntropyProfile,
)


def _epsilon(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values)))


def _create_test_profile(name: str, num_layers: int) -> ModelEntropyProfile:
    """Create a test ModelEntropyProfile with deterministic entropy values."""
    layer_profiles = {}
    for i in range(num_layers):
        entropy = 1.5 + i * 0.15
        variance = 0.1 + (i / num_layers) * 0.2
        layer_profiles[f"layers.{i}"] = LayerEntropyProfile(
            layer_name=f"layers.{i}",
            mean_entropy=entropy,
            entropy_variance=variance,
        )

    return ModelEntropyProfile.from_layer_profiles(name, layer_profiles)


class TestLayerEntropyProfile:
    """Tests for LayerEntropyProfile dataclass."""

    def test_profiles_hold_values(self) -> None:
        profile = LayerEntropyProfile(
            layer_name="layers.0",
            mean_entropy=1.0,
            entropy_variance=0.1,
        )

        assert profile.layer_name == "layers.0"
        assert profile.mean_entropy == 1.0
        assert profile.entropy_variance == 0.1


class TestModelEntropyProfile:
    """Tests for ModelEntropyProfile dataclass."""

    def test_from_layer_profiles_computes_statistics(self) -> None:
        layers = {
            "layers.0": LayerEntropyProfile(
                layer_name="layers.0",
                mean_entropy=1.0,
                entropy_variance=0.1,
            ),
            "layers.1": LayerEntropyProfile(
                layer_name="layers.1",
                mean_entropy=2.0,
                entropy_variance=0.2,
            ),
            "layers.2": LayerEntropyProfile(
                layer_name="layers.2",
                mean_entropy=3.0,
                entropy_variance=0.3,
            ),
        }

        profile = ModelEntropyProfile.from_layer_profiles("test_model", layers)

        assert profile.model_name == "test_model"
        assert abs(profile.mean_entropy - 2.0) < _epsilon(profile.mean_entropy, 2.0)
        assert abs(profile.entropy_variance - (2.0 / 3.0)) < _epsilon(
            profile.entropy_variance, 2.0 / 3.0
        )

    def test_empty_layers_returns_defaults(self) -> None:
        profile = ModelEntropyProfile.from_layer_profiles("empty", {})

        assert profile.mean_entropy == 0.0
        assert profile.entropy_variance == 0.0


class TestLayerMergeValidation:
    """Tests for LayerMergeValidation.

    Tests verify raw measurements, not classifications.
    entropy_ratio is the stability signal.
    """

    def test_perfect_merge_has_zero_ratio(self) -> None:
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.0,
        )

        assert validation.entropy_delta == 0.0
        assert validation.entropy_ratio == 0.0
        assert validation.knowledge_retention_score == 1.0

    def test_small_deviation_ratio_matches_expected(self) -> None:
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.5,
        )

        eps = _epsilon(2.0, 2.0, 2.5)
        expected_entropy = 2.0
        expected_delta = 0.5
        expected_ratio = expected_delta / (expected_entropy + eps)

        assert abs(validation.entropy_delta - expected_delta) < _epsilon(
            validation.entropy_delta, expected_delta
        )
        assert abs(validation.entropy_ratio - expected_ratio) < _epsilon(
            validation.entropy_ratio, expected_ratio
        )

    def test_ratio_ordering_reflects_deviation(self) -> None:
        stable = LayerMergeValidation.compute("l0", 2.0, 2.0, 2.0)
        moderate = LayerMergeValidation.compute("l1", 2.0, 2.0, 2.5)
        severe = LayerMergeValidation.compute("l2", 2.0, 2.0, 5.0)

        assert stable.entropy_ratio < moderate.entropy_ratio < severe.entropy_ratio


class TestMergeEntropyValidation:
    """Tests for MergeEntropyValidation.

    Tests verify raw aggregate measurements, not classifications.
    """

    def test_from_layer_validations_aggregates_metrics(self) -> None:
        layers = {
            "layers.0": LayerMergeValidation.compute("layers.0", 2.0, 2.0, 2.0),
            "layers.1": LayerMergeValidation.compute("layers.1", 2.0, 2.0, 2.05),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        mean_ratio = sum(v.entropy_ratio for v in layers.values()) / len(layers)
        max_ratio = max(v.entropy_ratio for v in layers.values())
        mean_retention = sum(v.knowledge_retention_score for v in layers.values()) / len(layers)

        assert abs(validation.mean_entropy_ratio - mean_ratio) < _epsilon(
            validation.mean_entropy_ratio, mean_ratio
        )
        assert abs(validation.max_entropy_ratio - max_ratio) < _epsilon(
            validation.max_entropy_ratio, max_ratio
        )
        assert abs(validation.mean_knowledge_retention - mean_retention) < _epsilon(
            validation.mean_knowledge_retention, mean_retention
        )

    def test_single_layer_aggregation(self) -> None:
        layer = LayerMergeValidation.compute("layers.0", 2.0, 2.0, 5.0)
        validation = MergeEntropyValidation.from_layer_validations(
            "source", "target", {"layers.0": layer}
        )

        assert abs(validation.mean_entropy_ratio - layer.entropy_ratio) < _epsilon(
            validation.mean_entropy_ratio, layer.entropy_ratio
        )
        assert abs(validation.max_entropy_ratio - layer.entropy_ratio) < _epsilon(
            validation.max_entropy_ratio, layer.entropy_ratio
        )
        assert abs(
            validation.mean_knowledge_retention - layer.knowledge_retention_score
        ) < _epsilon(validation.mean_knowledge_retention, layer.knowledge_retention_score)

    def test_layers_by_entropy_ratio(self) -> None:
        layers = {
            "best": LayerMergeValidation.compute("best", 2.0, 2.0, 2.0),
            "mid": LayerMergeValidation.compute("mid", 2.0, 2.0, 2.5),
            "worst": LayerMergeValidation.compute("worst", 2.0, 2.0, 4.0),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        worst_first = validation.layers_by_entropy_ratio(descending=True)
        assert worst_first[0] == "worst"
        assert worst_first[-1] == "best"

        best_first = validation.layers_by_entropy_ratio(descending=False)
        assert best_first[0] == "best"
        assert best_first[-1] == "worst"

    def test_summary_formatting(self) -> None:
        layers = {
            "layers.0": LayerMergeValidation.compute("layers.0", 2.0, 2.0, 2.0),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        assert "Mean entropy ratio" in validation.summary
        assert "Knowledge retention" in validation.summary


class TestEntropyMergeValidator:
    """Tests for EntropyMergeValidator."""

    @pytest.fixture
    def validator(self) -> EntropyMergeValidator:
        return EntropyMergeValidator()

    def test_create_layer_profile(self, validator: EntropyMergeValidator) -> None:
        entropies = [1.0, 1.1, 0.9, 1.0, 1.0]
        profile = validator.create_layer_profile("layers.0", entropies)

        assert profile.layer_name == "layers.0"
        assert abs(profile.mean_entropy - 1.0) < _epsilon(profile.mean_entropy, 1.0)

    def test_create_layer_profile_empty(self, validator: EntropyMergeValidator) -> None:
        profile = validator.create_layer_profile("layers.0", [])

        assert profile.mean_entropy == 0.0
        assert profile.entropy_variance == 0.0

    def test_create_test_profile_structure(self, validator: EntropyMergeValidator) -> None:
        profile = _create_test_profile("test_model", num_layers=10)

        assert profile.model_name == "test_model"
        assert len(profile.layer_profiles) == 10
        assert "layers.0" in profile.layer_profiles
        assert "layers.9" in profile.layer_profiles

    def test_validate_merge(self, validator: EntropyMergeValidator) -> None:
        source_entropies = {"layers.0": 2.0, "layers.1": 2.5}
        target_entropies = {"layers.0": 2.1, "layers.1": 2.4}
        merged_entropies = {"layers.0": 2.05, "layers.1": 2.45}

        validation = validator.validate_merge(
            source_entropies=source_entropies,
            target_entropies=target_entropies,
            merged_entropies=merged_entropies,
        )

        assert len(validation.layer_validations) == 2
        mean_ratio = sum(
            v.entropy_ratio for v in validation.layer_validations.values()
        ) / len(validation.layer_validations)
        assert abs(validation.mean_entropy_ratio - mean_ratio) < _epsilon(
            validation.mean_entropy_ratio, mean_ratio
        )

    def test_validate_merge_missing_layers(self, validator: EntropyMergeValidator) -> None:
        source_entropies = {"layers.0": 2.0, "layers.1": 2.5}
        target_entropies = {"layers.0": 2.1}
        merged_entropies = {"layers.0": 2.05, "layers.1": 2.45}

        validation = validator.validate_merge(
            source_entropies=source_entropies,
            target_entropies=target_entropies,
            merged_entropies=merged_entropies,
        )

        assert len(validation.layer_validations) == 1
        assert "layers.0" in validation.layer_validations
