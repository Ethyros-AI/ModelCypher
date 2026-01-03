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

"""Tests for ProfileComparison - geometric comparison between two models."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.model_profile import (
    DensitySummary,
    LayerProfile,
    ModelProfile,
    SemanticSignature,
    TopologySummary,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.profile_comparison import (
    LayerComparison,
    ProfileComparison,
    compare_profiles,
)


def _eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestLayerComparison:
    """Tests for LayerComparison dataclass."""

    def test_create_minimal(self) -> None:
        """Should create LayerComparison with minimal fields."""
        lc = LayerComparison(
            source_layer_idx=0,
            target_layer_idx=0,
        )
        assert lc.source_layer_idx == 0
        assert lc.target_layer_idx == 0
        assert lc.sectional_curvature_diff == 0.0
        assert lc.curvature_sign_match

    def test_create_full(self) -> None:
        """Should create LayerComparison with all fields."""
        lc = LayerComparison(
            source_layer_idx=5,
            target_layer_idx=6,
            sectional_curvature_diff=-0.05,
            ollivier_ricci_diff=-0.08,
            intrinsic_dimension_diff=10.0,
            dimension_ratio=1.2,
            shannon_entropy_diff=0.5,
            betti_0_diff=0,
            betti_1_diff=1,
            curvature_sign_match=False,
        )
        assert lc.source_layer_idx == 5
        assert lc.target_layer_idx == 6
        assert lc.dimension_ratio == 1.2
        assert not lc.curvature_sign_match

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip LayerComparison."""
        original = LayerComparison(
            source_layer_idx=3,
            target_layer_idx=4,
            sectional_curvature_diff=-0.03,
            ollivier_ricci_diff=-0.05,
            intrinsic_dimension_diff=5.0,
            dimension_ratio=1.1,
        )
        d = original.to_dict()
        restored = LayerComparison.from_dict(d)

        assert restored.source_layer_idx == original.source_layer_idx
        assert restored.target_layer_idx == original.target_layer_idx
        assert restored.sectional_curvature_diff == original.sectional_curvature_diff


class TestProfileComparison:
    """Tests for ProfileComparison dataclass."""

    def test_create_minimal(self) -> None:
        """Should create ProfileComparison with minimal fields."""
        pc = ProfileComparison(
            source_path="/path/to/source",
            target_path="/path/to/target",
        )
        assert pc.source_path == "/path/to/source"
        assert pc.target_path == "/path/to/target"
        assert pc.architecture_match is False
        assert pc.mean_sectional_curvature_diff == 0.0
        assert pc.aligned is False

    def test_create_full(self) -> None:
        """Should create ProfileComparison with all fields."""
        pc = ProfileComparison(
            source_path="/path/to/source",
            target_path="/path/to/target",
            architecture_match=True,
            hidden_dim_ratio=1.0,
            layer_count_ratio=1.0,
            mean_sectional_curvature_diff=0.01,
            mean_ollivier_ricci_diff=0.02,
            mean_intrinsic_dimension_diff=0.03,
            topology_betti_diff=1,
            topology_persistence_diff=0.1,
            semantic_cosine_similarity=0.88,
            layer_mapping={0: 0, 1: 1, 2: 2},
            layer_comparisons=[
                LayerComparison(0, 0),
                LayerComparison(1, 1),
            ],
            aligned=False,
        )
        assert pc.architecture_match
        assert pc.mean_sectional_curvature_diff == 0.01
        assert len(pc.layer_comparisons) == 2

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip ProfileComparison."""
        original = ProfileComparison(
            source_path="/path/to/source",
            target_path="/path/to/target",
            architecture_match=True,
            mean_sectional_curvature_diff=0.01,
            layer_mapping={0: 0, 1: 1},
            layer_comparisons=[
                LayerComparison(0, 0),
            ],
        )
        d = original.to_dict()
        restored = ProfileComparison.from_dict(d)

        assert restored.source_path == original.source_path
        assert restored.architecture_match == original.architecture_match
        assert restored.mean_sectional_curvature_diff == original.mean_sectional_curvature_diff
        assert len(restored.layer_comparisons) == 1


class TestCompareProfiles:
    """Tests for compare_profiles function."""

    def test_compare_identical_profiles(self) -> None:
        """Identical profiles should have near-zero diffs."""
        profile = ModelProfile(
            model_path="/path/to/model",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=32,
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.1,
                    ollivier_ricci_mean=-0.15,
                    intrinsic_dimension=50.0,
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        comparison = compare_profiles(profile, profile)

        eps = _eps()
        assert comparison.architecture_match
        assert comparison.hidden_dim_ratio == 1.0
        assert comparison.layer_count_ratio == 1.0
        # Identical profiles should have near-zero diffs
        assert abs(comparison.mean_sectional_curvature_diff) <= eps
        assert abs(comparison.mean_ollivier_ricci_diff) <= eps
        assert abs(comparison.mean_intrinsic_dimension_diff) <= eps
        assert comparison.aligned is True

    def test_compare_same_architecture_different_curvature(self) -> None:
        """Same architecture with different curvature should still be compatible."""
        source = ModelProfile(
            model_path="/path/to/source",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=32,
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.1,
                    ollivier_ricci_mean=-0.15,
                    intrinsic_dimension=50.0,
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=32,
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.2,  # Different curvature
                    ollivier_ricci_mean=-0.25,
                    intrinsic_dimension=55.0,
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        comparison = compare_profiles(source, target)
        identical = compare_profiles(source, source)

        assert comparison.architecture_match
        assert comparison.hidden_dim_ratio == 1.0
        assert comparison.mean_sectional_curvature_diff >= identical.mean_sectional_curvature_diff
        assert comparison.mean_ollivier_ricci_diff >= identical.mean_ollivier_ricci_diff
        assert comparison.aligned is False

    def test_compare_different_architectures(self) -> None:
        """Different architectures should be flagged."""
        source = ModelProfile(
            model_path="/path/to/qwen",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=32,
        )

        target = ModelProfile(
            model_path="/path/to/llama",
            model_family="llama",
            architecture="llama",
            hidden_dim=4096,
            num_layers=32,
        )

        comparison = compare_profiles(source, target)

        assert not comparison.architecture_match
        assert comparison.hidden_dim_ratio == 2.0

    def test_compare_different_layer_counts(self) -> None:
        """Different layer counts should be reflected in ratio."""
        source = ModelProfile(
            model_path="/path/to/small",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=16,
            layer_profiles=[
                LayerProfile(layer_idx=i, sectional_curvature_mean=-0.1)
                for i in range(4)
            ],
        )

        target = ModelProfile(
            model_path="/path/to/large",
            model_family="qwen",
            architecture="qwen2",
            hidden_dim=2048,
            num_layers=32,
            layer_profiles=[
                LayerProfile(layer_idx=i, sectional_curvature_mean=-0.1)
                for i in range(8)
            ],
        )

        comparison = compare_profiles(source, target)

        assert comparison.layer_count_ratio == 2.0
        # Layer mapping should still be computed
        assert len(comparison.layer_mapping) > 0

    def test_compare_with_topology_summaries(self) -> None:
        """Should compute topology diffs when available."""
        source = ModelProfile(
            model_path="/path/to/source",
            topology_summary=TopologySummary(
                component_count=1,
                cycle_count=3,
                max_persistence=0.5,
            ),
        )

        target = ModelProfile(
            model_path="/path/to/target",
            topology_summary=TopologySummary(
                component_count=1,
                cycle_count=4,  # One more cycle
                max_persistence=0.6,
            ),
        )

        comparison = compare_profiles(source, target)

        assert comparison.topology_betti_diff == 1
        assert comparison.topology_persistence_diff == 0.1

    def test_compare_with_semantic_signatures(self) -> None:
        """Should compute semantic cosine when available."""
        source = ModelProfile(
            model_path="/path/to/source",
            semantic_signature=SemanticSignature(
                vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            ),
        )

        target = ModelProfile(
            model_path="/path/to/target",
            semantic_signature=SemanticSignature(
                vector=[0.8, 0.6, 0.0, 0.0, 0.0],  # Similar but not identical
            ),
        )

        comparison = compare_profiles(source, target)

        assert comparison.semantic_cosine_similarity is not None
        expected = 0.8 / sqrt_scalar(0.8**2 + 0.6**2, get_default_backend())
        assert abs(comparison.semantic_cosine_similarity - expected) <= _eps()

    def test_compare_orthogonal_semantic_signatures(self) -> None:
        """Orthogonal semantic signatures should have zero cosine."""
        source = ModelProfile(
            model_path="/path/to/source",
            semantic_signature=SemanticSignature(
                vector=[1.0, 0.0, 0.0],
            ),
        )

        target = ModelProfile(
            model_path="/path/to/target",
            semantic_signature=SemanticSignature(
                vector=[0.0, 1.0, 0.0],  # Orthogonal
            ),
        )

        comparison = compare_profiles(source, target)

        assert comparison.semantic_cosine_similarity is not None
        assert abs(comparison.semantic_cosine_similarity) <= _eps()

    def test_layer_mapping_across_different_counts(self) -> None:
        """Should map layers by relative position."""
        source = ModelProfile(
            model_path="/path/to/source",
            num_layers=4,
            layer_profiles=[
                LayerProfile(layer_idx=0),
                LayerProfile(layer_idx=1),
                LayerProfile(layer_idx=2),
                LayerProfile(layer_idx=3),
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            num_layers=8,
            layer_profiles=[
                LayerProfile(layer_idx=i) for i in range(8)
            ],
        )

        comparison = compare_profiles(source, target)

        # Source layer 0 (position 0.0) -> Target layer 0
        assert comparison.layer_mapping[0] == 0
        # Source layer 3 (position 1.0) -> Target layer 7
        assert comparison.layer_mapping[3] == 7


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_compare_empty_profiles(self) -> None:
        """Should handle profiles with no layer data."""
        source = ModelProfile(model_path="/path/to/source")
        target = ModelProfile(model_path="/path/to/target")

        comparison = compare_profiles(source, target)

        assert comparison.layer_mapping == {}
        assert comparison.layer_comparisons == []
        assert comparison.aligned is False

    def test_compare_one_empty_profile(self) -> None:
        """Should handle case where one profile has no layers."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(layer_idx=0, sectional_curvature_mean=-0.1),
            ],
        )
        target = ModelProfile(model_path="/path/to/target")

        comparison = compare_profiles(source, target)

        # Should compute what it can
        assert comparison.source_path == "/path/to/source"
        assert comparison.target_path == "/path/to/target"

    def test_compare_with_zero_hidden_dim(self) -> None:
        """Should handle zero hidden_dim gracefully."""
        source = ModelProfile(
            model_path="/path/to/source",
            hidden_dim=0,
        )
        target = ModelProfile(
            model_path="/path/to/target",
            hidden_dim=2048,
        )

        comparison = compare_profiles(source, target)

        # Should default to 1.0 ratio when source is zero
        assert comparison.hidden_dim_ratio == 1.0

    def test_compare_with_mismatched_semantic_vector_lengths(self) -> None:
        """Should handle mismatched semantic vector lengths."""
        source = ModelProfile(
            model_path="/path/to/source",
            semantic_signature=SemanticSignature(
                vector=[1.0, 0.0, 0.0],
            ),
        )
        target = ModelProfile(
            model_path="/path/to/target",
            semantic_signature=SemanticSignature(
                vector=[1.0, 0.0],  # Different length!
            ),
        )

        comparison = compare_profiles(source, target)

        # Should not compute semantic cosine for mismatched lengths
        assert comparison.semantic_cosine_similarity is None

    def test_compare_with_empty_semantic_vectors(self) -> None:
        """Should handle empty semantic vectors."""
        source = ModelProfile(
            model_path="/path/to/source",
            semantic_signature=SemanticSignature(vector=[]),
        )
        target = ModelProfile(
            model_path="/path/to/target",
            semantic_signature=SemanticSignature(vector=[]),
        )

        comparison = compare_profiles(source, target)

        assert comparison.semantic_cosine_similarity is None

    def test_compare_with_zero_norm_semantic_vectors(self) -> None:
        """Should handle zero-norm semantic vectors."""
        source = ModelProfile(
            model_path="/path/to/source",
            semantic_signature=SemanticSignature(
                vector=[0.0, 0.0, 0.0],
            ),
        )
        target = ModelProfile(
            model_path="/path/to/target",
            semantic_signature=SemanticSignature(
                vector=[1.0, 0.0, 0.0],
            ),
        )

        comparison = compare_profiles(source, target)

        # Should not compute for zero-norm vectors
        assert comparison.semantic_cosine_similarity is None


class TestBaselineZScores:
    """Tests for baseline-relative z-score computation."""

    def test_compare_without_baseline(self) -> None:
        """Should not compute z-scores without baseline."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(layer_idx=0, ollivier_ricci_mean=0.1),
            ],
        )
        target = ModelProfile(
            model_path="/path/to/target",
            layer_profiles=[
                LayerProfile(layer_idx=0, ollivier_ricci_mean=0.2),
            ],
        )

        comparison = compare_profiles(source, target, baseline=None)

        assert comparison.sectional_z_score is None
        assert comparison.ricci_z_score is None
        assert comparison.dimension_z_score is None
        assert comparison.baseline_family is None
        assert comparison.baseline_model_count is None

    def test_compare_with_baseline(self) -> None:
        """Should compute z-scores with baseline."""
        from modelcypher.core.domain.geometry.curvature_profile import FamilyBaseline

        source = ModelProfile(
            model_path="/path/to/source",
            num_layers=3,
            layer_profiles=[
                LayerProfile(layer_idx=0, ollivier_ricci_mean=0.1, intrinsic_dimension=10.0),
                LayerProfile(layer_idx=1, ollivier_ricci_mean=0.15, intrinsic_dimension=8.0),
                LayerProfile(layer_idx=2, ollivier_ricci_mean=0.2, intrinsic_dimension=6.0),
            ],
        )
        target = ModelProfile(
            model_path="/path/to/target",
            num_layers=3,
            layer_profiles=[
                LayerProfile(layer_idx=0, ollivier_ricci_mean=0.2, intrinsic_dimension=12.0),
                LayerProfile(layer_idx=1, ollivier_ricci_mean=0.25, intrinsic_dimension=10.0),
                LayerProfile(layer_idx=2, ollivier_ricci_mean=0.3, intrinsic_dimension=8.0),
            ],
        )

        baseline = FamilyBaseline(
            family="test",
            layer_positions=[0.0, 0.5, 1.0],
            sectional_mean_by_position=[0.0, 0.0, 0.0],
            sectional_std_by_position=[0.1, 0.1, 0.1],  # Non-zero std
            ollivier_ricci_mean_by_position=[0.15, 0.2, 0.25],
            ollivier_ricci_std_by_position=[0.05, 0.05, 0.05],  # Non-zero std
            intrinsic_dimension_by_position=[10.0, 8.0, 6.0],
            contributing_models=["model1", "model2"],
            sample_count=2,
        )

        comparison = compare_profiles(source, target, baseline=baseline)

        eps = _eps()
        assert comparison.baseline_family == "test"
        assert comparison.baseline_model_count == 2
        assert comparison.ricci_z_score is not None
        assert comparison.ricci_z_score > eps
        assert comparison.dimension_z_score is not None
        assert comparison.dimension_z_score > eps

    def test_compare_identical_with_baseline(self) -> None:
        """Should give z-score of 0 for identical profiles."""
        from modelcypher.core.domain.geometry.curvature_profile import FamilyBaseline

        source = ModelProfile(
            model_path="/path/to/model",
            num_layers=2,
            layer_profiles=[
                LayerProfile(layer_idx=0, ollivier_ricci_mean=0.1, intrinsic_dimension=10.0),
                LayerProfile(layer_idx=1, ollivier_ricci_mean=0.2, intrinsic_dimension=8.0),
            ],
        )

        baseline = FamilyBaseline(
            family="test",
            layer_positions=[0.0, 1.0],
            ollivier_ricci_std_by_position=[0.05, 0.05],
            intrinsic_dimension_by_position=[10.0, 8.0],
            sample_count=2,
        )

        comparison = compare_profiles(source, source, baseline=baseline)

        eps = _eps()
        # Identical profiles should have z-score of 0
        assert abs(comparison.ricci_z_score) <= eps
        assert abs(comparison.dimension_z_score) <= eps

    def test_baseline_with_zero_std_skips_metric(self) -> None:
        """Should skip z-score computation when baseline std is zero."""
        from modelcypher.core.domain.geometry.curvature_profile import FamilyBaseline

        source = ModelProfile(
            model_path="/path/to/source",
            num_layers=2,
            layer_profiles=[
                LayerProfile(layer_idx=0, sectional_curvature_mean=0.1),
                LayerProfile(layer_idx=1, sectional_curvature_mean=0.2),
            ],
        )
        target = ModelProfile(
            model_path="/path/to/target",
            num_layers=2,
            layer_profiles=[
                LayerProfile(layer_idx=0, sectional_curvature_mean=0.3),
                LayerProfile(layer_idx=1, sectional_curvature_mean=0.4),
            ],
        )

        baseline = FamilyBaseline(
            family="test",
            layer_positions=[0.0, 1.0],
            sectional_std_by_position=[0.0, 0.0],  # Zero std
            sample_count=1,
        )

        comparison = compare_profiles(source, target, baseline=baseline)

        # Should skip sectional z-score due to zero std
        assert comparison.sectional_z_score is None

    def test_z_score_serialization_roundtrip(self) -> None:
        """Z-score fields should survive serialization."""
        comparison = ProfileComparison(
            source_path="/source",
            target_path="/target",
            sectional_z_score=1.5,
            ricci_z_score=0.8,
            dimension_z_score=2.1,
            baseline_family="qwen",
            baseline_model_count=3,
        )

        d = comparison.to_dict()
        restored = ProfileComparison.from_dict(d)

        assert restored.sectional_z_score == 1.5
        assert restored.ricci_z_score == 0.8
        assert restored.dimension_z_score == 2.1
        assert restored.baseline_family == "qwen"
        assert restored.baseline_model_count == 3
