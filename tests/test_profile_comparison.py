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

"""Tests for ProfileComparison - the alignment story between two models."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.model_profile import (
    DensitySummary,
    LayerProfile,
    ModelProfile,
    SemanticSignature,
    TopologySummary,
)
from modelcypher.core.domain.geometry.profile_comparison import (
    LayerComparison,
    ProfileComparison,
    compare_profiles,
)


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
        assert lc.alignment_effort == 0.0
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
            alignment_effort=0.6,
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
            alignment_effort=0.4,
        )
        d = original.to_dict()
        restored = LayerComparison.from_dict(d)

        assert restored.source_layer_idx == original.source_layer_idx
        assert restored.target_layer_idx == original.target_layer_idx
        assert restored.sectional_curvature_diff == original.sectional_curvature_diff
        assert restored.alignment_effort == original.alignment_effort


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
        assert pc.overall_compatibility == 0.0

    def test_create_full(self) -> None:
        """Should create ProfileComparison with all fields."""
        pc = ProfileComparison(
            source_path="/path/to/source",
            target_path="/path/to/target",
            architecture_match=True,
            hidden_dim_ratio=1.0,
            layer_count_ratio=1.0,
            curvature_compatibility=0.85,
            ricci_compatibility=0.90,
            dimension_compatibility=0.80,
            overall_compatibility=0.87,
            topology_similarity=0.92,
            semantic_alignment=0.88,
            layer_mapping={0: 0, 1: 1, 2: 2},
            layer_comparisons=[
                LayerComparison(0, 0, alignment_effort=0.2),
                LayerComparison(1, 1, alignment_effort=0.3),
            ],
            critical_layers=[],
            total_alignment_effort=0.5,
            mean_alignment_effort=0.25,
            max_alignment_effort=0.3,
            recommended_strategy="procrustes",
        )
        assert pc.architecture_match
        assert pc.overall_compatibility == 0.87
        assert len(pc.layer_comparisons) == 2

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip ProfileComparison."""
        original = ProfileComparison(
            source_path="/path/to/source",
            target_path="/path/to/target",
            architecture_match=True,
            curvature_compatibility=0.85,
            layer_mapping={0: 0, 1: 1},
            layer_comparisons=[
                LayerComparison(0, 0, alignment_effort=0.2),
            ],
            recommended_strategy="curvature_flow",
        )
        d = original.to_dict()
        restored = ProfileComparison.from_dict(d)

        assert restored.source_path == original.source_path
        assert restored.architecture_match == original.architecture_match
        assert restored.curvature_compatibility == original.curvature_compatibility
        assert restored.recommended_strategy == original.recommended_strategy
        assert len(restored.layer_comparisons) == 1


class TestCompareProfiles:
    """Tests for compare_profiles function."""

    def test_compare_identical_profiles(self) -> None:
        """Identical profiles should have high compatibility."""
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

        assert comparison.architecture_match
        assert comparison.hidden_dim_ratio == 1.0
        assert comparison.layer_count_ratio == 1.0
        # Identical profiles should have high compatibility
        assert comparison.curvature_compatibility > 0.9
        assert comparison.overall_compatibility > 0.9
        assert comparison.mean_alignment_effort < 0.1

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

        assert comparison.architecture_match
        assert comparison.hidden_dim_ratio == 1.0
        # Should still have reasonable compatibility
        assert comparison.overall_compatibility > 0.5
        # But some alignment effort needed
        assert comparison.mean_alignment_effort > 0.0

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
        """Should compute topology similarity when available."""
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

        assert comparison.topology_similarity is not None
        assert 0.0 <= comparison.topology_similarity <= 1.0

    def test_compare_with_semantic_signatures(self) -> None:
        """Should compute semantic alignment when available."""
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

        assert comparison.semantic_alignment is not None
        # Cosine similarity of [1,0,0,0,0] and [0.8,0.6,0,0,0]
        # = 0.8 / (1.0 * 1.0) = 0.8
        assert 0.7 < comparison.semantic_alignment < 0.9

    def test_compare_orthogonal_semantic_signatures(self) -> None:
        """Orthogonal semantic signatures should have zero alignment."""
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

        assert comparison.semantic_alignment is not None
        assert abs(comparison.semantic_alignment) < 0.01

    def test_recommended_strategy_procrustes(self) -> None:
        """Should recommend procrustes for well-matched profiles."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.1,
                    ollivier_ricci_mean=-0.1,
                    intrinsic_dimension=50.0,
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.12,  # Similar curvature
                    ollivier_ricci_mean=-0.12,
                    intrinsic_dimension=52.0,  # Similar dimension
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        comparison = compare_profiles(source, target)

        assert comparison.recommended_strategy == "procrustes"

    def test_recommended_strategy_projection_first(self) -> None:
        """Should recommend projection_first when dimensions differ significantly."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    intrinsic_dimension=50.0,
                )
                for i in range(4)
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    intrinsic_dimension=100.0,  # Double the dimension
                )
                for i in range(4)
            ],
        )

        comparison = compare_profiles(source, target)

        assert comparison.recommended_strategy == "projection_first"

    def test_recommended_strategy_curvature_flow(self) -> None:
        """Should recommend curvature_flow when curvature signs mismatch."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=-0.1,
                    ollivier_ricci_mean=-0.8,  # Strongly negative
                    intrinsic_dimension=50.0,
                    dominant_curvature_sign="negative",
                )
                for i in range(4)
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            layer_profiles=[
                LayerProfile(
                    layer_idx=i,
                    sectional_curvature_mean=0.1,  # Different sign
                    ollivier_ricci_mean=0.8,  # Strongly positive
                    intrinsic_dimension=50.0,
                    dominant_curvature_sign="positive",
                )
                for i in range(4)
            ],
        )

        comparison = compare_profiles(source, target)

        # Should recommend curvature_flow due to sign mismatch
        assert comparison.recommended_strategy == "curvature_flow"

    def test_critical_layers_detection(self) -> None:
        """Should detect layers requiring high alignment effort."""
        source = ModelProfile(
            model_path="/path/to/source",
            layer_profiles=[
                LayerProfile(
                    layer_idx=0,
                    ollivier_ricci_mean=-0.1,
                    intrinsic_dimension=50.0,
                ),
                LayerProfile(
                    layer_idx=1,
                    ollivier_ricci_mean=-0.1,
                    intrinsic_dimension=50.0,
                ),
                LayerProfile(
                    layer_idx=2,
                    ollivier_ricci_mean=-0.1,
                    intrinsic_dimension=50.0,
                ),
            ],
        )

        target = ModelProfile(
            model_path="/path/to/target",
            layer_profiles=[
                LayerProfile(
                    layer_idx=0,
                    ollivier_ricci_mean=-0.12,  # Similar
                    intrinsic_dimension=52.0,
                ),
                LayerProfile(
                    layer_idx=1,
                    ollivier_ricci_mean=-0.9,  # Very different!
                    intrinsic_dimension=100.0,  # Very different!
                ),
                LayerProfile(
                    layer_idx=2,
                    ollivier_ricci_mean=-0.12,  # Similar
                    intrinsic_dimension=52.0,
                ),
            ],
        )

        comparison = compare_profiles(source, target)

        # Layer 1 should be in critical layers due to high effort
        assert 1 in comparison.critical_layers

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
        assert comparison.total_alignment_effort == 0.0

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

        # Should not compute semantic alignment for mismatched lengths
        assert comparison.semantic_alignment is None

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

        assert comparison.semantic_alignment is None

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
        assert comparison.semantic_alignment is None
