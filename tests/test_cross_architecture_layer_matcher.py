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

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    Configuration,
    CrossArchitectureLayerMatcher,
)


def _test_config() -> Configuration:
    """Create test Configuration with explicit thresholds."""
    return Configuration.with_thresholds(
        high_confidence_threshold=0.75,
        medium_confidence_threshold=0.5,
    )


def _build_crm(model_id: str) -> ConceptResponseMatrix:
    anchor_ids = ["prime:A", "prime:B", "prime:C"]
    metadata = AnchorMetadata(
        total_count=3,
        semantic_prime_count=3,
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier=model_id,
        layer_count=2,
        hidden_dim=2,
        anchor_metadata=metadata,
    )
    crm.record_activations("prime:A", {0: [1.0, 0.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:B", {0: [0.0, 1.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:C", {0: [1.0, 1.0], 1: [0.0, 1.0]})
    return crm


def test_layer_matcher_basic_alignment() -> None:
    source = _build_crm("source")
    target = _build_crm("target")
    result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())
    assert len(result.mappings) == 2
    assert result.mappings[0].source_layer == 0
    assert result.mappings[0].target_layer == 0
    assert result.mappings[1].source_layer == 1
    assert result.mappings[1].target_layer == 1
    assert result.alignment_quality > 0.9


def test_layer_matcher_with_jaccard() -> None:
    source = _build_crm("source")
    target = _build_crm("target")
    jaccard = [[1.0, 0.0], [0.0, 1.0]]
    result = CrossArchitectureLayerMatcher.find_correspondence(
        source, target, _test_config(), jaccard_matrix=jaccard
    )
    assert result.visualization_data.combined_matrix is not None


def _build_crm_with_dims(model_id: str, hidden_dim: int, layer_count: int) -> ConceptResponseMatrix:
    """Build a CRM with specific dimensions."""
    anchor_ids = ["prime:A", "prime:B", "prime:C"]
    metadata = AnchorMetadata(
        total_count=3,
        semantic_prime_count=3,
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier=model_id,
        layer_count=layer_count,
        hidden_dim=hidden_dim,
        anchor_metadata=metadata,
    )
    # Generate activations with appropriate dimension
    for layer in range(layer_count):
        crm.record_activations("prime:A", {layer: [1.0] * hidden_dim})
        crm.record_activations("prime:B", {layer: [0.5] * hidden_dim})
        crm.record_activations("prime:C", {layer: [0.0] * hidden_dim})
    return crm


class TestCrossArchitectureEdgeCases:
    """Edge case tests for cross-architecture layer matching."""

    def test_768d_vs_1024d_alignment(self) -> None:
        """Test alignment between common transformer dimensions."""
        source = _build_crm_with_dims("source-768", hidden_dim=768, layer_count=2)
        target = _build_crm_with_dims("target-1024", hidden_dim=1024, layer_count=2)

        result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())

        # Should produce mappings despite dimension mismatch
        assert len(result.mappings) == 2
        assert result.alignment_quality >= 0.0

    def test_different_layer_counts(self) -> None:
        """Test alignment with different layer counts."""
        source = _build_crm_with_dims("source-shallow", hidden_dim=2, layer_count=2)
        target = _build_crm_with_dims("target-deep", hidden_dim=2, layer_count=4)

        result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())

        # Should map source layers to some target layers
        assert len(result.mappings) == 2  # Source has 2 layers

    def test_empty_crm_handling(self) -> None:
        """Test handling of CRMs with no activations."""
        anchor_ids = ["prime:A"]
        metadata = AnchorMetadata(
            total_count=1,
            semantic_prime_count=1,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        empty_crm = ConceptResponseMatrix(
            model_identifier="empty",
            layer_count=2,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        # No activations recorded

        target = _build_crm("target")

        # Should handle gracefully, may produce empty or low-quality result
        result = CrossArchitectureLayerMatcher.find_correspondence(empty_crm, target, _test_config())

        # Result should exist, quality may be 0
        assert result is not None

    def test_single_layer_models(self) -> None:
        """Test alignment with single-layer models."""
        source = _build_crm_with_dims("source-1", hidden_dim=2, layer_count=1)
        target = _build_crm_with_dims("target-1", hidden_dim=2, layer_count=1)

        result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())

        assert len(result.mappings) == 1
        assert result.mappings[0].source_layer == 0
        assert result.mappings[0].target_layer == 0

    def test_many_layers_models(self) -> None:
        """Test alignment with many layers."""
        source = _build_crm_with_dims("source-32", hidden_dim=4, layer_count=32)
        target = _build_crm_with_dims("target-32", hidden_dim=4, layer_count=32)

        result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())

        assert len(result.mappings) == 32
        assert result.alignment_quality >= 0.0

    def test_very_different_architectures(self) -> None:
        """Test alignment between very different architectures."""
        # Small model with few layers and low dimension
        small = _build_crm_with_dims("small", hidden_dim=64, layer_count=4)
        # Large model with many layers and high dimension
        large = _build_crm_with_dims("large", hidden_dim=4096, layer_count=32)

        result = CrossArchitectureLayerMatcher.find_correspondence(small, large, _test_config())

        # Should still produce some mappings
        assert len(result.mappings) == 4  # Limited by smaller model

    def test_jaccard_matrix_shape_mismatch(self) -> None:
        """Test with jaccard matrix of wrong shape."""
        source = _build_crm("source")  # 2 layers
        target = _build_crm("target")  # 2 layers

        # Wrong shape jaccard (3x3 instead of 2x2)
        wrong_jaccard = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        # Should either handle gracefully or use default behavior
        try:
            result = CrossArchitectureLayerMatcher.find_correspondence(
                source, target, _test_config(), jaccard_matrix=wrong_jaccard
            )
            # If it doesn't raise, result should still be valid
            assert result is not None
        except (ValueError, IndexError):
            # Raising an error for shape mismatch is also acceptable
            pass

    def test_all_zero_activations(self) -> None:
        """Test with all-zero activations."""
        anchor_ids = ["prime:A", "prime:B"]
        metadata = AnchorMetadata(
            total_count=2,
            semantic_prime_count=2,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        source = ConceptResponseMatrix(
            model_identifier="zero-source",
            layer_count=2,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        source.record_activations("prime:A", {0: [0.0, 0.0], 1: [0.0, 0.0]})
        source.record_activations("prime:B", {0: [0.0, 0.0], 1: [0.0, 0.0]})

        target = _build_crm("target")

        result = CrossArchitectureLayerMatcher.find_correspondence(source, target, _test_config())

        # Should not crash on zero activations
        assert result is not None
