from __future__ import annotations

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.cross_architecture_layer_matcher import CrossArchitectureLayerMatcher


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
    result = CrossArchitectureLayerMatcher.find_correspondence(source, target)
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
    result = CrossArchitectureLayerMatcher.find_correspondence(source, target, jaccard_matrix=jaccard)
    assert result.visualization_data.combined_matrix is not None
