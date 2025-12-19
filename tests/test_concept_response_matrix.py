from __future__ import annotations

from datetime import datetime, timezone

from modelcypher.core.domain.concept_response_matrix import (
    AnchorMetadata,
    ConceptResponseMatrix,
)


def _build_crm() -> ConceptResponseMatrix:
    anchor_ids = ["prime:A", "prime:B", "prime:C"]
    metadata = AnchorMetadata(
        total_count=3,
        semantic_prime_count=3,
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier="test-model",
        layer_count=2,
        hidden_dim=2,
        anchor_metadata=metadata,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    crm.record_activations("prime:A", {0: [1.0, 0.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:B", {0: [0.0, 1.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:C", {0: [1.0, 1.0], 1: [0.0, 1.0]})
    return crm


def test_activation_matrix_order() -> None:
    crm = _build_crm()
    matrix = crm.activation_matrix(0)
    assert matrix == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_cka_matrix_values() -> None:
    crm = _build_crm()
    cka = crm.compute_cka_matrix(crm)
    assert len(cka) == 2
    assert abs(cka[0][0] - 1.0) < 1e-6
    assert abs(cka[1][1] - 1.0) < 1e-6
    assert abs(cka[0][1] - 0.3162277) < 1e-6
    assert abs(cka[1][0] - 0.3162277) < 1e-6


def test_compare_report() -> None:
    crm = _build_crm()
    report = crm.compare(crm)
    assert report.common_anchor_count == 3
    assert len(report.layer_correspondence) == 2
    assert report.layer_correspondence[0].source_layer == 0
    assert report.layer_correspondence[0].target_layer == 0
