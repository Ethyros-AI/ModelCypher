from __future__ import annotations

import pytest

from modelcypher.core.domain.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.generalized_procrustes import Config, GeneralizedProcrustes


def test_align_requires_min_models() -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    config = Config(min_models=2, max_iterations=5)
    assert GeneralizedProcrustes.align([matrix], config=config) is None


def test_align_identity_consensus() -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    result = GeneralizedProcrustes.align([matrix, matrix], config=Config(max_iterations=5))
    assert result is not None
    assert result.alignment_error == pytest.approx(0.0, abs=1e-6)
    assert result.consensus_variance_ratio == pytest.approx(1.0, abs=1e-6)
    assert result.dimension == 2
    assert result.model_count == 2


def test_align_crms_with_dimension_mismatch() -> None:
    metadata = AnchorMetadata(
        total_count=2,
        semantic_prime_count=2,
        computational_gate_count=0,
        anchor_ids=["prime:a", "prime:b"],
    )
    crm_a = ConceptResponseMatrix(
        model_identifier="a",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )
    crm_b = ConceptResponseMatrix(
        model_identifier="b",
        layer_count=1,
        hidden_dim=3,
        anchor_metadata=metadata,
    )
    crm_a.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }
    crm_b.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0, 0.0]),
        }
    }

    result = GeneralizedProcrustes.align_crms([crm_a, crm_b], layer=0, config=Config(max_iterations=5))
    assert result is not None
    assert result.dimension == 2
    assert result.sample_count == 2
