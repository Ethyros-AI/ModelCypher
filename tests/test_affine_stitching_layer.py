from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.affine_stitching_layer import (
    AffineStitchingLayer,
    AnchorPair,
    Config,
)
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)


def test_train_min_samples() -> None:
    data = [
        AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
        AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
    ]
    config = Config(min_samples=3, max_iterations=10)
    assert AffineStitchingLayer.train(data, config=config) is None


def test_train_identity_mapping() -> None:
    data = [
        AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
        AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
        AnchorPair(source_activation=[1.0, 1.0], target_activation=[1.0, 1.0]),
    ]
    config = Config(
        min_samples=1,
        max_iterations=5,
        weight_decay=0.0,
        use_momentum=False,
        use_procrustes_warm_start=True,
    )
    result = AffineStitchingLayer.train(data, config=config)
    assert result is not None
    assert result.forward_error == pytest.approx(0.0, abs=1e-6)
    assert result.backward_error == pytest.approx(0.0, abs=1e-6)
    assert result.weights[0][0] == pytest.approx(1.0, abs=1e-5)
    assert result.weights[1][1] == pytest.approx(1.0, abs=1e-5)
    assert result.weights[0][1] == pytest.approx(0.0, abs=1e-5)
    assert result.weights[1][0] == pytest.approx(0.0, abs=1e-5)


def test_apply_and_inverse() -> None:
    weights = [[1.0, 0.0], [0.0, 1.0]]
    bias = [0.0, 0.0]
    activations = [[1.0, 2.0], [-1.0, 0.5]]

    forward = AffineStitchingLayer.apply(activations, weights, bias)
    assert forward == activations

    inverse = AffineStitchingLayer.apply_inverse(activations, weights)
    assert inverse == activations


def test_train_from_crms() -> None:
    metadata = AnchorMetadata(
        total_count=2,
        semantic_prime_count=2,
        computational_gate_count=0,
        anchor_ids=["prime:a", "prime:b"],
    )
    source = ConceptResponseMatrix(
        model_identifier="source",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )
    target = ConceptResponseMatrix(
        model_identifier="target",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )

    source.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }
    target.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }

    config = Config(min_samples=1, max_iterations=3, weight_decay=0.0, use_momentum=False)
    result = AffineStitchingLayer.train_from_crms(source, target, layer=0, config=config)
    assert result is not None
