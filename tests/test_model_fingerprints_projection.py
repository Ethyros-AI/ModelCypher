from __future__ import annotations

import pytest
from modelcypher.core.domain.geometry.exceptions import ProjectionError
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ActivatedDimension,
    ActivationFingerprint,
    ModelFingerprints,
    ProbeSpace,
)
from modelcypher.core.domain.geometry.model_fingerprints_projection import ModelFingerprintsProjection


def test_model_fingerprints_projection_pca() -> None:
    fingerprints = [
        ActivationFingerprint(
            prime_id="prime-a",
            prime_text="A",
            activated_dimensions={0: [ActivatedDimension(index=0, activation=1.0)]},
        ),
        ActivationFingerprint(
            prime_id="prime-b",
            prime_text="B",
            activated_dimensions={0: [ActivatedDimension(index=1, activation=1.0)]},
        ),
    ]
    bundle = ModelFingerprints(
        model_id="model-1",
        probe_space=ProbeSpace.output_logits,
        probe_capture_key=None,
        hidden_dim=2,
        layer_count=1,
        fingerprints=fingerprints,
    )

    projection = ModelFingerprintsProjection.project_2d(bundle, max_features=4)
    assert len(projection.points) == 2
    assert len(projection.features) >= 2


def test_model_fingerprints_projection_empty() -> None:
    """Empty fingerprints should raise ProjectionError."""
    bundle = ModelFingerprints(
        model_id="model-empty",
        probe_space=ProbeSpace.output_logits,
        probe_capture_key=None,
        hidden_dim=128,
        layer_count=32,
        fingerprints=[],
    )
    with pytest.raises(ProjectionError, match="No fingerprints available"):
        ModelFingerprintsProjection.project_2d(bundle)


def test_model_fingerprints_projection_single_point() -> None:
    """Single fingerprint should raise ProjectionError (needs >= 2 for PCA)."""
    fingerprints = [
        ActivationFingerprint(
            prime_id="prime-1",
            prime_text="One",
            activated_dimensions={0: [ActivatedDimension(index=0, activation=1.0)]},
        )
    ]
    bundle = ModelFingerprints(
        model_id="model-single",
        probe_space=ProbeSpace.prelogits_hidden,
        probe_capture_key="layer_0",
        hidden_dim=2,
        layer_count=1,
        fingerprints=fingerprints,
    )
    with pytest.raises(ProjectionError, match="at least 2 fingerprints"):
        ModelFingerprintsProjection.project_2d(bundle)


def test_model_fingerprints_projection_dimensionality_mismatch() -> None:
    # Test that it handles fingerprints with inconsistent layer coverage
    fingerprints = [
        ActivationFingerprint(
            prime_id="p1",
            prime_text="A",
            activated_dimensions={0: [ActivatedDimension(index=0, activation=1.0)]},
        ),
        ActivationFingerprint(
            prime_id="p2",
            prime_text="B",
            activated_dimensions={1: [ActivatedDimension(index=0, activation=1.0)]}, # Different layer
        ),
    ]
    bundle = ModelFingerprints(
        model_id="model-mismatch",
        probe_space=ProbeSpace.output_logits,
        probe_capture_key=None,
        hidden_dim=2,
        layer_count=2,
        fingerprints=fingerprints,
    )
    projection = ModelFingerprintsProjection.project_2d(bundle)
    assert len(projection.points) == 2
