from __future__ import annotations

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
    bundle = ModelFingerprints(
        model_id="model-empty",
        probe_space=ProbeSpace.output_logits,
        probe_capture_key=None,
        hidden_dim=128,
        layer_count=32,
        fingerprints=[],
    )
    projection = ModelFingerprintsProjection.project_2d(bundle)
    assert len(projection.points) == 0
    assert len(projection.features) == 0


def test_model_fingerprints_projection_single_point() -> None:
    fingerprints = [
        ActivationFingerprint(
            prime_id="prime-1",
            prime_text="One",
            activated_dimensions={0: [ActivatedDimension(index=0, activation=1.0)]},
        )
    ]
    bundle = ModelFingerprints(
        model_id="model-single",
        probe_space=ProbeSpace.residual_stream,
        probe_capture_key="layer_0",
        hidden_dim=2,
        layer_count=1,
        fingerprints=fingerprints,
    )
    projection = ModelFingerprintsProjection.project_2d(bundle)
    assert len(projection.points) == 1
    # Point should be at origin for single point PCA
    assert projection.points[0].x == pytest.approx(0.0)
    assert projection.points[0].y == pytest.approx(0.0)


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
