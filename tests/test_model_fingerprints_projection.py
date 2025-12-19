from __future__ import annotations

from modelcypher.core.domain.manifold_stitcher import ActivatedDimension, ActivationFingerprint, ModelFingerprints, ProbeSpace
from modelcypher.core.domain.model_fingerprints_projection import ModelFingerprintsProjection


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
