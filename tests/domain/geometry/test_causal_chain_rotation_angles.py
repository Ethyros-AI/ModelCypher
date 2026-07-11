from __future__ import annotations

from modelcypher.core.domain.geometry.causal_chain import (
    assemble_chain_profile,
    compute_layer_rotation_angles,
)


def test_chain_profile_emits_rotation_angle_keys_not_curvature_keys() -> None:
    sublayer_data = [
        {
            "h_in": [[1.0, 0.0], [0.0, 1.0]],
            "h_out": [[0.0, 1.0], [-1.0, 0.0]],
            "h_post_attn": [[1.0, 1.0], [-1.0, 1.0]],
            "has_decomposition": True,
        },
        {
            "h_in": [[0.0, 1.0], [-1.0, 0.0]],
            "h_out": [[-1.0, 0.0], [0.0, -1.0]],
            "h_post_attn": [[-1.0, 1.0], [-1.0, -1.0]],
            "has_decomposition": True,
        },
        {
            "h_in": [[-1.0, 0.0], [0.0, -1.0]],
            "h_out": [[0.0, -1.0], [1.0, 0.0]],
            "h_post_attn": [[-1.0, -1.0], [1.0, -1.0]],
            "has_decomposition": True,
        },
    ]

    measurements = compute_layer_rotation_angles(sublayer_data)
    profile = assemble_chain_profile(
        model_path="synthetic",
        num_layers=len(sublayer_data),
        hidden_dim=2,
        probe_count=2,
        rotation_measurements=measurements,
        entropies=[1.0, 1.5, 2.0],
    )

    payload = profile.as_dict()
    layer_payload = payload["layers"][0]
    correlation_payload = payload["correlations"]

    assert "layerRotationAngle" in layer_payload
    assert "cumulativeLayerRotationAngle" in layer_payload
    assert "attnRotationAngle" in layer_payload
    assert "mlpRotationAngle" in layer_payload
    assert "entropyToLayerRotationAngle" in correlation_payload
    assert "cumulativeLayerRotationAngleToId" in correlation_payload
    assert all("Curvature" not in key for key in layer_payload)
    assert all("Curvature" not in key for key in correlation_payload)
