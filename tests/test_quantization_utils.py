from __future__ import annotations

from modelcypher.core.use_cases.quantization_utils import (
    quantization_config_from_payload,
    quantization_hint_for_key,
    resolve_quantization,
)


def test_quantization_config_inherits_mode() -> None:
    payload = {
        "quantization": {
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
            "model.layers.0.mlp": {"group_size": 64, "bits": 8},
        }
    }
    config = quantization_config_from_payload(payload)
    assert config is not None
    assert config.default is not None
    assert config.default.mode == "mxfp4"

    hint = quantization_hint_for_key("model.layers.0.mlp.weight", config)
    assert hint is not None
    assert hint.bits == 8
    assert hint.group_size == 64
    assert hint.mode == "mxfp4"


def test_resolve_quantization_biasless_mxfp4() -> None:
    params = resolve_quantization(
        base_key="model.layers.0.mlp.gate_proj.weight",
        weight_shape=(16, 8),
        scales_shape=(16, 2),
        hint=None,
        biases_present=False,
    )
    assert params is not None
    assert params.bits == 4
    assert params.group_size == 32
    assert params.mode == "mxfp4"
