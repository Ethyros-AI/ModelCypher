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
