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

"""Tests for ModelProfilerService identity updates."""

from __future__ import annotations

from modelcypher.core.domain.geometry.model_profile import ModelIdentity, ModelProfile
from modelcypher.core.use_cases.model_profiler_service import ModelProfilerService
from modelcypher.ports.model_probe import LayerInfo, ModelProbeResult


def test_update_identity_prefers_layer_count_config() -> None:
    probe_result = ModelProbeResult(
        architecture="llama",
        parameter_count=123,
        layers=[
            LayerInfo(name="layer.0", type="attn", parameters=1, shape=[1]),
            LayerInfo(name="layer.1", type="attn", parameters=1, shape=[1]),
        ],
        vocab_size=32000,
        hidden_size=1024,
        num_attention_heads=16,
        layer_count_config=12,
    )
    identity = ModelIdentity(
        model_id="model-abc",
        config_hash="cfg-hash",
        weights_hash="wt-hash",
        model_path="/tmp/model",
    )
    profile = ModelProfile(model_path="/tmp/model")

    service = ModelProfilerService()
    updated = service.update_identity(
        profile,
        model_path="/tmp/model",
        probe_result=probe_result,
        identity=identity,
    )

    assert updated.model_id == "model-abc"
    assert updated.config_hash == "cfg-hash"
    assert updated.weights_hash == "wt-hash"
    assert updated.num_layers == 12
    assert updated.weight_tensor_count == 2
