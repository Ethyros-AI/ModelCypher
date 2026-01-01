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

from modelcypher.core.use_cases.merge.stages.permute import stage_permute


def test_stage_permute_aligns_mlp_blocks(any_backend) -> None:
    backend = any_backend
    backend.random_seed(7)

    embed = backend.random_normal((4, 4))
    up = backend.random_normal((4, 4))
    gate = backend.random_normal((4, 4))
    down = backend.random_normal((4, 4))
    backend.eval(embed, up, gate, down)

    source_weights = {
        "model.embed_tokens.weight": embed,
        "model.layers.0.mlp.up_proj.weight": up,
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.down_proj.weight": down,
    }
    target_weights = {
        "model.embed_tokens.weight": embed,
        "model.layers.0.mlp.up_proj.weight": up,
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.down_proj.weight": down,
    }

    result = stage_permute(
        source_weights=source_weights,
        target_weights=target_weights,
        intersection_map_obj=None,
        layer_confidences={},
        infer_hidden_dim_fn=lambda _weights: 4,
        backend=backend,
    )

    assert "layers_permuted" in result.metrics
    assert result.metrics["layers_permuted"] >= 1
    assert "model.layers.0.mlp.up_proj.weight" in result.weights
    assert "model.layers.0.mlp.gate_proj.weight" in result.weights
    assert "model.layers.0.mlp.down_proj.weight" in result.weights
