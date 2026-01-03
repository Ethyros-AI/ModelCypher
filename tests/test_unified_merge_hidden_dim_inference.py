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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge import UnifiedGeometricMerger


class _MockLoader:
    def load_model_for_training(self, model_path, lora_config=None):  # pragma: no cover
        raise NotImplementedError

    def load_weights_as_numpy(self, model_path):  # pragma: no cover
        raise NotImplementedError

    def load_weights(self, model_path):  # pragma: no cover
        raise NotImplementedError


def test_infer_hidden_dim_prefers_norm_weight_over_quant_metadata():
    backend = get_default_backend()
    merger = UnifiedGeometricMerger(model_loader=_MockLoader())

    # Simulate a quantized layer with per-group scales (in_dim=3584, group_size=64 -> 56 groups)
    weights = {
        "model.layers.0.self_attn.q_proj.scales": backend.zeros((3584, 56)),
        "model.layers.0.input_layernorm.weight": backend.zeros((3584,)),
    }

    assert merger._infer_hidden_dim(weights) == 3584


def test_infer_hidden_dim_falls_back_to_attention_projection():
    backend = get_default_backend()
    merger = UnifiedGeometricMerger(model_loader=_MockLoader())

    # GQA K-proj shape: [kv_dim, hidden] (e.g., 512 x 3584)
    weights = {
        "model.layers.0.self_attn.k_proj.weight": backend.zeros((512, 3584)),
    }

    assert merger._infer_hidden_dim(weights) == 3584
