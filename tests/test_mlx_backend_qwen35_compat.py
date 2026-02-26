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

import json

from modelcypher.backends.mlx_backend import MLXBackend


def test_normalize_qwen35_text_config_derives_qwen3_next_fields():
    raw_config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "eos_token_id": [248046, 248044],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 262144,
            "head_dim": 256,
            "mlp_only_layers": [],
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
    }

    normalized = MLXBackend._normalize_qwen35_text_config(raw_config)

    assert normalized["model_type"] == "qwen3_next"
    assert normalized["rope_theta"] == 10000000
    assert normalized["partial_rotary_factor"] == 0.25
    assert normalized["num_experts"] == 0
    assert normalized["num_experts_per_tok"] == 0
    assert normalized["decoder_sparse_step"] == 1
    assert normalized["shared_expert_intermediate_size"] == 17408
    assert normalized["moe_intermediate_size"] == 17408
    assert normalized["tie_word_embeddings"] is False
    assert normalized["eos_token_id"] == [248046, 248044]


def test_remap_qwen35_weights_for_qwen3_next_transforms_expected_keys():
    weights = {
        "model.language_model.embed_tokens.weight": "embed",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a",
        "model.language_model.layers.0.linear_attn.out_proj.weight": "out",
        "model.language_model.layers.0.mlp.up_proj.weight": "up",
        "model.visual.blocks.0.attn.qkv.weight": "vision",
        "mtp.fc.weight": "mtp",
        "lm_head.weight": "head",
    }

    def concat(arrays, axis=0):
        return ("concat", axis, tuple(arrays))

    remapped = MLXBackend._remap_qwen35_weights_for_qwen3_next(weights, concatenate=concat)

    assert remapped["model.embed_tokens.weight"] == "embed"
    assert remapped["model.layers.0.linear_attn.out_proj.weight"] == "out"
    assert remapped["model.layers.0.mlp.up_proj.weight"] == "up"
    assert remapped["lm_head.weight"] == "head"
    assert remapped["model.layers.0.linear_attn.in_proj_qkvz.weight"] == (
        "concat",
        0,
        ("qkv", "z"),
    )
    assert remapped["model.layers.0.linear_attn.in_proj_ba.weight"] == (
        "concat",
        0,
        ("b", "a"),
    )

    stale_keys = (
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "mtp.fc.weight",
    )
    for key in stale_keys:
        assert key not in remapped


def test_is_qwen35_config_path_detects_new_model_type(tmp_path):
    model_dir = tmp_path / "qwen35"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text"},
            }
        ),
        encoding="utf-8",
    )

    assert MLXBackend._is_qwen35_config_path(str(model_dir)) is True
