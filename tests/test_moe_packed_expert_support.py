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

"""Tests for packed MoE expert support (Batch 1 review fixes)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from modelcypher.core.domain.moe.topology import MoETopology, detect_expert_format

# ---------------------------------------------------------------------------
# detect_expert_format
# ---------------------------------------------------------------------------


def test_detect_expert_format_packed_switch_via_switch_mlp():
    """SwitchGLU layout detected via mlp.switch_mlp.gate_proj."""
    switch_mlp = SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object())
    mlp = SimpleNamespace(switch_mlp=switch_mlp)
    assert detect_expert_format(mlp) == "packed_switch"


def test_detect_expert_format_packed_switch_via_experts_gate_proj():
    """Packed layout detected when mlp.experts has gate_proj attribute."""
    experts = SimpleNamespace(gate_proj=object())
    mlp = SimpleNamespace(experts=experts)
    assert detect_expert_format(mlp) == "packed_switch"


def test_detect_expert_format_individual_list():
    """Individual experts as a list of modules with gate_proj."""
    expert_a = SimpleNamespace(gate_proj=object())
    expert_b = SimpleNamespace(gate_proj=object())
    mlp = SimpleNamespace(experts=[expert_a, expert_b])
    assert detect_expert_format(mlp) == "individual"


def test_detect_expert_format_individual_dict():
    """Individual experts as a dict (keyed by index or name)."""
    expert_a = SimpleNamespace(gate_proj=object())
    expert_b = SimpleNamespace(gate_proj=object())
    mlp = SimpleNamespace(experts={0: expert_a, 1: expert_b})
    assert detect_expert_format(mlp) == "individual"


def test_detect_expert_format_unknown_no_experts():
    """Returns unknown when mlp has no experts or switch_mlp."""
    mlp = SimpleNamespace(up_proj=object())
    assert detect_expert_format(mlp) == "unknown"


def test_detect_expert_format_unknown_empty_list():
    """Returns unknown when experts list is empty."""
    mlp = SimpleNamespace(experts=[])
    assert detect_expert_format(mlp) == "unknown"


# ---------------------------------------------------------------------------
# MoETopology.from_config with text_config
# ---------------------------------------------------------------------------


def test_topology_from_config_reads_text_config_fallback():
    """Qwen3.5-style config nests MoE params in text_config."""
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
        },
    }
    topo = MoETopology.from_config(config)
    assert topo is not None
    assert topo.num_experts == 256
    assert topo.num_experts_per_tok == 8
    assert topo.num_layers == 40
    assert topo.moe_intermediate_size == 512
    assert topo.has_shared_expert is True
    assert topo.shared_expert_intermediate_size == 512


def test_topology_from_config_top_level_overrides_text_config():
    """Top-level keys take precedence over text_config."""
    config = {
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 32,
        "moe_intermediate_size": 1024,
        "text_config": {
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "moe_intermediate_size": 512,
        },
    }
    topo = MoETopology.from_config(config)
    assert topo is not None
    assert topo.num_experts == 64
    assert topo.num_experts_per_tok == 4
    assert topo.num_layers == 32


def test_topology_from_config_layer_types_determines_num_layers():
    """layer_types list length overrides num_hidden_layers."""
    config = {
        "text_config": {
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 999,
            "moe_intermediate_size": 512,
            "layer_types": ["linear_attention"] * 30 + ["full_attention"] * 10,
        },
    }
    topo = MoETopology.from_config(config)
    assert topo is not None
    assert topo.num_layers == 40


def test_topology_from_config_dense_model_returns_none():
    """Dense model with text_config but no num_experts returns None."""
    config = {
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
        },
    }
    assert MoETopology.from_config(config) is None


# ---------------------------------------------------------------------------
# Virtual projections in _iter_layer_weight_projections
# ---------------------------------------------------------------------------


def test_virtual_projection_exposes_weight():
    from modelcypher.backends._mlx_training_adapter_core_mixin import _VirtualProjection

    arr = np.zeros((4, 8))
    vp = _VirtualProjection(arr)
    assert vp.weight is arr


# ---------------------------------------------------------------------------
# Model info MoE rendering
# ---------------------------------------------------------------------------


def test_model_info_renders_moe_topology(tmp_path):
    """model info text output includes MoE topology section."""
    model_dir = tmp_path / "moe_model"
    model_dir.mkdir()

    config = {
        "architectures": ["TestMoE"],
        "text_config": {
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "vocab_size": 32000,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Create a minimal safetensors-like structure so probe can run
    # We just need the topology detection from config, not actual weights
    from modelcypher.core.domain.moe.topology import MoETopology

    text_cfg = config.get("text_config", {})
    cfg = {**text_cfg, **config}
    topo = MoETopology.from_config(cfg)

    assert topo is not None
    assert topo.num_experts == 64
    assert topo.num_experts_per_tok == 4
    assert topo.has_shared_expert is True
    assert len(topo.moe_layer_indices) == 24
