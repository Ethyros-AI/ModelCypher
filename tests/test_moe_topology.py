# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

from modelcypher.core.domain.moe.topology import MoETopology


def test_dense_model_returns_none():
    config = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
    }
    assert MoETopology.from_config(config) is None


def test_qwen_moe_detects_all_layers_when_sparse_step_is_one():
    config = {
        "model_type": "qwen3",
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "decoder_sparse_step": 1,
    }
    topology = MoETopology.from_config(config)
    assert topology is not None
    assert topology.num_experts == 256
    assert topology.num_experts_per_tok == 8
    assert topology.moe_intermediate_size == 512
    assert topology.num_layers == 40
    assert topology.moe_layer_indices == list(range(40))
    assert topology.total_experts == 256 * 40
    assert topology.uniform_routing_frequency == 1.0 / 256.0


def test_decoder_sparse_step_builds_sparse_layer_indices():
    config = {
        "model_type": "mixtral",
        "num_hidden_layers": 12,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 14336,
        "decoder_sparse_step": 3,
    }
    topology = MoETopology.from_config(config)
    assert topology is not None
    assert topology.num_experts == 8
    assert topology.moe_layer_indices == [0, 3, 6, 9]


def test_explicit_moe_layer_indices_override_sparse_step():
    config = {
        "num_hidden_layers": 16,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "decoder_sparse_step": 2,
        "moe_layer_indices": [1, 4, 7, 10],
    }
    topology = MoETopology.from_config(config)
    assert topology is not None
    assert topology.moe_layer_indices == [1, 4, 7, 10]


def test_shared_expert_detection():
    config = {
        "num_hidden_layers": 4,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 512,
    }
    topology = MoETopology.from_config(config)
    assert topology is not None
    assert topology.has_shared_expert is True
    assert topology.shared_expert_intermediate_size == 512
