# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

from modelcypher.core.domain.moe.expert_selection import select_expert_targets
from modelcypher.core.domain.moe.routing_analysis import RoutingProfile
from modelcypher.core.domain.moe.topology import MoETopology
from modelcypher.core.domain.training.geometric_lora import LayerGeometry


def _geom(layer: int, expert: int, proj: str, tail_dims: int) -> LayerGeometry:
    return LayerGeometry(
        layer_key=f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
        shape=(16, 8),
        sigma_max=2.0,
        sigma_k=0.5,
        effective_rank=8,
        full_rank=8,
        decay_ratio=4.0,
        tail_dims=tail_dims,
        shannon_effective_rank=4.0,
        spectral_gap=0.25,
    )


def test_hybrid_selection_primary_expansion_saturated_and_skipped():
    topology = MoETopology.from_config({
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 512,
    })
    assert topology is not None

    routing_profile = RoutingProfile.from_routing_decisions(
        routing_decisions={
            0: [[0], [0], [0], [1]],  # expert 0 primary (0.75)
            1: [[1], [1], [1], [1]],  # expert 1 saturated candidate (1.0)
        },
        topology=topology,
    )

    expert_geometries: dict[tuple[int, int], dict[str, LayerGeometry]] = {}
    for layer in topology.moe_layer_indices:
        for expert in range(topology.num_experts):
            tail = 1
            if (layer, expert) == (0, 0):
                tail = 5   # primary
            elif (layer, expert) == (0, 2):
                tail = 8   # expansion (underutilized + high capacity)
            elif (layer, expert) == (1, 1):
                tail = 0   # saturated
            expert_geometries[(layer, expert)] = {
                "gate_proj": _geom(layer, expert, "gate_proj", tail),
                "up_proj": _geom(layer, expert, "up_proj", tail),
                "down_proj": _geom(layer, expert, "down_proj", tail),
            }

    selection = select_expert_targets(
        routing_profile=routing_profile,
        expert_geometries=expert_geometries,
        topology=topology,
    )

    selected_pairs = {(t.layer_idx, t.expert_idx): t.category for t in selection.targets}
    assert selected_pairs[(0, 0)] == "primary"
    assert selected_pairs[(0, 2)] == "expansion"
    assert (1, 1) in selection.saturated
    assert selection.n_trainable_experts == 2
    assert selection.estimated_params > 0

    module_keys = selection.target_module_keys
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in module_keys
    assert "model.layers.0.mlp.experts.0.up_proj.weight" in module_keys
    assert "model.layers.0.mlp.experts.0.down_proj.weight" in module_keys
