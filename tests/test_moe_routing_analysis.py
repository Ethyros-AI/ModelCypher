# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

from modelcypher.core.domain.moe.routing_analysis import RoutingProfile
from modelcypher.core.domain.moe.topology import MoETopology


def _topology() -> MoETopology:
    topology = MoETopology.from_config({
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 256,
        "decoder_sparse_step": 1,
    })
    assert topology is not None
    return topology


def test_profile_computes_per_expert_frequency():
    topology = _topology()
    routing = {
        0: [[0], [0], [0], [1]],
        1: [[2], [2], [3], [3]],
    }
    profile = RoutingProfile.from_routing_decisions(routing, topology)

    assert profile.total_tokens == 4
    assert profile.stats[(0, 0)].frequency == 0.75
    assert profile.stats[(0, 1)].frequency == 0.25
    assert profile.stats[(0, 2)].frequency == 0.0
    assert profile.stats[(0, 0)].token_count == 3
    assert profile.stats[(0, 0)].mean_routing_prob == 0.0


def test_task_relevant_and_underutilized_use_uniform_baseline():
    topology = _topology()
    routing = {
        0: [[0], [0], [0], [1]],  # expert 0 gets 3/4 = 0.75
        1: [[1], [1], [1], [1]],
    }
    profile = RoutingProfile.from_routing_decisions(routing, topology)

    # uniform = 1/4 = 0.25, relevant threshold = 3 * uniform = 0.75
    assert profile.task_relevant_experts() == [(0, 0), (1, 1)]
    assert profile.underutilized_experts(0) == [(0, 2), (0, 3)]


def test_layer_entropy_is_shannon_entropy():
    topology = _topology()
    routing = {
        0: [[0], [0], [0], [1]],  # [0.75, 0.25, 0, 0]
        1: [[0], [1], [2], [3]],  # uniform
    }
    profile = RoutingProfile.from_routing_decisions(routing, topology)

    entropy_layer0 = profile.layer_routing_entropy(0)
    entropy_layer1 = profile.layer_routing_entropy(1)

    assert entropy_layer0 == -(
        0.75 * math.log(0.75) + 0.25 * math.log(0.25)
    )
    assert entropy_layer1 == -4.0 * (0.25 * math.log(0.25))
    assert entropy_layer1 > entropy_layer0
