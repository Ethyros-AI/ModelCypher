# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

from modelcypher.core.domain.moe.routing_analysis import (
    RoutingProfile,
    routing_kl_divergence,
)
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

    # Layer 0 entropy-derived threshold ≈ 0.57, layer 1 threshold = 1.0
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


def test_routing_kl_divergence_identical_profiles_returns_zero():
    """KL(P||P) = 0 by definition."""
    topology = _topology()
    routing = {
        0: [[0], [0], [0], [1]],
        1: [[2], [2], [3], [3]],
    }
    profile = RoutingProfile.from_routing_decisions(routing, topology)
    kl = routing_kl_divergence(profile, profile)
    assert abs(kl) < 1e-12


def test_routing_kl_divergence_shifted_distribution():
    """Shifted routing should produce KL > 0."""
    topology = _topology()
    pre_routing = {
        0: [[0], [0], [0], [1]],
        1: [[2], [2], [3], [3]],
    }
    post_routing = {
        0: [[1], [1], [1], [0]],
        1: [[2], [3], [3], [3]],
    }
    pre = RoutingProfile.from_routing_decisions(pre_routing, topology)
    post = RoutingProfile.from_routing_decisions(post_routing, topology)
    kl = routing_kl_divergence(pre, post)
    assert kl > 0.0


def test_routing_kl_divergence_handles_zero_frequency_experts():
    """Experts with zero frequency in pre should not cause log(0)."""
    topology = _topology()
    # Pre: all tokens to expert 0 (experts 1,2,3 have zero frequency)
    pre_routing = {
        0: [[0], [0], [0], [0]],
        1: [[0], [0], [0], [0]],
    }
    # Post: tokens spread to expert 1 (which was zero in pre)
    post_routing = {
        0: [[0], [0], [1], [1]],
        1: [[0], [0], [1], [1]],
    }
    pre = RoutingProfile.from_routing_decisions(pre_routing, topology)
    post = RoutingProfile.from_routing_decisions(post_routing, topology)
    kl = routing_kl_divergence(pre, post)
    assert math.isfinite(kl)
    assert kl > 0.0


def test_routing_kl_divergence_empty_profiles():
    """Profiles with no MoE layers return 0.0."""
    pre = RoutingProfile(stats={}, total_tokens=0, num_layers=0, num_experts=4)
    post = RoutingProfile(stats={}, total_tokens=0, num_layers=0, num_experts=4)
    kl = routing_kl_divergence(pre, post)
    assert kl == 0.0
