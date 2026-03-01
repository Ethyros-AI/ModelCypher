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

"""Routing-profile analysis for Mixture-of-Experts models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelcypher.core.domain.moe.topology import MoETopology


def _to_nested_ints(value: object) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        return []
    if not value:
        return []

    rows: list[list[int]] = []
    for item in value:
        if isinstance(item, list):
            row = [int(v) for v in item]
        else:
            row = [int(item)]
        rows.append(row)
    return rows


@dataclass(frozen=True)
class ExpertRoutingStats:
    """Routing statistics for a single expert."""

    layer_idx: int
    expert_idx: int
    frequency: float
    mean_routing_prob: float
    token_count: int


@dataclass(frozen=True)
class RoutingProfile:
    """Complete routing profile from a dataset pass."""

    stats: dict[tuple[int, int], ExpertRoutingStats]
    total_tokens: int
    num_layers: int
    num_experts: int

    @classmethod
    def from_routing_decisions(
        cls,
        routing_decisions: dict[int, object],
        topology: MoETopology,
    ) -> RoutingProfile:
        """Build profile from selected expert IDs per token/layer."""
        stats: dict[tuple[int, int], ExpertRoutingStats] = {}
        total_tokens = 0

        for layer_idx in topology.moe_layer_indices:
            selected = _to_nested_ints(routing_decisions.get(layer_idx, []))
            layer_tokens = len(selected)
            total_tokens = max(total_tokens, layer_tokens)
            top_k = 0
            for row in selected:
                top_k = max(top_k, len(row))
            if top_k <= 0:
                top_k = max(1, topology.num_experts_per_tok)

            slot_total = max(1, layer_tokens * top_k)
            counts = [0 for _ in range(topology.num_experts)]

            for row in selected:
                for idx in row:
                    if 0 <= idx < topology.num_experts:
                        counts[idx] += 1

            for expert_idx in range(topology.num_experts):
                count = counts[expert_idx]
                frequency = float(count) / float(slot_total)
                stats[(layer_idx, expert_idx)] = ExpertRoutingStats(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    frequency=frequency,
                    # Routing probabilities are not captured in indices-only hooks.
                    # Keep the field populated with measured frequency.
                    mean_routing_prob=frequency,
                    token_count=count,
                )

        return cls(
            stats=stats,
            total_tokens=total_tokens,
            num_layers=topology.num_layers,
            num_experts=topology.num_experts,
        )

    @property
    def uniform_frequency(self) -> float:
        if self.num_experts <= 0:
            return 0.0
        return 1.0 / float(self.num_experts)

    def task_relevant_experts(
        self,
        affinity_threshold: float = 3.0,
    ) -> list[tuple[int, int]]:
        """Experts with routing >= threshold * uniform fair-share baseline."""
        threshold = affinity_threshold * self.uniform_frequency
        selected = [
            key
            for key, value in self.stats.items()
            if value.frequency >= threshold
        ]
        return sorted(selected)

    def underutilized_experts(
        self,
        layer_idx: int,
    ) -> list[tuple[int, int]]:
        """Experts below uniform fair-share frequency in one layer."""
        threshold = self.uniform_frequency
        selected = [
            key
            for key, value in self.stats.items()
            if key[0] == layer_idx and value.frequency < threshold
        ]
        return sorted(selected)

    def layer_routing_entropy(self, layer_idx: int) -> float:
        """Shannon entropy of routing frequencies in one layer."""
        freqs = [
            value.frequency
            for (idx, _expert), value in self.stats.items()
            if idx == layer_idx
        ]
        if not freqs:
            return 0.0

        total = sum(freqs)
        if total <= 0.0:
            return 0.0

        normalized = [f / total for f in freqs]
        entropy = 0.0
        for p in normalized:
            if p > 0.0:
                entropy -= p * math.log(p)
        return entropy


def build_routing_profile(
    routing_decisions: dict[int, object],
    topology: MoETopology,
) -> RoutingProfile:
    """Convenience wrapper for profile construction."""
    return RoutingProfile.from_routing_decisions(routing_decisions, topology)


__all__ = [
    "ExpertRoutingStats",
    "RoutingProfile",
    "build_routing_profile",
]
