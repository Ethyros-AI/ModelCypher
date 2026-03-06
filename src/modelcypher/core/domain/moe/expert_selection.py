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

"""Hybrid MoE expert selection from routing affinity x geometric capacity."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from modelcypher.core.domain.moe.routing_analysis import RoutingProfile
from modelcypher.core.domain.moe.topology import MoETopology
from modelcypher.core.domain.training.geometric_lora import LayerGeometry


def _representative_geometry(geometries: dict[str, LayerGeometry]) -> LayerGeometry | None:
    if not geometries:
        return None
    gate = geometries.get("gate_proj")
    if gate is not None:
        return gate
    for key in sorted(geometries):
        if key.endswith("gate_proj"):
            return geometries[key]
    return next(iter(geometries.values()))


def _nb_lora_params(rank: int, geom: LayerGeometry) -> int:
    out_features, in_features = geom.shape
    return int(rank * (int(in_features) + int(out_features) + 1))


@dataclass(frozen=True)
class ExpertTarget:
    """A selected expert for NB-LoRA training."""

    layer_idx: int
    expert_idx: int
    category: str
    routing_frequency: float
    geometry: LayerGeometry
    rank: int


@dataclass(frozen=True)
class ExpertTargetSelection:
    """Complete MoE expert-selection result."""

    targets: list[ExpertTarget]
    saturated: list[tuple[int, int]]
    skipped: list[tuple[int, int]]
    topology: MoETopology
    estimated_params_total: int = 0

    @property
    def target_module_keys(self) -> list[str]:
        """Layer keys suitable for inject_nb_lora()."""
        keys: list[str] = []
        for target in self.targets:
            prefix = f"model.layers.{target.layer_idx}.mlp.experts.{target.expert_idx}"
            keys.extend([
                f"{prefix}.gate_proj.weight",
                f"{prefix}.up_proj.weight",
                f"{prefix}.down_proj.weight",
            ])
        return sorted(set(keys))

    @property
    def n_trainable_experts(self) -> int:
        return len(self.targets)

    @property
    def estimated_params(self) -> int:
        """Total trainable NB-LoRA parameters across selected experts."""
        return int(self.estimated_params_total)


def select_expert_targets(
    routing_profile: RoutingProfile,
    expert_geometries: dict[tuple[int, int], dict[str, LayerGeometry]],
    topology: MoETopology,
) -> ExpertTargetSelection:
    """Select experts by routing affinity and geometric headroom.

    Primary threshold per layer is derived from routing entropy:
      effective_K = exp(H)  where H = Shannon entropy of routing frequencies
      threshold   = 1 / effective_K

    When routing is uniform (H = log K): threshold = 1/K, all experts near boundary.
    When routing is concentrated on M experts (H ≈ log M): threshold ≈ 1/M.
    This replaces a fixed 3× multiplier with a measurement-derived value.
    """
    uniform = topology.uniform_routing_frequency

    # Per-layer entropy-derived primary threshold.
    per_layer_threshold: dict[int, float] = {}
    for layer_idx in topology.moe_layer_indices:
        h = routing_profile.layer_routing_entropy(layer_idx)
        if h > 0:
            per_layer_threshold[layer_idx] = 1.0 / math.exp(h)
        else:
            # H=0: all routing to one expert; threshold = 1.0
            per_layer_threshold[layer_idx] = 1.0

    tail_dims_all: list[int] = []
    representative: dict[tuple[int, int], LayerGeometry] = {}
    for key, geoms in expert_geometries.items():
        rep = _representative_geometry(geoms)
        if rep is None:
            continue
        representative[key] = rep
        tail_dims_all.append(rep.tail_dims)

    median_tail = (
        float(statistics.median(tail_dims_all))
        if tail_dims_all
        else 0.0
    )

    targets: list[ExpertTarget] = []
    saturated: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    primary_layers: set[int] = set()

    # Pass 1: primary + saturated.
    for key, stats in sorted(routing_profile.stats.items()):
        rep = representative.get(key)
        if rep is None:
            skipped.append(key)
            continue
        layer_threshold = per_layer_threshold.get(key[0], uniform)
        if stats.frequency >= layer_threshold and rep.tail_dims > 0:
            targets.append(ExpertTarget(
                layer_idx=key[0],
                expert_idx=key[1],
                category="primary",
                routing_frequency=stats.frequency,
                geometry=rep,
                rank=rep.tail_dims,
            ))
            primary_layers.add(key[0])
        elif stats.frequency >= layer_threshold and rep.tail_dims == 0:
            saturated.append(key)

    # Pass 2: expansion in layers that already have primary experts.
    already_selected = {(t.layer_idx, t.expert_idx) for t in targets}
    for key, stats in sorted(routing_profile.stats.items()):
        if key in already_selected:
            continue
        rep = representative.get(key)
        if rep is None:
            continue
        if (
            key[0] in primary_layers
            and stats.frequency < uniform
            and float(rep.tail_dims) > median_tail
            and rep.tail_dims > 0
        ):
            targets.append(ExpertTarget(
                layer_idx=key[0],
                expert_idx=key[1],
                category="expansion",
                routing_frequency=stats.frequency,
                geometry=rep,
                rank=rep.tail_dims,
            ))
        elif key not in saturated:
            skipped.append(key)

    estimated_params_total = 0
    for target in targets:
        geoms = expert_geometries.get((target.layer_idx, target.expert_idx), {})
        if geoms:
            estimated_params_total += sum(
                _nb_lora_params(target.rank, geom) for geom in geoms.values()
            )
        else:
            estimated_params_total += 3 * _nb_lora_params(target.rank, target.geometry)

    return ExpertTargetSelection(
        targets=targets,
        saturated=sorted(set(saturated)),
        skipped=sorted(set(skipped)),
        topology=topology,
        estimated_params_total=estimated_params_total,
    )


__all__ = [
    "ExpertTarget",
    "ExpertTargetSelection",
    "select_expert_targets",
]
