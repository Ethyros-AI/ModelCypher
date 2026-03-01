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

"""MoE topology detection from model config."""

from __future__ import annotations

from dataclasses import dataclass


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        ivalue = int(value)
        return ivalue if ivalue > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("-"):
            return None
        if stripped.isdigit():
            ivalue = int(stripped)
            return ivalue if ivalue > 0 else None
    return None


def _as_layer_indices(value: object, num_layers: int) -> list[int]:
    if not isinstance(value, list):
        return []

    parsed: list[int] = []
    for item in value:
        idx = _as_positive_int(item)
        if idx is None:
            if isinstance(item, int) and item == 0:
                idx = 0
            else:
                continue
        if num_layers > 0 and idx >= num_layers:
            continue
        parsed.append(idx)
    return sorted(set(parsed))


@dataclass(frozen=True)
class MoETopology:
    """MoE architecture topology extracted from config.json."""

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    has_shared_expert: bool
    shared_expert_intermediate_size: int | None
    moe_layer_indices: list[int]
    num_layers: int

    @classmethod
    def from_config(cls, config: dict) -> MoETopology | None:
        """Return topology for MoE config, else None for dense models."""
        num_experts = _as_positive_int(
            config.get("num_experts", config.get("num_local_experts")),
        )
        if num_experts is None:
            return None

        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            num_layers = len(layer_types)
        else:
            num_layers = _as_positive_int(config.get("num_hidden_layers")) or 0

        num_experts_per_tok = _as_positive_int(config.get("num_experts_per_tok")) or 1
        num_experts_per_tok = min(num_experts_per_tok, num_experts)

        moe_intermediate_size = (
            _as_positive_int(config.get("moe_intermediate_size"))
            or _as_positive_int(config.get("intermediate_size"))
            or 0
        )

        shared_size = _as_positive_int(config.get("shared_expert_intermediate_size"))
        has_shared_expert = (
            shared_size is not None
            or bool(config.get("shared_expert_intermediate_size"))
            or bool(config.get("has_shared_expert"))
        )

        explicit_indices = _as_layer_indices(config.get("moe_layer_indices"), num_layers)
        if explicit_indices:
            moe_layer_indices = explicit_indices
        else:
            step = _as_positive_int(config.get("decoder_sparse_step"))
            if step is not None and num_layers > 0:
                moe_layer_indices = list(range(0, num_layers, step))
            elif num_layers > 0:
                moe_layer_indices = list(range(num_layers))
            else:
                moe_layer_indices = []

        if not moe_layer_indices and num_layers > 0:
            moe_layer_indices = list(range(num_layers))

        return cls(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            has_shared_expert=has_shared_expert,
            shared_expert_intermediate_size=shared_size,
            moe_layer_indices=moe_layer_indices,
            num_layers=num_layers,
        )

    @property
    def total_experts(self) -> int:
        """Total routed experts across all sparse layers."""
        return self.num_experts * len(self.moe_layer_indices)

    @property
    def uniform_routing_frequency(self) -> float:
        """Uniform fair-share routing baseline for one expert."""
        if self.num_experts <= 0:
            return 0.0
        return 1.0 / float(self.num_experts)


__all__ = ["MoETopology"]
