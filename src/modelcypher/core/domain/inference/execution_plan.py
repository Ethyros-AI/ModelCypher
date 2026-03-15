# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
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

"""Execution-plan domain types for inference-time layer routing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PlanKind = Literal["identity", "explicit", "rys"]


@dataclass(frozen=True)
class LayerExecutionPlan:
    """Validated execution plan over a fixed base layer stack."""

    base_layer_count: int
    layer_indices: tuple[int, ...]
    plan_kind: PlanKind
    label: str | None = None

    def __post_init__(self) -> None:
        if self.base_layer_count <= 0:
            raise ValueError("base_layer_count must be > 0")
        if not self.layer_indices:
            raise ValueError("layer_indices must be non-empty")
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= self.base_layer_count:
                raise ValueError(
                    f"layer index {layer_idx} out of bounds for base_layer_count={self.base_layer_count}"
                )
        if self.plan_kind not in ("identity", "explicit", "rys"):
            raise ValueError(f"Unsupported plan_kind: {self.plan_kind}")

    @classmethod
    def identity(cls, base_layer_count: int) -> "LayerExecutionPlan":
        """Construct the identity execution plan."""
        return cls(
            base_layer_count=base_layer_count,
            layer_indices=tuple(range(base_layer_count)),
            plan_kind="identity",
            label="identity",
        )

    @classmethod
    def from_indices(
        cls,
        base_layer_count: int,
        layer_indices: tuple[int, ...] | list[int],
        *,
        label: str | None = None,
    ) -> "LayerExecutionPlan":
        """Construct an explicit execution plan from layer indices."""
        indices = tuple(int(idx) for idx in layer_indices)
        return cls(
            base_layer_count=base_layer_count,
            layer_indices=indices,
            plan_kind="explicit",
            label=label,
        )

    @classmethod
    def from_rys(
        cls,
        base_layer_count: int,
        start: int,
        end: int,
        *,
        label: str | None = None,
    ) -> "LayerExecutionPlan":
        """Construct a Repeat-Your-Self execution plan."""
        if base_layer_count <= 0:
            raise ValueError("base_layer_count must be > 0")
        if start < 0 or start >= base_layer_count:
            raise ValueError(
                f"start must be in [0, {base_layer_count - 1}] for base_layer_count={base_layer_count}"
            )
        if end <= start or end > base_layer_count:
            raise ValueError(
                f"end must satisfy {start} < end <= {base_layer_count}, got {end}"
            )

        indices = tuple(range(end)) + tuple(range(start, base_layer_count))
        return cls(
            base_layer_count=base_layer_count,
            layer_indices=indices,
            plan_kind="rys",
            label=label or f"rys_{start}_{end}",
        )

    @property
    def execution_layer_count(self) -> int:
        """Number of execution steps in the plan."""
        return len(self.layer_indices)

    def to_dict(self) -> dict[str, int | str | None | list[int]]:
        """Return a JSON-friendly representation."""
        return {
            "baseLayerCount": self.base_layer_count,
            "layerIndices": list(self.layer_indices),
            "planKind": self.plan_kind,
            "label": self.label,
        }


__all__ = ["LayerExecutionPlan", "PlanKind"]
