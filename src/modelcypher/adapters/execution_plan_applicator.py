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

"""Execution-plan adapters for MLX-first layer-routing experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from modelcypher.core.domain.geometry.model_utils import resolve_model_base
from modelcypher.core.domain.inference import LayerExecutionPlan


@contextmanager
def apply_execution_plan(model: Any, plan: LayerExecutionPlan) -> Iterator[Any]:
    """Apply a validated execution plan to a mutable MLX-style layer list."""
    base = resolve_model_base(model)
    if not hasattr(base, "layers"):
        raise NotImplementedError("Resolved backbone does not expose a layers attribute")

    original_layers = getattr(base, "layers")
    if not isinstance(original_layers, list):
        raise NotImplementedError(
            "Execution-plan routing currently requires base.layers to be a mutable Python list"
        )
    if len(original_layers) != plan.base_layer_count:
        raise ValueError(
            "Execution plan base_layer_count does not match the loaded model "
            f"({plan.base_layer_count} != {len(original_layers)})"
        )

    planned_layers = [original_layers[layer_idx] for layer_idx in plan.layer_indices]
    base.layers = planned_layers
    try:
        yield model
    finally:
        base.layers = original_layers


__all__ = ["apply_execution_plan"]
