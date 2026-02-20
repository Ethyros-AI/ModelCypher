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

"""Domain types for experimental multi-adapter routing measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterIdentity:
    """Identity and geometry metadata for one adapter."""

    id: str
    path: str
    rank: int
    scale: float
    module_keys: tuple[str, ...]


@dataclass(frozen=True)
class LayerRoutingMeasurement:
    """Per-layer divergence signals for one adapter relative to base."""

    layer_index: int
    adapter_id: str
    kl_divergence: float
    activation_norm_ratio: float
    cosine_similarity: float


@dataclass(frozen=True)
class LayerRoutingSnapshot:
    """All adapter measurements for a single layer."""

    layer_index: int
    measurements: tuple[LayerRoutingMeasurement, ...]
    base_activation_norm: float


@dataclass(frozen=True)
class RoutingTrace:
    """End-to-end routing measurements for one prompt."""

    prompt: str
    n_adapters: int
    n_layers: int
    layer_snapshots: tuple[LayerRoutingSnapshot, ...]
    adapter_identities: tuple[AdapterIdentity, ...]
    selected_adapter_per_layer: dict[int, str]
    selection_method: str
    output_kl_vs_single: dict[str, float] | None


@dataclass
class AdapterPool:
    """Loaded base model and adapter variants for routing experiments."""

    base_model: Any
    base_tokenizer: Any
    base_model_path: str
    adapter_models: dict[str, tuple[Any, Any]]
    adapter_identities: tuple[AdapterIdentity, ...]


__all__ = [
    "AdapterIdentity",
    "LayerRoutingMeasurement",
    "LayerRoutingSnapshot",
    "RoutingTrace",
    "AdapterPool",
]

