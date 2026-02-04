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

"""Update Strategy - Strategy pattern for weight update routing.

This module defines the UpdateStrategy protocol for routing weight updates
to different targets:

1. **DirectWeightStrategy**: Immediate weight modification (original behavior)
2. **LoRAAccumulateStrategy**: Buffer updates for later LoRA training

The strategy pattern enables KnowledgeEncoder to be agnostic about where
updates go, allowing the same consolidation pipeline to either:
- Directly modify weights (fast, ephemeral)
- Accumulate to LoRA (persistent, mergeable)

This is the foundation for two-tier memory:
- Hippocampus (LoRA): Fast binding, session-level learning
- Neocortex (Base Weights): Slow consolidation via null-space merge

Usage:
    # Direct weight updates (original behavior)
    encoder = KnowledgeEncoder(
        model=model,
        null_space_tracker=tracker,
        update_strategy=DirectWeightStrategy(model),
    )

    # LoRA accumulation (two-tier memory)
    lora_store = LoRAMemoryStore(agent_id="agent-001", base_model_path=path)
    encoder = KnowledgeEncoder(
        model=model,
        null_space_tracker=tracker,
        update_strategy=LoRAAccumulateStrategy(lora_store),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class UpdateResult:
    """Result of applying a weight update via a strategy.

    Attributes
    ----------
    layer_id : int
        Layer that was updated.
    weight_name : str
        Name of the weight matrix (e.g., "mlp.up_proj").
    applied : bool
        Whether the update was successfully applied/accumulated.
    strategy_name : str
        Name of the strategy that handled this update.
    metadata : dict
        Strategy-specific metadata.
    """

    layer_id: int
    weight_name: str
    applied: bool
    strategy_name: str
    metadata: dict[str, Any]


@runtime_checkable
class UpdateStrategy(Protocol):
    """Protocol for routing weight updates.

    Implementations determine where weight deltas go:
    - DirectWeightStrategy: Immediate application to model weights
    - LoRAAccumulateStrategy: Buffer for later LoRA training

    The strategy receives the projected delta (already null-space safe)
    and determines how to apply it.
    """

    @property
    def name(self) -> str:
        """Return the strategy name."""
        ...

    def apply_update(
        self,
        layer: Any,
        layer_id: int,
        weight_name: str,
        delta: "Array",
        hidden_state: "Array",
    ) -> UpdateResult:
        """Apply a weight update using this strategy.

        Parameters
        ----------
        layer : Any
            The transformer layer object.
        layer_id : int
            Index of the layer.
        weight_name : str
            Dot-separated path to weight (e.g., "mlp.up_proj").
        delta : Array
            Weight delta to apply, already null-space projected.
        hidden_state : Array
            Hidden state that produced this delta (for LoRA training).

        Returns
        -------
        UpdateResult
            Result of the update operation.
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get strategy-specific statistics."""
        ...


class DirectWeightStrategy:
    """Apply weight updates directly to model weights.

    This is the original KnowledgeEncoder behavior, extracted into a strategy.
    Updates are applied immediately by modifying the weight matrix in-place.

    Parameters
    ----------
    model : Any
        The model whose weights will be modified.
    backend : Backend, optional
        Compute backend.
    """

    def __init__(
        self,
        model: Any,
        backend: "Backend | None" = None,
    ) -> None:
        self._model = model
        self._backend = backend or get_default_backend()
        self._update_count = 0

    @property
    def name(self) -> str:
        return "DirectWeight"

    def apply_update(
        self,
        layer: Any,
        layer_id: int,
        weight_name: str,
        delta: "Array",
        hidden_state: "Array",
    ) -> UpdateResult:
        """Apply delta directly to model weights."""
        b = self._backend

        # Navigate to the weight holder (e.g., layer.mlp.up_proj)
        obj = layer
        attrs = weight_name.split(".")

        for attr in attrs:
            obj = getattr(obj, attr, None)
            if obj is None:
                return UpdateResult(
                    layer_id=layer_id,
                    weight_name=weight_name,
                    applied=False,
                    strategy_name=self.name,
                    metadata={"error": f"Path not found: {weight_name}"},
                )

        weight_holder = obj

        # Get current weight
        current_weight = getattr(weight_holder, "weight", None)
        if current_weight is None:
            return UpdateResult(
                layer_id=layer_id,
                weight_name=weight_name,
                applied=False,
                strategy_name=self.name,
                metadata={"error": "No weight attribute found"},
            )

        # Verify shape match
        if current_weight.shape != delta.shape:
            return UpdateResult(
                layer_id=layer_id,
                weight_name=weight_name,
                applied=False,
                strategy_name=self.name,
                metadata={
                    "error": "Shape mismatch",
                    "weight_shape": current_weight.shape,
                    "delta_shape": delta.shape,
                },
            )

        # Apply update
        new_weight = current_weight + delta
        b.eval(new_weight)

        # Set new weight (some backends use frozen arrays, so replace whole thing)
        setattr(weight_holder, "weight", new_weight)

        self._update_count += 1

        return UpdateResult(
            layer_id=layer_id,
            weight_name=weight_name,
            applied=True,
            strategy_name=self.name,
            metadata={"update_count": self._update_count},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics."""
        return {
            "strategy": self.name,
            "update_count": self._update_count,
        }


class LoRAAccumulateStrategy:
    """Accumulate weight updates for later LoRA training.

    Instead of modifying weights directly, this strategy buffers
    (hidden_state, delta) pairs for batch LoRA training. The deltas
    are used to derive target embeddings for the LoRA adapter.

    This implements the "hippocampus" part of two-tier memory:
    - Fast: Just appends to a buffer
    - Ephemeral: Buffer is cleared after LoRA training
    - Trainable: Accumulated pairs become LoRA training data

    Parameters
    ----------
    accumulator : Any
        Object with `accumulate(hidden_state, delta, layer_id, weight_name)` method.
        Typically a LoRAMemoryStore instance.
    backend : Backend, optional
        Compute backend.
    """

    def __init__(
        self,
        accumulator: Any,
        backend: "Backend | None" = None,
    ) -> None:
        self._accumulator = accumulator
        self._backend = backend or get_default_backend()
        self._accumulate_count = 0

    @property
    def name(self) -> str:
        return "LoRAAccumulate"

    def apply_update(
        self,
        layer: Any,
        layer_id: int,
        weight_name: str,
        delta: "Array",
        hidden_state: "Array",
    ) -> UpdateResult:
        """Accumulate update for later LoRA training.

        The hidden_state becomes the "input" and delta provides the
        "target direction" for LoRA training.
        """
        b = self._backend

        # Compute target from delta
        # For MLP weights [out_dim, in_dim], target = (W + delta) @ hidden_state
        # But we can use the delta directly as the direction to learn
        # The accumulator will handle the actual training data format

        # Call accumulator's accumulate method
        # Expected signature: accumulate(hidden_state, delta, layer_id, weight_name)
        try:
            result = self._accumulator.accumulate(
                hidden_state=hidden_state,
                delta=delta,
                layer_id=layer_id,
                weight_name=weight_name,
            )
            self._accumulate_count += 1

            return UpdateResult(
                layer_id=layer_id,
                weight_name=weight_name,
                applied=True,
                strategy_name=self.name,
                metadata={
                    "accumulate_count": self._accumulate_count,
                    "buffer_size": getattr(self._accumulator, "buffer_size", None),
                },
            )

        except Exception as e:
            return UpdateResult(
                layer_id=layer_id,
                weight_name=weight_name,
                applied=False,
                strategy_name=self.name,
                metadata={"error": str(e)},
            )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics."""
        return {
            "strategy": self.name,
            "accumulate_count": self._accumulate_count,
            "buffer_size": getattr(self._accumulator, "buffer_size", None),
        }
