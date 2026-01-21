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

"""
Knowledge Encoder - Compute and project weight deltas for continual learning.

This module computes weight updates that encode new knowledge and projects
them into the null-space to avoid catastrophic forgetting. It is the core
algorithm for inference-time adaptation.

The key insight: The same null-space projection used for model merging can
be used for continual learning. Instead of merging knowledge from another
model, we encode knowledge from surprising events during inference.

Algorithm:
    1. Receive surprise event with context and target token
    2. Compute gradient-like signal (what weight change would help?)
    3. Project delta into null-space (don't interfere with existing knowledge)
    4. Apply update via UpdateStrategy (direct or LoRA accumulate)

Weight update strategies:
1. **Embedding update**: Add new token-concept associations
2. **MLP update**: Strengthen concept-concept connections
3. **Attention update**: Adjust attention patterns (careful - affects everything)

The encoder selects target layers from null-space capacity, ensuring
updates only occur where the model has available geometric room.

Update Routing (via UpdateStrategy):
- **DirectWeightStrategy**: Original behavior - immediate weight modification
- **LoRAAccumulateStrategy**: Buffer updates for later LoRA training (two-tier memory)

Math:
    delta_ideal = gradient(loss, weights)  # What we WANT to change
    delta_safe = P_null @ delta_ideal       # What we CAN change safely
    weights_new = weights + lr * delta_safe  # Or accumulated to LoRA

References:
    - GNSP: Gradient Null Space Projection (arXiv:2507.19839)
    - PNSP: Primary Null Space Projection (ScienceDirect 2024)
    - Titans: Learning to Memorize at Test Time (arXiv:2501.00663)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker
from modelcypher.core.domain.continual.surprise_detector import SurpriseEvent
from modelcypher.core.domain.continual.update_strategy import (
    DirectWeightStrategy,
    UpdateStrategy,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class EncodingResult:
    """Result of a knowledge encoding operation.

    Attributes:
        layer_id: Layer that was updated.
        weight_name: Name of the weight matrix updated.
        delta_norm_before: Norm of ideal delta (before projection).
        delta_norm_after: Norm of safe delta (after projection).
        preserved_fraction: Fraction of delta preserved (after/before).
        null_space_rank: Available null-space dimensions.
        learning_rate: Learning rate used.
        applied: Whether the update was actually applied.
    """

    layer_id: int
    weight_name: str
    delta_norm_before: float
    delta_norm_after: float
    preserved_fraction: float
    null_space_rank: int
    learning_rate: float
    applied: bool

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "layer_id": self.layer_id,
            "weight_name": self.weight_name,
            "delta_norm_before": self.delta_norm_before,
            "delta_norm_after": self.delta_norm_after,
            "preserved_fraction": self.preserved_fraction,
            "null_space_rank": self.null_space_rank,
            "learning_rate": self.learning_rate,
            "applied": self.applied,
        }


class KnowledgeEncoder:
    """Encodes new knowledge via null-space projected weight updates.

    The encoder works with a NullSpaceTracker to ensure updates don't
    interfere with existing capabilities.

    Usage:
        encoder = KnowledgeEncoder(model, tracker)

        for surprise_event in surprise_stream:
            # Caller decides encoding based on raw metrics
            if should_encode(surprise_event):  # Application-specific decision
                results = encoder.encode(
                    event=surprise_event,
                    hidden_state=current_hidden,
                )

                for result in results:
                    print(f"Encoded to {result.weight_name}: "
                          f"{result.preserved_fraction:.1%} preserved")
    """

    def __init__(
        self,
        model: Any,
        null_space_tracker: NullSpaceTracker,
        backend: "Backend",
        update_strategy: UpdateStrategy | None = None,
    ) -> None:
        """Initialize the knowledge encoder.

        Args:
            model: The model to update (must expose layers for modification).
            null_space_tracker: Tracker for null-space availability.
            backend: Compute backend.
            update_strategy: Strategy for applying weight updates. Defaults to
                DirectWeightStrategy (immediate weight modification). Use
                LoRAAccumulateStrategy for two-tier memory accumulation.
        """
        self._backend = backend
        self._model = model
        self._tracker = null_space_tracker

        # Initialize update strategy (default: direct weight modification)
        if update_strategy is None:
            self._update_strategy: UpdateStrategy = DirectWeightStrategy(
                model=model,
                backend=self._backend,
            )
        else:
            self._update_strategy = update_strategy

        # Derive learning rate from machine precision
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        eps = machine_epsilon(self._backend, self._backend.array([1.0]))
        # sqrt(eps) is the precision limit for meaningful updates
        self._learning_rate = eps ** 0.5

        # Minimum preserved fraction derived from precision
        # Updates below sqrt(eps) are numerically meaningless
        eps = machine_epsilon(self._backend, self._backend.array([1.0]))
        self._min_preserved_fraction = eps ** 0.5

        # Track encoding history
        self._encoding_count = 0
        self._total_preserved = 0.0

    def encode(
        self,
        event: SurpriseEvent,
        hidden_state: Array,
    ) -> list[EncodingResult]:
        """Encode knowledge from a surprise event.

        Args:
            event: The surprise event to encode.
            hidden_state: Hidden state at the surprise point.

        Returns:
            List of EncodingResults, one per updated weight.
        """
        # Determine target layers based on null-space availability
        target_layers = self._get_target_layers()

        results = []

        for layer_id in target_layers:
            # Encode to MLP weights (primary knowledge storage)
            mlp_result = self._encode_to_mlp(
                layer_id=layer_id,
                event=event,
                hidden_state=hidden_state,
            )
            if mlp_result is not None:
                results.append(mlp_result)

        return results

    def _get_target_layers(self) -> list[int]:
        """Get target layers based on null-space availability."""
        n_layers = self._tracker.n_layers

        target_layers: list[int] = []
        for layer_id in range(n_layers):
            state = self._tracker.get_layer_state(layer_id)
            if state.null_rank > 0:
                target_layers.append(layer_id)

        return target_layers

    def _encode_to_mlp(
        self,
        layer_id: int,
        event: SurpriseEvent,
        hidden_state: Array,
    ) -> EncodingResult | None:
        """Encode knowledge to MLP weights via null-space projection.

        Targets the up_proj or gate_proj weights of the MLP.
        """
        b = self._backend

        # Get the layer
        layer = self._get_layer(layer_id)
        if layer is None:
            return None

        # Get MLP weights (try different naming conventions)
        mlp_weights, weight_name = self._get_mlp_weights(layer)
        if mlp_weights is None:
            return None

        # Compute ideal delta (gradient-like signal)
        # We want the model to predict the correct token with higher probability
        # This is a simplified outer-product update
        delta_ideal, transformed_hidden = self._compute_mlp_delta(
            hidden_state=hidden_state,
            target_token_id=event.token_id,
            weights=mlp_weights,
        )

        # Compute BEHAVIORAL norms (actual output change, not weight mass)
        # Use transformed_hidden to ensure shape matches weight dimensions
        delta_norm_before = self._compute_behavioral_norm(delta_ideal, transformed_hidden)

        # Project to null-space
        delta_safe = self._tracker.project_to_null_space(
            layer_id=layer_id,
            delta=delta_ideal,
            use_variance_weighting=True,
        )

        if delta_safe is None:
            # Null-space not ready yet
            return EncodingResult(
                layer_id=layer_id,
                weight_name=weight_name,
                delta_norm_before=delta_norm_before,
                delta_norm_after=0.0,
                preserved_fraction=0.0,
                null_space_rank=0,
                learning_rate=self._learning_rate,
                applied=False,
            )

        # Compute BEHAVIORAL norm after projection
        # Use transformed_hidden to ensure shape matches weight dimensions
        delta_norm_after = self._compute_behavioral_norm(delta_safe, transformed_hidden)

        # Preserved fraction based on BEHAVIORAL impact, not weight magnitude
        if delta_norm_before > 0:
            preserved_fraction = delta_norm_after / delta_norm_before
        else:
            preserved_fraction = 0.0

        # Get null-space state
        null_state = self._tracker.get_layer_state(layer_id)

        # Check if enough of the update survived
        if preserved_fraction < self._min_preserved_fraction:
            return EncodingResult(
                layer_id=layer_id,
                weight_name=weight_name,
                delta_norm_before=delta_norm_before,
                delta_norm_after=delta_norm_after,
                preserved_fraction=preserved_fraction,
                null_space_rank=null_state.null_rank,
                learning_rate=self._learning_rate,
                applied=False,
            )

        # Apply the update via strategy (direct or LoRA accumulate)
        applied = self._apply_weight_update(
            layer=layer,
            layer_id=layer_id,
            weight_name=weight_name,
            delta=delta_safe,
            hidden_state=transformed_hidden,
        )

        if not applied:
            return EncodingResult(
                layer_id=layer_id,
                weight_name=weight_name,
                delta_norm_before=delta_norm_before,
                delta_norm_after=delta_norm_after,
                preserved_fraction=preserved_fraction,
                null_space_rank=null_state.null_rank,
                learning_rate=self._learning_rate,
                applied=False,
            )

        # Track statistics
        self._encoding_count += 1
        self._total_preserved += preserved_fraction

        return EncodingResult(
            layer_id=layer_id,
            weight_name=weight_name,
            delta_norm_before=delta_norm_before,
            delta_norm_after=delta_norm_after,
            preserved_fraction=preserved_fraction,
            null_space_rank=null_state.null_rank,
            learning_rate=self._learning_rate,
            applied=True,
        )

    def _compute_mlp_delta(
        self,
        hidden_state: Array,
        target_token_id: int,
        weights: Array,
    ) -> tuple[Array, Array]:
        """Compute MLP weight delta that would encode the target.

        The delta is shaped to match the weight matrix exactly.
        Uses a projection-based update that respects the weight geometry.

        For MLP weights [out_dim, in_dim]:
            delta = lr * (target_direction @ hidden_state.T)

        where target_direction is projected to match out_dim.

        Returns:
            Tuple of (delta, transformed_hidden_state) where both are
            shaped to match weight dimensions. The transformed hidden
            state should be used for behavioral norm computation.
        """
        b = self._backend

        # Get weight shape
        out_dim = int(weights.shape[0])
        in_dim = int(weights.shape[1])

        # Flatten hidden state to [in_dim]
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        # Ensure hidden_state matches in_dim
        hidden_dim = int(hidden_state.shape[0])
        if hidden_dim != in_dim:
            # Project or pad to match
            if hidden_dim > in_dim:
                hidden_state = b.take(
                    hidden_state, b.arange(in_dim), axis=0
                )
            else:
                # Pad with zeros
                padding = b.zeros((in_dim - hidden_dim,))
                hidden_state = b.concatenate([hidden_state, padding], axis=0)

        # Save transformed hidden state for behavioral norm
        transformed_hidden = hidden_state

        # Get target embedding for direction
        target_embed = self._get_token_embedding(target_token_id)

        if target_embed is None:
            # Use hidden state as target direction (self-reinforcement)
            target_direction = hidden_state
        else:
            target_direction = target_embed

        # Project target_direction to out_dim
        target_dim = int(target_direction.shape[0])
        if target_dim != out_dim:
            if target_dim > out_dim:
                target_direction = b.take(
                    target_direction, b.arange(out_dim), axis=0
                )
            else:
                # Project via linear interpolation
                # This maintains structure while changing dimension
                indices = b.arange(out_dim) * target_dim // out_dim
                target_direction = b.take(target_direction, indices, axis=0)

        # Outer product: [out_dim, in_dim] - matches weight shape exactly
        delta = b.matmul(target_direction[:, None], hidden_state[None, :])

        # Scale by learning rate
        delta = delta * self._learning_rate

        b.eval(delta, transformed_hidden)
        return delta, transformed_hidden

    def _compute_frobenius_norm(self, tensor: Array) -> float:
        """Compute Frobenius norm of a tensor."""
        b = self._backend
        norm_sq = b.sum(tensor * tensor)
        b.eval(norm_sq)
        return float(b.to_scalar(norm_sq)) ** 0.5

    def _compute_behavioral_norm(
        self,
        delta_weight: Array,
        input_activations: Array,
    ) -> float:
        """Compute behavioral norm of a weight delta.

        Behavioral norm measures actual output change:
            ||A @ delta_W.T||_F

        where A is input activations. This is the TRUE measure of
        how much the weight change affects model behavior.

        After null-space projection:
        - Frobenius might say "47% preserved" (weight mass)
        - Behavioral shows "0.0002% preserved" (actual impact)

        The behavioral norm is what we care about.
        """
        b = self._backend

        # Ensure input_activations is 2D [n_samples, in_dim]
        if input_activations.ndim == 1:
            input_activations = input_activations[None, :]

        # Output change: A @ delta_W.T
        # delta_weight is [out_dim, in_dim]
        # input_activations is [n_samples, in_dim]
        # result is [n_samples, out_dim]
        output_change = b.matmul(input_activations, b.transpose(delta_weight))

        # Frobenius norm of output change
        norm_sq = b.sum(output_change * output_change)
        b.eval(norm_sq)

        return float(b.to_scalar(norm_sq)) ** 0.5

    def _get_layer(self, layer_id: int) -> Any | None:
        """Get layer object from model."""
        # Handle wrapped models
        base_model = getattr(self._model, "model", self._model)
        layers = getattr(base_model, "layers", None)

        if layers is None or layer_id >= len(layers):
            return None

        return layers[layer_id]

    def _get_mlp_weights(self, layer: Any) -> tuple[Array | None, str]:
        """Get MLP weights from a layer.

        Tries various naming conventions used by different architectures.
        """
        # Try common MLP weight names
        weight_names = [
            ("mlp.up_proj", "weight"),
            ("mlp.gate_proj", "weight"),
            ("mlp.fc1", "weight"),
            ("feed_forward.w1", None),
            ("mlp.dense_h_to_4h", "weight"),
        ]

        for attr_path, weight_attr in weight_names:
            obj = layer
            found = True

            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    found = False
                    break

            if found:
                if weight_attr:
                    weights = getattr(obj, weight_attr, None)
                else:
                    # Try to get .weight from a Linear-like layer
                    weights = getattr(obj, "weight", obj)

                if weights is not None:
                    return weights, attr_path

        return None, ""

    def _get_token_embedding(self, token_id: int) -> Array | None:
        """Get embedding for a token ID."""
        # Handle wrapped models
        base_model = getattr(self._model, "model", self._model)

        # Try common embedding attribute names
        embed_attrs = ["embed_tokens", "wte", "embedding", "token_embedding"]

        for attr in embed_attrs:
            embed_layer = getattr(base_model, attr, None)
            if embed_layer is not None:
                # Get the weight matrix
                weight = getattr(embed_layer, "weight", None)
                if weight is None:
                    # Some models have the weight directly
                    weight = embed_layer

                if weight is not None:
                    b = self._backend
                    # Index into embedding: [vocab_size, embed_dim]
                    embedding = b.take(weight, b.array([token_id]), axis=0)[0]
                    b.eval(embedding)
                    return embedding

        return None

    def _apply_weight_update(
        self,
        layer: Any,
        layer_id: int,
        weight_name: str,
        delta: "Array",
        hidden_state: "Array",
    ) -> bool:
        """Apply a weight update using the configured strategy.

        The update is routed through the UpdateStrategy, which determines
        where the delta goes:
        - DirectWeightStrategy: Immediate modification of model weights
        - LoRAAccumulateStrategy: Buffer for later LoRA training

        Args:
            layer: The transformer layer containing the weight.
            layer_id: Index of the layer.
            weight_name: Dot-separated path like "mlp.up_proj".
            delta: Weight delta to add, must match weight shape.
            hidden_state: Hidden state that produced this delta (for LoRA).

        Returns:
            True if update was applied/accumulated successfully.
        """
        result = self._update_strategy.apply_update(
            layer=layer,
            layer_id=layer_id,
            weight_name=weight_name,
            delta=delta,
            hidden_state=hidden_state,
        )
        return result.applied

    def get_stats(self) -> dict[str, Any]:
        """Get encoding statistics including strategy stats."""
        if self._encoding_count == 0:
            avg_preserved = 0.0
        else:
            avg_preserved = self._total_preserved / self._encoding_count

        stats: dict[str, Any] = {
            "encoding_count": self._encoding_count,
            "average_preserved_fraction": avg_preserved,
            "strategy": self._update_strategy.get_stats(),
        }
        return stats

    def reset_stats(self) -> None:
        """Reset encoding statistics."""
        self._encoding_count = 0
        self._total_preserved = 0.0

    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        return self._learning_rate

    @property
    def encoding_count(self) -> int:
        """Number of encodings performed."""
        return self._encoding_count

    @property
    def update_strategy(self) -> UpdateStrategy:
        """Current update strategy."""
        return self._update_strategy

    @property
    def strategy_name(self) -> str:
        """Name of the current update strategy."""
        return self._update_strategy.name
