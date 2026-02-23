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
Feature Steering for Activation Intervention.

Modifies model behavior by adding/subtracting directions in activation space
during inference. Supports two modes:

1. Direct steering: activation += direction * strength
2. Null-space constrained (AlphaSteer): Project steering into directions that
   don't affect core capability while achieving desired behavioral change.

Sources for steering directions:
- Contrastive pairs (harmful vs harmless)
- SAE feature directions
- Refusal direction (from refusal_direction_detector.py)
- Mean difference between any two concept sets

Key principles:
    - Geodesic geometry (not Euclidean)
    - No hardcoded thresholds (strength derived from activation scale)
    - Backend-agnostic

References:
    - "Activation Addition" (Turner et al., 2023)
    - "AlphaSteer" (2024) - Null-space constrained steering
    - "Representation Engineering" (Zou et al., 2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms, frechet_mean

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class SteeringSource(str, Enum):
    """Source of the steering direction."""

    contrastive = "contrastive"
    refusal = "refusal"
    mean_difference = "mean_difference"
    custom = "custom"


@dataclass(frozen=True)
class SteeringVector:
    """A direction for steering model behavior.

    Attributes
    ----------
    direction : Array
        Unit direction vector. Shape: [hidden_dim].
    layer : int
        Layer to apply steering at.
    source : SteeringSource
        How this direction was computed.
    label : str
        Human-readable label for this steering direction.
    strength_range : tuple[float, float]
        Recommended strength range (derived from data, not hardcoded).
    """

    direction: Any
    layer: int
    source: SteeringSource
    label: str = ""
    strength_range: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class SteeringConfig:
    """Configuration for steering during inference.

    Attributes
    ----------
    vectors : list[SteeringVector]
        Steering vectors to apply.
    strengths : list[float]
        Strength for each vector. Positive = add, negative = subtract.
    null_space_constrained : bool
        Whether to project steering into null space (AlphaSteer).
    position : int | None
        Token position to steer. None = all positions.
    """

    vectors: list[SteeringVector] = field(default_factory=list)
    strengths: list[float] = field(default_factory=list)
    null_space_constrained: bool = True
    position: int | None = None


@dataclass(frozen=True)
class SteeringResult:
    """Result of applying steering.

    Attributes
    ----------
    original_output : Any
        Output without steering.
    steered_output : Any
        Output with steering applied.
    applied_vectors : list[SteeringVector]
        Vectors that were applied.
    effective_strengths : list[float]
        Actual strengths used (may differ from requested if null-space
        constrained).
    projection_loss : float
        Fraction of steering lost to null-space projection (0-1).
    """

    original_output: Any
    steered_output: Any
    applied_vectors: list[SteeringVector]
    effective_strengths: list[float]
    projection_loss: float


class FeatureSteering:
    """Applies steering vectors during model inference.

    Example
    -------
    >>> steering = FeatureSteering(model)
    >>> refusal_vec = steering.extract_contrastive_direction(
    ...     positive_prompts=harmful_prompts,
    ...     negative_prompts=harmless_prompts,
    ...     layer=16,
    ... )
    >>> config = SteeringConfig(
    ...     vectors=[refusal_vec],
    ...     strengths=[-0.5],  # Reduce refusal
    ... )
    >>> result = steering.generate_with_steering(input_ids, config)
    """

    def __init__(self, model: Any, backend: "Backend | None" = None) -> None:
        """Initialize steering.

        Parameters
        ----------
        model : Any
            Model to steer.
        backend : Backend, optional
            Computation backend.
        """
        self._model = model
        self._backend = backend or get_default_backend()
        self._layers = self._get_layers()
        self._null_space_filter = GeodesicNullSpaceFilter(self._backend)

    def _get_layers(self) -> list[Any]:
        """Get model layers."""
        base_model = getattr(self._model, "model", self._model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise RuntimeError("Model does not expose transformer layers.")
        return layers

    def extract_contrastive_direction(
        self,
        positive_activations: Any,
        negative_activations: Any,
        layer: int,
        label: str = "contrastive",
    ) -> SteeringVector:
        """Extract steering direction from contrastive activations.

        Computes the direction from negative to positive concept centroid.

        Parameters
        ----------
        positive_activations : Array
            Activations for positive examples. Shape: [n_pos, hidden_dim].
        negative_activations : Array
            Activations for negative examples. Shape: [n_neg, hidden_dim].
        layer : int
            Layer these activations were captured from.
        label : str
            Label for this direction.

        Returns
        -------
        SteeringVector
            Contrastive steering direction.
        """
        b = self._backend
        pos = b.array(positive_activations) if not hasattr(positive_activations, "shape") else positive_activations
        neg = b.array(negative_activations) if not hasattr(negative_activations, "shape") else negative_activations
        b.eval(pos, neg)

        # Compute Fréchet means (geodesic centroids)
        pos_mean = frechet_mean(pos, backend=b)
        neg_mean = frechet_mean(neg, backend=b)
        b.eval(pos_mean, neg_mean)

        # Direction from negative to positive
        direction = pos_mean - neg_mean

        # Normalize to unit geodesic norm
        norm = geodesic_norms(b.reshape(direction, (1, -1)), b)
        b.eval(norm)
        norm_val = float(b.to_scalar(norm[0]))

        eps = regularization_epsilon(b, direction)
        if norm_val > eps:
            direction = direction / norm_val
        b.eval(direction)

        # Derive strength range from activation scale
        pos_norms = geodesic_norms(pos, b)
        neg_norms = geodesic_norms(neg, b)
        all_norms = b.concatenate([pos_norms, neg_norms], axis=0)
        mean_norm = b.mean(all_norms)
        b.eval(mean_norm)
        mean_norm_val = float(b.to_scalar(mean_norm))

        # Strength range: fraction of typical activation magnitude
        strength_range = (-mean_norm_val, mean_norm_val)

        return SteeringVector(
            direction=direction,
            layer=layer,
            source=SteeringSource.contrastive,
            label=label,
            strength_range=strength_range,
        )

    def extract_mean_difference_direction(
        self,
        concept_a_activations: Any,
        concept_b_activations: Any,
        layer: int,
        label: str = "mean_diff",
    ) -> SteeringVector:
        """Extract steering direction from mean difference between concepts.

        Parameters
        ----------
        concept_a_activations : Array
            Activations for concept A. Shape: [n_a, hidden_dim].
        concept_b_activations : Array
            Activations for concept B. Shape: [n_b, hidden_dim].
        layer : int
            Layer these activations were captured from.
        label : str
            Label for this direction.

        Returns
        -------
        SteeringVector
            Direction from concept B centroid to concept A centroid.
        """
        return self.extract_contrastive_direction(
            positive_activations=concept_a_activations,
            negative_activations=concept_b_activations,
            layer=layer,
            label=label,
        )._replace(source=SteeringSource.mean_difference)

    def project_to_null_space(
        self,
        steering_direction: Any,
        prior_activations: Any,
    ) -> tuple[Any, float]:
        """Project steering direction into null space of prior activations.

        This ensures steering doesn't interfere with core model capability.
        The AlphaSteer approach.

        Parameters
        ----------
        steering_direction : Array
            Direction to project. Shape: [hidden_dim].
        prior_activations : Array
            Activations defining the occupied space. Shape: [n, hidden_dim].

        Returns
        -------
        tuple[Array, float]
            Projected direction and fraction lost to projection.
        """
        b = self._backend
        direction = b.array(steering_direction) if not hasattr(steering_direction, "shape") else steering_direction
        direction = b.reshape(direction, (1, -1))
        b.eval(direction)

        result = self._null_space_filter.filter_delta(
            direction,
            prior_activations,
            delta_space="activations",
        )

        projected = b.reshape(result.filtered_delta, (-1,))
        b.eval(projected)

        projection_loss = result.projection_loss

        return projected, projection_loss

    def apply_steering(
        self,
        input_ids: Any,
        config: SteeringConfig,
        prior_activations: Any | None = None,
    ) -> SteeringResult:
        """Apply steering during inference.

        Parameters
        ----------
        input_ids : Array
            Input token IDs.
        config : SteeringConfig
            Steering configuration.
        prior_activations : Array, optional
            Activations for null-space projection. Required if
            null_space_constrained=True.

        Returns
        -------
        SteeringResult
            Result with original and steered outputs.
        """
        b = self._backend
        input_ids = b.array(input_ids) if not hasattr(input_ids, "shape") else input_ids
        b.eval(input_ids)

        if len(config.vectors) != len(config.strengths):
            raise ValueError("Must have same number of vectors and strengths")

        # Run original forward pass
        original_output = self._model(input_ids)
        b.eval(original_output)

        # Prepare steering vectors
        effective_strengths = list(config.strengths)
        projection_loss = 0.0

        # Group vectors by layer
        layer_to_vectors: dict[int, list[tuple[Any, float]]] = {}
        for i, vec in enumerate(config.vectors):
            direction = vec.direction
            strength = config.strengths[i]

            # Project to null space if requested
            if config.null_space_constrained and prior_activations is not None:
                direction, loss = self.project_to_null_space(
                    direction, prior_activations
                )
                projection_loss = max(projection_loss, loss)
                # Adjust effective strength based on what remains
                effective_strengths[i] = strength * (1.0 - loss)

            if vec.layer not in layer_to_vectors:
                layer_to_vectors[vec.layer] = []
            layer_to_vectors[vec.layer].append((direction, strength))

        # Create steering callback
        def steering_callback(layer_idx: int, output: Any) -> Any:
            if layer_idx not in layer_to_vectors:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Apply all steering vectors for this layer
            for direction, strength in layer_to_vectors[layer_idx]:
                hidden = self._add_steering(
                    hidden, direction, strength, config.position
                )
            b.eval(hidden)

            if rest is not None:
                return (hidden,) + rest
            return hidden

        # Run steered forward pass
        target_layers = set(layer_to_vectors.keys())
        with _SteeringContext(self._layers, steering_callback, target_layers):
            steered_output = self._model(input_ids)
            b.eval(steered_output)

        return SteeringResult(
            original_output=original_output,
            steered_output=steered_output,
            applied_vectors=list(config.vectors),
            effective_strengths=effective_strengths,
            projection_loss=projection_loss,
        )

    def _add_steering(
        self,
        hidden: Any,
        direction: Any,
        strength: float,
        position: int | None,
    ) -> Any:
        """Add steering direction to hidden states."""
        b = self._backend
        hidden_shape = b.shape(hidden)

        # Prepare direction for broadcasting
        if len(hidden_shape) == 3:
            # [batch, seq, hidden]
            direction_broadcast = b.reshape(direction, (1, 1, -1))
        elif len(hidden_shape) == 2:
            # [seq, hidden]
            direction_broadcast = b.reshape(direction, (1, -1))
        else:
            direction_broadcast = direction

        if position is not None:
            # Only steer at specific position
            if len(hidden_shape) == 3:
                seq_len = int(hidden_shape[1])
                pos = position if position >= 0 else seq_len + position
                indices = b.arange(seq_len)
                mask = b.reshape(indices == pos, (1, seq_len, 1))
            elif len(hidden_shape) == 2:
                seq_len = int(hidden_shape[0])
                pos = position if position >= 0 else seq_len + position
                indices = b.arange(seq_len)
                mask = b.reshape(indices == pos, (seq_len, 1))
            else:
                mask = b.ones_like(hidden)

            steering = strength * direction_broadcast * b.astype(mask, "float32")
        else:
            steering = strength * direction_broadcast

        result = hidden + steering
        b.eval(result)
        return result


class _SteeringContext:
    """Context manager for applying steering during forward pass."""

    def __init__(
        self,
        layers: list[Any],
        steering_fn: Callable[[int, Any], Any],
        target_layers: set[int],
    ) -> None:
        self._layers = layers
        self._steering_fn = steering_fn
        self._target_layers = target_layers
        self._original: list[Any] | None = None

    def __enter__(self) -> "_SteeringContext":
        self._original = list(self._layers)
        wrapped = []
        for idx, layer in enumerate(self._layers):
            if idx in self._target_layers:
                wrapped.append(_SteeringWrapper(layer, idx, self._steering_fn))
            else:
                wrapped.append(layer)
        self._layers[:] = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original is not None:
            self._layers[:] = self._original


class _SteeringWrapper:
    """Wrapper that applies steering to layer output."""

    __slots__ = ("_layer", "_idx", "_steering_fn")

    def __init__(
        self,
        layer: Any,
        idx: int,
        steering_fn: Callable[[int, Any], Any],
    ) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_steering_fn", steering_fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._layer(*args, **kwargs)
        return self._steering_fn(self._idx, output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


__all__ = [
    "SteeringSource",
    "SteeringVector",
    "SteeringConfig",
    "SteeringResult",
    "FeatureSteering",
]
