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
Confidence Embedding - Embed entropy state into the residual stream.

This module creates a learnable embedding that encodes the model's current
entropy/confidence state. When added to the residual stream, it allows the
model to "see" its own uncertainty and potentially respond to it.

The key insight: Models already have positional embeddings (where am I in
the sequence?) and token embeddings (what token is this?). We add a third
type: confidence embeddings (how uncertain am I?).

Architecture:
    Input: EntropyState (entropy, derivative, acceleration, variance)
    Output: [hidden_dim] embedding to add to residual stream

The embedding uses a small MLP to project the low-dimensional entropy state
into the high-dimensional hidden space. The projection is learned to place
confidence information in directions the model can attend to.

Math:
    features = [H, H_norm, dH/dt, d²H/dt², var]  # [5]
    embedding = W2 @ tanh(W1 @ features + b1) + b2  # [hidden_dim]
    residual' = residual + scale * embedding

The scale factor controls how strongly confidence affects generation.
It can be learned or set to a fixed value.

References:
    - Emergent Introspective Awareness (Anthropic, 2025)
    - LLMs Have Intrinsic Meta-Cognition (arXiv:2506.08410)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.entropy_analyzer import EntropyState

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class ConfidenceEmbedding:
    """Creates embeddings from entropy state for residual stream injection.

    The embedding encodes uncertainty information into the model's hidden space,
    allowing the model to attend to its own confidence state.

    Usage:
        embedding = ConfidenceEmbedding(hidden_dim)

        for step in generation:
            entropy_state = analyzer.analyze(logits)
            conf_embed = embedding.encode(entropy_state)

            # Inject into residual stream before next forward pass
            hidden_state = hidden_state + conf_embed

    The embedding can be trained via:
    1. End-to-end backprop through the generation loop
    2. Supervised learning from labeled confidence-action pairs
    3. Contrastive learning (similar confidence -> similar embedding)
    """

    # Number of input features from entropy state
    _NUM_FEATURES = 5

    def __init__(
        self,
        hidden_dim: int,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the confidence embedding.

        Args:
            hidden_dim: Target hidden dimension (must match model).
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._hidden_dim = hidden_dim

        # Derive parameters from geometry if not set
        self._intermediate_dim = self._derive_intermediate_dim()
        self._scale_value = self._derive_scale()

        self._init_weights()

    def _derive_intermediate_dim(self) -> int:
        """Derive intermediate_dim from hidden_dim if not set.

        Uses sqrt(hidden_dim) as the natural compression scale.
        """
        dim = int(self._hidden_dim ** 0.5)
        return max(1, dim)

    def _derive_scale(self) -> float:
        """Derive scale from hidden_dim if not set.

        Uses 1/sqrt(hidden_dim) so the embedding contribution has unit norm
        in expectation. This follows from Xavier/He initialization theory:
        for a d-dimensional embedding, 1/sqrt(d) maintains variance.
        """
        # 1/sqrt(hidden_dim) maintains unit contribution in expectation
        return 1.0 / (self._hidden_dim ** 0.5)

    def _init_weights(self) -> None:
        """Initialize MLP weights using Xavier initialization."""
        b = self._backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        # Layer 1: [num_features, intermediate_dim]
        std1 = sqrt_scalar(2.0 / (self._NUM_FEATURES + self._intermediate_dim), b)
        self._w1 = b.random_normal((self._NUM_FEATURES, self._intermediate_dim)) * std1
        self._b1 = b.zeros((self._intermediate_dim,))

        # Layer 2: [intermediate_dim, hidden_dim]
        std2 = sqrt_scalar(
            2.0 / (self._intermediate_dim + self._hidden_dim), b
        )
        self._w2 = b.random_normal((self._intermediate_dim, self._hidden_dim)) * std2
        self._b2 = b.zeros((self._hidden_dim,))

        # Learnable scale (initialized to derived value)
        self._scale = b.array([self._scale_value])

        b.eval(self._w1, self._b1, self._w2, self._b2, self._scale)

    def encode(self, entropy_state: EntropyState) -> Array:
        """Encode entropy state into hidden-dimension embedding.

        Args:
            entropy_state: Current entropy analysis.

        Returns:
            Embedding tensor [hidden_dim] ready for residual injection.
        """
        b = self._backend

        # Build feature vector
        features = b.array(
            [
                entropy_state.entropy,
                entropy_state.entropy_normalized,
                entropy_state.entropy_derivative,
                entropy_state.entropy_acceleration,
                entropy_state.logit_variance,
            ]
        )

        # Forward pass through MLP
        # Layer 1
        hidden = b.matmul(features[None, :], self._w1)[0] + self._b1

        # Activation
        if self._config.use_tanh:
            # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            exp_2x = b.exp(2.0 * hidden)
            hidden = (exp_2x - 1.0) / (exp_2x + 1.0)
        else:
            hidden = b.maximum(hidden, b.zeros_like(hidden))  # ReLU

        # Layer 2
        embedding = b.matmul(hidden[None, :], self._w2)[0] + self._b2

        # Apply scale
        embedding = embedding * self._scale

        b.eval(embedding)
        return embedding

    def encode_batch(self, entropy_states: list[EntropyState]) -> Array:
        """Encode batch of entropy states.

        Args:
            entropy_states: List of entropy states.

        Returns:
            Stacked embeddings [batch, hidden_dim].
        """
        embeddings = [self.encode(state) for state in entropy_states]
        return self._backend.stack(embeddings, axis=0)

    def inject_into_residual(
        self,
        hidden_state: Array,
        entropy_state: EntropyState,
    ) -> Array:
        """Convenience method to encode and inject in one step.

        Args:
            hidden_state: Current hidden state [batch, seq, hidden] or [seq, hidden].
            entropy_state: Current entropy analysis.

        Returns:
            Modified hidden state with confidence embedding added.
        """
        b = self._backend
        embedding = self.encode(entropy_state)

        # Handle different input shapes
        if hidden_state.ndim == 3:
            # [batch, seq, hidden] - add to last position
            # Create zeros for all but last position
            batch_size = int(hidden_state.shape[0])
            seq_len = int(hidden_state.shape[1])

            # Add embedding only to last token position
            # Shape: [1, 1, hidden_dim]
            embedding_expanded = b.reshape(embedding, (1, 1, -1))

            # Create mask: 1 at last position, 0 elsewhere
            position_weights = b.zeros((1, seq_len, 1))
            # Set last position to 1
            # Since we can't do direct indexing assignment, we create it differently
            weights_list = [[0.0] * (seq_len - 1) + [1.0]]
            position_weights = b.array(weights_list)
            position_weights = b.reshape(position_weights, (1, seq_len, 1))

            # Broadcast and add
            injection = embedding_expanded * position_weights
            result = hidden_state + injection

        elif hidden_state.ndim == 2:
            # [seq, hidden] - add to last position
            seq_len = int(hidden_state.shape[0])
            weights = b.zeros((seq_len, 1))
            weights_list = [0.0] * (seq_len - 1) + [1.0]
            weights = b.array(weights_list)
            weights = b.reshape(weights, (seq_len, 1))

            injection = embedding[None, :] * weights
            result = hidden_state + injection

        else:
            # [hidden] - add directly
            result = hidden_state + embedding

        b.eval(result)
        return result

    def get_parameters(self) -> dict[str, Array]:
        """Get learnable parameters for training."""
        return {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
            "scale": self._scale,
        }

    def set_parameters(self, params: dict[str, Array]) -> None:
        """Set parameters from training."""
        if "w1" in params:
            self._w1 = params["w1"]
        if "b1" in params:
            self._b1 = params["b1"]
        if "w2" in params:
            self._w2 = params["w2"]
        if "b2" in params:
            self._b2 = params["b2"]
        if "scale" in params:
            self._scale = params["scale"]

    @property
    def config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self._config

    @property
    def hidden_dim(self) -> int:
        """Target hidden dimension."""
        return self._config.hidden_dim
