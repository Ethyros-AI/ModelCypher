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
JAX Activation Provider - Collects activations from JAX/Flax models.

This is an ADAPTER in hexagonal architecture. It implements the ActivationProvider
protocol for the JAX backend (Linux/TPU, CUDA GPU, or CPU).

JAX models can come from various frameworks (Flax, Haiku, Equinox). This provider
abstracts over the differences to provide a consistent activation collection interface.

Usage:
    from modelcypher.adapters.jax_activation_provider import JAXActivationProvider

    provider = JAXActivationProvider()
    hidden_acts = provider.collect_hidden_activations(model, tokenizer, text)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


class JAXActivationProvider:
    """
    JAX implementation of ActivationProvider protocol.

    Collects activations from JAX-based models (Flax, Haiku, Equinox).
    Keeps all tensors on TPU/GPU device memory.

    Supports:
    - Flax Linen modules with intermediate=True pattern
    - HuggingFace Transformers with FlaxPreTrainedModel
    - Custom JAX models with explicit layer iteration
    """

    def __init__(self) -> None:
        """Initialize JAX activation provider."""
        try:
            import jax
            import jax.numpy as jnp

            self.jax = jax
            self.jnp = jnp
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("JAX not available. Install with: pip install jax jaxlib")

    @property
    def available(self) -> bool:
        """Check if JAX backend is available."""
        return self._available

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer hidden state activations for a text input.

        Runs the text through the model and extracts the final hidden state
        (mean-pooled over sequence length) at each layer.

        Supports:
        - FlaxPreTrainedModel with output_hidden_states=True
        - Flax Linen modules with __call__(x, train=False)
        - Custom models with forward_with_hidden_states method

        Returns JAX arrays directly (stays on TPU/GPU).
        """
        if not self._available:
            raise RuntimeError("JAX backend not available. Install: pip install jax jaxlib")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = self.jnp.array([token_ids])

        activations: dict[int, "Array"] = {}

        try:
            # Try HuggingFace FlaxPreTrainedModel pattern
            if hasattr(model, "__call__") and hasattr(model, "config"):
                try:
                    outputs = model(input_ids, output_hidden_states=True)
                    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        for layer_idx, hidden in enumerate(outputs.hidden_states):
                            pooled = self.jnp.mean(hidden, axis=(0, 1))
                            activations[layer_idx] = pooled
                        return activations
                except Exception as e:
                    logger.debug("HuggingFace pattern failed: %s", e)

            # Try custom forward_with_hidden_states pattern
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    pooled = self.jnp.mean(hidden, axis=(0, 1))
                    activations[layer_idx] = pooled
                return activations

            # Try Flax Linen pattern with layer iteration
            if hasattr(model, "apply") and hasattr(model, "params"):
                # Flax Linen module - need to trace through layers
                # This requires model-specific knowledge
                logger.debug("Flax Linen model detected - using apply with mutable intermediates")
                try:
                    # Some Flax models support capture_intermediates
                    outputs, intermediates = model.apply(
                        {"params": model.params},
                        input_ids,
                        train=False,
                        mutable=["intermediates"],
                    )
                    if "intermediates" in intermediates:
                        for key, value in intermediates["intermediates"].items():
                            if "hidden" in key.lower() or "layer" in key.lower():
                                layer_idx = len(activations)
                                pooled = self.jnp.mean(value, axis=(0, 1))
                                activations[layer_idx] = pooled
                except Exception as e:
                    logger.debug("Flax intermediate capture failed: %s", e)

            # Fallback: run model and capture single output
            if not activations:
                try:
                    if hasattr(model, "apply"):
                        output = model.apply({"params": model.params}, input_ids, train=False)
                    else:
                        output = model(input_ids)
                    pooled = self.jnp.mean(output, axis=(0, 1))
                    activations[0] = pooled
                except Exception as e:
                    logger.warning("JAX model inference failed: %s", e)

        except Exception as e:
            logger.warning("Activation collection failed for text '%s...': %s", text[:30], e)

        if not activations:
            logger.debug("No activations collected for text: %s", text[:50])

        return activations

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> "Array":
        """
        Collect post-embedding activation for a text input.

        Uses the embedding output from hidden_states[0] when available.
        """
        if not self._available:
            raise RuntimeError("JAX backend not available. Install: pip install jax jaxlib")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = self.jnp.array([token_ids])

        # HuggingFace FlaxPreTrainedModel pattern
        if hasattr(model, "__call__") and hasattr(model, "config"):
            outputs = model(input_ids, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                emb = outputs.hidden_states[0]
                return self.jnp.mean(emb, axis=(0, 1))

        # Custom forward_with_hidden_states pattern
        if hasattr(model, "forward_with_hidden_states"):
            _, hidden_states = model.forward_with_hidden_states(input_ids)
            if hidden_states:
                emb = hidden_states[0]
                return self.jnp.mean(emb, axis=(0, 1))

        raise RuntimeError(
            "Embedding activations unavailable for this JAX model. "
            "Model must expose hidden_states with embedding output."
        )

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer MLP intermediate activations for a text input.

        For JAX/Flax models, this captures the intermediate MLP activations
        (after gating, before output projection).

        Returns JAX arrays directly (stays on TPU/GPU).
        """
        if not self._available:
            raise RuntimeError("JAX backend not available. Install: pip install jax jaxlib")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = self.jnp.array([token_ids])

        activations: dict[int, "Array"] = {}

        try:
            # Try HuggingFace FlaxPreTrainedModel with custom intermediate capture
            if hasattr(model, "__call__") and hasattr(model, "config"):
                # FlaxPreTrainedModel doesn't expose intermediate MLP activations
                # by default, so we'd need to modify the model or use hooks
                logger.debug("Intermediate activation collection not fully supported for FlaxPreTrainedModel")

            # Try Flax Linen with mutable intermediates
            if hasattr(model, "apply") and hasattr(model, "params"):
                try:
                    # Capture MLP intermediates if model supports it
                    outputs, intermediates = model.apply(
                        {"params": model.params},
                        input_ids,
                        train=False,
                        mutable=["intermediates"],
                    )
                    if "intermediates" in intermediates:
                        for key, value in intermediates["intermediates"].items():
                            if "mlp" in key.lower() or "ff" in key.lower():
                                layer_idx = len(activations)
                                pooled = self.jnp.mean(value, axis=(0, 1))
                                activations[layer_idx] = pooled
                except Exception as e:
                    logger.debug("Flax MLP intermediate capture failed: %s", e)

        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """
        Collect hidden + intermediate + embedding activations for multiple texts.
        """
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        hidden: list[dict[int, "Array"]] = []
        intermediate: list[dict[int, "Array"]] = []
        embedding: list["Array"] = []

        for text in texts:
            hidden.append(self.collect_hidden_activations(model, tokenizer, text))
            intermediate.append(self.collect_intermediate_activations(model, tokenizer, text))
            embedding.append(self.collect_embedding_activations(model, tokenizer, text))

        return ProbeActivationBatch(
            hidden=hidden,
            intermediate=intermediate,
            embedding=embedding,
        )

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        """
        Collect per-layer attention Q and KV activations for a text input.

        Returns TWO dicts:
        1. Q activations: [num_heads * head_dim]
        2. KV activations: [num_kv_heads * head_dim] (for GQA models)

        Returns JAX arrays directly (stays on TPU/GPU).
        """
        if not self._available:
            raise RuntimeError("JAX backend not available. Install: pip install jax jaxlib")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = self.jnp.array([token_ids])

        q_activations: dict[int, "Array"] = {}
        kv_activations: dict[int, "Array"] = {}

        try:
            # Try HuggingFace FlaxPreTrainedModel with attention outputs
            if hasattr(model, "__call__") and hasattr(model, "config"):
                try:
                    outputs = model(input_ids, output_attentions=True)
                    if hasattr(outputs, "attentions") and outputs.attentions:
                        # FlaxPreTrainedModel returns attention weights, not Q/K/V directly
                        # We'd need model surgery to get Q/K/V projections
                        logger.debug("Attention Q/KV extraction requires model surgery for FlaxPreTrainedModel")
                except Exception as e:
                    logger.debug("HuggingFace attention pattern failed: %s", e)

            # Try Flax Linen with mutable intermediates
            if hasattr(model, "apply") and hasattr(model, "params"):
                try:
                    outputs, intermediates = model.apply(
                        {"params": model.params},
                        input_ids,
                        train=False,
                        mutable=["intermediates"],
                    )
                    if "intermediates" in intermediates:
                        for key, value in intermediates["intermediates"].items():
                            layer_idx = len(q_activations)
                            if "query" in key.lower() or "q_proj" in key.lower():
                                pooled = self.jnp.mean(value, axis=(0, 1))
                                q_activations[layer_idx] = pooled
                            elif "key" in key.lower() or "k_proj" in key.lower():
                                pooled = self.jnp.mean(value, axis=(0, 1))
                                kv_activations[layer_idx] = pooled
                except Exception as e:
                    logger.debug("Flax attention intermediate capture failed: %s", e)

        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, kv_activations


def get_activation_provider() -> JAXActivationProvider:
    """Get the JAX activation provider instance."""
    return JAXActivationProvider()


__all__ = ["JAXActivationProvider", "get_activation_provider"]
