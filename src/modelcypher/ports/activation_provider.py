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
Activation Provider Protocol for model inference and activation collection.

This is a PORT in hexagonal architecture. Domain code imports ActivationProvider,
never concrete implementations. The adapters (MLXActivationProvider, etc.) are
injected at runtime based on platform detection.

Why a Separate Port?
--------------------
The Backend protocol is for tensor operations (matmul, svd, etc.).
Activation collection is a model-level operation that requires:
- Running inference
- Accessing internal layer states
- Model structure knowledge (layers, embeddings, etc.)

This belongs in a separate port because:
1. It operates on models, not just tensors
2. It requires model-specific knowledge (layer structure, embedding access)
3. It produces activations that then flow into Backend tensor operations

Architecture:

    core/domain/ --> ports/ActivationProvider (this file)
                                              <-- adapters/mlx_activation_provider.py
                                              <-- adapters/jax_activation_provider.py

Dependencies point inward. Domain never imports adapters directly.

Usage:

    from modelcypher.ports.activation_provider import ActivationProvider

    def my_function(provider: ActivationProvider, model, tokenizer, text):
        hidden_acts = provider.collect_hidden_activations(model, tokenizer, text)
        intermediate_acts = provider.collect_intermediate_activations(model, tokenizer, text)
        q_acts, kv_acts = provider.collect_attention_activations(model, tokenizer, text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

# TypeVar for array types - same as Backend protocol
Array = TypeVar("Array")


@dataclass(frozen=True)
class ProbeActivationBatch:
    """Batch activations for probe alignment in a single forward pass."""

    hidden: list[dict[int, Array]]
    intermediate: list[dict[int, Array]]
    gate: list[dict[int, Array]]
    embedding: list[Array]


@runtime_checkable
class ActivationProvider(Protocol):
    """
    Abstract protocol for collecting activations from neural network models.

    Implementations: MLXActivationProvider (macOS), JAXActivationProvider (Linux).
    Domain code should depend on this protocol, not concrete implementations.

    All methods return backend-native arrays (never NumPy) to stay on GPU.
    """

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Array]:
        """
        Collect per-layer hidden state activations for a text input.

        Runs the text through the model and extracts the final hidden state
        (mean-pooled over sequence length) at each layer.

        Args:
            model: The loaded model (e.g., mlx_lm model).
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input (skips tokenization if provided).

        Returns:
            Dict mapping layer_idx -> activation array of shape [hidden_dim].
            E.g., {0: Array([...], shape=(960,)), 1: Array([...], shape=(960,)), ...}
        """
        ...

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Array:
        """
        Collect post-embedding activation for a text input.

        This captures the OUTPUT of embed_tokens (before layer 0 input_layernorm).
        Shape: [hidden_dim] (mean-pooled over sequence length).

        Used for GramAlign at the 1D→2D interface (token IDs → embedding space).
        Linear alignment is exact on probes; geodesic CKA is the overlap diagnostic
        at the embedding dimension.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input.

        Returns:
            Activation array of shape [hidden_dim] representing the mean-pooled
            embedding output.
        """
        ...

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Array]:
        """
        Collect per-layer MLP intermediate activations for a text input.

        Captures the activation INSIDE the MLP (after gate_proj * up_proj, before down_proj).
        This is the intermediate representation space, distinct from the hidden space.

        Shape: [intermediate_dim] (e.g., 2560 for SmolLM, 4864 for Qwen)

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input.

        Returns:
            Dict mapping layer_idx -> activation array of shape [intermediate_dim].
        """
        ...

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Array], dict[int, Array]]:
        """
        Collect per-layer attention Q and KV activations for a text input.

        Returns TWO dicts:
        1. Q activations: [num_heads * head_dim] (e.g., 960 for SmolLM, 896 for Qwen)
        2. KV activations: [num_kv_heads * head_dim] (e.g., 320 for SmolLM, 128 for Qwen)

        For Grouped Query Attention (GQA) models, Q and KV have different dimensions:
        - SmolLM: Q = 15 heads × 64 = 960, KV = 5 heads × 64 = 320
        - Qwen: Q = 14 heads × 64 = 896, KV = 2 heads × 64 = 128

        Separate transforms are needed for each space:
        - Q activations: For q_proj and o_proj weight stitching
        - KV activations: For k_proj and v_proj weight stitching

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input.

        Returns:
            Tuple of (q_activations, kv_activations), each a dict mapping
            layer_idx -> activation array.
        """
        ...

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> ProbeActivationBatch:
        """
        Collect hidden + intermediate + gate + embedding activations for multiple texts in one pass.

        This is the strict, efficient probe path used for merge alignment. Implementations
        must not fall back to sequential collection.
        """
        ...

    # ==========================================================================
    # BATCHED METHODS - Process multiple texts in a single forward pass
    # ==========================================================================
    # These methods are OPTIONAL. Implementations that don't support batching
    # should raise NotImplementedError, and callers should fall back to
    # sequential processing.

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Array]]:
        """
        Collect per-layer hidden state activations for multiple texts in one pass.

        This is more efficient than calling collect_hidden_activations() multiple
        times because it batches the forward passes, reducing kernel launch overhead.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding texts.
            texts: List of text inputs to process.

        Returns:
            List of dicts, one per input text, each mapping layer_idx -> activation.

        Raises:
            NotImplementedError: If batching is not supported by this implementation.
        """
        ...

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Array]]:
        """
        Collect per-layer MLP intermediate activations for multiple texts in one pass.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding texts.
            texts: List of text inputs to process.

        Returns:
            List of dicts, one per input text, each mapping layer_idx -> activation.

        Raises:
            NotImplementedError: If batching is not supported by this implementation.
        """
        ...

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Array]]:
        """
        Collect per-layer PRE-SiLU gate activations for multiple texts in one pass.

        This is the correct activation space for gate_proj/up_proj stitching.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding texts.
            texts: List of text inputs to process.

        Returns:
            List of dicts, one per input text, each mapping layer_idx -> activation.

        Raises:
            NotImplementedError: If batching is not supported by this implementation.
        """
        ...

    def collect_attention_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> tuple[list[dict[int, Array]], list[dict[int, Array]]]:
        """
        Collect per-layer attention Q and KV activations for multiple texts in one pass.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding texts.
            texts: List of text inputs to process.

        Returns:
            Tuple of (q_activations_list, kv_activations_list), each a list of dicts
            mapping layer_idx -> activation array.

        Raises:
            NotImplementedError: If batching is not supported by this implementation.
        """
        ...


__all__ = ["ActivationProvider", "Array", "ProbeActivationBatch"]
