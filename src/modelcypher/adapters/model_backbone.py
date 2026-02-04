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

"""Model backbone resolution utilities.

Provides functions to resolve model structure across different architectures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


def resolve_model_backbone(model, model_type: str | None = None):
    """Resolve the text backbone components from various model architectures.

    Handles multiple architecture styles:
    - Standard mlx_lm models (Qwen, Llama, Mistral)
    - Multimodal VL wrappers (Qwen-VL, LLaVA, GLM-4V)
    - Direct model structure (smaller models)
    - Deep search as fallback

    Args:
        model: The loaded model
        model_type: Optional model type hint (from model.model_type)

    Returns:
        Tuple of (embed_tokens, layers, norm) or None if resolution fails.
    """
    embed_tokens = None
    layers = None
    norm = None

    # Strategy 1: Standard mlx_lm structure (Qwen, Llama, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        layers = model.model.layers
        norm = getattr(model.model, "norm", None)
        return (embed_tokens, layers, norm)

    # Strategy 2: Multimodal VL wrapper with nested language_model
    if hasattr(model, "language_model"):
        lm = model.language_model

        # GLM-4V uses transformer instead of model
        if hasattr(lm, "transformer"):
            transformer = lm.transformer
            embed_tokens = getattr(transformer, "embedding", None)
            if embed_tokens is not None:
                embed_tokens = getattr(embed_tokens, "word_embeddings", embed_tokens)
            layers = getattr(transformer, "encoder", None)
            if layers is not None:
                layers = getattr(layers, "layers", layers)
            norm = getattr(transformer, "output_layer_norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

        # Standard LM structure (Qwen-VL, LLaVA, etc.)
        if hasattr(lm, "model"):
            embed_tokens = getattr(lm.model, "embed_tokens", None)
            layers = getattr(lm.model, "layers", None)
            norm = getattr(lm.model, "norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

    # Strategy 3: Direct model structure (some smaller models)
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        embed_tokens = model.embed_tokens
        layers = model.layers
        norm = getattr(model, "norm", None)
        return (embed_tokens, layers, norm)

    # Strategy 4: Deep search through model tree
    def find_deep(module):
        found_embed = None
        found_layers = None
        found_norm = None

        for name, m in module.named_modules():
            if any(x in name.lower() for x in ["layers", "blocks", "h", "encoder"]):
                if hasattr(m, "__getitem__") and hasattr(m, "__len__") and len(m) > 0:
                    first = m[0]
                    if (
                        hasattr(first, "self_attn")
                        or hasattr(first, "attention")
                        or hasattr(first, "mlp")
                    ):
                        found_layers = m
                        break

        for name, m in module.named_modules():
            if "embed" in name.lower() and hasattr(m, "__call__") and not hasattr(m, "__getitem__"):
                found_embed = m
                break

        for name, m in module.named_modules():
            if "norm" in name.lower() and "layer" not in name.lower():
                found_norm = m

        return found_embed, found_layers, found_norm

    embed_tokens, layers, norm = find_deep(model)
    if embed_tokens and layers:
        return (embed_tokens, layers, norm)

    return None


def _apply_layer_with_mask(layer, hidden, mask):
    """Apply a transformer layer with best-effort mask handling."""
    try:
        return layer(hidden, mask=mask)
    except (TypeError, ValueError):
        try:
            return layer(hidden, mask)
        except (TypeError, ValueError):
            return layer(hidden)


def forward_through_backbone(
    input_ids,
    embed_tokens,
    layers,
    norm,
    target_layer: int,
    backend: "Backend",
):
    """Forward pass through text backbone to extract hidden states.

    Args:
        input_ids: Token IDs [batch, seq_len]
        embed_tokens: Embedding layer
        layers: List of transformer layers
        norm: Optional final normalization layer
        target_layer: Which layer to extract from (0-indexed, -1 for last)
        backend: Backend for tensor operations

    Returns:
        Hidden states at target layer [batch, seq_len, hidden_dim]
    """
    hidden = embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    mask = backend.create_causal_mask(seq_len, hidden.dtype)

    actual_target = target_layer if target_layer >= 0 else len(layers) - 1

    for i, layer in enumerate(layers):
        hidden = _apply_layer_with_mask(layer, hidden, mask)
        if i == actual_target:
            break

    if norm is not None and actual_target == len(layers) - 1:
        hidden = norm(hidden)

    return hidden
