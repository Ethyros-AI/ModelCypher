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

"""Shared helpers for geometry CLI commands.

This module provides canonical implementations of helper functions
used across geometry CLI commands. All CLI geometry commands should
use these helpers instead of reimplementing them.

Functions:
- resolve_model_backbone: Extract text backbone from various model architectures
- forward_through_backbone: Forward pass through text backbone
- forward_through_backbone_embeddings: Forward pass from pre-embedded inputs
- extract_anchor_activations: Extract activations for a list of anchors
- save_activations_json: Save activations to JSON file
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


def resolve_model_backbone(model, model_type: str | None = None):
    """Resolve the text backbone components from various model architectures.

    This is the canonical implementation for resolving model structure.
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
        - embed_tokens: Token embedding layer
        - layers: List of transformer layers
        - norm: Final normalization layer (may be None)
    """
    embed_tokens = None
    layers = None
    norm = None

    # Strategy 1: Standard mlx_lm structure (Qwen, Llama, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        logger.debug("Using Strategy 1 (Standard mlx_lm structure)")
        embed_tokens = model.model.embed_tokens
        layers = model.model.layers
        norm = getattr(model.model, "norm", None)
        return (embed_tokens, layers, norm)

    # Strategy 2: Multimodal VL wrapper with nested language_model
    if hasattr(model, "language_model"):
        logger.debug("Using Strategy 2 (Multimodal VL wrapper)")
        lm = model.language_model

        # GLM-4V uses transformer instead of model
        if hasattr(lm, "transformer"):
            logger.debug("Detected transformer in language_model (GLM-4V style)")
            transformer = lm.transformer
            embed_tokens = getattr(transformer, "embedding", None)
            if embed_tokens is not None:
                # GLM-4 uses embedding.word_embeddings
                embed_tokens = getattr(embed_tokens, "word_embeddings", embed_tokens)
            layers = getattr(transformer, "encoder", None)
            if layers is not None:
                layers = getattr(layers, "layers", layers)
            norm = getattr(transformer, "output_layer_norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

        # Standard LM structure (Qwen-VL, LLaVA, etc.)
        if hasattr(lm, "model"):
            logger.debug("Detected model in language_model (standard VL structure)")
            embed_tokens = getattr(lm.model, "embed_tokens", None)
            layers = getattr(lm.model, "layers", None)
            norm = getattr(lm.model, "norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

    # Strategy 3: Direct model structure (some smaller models)
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        logger.debug("Using Strategy 3 (Direct model structure)")
        embed_tokens = model.embed_tokens
        layers = model.layers
        norm = getattr(model, "norm", None)
        return (embed_tokens, layers, norm)

    # Strategy 4: Deep search through model tree
    def find_deep(module):
        """Recursively search for embed_tokens and layers."""
        found_embed = None
        found_layers = None
        found_norm = None

        # Look for layers first (ModuleList or list of transformer blocks)
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
                        logger.debug(f"Found layers at {name}")
                        break

        # Look for embedding
        for name, m in module.named_modules():
            if "embed" in name.lower() and hasattr(m, "__call__") and not hasattr(m, "__getitem__"):
                found_embed = m
                logger.debug(f"Found embedding at {name}")
                break

        # Look for final norm
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

    This is the canonical implementation for extracting hidden states.
    Bypasses the full model forward pass and directly runs through
    the text backbone components, making it work for both text-only
    and multimodal models.

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
    # Embed tokens
    hidden = embed_tokens(input_ids)

    # Create causal mask
    seq_len = input_ids.shape[1]
    mask = backend.create_causal_mask(seq_len, hidden.dtype)

    # Handle -1 for last layer
    actual_target = target_layer if target_layer >= 0 else len(layers) - 1

    # Forward through layers
    for i, layer in enumerate(layers):
        hidden = _apply_layer_with_mask(layer, hidden, mask)

        if i == actual_target:
            break

    # Apply final norm if available and we went through all layers
    if norm is not None and actual_target == len(layers) - 1:
        hidden = norm(hidden)

    return hidden


def forward_through_backbone_embeddings(
    embedded,
    layers,
    norm,
    target_layer: int,
    backend: "Backend",
):
    """Forward pass through text backbone starting from embeddings."""
    hidden = embedded

    seq_len = embedded.shape[1]
    mask = backend.create_causal_mask(seq_len, hidden.dtype)

    actual_target = target_layer if target_layer >= 0 else len(layers) - 1

    for i, layer in enumerate(layers):
        hidden = _apply_layer_with_mask(layer, hidden, mask)
        if i == actual_target:
            break

    if norm is not None and actual_target == len(layers) - 1:
        hidden = norm(hidden)

    return hidden


def extract_anchor_activations(
    anchors: list,
    tokenizer,
    embed_tokens,
    layers,
    norm,
    target_layer: int,
    backend: "Backend",
    prompt_attr: str = "prompt",
    name_attr: str = "name",
    verbose: bool = False,
) -> dict:
    """Extract and mean-pool activations for a list of anchors.

    Uses async_eval for pipeline parallelism: GPU processes current anchor
    while CPU tokenizes the next one. Final sync at the end.

    Args:
        anchors: List of anchor objects with prompt and name attributes
        tokenizer: Tokenizer for encoding prompts
        embed_tokens: Embedding layer
        layers: List of transformer layers
        norm: Optional final normalization layer
        target_layer: Which layer to extract from
        backend: Backend for tensor operations
        prompt_attr: Attribute name for the prompt text (default: "prompt")
        name_attr: Attribute name for the anchor name (default: "name")
        verbose: If True, log progress

    Returns:
        Dictionary mapping anchor names to activation vectors
    """
    activations = {}
    pending_activations = []  # Collect for final eval

    for anchor in anchors:
        name = getattr(anchor, name_attr) if hasattr(anchor, name_attr) else str(anchor)
        prompt = getattr(anchor, prompt_attr) if hasattr(anchor, prompt_attr) else str(anchor)

        try:
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])

            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                norm,
                target_layer=target_layer,
                backend=backend,
            )

            # Mean pool across sequence length
            activation = backend.mean(hidden[0], axis=0)
            # Use async_eval for pipeline parallelism - don't block here
            backend.async_eval(activation)
            activations[name] = activation
            pending_activations.append(activation)

            if verbose:
                logger.info(f"Extracted activation for {name}")

        except Exception as e:
            logger.warning(f"Failed to extract activation for {name}: {e}")

    # Final sync - materialize all pending activations
    if pending_activations:
        backend.eval(*pending_activations)

    return activations


def save_activations_json(
    activations: dict,
    output_path: str | Path,
    backend: "Backend",
) -> None:
    """Save activations dict to JSON file.

    Args:
        activations: Dictionary mapping names to activation arrays
        output_path: Path to save JSON file
        backend: Backend for array conversion
    """
    activations_json = {name: array_to_list(backend, act) for name, act in activations.items()}
    Path(output_path).write_text(json.dumps(activations_json, indent=2))


def load_activations_json(
    input_path: str | Path,
    backend: "Backend",
) -> dict:
    """Load activations from JSON file.

    Args:
        input_path: Path to JSON file
        backend: Backend for array creation

    Returns:
        Dictionary mapping names to activation arrays
    """
    data = json.loads(Path(input_path).read_text())
    return {name: backend.array(vec) for name, vec in data.items()}


class BackboneActivationProvider:
    """Activation provider using model backbone components.

    This is the canonical implementation for collecting activations from
    a model's backbone (embed_tokens, layers, norm). All CLI geometry
    commands should use this instead of reimplementing activation collection.

    Args:
        tokenizer: Tokenizer for encoding text
        embed_tokens: Token embedding layer
        layers: List of transformer layers
        norm: Optional final normalization layer
        backend: Backend for tensor operations
        pooling: Pooling strategy - "frechet" for manifold-aware, "mean" for arithmetic
        frechet_k_neighbors: k for frechet mean (only if pooling="frechet")
        frechet_max_k_neighbors: max k for frechet mean (only if pooling="frechet")
    """

    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend: "Backend",
        pooling: str = "mean",
        frechet_k_neighbors: int | None = None,
        frechet_max_k_neighbors: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend
        self._pooling = pooling
        self._frechet_k_neighbors = frechet_k_neighbors
        self._frechet_max_k_neighbors = frechet_max_k_neighbors

    def _pool(self, hidden):
        """Pool hidden states using configured strategy."""
        if self._pooling == "frechet":
            from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
            return frechet_mean(
                hidden,
                backend=self._backend,
                k_neighbors=self._frechet_k_neighbors,
                max_k_neighbors=self._frechet_max_k_neighbors,
            )
        return self._backend.mean(hidden, axis=0)

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        """Get pooled activations for texts at a specific layer.

        Args:
            texts: List of text inputs
            layer: Layer index to extract from

        Returns:
            List of activation vectors as Python lists
        """
        activations = []
        pending = []

        for text in texts:
            if not text:
                continue
            try:
                tokens = self._tokenizer.encode(text)
                if not tokens:
                    continue
                input_ids = self._backend.array([tokens])
                hidden = forward_through_backbone(
                    input_ids,
                    self._embed_tokens,
                    self._layers,
                    self._norm,
                    target_layer=layer,
                    backend=self._backend,
                )
                mean = self._pool(hidden[0])
                self._backend.async_eval(mean)
                pending.append(mean)
                activations.append(mean)
            except Exception as exc:
                logger.debug("Activation failed for text '%s': %s", text, exc)
                continue

        if pending:
            self._backend.eval(*pending)

        return [array_to_list(self._backend, vec) for vec in activations]


def load_model_and_provider(
    model_path: str,
    pooling: str = "mean",
    frechet_k_neighbors: int | None = None,
    frechet_max_k_neighbors: int | None = None,
):
    """Load model and create activation provider.

    This is the canonical factory for loading a model and creating a
    BackboneActivationProvider. All CLI geometry commands should use this.

    Args:
        model_path: Path to the model directory
        pooling: Pooling strategy - "frechet" or "mean"
        frechet_k_neighbors: k for frechet mean (only if pooling="frechet")
        frechet_max_k_neighbors: max k for frechet mean (only if pooling="frechet")

    Returns:
        Tuple of (model, tokenizer, backend, provider, num_layers)

    Raises:
        typer.BadParameter: If model architecture cannot be resolved
    """
    import typer
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.cli.composition import get_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)

    backend = get_backend()
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        pooling=pooling,
        frechet_k_neighbors=frechet_k_neighbors,
        frechet_max_k_neighbors=frechet_max_k_neighbors,
    )

    return model, tokenizer, backend, provider, num_layers


def cleanup_memory() -> None:
    """Aggressively clean up memory between model operations.

    This is critical when profiling multiple models sequentially.
    Without cleanup, memory accumulates and can crash the system.
    """
    import gc
    import time
    from modelcypher.cli.composition import get_backend

    gc.collect()
    gc.collect()  # Second pass catches circular refs

    try:
        backend = get_backend()
        backend.clear_cache()
    except Exception as exc:
        logger.debug("Failed to clear backend cache: %s", exc)
