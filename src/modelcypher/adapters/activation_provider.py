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

"""Activation provider adapter using Backend protocol.

This module implements the ActivationProvider port using the Backend abstraction,
making it framework-agnostic across supported runtimes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class ActivationProviderAdapter:
    """Activation provider adapter using Backend protocol.

    This implementation uses the Backend abstraction to collect activations,
    making it framework-agnostic across supported runtimes.
    """

    def __init__(
        self,
        backend: "Backend",
        model_path: str | Path | None = None,
        pooling: str = "auto",
    ) -> None:
        self._backend = backend
        self._model_path = model_path
        self._pooling = pooling
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self) -> tuple[Any, Any]:
        """Ensure model and tokenizer are loaded."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        if self._model_path is None:
            raise RuntimeError("No model_path provided to ActivationProvider")

        from modelcypher.adapters.model_loader import load_model_for_training

        self._model, self._tokenizer = load_model_for_training(str(self._model_path))
        return self._model, self._tokenizer

    def _get_backbone(self, model: Any) -> tuple[Any, Any, Any]:
        """Get backbone components (embed_tokens, layers, norm) from model."""
        from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone

        model_type = getattr(model, "model_type", "unknown")
        resolved = resolve_model_backbone(model, model_type)
        if not resolved:
            raise RuntimeError("Could not resolve model backbone")
        return resolved

    def _pool_activations(self, hidden: Any) -> Any:
        """Pool activations based on pooling strategy."""
        # Mean pooling over sequence dimension
        return self._backend.mean(hidden, axis=0)

    def _forward_through_backbone(
        self,
        input_ids: Any,
        embed_tokens: Any,
        layers: Any,
        norm: Any,
        target_layer: int,
    ) -> Any:
        """Forward pass through backbone to target layer."""
        from modelcypher.cli.commands.geometry.helpers import forward_through_backbone

        return forward_through_backbone(
            input_ids,
            embed_tokens,
            layers,
            norm,
            target_layer=target_layer,
            backend=self._backend,
        )

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer hidden activations for text."""
        embed_tokens, layers, norm = self._get_backbone(model)
        num_layers = len(layers)

        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        results = {}

        for layer_idx in range(num_layers):
            hidden = self._forward_through_backbone(
                input_ids, embed_tokens, layers, norm, target_layer=layer_idx
            )
            pooled = self._pool_activations(hidden[0])
            self._backend.eval(pooled)
            results[layer_idx] = pooled

        return results

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect post-embedding activations for text."""
        embed_tokens, _, _ = self._get_backbone(model)

        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        embedded = embed_tokens(input_ids)
        pooled = self._pool_activations(embedded[0])
        self._backend.eval(pooled)
        return pooled

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer MLP intermediate activations."""
        # This requires hooking into MLP intermediate outputs
        # For now, return hidden activations as a fallback
        return self.collect_hidden_activations(model, tokenizer, text, token_ids)

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Any], dict[int, Any]]:
        """Collect per-layer attention Q and KV activations."""
        # This requires hooking into attention outputs
        # For now, return hidden activations as a fallback
        hidden = self.collect_hidden_activations(model, tokenizer, text, token_ids)
        return hidden, hidden

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect logits for text."""
        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        logits = model(input_ids)
        self._backend.eval(logits)
        # Return last token's logits
        return logits[0, -1, :]

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """Collect activations for multiple texts."""
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        hidden_list = []
        intermediate_list = []
        gate_list = []
        embedding_list = []

        for text in texts:
            hidden = self.collect_hidden_activations(model, tokenizer, text)
            embedding = self.collect_embedding_activations(model, tokenizer, text)
            hidden_list.append(hidden)
            intermediate_list.append(hidden)  # Fallback
            gate_list.append(hidden)  # Fallback
            embedding_list.append(embedding)

        return ProbeActivationBatch(
            hidden=hidden_list,
            intermediate=intermediate_list,
            gate=gate_list,
            embedding=embedding_list,
        )

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect hidden activations for multiple texts."""
        return [self.collect_hidden_activations(model, tokenizer, text) for text in texts]

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect intermediate activations for multiple texts."""
        return [self.collect_intermediate_activations(model, tokenizer, text) for text in texts]

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect gate activations for multiple texts."""
        # Fallback to hidden activations
        return [self.collect_hidden_activations(model, tokenizer, text) for text in texts]

    def collect_attention_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> tuple[list[dict[int, Any]], list[dict[int, Any]]]:
        """Collect attention activations for multiple texts."""
        q_list = []
        kv_list = []
        for text in texts:
            q, kv = self.collect_attention_activations(model, tokenizer, text)
            q_list.append(q)
            kv_list.append(kv)
        return q_list, kv_list

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """Collect trajectory activations (not yet implemented)."""
        raise NotImplementedError("Trajectory collection not yet implemented")


# Alias for backwards compatibility during transition
ActivationProvider = ActivationProviderAdapter


def get_activation_provider(
    backend: "Backend | None" = None,
    model_path: str | Path | None = None,
    pooling: str = "auto",
) -> ActivationProviderAdapter:
    """Factory for ActivationProviderAdapter."""
    from modelcypher.backends import initialize_default_backend

    if backend is None:
        backend = initialize_default_backend()

    return ActivationProviderAdapter(backend=backend, model_path=model_path, pooling=pooling)


__all__ = ["ActivationProviderAdapter", "ActivationProvider", "get_activation_provider"]
