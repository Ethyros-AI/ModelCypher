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

"""Unified Activation Provider - delegates to Backend.

This is THE activation provider. It uses the Backend protocol for ALL
tensor operations and model operations. No framework imports here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from modelcypher.ports.activation_provider import (
    ActivationProvider as ActivationProviderProtocol,
    ProbeActivationBatch,
    TrajectoryActivations,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_architecture import ModelArchitecturePort

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["auto", "last", "mean", "max"]


class ActivationProvider(ActivationProviderProtocol):
    """Unified activation provider - delegates ALL operations to Backend.

    Handles MLX, JAX, and CUDA models through the Backend protocol.
    NO framework imports in this file.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        model_path: str | Path | None = None,
        pooling: PoolingStrategy = "auto",
    ) -> None:
        """Initialize with optional backend.

        Args:
            backend: If None, auto-detects from platform.
            model_path: Optional path to model directory (for config.json).
            pooling: Default pooling strategy for activation collection.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self._backend = backend
        self._model_path = Path(model_path) if model_path else None
        self._default_pooling = pooling
        self._architecture_cache: dict[int, "ModelArchitecturePort"] = {}

    def _get_architecture(self, model: Any) -> "ModelArchitecturePort":
        """Get or create architecture wrapper for model."""
        from modelcypher.adapters.model_architecture import get_model_architecture, load_config

        model_id = id(model)
        if model_id in self._architecture_cache:
            return self._architecture_cache[model_id]

        config = None
        if self._model_path is not None:
            config = load_config(self._model_path)

        arch = get_model_architecture(model, config=config, model_path=self._model_path)
        self._architecture_cache[model_id] = arch
        return arch

    def _resolve_pooling(self, model: Any, pooling: PoolingStrategy | None = None) -> str:
        """Resolve pooling strategy, using architecture-aware auto-detection."""
        strategy = pooling or self._default_pooling

        if strategy == "auto":
            try:
                arch = self._get_architecture(model)
                return "last" if arch.is_causal else "mean"
            except (ValueError, AttributeError):
                logger.debug("Architecture detection failed, using mean pooling")
                return "mean"

        return strategy

    def _tokenize(self, tokenizer: Any, text: str, token_ids: list[int] | None = None) -> list[int]:
        """Tokenize text or return provided token_ids."""
        if token_ids is not None:
            return token_ids
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            return tokens
        return list(tokens.ids)

    # =========================================================================
    # Core collection methods - ALL delegate to Backend
    # =========================================================================

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """Collect per-layer hidden state activations for a text input."""
        return self._backend.collect_hidden_activations(model, tokenizer, [text])[0] if text else {}

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> "Array":
        """Collect post-embedding activation for a text input."""
        return self._backend.collect_embedding_activations(model, tokenizer, text, token_ids)

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """Collect per-layer MLP intermediate activations."""
        return self._backend.collect_intermediate_activations(model, tokenizer, text, token_ids)

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"], dict[int, "Array"]]:
        """Collect per-layer attention Q, K, V activations."""
        return self._backend.collect_attention_activations(model, tokenizer, text, token_ids)

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> "Array":
        """Collect logits for a text input."""
        return self._backend.collect_logits(model, tokenizer, text, token_ids)

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> ProbeActivationBatch:
        """Collect hidden + intermediate + gate + embedding activations in one batched pass."""
        return self._backend.collect_probe_activations_batch(model, tokenizer, texts)

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer hidden state activations for multiple texts."""
        return self._backend.collect_hidden_activations_batch(model, tokenizer, texts)

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer MLP intermediate activations for multiple texts."""
        return self._backend.collect_intermediate_activations_batch(model, tokenizer, texts)

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer PRE-SiLU gate activations for multiple texts."""
        return self._backend.collect_gate_activations_batch(model, tokenizer, texts)

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> TrajectoryActivations:
        """Collect full trajectory activations for geometric manifold mapping."""
        return self._backend.collect_trajectory_batch(model, tokenizer, texts)


def get_activation_provider(backend: "Backend | None" = None) -> ActivationProvider:
    """Get an activation provider instance."""
    return ActivationProvider(backend)


__all__ = ["ActivationProvider", "get_activation_provider"]
