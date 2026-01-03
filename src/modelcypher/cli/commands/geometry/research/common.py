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

from __future__ import annotations

import logging

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.core.support.array_utils import array_to_list

logger = logging.getLogger(__name__)


def get_context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    """Activation provider for knowledge density analysis.

    Uses arithmetic mean for token aggregation (mean-pooling). This is
    appropriate for aggregating tokens within a single sequence. The
    manifold-aware Frechet mean is used later in the intrinsic dimension
    estimation when comparing across texts/concepts.
    """

    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
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
                # Arithmetic mean for token aggregation (mean-pooling)
                # Frechet mean is used later in intrinsic dimension estimation
                # when comparing across texts, not for within-text pooling
                mean = self._backend.mean(hidden[0], axis=0)
                self._backend.async_eval(mean)
                pending.append(mean)
                activations.append(mean)
            except Exception as exc:
                logger.debug("Activation failed for text '%s': %s", text, exc)
                continue

        if pending:
            self._backend.eval(*pending)

        return [array_to_list(self._backend, vec) for vec in activations]


def load_model_and_provider(model_path: str):
    """Load model and create activation provider.

    Args:
        model_path: Path to the model directory.
    """
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)

    backend = get_default_backend()
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
    )

    return model, tokenizer, backend, provider, num_layers


def cleanup_memory() -> None:
    """Aggressively clean up memory between model operations.

    This is critical when profiling multiple models sequentially.
    Without cleanup, memory accumulates and can crash the system.
    """
    import gc

    # Force Python garbage collection
    gc.collect()
    gc.collect()  # Second pass catches circular refs

    # Clear MLX cache if available
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Brief pause to let system reclaim memory
    import time

    time.sleep(1)
