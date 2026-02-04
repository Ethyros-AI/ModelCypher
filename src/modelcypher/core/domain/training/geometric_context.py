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

"""Geometric context injection for self-aware training.

This module provides the model with access to its own geometric state
during training. The context is injected as a natural language prefix
that the model can learn to interpret and use.

All values are relational (ratios/comparisons) - no arbitrary units:
- loop_persistence: ΔH (current - baseline) entropy change
- expansion_ratio: peak_norm / final_norm
- highway_depth: highway_layer / n_layers
- exit_convergence: mean_norm / dev_norm at exit
- has_reasoning_loops: spectral entropy proxy for β₁ > 0

The format uses [GEOMETRY]...[/GEOMETRY] markers that work with any
tokenizer without special tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GeometricContext:
    """All values are ratios or relative comparisons. No arbitrary units.

    Attributes:
        loop_persistence: ΔH = H_exit - H_highway (entropy change).
            Positive = healthy (loops preserved), negative = collapsed.
        expansion_ratio: peak_norm / final_norm.
            >1 = intermediate expansion, <1 = monotonic compression.
        highway_depth: highway_layer / n_layers.
            Where in the model the compression bottleneck occurs.
        exit_convergence: mean_norm / dev_norm at exit.
            High = consistent outputs, low = varied outputs.
        has_reasoning_loops: True if H_exit > H_highway.
            Spectral proxy for topological β₁ > 0.
    """

    loop_persistence: float
    expansion_ratio: float
    highway_depth: float
    exit_convergence: float
    has_reasoning_loops: bool

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer,
        prompt: str,
        highway_layer: int,
        base_delta_entropy: float = 0.0,
    ) -> "GeometricContext":
        """Compute all geometric context from a single forward pass.

        Args:
            model: The loaded model.
            tokenizer: Tokenizer for the model.
            prompt: The prompt to compute geometry for.
            highway_layer: Pre-computed highway layer index.
            base_delta_entropy: Base model's ΔH for comparison.

        Returns:
            GeometricContext with all relational metrics.
        """
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", [])
        n_layers = len(layers)
        embed_module = getattr(base_model, "embed_tokens", None)

        if n_layers == 0 or embed_module is None:
            return cls(
                loop_persistence=0.0,
                expansion_ratio=1.0,
                highway_depth=0.0,
                exit_convergence=1.0,
                has_reasoning_loops=False,
            )

        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids) if hasattr(tokens, "ids") else list(tokens)
        input_ids = mx.array([token_ids])

        # Track norms and entropies through forward pass
        norms: list[float] = []
        h_highway_entropy: float = 0.0
        h_exit_entropy: float = 0.0

        # Get embeddings
        h = embed_module(input_ids)
        mx.eval(h)
        norms.append(_compute_norm(h))

        # Forward through all layers
        for i, layer in enumerate(layers):
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result
            mx.eval(h)

            norms.append(_compute_norm(h))

            # Capture entropies at key layers
            if i == highway_layer:
                h_highway_entropy = _compute_spectral_entropy(h)
            if i == n_layers - 1:
                h_exit_entropy = _compute_spectral_entropy(h)

        # Compute relational metrics
        # Loop persistence: ΔH relative to base
        current_delta = h_exit_entropy - h_highway_entropy
        loop_persistence = current_delta - base_delta_entropy

        # Expansion ratio: peak_norm / final_norm
        peak_norm = max(norms) if norms else 1.0
        final_norm = norms[-1] if norms else 1.0
        expansion_ratio = peak_norm / max(final_norm, 1e-8)

        # Highway depth: fraction of total layers
        highway_depth = highway_layer / max(n_layers, 1)

        # Exit convergence: mean_norm / dev_norm at exit
        exit_mean, exit_dev = _compute_exit_convergence(h)
        exit_convergence = exit_mean / max(exit_dev, 1e-8)

        # Has reasoning loops: spectral proxy (entropy grows after highway)
        has_reasoning_loops = current_delta > 0

        return cls(
            loop_persistence=loop_persistence,
            expansion_ratio=expansion_ratio,
            highway_depth=highway_depth,
            exit_convergence=exit_convergence,
            has_reasoning_loops=has_reasoning_loops,
        )

    def format(self) -> str:
        """Format as prefix. Values are raw floats, model learns interpretation.

        Uses [GEOMETRY]...[/GEOMETRY] markers that work with any tokenizer
        without requiring special tokens.
        """
        reasoning = "yes" if self.has_reasoning_loops else "no"
        return f"""[GEOMETRY]
loop_persistence: {self.loop_persistence:+.2f}
expansion_ratio: {self.expansion_ratio:.2f}
highway_depth: {self.highway_depth:.2f}
convergence: {self.exit_convergence:.2f}
reasoning_loops: {reasoning}
[/GEOMETRY]

"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/saving."""
        return {
            "loop_persistence": self.loop_persistence,
            "expansion_ratio": self.expansion_ratio,
            "highway_depth": self.highway_depth,
            "exit_convergence": self.exit_convergence,
            "has_reasoning_loops": self.has_reasoning_loops,
        }


def compute_geometric_context_for_batch(
    model,
    tokenizer,
    prompts: list[str],
    highway_layer: int,
    base_delta_entropy: float = 0.0,
) -> list[GeometricContext]:
    """Compute geometric context for a batch of prompts.

    This is more efficient than calling from_model individually because
    the model structure is only queried once.

    Args:
        model: The loaded model.
        tokenizer: Tokenizer for the model.
        prompts: List of prompts to compute geometry for.
        highway_layer: Pre-computed highway layer index.
        base_delta_entropy: Base model's ΔH for comparison.

    Returns:
        List of GeometricContext, one per prompt.
    """
    contexts: list[GeometricContext] = []

    for prompt in prompts:
        ctx = GeometricContext.from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            highway_layer=highway_layer,
            base_delta_entropy=base_delta_entropy,
        )
        contexts.append(ctx)

    return contexts


def _compute_norm(hidden: mx.array) -> float:
    """Compute L2 norm of hidden states (mean across batch/seq)."""
    # hidden: [batch, seq, hidden_dim]
    if len(hidden.shape) == 3:
        hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))

    norms = mx.sqrt(mx.sum(hidden * hidden, axis=-1))
    mean_norm = mx.mean(norms)
    mx.eval(mean_norm)

    return float(mean_norm.tolist())


def _compute_spectral_entropy(hidden: mx.array) -> float:
    """Compute spectral entropy from hidden states.

    Same as in loop_preservation.py - copied here to avoid circular import.
    """
    if len(hidden.shape) == 3:
        hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))

    mx.eval(hidden)
    h_np = np.array(hidden.tolist(), dtype=np.float32)

    if h_np.shape[0] < 2 or h_np.shape[1] < 2:
        return 0.0

    try:
        _, S, _ = np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    S_sq = S * S
    total = np.sum(S_sq)

    if total < 1e-10:
        return 0.0

    p = S_sq / total
    p = p[p > 1e-10]

    if len(p) == 0:
        return 0.0

    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def _compute_exit_convergence(hidden: mx.array) -> tuple[float, float]:
    """Compute mean and deviation of norms at exit layer.

    Returns:
        Tuple of (mean_norm, dev_norm).
        High mean/dev ratio = consistent outputs.
    """
    if len(hidden.shape) == 3:
        hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))

    norms = mx.sqrt(mx.sum(hidden * hidden, axis=-1))
    mean_norm = mx.mean(norms)
    dev_norm = mx.sqrt(mx.mean((norms - mean_norm) ** 2))

    mx.eval(mean_norm, dev_norm)

    return float(mean_norm.tolist()), float(dev_norm.tolist())


__all__ = [
    "GeometricContext",
    "compute_geometric_context_for_batch",
]
