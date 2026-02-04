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

"""Loop preservation loss for reasoning training.

Penalizes spectral entropy collapse during training. The key insight from
our research is that:
- Δβ₁ > 0 (loops grow) ↔ spectral entropy increases toward exit
- Δβ₁ < 0 (loops collapse) ↔ spectral entropy drops before exit

Since β₁ (ripser) is O(n³) and too slow for training, we use spectral
entropy trajectory as a differentiable proxy for topological complexity.

All parameters are derived from model geometry - no arbitrary constants:
- highway_layer: argmin(intrinsic_dimension) across layers
- lambda_scale: 1 / σ_max (inverse of largest singular value)
- base_delta_entropy: H_exit - H_highway (base model's trajectory)

The loss penalizes when training makes the entropy trajectory WORSE
than the base model. If base model has ΔH = +0.5 (healthy: entropy grows)
and adapter produces ΔH = -0.2 (collapse), the loss is:
    L = λ * max(0, 0.5 - (-0.2)) = λ * 0.7

This preserves the topological loop structure that correlates with
reasoning capability.
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
class LoopPreservationConfig:
    """All parameters derived from base model geometry.

    Attributes:
        highway_layer: Layer index where intrinsic dimension is minimum.
            This is the "highway entry" - the compression bottleneck.
        base_delta_entropy: Base model's H_exit - H_highway.
            Positive = healthy (entropy grows after highway).
            Negative = collapsing (entropy drops after highway).
        lambda_scale: 1 / σ_max of base model.
            Natural scale factor for the loss weight.
    """

    highway_layer: int
    base_delta_entropy: float
    lambda_scale: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/saving."""
        return {
            "highway_layer": self.highway_layer,
            "base_delta_entropy": self.base_delta_entropy,
            "lambda_scale": self.lambda_scale,
        }


def detect_highway_layer(
    model,
    tokenizer,
    probe_prompts: list[str],
) -> int:
    """Find highway entry point from intrinsic dimension profile.

    The highway is where the model compresses representations before
    the final reasoning/output layers. It's characterized by a dip
    in intrinsic dimension - information is concentrated.

    Uses IntrinsicDimension.compute_two_nn() which is O(n log n).

    Args:
        model: The loaded model.
        tokenizer: Tokenizer for the model.
        probe_prompts: Prompts to use for computing activations.
            More prompts = more stable estimate.

    Returns:
        Layer index where intrinsic dimension is minimum (highway entry).
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])
    n_layers = len(layers)

    if n_layers == 0:
        return 0

    # Collect activations at each layer for probe prompts
    layer_dims: list[float] = []
    estimator = IntrinsicDimension()

    for layer_idx in range(n_layers):
        # Collect hidden states at this layer across all prompts
        all_hidden = []

        for prompt in probe_prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids) if hasattr(tokens, "ids") else list(tokens)
            input_ids = mx.array([token_ids])

            # Forward pass to get hidden states
            hidden = _get_hidden_at_layer(model, input_ids, layer_idx)
            mx.eval(hidden)

            # Flatten to [n_tokens, hidden_dim]
            if len(hidden.shape) == 3:
                hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))

            all_hidden.append(hidden)

        # Concatenate all hidden states
        combined = mx.concatenate(all_hidden, axis=0)
        mx.eval(combined)

        # Compute intrinsic dimension
        try:
            points = np.array(combined.tolist(), dtype=np.float32)
            estimate = estimator.compute(mx.array(points))
            layer_dims.append(estimate.intrinsic_dimension)
        except Exception as e:
            logger.warning("Failed to compute ID at layer %d: %s", layer_idx, e)
            layer_dims.append(float("inf"))

    # Find layer with minimum intrinsic dimension
    if not layer_dims or all(d == float("inf") for d in layer_dims):
        # Fallback: use 1/3 of layers as rough estimate
        return n_layers // 3

    min_dim = min(d for d in layer_dims if d != float("inf"))
    highway = layer_dims.index(min_dim)

    logger.info(
        "Highway detected at layer %d (ID=%.2f, n_layers=%d)",
        highway,
        min_dim,
        n_layers,
    )

    return highway


def compute_base_entropy_trajectory(
    model,
    tokenizer,
    probe_prompts: list[str],
    highway_layer: int,
) -> float:
    """Compute base model's spectral entropy trajectory.

    Spectral entropy measures the distribution of singular values in
    the activation space. Higher entropy = more uniform distribution =
    more dimensions active = richer representation.

    Returns:
        ΔH = H_exit - H_highway (the baseline delta to preserve).
        Positive means entropy grows after highway (healthy reasoning).
        Negative means entropy collapses after highway (degraded).
    """
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])
    n_layers = len(layers)

    if n_layers == 0:
        return 0.0

    exit_layer = n_layers - 1
    highway_layer = min(highway_layer, n_layers - 1)

    # Collect entropies at highway and exit layers
    highway_entropies: list[float] = []
    exit_entropies: list[float] = []

    for prompt in probe_prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids) if hasattr(tokens, "ids") else list(tokens)
        input_ids = mx.array([token_ids])

        # Get hidden states at highway and exit
        h_highway = _get_hidden_at_layer(model, input_ids, highway_layer)
        h_exit = _get_hidden_at_layer(model, input_ids, exit_layer)
        mx.eval(h_highway, h_exit)

        # Compute spectral entropy for each
        highway_entropies.append(_compute_spectral_entropy(h_highway))
        exit_entropies.append(_compute_spectral_entropy(h_exit))

    # Average across prompts
    mean_highway = np.mean(highway_entropies) if highway_entropies else 0.0
    mean_exit = np.mean(exit_entropies) if exit_entropies else 0.0

    delta = mean_exit - mean_highway

    logger.info(
        "Base entropy trajectory: H_highway=%.4f, H_exit=%.4f, ΔH=%+.4f",
        mean_highway,
        mean_exit,
        delta,
    )

    return float(delta)


def loop_preservation_loss(
    current_trajectory: dict[int, float],
    config: LoopPreservationConfig,
) -> tuple[float, float]:
    """Compute loop preservation loss.

    Loss = λ * max(0, base_delta - current_delta)

    This penalizes when the current model's entropy trajectory is
    WORSE than the base model's. "Worse" means:
    - Base has positive ΔH (healthy) but current has lower or negative ΔH
    - The gap between base and current represents loop collapse

    Args:
        current_trajectory: Dict mapping layer_idx -> spectral_entropy.
            Must contain at least highway_layer and the exit layer.
        config: Loop preservation configuration from base model.

    Returns:
        Tuple of (loss_value, current_delta_entropy).
        loss_value is already scaled by λ.
    """
    if not current_trajectory:
        return 0.0, 0.0

    # Get current delta entropy
    layers = sorted(current_trajectory.keys())
    if len(layers) < 2:
        return 0.0, 0.0

    highway_layer = config.highway_layer
    exit_layer = max(layers)

    # Find closest layer to highway if exact not available
    if highway_layer not in current_trajectory:
        highway_layer = min(layers, key=lambda x: abs(x - config.highway_layer))

    h_highway = current_trajectory.get(highway_layer, 0.0)
    h_exit = current_trajectory.get(exit_layer, 0.0)

    current_delta = h_exit - h_highway

    # Loss: penalize when current is worse than base
    # If base_delta = 0.5 and current_delta = -0.2, loss = 0.7
    # If base_delta = 0.5 and current_delta = 0.6, loss = 0 (better)
    gap = config.base_delta_entropy - current_delta
    loss = config.lambda_scale * max(0.0, gap)

    return loss, current_delta


def compute_entropy_trajectory(
    model,
    input_ids: mx.array,
    layers_to_sample: list[int] | None = None,
) -> dict[int, float]:
    """Compute spectral entropy at each layer for a batch.

    This is the per-batch computation used during training.

    Args:
        model: The model (with or without LoRA adapters).
        input_ids: Input token IDs [batch, seq].
        layers_to_sample: Which layers to compute entropy at.
            If None, samples at highway (1/3), middle (1/2), exit (last).

    Returns:
        Dict mapping layer_idx -> spectral_entropy.
    """
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])
    n_layers = len(layers)

    if n_layers == 0:
        return {}

    # Default: sample at highway, middle, exit
    if layers_to_sample is None:
        highway = n_layers // 3
        middle = n_layers // 2
        exit_layer = n_layers - 1
        layers_to_sample = sorted(set([highway, middle, exit_layer]))

    trajectory: dict[int, float] = {}

    for layer_idx in layers_to_sample:
        if layer_idx >= n_layers:
            continue

        hidden = _get_hidden_at_layer(model, input_ids, layer_idx)
        mx.eval(hidden)

        entropy = _compute_spectral_entropy(hidden)
        trajectory[layer_idx] = entropy

    return trajectory


def derive_loop_config(
    model,
    tokenizer,
    probe_prompts: list[str],
    sigma_max: float,
) -> LoopPreservationConfig:
    """Derive complete loop preservation config from model geometry.

    This is the convenience function to call before training.

    Args:
        model: The base model (before any LoRA).
        tokenizer: Tokenizer for the model.
        probe_prompts: Prompts for computing activation geometry.
        sigma_max: Largest singular value from layer geometry
            (use geometries[target_modules[0]].sigma_max).

    Returns:
        LoopPreservationConfig with all parameters derived from geometry.
    """
    # Detect highway layer from intrinsic dimension profile
    highway_layer = detect_highway_layer(model, tokenizer, probe_prompts)

    # Compute base model's entropy trajectory
    base_delta_entropy = compute_base_entropy_trajectory(
        model, tokenizer, probe_prompts, highway_layer
    )

    # Lambda scale = 1 / σ_max (natural spectral scale)
    lambda_scale = 1.0 / max(sigma_max, 1e-8)

    config = LoopPreservationConfig(
        highway_layer=highway_layer,
        base_delta_entropy=base_delta_entropy,
        lambda_scale=lambda_scale,
    )

    logger.info("Loop preservation config: %s", config.to_dict())

    return config


def _get_hidden_at_layer(model, input_ids: mx.array, layer_idx: int) -> mx.array:
    """Get hidden states at a specific layer.

    Runs forward pass up to and including the specified layer.
    """
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])
    embed_module = getattr(base_model, "embed_tokens", None)

    if embed_module is None:
        raise ValueError("Could not find embed_tokens module")

    # Get embeddings
    h = embed_module(input_ids)
    mx.eval(h)

    # Forward through layers up to layer_idx
    for i in range(min(layer_idx + 1, len(layers))):
        result = layers[i](h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result
        mx.eval(h)

    return h


def _compute_spectral_entropy(hidden: mx.array) -> float:
    """Compute spectral entropy from hidden states.

    Spectral entropy = Shannon entropy of normalized singular values.
    H = -Σ p_i log(p_i) where p_i = σ_i² / Σσ²

    Higher entropy = more uniform singular values = richer representation.
    Lower entropy = concentrated singular values = compressed representation.
    """
    # Flatten to [n_samples, hidden_dim]
    if len(hidden.shape) == 3:
        hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))

    mx.eval(hidden)

    # Convert to numpy for SVD
    h_np = np.array(hidden.tolist(), dtype=np.float32)

    if h_np.shape[0] < 2 or h_np.shape[1] < 2:
        return 0.0

    # Compute singular values
    try:
        _, S, _ = np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    # Compute spectral entropy
    S_sq = S * S
    total = np.sum(S_sq)

    if total < 1e-10:
        return 0.0

    p = S_sq / total
    # Filter out near-zero probabilities to avoid log(0)
    p = p[p > 1e-10]

    if len(p) == 0:
        return 0.0

    entropy = -np.sum(p * np.log(p))

    return float(entropy)


__all__ = [
    "LoopPreservationConfig",
    "detect_highway_layer",
    "compute_base_entropy_trajectory",
    "loop_preservation_loss",
    "compute_entropy_trajectory",
    "derive_loop_config",
]
