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

"""CKA-based loss proxy for mode connectivity analysis.

Provides an activation-based loss function for mode_connectivity.py, enabling
loss barrier analysis without requiring labeled data or true loss computation.

The proxy measures how much interpolated model representations diverge from
source activations:

    loss(weights) = 1 - CKA(source_activations, activations_at_weights)

Key properties:
    - At source weights: CKA = 1.0, loss = 0
    - As weights interpolate: CKA decreases, loss increases
    - Barrier height = maximum divergence along path

Use case: LoRA insertion safety. Check if applying a LoRA pushes the model
out of its loss basin (high barrier = LoRA fighting base model structure).

References:
    - Kornblith et al. (2019) "Similarity of Neural Network Representations
      Revisited" - CKA as representation similarity metric
    - Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast
      Ensembling of DNNs" - mode connectivity and loss barriers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CKALossProxyConfig:
    """Configuration for CKA loss proxy."""

    # Whether to center activations before CKA (recommended)
    center: bool = True


def make_cka_loss_proxy(
    source_activations: "Array",
    forward_fn: Callable[["Array", "Array"], "Array"],
    probe_inputs: "Array",
    backend: "Backend | None" = None,
    config: CKALossProxyConfig | None = None,
) -> Callable[["Array"], float]:
    """Create a loss function for mode connectivity analysis.

    Returns a callable that computes:
        loss(weights) = 1 - CKA(source_activations, forward_fn(weights, probe_inputs))

    This enables using mode_connectivity.analyze_mode_connectivity() with
    activation-based divergence instead of requiring true loss computation.

    Args:
        source_activations: Reference activations [n_samples, d].
            These are cached and compared against at each evaluation.
        forward_fn: Function (weights, inputs) -> activations.
            Called at each point along the interpolation path.
        probe_inputs: Fixed probe inputs [n_samples, input_dim].
            Must produce activations compatible with source_activations.
        backend: Optional Backend for computation.
        config: Optional configuration.

    Returns:
        Callable that takes weights and returns loss (1 - CKA).

    Example:
        >>> source_weights = model.get_layer_weights(layer_idx)
        >>> source_acts = model.forward(probe_inputs, layer_idx)
        >>> loss_fn = make_cka_loss_proxy(source_acts, model.forward, probe_inputs, backend)
        >>> result = analyze_mode_connectivity(source_weights, target_weights, loss_fn)
    """
    b = backend or get_default_backend()
    cfg = config or CKALossProxyConfig()

    # Cast to precision dtype and cache source activations
    source = b.astype(source_activations, precision_dtype(b, reference=source_activations))
    probes = b.astype(probe_inputs, precision_dtype(b, reference=probe_inputs))
    b.eval(source, probes)

    # Center source if configured (improves CKA meaningfulness)
    if cfg.center:
        source_mean = b.mean(source, axis=0, keepdims=True)
        source_centered = source - source_mean
        b.eval(source_centered)
    else:
        source_centered = source

    def loss_fn(weights: "Array") -> float:
        """Compute CKA divergence loss: 1 - CKA(source, current)."""
        # Forward pass with current weights
        current_activations = forward_fn(weights, probes)
        current = b.astype(current_activations, precision_dtype(b, reference=current_activations))
        b.eval(current)

        # Center if configured
        if cfg.center:
            current_mean = b.mean(current, axis=0, keepdims=True)
            current_centered = current - current_mean
            b.eval(current_centered)
        else:
            current_centered = current

        # Compute linear CKA (fast, no geodesics)
        cka = compute_linear_cka_from_activations(source_centered, current_centered, b)

        # Return loss: higher = worse alignment
        loss = 1.0 - cka

        logger.debug("CKA loss proxy: cka=%.4f, loss=%.4f", cka, loss)
        return loss

    return loss_fn


def make_simple_cka_loss_proxy(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> Callable[["Array"], float]:
    """Create a simplified loss proxy for pure activation interpolation.

    This version doesn't require a forward function - it directly interpolates
    between source and target activations. Useful for testing and synthetic
    experiments.

    The loss measures divergence from source as t varies from 0 to 1:
        activations(t) = (1-t) * source + t * target
        loss(t) = 1 - CKA(source, activations(t))

    Args:
        source_activations: Source activations [n_samples, d].
        target_activations: Target activations [n_samples, d].
        backend: Optional Backend for computation.

    Returns:
        Callable that takes interpolation parameter t (as 1D array) and returns loss.
    """
    b = backend or get_default_backend()

    source = b.astype(source_activations, precision_dtype(b, reference=source_activations))
    target = b.astype(target_activations, precision_dtype(b, reference=target_activations))
    b.eval(source, target)

    # Center both
    source_mean = b.mean(source, axis=0, keepdims=True)
    target_mean = b.mean(target, axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    b.eval(source_centered, target_centered)

    def loss_fn(t_value: "Array") -> float:
        """Compute CKA loss at interpolation point t."""
        # Handle scalar or array input
        if hasattr(t_value, "shape") and len(b.shape(t_value)) > 0:
            t = float(b.to_scalar(t_value))
        else:
            t = float(t_value) if not isinstance(t_value, float) else t_value

        t = max(0.0, min(1.0, t))

        # Interpolate activations
        interpolated = (1.0 - t) * source_centered + t * target_centered
        b.eval(interpolated)

        # Compute CKA against source
        cka = compute_linear_cka_from_activations(source_centered, interpolated, b)

        return 1.0 - cka

    return loss_fn


__all__ = [
    "CKALossProxyConfig",
    "make_cka_loss_proxy",
    "make_simple_cka_loss_proxy",
]
