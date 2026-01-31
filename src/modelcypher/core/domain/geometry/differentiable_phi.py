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

"""Differentiable phi loss for end-to-end geometric alignment training.

The comp/phi ratio measures reasoning alignment:
- comp/phi = 1.0: aligned reasoning (golden ratio expansion-compression)
- comp/phi != 1.0: geometry deviates from golden ratio

The TwoNN-based comp/phi uses k-NN which is non-differentiable. This module
provides a differentiable proxy using activation norm trajectories.

Key insight: The TRAJECTORY of activation norms IS differentiable.
We don't need to differentiate through TwoNN.

Mathematical basis:
    expansion_rate = (peak - initial) / peak_layer
    compression_rate = (peak - final) / (n_layers - peak_layer)
    comp_phi = compression_rate / (expansion_rate * phi)

    Loss = |comp_phi - 1.0|

No heuristics: All numerical guards are dtype-derived (sqrt(eps)).

References:
    Facco et al. (2017) "Estimating the intrinsic dimension of datasets"
    - TwoNN method for intrinsic dimension (non-differentiable ground truth)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Golden ratio - mathematical constant, not a heuristic
PHI = 1.618033988749895


def _sqrt_eps(dtype) -> float:
    """Return sqrt(machine epsilon) for the given dtype.

    This is the standard numerical analysis threshold for relative precision.
    Values below sqrt(eps) × scale are indistinguishable from roundoff noise.
    """
    eps = mx.finfo(dtype).eps
    return float(mx.sqrt(mx.array(eps)))


@dataclass
class PhiTrajectory:
    """Layer-wise activation norms with differentiable peak detection.

    Captures the geometric trajectory of activations through the model,
    enabling differentiable computation of the comp/phi ratio.

    Attributes:
        norms: Layer-wise L2 norms of activations [n_layers+1]
        soft_peak_val: Differentiable peak value (weighted by soft argmax)
        soft_peak_idx: Differentiable peak position (fractional layer index)
        initial_norm: Layer 0 (embedding) norm
        final_norm: Final layer norm
    """

    norms: mx.array
    soft_peak_val: mx.array
    soft_peak_idx: mx.array
    initial_norm: mx.array
    final_norm: mx.array


def compute_trajectory_norms(model: Any, input_ids: mx.array) -> mx.array:
    """Compute L2 norms of activations at each layer (fully differentiable).

    This is the core function that makes phi-loss trainable. All operations
    are MLX array operations, preserving the computation graph for backprop.

    Args:
        model: MLX language model with model.model.embed_tokens and model.model.layers
        input_ids: Token IDs [batch, seq_len] or [seq_len]

    Returns:
        Stacked L2 norms [n_layers + 1] including embedding layer

    Note:
        Does NOT call mx.eval() - keeps computation graph intact for gradients.
    """
    # Ensure 2D input
    if input_ids.ndim == 1:
        input_ids = mx.expand_dims(input_ids, axis=0)

    # Get base model (handle wrapper patterns)
    base_model = getattr(model, "model", model)
    embed_tokens = getattr(base_model, "embed_tokens", None)
    layers = getattr(base_model, "layers", None)

    if embed_tokens is None or layers is None:
        raise ValueError("Model must have model.embed_tokens and model.layers")

    # Embedding layer
    hidden = embed_tokens(input_ids)
    norms = [mx.sqrt(mx.sum(hidden * hidden))]

    # Transformer layers
    for layer in layers:
        hidden = layer(hidden, mask=None, cache=None)
        # Handle tuple return (hidden, cache)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        norms.append(mx.sqrt(mx.sum(hidden * hidden)))

    return mx.stack(norms)


def soft_argmax(values: mx.array) -> tuple[mx.array, mx.array]:
    """L2-weighted soft argmax - differentiable peak detection.

    Uses L2 (Euclidean) weighting: weight_i = (value_i - min + baseline)^2

    L2 is the natural metric for vector spaces (Euclidean norm). Power=2
    is not arbitrary - it corresponds to minimizing squared distance,
    the foundation of least squares regression.

    The baseline ensures non-zero weights for uniform values and is set
    to sqrt(eps) × scale, the precision floor for the dtype.

    Args:
        values: 1D array of values to find peak in

    Returns:
        (soft_peak_idx, soft_peak_val) where:
        - soft_peak_idx: Differentiable fractional index of peak
        - soft_peak_val: Differentiable value at peak

    Mathematical note:
        weights_i = (values_i - min + baseline)^2 / sum(...)
        soft_idx = sum(weights_i * i)
        soft_val = sum(weights_i * values_i)
    """
    n = values.shape[0]
    dtype = values.dtype

    # Dtype-derived precision floor
    eps = mx.finfo(dtype).eps

    # Shift to make minimum = 0 (all weights non-negative)
    min_val = mx.min(values)
    max_val = mx.max(values)
    scale = mx.maximum(mx.abs(max_val), mx.array(1.0))

    # Baseline: sqrt(eps) × scale ensures non-zero weights for uniform values
    # This is the precision floor - not a heuristic
    sqrt_eps = mx.sqrt(mx.array(eps))
    baseline = sqrt_eps * scale
    shifted = values - min_val + baseline

    # L2 weighting (power=2 is Euclidean, not arbitrary)
    weights = shifted * shifted
    weight_sum = mx.sum(weights)
    # Guard against division by zero with eps (not sqrt_eps)
    # Since baseline > 0, weight_sum >= n * baseline² > 0 always
    # The eps guard is for numerical safety only
    weights = weights / mx.maximum(weight_sum, mx.array(eps))

    # Weighted index (soft argmax)
    indices = mx.arange(n, dtype=dtype)
    soft_idx = mx.sum(weights * indices)

    # Weighted value
    soft_val = mx.sum(weights * values)

    return soft_idx, soft_val


def compute_phi_trajectory(model: Any, input_ids: mx.array) -> PhiTrajectory:
    """Compute full phi trajectory with soft peak detection.

    Combines trajectory norm computation with soft argmax to produce
    all quantities needed for phi-loss in a differentiable way.

    Args:
        model: MLX language model
        input_ids: Token IDs

    Returns:
        PhiTrajectory with norms, soft peak, and boundary values
    """
    norms = compute_trajectory_norms(model, input_ids)
    soft_peak_idx, soft_peak_val = soft_argmax(norms)

    return PhiTrajectory(
        norms=norms,
        soft_peak_val=soft_peak_val,
        soft_peak_idx=soft_peak_idx,
        initial_norm=norms[0],
        final_norm=norms[-1],
    )


def differentiable_phi_loss(trajectory: mx.array) -> tuple[mx.array, mx.array]:
    """Compute differentiable comp/phi loss.

    Loss = |comp/phi - 1.0|

    This is the ONLY loss. No peak position regularization - we let the
    geometry emerge from optimizing comp/phi = 1.0. Any auxiliary loss
    would inject a prior we don't have evidence for.

    Args:
        trajectory: Layer-wise norms from compute_trajectory_norms() [n_layers+1]

    Returns:
        (loss, comp_phi) where:
        - loss: |comp/phi - 1.0| for training
        - comp_phi: The computed comp/phi ratio for monitoring

    Mathematical basis:
        expansion_rate = (peak - initial) / peak_layer
        compression_rate = (peak - final) / (n_layers - peak_layer)
        comp_phi = compression_rate / (expansion_rate * phi)

    Numerical stability:
        All guards use sqrt(eps), the dtype-derived precision floor.
    """
    dtype = trajectory.dtype
    sqrt_eps = mx.array(_sqrt_eps(dtype))
    phi = mx.array(PHI)
    n = trajectory.shape[0]
    n_float = mx.array(float(n))

    # Soft peak detection
    soft_peak_idx, soft_peak_val = soft_argmax(trajectory)

    # Boundary values
    initial = trajectory[0]
    final = trajectory[-1]

    # Expansion rate (initial -> peak)
    # Guard: sqrt(eps) prevents division by zero while preserving gradients
    expansion_layers = mx.maximum(soft_peak_idx, sqrt_eps)
    expansion_rate = (soft_peak_val - initial) / expansion_layers

    # Compression rate (peak -> final)
    compression_layers = mx.maximum(n_float - soft_peak_idx - mx.array(1.0), sqrt_eps)
    compression_rate = (soft_peak_val - final) / compression_layers

    # comp/phi ratio
    denominator = mx.maximum(mx.abs(expansion_rate) * phi, sqrt_eps)
    comp_phi = compression_rate / denominator

    # Loss: distance from comp/phi = 1.0
    # This is the ONLY objective - no auxiliary losses
    loss = mx.abs(comp_phi - mx.array(1.0))

    return loss, comp_phi


def compute_phi_metrics(trajectory: mx.array, exact: bool = True) -> dict[str, float]:
    """Compute phi-related metrics for monitoring (non-training).

    This is the monitoring version that returns Python floats.
    Use differentiable_phi_loss() for training.

    Args:
        trajectory: Layer-wise norms [n_layers+1]
        exact: If True (default), use actual argmax for accurate monitoring.
               If False, use soft_argmax (matches training but less accurate).

    Returns:
        Dict with comp_phi and component metrics for analysis.
    """
    mx.eval(trajectory)

    dtype = trajectory.dtype
    sqrt_eps = _sqrt_eps(dtype)
    n = trajectory.shape[0]
    n_float = float(n)

    if exact:
        # Use actual argmax for accurate monitoring (non-differentiable)
        peak_idx_arr = mx.argmax(trajectory)
        mx.eval(peak_idx_arr)
        peak_idx = float(peak_idx_arr)
        peak_val = float(trajectory[int(peak_idx)])
    else:
        # Use soft_argmax (differentiable but less accurate)
        soft_peak_idx, soft_peak_val = soft_argmax(trajectory)
        mx.eval(soft_peak_idx, soft_peak_val)
        peak_idx = float(soft_peak_idx)
        peak_val = float(soft_peak_val)

    initial = float(trajectory[0])
    final = float(trajectory[-1])

    # Expansion rate
    expansion_layers = max(peak_idx, sqrt_eps)
    expansion_rate = (peak_val - initial) / expansion_layers

    # Compression rate
    compression_layers = max(n_float - peak_idx - 1.0, sqrt_eps)
    compression_rate = (peak_val - final) / compression_layers

    # comp/phi
    denominator = max(abs(expansion_rate) * PHI, sqrt_eps)
    comp_phi = compression_rate / denominator

    return {
        "comp_phi": comp_phi,
        "peak_layer": peak_idx,
        "peak_norm": peak_val,
        "expansion_rate": expansion_rate,
        "compression_rate": compression_rate,
        "initial_norm": initial,
        "final_norm": final,
        "n_layers": n - 1,  # Subtract 1 for embedding layer
    }


class PhiLossTracker:
    """Track phi metrics across training for monitoring.

    Records metrics during training to:
    - Monitor phi-loss convergence
    - Detect when training destabilizes geometry
    - Provide data for analysis (no heuristic decisions made here)
    """

    def __init__(self) -> None:
        """Initialize tracker with empty history."""
        self.history: list[dict] = []

    def record(self, metrics: dict[str, float], epoch: int, step: int) -> None:
        """Record metrics for a training step.

        Args:
            metrics: Dict from compute_phi_metrics()
            epoch: Current epoch
            step: Current step within epoch
        """
        record = {
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        self.history.append(record)

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics across all recorded steps.

        Returns measured statistics, no heuristic thresholds.
        The caller decides what to do with these numbers.
        """
        if not self.history:
            return {}

        comp_phis = [h["comp_phi"] for h in self.history]
        n = len(comp_phis)

        mean = sum(comp_phis) / n
        variance = sum((x - mean) ** 2 for x in comp_phis) / n
        std = variance**0.5

        return {
            "comp_phi_mean": mean,
            "comp_phi_std": std,
            "comp_phi_min": min(comp_phis),
            "comp_phi_max": max(comp_phis),
            "n_samples": n,
            "distance_from_target": abs(mean - 1.0),
        }
