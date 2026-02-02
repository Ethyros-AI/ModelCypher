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

"""Differentiable expansion ratio loss for geometric alignment training.

EXPERIMENTAL: This loss function assumes expansion_ratio=1.0 is optimal.
This hypothesis is under investigation. Before using for training:
1. Measure natural expansion_ratio distribution for your task types
2. Consider whether a single target value makes sense for your use case
3. See scripts/measure_expansion_distribution.py for empirical data gathering

Measures geometric expansion/compression cycle:
- expansion_ratio = 1.0: balanced expansion and compression
- expansion_ratio != 1.0: asymmetric geometry

Mathematical basis:
    expansion_rate = (peak - initial) / peak_layer
    compression_rate = (peak - final) / (n_layers - peak_layer)
    expansion_ratio = compression_rate / expansion_rate

    Loss = |expansion_ratio - 1.0|

All numerical guards are dtype-derived (sqrt(eps)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _sqrt_eps(dtype) -> float:
    """Return sqrt(machine epsilon) for the given dtype."""
    eps = mx.finfo(dtype).eps
    return float(mx.sqrt(mx.array(eps)))


@dataclass
class ExpansionTrajectory:
    """Layer-wise activation norms with differentiable peak detection."""

    norms: mx.array
    soft_peak_val: mx.array
    soft_peak_idx: mx.array
    initial_norm: mx.array
    final_norm: mx.array


def compute_trajectory_norms(model: Any, input_ids: mx.array) -> mx.array:
    """Compute L2 norms of activations at each layer (fully differentiable)."""
    if input_ids.ndim == 1:
        input_ids = mx.expand_dims(input_ids, axis=0)

    base_model = getattr(model, "model", model)
    embed_tokens = getattr(base_model, "embed_tokens", None)
    layers = getattr(base_model, "layers", None)

    if embed_tokens is None or layers is None:
        raise ValueError("Model must have model.embed_tokens and model.layers")

    hidden = embed_tokens(input_ids)
    norms = [mx.sqrt(mx.sum(hidden * hidden))]

    for layer in layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        norms.append(mx.sqrt(mx.sum(hidden * hidden)))

    return mx.stack(norms)


def soft_argmax(values: mx.array) -> tuple[mx.array, mx.array]:
    """L2-weighted soft argmax - differentiable peak detection."""
    n = values.shape[0]
    dtype = values.dtype
    eps = mx.finfo(dtype).eps

    min_val = mx.min(values)
    max_val = mx.max(values)
    scale = mx.maximum(mx.abs(max_val), mx.array(1.0))

    sqrt_eps = mx.sqrt(mx.array(eps))
    baseline = sqrt_eps * scale
    shifted = values - min_val + baseline

    weights = shifted * shifted
    weight_sum = mx.sum(weights)
    weights = weights / mx.maximum(weight_sum, mx.array(eps))

    indices = mx.arange(n, dtype=dtype)
    soft_idx = mx.sum(weights * indices)
    soft_val = mx.sum(weights * values)

    return soft_idx, soft_val


def compute_expansion_trajectory(model: Any, input_ids: mx.array) -> ExpansionTrajectory:
    """Compute full expansion trajectory with soft peak detection."""
    norms = compute_trajectory_norms(model, input_ids)
    soft_peak_idx, soft_peak_val = soft_argmax(norms)

    return ExpansionTrajectory(
        norms=norms,
        soft_peak_val=soft_peak_val,
        soft_peak_idx=soft_peak_idx,
        initial_norm=norms[0],
        final_norm=norms[-1],
    )


def differentiable_expansion_loss(trajectory: mx.array) -> tuple[mx.array, mx.array]:
    """Compute differentiable expansion ratio loss.

    Loss = |expansion_ratio - 1.0|

    Returns:
        (loss, expansion_ratio)
    """
    dtype = trajectory.dtype
    sqrt_eps = mx.array(_sqrt_eps(dtype))
    n = trajectory.shape[0]
    n_float = mx.array(float(n))

    soft_peak_idx, soft_peak_val = soft_argmax(trajectory)

    initial = trajectory[0]
    final = trajectory[-1]

    expansion_layers = mx.maximum(soft_peak_idx, sqrt_eps)
    expansion_rate = (soft_peak_val - initial) / expansion_layers

    compression_layers = mx.maximum(n_float - soft_peak_idx - mx.array(1.0), sqrt_eps)
    compression_rate = (soft_peak_val - final) / compression_layers

    denominator = mx.maximum(mx.abs(expansion_rate), sqrt_eps)
    expansion_ratio = compression_rate / denominator

    loss = mx.abs(expansion_ratio - mx.array(1.0))

    return loss, expansion_ratio


def compute_expansion_metrics(trajectory: mx.array, exact: bool = True) -> dict[str, float]:
    """Compute expansion metrics for monitoring (non-training)."""
    mx.eval(trajectory)

    dtype = trajectory.dtype
    sqrt_eps = _sqrt_eps(dtype)
    n = trajectory.shape[0]
    n_float = float(n)

    if exact:
        peak_idx_arr = mx.argmax(trajectory)
        mx.eval(peak_idx_arr)
        peak_idx = float(peak_idx_arr)
        peak_val = float(trajectory[int(peak_idx)])
    else:
        soft_peak_idx, soft_peak_val = soft_argmax(trajectory)
        mx.eval(soft_peak_idx, soft_peak_val)
        peak_idx = float(soft_peak_idx)
        peak_val = float(soft_peak_val)

    initial = float(trajectory[0])
    final = float(trajectory[-1])

    expansion_layers = max(peak_idx, sqrt_eps)
    expansion_rate = (peak_val - initial) / expansion_layers

    compression_layers = max(n_float - peak_idx - 1.0, sqrt_eps)
    compression_rate = (peak_val - final) / compression_layers

    denominator = max(abs(expansion_rate), sqrt_eps)
    expansion_ratio = compression_rate / denominator

    return {
        "expansion_ratio": expansion_ratio,
        "peak_layer": peak_idx,
        "peak_norm": peak_val,
        "expansion_rate": expansion_rate,
        "compression_rate": compression_rate,
        "initial_norm": initial,
        "final_norm": final,
        "n_layers": n - 1,
    }


class ExpansionLossTracker:
    """Track expansion metrics across training."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def record(self, metrics: dict[str, float], epoch: int, step: int) -> None:
        record = {"epoch": epoch, "step": step, **metrics}
        self.history.append(record)

    def get_summary(self) -> dict[str, float]:
        if not self.history:
            return {}

        ratios = [h["expansion_ratio"] for h in self.history if "expansion_ratio" in h]
        if not ratios:
            return {}

        n = len(ratios)
        mean = sum(ratios) / n
        variance = sum((x - mean) ** 2 for x in ratios) / n
        std = variance**0.5

        return {
            "expansion_ratio_mean": mean,
            "expansion_ratio_std": std,
            "expansion_ratio_min": min(ratios),
            "expansion_ratio_max": max(ratios),
            "n_samples": n,
            "distance_from_target": abs(mean - 1.0),
        }
