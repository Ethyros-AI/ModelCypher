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

"""Mode Connectivity and Loss Barrier analysis for merge compatibility.

Models in the same loss basin (low barrier) merge better than models in
disconnected modes (high barrier). This module provides tools for analyzing
the loss landscape between weight configurations.

Key Concepts:
    - **Mode**: A local minimum in the loss landscape
    - **Barrier**: Maximum loss along the path between two modes
    - **Connectivity**: Models are connected if barrier is low (same basin)

Use Cases:
    - Predict merge success before expensive operations
    - Choose between linear and geodesic interpolation
    - Identify when models are too different to merge

References:
    - Draxler et al. (2018) "Essentially No Barriers in Neural Network Energy Landscape"
    - Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"
    - Entezari et al. (2022) "The Role of Permutation Invariance in Linear Mode Connectivity"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Method for interpolating between weight configurations."""

    LINEAR = "linear"  # W(t) = (1-t)*W_0 + t*W_1
    GEODESIC = "geodesic"  # Riemannian geodesic (for rotation matrices)


@dataclass
class ModeConnectivityResult:
    """Result of mode connectivity analysis.

    Contains raw measurements only. No interpretations or recommendations.
    Threshold decisions are left to the caller.
    """

    # Loss values along the interpolation path
    path_losses: list[float]

    # Interpolation parameters (0=source, 1=target)
    path_t_values: list[float]

    # Barrier height: max(path_losses) - min(source_loss, target_loss)
    barrier_height: float

    # Normalized barrier: barrier_height / mean(endpoint_losses)
    # Lower indicates models are closer in the loss landscape.
    normalized_barrier: float

    # Loss at source (t=0)
    source_loss: float

    # Loss at target (t=1)
    target_loss: float

    # Location of maximum loss (t value)
    barrier_location: float

    # Interpolation method used
    method: InterpolationMethod


@dataclass
class LossBarrierProfile:
    """Detailed profile of the loss barrier between two weight configurations."""

    # Main connectivity result
    connectivity: ModeConnectivityResult

    # First derivative of loss (approximated from path)
    # Positive = loss increasing, Negative = loss decreasing
    gradient_sign_changes: int

    # Location of any local minima along the path (excluding endpoints)
    local_minima_t: list[float]

    # Is the path monotonic (no internal peaks)?
    is_monotonic: bool

    # Estimated Lipschitz constant (max |dL/dt|)
    lipschitz_estimate: float


def linear_interpolate(
    W0: "Array",
    W1: "Array",
    t: float,
    backend: "Backend",
) -> "Array":
    """Linear interpolation between weight matrices.

    W(t) = (1-t) * W0 + t * W1

    Parameters
    ----------
    W0 : Array
        Source weights.
    W1 : Array
        Target weights.
    t : float
        Interpolation parameter in [0, 1].
    backend : Backend
        Compute backend.

    Returns
    -------
    Array
        Interpolated weights.
    """
    t = max(0.0, min(1.0, t))
    result = (1 - t) * W0 + t * W1
    backend.eval(result)
    return result


def geodesic_interpolate(
    W0: "Array",
    W1: "Array",
    t: float,
    backend: "Backend",
) -> "Array":
    """Geodesic interpolation in flat weight space.

    Weight space is Euclidean (flat + spectral), so the geodesic reduces
    exactly to linear interpolation.
    """
    return linear_interpolate(W0, W1, t, backend)


def compute_path_losses(
    source_weights: "Array",
    target_weights: "Array",
    loss_fn: Callable[["Array"], float],
    n_steps: int | None = None,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    backend: "Backend | None" = None,
) -> tuple[list[float], list[float]]:
    """Compute losses along interpolation path.

    Parameters
    ----------
    source_weights : Array
        Source model weights.
    target_weights : Array
        Target model weights.
    loss_fn : Callable
        Function that takes weights and returns loss value.
    n_steps : int
        Number of points along the path (including endpoints). Must be provided
        explicitly (no heuristic default).
    method : InterpolationMethod
        Interpolation method to use.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    tuple[list[float], list[float]]
        (t_values, losses) along the path.
    """
    b = backend or get_default_backend()
    if n_steps is None:
        raise ValueError("n_steps must be provided explicitly")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    source = b.array(source_weights)
    target = b.array(target_weights)
    b.eval(source, target)

    # Select interpolation function
    if method == InterpolationMethod.LINEAR:
        interp_fn = linear_interpolate
    elif method == InterpolationMethod.GEODESIC:
        interp_fn = geodesic_interpolate
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    t_values = [i / (n_steps - 1) for i in range(n_steps)]
    losses = []

    for t in t_values:
        W_t = interp_fn(source, target, t, b)
        loss = loss_fn(W_t)
        losses.append(loss)

    return t_values, losses


def analyze_mode_connectivity(
    source_weights: "Array",
    target_weights: "Array",
    loss_fn: Callable[["Array"], float],
    n_steps: int | None = None,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    backend: "Backend | None" = None,
) -> ModeConnectivityResult:
    """Analyze mode connectivity between two weight configurations.

    Computes the loss barrier along the interpolation path. Returns raw
    measurements only - threshold interpretation is left to the caller.

    Parameters
    ----------
    source_weights : Array
        Source model weights.
    target_weights : Array
        Target model weights.
    loss_fn : Callable
        Function that takes weights and returns loss value.
    n_steps : int
        Number of points along the path. Must be provided explicitly.
    method : InterpolationMethod
        Interpolation method.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    ModeConnectivityResult
        Analysis result with raw barrier metrics.
    """
    t_values, losses = compute_path_losses(
        source_weights,
        target_weights,
        loss_fn,
        n_steps=n_steps,
        method=method,
        backend=backend,
    )

    source_loss = losses[0]
    target_loss = losses[-1]
    max_loss = max(losses)
    min_endpoint = min(source_loss, target_loss)
    max_idx = losses.index(max_loss)

    barrier_height = max_loss - min_endpoint
    mean_endpoint = (source_loss + target_loss) / 2

    if mean_endpoint > 0:
        normalized_barrier = barrier_height / mean_endpoint
    else:
        raise ValueError(
            "Cannot normalize barrier with non-positive mean endpoint loss "
            f"(source={source_loss}, target={target_loss})"
        )

    logger.info(
        "MODE CONNECTIVITY: barrier=%.4f, normalized=%.3f, source_loss=%.4f, target_loss=%.4f",
        barrier_height,
        normalized_barrier,
        source_loss,
        target_loss,
    )

    return ModeConnectivityResult(
        path_losses=losses,
        path_t_values=t_values,
        barrier_height=barrier_height,
        normalized_barrier=normalized_barrier,
        source_loss=source_loss,
        target_loss=target_loss,
        barrier_location=t_values[max_idx],
        method=method,
    )


def compute_loss_barrier_profile(
    source_weights: "Array",
    target_weights: "Array",
    loss_fn: Callable[["Array"], float],
    n_steps: int | None = None,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    backend: "Backend | None" = None,
) -> LossBarrierProfile:
    """Compute detailed loss barrier profile.

    Provides additional analysis beyond basic connectivity, including
    gradient information, local minima detection, and Lipschitz estimate.

    Parameters
    ----------
    source_weights : Array
        Source model weights.
    target_weights : Array
        Target model weights.
    loss_fn : Callable
        Function that takes weights and returns loss value.
    n_steps : int
        Number of points along the path (more = finer resolution). Must be
        provided explicitly.
    method : InterpolationMethod
        Interpolation method.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    LossBarrierProfile
        Detailed profile with gradient info and local minima.
    """
    connectivity = analyze_mode_connectivity(
        source_weights,
        target_weights,
        loss_fn,
        n_steps=n_steps,
        method=method,
        backend=backend,
    )

    losses = connectivity.path_losses
    t_values = connectivity.path_t_values
    n = len(losses)

    # Compute approximate gradients
    gradients = []
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        dL = losses[i] - losses[i - 1]
        gradients.append(dL / dt if dt > 0 else 0.0)

    # Count sign changes in gradient
    gradient_sign_changes = 0
    for i in range(1, len(gradients)):
        if gradients[i] * gradients[i - 1] < 0:
            gradient_sign_changes += 1

    # Find local minima (excluding endpoints)
    local_minima_t = []
    for i in range(1, n - 1):
        if losses[i] < losses[i - 1] and losses[i] < losses[i + 1]:
            local_minima_t.append(t_values[i])

    # Check monotonicity
    is_monotonic = gradient_sign_changes == 0 or (
        gradient_sign_changes == 1 and len(local_minima_t) == 0
    )

    # Estimate Lipschitz constant
    lipschitz_estimate = max(abs(g) for g in gradients) if gradients else 0.0

    return LossBarrierProfile(
        connectivity=connectivity,
        gradient_sign_changes=gradient_sign_changes,
        local_minima_t=local_minima_t,
        is_monotonic=is_monotonic,
        lipschitz_estimate=lipschitz_estimate,
    )


__all__ = [
    "InterpolationMethod",
    "ModeConnectivityResult",
    "LossBarrierProfile",
    "linear_interpolate",
    "geodesic_interpolate",
    "compute_path_losses",
    "analyze_mode_connectivity",
    "compute_loss_barrier_profile",
]
