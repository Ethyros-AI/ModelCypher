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

"""
Birkhoff Router for multi-channel model merging.

This module implements multi-channel routing via doubly stochastic matrices,
enabling stable combination of knowledge channels from multiple source models.
Based on DeepSeek mHC (arXiv:2512.24880) principles.

Key insight from mHC-null-space connection (docs/research/mhc_null_space_connection.md):
Both doubly stochastic routing and null-space projection are "invariant-preserving
projections onto constrained manifolds." Combining them enables multi-modal
knowledge addition without interference.

Properties:
    1. CKA = 1.0 per channel (null-space preserves geometry)
    2. Stable combination (doubly stochastic spectral norm ≤ 1.0)
    3. No interference (channels add, not blend)

Usage:
    router = BirkhoffRouter(backend)
    result = router.compute_routing(channel_deltas)
    combined = router.apply_routing(result.routing_matrix, channel_deltas)

References:
    - DeepSeek-AI (2025), mHC: Manifold-Constrained Hyper-Connections
    - docs/research/mhc_null_space_connection.md
    - docs/DIMENSIONAL_COMPRESSION.md (Multi-Modal Extension)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.birkhoff_projector import BirkhoffProjector
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
    geodesic_svd,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Initialization mode for routing matrix."""

    UNIFORM = "uniform"  # Equal weight to all channels (1/n)
    IDENTITY = "identity"  # Each output gets its corresponding input only
    DIAGONAL_WEIGHTED = "diagonal_weighted"  # Weighted diagonal based on delta norms


@dataclass
class BirkhoffRoutingResult:
    """Result of computing Birkhoff routing for multiple channels."""

    # The doubly stochastic routing matrix [n_channels, n_channels]
    routing_matrix: Any

    # Number of channels routed
    n_channels: int

    # Number of Sinkhorn iterations used
    iterations_used: int

    # Final max deviation from doubly stochastic (should be < epsilon)
    convergence_error: float

    # Whether Sinkhorn converged within max iterations
    converged: bool

    # Spectral norm of routing matrix (should be ≤ 1.0)
    spectral_norm: float

    # Whether spectral clipping was applied
    spectral_clipped: bool

    # Initialization mode used
    init_mode: RoutingMode


@dataclass
class ApplyRoutingResult:
    """Result of applying routing to channel deltas."""

    # Combined delta after routing
    combined_delta: Any

    # Per-channel contributions [n_channels, *delta_shape]
    channel_contributions: list[Any]

    # Input norms before routing
    input_norms: list[float]

    # Output norm after routing
    output_norm: float

    # Routing matrix used
    routing_matrix: Any


class BirkhoffRouter:
    """
    Routes multiple knowledge channels via doubly stochastic mixing.

    This class implements multi-channel routing for model merging, where
    each channel represents knowledge from a different source model
    (e.g., spatial from world model, temporal from video model, text from LLM).

    The routing matrix H satisfies:
    - All entries ≥ 0
    - All rows sum to 1
    - All columns sum to 1
    - Spectral norm ≤ 1.0

    These properties ensure:
    1. Stable combination (no signal explosion)
    2. Conservation (total information preserved)
    3. Bounded mixing (no single channel dominates unboundedly)

    Mathematical Foundation:
        From the unified formula in mhc_null_space_connection.md:

        combined_delta = Σ_i Σ_j H[i,j] × δW_safe_j

        Where:
        - H is doubly stochastic [n_channels, n_channels]
        - δW_safe_j is the null-space-projected delta for channel j
        - The result is a weighted sum with bounded total weight

    Integration with Null-Space:
        This router is designed to work with GeodesicNullSpaceFilter:

        1. Each channel's delta is projected to target's null space (CKA=1.0)
        2. Channels are combined via this router (stable mixing)
        3. Result is added to target weights (geometric addition)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._projector = BirkhoffProjector(self._backend)

    def compute_routing(
        self,
        channel_deltas: list["Array"],
        *,
        init_mode: RoutingMode | str = RoutingMode.UNIFORM,
    ) -> BirkhoffRoutingResult:
        """
        Compute doubly stochastic routing matrix for channel combination.

        The routing matrix H[i,j] determines how much of channel j contributes
        to output channel i. Since we typically want the combined output
        (not per-channel outputs), the actual combination uses H.sum(axis=0)
        as effective weights per input channel.

        Args:
            channel_deltas: List of n channel deltas [n_channels x weight_shape].
            init_mode: How to initialize the routing matrix before projection.
                - "uniform": All channels contribute equally (1/n)
                - "identity": Each output gets only its corresponding input
                - "diagonal_weighted": Weight based on delta norms

        Returns:
            BirkhoffRoutingResult with routing matrix and diagnostics.
        """
        backend = self._backend
        n = len(channel_deltas)

        if n == 0:
            raise ValueError("At least one channel delta required")

        if isinstance(init_mode, str):
            init_mode = RoutingMode(init_mode)

        # Initialize routing matrix based on mode
        H_init = self._initialize_routing(channel_deltas, init_mode)
        backend.eval(H_init)

        # Project to Birkhoff polytope using existing BirkhoffProjector
        result = self._projector.project(H_init, ensure_positive=False)

        return BirkhoffRoutingResult(
            routing_matrix=result.projected_matrix,
            n_channels=n,
            iterations_used=result.iterations_used,
            convergence_error=result.max_marginal_error,
            converged=result.converged,
            spectral_norm=result.spectral_norm_after,
            spectral_clipped=result.spectral_clipped,
            init_mode=init_mode,
        )

    def _initialize_routing(
        self,
        channel_deltas: list["Array"],
        mode: RoutingMode,
    ) -> "Array":
        """Initialize routing matrix before Birkhoff projection."""
        backend = self._backend
        n = len(channel_deltas)

        if mode == RoutingMode.UNIFORM:
            # Equal weight: H[i,j] = 1/n
            return backend.full((n, n), 1.0 / n)

        elif mode == RoutingMode.IDENTITY:
            # Identity: H[i,j] = δ_ij (only diagonal)
            return backend.eye(n)

        elif mode == RoutingMode.DIAGONAL_WEIGHTED:
            # Weight diagonal by delta norms, then add small uniform component
            # This biases toward keeping channels separate while allowing mixing
            norms = []
            for delta in channel_deltas:
                delta_arr = backend.array(delta)
                flat = backend.reshape(delta_arr, (-1,))
                norm_sq = backend.sum(flat * flat)
                backend.eval(norm_sq)
                norms.append(float(backend.to_scalar(norm_sq)) ** 0.5)

            total_norm = sum(norms)
            if total_norm > 0:
                weights = [norm / total_norm for norm in norms]
            else:
                weights = [1.0 / n] * n

            # Create diagonal-heavy matrix with small off-diagonal mixing
            eps = division_epsilon(backend, channel_deltas[0])
            H = backend.full((n, n), eps)  # Small off-diagonal
            for i in range(n):
                # Set diagonal element
                idx_i = backend.array([i])
                row = backend.take(H, idx_i, axis=0)
                row = backend.reshape(row, (n,))
                # Update diagonal via full matrix construction
                H_list = []
                for j in range(n):
                    if i == j:
                        H_list.append(weights[i])
                    else:
                        H_list.append(eps)
                H_row = backend.array(H_list)
                H_list_2d = []
                for k in range(n):
                    if k == i:
                        H_list_2d.append(H_row)
                    else:
                        idx_k = backend.array([k])
                        H_list_2d.append(
                            backend.reshape(backend.take(H, idx_k, axis=0), (n,))
                        )
                H = backend.stack(H_list_2d, axis=0)
            backend.eval(H)
            return H

        else:
            raise ValueError(f"Unknown routing mode: {mode}")

    def apply_routing(
        self,
        routing_matrix: "Array",
        channel_deltas: list["Array"],
    ) -> ApplyRoutingResult:
        """
        Apply routing matrix to combine channel deltas.

        The combined delta is computed as:
            combined = Σ_j (Σ_i H[i,j]) × δW_j
                     = Σ_j col_sum[j] × δW_j

        Since H is doubly stochastic, Σ_i H[i,j] = 1 for all j, so this
        effectively averages the channels. For uniform H, this gives equal
        weight to each channel.

        Note: For multi-output scenarios (different target channels), use
        the full H[i,j] × δW_j formulation instead.

        Args:
            routing_matrix: Doubly stochastic [n_channels, n_channels].
            channel_deltas: List of n channel deltas.

        Returns:
            ApplyRoutingResult with combined delta and diagnostics.
        """
        backend = self._backend
        n = len(channel_deltas)

        if n == 0:
            raise ValueError("At least one channel delta required")

        routing_matrix = backend.array(routing_matrix)
        backend.eval(routing_matrix)

        # Compute input norms
        input_norms = []
        for delta in channel_deltas:
            delta_arr = backend.array(delta)
            flat = backend.reshape(delta_arr, (-1,))
            norm_sq = backend.sum(flat * flat)
            backend.eval(norm_sq)
            input_norms.append(float(backend.to_scalar(norm_sq)) ** 0.5)

        # For single-output combination: use column sums as weights
        # Column sum of doubly stochastic = 1, so this is effectively
        # an equal-weighted combination
        col_sums = backend.sum(routing_matrix, axis=0)  # [n_channels]
        backend.eval(col_sums)

        # Combine deltas: combined = Σ_j col_sum[j] × δW_j
        # Start with zeros of same shape as first delta
        combined = backend.zeros_like(channel_deltas[0])
        channel_contributions = []

        for j in range(n):
            delta_j = backend.array(channel_deltas[j])
            idx_j = backend.array([j])
            weight_j = backend.take(col_sums, idx_j, axis=0)
            weight_j = backend.reshape(weight_j, (1,) * len(delta_j.shape))
            contribution = weight_j * delta_j
            backend.eval(contribution)
            channel_contributions.append(contribution)
            combined = combined + contribution

        backend.eval(combined)

        # Compute output norm
        flat_combined = backend.reshape(combined, (-1,))
        output_norm_sq = backend.sum(flat_combined * flat_combined)
        backend.eval(output_norm_sq)
        output_norm = float(backend.to_scalar(output_norm_sq)) ** 0.5

        return ApplyRoutingResult(
            combined_delta=combined,
            channel_contributions=channel_contributions,
            input_norms=input_norms,
            output_norm=output_norm,
            routing_matrix=routing_matrix,
        )

    def route_channels(
        self,
        channel_deltas: list["Array"],
        *,
        init_mode: RoutingMode | str = RoutingMode.UNIFORM,
    ) -> tuple["Array", BirkhoffRoutingResult]:
        """
        Convenience method: compute routing and apply in one call.

        This is the typical entry point for multi-channel merging:

            combined, routing_result = router.route_channels(channel_deltas)
            merged_weights = target_weights + combined

        Args:
            channel_deltas: List of per-channel deltas (already null-space filtered).
            init_mode: Routing initialization mode.

        Returns:
            Tuple of (combined_delta, routing_result).
        """
        routing_result = self.compute_routing(channel_deltas, init_mode=init_mode)
        apply_result = self.apply_routing(routing_result.routing_matrix, channel_deltas)
        return apply_result.combined_delta, routing_result


__all__ = [
    "ApplyRoutingResult",
    "BirkhoffRouter",
    "BirkhoffRoutingResult",
    "RoutingMode",
]
