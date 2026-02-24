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

"""Hierarchical Optimal Transport (HOT) layer matcher.

Implements the SOTA approach to cross-architecture layer matching using
two-level optimal transport. Unlike Hungarian (rigid 1-to-1) or DP (monotonic),
HOT produces soft couplings that allow one layer to map to multiple layers.

Key properties:
- Soft coupling: Entries in [0,1], rows/columns sum to marginals
- Global optimization: Joint optimization at layer and neuron level
- Depth mismatch handling: Mass naturally distributes across multiple layers
- Emergent hierarchy: Diagonal structure emerges from optimization, not imposed

References:
    - Shah, S. & Khosla, M. (2025). "Representational Alignment Across Model
      Layers and Brain Regions with Hierarchical Optimal Transport."
      arXiv:2510.01706, ICLR 2026.
    - Cuturi, M. (2013). "Sinkhorn Distances." NeurIPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision_float32 as _promote_precision,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.optimal_transport import SinkhornSolver

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class HOTLayerMatchingResult:
    """Result of Hierarchical Optimal Transport layer matching.

    Unlike Hungarian (which produces 1-to-1 assignment), HOT produces
    soft couplings where mass can distribute across multiple layers.
    """

    # Soft coupling matrix [n_source_layers, n_target_layers]
    # Each entry in [0, 1], rows sum to source marginal, cols to target marginal
    layer_coupling: "Array"

    # Global alignment score (single scalar) - total transport cost
    alignment_score: float

    # Per-pair inner OT costs [n_source, n_target]
    pairwise_costs: dict[tuple[int, int], float]

    # Source and target layer indices used
    source_layers: list[int]
    target_layers: list[int]

    # Neuron-level transport plans (optional, for diagnostics)
    # Key: (source_layer_idx, target_layer_idx) -> transport plan
    neuron_transport: dict[tuple[int, int], "Array"] | None = None

    # Convergence info
    converged: bool = True
    iterations: int = 0


def _compute_correlation_cost(
    source: "Array",
    target: "Array",
    backend: "Backend",
) -> "Array":
    """Compute correlation-based cost matrix between neuron activations.

    Cost = 1 - correlation, which ranges from 0 (perfect correlation)
    to 2 (perfect anti-correlation).

    Args:
        source: Source activations [n_samples, n_source_neurons]
        target: Target activations [n_samples, n_target_neurons]
        backend: Backend for tensor operations.

    Returns:
        Cost matrix [n_source_neurons, n_target_neurons]
    """
    b = backend
    eps = division_epsilon(b, source)

    # Center the activations (subtract mean along sample axis)
    source_mean = b.mean(source, axis=0, keepdims=True)
    target_mean = b.mean(target, axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    b.eval(source_centered, target_centered)

    # Compute standard deviations
    n_samples = int(source.shape[0])
    source_std = b.sqrt(b.sum(source_centered * source_centered, axis=0) / n_samples)
    target_std = b.sqrt(b.sum(target_centered * target_centered, axis=0) / n_samples)
    b.eval(source_std, target_std)

    # Floor to avoid division by zero
    eps_arr_s = b.full(source_std.shape, eps)
    eps_arr_t = b.full(target_std.shape, eps)
    source_std = b.maximum(source_std, eps_arr_s)
    target_std = b.maximum(target_std, eps_arr_t)

    # Normalize
    source_norm = source_centered / b.reshape(source_std, (1, -1))
    target_norm = target_centered / b.reshape(target_std, (1, -1))
    b.eval(source_norm, target_norm)

    # Correlation matrix: corr[i,j] = (source[:,i] · target[:,j]) / n_samples
    # Shape: [n_source_neurons, n_target_neurons]
    corr = b.matmul(b.transpose(source_norm), target_norm) / n_samples
    b.eval(corr)

    # Clip correlation to [-1, 1]
    corr = b.clip(corr, -1.0, 1.0)
    b.eval(corr)

    # Cost = 1 - correlation (range [0, 2])
    cost = 1.0 - corr
    b.eval(cost)

    return cost


class HOTLayerMatcher:
    """Hierarchical Optimal Transport layer matcher.

    Performs two-level optimal transport:
    1. Inner OT: For each (source_layer, target_layer) pair, compute neuron-level
       transport plan and aggregate cost.
    2. Outer OT: Solve layer-level OT using aggregated costs to get soft coupling.

    Example:
        >>> matcher = HOTLayerMatcher(backend)
        >>> result = matcher.match(source_activations, target_activations)
        >>> # result.layer_coupling is a soft matrix, not 1-to-1 assignment
        >>> # To get the "best" mapping for a target layer:
        >>> best_source = backend.argmax(result.layer_coupling[:, target_idx])
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize HOT layer matcher.

        Args:
            backend: Backend for tensor operations. If None, uses default.
        """
        self._backend = backend or get_default_backend()
        self._sinkhorn = SinkhornSolver(self._backend)

    def match(
        self,
        source_layer_activations: dict[int, "Array"],
        target_layer_activations: dict[int, "Array"],
        store_neuron_transport: bool = False,
    ) -> HOTLayerMatchingResult:
        """Find soft layer correspondences using Hierarchical Optimal Transport.

        Unlike Hungarian matching, this returns a soft coupling matrix where
        one source layer can distribute mass across multiple target layers.
        This naturally handles depth mismatches and allows for many-to-many
        correspondences.

        Args:
            source_layer_activations: Source activations by layer index.
                Each value is [n_samples, n_neurons].
            target_layer_activations: Target activations by layer index.
                Each value is [n_samples, n_neurons].
            store_neuron_transport: If True, store neuron-level transport plans
                for all layer pairs (increases memory usage).

        Returns:
            HOTLayerMatchingResult with soft coupling and diagnostics.
        """
        b = self._backend

        source_layers = sorted(source_layer_activations.keys())
        target_layers = sorted(target_layer_activations.keys())

        n_source = len(source_layers)
        n_target = len(target_layers)

        logger.info(
            "HOT LAYER MATCH: Computing %d x %d layer pairs",
            n_source,
            n_target,
        )

        if n_source == 0 or n_target == 0:
            logger.warning("HOT LAYER MATCH: Empty layer activations")
            return HOTLayerMatchingResult(
                layer_coupling=b.zeros((n_source, n_target)),
                alignment_score=0.0,
                pairwise_costs={},
                source_layers=source_layers,
                target_layers=target_layers,
                neuron_transport=None,
                converged=True,
                iterations=0,
            )

        # Pre-process activations: stack and promote precision
        source_acts_cached: dict[int, "Array"] = {}
        for layer_idx in source_layers:
            acts = source_layer_activations[layer_idx]
            if isinstance(acts, list):
                acts = b.stack(acts, axis=0)
            acts = _promote_precision(acts, b)
            b.eval(acts)
            source_acts_cached[layer_idx] = acts

        target_acts_cached: dict[int, "Array"] = {}
        for layer_idx in target_layers:
            acts = target_layer_activations[layer_idx]
            if isinstance(acts, list):
                acts = b.stack(acts, axis=0)
            acts = _promote_precision(acts, b)
            b.eval(acts)
            target_acts_cached[layer_idx] = acts

        # Phase 1: Compute inner OT costs for all layer pairs
        pairwise_costs: dict[tuple[int, int], float] = {}
        neuron_transport: dict[tuple[int, int], "Array"] | None = (
            {} if store_neuron_transport else None
        )

        layer_cost_matrix = []

        for src_idx, src_layer in enumerate(source_layers):
            src_acts = source_acts_cached[src_layer]
            row_costs = []

            for tgt_idx, tgt_layer in enumerate(target_layers):
                tgt_acts = target_acts_cached[tgt_layer]

                # Ensure same sample count
                n_src = int(b.shape(src_acts)[0])
                n_tgt = int(b.shape(tgt_acts)[0])
                n_samples = min(n_src, n_tgt)

                if n_samples < 2:
                    # Degenerate case: use max cost
                    cost = 2.0  # Max correlation distance
                    pairwise_costs[(src_layer, tgt_layer)] = cost
                    row_costs.append(cost)
                    continue

                src_subset = src_acts[:n_samples]
                tgt_subset = tgt_acts[:n_samples]
                b.eval(src_subset, tgt_subset)

                # Compute neuron-level cost matrix (correlation distance)
                neuron_cost = _compute_correlation_cost(src_subset, tgt_subset, b)

                # Solve inner OT (neuron level)
                inner_result = self._sinkhorn.solve(neuron_cost)

                # Aggregate cost for this layer pair
                cost = inner_result.cost
                pairwise_costs[(src_layer, tgt_layer)] = cost
                row_costs.append(cost)

                # Store neuron transport if requested
                if neuron_transport is not None:
                    neuron_transport[(src_layer, tgt_layer)] = inner_result.plan

            layer_cost_matrix.append(row_costs)

        # Build layer-level cost matrix
        layer_cost_arr = b.array(layer_cost_matrix)
        b.eval(layer_cost_arr)

        # Phase 2: Solve outer OT (layer level) with uniform marginals
        work_dtype = precision_dtype(b, reference=layer_cost_arr)
        source_marginal = b.ones((n_source,), dtype=work_dtype) / float(n_source)
        target_marginal = b.ones((n_target,), dtype=work_dtype) / float(n_target)
        b.eval(source_marginal, target_marginal)

        outer_result = self._sinkhorn.solve(
            layer_cost_arr,
            source_marginal=source_marginal,
            target_marginal=target_marginal,
        )

        # The layer coupling is the transport plan scaled to have interpretable values
        # Each entry represents how much "mass" flows from source layer to target layer
        layer_coupling = outer_result.plan
        b.eval(layer_coupling)

        # Total alignment score is the total transport cost
        alignment_score = outer_result.cost

        logger.info(
            "HOT LAYER MATCH: Completed, alignment_score=%.4f, converged=%s",
            alignment_score,
            outer_result.converged,
        )

        # Log the dominant correspondences
        for tgt_idx, tgt_layer in enumerate(target_layers):
            col = b.take(layer_coupling, b.array([tgt_idx]), axis=1)
            col = b.reshape(col, (-1,))
            best_src_idx = b.argmax(col)
            b.eval(best_src_idx)
            best_src_idx_val = int(b.to_scalar(best_src_idx))
            best_src_layer = source_layers[best_src_idx_val]
            mass = b.take(col, best_src_idx)
            b.eval(mass)
            mass_val = float(b.to_scalar(mass))
            logger.info(
                "  target[%d] <- source[%d] (mass=%.4f)",
                tgt_layer,
                best_src_layer,
                mass_val,
            )

        return HOTLayerMatchingResult(
            layer_coupling=layer_coupling,
            alignment_score=alignment_score,
            pairwise_costs=pairwise_costs,
            source_layers=source_layers,
            target_layers=target_layers,
            neuron_transport=neuron_transport,
            converged=outer_result.converged,
            iterations=outer_result.iterations,
        )


def hot_layer_matching(
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    backend: "Backend | None" = None,
) -> HOTLayerMatchingResult:
    """Convenience function for HOT layer matching.

    Args:
        source_layer_activations: Source activations by layer index.
        target_layer_activations: Target activations by layer index.
        backend: Optional backend. If None, uses default.

    Returns:
        HOTLayerMatchingResult with soft layer coupling.
    """
    matcher = HOTLayerMatcher(backend)
    return matcher.match(source_layer_activations, target_layer_activations)


def coupling_to_assignment(
    layer_coupling: "Array",
    source_layers: list[int],
    target_layers: list[int],
    backend: "Backend",
) -> dict[int, int]:
    """Convert soft coupling to hard 1-to-1 assignment (for compatibility).

    This extracts the dominant source layer for each target layer.
    Use this when you need a discrete mapping, but note that this
    loses the soft coupling information.

    Args:
        layer_coupling: Soft coupling matrix [n_source, n_target].
        source_layers: List of source layer indices.
        target_layers: List of target layer indices.
        backend: Backend for tensor operations.

    Returns:
        Mapping target_layer_idx -> source_layer_idx.
    """
    b = backend
    mapping: dict[int, int] = {}

    for tgt_idx, tgt_layer in enumerate(target_layers):
        col = b.take(layer_coupling, b.array([tgt_idx]), axis=1)
        col = b.reshape(col, (-1,))
        best_src_idx = b.argmax(col)
        b.eval(best_src_idx)
        best_src_idx_val = int(b.to_scalar(best_src_idx))

        if best_src_idx_val < len(source_layers):
            mapping[tgt_layer] = source_layers[best_src_idx_val]

    return mapping


__all__ = [
    "HOTLayerMatchingResult",
    "HOTLayerMatcher",
    "hot_layer_matching",
    "coupling_to_assignment",
]
