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

"""Trajectory-based manifold mapping with rank saturation detection.

This module implements geometric profiling of LLM activation spaces using
trajectory-based sampling. Unlike text-probe-at-a-time approaches, this:

1. Collects FULL trajectories (all token positions, not mean-pooled)
2. Computes velocities (first differences) to capture dynamics
3. Uses rank saturation detection for geometric termination
4. Employs domain-stratified sampling from atlas probes

The key insight: a 100-token text yields 199 samples (100 positions + 99 velocities)
instead of 1. This samples the manifold 200x more densely per forward pass.

Usage:
    mapper = ManifoldMapper(backend, activation_provider)
    result = mapper.map_manifold(model, tokenizer, probes)

    # result.profiles[layer_idx].trajectory_rank is the TRUE geometric ceiling
    # result.profiles[layer_idx].batches_to_saturation shows when rank stabilized
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    compute_numerical_rank,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.agents.unified_atlas import AtlasProbe
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerManifoldProfile:
    """Per-layer manifold geometry from trajectory mapping."""

    layer_idx: int
    hidden_dim: int

    # Rank measurements (from SVD with sqrt(eps) threshold)
    trajectory_rank: int  # True geometric ceiling (positions + velocities)
    activation_rank: int  # Achieved rank at saturation (may equal trajectory_rank)
    null_rank: int  # hidden_dim - trajectory_rank

    # Sample counts
    total_samples: int  # positions + velocities
    position_samples: int  # Raw token positions
    velocity_samples: int  # First differences

    # Domain coverage
    domains_sampled: set[str]
    probes_processed: int

    # Numerical stability
    gram_condition: float  # Condition number of Gram matrix

    # Saturation metrics
    batches_to_saturation: int  # How many batches until rank stabilized
    saturated: bool  # Whether this layer reached saturation

    # Weight matrix ranks (structural capacity) - fields with defaults must come last
    # These tell us what the layer CAN do, independent of what probes activate
    weight_rank_o_proj: int = 0  # Rank of attention output projection
    weight_rank_down_proj: int = 0  # Rank of MLP down projection
    structural_capacity: int = 0  # Min of weight ranks (true ceiling)

    # Capacity verification
    @property
    def is_probe_limited(self) -> bool:
        """True if probes didn't fully activate the structural capacity.

        If activation_rank < structural_capacity, our text probes don't
        cover the full space the model CAN use. This suggests we need
        more diverse probes, not that the capacity is unused.
        """
        return self.structural_capacity > 0 and self.activation_rank < self.structural_capacity

    @property
    def unused_capacity(self) -> int:
        """Dimensions the model CAN use but probes didn't activate."""
        if self.structural_capacity > 0:
            return max(0, self.structural_capacity - self.activation_rank)
        return 0


@dataclass
class ManifoldMapResult:
    """Result of manifold mapping operation."""

    # Per-layer profiles
    profiles: dict[int, LayerManifoldProfile]

    # Global metrics
    total_probes_processed: int
    total_batches: int
    all_layers_saturated: bool

    # Domain coverage (global)
    domains_covered: set[str]

    # Stored activations for merge reuse
    # positions[layer_idx] = [total_samples, hidden_dim]
    positions: dict[int, "Array"]
    velocities: dict[int, "Array"]


@dataclass
class _LayerState:
    """Internal state for tracking a layer during mapping."""

    layer_idx: int
    hidden_dim: int
    activation_rank: int = 0
    previous_rank: int = 0
    saturated_count: int = 0
    saturated: bool = False
    batches_to_saturation: int = 0
    domains_seen: set[str] = field(default_factory=set)
    probes_processed: int = 0


class ManifoldMapper:
    """Maps activation manifolds using trajectory-based sampling with rank saturation.

    This is the geometric alternative to text-probe-at-a-time profiling.
    Instead of running 4,596 probes and mean-pooling each to a single vector,
    we run batches of diverse probes and collect FULL trajectories.

    Key features:
    - Domain-stratified sampling ensures coverage of all semantic categories
    - Rank saturation detection provides geometric termination (not iteration limit)
    - Velocities (first differences) capture manifold dynamics
    - Single forward pass per batch minimizes inference overhead

    The mapping stops when all layers reach rank saturation:
    delta_rank < 1 for K consecutive batches = saturation
    """

    # Consecutive batches with no rank increase = saturation
    K_CONSECUTIVE = 3

    # Default batch size (probes per batch)
    DEFAULT_BATCH_SIZE = 20

    def __init__(
        self,
        backend: "Backend | None" = None,
        activation_provider: "ActivationProvider | None" = None,
    ):
        """Initialize ManifoldMapper.

        Args:
            backend: Backend for tensor operations. Uses default if None.
            activation_provider: Provider for collecting trajectories.
                                If None, creates MLXActivationProvider.
        """
        self._backend = backend or get_default_backend()

        if activation_provider is None:
            from modelcypher.adapters.mlx_activation_provider import (
                MLXActivationProvider,
            )

            activation_provider = MLXActivationProvider()
        self._activation_provider = activation_provider

    def map_manifold(
        self,
        model: Any,
        tokenizer: Any,
        probes: list["AtlasProbe"],
        batch_size: int | None = None,
        max_batches: int | None = None,
    ) -> ManifoldMapResult:
        """Map the model's activation manifold with rank saturation detection.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding probe texts.
            probes: List of atlas probes (with domain labels).
            batch_size: Probes per batch. Default 20.
            max_batches: Optional maximum batches (for testing). None = no limit.

        Returns:
            ManifoldMapResult with per-layer profiles and stored activations.
        """
        b = self._backend
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE

        if not probes:
            logger.warning("MANIFOLD MAPPER: No probes provided")
            return ManifoldMapResult(
                profiles={},
                total_probes_processed=0,
                total_batches=0,
                all_layers_saturated=True,
                domains_covered=set(),
                positions={},
                velocities={},
            )

        # Initialize layer states (we'll discover num_layers from first batch)
        layer_states: dict[int, _LayerState] = {}
        num_layers: int | None = None

        # Accumulated activations
        all_positions: dict[int, list["Array"]] = {}
        all_velocities: dict[int, list["Array"]] = {}

        # Global tracking
        total_probes = 0
        total_batches = 0
        domains_covered: set[str] = set()

        # Domain-stratified batch generator
        for batch_probes in self._domain_stratified_batches(probes, batch_size):
            # Extract probe texts
            batch_texts = [self._get_probe_text(p) for p in batch_probes]
            batch_texts = [t for t in batch_texts if t]  # Filter empty

            if not batch_texts:
                continue

            # Collect trajectories
            try:
                trajectories = self._activation_provider.collect_trajectory_batch(
                    model, tokenizer, batch_texts
                )
            except Exception as e:
                logger.warning("MANIFOLD MAPPER: Batch collection failed: %s", e)
                continue

            # Track domains
            for probe in batch_probes:
                domains_covered.add(str(probe.domain.value))

            total_probes += len(batch_texts)
            total_batches += 1

            # Initialize layer states on first batch
            if num_layers is None and trajectories.positions:
                num_layers = len(trajectories.positions)
                for layer_idx in trajectories.positions:
                    hidden_dim = trajectories.positions[layer_idx].shape[1]
                    layer_states[layer_idx] = _LayerState(
                        layer_idx=layer_idx,
                        hidden_dim=hidden_dim,
                    )
                    all_positions[layer_idx] = []
                    all_velocities[layer_idx] = []

            # Accumulate activations and check rank saturation
            all_saturated = True

            for layer_idx, state in layer_states.items():
                if state.saturated:
                    continue  # Already saturated, skip

                # Accumulate positions
                if layer_idx in trajectories.positions:
                    all_positions[layer_idx].append(trajectories.positions[layer_idx])

                # Accumulate velocities
                if layer_idx in trajectories.velocities:
                    all_velocities[layer_idx].append(trajectories.velocities[layer_idx])

                # Stack all accumulated samples
                if all_positions[layer_idx]:
                    stacked_positions = b.concatenate(all_positions[layer_idx], axis=0)
                else:
                    continue

                if all_velocities[layer_idx]:
                    stacked_velocities = b.concatenate(all_velocities[layer_idx], axis=0)
                    # Combine positions + velocities for full manifold sampling
                    combined = b.concatenate(
                        [stacked_positions, stacked_velocities], axis=0
                    )
                else:
                    combined = stacked_positions

                b.eval(combined)

                # Compute numerical rank
                new_rank, _ = compute_numerical_rank(combined, b)

                # Track domain coverage
                for probe in batch_probes:
                    state.domains_seen.add(str(probe.domain.value))
                state.probes_processed += len(batch_texts)

                # Check for saturation
                if new_rank == state.activation_rank:
                    state.saturated_count += 1
                    if state.saturated_count >= self.K_CONSECUTIVE:
                        state.saturated = True
                        state.batches_to_saturation = total_batches
                        logger.info(
                            "MANIFOLD MAPPER: Layer %d saturated at rank %d/%d "
                            "(batch %d, %d samples)",
                            layer_idx,
                            new_rank,
                            state.hidden_dim,
                            total_batches,
                            combined.shape[0],
                        )
                else:
                    state.saturated_count = 0
                    state.previous_rank = state.activation_rank
                    state.activation_rank = new_rank

                if not state.saturated:
                    all_saturated = False

            # Log progress periodically
            if total_batches % 5 == 0:
                saturated_layers = sum(1 for s in layer_states.values() if s.saturated)
                logger.info(
                    "MANIFOLD MAPPER: Batch %d, %d probes, %d/%d layers saturated",
                    total_batches,
                    total_probes,
                    saturated_layers,
                    len(layer_states),
                )

            # Check for global termination
            if all_saturated and layer_states:
                logger.info(
                    "MANIFOLD MAPPER: All %d layers saturated after %d batches",
                    len(layer_states),
                    total_batches,
                )
                break

            # Optional max batches limit
            if max_batches is not None and total_batches >= max_batches:
                logger.info(
                    "MANIFOLD MAPPER: Reached max_batches=%d, stopping", max_batches
                )
                break

        # Build final stacked activations
        final_positions: dict[int, "Array"] = {}
        final_velocities: dict[int, "Array"] = {}

        for layer_idx in all_positions:
            if all_positions[layer_idx]:
                final_positions[layer_idx] = b.concatenate(
                    all_positions[layer_idx], axis=0
                )
                b.eval(final_positions[layer_idx])
            if all_velocities[layer_idx]:
                final_velocities[layer_idx] = b.concatenate(
                    all_velocities[layer_idx], axis=0
                )
                b.eval(final_velocities[layer_idx])

        # Compute weight ranks for structural capacity verification
        logger.info("MANIFOLD MAPPER: Computing weight ranks for structural capacity...")
        weight_ranks = self.compute_weight_ranks(model)

        # Build layer profiles
        profiles: dict[int, LayerManifoldProfile] = {}
        for layer_idx, state in layer_states.items():
            position_samples = (
                final_positions[layer_idx].shape[0]
                if layer_idx in final_positions
                else 0
            )
            velocity_samples = (
                final_velocities[layer_idx].shape[0]
                if layer_idx in final_velocities
                else 0
            )

            # Compute Gram condition number
            if layer_idx in final_positions:
                gram_condition = self._compute_gram_condition(
                    final_positions[layer_idx], b
                )
            else:
                gram_condition = float("inf")

            # Get weight ranks for structural capacity
            o_proj_rank, down_proj_rank = weight_ranks.get(layer_idx, (0, 0))
            # Structural capacity is the minimum of both (the bottleneck)
            # If either is 0, we couldn't compute it - use hidden_dim as fallback
            if o_proj_rank > 0 and down_proj_rank > 0:
                structural_capacity = min(o_proj_rank, down_proj_rank)
            elif o_proj_rank > 0:
                structural_capacity = o_proj_rank
            elif down_proj_rank > 0:
                structural_capacity = down_proj_rank
            else:
                structural_capacity = state.hidden_dim  # Assume full capacity if unknown

            profiles[layer_idx] = LayerManifoldProfile(
                layer_idx=layer_idx,
                hidden_dim=state.hidden_dim,
                trajectory_rank=state.activation_rank,  # Final rank = trajectory rank
                activation_rank=state.activation_rank,
                null_rank=state.hidden_dim - state.activation_rank,
                weight_rank_o_proj=o_proj_rank,
                weight_rank_down_proj=down_proj_rank,
                structural_capacity=structural_capacity,
                total_samples=position_samples + velocity_samples,
                position_samples=position_samples,
                velocity_samples=velocity_samples,
                domains_sampled=state.domains_seen,
                probes_processed=state.probes_processed,
                gram_condition=gram_condition,
                batches_to_saturation=state.batches_to_saturation,
                saturated=state.saturated,
            )

            # Log if probe-limited
            if profiles[layer_idx].is_probe_limited:
                logger.info(
                    "MANIFOLD MAPPER: Layer %d is PROBE-LIMITED: "
                    "activation_rank=%d < structural_capacity=%d (unused=%d dims)",
                    layer_idx,
                    state.activation_rank,
                    structural_capacity,
                    profiles[layer_idx].unused_capacity,
                )

        all_saturated = all(p.saturated for p in profiles.values()) if profiles else True

        return ManifoldMapResult(
            profiles=profiles,
            total_probes_processed=total_probes,
            total_batches=total_batches,
            all_layers_saturated=all_saturated,
            domains_covered=domains_covered,
            positions=final_positions,
            velocities=final_velocities,
        )

    def _domain_stratified_batches(
        self, probes: list["AtlasProbe"], batch_size: int
    ) -> list[list["AtlasProbe"]]:
        """Generate batches with probes from all domains represented.

        This ensures early batches cover the full semantic space, maximizing
        rank increase per batch. Without stratification, we might process
        many probes from one domain before seeing others.

        Args:
            probes: All available probes.
            batch_size: Target probes per batch.

        Yields:
            Lists of probes, each batch containing probes from multiple domains.
        """
        from collections import defaultdict

        # Group probes by domain
        by_domain: dict[str, list["AtlasProbe"]] = defaultdict(list)
        for probe in probes:
            by_domain[str(probe.domain.value)].append(probe)

        domains = list(by_domain.keys())
        if not domains:
            return

        # Calculate probes per domain per batch
        probes_per_domain = max(1, batch_size // len(domains))

        # Track position in each domain's probe list
        domain_idx: dict[str, int] = {d: 0 for d in domains}

        while True:
            batch: list["AtlasProbe"] = []

            # Take probes from each domain
            for domain in domains:
                domain_probes = by_domain[domain]
                start = domain_idx[domain]

                # Take up to probes_per_domain from this domain
                end = min(start + probes_per_domain, len(domain_probes))
                batch.extend(domain_probes[start:end])
                domain_idx[domain] = end

            if not batch:
                break  # All domains exhausted

            yield batch

    def _get_probe_text(self, probe: "AtlasProbe") -> str | None:
        """Extract text from a probe for activation collection.

        Args:
            probe: AtlasProbe with support_texts.

        Returns:
            First non-empty support text, or None.
        """
        if hasattr(probe, "support_texts") and probe.support_texts:
            for text in probe.support_texts:
                if text and text.strip():
                    return text.strip()
        return None

    def _compute_gram_condition(
        self, activations: "Array", backend: "Backend"
    ) -> float:
        """Compute condition number of Gram matrix G = A^T @ A.

        Args:
            activations: [n_samples, hidden_dim] activation matrix.
            backend: Backend for computation.

        Returns:
            Condition number (max_eigenvalue / min_eigenvalue).
        """
        b = backend

        # Compute Gram matrix
        G = b.matmul(b.transpose(activations), activations)
        b.eval(G)

        # Eigenvalues of symmetric positive semi-definite matrix
        eigenvalues, _ = b.eigh(G)
        b.eval(eigenvalues)

        # Condition = max / min (for positive eigenvalues)
        eps = machine_epsilon(b, eigenvalues)
        threshold = sqrt_scalar(eps, b) * float(b.to_scalar(b.max(eigenvalues)))

        # Filter to positive eigenvalues above noise floor
        positive_mask = eigenvalues > threshold
        n_positive = int(b.to_scalar(b.sum(b.astype(positive_mask, "int32"))))

        if n_positive == 0:
            return float("inf")

        # Get max and min positive eigenvalues
        # Since eigenvalues are sorted ascending, max is last, min is first positive
        max_eig = float(b.to_scalar(eigenvalues[-1]))

        # Find minimum eigenvalue > threshold using backend ops
        # Replace values <= threshold with inf, then take min
        threshold_scalar = float(b.to_scalar(threshold))
        inf_arr = b.full(b.shape(eigenvalues), float("inf"), dtype=eigenvalues.dtype)
        masked = b.where(eigenvalues > threshold_scalar, eigenvalues, inf_arr)
        b.eval(masked)
        min_eig_arr = b.min(masked)
        b.eval(min_eig_arr)
        min_eig = float(b.to_scalar(min_eig_arr))

        if min_eig <= 0 or min_eig == float("inf"):
            return float("inf")

        return max_eig / min_eig

    def compute_weight_ranks(
        self, model: Any
    ) -> dict[int, tuple[int, int]]:
        """Compute structural capacity from weight matrix ranks.

        For each layer, computes the rank of:
        - o_proj: attention output projection (writes to hidden space)
        - down_proj: MLP down projection (writes to hidden space)

        The minimum of these ranks is the structural ceiling - the maximum
        rank that activations CAN achieve for that layer.

        This answers: "Is the activation rank limited by probe coverage or
        by the model's structural capacity?"

        Args:
            model: The loaded model with accessible weights.

        Returns:
            Dict mapping layer_idx -> (o_proj_rank, down_proj_rank).
        """
        b = self._backend
        weight_ranks: dict[int, tuple[int, int]] = {}

        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            logger.warning(
                "MANIFOLD MAPPER: Cannot access model layers for weight rank computation"
            )
            return weight_ranks

        layers = model.model.layers

        for layer_idx, layer in enumerate(layers):
            o_proj_rank = 0
            down_proj_rank = 0

            try:
                # Compute o_proj rank (attention output projection)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                    o_proj_weight = layer.self_attn.o_proj.weight
                    b.eval(o_proj_weight)
                    o_proj_rank, _ = compute_numerical_rank(o_proj_weight, b)
                    logger.debug(
                        "Layer %d o_proj rank: %d/%d",
                        layer_idx,
                        o_proj_rank,
                        b.shape(o_proj_weight)[0],
                    )
            except Exception as e:
                logger.debug("Could not compute o_proj rank for layer %d: %s", layer_idx, e)

            try:
                # Compute down_proj rank (MLP down projection)
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    down_proj_weight = layer.mlp.down_proj.weight
                    b.eval(down_proj_weight)
                    down_proj_rank, _ = compute_numerical_rank(down_proj_weight, b)
                    logger.debug(
                        "Layer %d down_proj rank: %d/%d",
                        layer_idx,
                        down_proj_rank,
                        b.shape(down_proj_weight)[0],
                    )
            except Exception as e:
                logger.debug("Could not compute down_proj rank for layer %d: %s", layer_idx, e)

            weight_ranks[layer_idx] = (o_proj_rank, down_proj_rank)

        # Log summary
        if weight_ranks:
            full_rank_layers = sum(
                1 for o, d in weight_ranks.values()
                if o > 0 and d > 0
            )
            logger.info(
                "MANIFOLD MAPPER: Computed weight ranks for %d layers (%d with full access)",
                len(weight_ranks),
                full_rank_layers,
            )

        return weight_ranks


__all__ = ["ManifoldMapper", "ManifoldMapResult", "LayerManifoldProfile"]
