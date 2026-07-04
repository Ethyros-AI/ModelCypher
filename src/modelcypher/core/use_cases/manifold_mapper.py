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

Collects per-token trajectories and velocity features, then estimates rank
saturation and layer profiles using domain-stratified atlas probes.

Usage:
    mapper = ManifoldMapper(backend, activation_provider)
    result = mapper.map_manifold(model, tokenizer, probes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from modelcypher.core.domain.geometry.null_space import compute_numerical_rank
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision,
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.atlas.unified_atlas import AtlasProbe
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_architecture import ModelArchitecturePort

logger = logging.getLogger(__name__)


def _get_model_architecture(model: Any) -> "ModelArchitecturePort":
    """Get architecture wrapper for model, inferring config if needed."""
    from modelcypher.ports.model_architecture_factory import get_model_architecture

    config: dict = {}
    if hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "to_dict"):
            config = model_config.to_dict()
        elif isinstance(model_config, dict):
            config = model_config
    elif hasattr(model, "model") and hasattr(model.model, "config"):
        model_config = model.model.config
        if hasattr(model_config, "to_dict"):
            config = model_config.to_dict()
        elif isinstance(model_config, dict):
            config = model_config

    # If config is empty, try to infer model_type from class name
    if not config.get("model_type"):
        model_class = type(model).__module__ + "." + type(model).__name__
        model_class_lower = model_class.lower()
        if "lfm2" in model_class_lower or "lfm" in model_class_lower:
            config["model_type"] = "lfm2"
            logger.info("Inferred model_type='lfm2' from class %s", model_class)
        elif "llama" in model_class_lower:
            config["model_type"] = "llama"
            logger.info("Inferred model_type='llama' from class %s", model_class)
        elif "qwen" in model_class_lower:
            config["model_type"] = "qwen2"
            logger.info("Inferred model_type='qwen2' from class %s", model_class)
        elif "gpt2" in model_class_lower:
            config["model_type"] = "gpt2"
            logger.info("Inferred model_type='gpt2' from class %s", model_class)
        elif "bert" in model_class_lower:
            config["model_type"] = "bert"
            logger.info("Inferred model_type='bert' from class %s", model_class)

    return get_model_architecture(model, config=config)


@dataclass
class ManifoldProgressEvent:
    """Progress event for AI-interpretable status updates.

    Contains semantic context that AI assistants can use to explain
    what's happening to humans.
    """

    model_name: str  # "source" or "target"
    batch: int
    probes_processed: int
    layers_saturated: int
    layers_total: int
    ranks: dict[int, int]  # layer_idx -> current_rank
    hidden_dims: dict[int, int]  # layer_idx -> hidden_dim
    layer_just_saturated: int | None = None  # If a layer just saturated


class ProgressCallback(Protocol):
    """Protocol for manifold mapping progress callbacks."""

    def __call__(self, event: ManifoldProgressEvent) -> None:
        """Called with progress updates during manifold mapping."""
        ...


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
    """Result of manifold mapping operation.

    For PERFECT merging, this stores ALL activation types collected during
    manifold mapping. Each activation type has:
    - Trajectories (all token positions) for rank saturation detection
    - Mean-pooled values (one per probe) for alignment transforms
    """

    # Per-layer profiles
    profiles: dict[int, LayerManifoldProfile]

    # Global metrics
    total_probes_processed: int
    total_batches: int
    all_layers_saturated: bool

    # Domain coverage (global)
    domains_covered: set[str]

    # === HIDDEN STATE ACTIVATIONS ===
    # Trajectories: positions[layer_idx] = [total_samples, hidden_dim]
    positions: dict[int, "Array"]
    velocities: dict[int, "Array"]

    # === PROBE METADATA for merge consistency ===
    probe_ids: list[str] = field(default_factory=list)
    probe_domains: list[str] = field(default_factory=list)

    # Mean-pooled per-probe activations (one vector per probe, for alignment)
    # mean_pooled[layer_idx] = [n_probes, hidden_dim]
    mean_pooled: dict[int, "Array"] = field(default_factory=dict)

    # === INTERMEDIATE (MLP) ACTIVATIONS ===
    # Trajectories: [total_samples, intermediate_dim]
    intermediate_positions: dict[int, "Array"] = field(default_factory=dict)
    # Mean-pooled: [n_probes, intermediate_dim]
    intermediate_mean_pooled: dict[int, "Array"] = field(default_factory=dict)

    # === EMBEDDING ACTIVATIONS ===
    # Trajectories: [total_samples, hidden_dim]
    embedding_positions: "Array | None" = None
    # Mean-pooled: [n_probes, hidden_dim]
    embedding_mean_pooled: list["Array"] = field(default_factory=list)

    # === ATTENTION Q/K/V ACTIVATIONS ===
    # Q trajectories: [total_samples, q_dim]
    q_positions: dict[int, "Array"] = field(default_factory=dict)
    # K trajectories: [total_samples, kv_dim]
    k_positions: dict[int, "Array"] = field(default_factory=dict)
    # V trajectories: [total_samples, kv_dim]
    v_positions: dict[int, "Array"] = field(default_factory=dict)
    # Mean-pooled: [n_probes, dim]
    q_mean_pooled: dict[int, "Array"] = field(default_factory=dict)
    k_mean_pooled: dict[int, "Array"] = field(default_factory=dict)
    v_mean_pooled: dict[int, "Array"] = field(default_factory=dict)

    # === GATE ACTIVATIONS ===
    # Trajectories: [total_samples, intermediate_dim]
    gate_positions: dict[int, "Array"] = field(default_factory=dict)
    # Mean-pooled: [n_probes, intermediate_dim]
    gate_mean_pooled: dict[int, "Array"] = field(default_factory=dict)


@dataclass
class _LayerState:
    """Internal state for tracking a layer during mapping."""

    layer_idx: int
    hidden_dim: int
    rank_ceiling: int = 0
    activation_rank: int = 0
    saturated: bool = False
    batches_to_saturation: int = 0
    domains_seen: set[str] = field(default_factory=set)
    probes_processed: int = 0
    position_samples: int = 0
    velocity_samples: int = 0
    gram: "Array | None" = None


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

    The mapping stops when all layers reach full rank.
    """

    # Default batch size (probes per batch)
    DEFAULT_BATCH_SIZE = 20

    def __init__(
        self,
        backend: "Backend",
        activation_provider: "ActivationProvider | None" = None,
    ):
        """Initialize ManifoldMapper.

        Args:
            backend: Backend for tensor operations.
            activation_provider: Provider for collecting trajectories.
                                If None, creates a default activation provider.
        """
        self._backend = backend

        if activation_provider is None:
            from modelcypher.ports.activation_provider import get_activation_provider
            activation_provider = get_activation_provider()
        self._activation_provider = activation_provider

    def map_manifold(
        self,
        model: Any,
        tokenizer: Any,
        probes: list["AtlasProbe"],
        batch_size: int | None = None,
        max_batches: int | None = None,
        model_name: str = "model",
        progress_callback: ProgressCallback | None = None,
        retain_trajectories: bool = True,
    ) -> ManifoldMapResult:
        """Map the model's activation manifold with rank saturation detection.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding probe texts.
            probes: List of atlas probes (with domain labels).
            batch_size: Probes per batch. Default 20.
            max_batches: Optional maximum batches (for testing). None = no limit.
            model_name: Name for progress reporting ("source" or "target").
            progress_callback: Optional callback for progress events.
            retain_trajectories: Whether to keep full trajectories in memory.

        Returns:
            ManifoldMapResult with per-layer profiles and stored activations.
        """
        b = self._backend
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        retain_trajectories = bool(retain_trajectories)
        weight_ranks = self.compute_weight_ranks(model)

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

        # Accumulated trajectories (optional; large memory footprint)
        all_positions: dict[int, list["Array"]] = {}
        all_velocities: dict[int, list["Array"]] = {}
        all_intermediate_positions: dict[int, list["Array"]] = {}
        all_embedding_positions: list["Array"] = []
        all_q_positions: dict[int, list["Array"]] = {}
        all_k_positions: dict[int, list["Array"]] = {}
        all_v_positions: dict[int, list["Array"]] = {}
        all_gate_positions: dict[int, list["Array"]] = {}

        # Global tracking
        total_probes = 0
        total_batches = 0
        domains_covered: set[str] = set()

        # === PROBE METADATA for merge consistency ===
        # Track probe_ids and domains in order, plus mean-pooled activations
        all_probe_ids: list[str] = []
        all_probe_domains: list[str] = []
        # Mean-pooled per-probe activations for ALL types
        all_mean_pooled: dict[int, list["Array"]] = {}  # hidden states
        all_intermediate_mean_pooled: dict[int, list["Array"]] = {}
        all_embedding_mean_pooled: list["Array"] = []
        all_q_mean_pooled: dict[int, list["Array"]] = {}
        all_k_mean_pooled: dict[int, list["Array"]] = {}
        all_v_mean_pooled: dict[int, list["Array"]] = {}
        all_gate_mean_pooled: dict[int, list["Array"]] = {}

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
                raise RuntimeError(
                    f"Manifold mapper batch collection failed: {e}. "
                    f"Cannot build manifold from partial data."
                ) from e

            # Track domains and probe metadata
            for probe in batch_probes:
                domains_covered.add(str(probe.domain.value))
                all_probe_ids.append(probe.probe_id)
                all_probe_domains.append(str(probe.domain.value))

            total_probes += len(batch_texts)
            total_batches += 1

            # Initialize layer states on first batch
            if num_layers is None and trajectories.positions:
                num_layers = len(trajectories.positions)
                for layer_idx in trajectories.positions:
                    # Ensure layer_idx and hidden_dim are Python ints, not backend arrays
                    layer_idx = int(layer_idx)
                    hidden_dim = int(trajectories.positions[layer_idx].shape[1])
                    o_proj_rank, down_proj_rank = weight_ranks.get(layer_idx, (0, 0))
                    if o_proj_rank > 0 and down_proj_rank > 0:
                        rank_ceiling = min(o_proj_rank, down_proj_rank)
                    elif o_proj_rank > 0:
                        rank_ceiling = o_proj_rank
                    elif down_proj_rank > 0:
                        rank_ceiling = down_proj_rank
                    else:
                        rank_ceiling = hidden_dim
                    layer_states[layer_idx] = _LayerState(
                        layer_idx=layer_idx,
                        hidden_dim=hidden_dim,
                        rank_ceiling=rank_ceiling,
                    )
                    # Mean-pooled storage (always kept)
                    all_mean_pooled[layer_idx] = []
                    all_intermediate_mean_pooled[layer_idx] = []
                    all_q_mean_pooled[layer_idx] = []
                    all_k_mean_pooled[layer_idx] = []
                    all_v_mean_pooled[layer_idx] = []
                    all_gate_mean_pooled[layer_idx] = []
                    if retain_trajectories:
                        # Trajectory storage (optional, large memory footprint)
                        all_positions[layer_idx] = []
                        all_velocities[layer_idx] = []
                        all_intermediate_positions[layer_idx] = []
                        all_q_positions[layer_idx] = []
                        all_k_positions[layer_idx] = []
                        all_v_positions[layer_idx] = []
                        all_gate_positions[layer_idx] = []

            # Compute mean-pooled per-probe activations for ALL activation types
            # This ensures profile-based merges produce identical results to probe-based
            text_lengths = trajectories.text_lengths

            # === HIDDEN STATE MEAN-POOLING ===
            for layer_idx_raw in trajectories.positions:
                layer_idx = int(layer_idx_raw)  # Ensure Python int
                positions = trajectories.positions[layer_idx_raw]
                # Split by text and mean-pool each
                offset = 0
                for length in text_lengths:
                    if length > 0:
                        text_positions = positions[offset : offset + length]
                        mean_pooled = b.mean(text_positions, axis=0)
                        b.eval(mean_pooled)
                        all_mean_pooled[layer_idx].append(mean_pooled)
                    offset += length

            # === INTERMEDIATE MEAN-POOLING ===
            for layer_idx_raw in trajectories.intermediate_positions:
                layer_idx = int(layer_idx_raw)
                int_positions = trajectories.intermediate_positions[layer_idx_raw]
                offset = 0
                for length in text_lengths:
                    if length > 0:
                        text_int = int_positions[offset : offset + length]
                        mean_int = b.mean(text_int, axis=0)
                        b.eval(mean_int)
                        all_intermediate_mean_pooled[layer_idx].append(mean_int)
                    offset += length

            # === EMBEDDING MEAN-POOLING ===
            if trajectories.embedding_positions is not None and b.shape(trajectories.embedding_positions)[0] > 0:
                emb_positions = trajectories.embedding_positions
                offset = 0
                for length in text_lengths:
                    if length > 0:
                        text_emb = emb_positions[offset : offset + length]
                        mean_emb = b.mean(text_emb, axis=0)
                        b.eval(mean_emb)
                        all_embedding_mean_pooled.append(mean_emb)
                    offset += length
                # Store embedding trajectory (optional)
                if retain_trajectories:
                    all_embedding_positions.append(emb_positions)

            # === ATTENTION Q/K/V MEAN-POOLING ===
            for layer_idx_raw in trajectories.q_positions:
                layer_idx = int(layer_idx_raw)
                q_pos = trajectories.q_positions[layer_idx_raw]
                k_pos = trajectories.k_positions.get(layer_idx_raw)
                v_pos = trajectories.v_positions.get(layer_idx_raw)
                offset = 0
                for length in text_lengths:
                    if length > 0:
                        text_q = q_pos[offset : offset + length]
                        mean_q = b.mean(text_q, axis=0)
                        b.eval(mean_q)
                        all_q_mean_pooled[layer_idx].append(mean_q)

                        if k_pos is not None:
                            text_k = k_pos[offset : offset + length]
                            mean_k = b.mean(text_k, axis=0)
                            b.eval(mean_k)
                            all_k_mean_pooled[layer_idx].append(mean_k)

                        if v_pos is not None:
                            text_v = v_pos[offset : offset + length]
                            mean_v = b.mean(text_v, axis=0)
                            b.eval(mean_v)
                            all_v_mean_pooled[layer_idx].append(mean_v)
                    offset += length

            # === GATE MEAN-POOLING ===
            for layer_idx_raw in trajectories.gate_positions:
                layer_idx = int(layer_idx_raw)
                gate_pos = trajectories.gate_positions[layer_idx_raw]
                offset = 0
                for length in text_lengths:
                    if length > 0:
                        text_gate = gate_pos[offset : offset + length]
                        mean_gate = b.mean(text_gate, axis=0)
                        b.eval(mean_gate)
                        all_gate_mean_pooled[layer_idx].append(mean_gate)
                    offset += length

            # Accumulate activations and check rank saturation
            all_saturated = True

            for layer_idx, state in layer_states.items():
                if state.saturated:
                    continue  # Already saturated, skip

                batch_positions = trajectories.positions.get(layer_idx)
                batch_velocities = trajectories.velocities.get(layer_idx)
                if batch_positions is None:
                    continue

                if not retain_trajectories:
                    batch_positions = _promote_precision(batch_positions, b)
                    b.eval(batch_positions)
                    state.position_samples += int(b.shape(batch_positions)[0])
                    if batch_velocities is not None:
                        batch_velocities = _promote_precision(batch_velocities, b)
                        b.eval(batch_velocities)
                        state.velocity_samples += int(b.shape(batch_velocities)[0])
                        combined = b.concatenate([batch_positions, batch_velocities], axis=0)
                    else:
                        combined = batch_positions

                    b.eval(combined)
                    batch_gram = b.matmul(b.transpose(combined), combined)
                    b.eval(batch_gram)
                    if state.gram is None:
                        state.gram = batch_gram
                    else:
                        state.gram = state.gram + batch_gram
                    b.eval(state.gram)

                    new_rank = self._compute_rank_from_gram(state.gram, b)
                else:
                    # === ACCUMULATE HIDDEN STATE TRAJECTORIES ===
                    all_positions[layer_idx].append(batch_positions)
                    if batch_velocities is not None:
                        all_velocities[layer_idx].append(batch_velocities)

                    # === ACCUMULATE INTERMEDIATE TRAJECTORIES ===
                    if layer_idx in trajectories.intermediate_positions:
                        all_intermediate_positions[layer_idx].append(
                            trajectories.intermediate_positions[layer_idx]
                        )

                    # === ACCUMULATE Q/K/V TRAJECTORIES ===
                    if layer_idx in trajectories.q_positions:
                        all_q_positions[layer_idx].append(trajectories.q_positions[layer_idx])
                    if layer_idx in trajectories.k_positions:
                        all_k_positions[layer_idx].append(trajectories.k_positions[layer_idx])
                    if layer_idx in trajectories.v_positions:
                        all_v_positions[layer_idx].append(trajectories.v_positions[layer_idx])

                    # === ACCUMULATE GATE TRAJECTORIES ===
                    if layer_idx in trajectories.gate_positions:
                        all_gate_positions[layer_idx].append(trajectories.gate_positions[layer_idx])

                    # Stack all accumulated samples
                    stacked_positions = b.concatenate(all_positions[layer_idx], axis=0)
                    if all_velocities[layer_idx]:
                        stacked_velocities = b.concatenate(all_velocities[layer_idx], axis=0)
                        # Combine positions + velocities for full manifold sampling
                        combined = b.concatenate(
                            [stacked_positions, stacked_velocities], axis=0
                        )
                    else:
                        combined = stacked_positions

                    b.eval(combined)

                    # Compute numerical rank (enforce monotonicity; exact rank cannot decrease)
                    new_rank, _ = compute_numerical_rank(combined, b)

                if new_rank < state.activation_rank:
                    logger.warning(
                        "MANIFOLD MAPPER: Numerical rank decreased for layer %d (%d -> %d); "
                        "clamping to preserve monotonic rank",
                        layer_idx,
                        state.activation_rank,
                        new_rank,
                    )
                    new_rank = state.activation_rank

                # Track domain coverage
                for probe in batch_probes:
                    state.domains_seen.add(str(probe.domain.value))
                state.probes_processed += len(batch_texts)

                if new_rank > state.activation_rank:
                    state.activation_rank = new_rank

                if state.activation_rank >= state.rank_ceiling and not state.saturated:
                    state.saturated = True
                    state.batches_to_saturation = total_batches
                    if progress_callback is not None:
                        progress_callback(
                            ManifoldProgressEvent(
                                model_name=model_name,
                                batch=total_batches,
                                probes_processed=total_probes,
                                layers_saturated=sum(
                                    1 for s in layer_states.values() if s.saturated
                                ),
                                layers_total=len(layer_states),
                                ranks={
                                    idx: s.activation_rank
                                    for idx, s in layer_states.items()
                                },
                                hidden_dims={
                                    idx: s.hidden_dim for idx, s in layer_states.items()
                                },
                                layer_just_saturated=layer_idx,
                            )
                        )

                if not state.saturated:
                    all_saturated = False

            # Log progress periodically (only when a progress callback is requested)
            if total_batches % 5 == 0 and progress_callback is not None:
                saturated_layers = sum(1 for s in layer_states.values() if s.saturated)
                logger.info(
                    "MANIFOLD MAPPER: Batch %d, %d probes, %d/%d layers saturated",
                    total_batches,
                    total_probes,
                    saturated_layers,
                    len(layer_states),
                )
                # Emit progress event
                progress_callback(
                    ManifoldProgressEvent(
                        model_name=model_name,
                        batch=total_batches,
                        probes_processed=total_probes,
                        layers_saturated=saturated_layers,
                        layers_total=len(layer_states),
                        ranks={
                            idx: s.activation_rank
                            for idx, s in layer_states.items()
                        },
                        hidden_dims={
                            idx: s.hidden_dim for idx, s in layer_states.items()
                        },
                        layer_just_saturated=None,
                    )
                )

            # Check for global termination
            if all_saturated and layer_states:
                logger.info(
                    "MANIFOLD MAPPER: All %d layers saturated after %d batches",
                    len(layer_states),
                    total_batches,
                )
                # Final progress event
                if progress_callback is not None:
                    progress_callback(
                        ManifoldProgressEvent(
                            model_name=model_name,
                            batch=total_batches,
                            probes_processed=total_probes,
                            layers_saturated=len(layer_states),
                            layers_total=len(layer_states),
                            ranks={
                                idx: s.activation_rank
                                for idx, s in layer_states.items()
                            },
                            hidden_dims={
                                idx: s.hidden_dim for idx, s in layer_states.items()
                            },
                            layer_just_saturated=None,
                        )
                    )
                break

            # Optional max batches limit
            if max_batches is not None and total_batches >= max_batches:
                logger.info(
                    "MANIFOLD MAPPER: Reached max_batches=%d, stopping", max_batches
                )
                break

        # Build final stacked activations for ALL types
        final_positions: dict[int, "Array"] = {}
        final_velocities: dict[int, "Array"] = {}
        final_intermediate_positions: dict[int, "Array"] = {}
        final_q_positions: dict[int, "Array"] = {}
        final_k_positions: dict[int, "Array"] = {}
        final_v_positions: dict[int, "Array"] = {}
        final_gate_positions: dict[int, "Array"] = {}

        if retain_trajectories:
            for layer_idx in all_positions:
                # Hidden states
                if all_positions[layer_idx]:
                    final_positions[layer_idx] = b.concatenate(
                        all_positions[layer_idx], axis=0
                    )
                    b.eval(final_positions[layer_idx])
                if all_velocities.get(layer_idx):
                    final_velocities[layer_idx] = b.concatenate(
                        all_velocities[layer_idx], axis=0
                    )
                    b.eval(final_velocities[layer_idx])

                # Intermediate
                if all_intermediate_positions.get(layer_idx):
                    final_intermediate_positions[layer_idx] = b.concatenate(
                        all_intermediate_positions[layer_idx], axis=0
                    )
                    b.eval(final_intermediate_positions[layer_idx])

                # Q/K/V
                if all_q_positions.get(layer_idx):
                    final_q_positions[layer_idx] = b.concatenate(
                        all_q_positions[layer_idx], axis=0
                    )
                    b.eval(final_q_positions[layer_idx])
                if all_k_positions.get(layer_idx):
                    final_k_positions[layer_idx] = b.concatenate(
                        all_k_positions[layer_idx], axis=0
                    )
                    b.eval(final_k_positions[layer_idx])
                if all_v_positions.get(layer_idx):
                    final_v_positions[layer_idx] = b.concatenate(
                        all_v_positions[layer_idx], axis=0
                    )
                    b.eval(final_v_positions[layer_idx])

                # Gate
                if all_gate_positions.get(layer_idx):
                    final_gate_positions[layer_idx] = b.concatenate(
                        all_gate_positions[layer_idx], axis=0
                    )
                    b.eval(final_gate_positions[layer_idx])

        # Embedding positions (single tensor, not per-layer)
        final_embedding_positions: "Array | None" = None
        if retain_trajectories and all_embedding_positions:
            final_embedding_positions = b.concatenate(all_embedding_positions, axis=0)
            b.eval(final_embedding_positions)

        # Stack mean-pooled activations for ALL types
        final_mean_pooled: dict[int, "Array"] = {}
        final_intermediate_mean_pooled: dict[int, "Array"] = {}
        final_q_mean_pooled: dict[int, "Array"] = {}
        final_k_mean_pooled: dict[int, "Array"] = {}
        final_v_mean_pooled: dict[int, "Array"] = {}
        final_gate_mean_pooled: dict[int, "Array"] = {}

        for layer_idx, mp_list in all_mean_pooled.items():
            if mp_list:
                final_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_mean_pooled[layer_idx])

        for layer_idx, mp_list in all_intermediate_mean_pooled.items():
            if mp_list:
                final_intermediate_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_intermediate_mean_pooled[layer_idx])

        for layer_idx, mp_list in all_q_mean_pooled.items():
            if mp_list:
                final_q_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_q_mean_pooled[layer_idx])

        for layer_idx, mp_list in all_k_mean_pooled.items():
            if mp_list:
                final_k_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_k_mean_pooled[layer_idx])

        for layer_idx, mp_list in all_v_mean_pooled.items():
            if mp_list:
                final_v_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_v_mean_pooled[layer_idx])

        for layer_idx, mp_list in all_gate_mean_pooled.items():
            if mp_list:
                final_gate_mean_pooled[layer_idx] = b.stack(mp_list, axis=0)
                b.eval(final_gate_mean_pooled[layer_idx])

        # Compute weight ranks for structural capacity verification
        logger.info("MANIFOLD MAPPER: Computing weight ranks for structural capacity...")
        weight_ranks = self.compute_weight_ranks(model)

        # Build layer profiles
        profiles: dict[int, LayerManifoldProfile] = {}
        for layer_idx, state in layer_states.items():
            if retain_trajectories:
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
            else:
                position_samples = state.position_samples
                velocity_samples = state.velocity_samples

            mean_pooled = final_mean_pooled.get(layer_idx)

            # Compute Gram condition number
            if retain_trajectories and layer_idx in final_positions:
                gram_condition = self._compute_gram_condition(
                    final_positions[layer_idx], b
                )
            elif state.gram is not None:
                gram_condition = self._compute_gram_condition_from_gram(state.gram, b)
            elif mean_pooled is not None:
                gram_condition = self._compute_gram_condition(mean_pooled, b)
            else:
                gram_condition = float("inf")

            # Get weight ranks for structural capacity
            o_proj_rank, down_proj_rank = weight_ranks.get(layer_idx, (0, 0))
            structural_capacity = state.rank_ceiling or state.hidden_dim

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

        # Embedding mean-pooled (list, not dict)
        final_embedding_mean_pooled: list["Array"] = all_embedding_mean_pooled

        logger.info(
            "MANIFOLD MAPPER: Collected ALL activation types - %d probes, %d domains, "
            "hidden=%d layers, intermediate=%d layers, embedding=%d samples, "
            "Q/K/V=%d layers, gate=%d layers",
            len(all_probe_ids),
            len(set(all_probe_domains)),
            len(final_mean_pooled),
            len(final_intermediate_mean_pooled),
            len(final_embedding_mean_pooled),
            len(final_q_mean_pooled),
            len(final_gate_mean_pooled),
        )

        return ManifoldMapResult(
            profiles=profiles,
            total_probes_processed=total_probes,
            total_batches=total_batches,
            all_layers_saturated=all_saturated,
            domains_covered=domains_covered,
            # Hidden states
            positions=final_positions,
            velocities=final_velocities,
            probe_ids=all_probe_ids,
            probe_domains=all_probe_domains,
            mean_pooled=final_mean_pooled,
            # Intermediate
            intermediate_positions=final_intermediate_positions,
            intermediate_mean_pooled=final_intermediate_mean_pooled,
            # Embedding
            embedding_positions=final_embedding_positions,
            embedding_mean_pooled=final_embedding_mean_pooled,
            # Q/K/V
            q_positions=final_q_positions,
            k_positions=final_k_positions,
            v_positions=final_v_positions,
            q_mean_pooled=final_q_mean_pooled,
            k_mean_pooled=final_k_mean_pooled,
            v_mean_pooled=final_v_mean_pooled,
            # Gate
            gate_positions=final_gate_positions,
            gate_mean_pooled=final_gate_mean_pooled,
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
        from collections import defaultdict, deque

        # Group probes by domain
        by_domain: dict[str, list["AtlasProbe"]] = defaultdict(list)
        for probe in probes:
            by_domain[str(probe.domain.value)].append(probe)

        domains = list(by_domain.keys())
        if not domains:
            return

        domain_queue = deque(domains)
        domain_idx: dict[str, int] = {d: 0 for d in domains}

        while domain_queue:
            batch: list["AtlasProbe"] = []

            while domain_queue and len(batch) < batch_size:
                domain = domain_queue.popleft()
                domain_probes = by_domain[domain]
                idx = domain_idx[domain]

                if idx < len(domain_probes):
                    batch.append(domain_probes[idx])
                    domain_idx[domain] = idx + 1
                    if domain_idx[domain] < len(domain_probes):
                        domain_queue.append(domain)

            if batch:
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

    def _compute_rank_from_gram(
        self, gram: "Array", backend: "Backend"
    ) -> int:
        """Compute numerical rank from a Gram matrix using a dtype-derived threshold."""
        b = backend
        gram = _promote_precision(gram, b)
        b.eval(gram)

        eigvals = b.eigvalsh(gram)
        b.eval(eigvals)

        max_eig_arr = eigvals[-1]
        b.eval(max_eig_arr)
        max_eig = float(b.to_scalar(max_eig_arr))
        if max_eig <= 0:
            return 0

        eps = float(machine_epsilon(b, eigvals))
        threshold = max_eig * eps
        rank_mask = eigvals > threshold
        rank_arr = b.sum(b.astype(rank_mask, "int32"))
        b.eval(rank_arr)
        return int(b.to_scalar(rank_arr))

    def _compute_gram_condition_from_gram(
        self, gram: "Array", backend: "Backend"
    ) -> float:
        """Compute condition number directly from a Gram matrix."""
        b = backend
        gram = _promote_precision(gram, b)
        b.eval(gram)

        eigenvalues = b.eigvalsh(gram)
        b.eval(eigenvalues)

        eps = machine_epsilon(b, eigenvalues)
        threshold = sqrt_scalar(eps, b) * float(b.to_scalar(b.max(eigenvalues)))

        positive_mask = eigenvalues > threshold
        n_positive = int(b.to_scalar(b.sum(b.astype(positive_mask, "int32"))))
        if n_positive == 0:
            return float("inf")

        max_eig = float(b.to_scalar(eigenvalues[-1]))
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
        - o_proj: attention/conv output projection (writes to hidden space)
        - down_proj: MLP down projection (writes to hidden space)

        The minimum of these ranks is the structural ceiling - the maximum
        rank that activations CAN achieve for that layer.

        Supports multiple architectures:
        - Llama/Qwen: self_attn.o_proj, mlp.down_proj
        - LFM: conv.out_proj, feed_forward.w2

        Args:
            model: The loaded model with accessible weights.

        Returns:
            Dict mapping layer_idx -> (o_proj_rank, down_proj_rank).
        """
        b = self._backend
        weight_ranks: dict[int, tuple[int, int]] = {}

        # Get model architecture via protocol
        arch = _get_model_architecture(model)
        if arch.num_layers == 0:
            logger.warning(
                "MANIFOLD MAPPER: Cannot access model layers for weight rank computation"
            )
            return weight_ranks

        for layer_idx in range(arch.num_layers):
            o_proj_rank = 0
            down_proj_rank = 0

            # Get layer accessor for normalized access
            layer_accessor = arch.layer_accessor(layer_idx)
            attn_module = layer_accessor.attention
            mlp_module = layer_accessor.mlp

            # Get output projection weight from attention module
            o_proj_weight = None
            try:
                if attn_module is not None:
                    # Try common output projection names
                    for proj_name in ("o_proj", "out_proj"):
                        proj = getattr(attn_module, proj_name, None)
                        if proj is not None:
                            o_proj_weight = getattr(proj, "weight", None)
                            break

                if o_proj_weight is not None:
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

            # Get down projection weight from MLP module
            down_proj_weight = None
            try:
                if mlp_module is not None:
                    # Try common down projection names
                    for proj_name in ("down_proj", "w2"):
                        proj = getattr(mlp_module, proj_name, None)
                        if proj is not None:
                            down_proj_weight = getattr(proj, "weight", None)
                            break

                if down_proj_weight is not None:
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
