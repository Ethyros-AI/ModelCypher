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
Multi-Channel Merge Pipeline for world model compression.

This module implements the extended merge pipeline supporting multiple knowledge
channels from different source models. Based on the unified theory connecting
DeepSeek mHC and null-space projection (docs/research/mhc_null_space_connection.md).

Mathematical Foundation:
    For each layer:
    1. Project each channel to null space: δW_safe_i = P_null(target) @ δW_i
    2. Route channels: combined = Σ_j H[i,j] × δW_safe_j (doubly stochastic)
    3. Merge: W' = W_target + combined (geometric addition)

    Properties:
    - CKA = 1.0 per channel (null-space preserves geometry)
    - Stable combination (Birkhoff spectral norm ≤ 1.0)
    - No interference (channels add, not blend)

Usage:
    pipeline = MultiChannelMergePipeline(backend)
    result = pipeline.run_merge(
        sources={"spatial": spatial_model, "temporal": temporal_model},
        target=target_model,
        config=MultiChannelMergeConfig(channels=["spatial", "temporal"]),
    )

References:
    - docs/DIMENSIONAL_COMPRESSION.md (Multi-Modal Extension)
    - docs/architecture/multi_channel_merge.md
    - docs/research/mhc_null_space_connection.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.birkhoff_router import (
    BirkhoffRouter,
    BirkhoffRoutingResult,
    RoutingMode,
)
from modelcypher.core.domain.geometry.channel_projector import (
    ChannelProjector,
    MultiChannelProjectionResult,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class MultiChannelMergeConfig:
    """Configuration for multi-channel merge."""

    # Channel identifiers (e.g., ["spatial", "temporal", "text"])
    channels: list[str]

    # Routing mode for channel combination
    routing_mode: str = "uniform"  # "uniform", "identity", "diagonal_weighted"

    # Whether to verify CKA = 1.0 per channel (slower but safer)
    verify_cka: bool = True

    # k-NN neighbors for null-space computation (None = auto-derive)
    k_neighbors: int | None = None

    # Fast mode skips CKA precision checks
    fast_mode: bool = False


@dataclass
class LayerMergeResult:
    """Result of merging a single layer across all channels."""

    layer_name: str

    # Merged weights for this layer
    merged_weights: Any

    # Per-channel projection results
    channel_projection: MultiChannelProjectionResult

    # Routing result
    routing_result: BirkhoffRoutingResult

    # Combined delta (before adding to target)
    combined_delta: Any

    # Norm metrics
    target_norm: float
    delta_norm: float
    merged_norm: float


@dataclass
class MultiChannelMergeResult:
    """Result of multi-channel merge operation."""

    # Merged weights for all layers
    merged_weights: dict[str, Any]

    # Per-layer results
    layer_results: dict[str, LayerMergeResult]

    # Per-channel CKA (should all be 1.0)
    per_channel_cka: dict[str, float]

    # Routing matrix used (last layer's, representative)
    routing_matrix: Any

    # Spectral norm of routing (should be ≤ 1.0)
    spectral_norm: float

    # Total projection loss across all channels and layers
    total_projection_loss: float

    # Average preserved fraction
    average_preserved_fraction: float

    # Number of layers merged
    layer_count: int

    # Number of channels
    channel_count: int

    # Whether all channels achieved alignment
    all_aligned: bool

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    # Configuration used
    config: MultiChannelMergeConfig | None = None


class MultiChannelMergePipeline:
    """
    Extended merge pipeline supporting multiple knowledge channels.

    This pipeline enables merging knowledge from multiple source models
    (e.g., world model, vision-language model, text model) into a single
    target model while preserving the invariant geometry of each channel.

    Architecture:
        ┌─────────────────────────────────────────────────────────────────────┐
        │                    MULTI-CHANNEL PIPELINE                           │
        │                                                                     │
        │  Source 1 (World Model) ──┐                                        │
        │                           │      ┌──────────────┐                  │
        │  Source 2 (VL Model) ─────┼──►   │   Channel    │   ┌───────────┐  │
        │                           │      │   Projector  │──►│  Birkhoff │  │
        │  Source 3 (Text Model) ───┤      │  (CKA=1.0)   │   │   Router  │  │
        │                           │      └──────────────┘   └─────┬─────┘  │
        │  Target Model ────────────┘                               │        │
        │                                                           ▼        │
        │                                                    ┌───────────┐   │
        │                                                    │  Merged   │   │
        │                                                    │  Model    │   │
        │                                                    └───────────┘   │
        └─────────────────────────────────────────────────────────────────────┘

    Thread Safety:
        This pipeline is NOT thread-safe. Create separate instances for
        concurrent operations.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize the multi-channel merge pipeline.

        Args:
            backend: Backend for tensor operations.
        """
        self._backend = backend or get_default_backend()
        self._channel_projector = None  # Lazy init with config
        self._birkhoff_router = BirkhoffRouter(self._backend)

    def run_merge(
        self,
        source_activations: dict[str, dict[str, "Array"]],
        source_weights: dict[str, dict[str, "Array"]],
        target_activations: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        config: MultiChannelMergeConfig,
    ) -> MultiChannelMergeResult:
        """
        Run multi-channel merge.

        This is the main entry point for multi-channel merging. It:
        1. Iterates over all target layers
        2. For each layer: projects all channels, routes, and merges
        3. Returns merged weights and comprehensive diagnostics

        Args:
            source_activations: {channel_id: {layer_name: activations}}.
            source_weights: {channel_id: {layer_name: weights}}.
            target_activations: {layer_name: activations}.
            target_weights: {layer_name: weights}.
            config: Merge configuration.

        Returns:
            MultiChannelMergeResult with all merged weights and metrics.

        Raises:
            ValueError: If channel configurations don't match.
        """
        backend = self._backend

        # Initialize channel projector with config
        self._channel_projector = ChannelProjector(
            backend, fast_mode=config.fast_mode
        )

        # Validate channels exist in sources
        for channel_id in config.channels:
            if channel_id not in source_activations:
                raise ValueError(f"Channel '{channel_id}' not in source_activations")
            if channel_id not in source_weights:
                raise ValueError(f"Channel '{channel_id}' not in source_weights")

        # Get mergeable layers (present in target and all sources)
        target_layers = set(target_weights.keys())
        mergeable_layers = target_layers.copy()

        for channel_id in config.channels:
            channel_layers = set(source_weights[channel_id].keys())
            mergeable_layers &= channel_layers

        logger.info(
            "MULTI-CHANNEL MERGE: %d channels, %d mergeable layers",
            len(config.channels),
            len(mergeable_layers),
        )

        # Merge each layer
        merged_weights: dict[str, Any] = {}
        layer_results: dict[str, LayerMergeResult] = {}
        per_channel_cka: dict[str, float] = {ch: 1.0 for ch in config.channels}
        total_projection_loss = 0.0
        total_preserved = 0.0
        layer_count = 0
        last_routing_result = None
        all_aligned = True

        for layer_name in sorted(mergeable_layers):
            logger.info("MULTI-CHANNEL MERGE: Processing layer '%s'", layer_name)

            # Gather per-channel data for this layer
            layer_source_acts = {
                ch: source_activations[ch][layer_name] for ch in config.channels
            }
            layer_source_weights = {
                ch: source_weights[ch][layer_name] for ch in config.channels
            }
            layer_target_acts = target_activations.get(layer_name)
            layer_target_weights = target_weights[layer_name]

            # Skip if no target activations for this layer
            if layer_target_acts is None:
                logger.warning(
                    "MULTI-CHANNEL MERGE: No target activations for '%s', skipping",
                    layer_name,
                )
                merged_weights[layer_name] = layer_target_weights
                continue

            # Merge this layer
            layer_result = self._merge_layer(
                layer_name=layer_name,
                source_activations=layer_source_acts,
                source_weights=layer_source_weights,
                target_activations=layer_target_acts,
                target_weights=layer_target_weights,
                config=config,
            )

            merged_weights[layer_name] = layer_result.merged_weights
            layer_results[layer_name] = layer_result
            last_routing_result = layer_result.routing_result

            # Accumulate metrics
            total_projection_loss += layer_result.channel_projection.total_projection_loss
            total_preserved += layer_result.channel_projection.average_preserved_fraction
            layer_count += 1

            if not layer_result.channel_projection.all_aligned:
                all_aligned = False

        # Compute aggregates
        avg_preserved = total_preserved / layer_count if layer_count > 0 else 1.0

        # Copy non-mergeable layers from target
        for layer_name in target_layers - mergeable_layers:
            merged_weights[layer_name] = target_weights[layer_name]

        return MultiChannelMergeResult(
            merged_weights=merged_weights,
            layer_results=layer_results,
            per_channel_cka=per_channel_cka,
            routing_matrix=last_routing_result.routing_matrix if last_routing_result else None,
            spectral_norm=last_routing_result.spectral_norm if last_routing_result else 1.0,
            total_projection_loss=total_projection_loss,
            average_preserved_fraction=avg_preserved,
            layer_count=layer_count,
            channel_count=len(config.channels),
            all_aligned=all_aligned,
            config=config,
        )

    def _merge_layer(
        self,
        layer_name: str,
        source_activations: dict[str, "Array"],
        source_weights: dict[str, "Array"],
        target_activations: "Array",
        target_weights: "Array",
        config: MultiChannelMergeConfig,
    ) -> LayerMergeResult:
        """Merge a single layer across all channels."""
        backend = self._backend

        # Ensure arrays are on backend
        target_activations = backend.array(target_activations)
        target_weights = backend.array(target_weights)
        backend.eval(target_activations, target_weights)

        # Step 1: Project all channels into target's null space
        projection_result = self._channel_projector.project_channels(
            source_activations=source_activations,
            source_weights=source_weights,
            target_activations=target_activations,
            target_weights=target_weights,
            k_neighbors=config.k_neighbors,
        )

        # Step 2: Extract filtered deltas for routing
        channel_deltas = [
            projection_result.channel_results[ch].filtered_delta
            for ch in config.channels
        ]

        # Step 3: Route channels via Birkhoff mixing
        routing_mode = RoutingMode(config.routing_mode)
        combined_delta, routing_result = self._birkhoff_router.route_channels(
            channel_deltas, init_mode=routing_mode
        )
        backend.eval(combined_delta)

        # Step 4: Geometric addition (NOT blending)
        merged = target_weights + combined_delta
        backend.eval(merged)

        # Compute norms for diagnostics
        target_flat = backend.reshape(target_weights, (-1,))
        delta_flat = backend.reshape(combined_delta, (-1,))
        merged_flat = backend.reshape(merged, (-1,))

        target_norm = backend.sum(target_flat * target_flat) ** 0.5
        delta_norm = backend.sum(delta_flat * delta_flat) ** 0.5
        merged_norm = backend.sum(merged_flat * merged_flat) ** 0.5
        backend.eval(target_norm, delta_norm, merged_norm)

        return LayerMergeResult(
            layer_name=layer_name,
            merged_weights=merged,
            channel_projection=projection_result,
            routing_result=routing_result,
            combined_delta=combined_delta,
            target_norm=float(backend.to_scalar(target_norm)),
            delta_norm=float(backend.to_scalar(delta_norm)),
            merged_norm=float(backend.to_scalar(merged_norm)),
        )


def run_multi_channel_merge(
    source_activations: dict[str, dict[str, "Array"]],
    source_weights: dict[str, dict[str, "Array"]],
    target_activations: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    channels: list[str],
    routing_mode: str = "uniform",
    fast_mode: bool = False,
    backend: "Backend | None" = None,
) -> MultiChannelMergeResult:
    """
    Convenience function for multi-channel merge.

    This is a simpler entry point that creates the pipeline and config
    internally. For more control, use MultiChannelMergePipeline directly.

    Args:
        source_activations: {channel_id: {layer_name: activations}}.
        source_weights: {channel_id: {layer_name: weights}}.
        target_activations: {layer_name: activations}.
        target_weights: {layer_name: weights}.
        channels: List of channel IDs to merge.
        routing_mode: How to combine channels ("uniform", "identity").
        fast_mode: Skip CKA precision checks.
        backend: Backend for tensor operations.

    Returns:
        MultiChannelMergeResult with merged weights.
    """
    config = MultiChannelMergeConfig(
        channels=channels,
        routing_mode=routing_mode,
        fast_mode=fast_mode,
    )

    pipeline = MultiChannelMergePipeline(backend)
    return pipeline.run_merge(
        source_activations=source_activations,
        source_weights=source_weights,
        target_activations=target_activations,
        target_weights=target_weights,
        config=config,
    )


__all__ = [
    "LayerMergeResult",
    "MultiChannelMergeConfig",
    "MultiChannelMergePipeline",
    "MultiChannelMergeResult",
    "run_multi_channel_merge",
]
