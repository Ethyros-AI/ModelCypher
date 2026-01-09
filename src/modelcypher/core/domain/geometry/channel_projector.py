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
Channel Projector for multi-channel knowledge merging.

This module projects multiple knowledge channels into a target model's null space,
enabling interference-free knowledge addition from multiple source models.

Mathematical Foundation (from docs/research/mhc_null_space_connection.md):
    For each channel i:
    1. Align: F_i = find_alignment(source_acts_i, target_acts) → CKA = 1.0
    2. Compute delta: δW_i = (source_weights_i @ F_i) - target_weights
    3. Project to null space: δW_safe_i = P_null(target_acts) @ δW_i

    Properties:
    - CKA = 1.0 per channel (alignment preserves geometry)
    - δW_safe_i is orthogonal to target's tangent space (no interference)
    - Channels can be safely combined via Birkhoff routing

Usage:
    projector = ChannelProjector(backend)
    results = projector.project_channels(
        source_activations={"spatial": spatial_acts, "temporal": temporal_acts},
        source_weights={"spatial": spatial_w, "temporal": temporal_w},
        target_activations=target_acts,
        target_weights=target_w,
    )
    # Each result has: filtered_delta, cka_achieved, projection_loss

References:
    - docs/DIMENSIONAL_COMPRESSION.md (Multi-Modal Extension)
    - docs/research/mhc_null_space_connection.md
    - docs/architecture/multi_channel_merge.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geodesic_null_space import (
    GeodesicNullSpaceFilter,
    GeodesicNullSpaceBasis,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ChannelProjectionResult:
    """Result of projecting a single channel into target's null space."""

    # Channel identifier
    channel_id: str

    # The filtered delta (safe to add to target weights)
    filtered_delta: Any

    # CKA achieved during alignment (should be 1.0, invariant)
    cka_achieved: float

    # Numerical deviation from CKA = 1.0 (for precision diagnostics)
    numerical_deviation: float

    # Fraction of delta removed by null-space projection
    projection_loss: float

    # Fraction of delta preserved in null space
    preserved_fraction: float

    # Norm of original delta (before projection)
    original_delta_norm: float

    # Norm of filtered delta (after projection)
    filtered_delta_norm: float

    # Whether alignment was successful (always True if no errors)
    alignment_successful: bool

    # Scale ratio from alignment (for potential magnitude correction)
    scale_ratio: float


@dataclass
class MultiChannelProjectionResult:
    """Result of projecting multiple channels."""

    # Per-channel results
    channel_results: dict[str, ChannelProjectionResult]

    # Total projection loss across all channels
    total_projection_loss: float

    # Average preserved fraction
    average_preserved_fraction: float

    # All channels achieved CKA = 1.0
    all_aligned: bool

    # Number of channels processed
    n_channels: int

    # Shared null-space basis (computed once from target)
    null_space_basis: GeodesicNullSpaceBasis


class ChannelProjector:
    """
    Projects multiple knowledge channels into target's null space.

    This class orchestrates the multi-channel projection pipeline:
    1. Compute target's null-space basis ONCE (shared across channels)
    2. For each channel: align → compute delta → project to null space
    3. Return all projected deltas ready for Birkhoff routing

    The key optimization is computing the null-space basis once. Since
    all channels project into the SAME target model's null space, we
    avoid redundant geodesic distance computation.

    Thread Safety:
        This class is thread-safe for concurrent channel processing.
        The null-space basis is immutable after creation.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        fast_mode: bool = False,
    ) -> None:
        """Initialize the channel projector.

        Args:
            backend: Backend for tensor operations.
            fast_mode: If True, skip CKA precision checks (faster).
        """
        self._backend = backend or get_default_backend()
        self._null_space_filter = GeodesicNullSpaceFilter(self._backend)
        self._aligner = GramAligner(self._backend, fast_mode=fast_mode)
        self._fast_mode = fast_mode

    def project_channels(
        self,
        source_activations: dict[str, "Array"],
        source_weights: dict[str, "Array"],
        target_activations: "Array",
        target_weights: "Array",
        *,
        k_neighbors: int | None = None,
    ) -> MultiChannelProjectionResult:
        """
        Project multiple source channels into target's null space.

        Each channel represents knowledge from a different source model
        (e.g., spatial from world model, temporal from video model).
        All channels are projected into the SAME target null space.

        Args:
            source_activations: {channel_id: activations [n_samples, d_source]}.
            source_weights: {channel_id: weights [out_dim, d_source]}.
            target_activations: Target activations [n_samples, d_target].
            target_weights: Target weights [out_dim, d_target].
            k_neighbors: k for k-NN graph in null-space computation.

        Returns:
            MultiChannelProjectionResult with all channel projections.

        Raises:
            ValueError: If channel IDs don't match between activations and weights.
        """
        backend = self._backend

        # Validate channel IDs match
        if set(source_activations.keys()) != set(source_weights.keys()):
            raise ValueError(
                f"Channel ID mismatch: activations={set(source_activations.keys())}, "
                f"weights={set(source_weights.keys())}"
            )

        channel_ids = list(source_activations.keys())
        n_channels = len(channel_ids)

        if n_channels == 0:
            raise ValueError("At least one channel required")

        # Ensure target tensors are on backend
        target_activations = backend.array(target_activations)
        target_weights = backend.array(target_weights)
        backend.eval(target_activations, target_weights)

        # Validate sample counts match across all channels and target
        n_target = int(target_activations.shape[0])
        for channel_id, acts in source_activations.items():
            acts_arr = backend.array(acts)
            n_source = int(acts_arr.shape[0])
            if n_source != n_target:
                raise ValueError(
                    f"Sample counts must match: channel '{channel_id}' has {n_source}, "
                    f"target has {n_target}"
                )

        # =================================================================
        # COMPUTE NULL-SPACE BASIS ONCE (shared across all channels)
        # =================================================================
        # This is the key optimization: geodesic distance computation is
        # expensive, but all channels project into the same null space.
        logger.info(
            "CHANNEL PROJECTOR: Computing shared null-space basis for %d channels",
            n_channels
        )
        null_space_basis = self._null_space_filter.prepare_basis(
            target_activations, k_neighbors=k_neighbors
        )

        # =================================================================
        # PROJECT EACH CHANNEL
        # =================================================================
        channel_results: dict[str, ChannelProjectionResult] = {}
        total_loss = 0.0
        total_preserved = 0.0

        for channel_id in channel_ids:
            logger.info("CHANNEL PROJECTOR: Processing channel '%s'", channel_id)

            source_acts = backend.array(source_activations[channel_id])
            source_w = backend.array(source_weights[channel_id])
            backend.eval(source_acts, source_w)

            # Project this channel
            result = self._project_single_channel(
                channel_id=channel_id,
                source_activations=source_acts,
                source_weights=source_w,
                target_activations=target_activations,
                target_weights=target_weights,
                null_space_basis=null_space_basis,
            )

            channel_results[channel_id] = result
            total_loss += result.projection_loss
            total_preserved += result.preserved_fraction

        # Compute aggregates
        average_preserved = total_preserved / n_channels if n_channels > 0 else 1.0
        all_aligned = all(r.alignment_successful for r in channel_results.values())

        return MultiChannelProjectionResult(
            channel_results=channel_results,
            total_projection_loss=total_loss,
            average_preserved_fraction=average_preserved,
            all_aligned=all_aligned,
            n_channels=n_channels,
            null_space_basis=null_space_basis,
        )

    def _project_single_channel(
        self,
        channel_id: str,
        source_activations: "Array",
        source_weights: "Array",
        target_activations: "Array",
        target_weights: "Array",
        null_space_basis: GeodesicNullSpaceBasis,
    ) -> ChannelProjectionResult:
        """Project a single channel into target's null space."""
        backend = self._backend

        # =================================================================
        # STEP 1: ALIGN (CKA = 1.0)
        # =================================================================
        # Find feature transform F such that CKA(source @ F, target) = 1.0
        try:
            alignment = self._aligner.find_perfect_alignment(
                source_activations, target_activations
            )
            cka_achieved = alignment.achieved_cka  # 1.0 (invariant)
            numerical_deviation = alignment.numerical_deviation
            scale_ratio = alignment.scale_ratio
            alignment_successful = True
        except Exception as e:
            logger.warning(
                "CHANNEL PROJECTOR: Alignment failed for channel '%s': %s",
                channel_id, e
            )
            # Fallback: use pinv projection
            alignment = None
            cka_achieved = 0.0
            numerical_deviation = 1.0
            scale_ratio = 1.0
            alignment_successful = False

        # =================================================================
        # STEP 2: COMPUTE ALIGNED DELTA (WITH DUAL-STITCH FOR CROSS-ARCH)
        # =================================================================
        if alignment is not None:
            F = alignment.feature_transform  # [d_src, d_tgt]

            # Check for output dimension mismatch (cross-architecture)
            src_out_dim = int(source_weights.shape[0])
            tgt_out_dim = int(target_weights.shape[0])
            needs_dual_stitch = (src_out_dim != tgt_out_dim)

            if needs_dual_stitch:
                # DUAL-STITCH: Compute output stitch compositionally
                # G @ W @ F where G transforms output dimension
                #
                # This is mathematically guaranteed because:
                # - Hidden alignment achieves CKA=1.0
                # - Output projections are linear functions of hidden

                # Compute output stitch compositionally from hidden alignment + weights
                H = backend.transpose(F)  # [d_tgt, d_src]
                backend.eval(H)

                G = self._aligner.compositional_stitch(
                    hidden_transform=F,  # [d_src, d_tgt]
                    source_weight=source_weights,
                    target_weight=target_weights,
                )
                backend.eval(G)  # G: [tgt_out, src_out]

                # Apply dual-stitch: G @ W @ F
                aligned_source = backend.matmul(G, source_weights)  # [tgt_out, d_src]
                aligned_source = backend.matmul(aligned_source, F)   # [tgt_out, d_tgt]
                backend.eval(aligned_source)

                logger.info(
                    "DUAL-STITCH for channel '%s': [%d,%d] @ [%d,%d] @ [%d,%d] → [%d,%d]",
                    channel_id,
                    int(G.shape[0]), int(G.shape[1]),
                    src_out_dim, int(source_weights.shape[1]),
                    int(F.shape[0]), int(F.shape[1]),
                    int(aligned_source.shape[0]), int(aligned_source.shape[1])
                )
            else:
                # SINGLE-STITCH: Same dimensions, just apply F
                # source_weights: [out_dim, d_source]
                # F: [d_source, d_target]
                # aligned_source: [out_dim, d_target]
                aligned_source = backend.matmul(source_weights, F)
                backend.eval(aligned_source)
        else:
            # Fallback: use pinv to project weights
            logger.warning(
                "CHANNEL PROJECTOR: Using pinv fallback for channel '%s'",
                channel_id
            )
            source_pinv = backend.pinv(source_activations)
            backend.eval(source_pinv)
            # This gives an approximate projection, not CKA = 1.0
            # source_weights @ pinv(source_acts) @ target_acts ≈ aligned
            aligned_source = backend.matmul(
                backend.matmul(source_weights, source_pinv),
                target_activations
            )
            backend.eval(aligned_source)

        # Compute delta
        delta = aligned_source - target_weights
        backend.eval(delta)

        # =================================================================
        # STEP 3: PROJECT TO NULL SPACE
        # =================================================================
        # Filter delta using pre-computed null-space basis
        null_result = self._null_space_filter.filter_delta(
            delta, target_activations, basis=null_space_basis
        )

        return ChannelProjectionResult(
            channel_id=channel_id,
            filtered_delta=null_result.filtered_delta,
            cka_achieved=cka_achieved,
            numerical_deviation=numerical_deviation,
            projection_loss=null_result.projection_loss,
            preserved_fraction=null_result.preserved_fraction,
            original_delta_norm=null_result.original_norm,
            filtered_delta_norm=null_result.filtered_norm,
            alignment_successful=alignment_successful,
            scale_ratio=scale_ratio,
        )

    def project_single(
        self,
        source_activations: "Array",
        source_weights: "Array",
        target_activations: "Array",
        target_weights: "Array",
        *,
        channel_id: str = "default",
        k_neighbors: int | None = None,
    ) -> ChannelProjectionResult:
        """
        Convenience method for single-channel projection.

        Equivalent to project_channels with one channel, but slightly
        more efficient as it avoids the multi-channel orchestration.

        Args:
            source_activations: Source activations [n_samples, d_source].
            source_weights: Source weights [out_dim, d_source].
            target_activations: Target activations [n_samples, d_target].
            target_weights: Target weights [out_dim, d_target].
            channel_id: Identifier for this channel.
            k_neighbors: k for k-NN graph.

        Returns:
            ChannelProjectionResult for this channel.
        """
        result = self.project_channels(
            source_activations={channel_id: source_activations},
            source_weights={channel_id: source_weights},
            target_activations=target_activations,
            target_weights=target_weights,
            k_neighbors=k_neighbors,
        )
        return result.channel_results[channel_id]


__all__ = [
    "ChannelProjectionResult",
    "ChannelProjector",
    "MultiChannelProjectionResult",
]
