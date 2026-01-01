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
Geometry Transport-Guided Merger Service.

Exposes optimal transport-guided model merging as CLI/MCP-consumable operations.
Uses Gromov-Wasserstein distance to compute neuron correspondences for weight merging.
"""

from __future__ import annotations

from modelcypher.core.domain.geometry.transport_guided_merger import (
    TransportGuidedMerger,
)


class GeometryTransportService:
    """Service for transport-guided model merging operations.

    All thresholds and parameters are derived from the geometry of the inputs.
    No configuration required - the transport plan IS the correspondence.
    """

    def __init__(self) -> None:
        self._merger = TransportGuidedMerger()

    def synthesize_weights(
        self,
        source_weights: list[list[float]],
        target_weights: list[list[float]],
        transport_plan: list[list[float]],
    ) -> list[list[float]] | None:
        """Synthesize merged weights using a transport plan.

        Uses the transport plan π[i,j] to guide weighted averaging:
        W_merged[j,:] = Σ_i π[i,j] * W_source[i,:]

        All thresholds derived from numerical precision - no configuration.

        Args:
            source_weights: Source model weight matrix [N x D]
            target_weights: Target model weight matrix [M x D]
            transport_plan: Transport coupling matrix [N x M]

        Returns:
            Merged weight matrix [M x D] or None if invalid
        """
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        src_arr = backend.array(source_weights)
        tgt_arr = backend.array(target_weights)
        plan_arr = backend.array(transport_plan)

        result = self._merger.synthesize(
            source_weights=src_arr,
            target_weights=tgt_arr,
            transport_plan=plan_arr,
        )

        if result is None:
            return None

        backend.eval(result)
        return backend.to_numpy(result).tolist()

    def synthesize_with_gw(
        self,
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        source_weights: list[list[float]],
        target_weights: list[list[float]],
    ) -> TransportGuidedMerger.Result | None:
        """Compute GW transport plan and synthesize merged weights.

        Computes pairwise distances from activations, solves for optimal
        transport plan using Gromov-Wasserstein, then applies transport-
        guided weight averaging.

        All parameters derived from geometry. No configuration.

        Args:
            source_activations: Activation samples from source model
            target_activations: Activation samples from target model
            source_weights: Source model weight matrix
            target_weights: Target model weight matrix

        Returns:
            MergeResult with merged weights and quality metrics
        """
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        src_act = backend.array(source_activations)
        tgt_act = backend.array(target_activations)
        src_w = backend.array(source_weights)
        tgt_w = backend.array(target_weights)

        return self._merger.synthesize_with_gw(
            source_activations=src_act,
            target_activations=tgt_act,
            source_weights=src_w,
            target_weights=tgt_w,
        )

    @staticmethod
    def merge_result_payload(result: TransportGuidedMerger.Result) -> dict:
        """Convert merge result to CLI/MCP payload."""
        return {
            "gwDistance": result.gw_distance,
            "marginalError": result.marginal_error,
            "effectiveRank": result.effective_rank,
            "converged": result.converged,
            "iterations": result.iterations,
            "dimensionConfidences": result.dimension_confidences,
            "mergedWeightShape": [
                len(result.merged_weights),
                len(result.merged_weights[0]) if result.merged_weights else 0,
            ],
        }

    @staticmethod
    def batch_result_payload(result: TransportGuidedMerger.BatchResult) -> dict:
        """Convert batch merge result to CLI/MCP payload."""
        return {
            "meanGWDistance": result.mean_gw_distance,
            "meanMarginalError": result.mean_marginal_error,
            "qualityScore": result.quality_score,
            "successfulLayers": len(result.layer_results),
            "failedLayers": result.failed_layers,
            "layerResults": {
                layer: GeometryTransportService.merge_result_payload(res)
                for layer, res in result.layer_results.items()
            },
        }
