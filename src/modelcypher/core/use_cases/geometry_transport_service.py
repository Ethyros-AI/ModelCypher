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

from dataclasses import dataclass

from modelcypher.core.domain.geometry.gromov_wasserstein import Config as GWConfig
from modelcypher.core.domain.geometry.transport_guided_merger import (
    TransportGuidedMerger,
)


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for transport-guided merge.

    All values are derived from geometry. No user configuration.
    """

    normalize_rows: bool = True
    min_samples: int = 5


class GeometryTransportService:
    """
    Service for transport-guided model merging operations.
    """

    def synthesize_weights(
        self,
        source_weights: list[list[float]],
        target_weights: list[list[float]],
        transport_plan: list[list[float]],
        normalize_rows: bool = True,
    ) -> list[list[float]] | None:
        """
        Synthesize merged weights using a transport plan.

        Uses the transport plan π[i,j] to guide weighted averaging:
        W_merged[j,:] = Σ_i π[i,j] * W_source[i,:]

        All parameters derived from geometry - no configuration.

        Args:
            source_weights: Source model weight matrix [N x D]
            target_weights: Target model weight matrix [M x D]
            transport_plan: Transport coupling matrix [N x M]
            normalize_rows: Whether to normalize transport plan rows

        Returns:
            Merged weight matrix [M x D] or None if invalid
        """
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

        # Coupling threshold from transport plan distribution
        flat_plan = [v for row in transport_plan for v in row]
        mean_coupling = sum(flat_plan) / len(flat_plan) if flat_plan else 0.0
        coupling_threshold = mean_coupling

        # Blend alpha from weight cosine similarity
        src_flat = [v for row in source_weights for v in row]
        tgt_flat = [v for row in target_weights for v in row]
        src_arr = backend.array(src_flat)
        tgt_arr = backend.array(tgt_flat)
        dot = backend.sum(src_arr * tgt_arr)
        norm_src = backend.sqrt(backend.sum(src_arr * src_arr))
        norm_tgt = backend.sqrt(backend.sum(tgt_arr * tgt_arr))
        backend.eval(dot, norm_src, norm_tgt)
        denom = float(backend.to_numpy(norm_src)) * float(backend.to_numpy(norm_tgt))
        similarity = float(backend.to_numpy(dot)) / denom if denom > 1e-9 else 0.0
        blend_alpha = similarity  # Direct: high similarity = blend more, low = transport-guided

        config = TransportGuidedMerger.Config(
            coupling_threshold=coupling_threshold,
            normalize_rows=normalize_rows,
            blend_alpha=blend_alpha,
        )
        return TransportGuidedMerger.synthesize(
            source_weights=source_weights,
            target_weights=target_weights,
            transport_plan=transport_plan,
            config=config,
        )

    def synthesize_with_gw(
        self,
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        source_weights: list[list[float]],
        target_weights: list[list[float]],
    ) -> TransportGuidedMerger.Result | None:
        """
        Compute GW transport plan and synthesize merged weights.

        Computes pairwise distances from activations, solves for optimal
        transport plan using Gromov-Wasserstein, then applies transport-
        guided weight averaging.

        Everything derived from geometry. No configuration.

        Args:
            source_activations: Activation samples from source model
            target_activations: Activation samples from target model
            source_weights: Source model weight matrix
            target_weights: Target model weight matrix

        Returns:
            MergeResult with merged weights and quality metrics
        """
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.cka import CKAComputer

        backend = get_default_backend()

        # GW epsilon from activation variance
        src_arr = backend.array(source_activations)
        mean_src = backend.mean(src_arr)
        var_src = backend.mean((src_arr - mean_src) ** 2)
        backend.eval(var_src)
        gw_epsilon = float(backend.to_numpy(backend.sqrt(var_src))) * 0.1

        # Max iterations from problem size
        n_src = len(source_activations)
        n_tgt = len(target_activations)
        gw_max_iterations = max(50, min(500, (n_src + n_tgt) * 2))

        # Coupling threshold from uniform distribution expectation
        coupling_threshold = 0.5 / max(1, n_src * n_tgt)

        # Blend alpha from CKA between activations
        tgt_arr = backend.array(target_activations)
        cka = CKAComputer(backend)
        similarity = cka.linear_cka(src_arr, tgt_arr)
        backend.eval(similarity)
        blend_alpha = float(backend.to_numpy(similarity))

        gw_config = GWConfig(
            epsilon=gw_epsilon,
            max_iterations=gw_max_iterations,
        )
        tgm_config = TransportGuidedMerger.Config(
            coupling_threshold=coupling_threshold,
            normalize_rows=True,
            blend_alpha=blend_alpha,
            min_samples=5,
            gw_config=gw_config,
        )
        return TransportGuidedMerger.synthesize_with_gw(
            source_activations=source_activations,
            target_activations=target_activations,
            source_weights=source_weights,
            target_weights=target_weights,
            config=tgm_config,
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
