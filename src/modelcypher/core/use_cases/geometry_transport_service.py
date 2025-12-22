"""
Geometry Transport-Guided Merger Service.

Exposes optimal transport-guided model merging as CLI/MCP-consumable operations.
Uses Gromov-Wasserstein distance to compute neuron correspondences for weight merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modelcypher.core.domain.geometry.transport_guided_merger import (
    TransportGuidedMerger,
)
from modelcypher.core.domain.geometry.gromov_wasserstein import Config as GWConfig


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for transport-guided merge."""
    coupling_threshold: float = 0.001
    normalize_rows: bool = True
    blend_alpha: float = 0.5
    min_samples: int = 5
    gw_epsilon: float = 0.05
    gw_max_iterations: int = 50


class GeometryTransportService:
    """
    Service for transport-guided model merging operations.
    """

    def synthesize_weights(
        self,
        source_weights: list[list[float]],
        target_weights: list[list[float]],
        transport_plan: list[list[float]],
        coupling_threshold: float = 0.001,
        normalize_rows: bool = True,
        blend_alpha: float = 0.5,
    ) -> Optional[list[list[float]]]:
        """
        Synthesize merged weights using a transport plan.

        Uses the transport plan π[i,j] to guide weighted averaging:
        W_merged[j,:] = Σ_i π[i,j] * W_source[i,:]

        Args:
            source_weights: Source model weight matrix [N x D]
            target_weights: Target model weight matrix [M x D]
            transport_plan: Transport coupling matrix [N x M]
            coupling_threshold: Minimum coupling to consider
            normalize_rows: Whether to normalize transport plan rows
            blend_alpha: Blend factor with target (0 = transport-only)

        Returns:
            Merged weight matrix [M x D] or None if invalid
        """
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
        config: Optional[MergeConfig] = None,
    ) -> Optional[TransportGuidedMerger.Result]:
        """
        Compute GW transport plan and synthesize merged weights.

        Computes pairwise distances from activations, solves for optimal
        transport plan using Gromov-Wasserstein, then applies transport-
        guided weight averaging.

        Args:
            source_activations: Activation samples from source model
            target_activations: Activation samples from target model
            source_weights: Source model weight matrix
            target_weights: Target model weight matrix
            config: Optional merge configuration

        Returns:
            MergeResult with merged weights and quality metrics
        """
        if config is None:
            config = MergeConfig()

        gw_config = GWConfig(
            epsilon=config.gw_epsilon,
            max_iterations=config.gw_max_iterations,
        )
        tgm_config = TransportGuidedMerger.Config(
            coupling_threshold=config.coupling_threshold,
            normalize_rows=config.normalize_rows,
            blend_alpha=config.blend_alpha,
            min_samples=config.min_samples,
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
            "mergedWeightShape": [len(result.merged_weights), len(result.merged_weights[0]) if result.merged_weights else 0],
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
