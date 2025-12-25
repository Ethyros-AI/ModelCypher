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
Transport-Guided Model Merger using Optimal Transport (OT).

Scientific Foundation:
Uses the Gromov-Wasserstein transport plan to guide weighted parameter averaging.
The transport plan π[i,j] represents the optimal soft correspondence between
source neuron i and target neuron j based on their relational structure.

Math:
W_merged[i,:] = Σ_k π[i,k] * W_source[k,:]

References:
- Peyré & Cuturi (2019) "Computational Optimal Transport"
- Mémoli (2011) "Gromov-Wasserstein distances and the metric approach to object matching"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    Config as GWConfig,
)
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Import canonical IntersectionMap from manifold_stitcher (not placeholder)


class TransportGuidedMerger:
    """Transport-guided model merger using GPU-accelerated Gromov-Wasserstein."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._gw = GromovWassersteinDistance(self._backend)

    @dataclass
    class Config:
        coupling_threshold: float = 0.001
        normalize_rows: bool = True
        blend_alpha: float = 0.5
        use_intersection_confidence: bool = True
        min_samples: int = 5
        gw_config: GWConfig = field(default_factory=GWConfig)

    @dataclass
    class Result:
        merged_weights: list[list[float]]
        gw_distance: float
        marginal_error: float
        effective_rank: int
        converged: bool
        iterations: int
        dimension_confidences: list[float]

    @dataclass
    class BatchResult:
        layer_results: dict[str, "TransportGuidedMerger.Result"]
        mean_gw_distance: float
        mean_marginal_error: float
        failed_layers: list[str]

        @property
        def quality_score(self) -> float:
            total_attempted = max(1, len(self.layer_results) + len(self.failed_layers))
            success_rate = len(self.layer_results) / total_attempted

            converged_count = sum(1 for r in self.layer_results.values() if r.converged)
            convergence_rate = converged_count / max(1, len(self.layer_results))

            distance_score = max(0.0, 1.0 - self.mean_gw_distance)
            return (success_rate + convergence_rate + distance_score) / 3.0

    # MARK: - Core Synthesis

    def synthesize(
        self,
        source_weights: "Array",
        target_weights: "Array",
        transport_plan: "Array",
        config: "TransportGuidedMerger.Config | None" = None,
    ) -> "Array | None":
        """Synthesize merged weights using transport plan."""
        if config is None:
            config = TransportGuidedMerger.Config()

        backend = self._backend
        n = source_weights.shape[0]
        m = target_weights.shape[0]

        if n == 0 or m == 0:
            return None
        if transport_plan.shape[0] != n or transport_plan.shape[1] != m:
            return None

        d_source = source_weights.shape[1]
        d_target = target_weights.shape[1]

        # Apply threshold and normalize if configured
        processed_plan = transport_plan
        if config.coupling_threshold > 0:
            # Zero out values below threshold
            mask = processed_plan >= config.coupling_threshold
            processed_plan = backend.where(mask, processed_plan, backend.zeros_like(processed_plan))

        if config.normalize_rows:
            # Normalize rows to sum to 1
            row_sums = backend.sum(processed_plan, axis=1, keepdims=True)
            row_sums = backend.maximum(row_sums, backend.full(row_sums.shape, 1e-10))
            processed_plan = processed_plan / row_sums

        # Compute transport-merged weights: W_merged = π^T @ W_source
        # π is [n, m], W_source is [n, d_source], result is [m, d_source]
        merged_weights = backend.matmul(backend.transpose(processed_plan), source_weights)

        # Blend with target if dimensions match and alpha > 0
        if d_source == d_target and config.blend_alpha > 0:
            alpha = config.blend_alpha
            merged_weights = (1.0 - alpha) * merged_weights + alpha * target_weights

        return merged_weights

    def synthesize_with_gw(
        self,
        source_activations: "Array",
        target_activations: "Array",
        source_weights: "Array",
        target_weights: "Array",
        config: "TransportGuidedMerger.Config | None" = None,
    ) -> "TransportGuidedMerger.Result | None":
        """Synthesize merged weights using Gromov-Wasserstein transport."""
        if config is None:
            config = TransportGuidedMerger.Config()

        backend = self._backend
        sample_count = source_activations.shape[0]

        if sample_count < config.min_samples:
            return None
        if sample_count != target_activations.shape[0]:
            return None
        if source_weights.shape[0] == 0 or target_weights.shape[0] == 0:
            return None

        source_points = self._align_activations_to_weights(
            source_activations, source_weights.shape[0]
        )
        target_points = self._align_activations_to_weights(
            target_activations, target_weights.shape[0]
        )

        if source_points is None or target_points is None:
            return None

        # Compute pairwise distances
        source_dist = self._gw.compute_pairwise_distances(source_points)
        target_dist = self._gw.compute_pairwise_distances(target_points)

        # Compute GW transport plan
        gw_result = self._gw.compute(
            source_distances=source_dist, target_distances=target_dist, config=config.gw_config
        )

        if not (gw_result.converged or gw_result.iterations > 0):
            return None

        # Metrics
        row_error, col_error = self._compute_marginal_error(gw_result.coupling)
        marginal_error = max(row_error, col_error)

        effective_rank = self._compute_effective_rank(
            gw_result.coupling, config.coupling_threshold
        )
        dim_confidences = self._compute_dimension_confidences(gw_result.coupling)

        # Synthesize
        merged = self.synthesize(
            source_weights=source_weights,
            target_weights=target_weights,
            transport_plan=gw_result.coupling,
            config=config,
        )

        if merged is None:
            return None

        # Convert merged to list for result (backward compat)
        backend.eval(merged)
        merged_list = backend.to_numpy(merged).tolist()

        return TransportGuidedMerger.Result(
            merged_weights=merged_list,
            gw_distance=gw_result.distance,
            marginal_error=marginal_error,
            effective_rank=effective_rank,
            converged=gw_result.converged,
            iterations=gw_result.iterations,
            dimension_confidences=dim_confidences,
        )

    def synthesize_from_crms(
        self,
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_weights: dict[int, "Array"],
        target_weights: dict[int, "Array"],
        config: "TransportGuidedMerger.Config | None" = None,
    ) -> "TransportGuidedMerger.BatchResult":
        """Synthesize merged weights from CRMs for all common layers."""
        if config is None:
            config = TransportGuidedMerger.Config()

        layer_results: dict[str, TransportGuidedMerger.Result] = {}
        failed_layers: list[str] = []
        total_gw_dist = 0.0
        total_marginal = 0.0

        # If commonAnchorIDs logic exists in ConceptResponseMatrix
        common_anchors = (
            source_crm.common_anchor_ids(target_crm)
            if hasattr(source_crm, "common_anchor_ids")
            else []
        )
        if not common_anchors:
            return TransportGuidedMerger.BatchResult({}, 0.0, 0.0, [])

        source_layers = set(source_weights.keys())
        target_layers = set(target_weights.keys())
        common_layers = sorted(list(source_layers.intersection(target_layers)))

        for layer in common_layers:
            layer_key = f"layer_{layer}"

            source_act = source_crm.activation_matrix(layer, common_anchors)
            target_act = target_crm.activation_matrix(layer, common_anchors)
            src_w = source_weights.get(layer)
            tgt_w = target_weights.get(layer)

            if source_act is None or target_act is None or src_w is None or tgt_w is None:
                failed_layers.append(layer_key)
                continue

            result = self.synthesize_with_gw(
                source_activations=source_act,
                target_activations=target_act,
                source_weights=src_w,
                target_weights=tgt_w,
                config=config,
            )

            if result:
                layer_results[layer_key] = result
                total_gw_dist += result.gw_distance
                total_marginal += result.marginal_error
            else:
                failed_layers.append(layer_key)

        count = max(1, len(layer_results))
        return TransportGuidedMerger.BatchResult(
            layer_results=layer_results,
            mean_gw_distance=total_gw_dist / count,
            mean_marginal_error=total_marginal / count,
            failed_layers=failed_layers,
        )

    # MARK: - Utilities

    @staticmethod
    def _align_activations_to_weights(
        activations: list[list[float]], weight_count: int
    ) -> list[list[float]] | None:
        if not activations:
            return None
        n_rows = len(activations)
        n_cols = len(activations[0])

        # If rows match weight count (N neurons), use as is
        if n_rows == weight_count:
            return activations
        # If cols match weight count, transpose (samples x neurons -> neurons x samples)
        # Wait, GW usually expects point clouds.
        # Swift code:
        #   if activations.count == weightCount { return activations }
        #   if firstRow.count == weightCount { return transpose(activations) }
        # So it wants [Neuron x Features].

        if n_cols == weight_count:
            return TransportGuidedMerger._transpose(activations)

        return None

    @staticmethod
    def _apply_threshold(plan: list[list[float]], threshold: float) -> list[list[float]]:
        return [[(val if val >= threshold else 0.0) for val in row] for row in plan]

    @staticmethod
    def _normalize_rows(plan: list[list[float]]) -> list[list[float]]:
        normalized = []
        for row in plan:
            row_sum = sum(row)
            if row_sum > 0:
                normalized.append([val / row_sum for val in row])
            else:
                normalized.append(row)
        return normalized

    @staticmethod
    def _transpose(matrix: list[list[float]]) -> list[list[float]]:
        if not matrix:
            return []
        return [list(col) for col in zip(*matrix)]

    @staticmethod
    def _compute_marginal_error(coupling: list[list[float]]) -> tuple[float, float]:
        n = len(coupling)
        if n == 0:
            return (0.0, 0.0)
        m = len(coupling[0])
        if m == 0:
            return (0.0, 0.0)

        expected_row = 1.0 / n
        expected_col = 1.0 / m

        max_row_error = 0.0
        for i in range(n):
            row_sum = sum(coupling[i])
            max_row_error = max(max_row_error, abs(row_sum - expected_row))

        max_col_error = 0.0
        # Column sums
        col_sums = [0.0] * m
        for i in range(n):
            for j in range(m):
                col_sums[j] += coupling[i][j]

        for j in range(m):
            max_col_error = max(max_col_error, abs(col_sums[j] - expected_col))

        return (max_row_error, max_col_error)

    @staticmethod
    def _compute_effective_rank(coupling: list[list[float]], threshold: float) -> int:
        count = 0
        for row in coupling:
            for val in row:
                if val >= threshold:
                    count += 1
        return count

    @staticmethod
    def _compute_dimension_confidences(coupling: list[list[float]]) -> list[float]:
        n = len(coupling)
        if n == 0:
            return []
        confidences = []

        for row in coupling:
            row_sum = sum(row)
            if row_sum <= 0:
                confidences.append(0.0)
                continue

            # Entropy
            entropy = 0.0
            for val in row:
                p = val / row_sum
                if p > 0:
                    entropy -= p * math.log(p)

            m = len(row)
            max_entropy = math.log(m) if m > 1 else 1.0
            normalized = entropy / max_entropy
            confidences.append(max(0.0, 1.0 - normalized))

        return confidences
