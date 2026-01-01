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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.gromov_wasserstein import GromovWassersteinDistance
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class TransportGuidedMerger:
    """Transport-guided model merger using GPU-accelerated Gromov-Wasserstein.

    All thresholds are derived from the geometry of the transport plan and
    numerical precision of the dtype. No arbitrary parameters.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._gw = GromovWassersteinDistance(self._backend)

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

    # MARK: - Core Synthesis

    def synthesize(
        self,
        source_weights: "Array",
        target_weights: "Array",
        transport_plan: "Array",
    ) -> "Array | None":
        """Synthesize merged weights using transport plan.

        The transport plan determines the correspondence. Coupling threshold
        is derived from numerical precision (machine epsilon). No arbitrary
        blending - the transport plan IS the blending.
        """
        backend = self._backend
        n = source_weights.shape[0]
        m = target_weights.shape[0]

        if n == 0 or m == 0:
            return None
        if transport_plan.shape[0] != n or transport_plan.shape[1] != m:
            return None

        # Derive coupling threshold from numerical precision
        # Values below sqrt(eps) are numerically meaningless
        eps = machine_epsilon(backend, transport_plan)
        coupling_threshold = math.sqrt(float(backend.to_numpy(eps)))

        # Zero out values below numerical precision threshold
        processed_plan = transport_plan
        mask = processed_plan >= coupling_threshold
        processed_plan = backend.where(mask, processed_plan, backend.zeros_like(processed_plan))

        # Normalize rows to sum to 1 (always - this is mathematical correctness, not config)
        row_sums = backend.sum(processed_plan, axis=1, keepdims=True)
        row_sums = backend.maximum(row_sums, backend.full(row_sums.shape, coupling_threshold))
        processed_plan = processed_plan / row_sums

        # Compute transport-merged weights: W_merged = π^T @ W_source
        # π is [n, m], W_source is [n, d_source], result is [m, d_source]
        merged_weights = backend.matmul(backend.transpose(processed_plan), source_weights)

        return merged_weights

    def synthesize_with_gw(
        self,
        source_activations: "Array",
        target_activations: "Array",
        source_weights: "Array",
        target_weights: "Array",
    ) -> "TransportGuidedMerger.Result | None":
        """Synthesize merged weights using Gromov-Wasserstein transport.

        All parameters are derived from the geometry. Minimum sample count
        is 2 (minimum for meaningful distance computation).
        """
        backend = self._backend
        sample_count = source_activations.shape[0]

        # Minimum 2 samples required for pairwise distances
        if sample_count < 2:
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

        # Compute GW transport plan - all params derived from dtype
        gw_result = self._gw.compute(
            source_distances=source_dist, target_distances=target_dist
        )

        if not (gw_result.converged or gw_result.iterations > 0):
            return None

        # Metrics
        row_error, col_error = self._compute_marginal_error(gw_result.coupling)
        marginal_error = max(row_error, col_error)

        effective_rank = self._compute_effective_rank(gw_result.coupling)
        dim_confidences = self._compute_dimension_confidences(gw_result.coupling)

        # Synthesize
        merged = self.synthesize(
            source_weights=source_weights,
            target_weights=target_weights,
            transport_plan=gw_result.coupling,
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
    ) -> "TransportGuidedMerger.BatchResult":
        """Synthesize merged weights from CRMs for all common layers."""
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

    def _align_activations_to_weights(
        self, activations: "Array", weight_count: int
    ) -> "Array | None":
        """Align activations to weight dimensions."""
        if activations.shape[0] == 0:
            return None

        n_rows = activations.shape[0]
        n_cols = activations.shape[1] if len(activations.shape) > 1 else 1

        # If rows match weight count (N neurons), use as is
        if n_rows == weight_count:
            return activations
        # If cols match weight count, transpose (samples x neurons -> neurons x samples)
        if n_cols == weight_count:
            return self._backend.transpose(activations)

        return None

    def _compute_marginal_error(self, coupling: "Array") -> tuple[float, float]:
        """Compute marginal error for coupling matrix."""
        backend = self._backend
        n = coupling.shape[0]
        m = coupling.shape[1]

        if n == 0 or m == 0:
            return (0.0, 0.0)

        expected_row = 1.0 / n
        expected_col = 1.0 / m

        row_sums = backend.sum(coupling, axis=1)
        col_sums = backend.sum(coupling, axis=0)

        row_errors = backend.abs(row_sums - expected_row)
        col_errors = backend.abs(col_sums - expected_col)

        max_row_error = backend.max(row_errors)
        max_col_error = backend.max(col_errors)

        backend.eval(max_row_error, max_col_error)
        return (float(backend.to_numpy(max_row_error)), float(backend.to_numpy(max_col_error)))

    def _compute_effective_rank(self, coupling: "Array") -> int:
        """Compute effective rank (number of entries above numerical noise floor).

        Threshold is derived from machine epsilon - values below sqrt(eps)
        are numerically meaningless.
        """
        backend = self._backend
        eps = machine_epsilon(backend, coupling)
        threshold = math.sqrt(float(backend.to_numpy(eps)))
        mask = coupling >= threshold
        count = backend.sum(backend.astype(mask, coupling.dtype))
        backend.eval(count)
        return int(backend.to_numpy(count))

    def _compute_dimension_confidences(self, coupling: "Array") -> list[float]:
        """Compute per-dimension confidence from coupling entropy."""
        backend = self._backend
        n = coupling.shape[0]
        m = coupling.shape[1]

        if n == 0:
            return []

        # Row sums
        row_sums = backend.sum(coupling, axis=1, keepdims=True)
        eps = division_epsilon(backend, coupling)
        row_sums = backend.maximum(row_sums, backend.full(row_sums.shape, eps))

        # Normalized probabilities
        probs = coupling / row_sums

        # Entropy per row: -sum(p * log(p))
        # Avoid log(0) by clamping
        probs_safe = backend.maximum(probs, backend.full(probs.shape, eps))
        log_probs = backend.log(probs_safe)
        entropy_per_row = -backend.sum(probs * log_probs, axis=1)

        # Max entropy
        max_entropy = math.log(m) if m > 1 else 1.0

        # Confidence = 1 - normalized_entropy
        normalized_entropy = entropy_per_row / max_entropy
        confidence = 1.0 - normalized_entropy
        confidence = backend.maximum(confidence, backend.zeros_like(confidence))

        backend.eval(confidence)
        return backend.to_numpy(confidence).tolist()
