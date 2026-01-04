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

"""Cross-Architecture Layer Matcher.

Finds optimal layer correspondence between cross-architecture models using
dynamic programming for monotonic alignment with CKA similarity.

Notes
-----
Fundamental Principle:
    Concepts occupy fixed probability clouds in hyperspace. Knowledge is a
    high-dimensional shape that is invariant across models. Every LLM learns
    the same conceptual shapes because those shapes represent knowledge itself.
    Model family (Qwen, Llama, Mistral, etc.) is irrelevant - the geometry
    of knowledge is universal. Think of LLM weights as high-dimensional
    Legos that precisely fit every other Lego.

Theoretical Foundation:
    Different neural architectures have functionally equivalent layers at different indices.
    A 12-layer transformer and a 24-layer transformer may have corresponding "attention to
    syntax" functionality at layers 4 and 8 respectively. This matcher finds such correspondences.

Algorithm:
    1. Compute CKA similarity matrix between all layer pairs
    2. Use dynamic programming for monotonic alignment (layers must correspond in order)

    Unlike greedy matching, DP-based alignment respects the sequential nature of neural
    network layers - earlier layers in model A should map to earlier layers in model B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_nan,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_pairwise_metrics

logger = logging.getLogger(__name__)

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorCategory,
    ConceptResponseMatrix,
)


@dataclass(frozen=True)
class AnchorCategoryWeights:
    """Per-category weights for anchor-specific CKA computation."""

    semantic_prime: float = 0.0
    computational_gate: float = 0.0
    sequence_invariant: float = 0.0
    metaphor_invariant: float = 0.0
    conceptual_genealogy: float = 0.0

    @staticmethod
    def uniform() -> AnchorCategoryWeights:
        """Create uniform weights (1.0 for all categories)."""
        return AnchorCategoryWeights(
            semantic_prime=1.0,
            computational_gate=1.0,
            sequence_invariant=1.0,
            metaphor_invariant=1.0,
            conceptual_genealogy=1.0,
        )

    @property
    def total_weight(self) -> float:
        """Sum of all weights."""
        return (
            self.semantic_prime
            + self.computational_gate
            + self.sequence_invariant
            + self.metaphor_invariant
            + self.conceptual_genealogy
        )

    def weight_for(self, category: AnchorCategory) -> float:
        """Get weight for a specific anchor category."""
        # Map category values to weights
        mapping = {
            "prime": self.semantic_prime,
            "gate": self.computational_gate,
            "sequence": self.sequence_invariant,
            "metaphor": self.metaphor_invariant,
            "genealogy": self.conceptual_genealogy,
        }
        return mapping.get(category.value, 0.0)

    def normalized(self, available_categories: set[AnchorCategory]) -> dict[AnchorCategory, float]:
        """Get normalized weights for available categories."""
        weights: dict[AnchorCategory, float] = {}
        for category in AnchorCategory:
            if category in available_categories:
                value = self.weight_for(category)
                if value > 0:
                    weights[category] = value

        total = sum(weights.values())
        if total <= 0:
            return {}

        return {k: v / total for k, v in weights.items()}


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All matching uses uniform anchor weights - all concept types contribute equally.
# There is exactly ONE correct way to match layers across architectures.
# =============================================================================


# ConfidenceLevel enum removed - use raw CKA values directly.


@dataclass(frozen=True)
class LayerMapping:
    """Layer mapping between source and target models.

    Attributes
    ----------
    source_layer : int
        Source model layer index.
    target_layer : int
        Target model layer index.
    cka : float
        CKA similarity (0-1), which measures the confidence.
    combined_score : float
        Combined score (CKA only).
    """

    source_layer: int
    target_layer: int
    cka: float
    combined_score: float
    is_skipped: bool = False


@dataclass(frozen=True)
class H2ValidationResult:
    """H2 validation result for layer correspondence.

    Uses raw CKA values as the signal - no binning into confidence levels.

    Attributes
    ----------
    mean_cka : float
        Mean CKA across mappings.
    min_cka : float
        Minimum CKA across mappings.
    max_cka : float
        Maximum CKA across mappings.
    position_correlation : float
        Correlation between source and target layer positions.
    """

    mean_cka: float
    min_cka: float
    max_cka: float
    position_correlation: float


@dataclass(frozen=True)
class VisualizationData:
    cka_matrix: list[list[float]]
    combined_matrix: list[list[float]] | None
    alignment_path: list[tuple[int, int]]
    source_layer_count: int
    target_layer_count: int


@dataclass(frozen=True)
class Result:
    mappings: list[LayerMapping]
    mean_cka: float
    aligned: bool
    h2_validation: H2ValidationResult
    visualization_data: VisualizationData
    source_model: str
    target_model: str


class CrossArchitectureLayerMatcher:
    """Finds optimal layer correspondence between cross-architecture models.

    Uses dynamic programming for monotonic alignment with CKA similarity,
    optionally weighted by anchor category.
    """

    @staticmethod
    def find_correspondence(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        jaccard_matrix: list[list[float]] | None = None,
    ) -> Result:
        """Find layer correspondence between two concept response matrices.

        Uses dynamic programming to find optimal monotonic alignment between layers,
        respecting the sequential nature of neural network processing.

        All anchor categories contribute equally (uniform weights). There is exactly
        ONE correct way to match layers across architectures.

        Args:
            source_crm: Concept response matrix from source model.
            target_crm: Concept response matrix from target model.
            jaccard_matrix: Optional Jaccard similarity matrix (diagnostics only).

        Returns:
            Complete matching result with validation metrics.
        """
        # Always use uniform anchor weights - all concept types contribute equally
        anchor_weights = AnchorCategoryWeights.uniform()

        # Step 1: Compute CKA matrix (weighted by anchor category)
        weighted_cka = CrossArchitectureLayerMatcher._compute_weighted_cka_matrix(
            source_crm, target_crm, anchor_weights
        )
        cka_matrix = weighted_cka if weighted_cka else source_crm.compute_cka_matrix(target_crm)

        source_count = source_crm.layer_count
        target_count = target_crm.layer_count

        combined_matrix = cka_matrix

        dp_path, _ = CrossArchitectureLayerMatcher._dynamic_programming_alignment(cka_matrix)

        mappings: list[LayerMapping] = []
        for source, target in dp_path:
            cka = (
                cka_matrix[source][target]
                if source < len(cka_matrix) and target < len(cka_matrix[0])
                else 0.0
            )
            combined = (
                combined_matrix[source][target]
                if source < len(combined_matrix) and target < len(combined_matrix[0])
                else 0.0
            )
            mappings.append(
                LayerMapping(
                    source_layer=source,
                    target_layer=target,
                    cka=float(cka),
                    combined_score=float(combined),
                    is_skipped=False,
                )
            )

        h2_validation = CrossArchitectureLayerMatcher._validate_h2(mappings)
        mean_cka = sum(mapping.cka for mapping in mappings) / float(len(mappings)) if mappings else 0.0
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        aligned = bool(mappings) and all(mapping.cka >= 1.0 - eps for mapping in mappings)

        visualization = VisualizationData(
            cka_matrix=cka_matrix,
            combined_matrix=combined_matrix,
            alignment_path=dp_path,
            source_layer_count=source_count,
            target_layer_count=target_count,
        )

        return Result(
            mappings=mappings,
            mean_cka=float(mean_cka),
            aligned=aligned,
            h2_validation=h2_validation,
            visualization_data=visualization,
            source_model=source_crm.model_identifier,
            target_model=target_crm.model_identifier,
        )

    @staticmethod
    def _dynamic_programming_alignment(
        similarity_matrix: list[list[float]],
    ) -> tuple[list[tuple[int, int]], float]:
        m = len(similarity_matrix)
        if m == 0:
            return [], 0.0
        n = len(similarity_matrix[0]) if similarity_matrix[0] else 0
        if n == 0:
            return [], 0.0

        backend = get_default_backend()
        sim = backend.array(similarity_matrix, dtype="float32")
        backend.eval(sim)

        neg_inf = -float(backend.finfo().max)
        index = backend.arange(n, dtype="int32")
        row_idx = backend.reshape(index, (n, 1))
        col_idx = backend.reshape(index, (1, n))
        prefix_mask = col_idx <= row_idx
        neg_inf_mat = backend.full((n, n), neg_inf)

        first_row = backend.take(sim, backend.array([0], dtype="int32"), axis=0)
        first_row = backend.reshape(first_row, (n,))
        dp_rows = [first_row]
        parent: list[list[int]] = [[-1 for _ in range(n)]]

        for i in range(1, m):
            row = backend.take(sim, backend.array([i], dtype="int32"), axis=0)
            row = backend.reshape(row, (n,))
            dp_prev = dp_rows[-1]
            dp_prev_row = backend.broadcast_to(backend.reshape(dp_prev, (1, n)), (n, n))
            masked_prev = backend.where(prefix_mask, dp_prev_row, neg_inf_mat)
            prefix_max = backend.max(masked_prev, axis=1)
            prefix_arg = backend.argmax(masked_prev, axis=1)
            dp_row = row + prefix_max
            backend.eval(prefix_arg, dp_row)
            dp_rows.append(dp_row)
            parent.append([int(x) for x in backend.tolist(prefix_arg)])

        best_j_arr = backend.argmax(dp_rows[-1])
        best_score_arr = backend.max(dp_rows[-1])
        backend.eval(best_j_arr, best_score_arr)
        best_j = int(backend.to_scalar(best_j_arr))
        best_score = float(backend.to_scalar(best_score_arr))

        path: list[tuple[int, int]] = []
        j = best_j
        for i in range(m - 1, -1, -1):
            path.append((i, j))
            if i == 0:
                break
            j = parent[i][j]
        path.reverse()
        return path, best_score

    # _classify_confidence method removed - use raw CKA values directly.

    @staticmethod
    def _validate_h2(mappings: list[LayerMapping]) -> H2ValidationResult:
        """Compute layer correspondence statistics using raw CKA values.

        Returns raw measurements - callers determine validation thresholds.
        """
        if not mappings:
            return H2ValidationResult(
                mean_cka=0.0,
                min_cka=0.0,
                max_cka=0.0,
                position_correlation=0.0,
            )

        mean_cka = sum(mapping.cka for mapping in mappings) / float(len(mappings))
        min_cka = min(mapping.cka for mapping in mappings)
        max_cka = max(mapping.cka for mapping in mappings)

        source_positions = [float(mapping.source_layer) for mapping in mappings]
        target_positions = [float(mapping.target_layer) for mapping in mappings]
        position_corr = CrossArchitectureLayerMatcher._spearman_correlation(
            source_positions, target_positions
        )

        return H2ValidationResult(
            mean_cka=float(mean_cka),
            min_cka=float(min_cka),
            max_cka=float(max_cka),
            position_correlation=float(position_corr),
        )

    @staticmethod
    def _spearman_correlation(x: list[float], y: list[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        n = len(x)

        def ranks(values: list[float]) -> list[float]:
            sorted_indices = sorted(range(n), key=lambda idx: values[idx])
            result = [0.0] * n
            for rank, original_idx in enumerate(sorted_indices, start=1):
                result[original_idx] = float(rank)
            return result

        rank_x = ranks(x)
        rank_y = ranks(y)
        backend = get_default_backend()
        rank_x_arr = backend.array(rank_x)
        rank_y_arr = backend.array(rank_y)
        mean_x = backend.mean(rank_x_arr)
        mean_y = backend.mean(rank_y_arr)
        centered_x = rank_x_arr - mean_x
        centered_y = rank_y_arr - mean_y
        centered_x_mat = backend.reshape(centered_x, (1, -1))
        centered_y_mat = backend.reshape(centered_y, (1, -1))
        cos_arr, _ = geodesic_pairwise_metrics(centered_x_mat, centered_y_mat, backend)
        backend.eval(cos_arr)
        if cos_arr.size == 0:
            return 0.0
        corr = float(backend.to_scalar(cos_arr[0]))
        return 0.0 if is_nan(corr, backend) else corr

    @staticmethod
    def _compute_weighted_cka_matrix(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        weights: AnchorCategoryWeights | None,
    ) -> list[list[float]] | None:
        """Compute weighted CKA matrix using per-anchor-category weights.

        Args:
            source_crm: Source concept response matrix.
            target_crm: Target concept response matrix.
            weights: Anchor category weights.

        Returns:
            Weighted CKA matrix, or None if weights not provided or no valid categories.
        """
        if weights is None:
            return None

        common_anchors = source_crm.common_anchor_ids(target_crm)

        # Group anchors by category
        anchors_by_category: dict[AnchorCategory, list[str]] = {}
        for category in AnchorCategory:
            anchors = [a for a in common_anchors if a.startswith(category.prefix)]
            # Need at least two anchors to avoid degenerate CKA
            if len(anchors) >= 2:
                anchors_by_category[category] = anchors

        normalized = weights.normalized(set(anchors_by_category.keys()))
        if not normalized:
            return None

        rows = source_crm.layer_count
        cols = target_crm.layer_count
        combined = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for category, weight in normalized.items():
            anchors = anchors_by_category.get(category)
            if not anchors:
                continue
            matrix = CrossArchitectureLayerMatcher._compute_cka_matrix_for_anchors(
                source_crm,
                target_crm,
                anchors,
            )
            for i in range(rows):
                for j in range(cols):
                    combined[i][j] += weight * matrix[i][j]

        return combined

    @staticmethod
    def _compute_cka_matrix_for_anchors(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        anchors: list[str],
    ) -> list[list[float]]:
        """Compute CKA matrix restricted to a shared anchor list."""
        rows = source_crm.layer_count
        cols = target_crm.layer_count
        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]

        if not anchors:
            return matrix

        sorted_anchors = sorted(anchors)
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([1.0]))

        for source_layer in range(rows):
            source_acts = source_crm._extract_activations(source_layer, sorted_anchors)
            if source_acts is None:
                continue
            for target_layer in range(cols):
                target_acts = target_crm._extract_activations(target_layer, sorted_anchors)
                if target_acts is None:
                    continue
                cka = float(source_crm.compute_linear_cka(source_acts, target_acts))
                if cka < 0.0:
                    cka = 0.0
                if cka >= 1.0 - eps:
                    cka = 1.0
                matrix[source_layer][target_layer] = cka

        return matrix
