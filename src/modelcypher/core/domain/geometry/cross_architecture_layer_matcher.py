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

Theoretical Foundation:
    Different neural architectures have functionally equivalent layers at different indices.
    A 12-layer transformer and a 24-layer transformer may have corresponding "attention to
    syntax" functionality at layers 4 and 8 respectively. This matcher finds such correspondences.

Algorithm:
    1. Compute CKA similarity matrix between all layer pairs
    2. Optionally incorporate sparse fingerprint Jaccard similarity
    3. Use dynamic programming for monotonic alignment (layers must correspond in order)
    4. Classify matches by confidence level

Key Insight:
    Unlike greedy matching, DP-based alignment respects the sequential nature of neural
    network layers - earlier layers in model A should map to earlier layers in model B.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


@dataclass(frozen=True)
class Configuration:
    """Configuration for layer matching."""

    cka_weight: float = 0.7
    """Weight for dense CKA similarity (0-1)."""

    jaccard_weight: float = 0.3
    """Weight for sparse Jaccard similarity (0-1)."""

    max_skip: int = 3
    """Maximum consecutive layers that can be skipped."""

    skip_penalty: float = 0.2
    """Penalty per skipped layer (subtracted from score)."""

    min_cka_threshold: float = 0.3
    """Minimum CKA threshold for valid match."""

    high_confidence_threshold: float = 0.7
    """High confidence CKA threshold."""

    medium_confidence_threshold: float = 0.5
    """Medium confidence CKA threshold."""

    anchor_category_weights: AnchorCategoryWeights | None = None
    """Optional per-anchor-category weights for CKA computation."""

    @staticmethod
    def default() -> Configuration:
        """Default configuration."""
        return Configuration()

    @staticmethod
    def same_family() -> Configuration:
        """Configuration optimized for same-family models (stricter matching)."""
        return Configuration(
            cka_weight=0.8,
            jaccard_weight=0.2,
            max_skip=2,
            skip_penalty=0.3,
            min_cka_threshold=0.4,
            high_confidence_threshold=0.8,
            medium_confidence_threshold=0.6,
        )

    @staticmethod
    def cross_family() -> Configuration:
        """Configuration for cross-family models (more lenient matching)."""
        return Configuration(
            cka_weight=0.6,
            jaccard_weight=0.4,
            max_skip=4,
            skip_penalty=0.15,
            min_cka_threshold=0.25,
            high_confidence_threshold=0.6,
            medium_confidence_threshold=0.4,
        )


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    uncertain = "uncertain"


@dataclass(frozen=True)
class LayerMapping:
    source_layer: int
    target_layer: int
    cka: float
    combined_score: float
    confidence: ConfidenceLevel
    is_skipped: bool = False


@dataclass(frozen=True)
class H2ValidationResult:
    mean_cka: float
    high_confidence_proportion: float
    position_correlation: float
    is_validated: bool
    interpretation: str


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
    alignment_quality: float
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
        configuration: Configuration | None = None,
    ) -> Result:
        """Find layer correspondence between two concept response matrices.

        Uses dynamic programming to find optimal monotonic alignment between layers,
        respecting the sequential nature of neural network processing.

        Args:
            source_crm: Concept response matrix from source model.
            target_crm: Concept response matrix from target model.
            jaccard_matrix: Optional Jaccard similarity matrix from sparse fingerprints.
            configuration: Matching configuration.

        Returns:
            Complete matching result with validation metrics.
        """
        config = configuration or Configuration.default()

        # Step 1: Compute CKA matrix (optionally weighted by anchor category)
        weighted_cka = CrossArchitectureLayerMatcher._compute_weighted_cka_matrix(
            source_crm, target_crm, config.anchor_category_weights
        )
        cka_matrix = weighted_cka if weighted_cka else source_crm.compute_cka_matrix(target_crm)

        source_count = source_crm.layer_count
        target_count = target_crm.layer_count

        # Step 2: Combine CKA and Jaccard if provided
        if (
            jaccard_matrix is not None
            and len(jaccard_matrix) == source_count
            and jaccard_matrix
            and len(jaccard_matrix[0]) == target_count
        ):
            combined_matrix = CrossArchitectureLayerMatcher._combine_matrices(
                cka_matrix,
                jaccard_matrix,
                config.cka_weight,
                config.jaccard_weight,
            )
        else:
            combined_matrix = cka_matrix

        dp_path, _ = CrossArchitectureLayerMatcher._dynamic_programming_alignment(
            combined_matrix,
            max_skip=config.max_skip,
            skip_penalty=config.skip_penalty,
        )

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
            confidence = CrossArchitectureLayerMatcher._classify_confidence(cka, config)
            mappings.append(
                LayerMapping(
                    source_layer=source,
                    target_layer=target,
                    cka=float(cka),
                    combined_score=float(combined),
                    confidence=confidence,
                    is_skipped=cka < config.min_cka_threshold,
                )
            )

        h2_validation = CrossArchitectureLayerMatcher._validate_h2(mappings)
        valid_mappings = [mapping for mapping in mappings if not mapping.is_skipped]
        alignment_quality = (
            sum(mapping.cka for mapping in valid_mappings) / float(len(valid_mappings))
            if valid_mappings
            else 0.0
        )

        visualization = VisualizationData(
            cka_matrix=cka_matrix,
            combined_matrix=combined_matrix if jaccard_matrix is not None else None,
            alignment_path=dp_path,
            source_layer_count=source_count,
            target_layer_count=target_count,
        )

        return Result(
            mappings=mappings,
            alignment_quality=float(alignment_quality),
            h2_validation=h2_validation,
            visualization_data=visualization,
            source_model=source_crm.model_identifier,
            target_model=target_crm.model_identifier,
        )

    @staticmethod
    def _combine_matrices(
        cka: list[list[float]],
        jaccard: list[list[float]],
        cka_weight: float,
        jaccard_weight: float,
    ) -> list[list[float]]:
        rows = len(cka)
        if rows == 0:
            return cka
        cols = len(cka[0])
        combined = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                combined[i][j] = cka_weight * cka[i][j] + jaccard_weight * jaccard[i][j]
        return combined

    @staticmethod
    def _dynamic_programming_alignment(
        similarity_matrix: list[list[float]],
        max_skip: int,
        skip_penalty: float,
    ) -> tuple[list[tuple[int, int]], float]:
        m = len(similarity_matrix)
        if m == 0:
            return [], 0.0
        n = len(similarity_matrix[0]) if similarity_matrix[0] else 0
        if n == 0:
            return [], 0.0

        dp = [[float("-inf") for _ in range(n)] for _ in range(m)]
        parent: list[list[tuple[int, int] | None]] = [[None for _ in range(n)] for _ in range(m)]

        for j in range(n):
            dp[0][j] = float(similarity_matrix[0][j])

        for i in range(1, m):
            for j in range(n):
                score = float(similarity_matrix[i][j])
                for j_prev in range(0, j + 1):
                    skip_count = j - j_prev
                    if skip_count <= max_skip:
                        penalty = float(skip_count) * skip_penalty
                        candidate = dp[i - 1][j_prev] + score - penalty
                        if candidate > dp[i][j]:
                            dp[i][j] = candidate
                            parent[i][j] = (i - 1, j_prev)

                for i_prev in range(max(0, i - max_skip), i):
                    skip_count = i - i_prev
                    if skip_count <= max_skip:
                        penalty = float(skip_count) * skip_penalty
                        for j_prev in range(0, j + 1):
                            target_skip = j - j_prev
                            total_penalty = penalty + float(target_skip) * skip_penalty
                            candidate = dp[i_prev][j_prev] + score - total_penalty
                            if candidate > dp[i][j]:
                                dp[i][j] = candidate
                                parent[i][j] = (i_prev, j_prev)

        best_j = 0
        best_score = float("-inf")
        for j in range(n):
            if dp[m - 1][j] > best_score:
                best_score = dp[m - 1][j]
                best_j = j

        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = (m - 1, best_j)
        while current is not None:
            i, j = current
            path.append((i, j))
            current = parent[i][j]
        path.reverse()
        return path, float(best_score)

    @staticmethod
    def _classify_confidence(cka: float, config: Configuration) -> ConfidenceLevel:
        if cka >= config.high_confidence_threshold:
            return ConfidenceLevel.high
        if cka >= config.medium_confidence_threshold:
            return ConfidenceLevel.medium
        if cka >= config.min_cka_threshold:
            return ConfidenceLevel.low
        return ConfidenceLevel.uncertain

    @staticmethod
    def _validate_h2(mappings: list[LayerMapping]) -> H2ValidationResult:
        valid = [mapping for mapping in mappings if not mapping.is_skipped]
        if not valid:
            return H2ValidationResult(
                mean_cka=0.0,
                high_confidence_proportion=0.0,
                position_correlation=0.0,
                is_validated=False,
                interpretation="No valid layer mappings found.",
            )

        mean_cka = sum(mapping.cka for mapping in valid) / float(len(valid))
        high_count = sum(1 for mapping in valid if mapping.confidence == ConfidenceLevel.high)
        high_prop = float(high_count) / float(len(valid))

        source_positions = [float(mapping.source_layer) for mapping in valid]
        target_positions = [float(mapping.target_layer) for mapping in valid]
        position_corr = CrossArchitectureLayerMatcher._spearman_correlation(
            source_positions, target_positions
        )

        is_validated = mean_cka > 0.5 and high_prop > 0.6 and position_corr > 0.8
        if is_validated:
            interpretation = (
                "H2 validated: Layer correspondence hypothesis confirmed. Models have functionally equivalent "
                "layers at corresponding indices."
            )
        elif mean_cka > 0.3 and position_corr > 0.5:
            interpretation = (
                "H2 partially supported: Some functional correspondence exists, but models may organize concepts "
                "differently at certain depths."
            )
        else:
            interpretation = (
                "H2 not supported: Models have fundamentally different layer organization. Direct layer-to-layer "
                "mapping may not be meaningful."
            )

        return H2ValidationResult(
            mean_cka=float(mean_cka),
            high_confidence_proportion=float(high_prop),
            position_correlation=float(position_corr),
            is_validated=is_validated,
            interpretation=interpretation,
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
        mean_x = sum(rank_x) / float(n)
        mean_y = sum(rank_y) / float(n)

        sum_xy = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        for i in range(n):
            dx = rank_x[i] - mean_x
            dy = rank_y[i] - mean_y
            sum_xy += dx * dy
            sum_x2 += dx * dx
            sum_y2 += dy * dy
        denom = (sum_x2 * sum_y2) ** 0.5
        return sum_xy / denom if denom > 1e-10 else 0.0

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
            matrix = source_crm.compute_cka_matrix(target_crm, anchor_ids=anchors)
            for i in range(rows):
                for j in range(cols):
                    combined[i][j] += weight * matrix[i][j]

        return combined
