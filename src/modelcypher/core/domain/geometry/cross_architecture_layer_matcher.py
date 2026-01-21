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

"""Cross-architecture layer matcher.

Finds layer correspondence between models using Hierarchical Optimal Transport (HOT).
HOT produces soft couplings that are converted to discrete mappings.

References:
    - Shah, S. & Khosla, M. (2025). "Representational Alignment Across Model
      Layers and Brain Regions with Hierarchical Optimal Transport."
      arXiv:2510.01706, ICLR 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
from modelcypher.core.domain.geometry.hot_layer_matcher import (
    HOTLayerMatcher,
    coupling_to_assignment,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_nan,
    machine_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_pairwise_metrics

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


# All matching uses uniform anchor weights; all concept types contribute equally.


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
class WeightedLayerSource:
    """Weighted source layers for a target layer.

    Used when multiple source layers contribute to one target (many-to-one),
    weighted by CKA similarity.
    """

    target_layer: int
    source_weights: dict[int, float]  # source_layer -> weight (normalized to sum=1)


@dataclass(frozen=True)
class InterpolatedLayerMapping:
    """Interpolated mapping for one-to-many case.

    When target has more layers than source, a target layer may fall
    between two source layers. This provides interpolation weights.
    """

    target_layer: int
    source_layer_low: int  # Lower source layer index
    source_layer_high: int  # Higher source layer index (may equal low)
    weight_low: float  # Weight for lower source (0-1)
    weight_high: float  # Weight for higher source (0-1)


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
    # Weighted mappings for cross-architecture transfer
    many_to_one_weights: list[WeightedLayerSource] | None = None
    one_to_many_interpolation: list[InterpolatedLayerMapping] | None = None


class CrossArchitectureLayerMatcher:
    """Finds optimal layer correspondence between cross-architecture models.

    Uses Hierarchical Optimal Transport (HOT) for soft layer matching,
    then converts to discrete mappings.
    """

    @staticmethod
    def find_correspondence(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        jaccard_matrix: list[list[float]] | None = None,
    ) -> Result:
        """Find layer correspondence between two concept response matrices.

        Uses Hierarchical Optimal Transport (HOT) to find optimal alignment
        between layers. HOT produces soft couplings that naturally handle
        depth mismatches and allow many-to-many correspondences.

        All anchor categories contribute equally (uniform weights).

        Args:
            source_crm: Concept response matrix from source model.
            target_crm: Concept response matrix from target model.
            jaccard_matrix: Optional Jaccard similarity matrix (diagnostics only).

        Returns:
            Complete matching result with validation metrics.
        """
        backend = get_default_backend()

        # Step 1: Extract activation matrices from CRMs
        common_anchors = sorted(source_crm.common_anchor_ids(target_crm))
        if len(common_anchors) < 2:
            logger.warning("Insufficient common anchors for HOT matching: %d", len(common_anchors))
            return CrossArchitectureLayerMatcher._empty_result(source_crm, target_crm)

        source_activations = CrossArchitectureLayerMatcher._extract_layer_activations(
            source_crm, common_anchors, backend
        )
        target_activations = CrossArchitectureLayerMatcher._extract_layer_activations(
            target_crm, common_anchors, backend
        )

        if not source_activations or not target_activations:
            logger.warning("No valid layer activations for HOT matching")
            return CrossArchitectureLayerMatcher._empty_result(source_crm, target_crm)

        # Step 2: Run HOT layer matcher
        matcher = HOTLayerMatcher(backend)
        hot_result = matcher.match(source_activations, target_activations)

        # Step 3: Convert soft coupling to discrete assignment
        assignment = coupling_to_assignment(
            hot_result.layer_coupling,
            hot_result.source_layers,
            hot_result.target_layers,
            backend,
        )

        # Step 4: Compute CKA matrix for compatibility and diagnostics
        anchor_weights = AnchorCategoryWeights.uniform()
        weighted_cka = CrossArchitectureLayerMatcher._compute_weighted_cka_matrix(
            source_crm, target_crm, anchor_weights
        )
        cka_matrix = weighted_cka if weighted_cka else source_crm.compute_cka_matrix(target_crm)

        source_count = source_crm.layer_count
        target_count = target_crm.layer_count

        # Build alignment path from assignment
        alignment_path: list[tuple[int, int]] = []
        for target_layer in sorted(assignment.keys()):
            source_layer = assignment[target_layer]
            alignment_path.append((source_layer, target_layer))

        # Build LayerMapping objects with CKA scores
        mappings: list[LayerMapping] = []
        for source, target in alignment_path:
            cka = (
                cka_matrix[source][target]
                if source < len(cka_matrix) and target < len(cka_matrix[0])
                else 0.0
            )
            mappings.append(
                LayerMapping(
                    source_layer=source,
                    target_layer=target,
                    cka=float(cka),
                    combined_score=float(cka),
                    is_skipped=False,
                )
            )

        h2_validation = CrossArchitectureLayerMatcher._validate_h2(mappings)
        mean_cka = sum(mapping.cka for mapping in mappings) / float(len(mappings)) if mappings else 0.0
        eps = machine_epsilon(backend, backend.array([1.0]))
        aligned = bool(mappings) and all(mapping.cka >= 1.0 - eps for mapping in mappings)

        visualization = VisualizationData(
            cka_matrix=cka_matrix,
            combined_matrix=cka_matrix,
            alignment_path=alignment_path,
            source_layer_count=source_count,
            target_layer_count=target_count,
        )

        # Compute weighted mappings for cross-architecture transfer
        many_to_one = CrossArchitectureLayerMatcher.compute_many_to_one_weights(
            cka_matrix, alignment_path
        )
        one_to_many = CrossArchitectureLayerMatcher.compute_one_to_many_interpolation(
            cka_matrix, alignment_path
        )

        return Result(
            mappings=mappings,
            mean_cka=float(mean_cka),
            aligned=aligned,
            h2_validation=h2_validation,
            visualization_data=visualization,
            source_model=source_crm.model_identifier,
            target_model=target_crm.model_identifier,
            many_to_one_weights=many_to_one,
            one_to_many_interpolation=one_to_many,
        )

    @staticmethod
    def _extract_layer_activations(
        crm: ConceptResponseMatrix,
        anchors: list[str],
        backend: "Backend",
    ) -> dict[int, "Array"]:
        """Extract activation matrices from CRM for each layer.

        Args:
            crm: Concept response matrix.
            anchors: List of anchor IDs to extract.
            backend: Backend for array operations.

        Returns:
            Dict mapping layer index to activation matrix [n_anchors, hidden_dim].
        """
        from modelcypher.ports.backend import Array

        layer_activations: dict[int, Array] = {}

        for layer in range(crm.layer_count):
            acts = crm._extract_activations(layer, anchors)
            if acts is not None and len(acts) > 0:
                arr = backend.array(acts)
                backend.eval(arr)
                layer_activations[layer] = arr

        return layer_activations

    @staticmethod
    def _empty_result(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
    ) -> Result:
        """Return empty result when matching fails."""
        return Result(
            mappings=[],
            mean_cka=0.0,
            aligned=False,
            h2_validation=H2ValidationResult(
                mean_cka=0.0,
                min_cka=0.0,
                max_cka=0.0,
                position_correlation=0.0,
            ),
            visualization_data=VisualizationData(
                cka_matrix=[],
                combined_matrix=None,
                alignment_path=[],
                source_layer_count=source_crm.layer_count,
                target_layer_count=target_crm.layer_count,
            ),
            source_model=source_crm.model_identifier,
            target_model=target_crm.model_identifier,
            many_to_one_weights=None,
            one_to_many_interpolation=None,
        )

    @staticmethod
    def compute_many_to_one_weights(
        cka_matrix: list[list[float]],
        alignment_path: list[tuple[int, int]],
    ) -> list[WeightedLayerSource]:
        """Compute weighted source contributions for each target layer.

        When multiple source layers map to the same target layer (many-to-one),
        weights are derived from CKA similarity values - the invariant structure
        determines contribution, not an arbitrary blend parameter.

        Args:
            cka_matrix: CKA similarity matrix [source_layers x target_layers].
            alignment_path: DTW alignment path as (source, target) pairs.

        Returns:
            List of WeightedLayerSource, one per target layer.
        """
        if not alignment_path or not cka_matrix:
            return []

        backend = get_default_backend()
        n_target = len(cka_matrix[0]) if cka_matrix else 0

        # Group source layers by target
        target_to_sources: dict[int, list[tuple[int, float]]] = {}
        for source, target in alignment_path:
            cka = cka_matrix[source][target] if source < len(cka_matrix) else 0.0
            if target not in target_to_sources:
                target_to_sources[target] = []
            target_to_sources[target].append((source, cka))

        result: list[WeightedLayerSource] = []
        eps = division_epsilon(backend, backend.array([1.0]))

        for target_layer in range(n_target):
            if target_layer in target_to_sources:
                sources = target_to_sources[target_layer]
                # Normalize weights by CKA
                total_cka = sum(cka for _, cka in sources)
                if total_cka < eps:
                    # Equal weights if all CKA near zero
                    n = len(sources)
                    weights = {src: 1.0 / n for src, _ in sources}
                else:
                    weights = {src: cka / total_cka for src, cka in sources}
                result.append(WeightedLayerSource(target_layer, weights))
            else:
                # Target layer not in path - find nearest source by CKA
                best_source = 0
                best_cka = 0.0
                for src in range(len(cka_matrix)):
                    if cka_matrix[src][target_layer] > best_cka:
                        best_cka = cka_matrix[src][target_layer]
                        best_source = src
                result.append(WeightedLayerSource(target_layer, {best_source: 1.0}))

        return result

    @staticmethod
    def compute_one_to_many_interpolation(
        cka_matrix: list[list[float]],
        alignment_path: list[tuple[int, int]],
    ) -> list[InterpolatedLayerMapping]:
        """Compute interpolated source mapping for each target layer.

        When source has fewer layers than target (one-to-many), target layers
        between alignment points are interpolated from adjacent source layers.
        Interpolation weights are derived from relative position and CKA values.

        Args:
            cka_matrix: CKA similarity matrix [source_layers x target_layers].
            alignment_path: DTW alignment path as (source, target) pairs.

        Returns:
            List of InterpolatedLayerMapping, one per target layer.
        """
        if not alignment_path or not cka_matrix:
            return []

        backend = get_default_backend()
        n_source = len(cka_matrix)
        n_target = len(cka_matrix[0]) if cka_matrix else 0

        # Build mapping from alignment path
        # For each target layer, find bounding source layers
        path_targets = [t for _, t in alignment_path]
        path_sources = [s for s, _ in alignment_path]

        result: list[InterpolatedLayerMapping] = []
        eps = division_epsilon(backend, backend.array([1.0]))

        for target_layer in range(n_target):
            # Find where this target falls in the alignment path
            if target_layer in path_targets:
                # Direct mapping exists
                idx = path_targets.index(target_layer)
                source = path_sources[idx]
                result.append(InterpolatedLayerMapping(
                    target_layer=target_layer,
                    source_layer_low=source,
                    source_layer_high=source,
                    weight_low=1.0,
                    weight_high=0.0,
                ))
            else:
                # Target not in path - interpolate between adjacent path points
                # Find bounding path entries
                low_idx = -1
                high_idx = len(path_targets)

                for i, t in enumerate(path_targets):
                    if t < target_layer:
                        low_idx = i
                    elif t > target_layer and high_idx == len(path_targets):
                        high_idx = i
                        break

                if low_idx < 0:
                    # Before first path point - use first source
                    source = path_sources[0]
                    result.append(InterpolatedLayerMapping(
                        target_layer=target_layer,
                        source_layer_low=source,
                        source_layer_high=source,
                        weight_low=1.0,
                        weight_high=0.0,
                    ))
                elif high_idx >= len(path_targets):
                    # After last path point - use last source
                    source = path_sources[-1]
                    result.append(InterpolatedLayerMapping(
                        target_layer=target_layer,
                        source_layer_low=source,
                        source_layer_high=source,
                        weight_low=1.0,
                        weight_high=0.0,
                    ))
                else:
                    # Interpolate between low and high
                    source_low = path_sources[low_idx]
                    source_high = path_sources[high_idx]
                    target_low = path_targets[low_idx]
                    target_high = path_targets[high_idx]

                    # Linear interpolation based on target position
                    span = target_high - target_low
                    if span < eps:
                        weight_low = 0.5
                    else:
                        weight_low = (target_high - target_layer) / span

                    weight_high = 1.0 - weight_low

                    result.append(InterpolatedLayerMapping(
                        target_layer=target_layer,
                        source_layer_low=source_low,
                        source_layer_high=source_high,
                        weight_low=weight_low,
                        weight_high=weight_high,
                    ))

        return result

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
        backend = get_default_backend()

        # Vectorized ranking using argsort
        def ranks_vectorized(values_arr):
            # argsort gives indices that would sort the array
            sorted_indices = backend.argsort(values_arr)
            # Create rank array: for each position in sorted order, assign rank
            # We need the inverse: for each original position, what's its rank?
            ranks_arr = backend.zeros((n,), dtype=precision_dtype(backend))
            # Use scatter-like operation: ranks[sorted_indices[i]] = i + 1
            # Since we don't have scatter, use argsort of argsort
            inverse_indices = backend.argsort(sorted_indices)
            # Ranks are 1-indexed
            ranks_arr = backend.astype(
                inverse_indices, precision_dtype(backend, reference=inverse_indices)
            ) + 1.0
            return ranks_arr

        x_arr = backend.array(x, dtype=precision_dtype(backend))
        y_arr = backend.array(y, dtype=precision_dtype(backend))
        rank_x_arr = ranks_vectorized(x_arr)
        rank_y_arr = ranks_vectorized(y_arr)
        backend.eval(rank_x_arr, rank_y_arr)

        mean_x = backend.mean(rank_x_arr)
        mean_y = backend.mean(rank_y_arr)
        centered_x = rank_x_arr - mean_x
        centered_y = rank_y_arr - mean_y
        centered_x_mat = backend.reshape(centered_x, (1, -1))
        centered_y_mat = backend.reshape(centered_y, (1, -1))
        cos_arr, _ = geodesic_pairwise_metrics(centered_x_mat, centered_y_mat, backend)
        backend.eval(cos_arr)
        if int(cos_arr.shape[0]) == 0:
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
        backend = get_default_backend()

        # Vectorized weighted sum using backend operations
        combined_arr = backend.zeros((rows, cols), dtype=precision_dtype(backend))

        for category, weight in normalized.items():
            anchors = anchors_by_category.get(category)
            if not anchors:
                continue
            matrix = CrossArchitectureLayerMatcher._compute_cka_matrix_for_anchors(
                source_crm,
                target_crm,
                anchors,
            )
            matrix_arr = backend.array(matrix, dtype=precision_dtype(backend))
            combined_arr = combined_arr + weight * matrix_arr

        backend.eval(combined_arr)
        return backend.tolist(combined_arr)

    @staticmethod
    def _compute_cka_matrix_for_anchors(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        anchors: list[str],
    ) -> list[list[float]]:
        """Compute CKA matrix using Gram alignment.

        For each (source_layer, target_layer) pair, measures geodesic CKA
        without applying any alignment transform.

        Geodesic CKA near 1.0 suggests strong structural overlap.
        Lower CKA indicates divergent structure or limited probe coverage.
        """
        # Use geodesic CKA to MEASURE similarity between layers.
        # NOTE: We DO NOT use find_alignment here because:
        # - find_alignment TRANSFORMS source to match target (linear alignment on probes)
        # - Here we want to MEASURE native similarity WITHOUT transformation
        from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

        rows = source_crm.layer_count
        cols = target_crm.layer_count
        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]

        if not anchors:
            return matrix

        sorted_anchors = sorted(anchors)
        backend = get_default_backend()

        # Pre-cache layer activations for efficiency
        source_cache: dict[int, "Array"] = {}
        target_cache: dict[int, "Array"] = {}

        for layer in range(rows):
            acts = source_crm._extract_activations(layer, sorted_anchors)
            if acts is not None:
                arr = backend.array(acts)
                backend.eval(arr)
                source_cache[layer] = arr

        for layer in range(cols):
            acts = target_crm._extract_activations(layer, sorted_anchors)
            if acts is not None:
                arr = backend.array(acts)
                backend.eval(arr)
                target_cache[layer] = arr

        # Compute native CKA (WITHOUT transformation) for each layer pair
        for source_layer in range(rows):
            source_arr = source_cache.get(source_layer)
            if source_arr is None:
                continue

            for target_layer in range(cols):
                target_arr = target_cache.get(target_layer)
                if target_arr is None:
                    continue

                # Measure native CKA similarity between layers
                cka = compute_geodesic_cka(source_arr, target_arr, backend)
                matrix[source_layer][target_layer] = max(0.0, min(1.0, cka))

        return matrix
