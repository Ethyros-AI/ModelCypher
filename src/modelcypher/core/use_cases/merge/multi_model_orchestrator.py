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

"""Multi-model merge orchestrator with consensus correction.

Orchestrates N-model merging with two distinct phases:
1. CORRECTION: Fix concepts the target learned wrong (move to consensus)
2. ADDITION: Add concepts from sources that target doesn't have

The key insight: when 5 models agree on a concept's position and the 6th
differs, the 6th is WRONG. We use GPA (Generalized Procrustes Analysis)
to find consensus and detect outliers.

Unlike standard null-space addition which preserves target behavior,
correction intentionally CHANGES behavior because the current behavior
is incorrect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.consensus_corrector import ConsensusCorrector
from modelcypher.core.domain.geometry.outlier_detector import OutlierDetector

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.cross_grounding_transfer import (
        RelationalStressProfile,
    )
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptCorrection:
    """A correction to apply to a specific concept."""

    concept_name: str
    weight_delta: "Array"
    layer_name: str
    stress_reduction: float


@dataclass(frozen=True)
class MultiModelMergeResult:
    """Result of multi-model merge with correction + addition."""

    # Merged weights per layer
    merged_weights: dict[str, "Array"]

    # Correction details
    concepts_corrected: tuple[str, ...]
    concepts_added: tuple[str, ...]
    concepts_unchanged: tuple[str, ...]

    # Metrics
    total_corrections: int
    total_additions: int
    mean_stress_reduction: float

    # Per-model summary
    model_count: int
    outlier_models: tuple[int, ...]


@dataclass
class MultiModelMergeState:
    """Mutable state during multi-model merge."""

    # Stress profiles per model per concept
    stress_profiles: dict[str, list["RelationalStressProfile"]] = field(
        default_factory=dict
    )

    # Corrections to apply
    corrections: list[ConceptCorrection] = field(default_factory=list)

    # Concepts identified for each category
    outlier_concepts: set[str] = field(default_factory=set)
    source_only_concepts: set[str] = field(default_factory=set)
    shared_concepts: set[str] = field(default_factory=set)


class MultiModelMergeOrchestrator:
    """Orchestrate N-model merge with correction + addition.

    Two-phase pipeline:
    1. CORRECT: Move outlier concepts to consensus (no null-space)
    2. ADD: Add source-only concepts via Ghost Anchors (null-space)
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort",
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            model_loader: Port for loading model weights.
            backend: Compute backend (defaults to system default).
        """
        self._model_loader = model_loader
        self._backend = backend or get_default_backend()
        self._outlier_detector = OutlierDetector(self._backend)
        self._corrector = ConsensusCorrector(self._backend)

    def detect_outliers_from_profiles(
        self,
        profiles_per_model: list[list["RelationalStressProfile"]],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Detect which models are outliers across all concepts.

        Args:
            profiles_per_model: List of profiles per model, where each model
                has profiles for multiple concepts.

        Returns:
            (consensus_model_indices, outlier_model_indices)
        """
        n_models = len(profiles_per_model)
        if n_models < 2:
            return tuple(range(n_models)), ()

        # Aggregate per-model errors across concepts
        # For each model, compute mean distance to other models
        model_mean_distances = []

        for model_idx in range(n_models):
            total_dist = 0.0
            n_comparisons = 0

            for concept_idx in range(len(profiles_per_model[model_idx])):
                profile_i = profiles_per_model[model_idx][concept_idx]

                for other_idx in range(n_models):
                    if other_idx != model_idx:
                        if concept_idx < len(profiles_per_model[other_idx]):
                            profile_j = profiles_per_model[other_idx][concept_idx]
                            total_dist += profile_i.distance_to(profile_j)
                            n_comparisons += 1

            if n_comparisons > 0:
                model_mean_distances.append(total_dist / n_comparisons)
            else:
                model_mean_distances.append(0.0)

        # Use z-score detection on mean distances
        result = self._outlier_detector.detect_from_gpa(model_mean_distances)

        return result.consensus_indices, result.outlier_indices

    def compute_concept_corrections(
        self,
        target_positions: dict[str, "Array"],
        target_anchors: dict[str, "Array"],
        target_activations: "Array",
        target_weights: "Array",
        consensus_profiles: dict[str, "Array"],
        outlier_concepts: set[str],
    ) -> list[ConceptCorrection]:
        """Compute corrections for outlier concepts.

        Args:
            target_positions: Current positions per concept in target space.
            target_anchors: Universal anchor positions in target space.
            target_activations: Target activations for lstsq.
            target_weights: Target weights for the layer.
            consensus_profiles: Consensus stress vectors per concept.
            outlier_concepts: Concepts where target is an outlier.

        Returns:
            List of corrections to apply.
        """
        corrections = []

        for concept_name in outlier_concepts:
            if concept_name not in target_positions:
                continue
            if concept_name not in consensus_profiles:
                continue

            target_pos = target_positions[concept_name]
            consensus_stress = consensus_profiles[concept_name]

            result = self._corrector.compute_correction_delta(
                target_position=target_pos,
                consensus_stress=consensus_stress,
                target_anchors=target_anchors,
                target_activations=target_activations,
                target_weights=target_weights,
            )

            corrections.append(
                ConceptCorrection(
                    concept_name=concept_name,
                    weight_delta=result.weight_delta,
                    layer_name="",  # Set by caller
                    stress_reduction=result.stress_reduction,
                )
            )

            logger.info(
                "Computed correction for concept '%s': stress_reduction=%.4f",
                concept_name,
                result.stress_reduction,
            )

        return corrections

    def apply_corrections(
        self,
        target_weights: "Array",
        corrections: list[ConceptCorrection],
    ) -> "Array":
        """Apply all corrections to target weights.

        Args:
            target_weights: Original target weights.
            corrections: List of corrections to apply.

        Returns:
            Corrected weights.
        """
        b = self._backend
        result = target_weights

        for correction in corrections:
            result = self._corrector.apply_correction(result, correction.weight_delta)

        return result

    def merge_summary(
        self,
        corrections: list[ConceptCorrection],
        additions: list[str],
        model_count: int,
        outlier_model_indices: tuple[int, ...],
    ) -> MultiModelMergeResult:
        """Create summary of merge operation.

        Args:
            corrections: Applied corrections.
            additions: Concepts added via null-space.
            model_count: Number of models merged.
            outlier_model_indices: Which models were outliers.

        Returns:
            Summary result.
        """
        concepts_corrected = tuple(c.concept_name for c in corrections)

        if corrections:
            mean_stress = sum(c.stress_reduction for c in corrections) / len(
                corrections
            )
        else:
            mean_stress = 0.0

        return MultiModelMergeResult(
            merged_weights={},  # Filled by caller
            concepts_corrected=concepts_corrected,
            concepts_added=tuple(additions),
            concepts_unchanged=(),  # Filled by caller
            total_corrections=len(corrections),
            total_additions=len(additions),
            mean_stress_reduction=mean_stress,
            model_count=model_count,
            outlier_models=outlier_model_indices,
        )
