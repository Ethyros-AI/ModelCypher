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

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

# TypeVar for array types from any backend
Array = TypeVar("Array")

# --- Permutation Aligner Types ---


@dataclass
class PermutationAlignmentResult:
    permutation: Any  # Backend array or list
    signs: Any  # Backend array or list
    match_quality: float
    match_confidences: list[float]
    sign_flip_count: int
    is_sparse_permutation: bool = False
    assignment_indices: list[int] | None = None


@dataclass
class RebasinResult:
    aligned_weights: dict[str, Any]
    quality: float
    sign_flip_count: int


# --- Concept Detection Types ---


@dataclass(frozen=True)
class DetectedConcept:
    concept_id: str
    category: str
    similarity: float
    character_span: slice  # slice(start, end)
    trigger_text: str
    cross_modal_similarity: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    model_id: str
    prompt_id: str
    response_text: str
    detected_concepts: tuple[DetectedConcept, ...]
    mean_similarity: float
    mean_cross_modal_similarity: float | None
    timestamp: float = field(default_factory=time.time)

    @property
    def concept_sequence(self) -> tuple[str, ...]:
        return tuple(c.concept_id for c in self.detected_concepts)


@dataclass(frozen=True)
class ConceptComparisonResult:
    model_a: str
    model_b: str
    concept_path_a: tuple[str, ...]
    concept_path_b: tuple[str, ...]
    cka: float | None
    cosine_similarity: float | None
    aligned_concepts: tuple[str, ...]
    unique_to_a: tuple[str, ...]
    unique_to_b: tuple[str, ...]

    @property
    def alignment_ratio(self) -> float:
        total = len(set(self.concept_path_a) | set(self.concept_path_b))
        if total == 0:
            return 1.0
        return len(self.aligned_concepts) / float(total)


# --- Manifold Clusterer Types ---
# Note: ManifoldPoint, ManifoldRegion defined in manifold_profile.py
# Note: ClusteringResult defined in manifold_clusterer.py


# --- Intrinsic Dimension Types ---
# Note: TwoNNEstimate and LocalDimensionMap defined in intrinsic_dimension.py
# Note: IntrinsicDimensionResult for services in geometry_metrics_service.py


# --- Compositional Probes Types ---


class CompositionCategory(Enum):
    MENTAL_PREDICATE = "mentalPredicate"
    ACTION = "action"
    EVALUATIVE = "evaluative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIED = "quantified"
    RELATIONAL = "relational"


@dataclass(frozen=True)
class CompositionProbe:
    phrase: str
    components: tuple[str, ...]
    category: CompositionCategory


@dataclass(frozen=True)
class CompositionAnalysis:
    """Analysis of compositional structure for a probe.

    Compositionality is determined by geometric properties:
    - residual_norm: How much the composition misses the target (lower indicates smaller residual)
    - centroid_similarity: How well the barycenter approximates the phrase (higher indicates closer approximation)

    Both metrics are in [0, 1]. The midpoint (0.5) is the geometric boundary.
    """

    probe: CompositionProbe
    barycentric_weights: tuple[float, ...]
    residual_norm: float
    centroid_similarity: float
    component_angles: tuple[float, ...]

    @property
    def is_compositional(self) -> bool:
        """Compositional if residual exists and there's positive similarity.

        Uses boundary values (residual < 1.0, similarity > 0.0) instead of
        arbitrary midpoint thresholds. Callers should compare raw
        residual_norm and centroid_similarity values for nuanced decisions.
        """
        return self.residual_norm < 1.0 and self.centroid_similarity > 0.0


@dataclass(frozen=True)
class ConsistencyResult:
    """Cross-model compositional consistency.

    Attributes
    ----------
    probe_count : int
        Number of probes compared.
    analyses_a : tuple of CompositionAnalysis
        Analyses from model A.
    analyses_b : tuple of CompositionAnalysis
        Analyses from model B.
    barycentric_correlation : float
        Geodesic correlation of barycentric weights.
    angular_correlation : float
        Geodesic correlation of component angles.
    consistency_score : float
        Composite consistency score.
    """

    probe_count: int
    analyses_a: tuple[CompositionAnalysis, ...]
    analyses_b: tuple[CompositionAnalysis, ...]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float


# --- Generalized Procrustes Types ---


@dataclass(frozen=True)
class ProcrustesResult:
    consensus: tuple[tuple[float, ...], ...]
    rotations: tuple[tuple[tuple[float, ...], ...], ...]
    scales: tuple[float, ...]
    residuals: tuple[tuple[tuple[float, ...], ...], ...]
    converged: bool
    iterations: int
    alignment_error: float
    per_model_errors: tuple[float, ...]
    consensus_variance_ratio: float
    sample_count: int
    dimension: int
    model_count: int


# --- Pairwise Procrustes Alignment Types ---


@dataclass
class PairwiseProcrustesResult(Generic[Array]):
    """Result of pairwise Procrustes alignment.

    This is for aligning two matrices (source to target).
    For multi-model alignment, see ProcrustesResult above.

    Generic over Array type to support backend arrays.
    """

    rotation: Array  # Orthogonal rotation matrix [d, d]
    scale: float  # Optimal scale factor
    translation: Array  # Translation vector [d]
    residual: float  # Procrustes distance (sum of squared errors)
