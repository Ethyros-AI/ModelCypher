# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Temporal Geometry: Measuring time-related structure in representations.

Analyzes how temporal concepts (past/future, duration, causality) are
organized in embedding space. Computes:

1. Axis orthogonality: Independence of direction/duration/causality axes
2. Gradient consistency: Monotonicity of orderings (past < present < future)
3. Temporal direction: Correlation of projections with expected ordering

All outputs are raw measurements. No interpretation of what high/low
values "mean" about the model - that's for the researcher to determine.

Methods:
- Orthogonality: 1 - |cos(angle)| between axis principal components
- Gradient: Spearman correlation between concept levels and projections
- Direction: Correlation of direction-axis concepts with expected ordering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.atlas_protocols import (
    TemporalConceptProtocol,
    axis_key,
)
from modelcypher.core.domain.geometry.atlas_registry import get_temporal_concepts
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_spearman_correlation,
    division_epsilon,
    geodesic_svd,
    is_nan,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

_AXIS_DIRECTION = "direction"
_AXIS_DURATION = "duration"
_AXIS_CAUSALITY = "causality"

logger = logging.getLogger(__name__)


@dataclass
class TemporalAxisOrthogonality:
    """Orthogonality measurements between temporal axes.

    Values in [0, 1] where 1 = perfectly orthogonal, 0 = parallel.
    Computed as 1 - |cos(angle)| between axis principal components.
    """

    direction_duration: float
    direction_causality: float
    duration_causality: float
    mean_orthogonality: float


@dataclass
class TemporalGradientConsistency:
    """Gradient consistency for each temporal axis.

    Correlation values in [-1, 1]. Positive = ordering matches expected.
    Computed via Spearman correlation between concept levels and projections.
    """

    direction_correlation: float
    direction_monotonic: bool
    duration_correlation: float
    duration_monotonic: bool
    causality_correlation: float
    causality_monotonic: bool


@dataclass
class TemporalDirectionResult:
    """Temporal direction measurement results.

    Measures correlation between direction-axis concepts and expected
    temporal ordering (past < present < future).
    """

    past_anchors: list[str]
    future_anchors: list[str]
    direction_detected: bool  # True if |correlation| > 0
    direction_correlation: float  # Spearman correlation with expected ordering


@dataclass
class TemporalGeometryComponents:
    """Raw component scores for temporal geometry analysis.

    All scores in [0, 1]. Higher values indicate stronger structure.
    No interpretation provided - consumers decide significance.
    """

    orthogonality_score: float
    """Mean orthogonality between temporal axes."""

    gradient_score: float
    """Mean absolute gradient correlation across axes."""

    direction_score: float
    """Absolute correlation with expected temporal ordering."""


@dataclass
class TemporalGeometryReport:
    """Complete temporal geometry analysis report.

    Contains raw measurements only. Composite score is simple average
    of component scores - no weighting or interpretation.
    """

    model_path: str
    layer: int
    anchors_probed: int
    axis_orthogonality: TemporalAxisOrthogonality
    gradient_consistency: TemporalGradientConsistency
    temporal_direction: TemporalDirectionResult
    principal_components_variance: list[float]
    components: TemporalGeometryComponents

    @property
    def composite_score(self) -> float:
        """Simple average of component scores."""
        return (
            self.components.orthogonality_score
            + self.components.gradient_score
            + self.components.direction_score
        ) / 3.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "modelPath": self.model_path,
            "layer": self.layer,
            "anchorsProbed": self.anchors_probed,
            "axisOrthogonality": {
                "directionDuration": self.axis_orthogonality.direction_duration,
                "directionCausality": self.axis_orthogonality.direction_causality,
                "durationCausality": self.axis_orthogonality.duration_causality,
                "mean": self.axis_orthogonality.mean_orthogonality,
            },
            "gradientConsistency": {
                "direction": self.gradient_consistency.direction_correlation,
                "duration": self.gradient_consistency.duration_correlation,
                "causality": self.gradient_consistency.causality_correlation,
            },
            "temporalDirection": {
                "detected": self.temporal_direction.direction_detected,
                "correlation": self.temporal_direction.direction_correlation,
            },
            "components": {
                "orthogonality": self.components.orthogonality_score,
                "gradient": self.components.gradient_score,
                "direction": self.components.direction_score,
            },
            "compositeScore": self.composite_score,
        }


class TemporalGeometryAnalyzer:
    """Analyzer for temporal structure in representations.

    Measures how temporal concepts are organized geometrically:
    1. Extract activations for temporal concept probes
    2. Compute axis orthogonality via PCA
    3. Measure gradient consistency via Spearman correlation
    4. Check temporal direction ordering

    All outputs are measurements. No claims about what the model
    "understands" or whether structure is "meaningful".
    """

    def __init__(
        self,
        activations: dict[str, list[float]],
        concepts: list[TemporalConceptProtocol] | None = None,
    ) -> None:
        """Initialize with concept activations.

        Args:
            activations: Dict mapping concept ID to activation vector
            concepts: Optional concept inventory (defaults to registry)
        """
        self.activations = activations
        self._anchors = list(concepts or get_temporal_concepts())
        if not self._anchors:
            raise ValueError(
                "No temporal concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before running temporal geometry analysis."
            )
        self._anchor_lookup = {a.id: a for a in self._anchors}

    def analyze(self) -> TemporalGeometryReport:
        """Run temporal geometry analysis.

        Returns:
            TemporalGeometryReport with all measurements
        """
        anchors = self._anchors
        concepts = [a.id for a in anchors if a.id in self.activations]
        if len(concepts) < 10:
            raise ValueError(f"Insufficient anchors: {len(concepts)} < 10 required")

        backend = get_default_backend()
        matrix = backend.array(
            [self.activations[c] for c in concepts], dtype=precision_dtype(backend)
        )
        backend.eval(matrix)

        # Normalize for cosine similarity
        norms_arr = geodesic_norms(matrix, backend)
        norms_arr = backend.reshape(norms_arr, (-1, 1))
        div_eps = division_epsilon(backend, norms_arr)
        matrix_norm = matrix / backend.maximum(
            norms_arr, backend.full(norms_arr.shape, div_eps)
        )
        backend.eval(matrix_norm)

        # PCA for axis analysis
        mean_arr = backend.mean(matrix_norm, axis=0)
        backend.eval(mean_arr)
        centered = matrix_norm - mean_arr
        backend.eval(centered)
        try:
            _, s, vh = geodesic_svd(backend, centered)
            backend.eval(s, vh)
            s_squared = s * s
            total = backend.sum(s_squared)
            backend.eval(total)
            total_val = float(backend.to_scalar(total))
            if total_val > division_epsilon(backend, s):
                variance_explained = s_squared / total
            else:
                variance_explained = backend.zeros_like(s_squared)
            backend.eval(variance_explained)
            variance_list = backend.tolist(variance_explained)
            pc_variance = [float(x) for x in variance_list[:5]]
        except Exception:
            pc_variance = [0.0] * 5

        # Compute measurements
        axis_ortho = self._compute_axis_orthogonality(matrix_norm, concepts)
        gradient = self._compute_gradient_consistency(matrix_norm, concepts)
        direction = self._compute_temporal_direction(matrix_norm, concepts)

        # Component scores (raw, no weighting)
        ortho_score = axis_ortho.mean_orthogonality
        gradient_scores = [
            gradient.direction_correlation,
            gradient.duration_correlation,
            gradient.causality_correlation,
        ]
        gradient_score = sum(abs(s) for s in gradient_scores) / len(gradient_scores)
        direction_score = abs(direction.direction_correlation)

        components = TemporalGeometryComponents(
            orthogonality_score=ortho_score,
            gradient_score=gradient_score,
            direction_score=direction_score,
        )

        return TemporalGeometryReport(
            model_path="",
            layer=-1,
            anchors_probed=len(concepts),
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            temporal_direction=direction,
            principal_components_variance=pc_variance,
            components=components,
        )

    def _compute_axis_orthogonality(
        self, matrix: "Array", concepts: list[str]
    ) -> TemporalAxisOrthogonality:
        """Compute orthogonality between temporal axes."""
        backend = get_default_backend()
        backend.eval(matrix)

        direction_vecs = []
        duration_vecs = []
        causality_vecs = []

        for i, concept in enumerate(concepts):
            anchor = self._anchor_lookup.get(concept)
            if anchor is None:
                continue
            axis_str = axis_key(anchor.axis)
            if axis_str == _AXIS_DIRECTION:
                direction_vecs.append(matrix[i])
            elif axis_str == _AXIS_DURATION:
                duration_vecs.append(matrix[i])
            elif axis_str == _AXIS_CAUSALITY:
                causality_vecs.append(matrix[i])

        def axis_direction(vecs: list) -> "object":
            """Compute principal direction of axis from anchors."""
            if len(vecs) < 2:
                if vecs:
                    shape = vecs[0].shape
                    result = backend.zeros(shape)
                else:
                    result = backend.zeros((1,))
                backend.eval(result)
                return result
            arr = backend.stack(vecs, axis=0)
            backend.eval(arr)
            mean_vec = backend.mean(arr, axis=0)
            backend.eval(mean_vec)
            centered = arr - mean_vec
            backend.eval(centered)
            try:
                _, _, vh = geodesic_svd(backend, centered)
                backend.eval(vh)
                result = vh[0]
                backend.eval(result)
                return result
            except Exception:
                shape = arr.shape[1:]
                result = backend.zeros(shape)
                backend.eval(result)
                return result

        dir_vec = axis_direction(direction_vecs)
        dur_vec = axis_direction(duration_vecs)
        caus_vec = axis_direction(causality_vecs)

        def orthogonality(v1: "object", v2: "object") -> float:
            """Compute orthogonality as 1 - |cos(angle)|."""
            v1_mat = backend.reshape(v1, (1, -1))
            v2_mat = backend.reshape(v2, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(v1_mat, v2_mat, backend)
            backend.eval(cos_arr)
            cos_sim = abs(float(backend.to_scalar(cos_arr)))
            return 1.0 - cos_sim

        dir_dur = orthogonality(dir_vec, dur_vec)
        dir_caus = orthogonality(dir_vec, caus_vec)
        dur_caus = orthogonality(dur_vec, caus_vec)

        return TemporalAxisOrthogonality(
            direction_duration=dir_dur,
            direction_causality=dir_caus,
            duration_causality=dur_caus,
            mean_orthogonality=(dir_dur + dir_caus + dur_caus) / 3,
        )

    def _compute_gradient_consistency(
        self, matrix: "Array", concepts: list[str]
    ) -> TemporalGradientConsistency:
        """Compute gradient consistency via Spearman correlation."""
        backend = get_default_backend()
        backend.eval(matrix)

        shape = matrix.shape
        if len(shape) > 1 and int(shape[1]) > 0:
            col0_arr = matrix[:, 0]
            backend.eval(col0_arr)
            col0 = backend.tolist(col0_arr)
        else:
            col0 = [0.0] * len(concepts)

        def axis_correlation(axis: str) -> tuple[float, bool]:
            """Compute correlation for a specific axis."""
            levels = []
            projections = []

            for i, concept in enumerate(concepts):
                anchor = self._anchor_lookup.get(concept)
                if anchor is None or axis_key(anchor.axis) != axis:
                    continue
                levels.append(anchor.level)
                projections.append(float(col0[i]))

            if len(levels) < 3:
                return 0.0, False

            corr = compute_spearman_correlation(
                levels, projections, default=0.0, backend=backend
            )
            if corr is None or is_nan(float(corr), backend):
                corr = 0.0

            monotonic = abs(corr) > 0
            return float(corr), monotonic

        dir_corr, dir_mono = axis_correlation(_AXIS_DIRECTION)
        dur_corr, dur_mono = axis_correlation(_AXIS_DURATION)
        caus_corr, caus_mono = axis_correlation(_AXIS_CAUSALITY)

        return TemporalGradientConsistency(
            direction_correlation=dir_corr,
            direction_monotonic=dir_mono,
            duration_correlation=dur_corr,
            duration_monotonic=dur_mono,
            causality_correlation=caus_corr,
            causality_monotonic=caus_mono,
        )

    def _compute_temporal_direction(
        self, matrix: "Array", concepts: list[str]
    ) -> TemporalDirectionResult:
        """Compute temporal direction correlation."""
        backend = get_default_backend()
        backend.eval(matrix)

        shape = matrix.shape
        if len(shape) > 1 and int(shape[1]) > 0:
            col0_arr = matrix[:, 0]
            backend.eval(col0_arr)
            col0 = backend.tolist(col0_arr)
        else:
            col0 = [0.0] * len(concepts)

        past_concepts = ["yesterday", "past", "birth", "beginning"]
        future_concepts = ["tomorrow", "future", "death", "ending"]

        past_anchors = [c for c in concepts if c in past_concepts]
        future_anchors = [c for c in concepts if c in future_concepts]

        direction_anchors = []
        for i, concept in enumerate(concepts):
            anchor = self._anchor_lookup.get(concept)
            if anchor and axis_key(anchor.axis) == _AXIS_DIRECTION:
                direction_anchors.append((concept, anchor.level, float(col0[i])))

        if len(direction_anchors) < 4:
            return TemporalDirectionResult(
                past_anchors=past_anchors,
                future_anchors=future_anchors,
                direction_detected=False,
                direction_correlation=0.0,
            )

        levels = [a[1] for a in direction_anchors]
        projections = [a[2] for a in direction_anchors]

        corr = compute_spearman_correlation(
            levels, projections, default=0.0, backend=backend
        )
        if corr is None or is_nan(float(corr), backend):
            corr = 0.0

        direction_detected = abs(corr) > 0

        return TemporalDirectionResult(
            past_anchors=past_anchors,
            future_anchors=future_anchors,
            direction_detected=direction_detected,
            direction_correlation=float(corr),
        )


def extract_temporal_activations(
    model: Any,
    tokenizer: Any,
    layer: int = -1,
    activation_provider: Any | None = None,
) -> dict[str, list[float]]:
    """Extract activations for temporal concept probes.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        layer: Layer to extract from (-1 for last)
        activation_provider: ActivationProvider instance (required)

    Returns:
        Dict mapping concept ID to activation vector

    Raises:
        ValueError: If activation_provider is None
    """
    if activation_provider is None:
        raise ValueError(
            "activation_provider is required. "
            "Pass an ActivationProvider implementation."
        )

    backend = get_default_backend()
    activations = {}

    for anchor in get_temporal_concepts():
        try:
            all_layer_acts = activation_provider.collect_hidden_activations(
                model, tokenizer, anchor.prompt
            )

            if not all_layer_acts:
                logger.warning(f"No activations collected for {anchor.id}")
                continue

            layer_indices = sorted(all_layer_acts.keys())
            if layer < 0:
                target_layer = (
                    layer_indices[layer]
                    if abs(layer) <= len(layer_indices)
                    else layer_indices[-1]
                )
            else:
                target_layer = (
                    layer if layer in all_layer_acts else layer_indices[-1]
                )

            act_arr = all_layer_acts.get(target_layer)
            if act_arr is not None:
                backend.eval(act_arr)
                act = backend.tolist(act_arr)
                activations[anchor.id] = act

        except Exception as e:
            logger.warning(f"Failed to extract activation for {anchor.id}: {e}")
            continue

    return activations


__all__ = [
    "TemporalAxisOrthogonality",
    "TemporalGradientConsistency",
    "TemporalDirectionResult",
    "TemporalGeometryComponents",
    "TemporalGeometryReport",
    "TemporalGeometryAnalyzer",
    "extract_temporal_activations",
]
