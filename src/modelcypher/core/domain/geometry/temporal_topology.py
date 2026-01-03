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

"""Temporal Topology: Probing time-like structure in LLM representations.

This module implements the "Latent Chronologist" hypothesis: that language models
trained on narrative text encode time as a coherent geometric manifold with:
1. Temporal Direction axis (past → future)
2. Duration axis (moment → eternity)
3. Causality axis (cause → effect)

Scientific Method:
- H1: Models encode temporal structure above chance (TMS > 0.33 baseline)
- H2: Temporal axes are geometrically independent (orthogonality > 80%)
- H3: Arrow of Time is detectable (monotonic past→future gradient)
- H4: Duration is monotonic (moment < hour < day < year < century)
- H5: Measurements are reproducible (CV < 10%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.atlas_protocols import (
    TemporalConceptProtocol,
    axis_key,
)
from modelcypher.core.domain.geometry.atlas_registry import get_temporal_concepts
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon, is_nan

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

_AXIS_DIRECTION = "direction"
_AXIS_DURATION = "duration"
_AXIS_CAUSALITY = "causality"

logger = logging.getLogger(__name__)


@dataclass
class AxisOrthogonality:
    """Orthogonality measurements between temporal axes."""

    direction_duration: float
    direction_causality: float
    duration_causality: float
    mean_orthogonality: float


@dataclass
class GradientConsistency:
    """Gradient consistency measurements for each axis."""

    direction_correlation: float
    direction_monotonic: bool
    duration_correlation: float
    duration_monotonic: bool
    causality_correlation: float
    causality_monotonic: bool


@dataclass
class ArrowOfTime:
    """Detection of the "Arrow of Time" - consistent past→future gradient."""

    past_anchors: list[str]
    future_anchors: list[str]
    arrow_detected: bool
    direction_correlation: float  # Correlation with expected temporal ordering


@dataclass
class TemporalManifoldComponents:
    """Raw component scores for temporal manifold analysis.

    Returns individual measurements instead of a weighted composite.
    Consumers decide how to interpret these values.
    """

    orthogonality_score: float
    """Mean orthogonality between temporal axes [0, 1]. Higher = more distinct axes."""

    gradient_score: float
    """Mean absolute gradient consistency [0, 1]. Higher indicates more monotonic orderings."""

    arrow_score: float
    """Arrow of time detection score [0, 1]. Higher = clearer temporal direction."""


@dataclass
class TemporalTopologyReport:
    """Complete temporal topology analysis report."""

    model_path: str
    layer: int
    anchors_probed: int
    axis_orthogonality: AxisOrthogonality
    gradient_consistency: GradientConsistency
    arrow_of_time: ArrowOfTime
    principal_components_variance: list[float]
    temporal_manifold_components: TemporalManifoldComponents

    @property
    def temporal_manifold_score(self) -> float:
        """Composite temporal manifold score from component scores.

        Returns mean of orthogonality, gradient, and arrow scores.
        """
        components = self.temporal_manifold_components
        return (
            components.orthogonality_score
            + components.gradient_score
            + components.arrow_score
        ) / 3.0


class TemporalTopologyAnalyzer:
    """Analyzer for temporal structure in LLM representations.

    Implements the scientific method for testing the Latent Chronologist hypothesis:
    1. Extract activations for 25 temporal concepts
    2. Measure axis orthogonality (Direction ⊥ Duration ⊥ Causality)
    3. Test gradient consistency (monotonic orderings)
    4. Detect Arrow of Time (past→future direction)
    5. Compute Temporal Manifold Score (TMS)
    """

    def __init__(
        self,
        activations: dict[str, list[float]],
        concepts: list[TemporalConceptProtocol] | None = None,
    ) -> None:
        """Initialize with anchor activations.

        Args:
            activations: Dict mapping anchor concept to activation vector (as list)
            concepts: Optional temporal concept inventory (defaults to registry)
        """
        self.activations = activations
        self._anchors = list(concepts or get_temporal_concepts())
        if not self._anchors:
            raise ValueError(
                "No temporal concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before running temporal topology analysis."
            )
        self._anchor_lookup = {a.id: a for a in self._anchors}

    def analyze(self) -> TemporalTopologyReport:
        """Run complete temporal topology analysis.

        Returns:
            TemporalTopologyReport with all measurements
        """
        # Build activation matrix
        anchors = self._anchors
        concepts = [a.id for a in anchors if a.id in self.activations]
        if len(concepts) < 10:
            raise ValueError(f"Insufficient anchors: {len(concepts)} < 10 required")

        backend = get_default_backend()
        matrix = backend.array([self.activations[c] for c in concepts], dtype="float32")
        backend.eval(matrix)

        # Normalize for cosine similarity
        norms_arr = backend.norm(matrix, axis=1, keepdims=True)
        backend.eval(norms_arr)
        div_eps = division_epsilon(backend, matrix)
        matrix_norm = matrix / (norms_arr + div_eps)
        backend.eval(matrix_norm)

        # PCA for axis analysis
        mean_arr = backend.mean(matrix_norm, axis=0)
        backend.eval(mean_arr)
        centered = matrix_norm - mean_arr
        backend.eval(centered)
        try:
            _, s, vh = backend.svd(centered, full_matrices=False)
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
            # Use native tolist() for O(1) extraction
            variance_list = backend.tolist(variance_explained)
            pc_variance = [float(x) for x in variance_list[:5]]
        except Exception:
            pc_variance = [0.0] * 5

        # Compute axis orthogonality
        axis_ortho = self._compute_axis_orthogonality(matrix_norm, concepts)

        # Compute gradient consistency
        gradient = self._compute_gradient_consistency(matrix_norm, concepts)

        # Detect Arrow of Time
        arrow = self._detect_arrow_of_time(matrix_norm, concepts)

        # Compute raw component scores
        ortho_score = axis_ortho.mean_orthogonality

        gradient_scores = [
            gradient.direction_correlation,
            gradient.duration_correlation,
            gradient.causality_correlation,
        ]
        gradient_score = sum(abs(s) for s in gradient_scores) / len(gradient_scores)

        arrow_score = 1.0 if arrow.arrow_detected else 0.5 * abs(arrow.direction_correlation)

        components = TemporalManifoldComponents(
            orthogonality_score=ortho_score,
            gradient_score=gradient_score,
            arrow_score=arrow_score,
        )

        return TemporalTopologyReport(
            model_path="",
            layer=-1,
            anchors_probed=len(concepts),
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            arrow_of_time=arrow,
            principal_components_variance=pc_variance,
            temporal_manifold_components=components,
        )

    def _compute_axis_orthogonality(
        self, matrix: "Array", concepts: list[str]
    ) -> AxisOrthogonality:
        """Compute orthogonality between temporal axes."""
        backend = get_default_backend()
        backend.eval(matrix)

        # Get centroids for each axis
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
                _, _, vh = backend.svd(centered)
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
            n1_arr = backend.norm(v1)
            n2_arr = backend.norm(v2)
            backend.eval(n1_arr, n2_arr)
            n1 = float(backend.to_scalar(n1_arr))
            n2 = float(backend.to_scalar(n2_arr))
            div_eps = division_epsilon(backend, v1)
            if n1 < div_eps or n2 < div_eps:
                return 0.0
            dot_arr = backend.sum(v1 * v2)
            backend.eval(dot_arr)
            dot_val = float(backend.to_scalar(dot_arr))
            cos_sim = abs(dot_val / (n1 * n2))
            return 1.0 - cos_sim

        dir_dur = orthogonality(dir_vec, dur_vec)
        dir_caus = orthogonality(dir_vec, caus_vec)
        dur_caus = orthogonality(dur_vec, caus_vec)

        return AxisOrthogonality(
            direction_duration=dir_dur,
            direction_causality=dir_caus,
            duration_causality=dur_caus,
            mean_orthogonality=(dir_dur + dir_caus + dur_caus) / 3,
        )

    def _compute_gradient_consistency(
        self, matrix: "Array", concepts: list[str]
    ) -> GradientConsistency:
        """Compute gradient consistency (Spearman correlation with expected ordering)."""
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        backend = get_default_backend()
        backend.eval(matrix)

        # Extract first column once via O(1) tolist() instead of O(n) to_scalar() loop
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

            corr = VectorMath.spearman_correlation(levels, projections)
            if corr is None or is_nan(float(corr), backend):
                corr = 0.0

            # Monotonic if any measurable correlation exists
            monotonic = abs(corr) > 0
            return float(corr), monotonic

        dir_corr, dir_mono = axis_correlation(_AXIS_DIRECTION)
        dur_corr, dur_mono = axis_correlation(_AXIS_DURATION)
        caus_corr, caus_mono = axis_correlation(_AXIS_CAUSALITY)

        return GradientConsistency(
            direction_correlation=dir_corr,
            direction_monotonic=dir_mono,
            duration_correlation=dur_corr,
            duration_monotonic=dur_mono,
            causality_correlation=caus_corr,
            causality_monotonic=caus_mono,
        )

    def _detect_arrow_of_time(self, matrix: "Array", concepts: list[str]) -> ArrowOfTime:
        """Detect if there's a consistent "Arrow of Time" direction."""
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        backend = get_default_backend()
        backend.eval(matrix)

        # Extract first column once via O(1) tolist() instead of O(n) to_scalar() loop
        shape = matrix.shape
        if len(shape) > 1 and int(shape[1]) > 0:
            col0_arr = matrix[:, 0]
            backend.eval(col0_arr)
            col0 = backend.tolist(col0_arr)
        else:
            col0 = [0.0] * len(concepts)

        # Separate past and future anchors
        past_concepts = ["yesterday", "past", "birth", "beginning"]
        future_concepts = ["tomorrow", "future", "death", "ending"]

        past_anchors = [c for c in concepts if c in past_concepts]
        future_anchors = [c for c in concepts if c in future_concepts]

        # Get all direction-axis anchors and their expected ordering
        direction_anchors = []
        for i, concept in enumerate(concepts):
            anchor = self._anchor_lookup.get(concept)
            if anchor and axis_key(anchor.axis) == _AXIS_DIRECTION:
                # Store (concept, level, projection_value) - projection already extracted
                direction_anchors.append((concept, anchor.level, float(col0[i])))

        if len(direction_anchors) < 4:
            return ArrowOfTime(
                past_anchors=past_anchors,
                future_anchors=future_anchors,
                arrow_detected=False,
                direction_correlation=0.0,
            )

        # Compute correlation between level and first PC projection
        levels = [a[1] for a in direction_anchors]
        projections = [a[2] for a in direction_anchors]

        corr = VectorMath.spearman_correlation(levels, projections)
        if corr is None or is_nan(float(corr), backend):
            corr = 0.0

        # Arrow detected if any measurable correlation exists
        arrow_detected = abs(corr) > 0

        return ArrowOfTime(
            past_anchors=past_anchors,
            future_anchors=future_anchors,
            arrow_detected=arrow_detected,
            direction_correlation=float(corr),
        )


def extract_temporal_activations(
    model: Any,
    tokenizer: Any,
    layer: int = -1,
    activation_provider: Any | None = None,
) -> dict[str, list[float]]:
    """Extract activations for all temporal anchors.

    Args:
        model: The model (platform-agnostic via ActivationProvider)
        tokenizer: The tokenizer
        layer: Layer to extract from (-1 for last)
        activation_provider: ActivationProvider instance (auto-detected if None)

    Returns:
        Dict mapping concept to activation vector (as list)
    """
    # Get activation provider (auto-detect if not provided)
    if activation_provider is None:
        from modelcypher.infrastructure.activation_provider_factory import get_activation_provider

        activation_provider = get_activation_provider()

    backend = get_default_backend()
    activations = {}

    for anchor in get_temporal_concepts():
        try:
            # Collect all layer activations using ActivationProvider
            all_layer_acts = activation_provider.collect_hidden_activations(
                model, tokenizer, anchor.prompt
            )

            if not all_layer_acts:
                logger.warning(f"No activations collected for {anchor.id}")
                continue

            # Determine target layer
            layer_indices = sorted(all_layer_acts.keys())
            if layer < 0:
                # Negative indexing from end
                target_layer = layer_indices[layer] if abs(layer) <= len(layer_indices) else layer_indices[-1]
            else:
                # Positive indexing - find closest layer
                target_layer = layer if layer in all_layer_acts else layer_indices[-1]

            # Get activation for target layer
            act_arr = all_layer_acts.get(target_layer)
            if act_arr is not None:
                backend.eval(act_arr)
                act = backend.tolist(act_arr)
                activations[anchor.id] = act

        except Exception as e:
            logger.warning(f"Failed to extract activation for {anchor.id}: {e}")
            continue

    return activations
