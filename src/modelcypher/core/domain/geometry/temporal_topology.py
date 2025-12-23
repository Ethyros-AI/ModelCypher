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
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

logger = logging.getLogger(__name__)


class TemporalCategory(str, Enum):
    """Categories of temporal anchors."""
    TENSE = "tense"           # yesterday, today, tomorrow
    DURATION = "duration"     # moment, hour, day, year, century
    CAUSALITY = "causality"   # because, therefore, causes
    LIFECYCLE = "lifecycle"   # birth, childhood, adulthood, death
    SEQUENCE = "sequence"     # first, then, next, finally, last


class TemporalAxis(str, Enum):
    """Axes of the temporal manifold."""
    DIRECTION = "direction"   # Past (-1) to Future (+1)
    DURATION = "duration"     # Short (-1) to Long (+1)
    CAUSALITY = "causality"   # Cause (-1) to Effect (+1)


@dataclass(frozen=True)
class TemporalAnchor:
    """A probe anchor for temporal structure."""
    concept: str
    prompt: str
    category: TemporalCategory
    axis: TemporalAxis
    level: int  # 1-5 ordering within axis (1=lowest, 5=highest)


# The Temporal Prime Atlas: 23 anchors across 5 categories
TEMPORAL_PRIME_ATLAS: list[TemporalAnchor] = [
    # Tense (Past → Future) - Direction axis
    TemporalAnchor("yesterday", "The word yesterday represents", TemporalCategory.TENSE, TemporalAxis.DIRECTION, 1),
    TemporalAnchor("today", "The word today represents", TemporalCategory.TENSE, TemporalAxis.DIRECTION, 3),
    TemporalAnchor("tomorrow", "The word tomorrow represents", TemporalCategory.TENSE, TemporalAxis.DIRECTION, 5),
    TemporalAnchor("past", "The word past represents", TemporalCategory.TENSE, TemporalAxis.DIRECTION, 1),
    TemporalAnchor("future", "The word future represents", TemporalCategory.TENSE, TemporalAxis.DIRECTION, 5),

    # Duration (Short → Long) - Duration axis
    TemporalAnchor("moment", "The word moment represents", TemporalCategory.DURATION, TemporalAxis.DURATION, 1),
    TemporalAnchor("hour", "The word hour represents", TemporalCategory.DURATION, TemporalAxis.DURATION, 2),
    TemporalAnchor("day", "The word day represents", TemporalCategory.DURATION, TemporalAxis.DURATION, 3),
    TemporalAnchor("year", "The word year represents", TemporalCategory.DURATION, TemporalAxis.DURATION, 4),
    TemporalAnchor("century", "The word century represents", TemporalCategory.DURATION, TemporalAxis.DURATION, 5),

    # Causality (Cause → Effect) - Causality axis
    TemporalAnchor("because", "The word because represents", TemporalCategory.CAUSALITY, TemporalAxis.CAUSALITY, 1),
    TemporalAnchor("causes", "The word causes represents", TemporalCategory.CAUSALITY, TemporalAxis.CAUSALITY, 2),
    TemporalAnchor("leads", "The word leads represents", TemporalCategory.CAUSALITY, TemporalAxis.CAUSALITY, 3),
    TemporalAnchor("therefore", "The word therefore represents", TemporalCategory.CAUSALITY, TemporalAxis.CAUSALITY, 4),
    TemporalAnchor("results", "The word results represents", TemporalCategory.CAUSALITY, TemporalAxis.CAUSALITY, 5),

    # Lifecycle (Beginning → End) - Direction axis
    TemporalAnchor("birth", "The word birth represents", TemporalCategory.LIFECYCLE, TemporalAxis.DIRECTION, 1),
    TemporalAnchor("childhood", "The word childhood represents", TemporalCategory.LIFECYCLE, TemporalAxis.DIRECTION, 2),
    TemporalAnchor("adulthood", "The word adulthood represents", TemporalCategory.LIFECYCLE, TemporalAxis.DIRECTION, 3),
    TemporalAnchor("elderly", "The word elderly represents", TemporalCategory.LIFECYCLE, TemporalAxis.DIRECTION, 4),
    TemporalAnchor("death", "The word death represents", TemporalCategory.LIFECYCLE, TemporalAxis.DIRECTION, 5),

    # Sequence (First → Last) - Direction axis
    TemporalAnchor("beginning", "The word beginning represents", TemporalCategory.SEQUENCE, TemporalAxis.DIRECTION, 1),
    TemporalAnchor("middle", "The word middle represents", TemporalCategory.SEQUENCE, TemporalAxis.DIRECTION, 3),
    TemporalAnchor("ending", "The word ending represents", TemporalCategory.SEQUENCE, TemporalAxis.DIRECTION, 5),
]


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
class TemporalTopologyReport:
    """Complete temporal topology analysis report."""
    model_path: str
    layer: int
    anchors_probed: int
    axis_orthogonality: AxisOrthogonality
    gradient_consistency: GradientConsistency
    arrow_of_time: ArrowOfTime
    principal_components_variance: list[float]
    temporal_manifold_score: float
    has_temporal_manifold: bool
    verdict: str


class TemporalTopologyAnalyzer:
    """Analyzer for temporal structure in LLM representations.

    Implements the scientific method for testing the Latent Chronologist hypothesis:
    1. Extract activations for 23 temporal anchors
    2. Measure axis orthogonality (Direction ⊥ Duration ⊥ Causality)
    3. Test gradient consistency (monotonic orderings)
    4. Detect Arrow of Time (past→future direction)
    5. Compute Temporal Manifold Score (TMS)
    """

    def __init__(self, activations: dict[str, np.ndarray]) -> None:
        """Initialize with anchor activations.

        Args:
            activations: Dict mapping anchor concept to activation vector
        """
        self.activations = activations
        self._anchor_lookup = {a.concept: a for a in TEMPORAL_PRIME_ATLAS}

    def analyze(self) -> TemporalTopologyReport:
        """Run complete temporal topology analysis.

        Returns:
            TemporalTopologyReport with all measurements
        """
        # Build activation matrix
        concepts = [a.concept for a in TEMPORAL_PRIME_ATLAS if a.concept in self.activations]
        if len(concepts) < 10:
            raise ValueError(f"Insufficient anchors: {len(concepts)} < 10 required")

        # Cast to float32 for numpy linalg compatibility (float16 not supported)
        matrix = np.array([self.activations[c] for c in concepts], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)

        # PCA for axis analysis
        centered = matrix_norm - matrix_norm.mean(axis=0)
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            variance_explained = (s ** 2) / (s ** 2).sum()
            pc_variance = variance_explained[:5].tolist()
        except np.linalg.LinAlgError:
            pc_variance = [0.0] * 5

        # Compute axis orthogonality
        axis_ortho = self._compute_axis_orthogonality(matrix_norm, concepts)

        # Compute gradient consistency
        gradient = self._compute_gradient_consistency(matrix_norm, concepts)

        # Detect Arrow of Time
        arrow = self._detect_arrow_of_time(matrix_norm, concepts)

        # Compute Temporal Manifold Score (TMS)
        # Weighted: 30% orthogonality + 40% gradient + 30% arrow detection
        ortho_score = axis_ortho.mean_orthogonality

        gradient_scores = [
            gradient.direction_correlation,
            gradient.duration_correlation,
            gradient.causality_correlation,
        ]
        gradient_score = np.mean([abs(s) for s in gradient_scores])

        arrow_score = 1.0 if arrow.arrow_detected else 0.5 * abs(arrow.direction_correlation)

        tms = 0.30 * ortho_score + 0.40 * gradient_score + 0.30 * arrow_score

        # Determine verdict
        has_manifold = tms > 0.40
        if tms > 0.55:
            verdict = "STRONG TEMPORAL MANIFOLD - Clear direction/duration/causality axes detected."
        elif tms > 0.40:
            verdict = "MODERATE TEMPORAL MANIFOLD - Some temporal structure detected."
        else:
            verdict = "WEAK TEMPORAL MANIFOLD - Limited temporal geometry found."

        return TemporalTopologyReport(
            model_path="",
            layer=-1,
            anchors_probed=len(concepts),
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            arrow_of_time=arrow,
            principal_components_variance=pc_variance,
            temporal_manifold_score=tms,
            has_temporal_manifold=has_manifold,
            verdict=verdict,
        )

    def _compute_axis_orthogonality(
        self, matrix: np.ndarray, concepts: list[str]
    ) -> AxisOrthogonality:
        """Compute orthogonality between temporal axes."""
        # Get centroids for each axis
        direction_vecs = []
        duration_vecs = []
        causality_vecs = []

        for i, concept in enumerate(concepts):
            anchor = self._anchor_lookup.get(concept)
            if anchor is None:
                continue
            if anchor.axis == TemporalAxis.DIRECTION:
                direction_vecs.append(matrix[i])
            elif anchor.axis == TemporalAxis.DURATION:
                duration_vecs.append(matrix[i])
            elif anchor.axis == TemporalAxis.CAUSALITY:
                causality_vecs.append(matrix[i])

        def axis_direction(vecs: list[np.ndarray]) -> np.ndarray:
            """Compute principal direction of axis from anchors."""
            if len(vecs) < 2:
                return np.zeros(vecs[0].shape if vecs else 1)
            arr = np.array(vecs)
            centered = arr - arr.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                return vh[0]
            except np.linalg.LinAlgError:
                return np.zeros(arr.shape[1])

        dir_vec = axis_direction(direction_vecs)
        dur_vec = axis_direction(duration_vecs)
        caus_vec = axis_direction(causality_vecs)

        def orthogonality(v1: np.ndarray, v2: np.ndarray) -> float:
            """Compute orthogonality as 1 - |cos(angle)|."""
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0
            cos_sim = abs(np.dot(v1, v2) / (n1 * n2))
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
        self, matrix: np.ndarray, concepts: list[str]
    ) -> GradientConsistency:
        """Compute gradient consistency (Spearman correlation with expected ordering)."""
        from scipy import stats

        def axis_correlation(axis: TemporalAxis) -> tuple[float, bool]:
            """Compute correlation for a specific axis."""
            levels = []
            projections = []

            for i, concept in enumerate(concepts):
                anchor = self._anchor_lookup.get(concept)
                if anchor is None or anchor.axis != axis:
                    continue
                levels.append(anchor.level)
                # Project onto first PC direction
                projections.append(matrix[i, 0] if matrix.shape[1] > 0 else 0.0)

            if len(levels) < 3:
                return 0.0, False

            corr, _ = stats.spearmanr(levels, projections)
            if np.isnan(corr):
                corr = 0.0

            # Monotonic if |correlation| > 0.8
            monotonic = abs(corr) > 0.8
            return float(corr), monotonic

        dir_corr, dir_mono = axis_correlation(TemporalAxis.DIRECTION)
        dur_corr, dur_mono = axis_correlation(TemporalAxis.DURATION)
        caus_corr, caus_mono = axis_correlation(TemporalAxis.CAUSALITY)

        return GradientConsistency(
            direction_correlation=dir_corr,
            direction_monotonic=dir_mono,
            duration_correlation=dur_corr,
            duration_monotonic=dur_mono,
            causality_correlation=caus_corr,
            causality_monotonic=caus_mono,
        )

    def _detect_arrow_of_time(
        self, matrix: np.ndarray, concepts: list[str]
    ) -> ArrowOfTime:
        """Detect if there's a consistent "Arrow of Time" direction."""
        from scipy import stats

        # Separate past and future anchors
        past_concepts = ["yesterday", "past", "birth", "beginning"]
        future_concepts = ["tomorrow", "future", "death", "ending"]

        past_anchors = [c for c in concepts if c in past_concepts]
        future_anchors = [c for c in concepts if c in future_concepts]

        # Get all direction-axis anchors and their expected ordering
        direction_anchors = []
        for i, concept in enumerate(concepts):
            anchor = self._anchor_lookup.get(concept)
            if anchor and anchor.axis == TemporalAxis.DIRECTION:
                direction_anchors.append((concept, anchor.level, matrix[i]))

        if len(direction_anchors) < 4:
            return ArrowOfTime(
                past_anchors=past_anchors,
                future_anchors=future_anchors,
                arrow_detected=False,
                direction_correlation=0.0,
            )

        # Compute correlation between level and first PC projection
        levels = [a[1] for a in direction_anchors]
        projections = [a[2][0] for a in direction_anchors]

        corr, _ = stats.spearmanr(levels, projections)
        if np.isnan(corr):
            corr = 0.0

        # Arrow detected if |correlation| > 0.7
        arrow_detected = abs(corr) > 0.7

        return ArrowOfTime(
            past_anchors=past_anchors,
            future_anchors=future_anchors,
            arrow_detected=arrow_detected,
            direction_correlation=float(corr),
        )


def extract_temporal_activations(
    model: "mx.Module",
    tokenizer: "object",
    layer: int = -1,
) -> dict[str, np.ndarray]:
    """Extract activations for all temporal anchors.

    Args:
        model: The MLX model
        tokenizer: The tokenizer
        layer: Layer to extract from (-1 for last)

    Returns:
        Dict mapping concept to activation vector
    """
    import mlx.core as mx

    activations = {}

    for anchor in TEMPORAL_PRIME_ATLAS:
        # Tokenize prompt
        tokens = tokenizer.encode(anchor.prompt)
        input_ids = mx.array([tokens])

        # Forward pass with cache to get hidden states
        try:
            if hasattr(model, "model"):
                # Qwen-style architecture
                hidden = model.model.embed_tokens(input_ids)
                for i, layer_module in enumerate(model.model.layers):
                    hidden = layer_module(hidden, mask=None, cache=None)[0]
                    if i == layer or (layer == -1 and i == len(model.model.layers) - 1):
                        break
            else:
                # Try direct call
                outputs = model(input_ids)
                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs

            mx.eval(hidden)

            # Get last token's activation
            act = hidden[0, -1, :].tolist()
            activations[anchor.concept] = np.array(act)

        except Exception as e:
            logger.warning(f"Failed to extract activation for {anchor.concept}: {e}")
            continue

    return activations
