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

"""Merge Transformation Polytope.

Defines convex region from four diagnostic dimensions:
volume/overlap, mass/importance, stability, complexity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Types of transformations that may be needed for merging."""

    REDUCE_ALPHA = "reduce_alpha"  # Lower blend coefficient
    NULL_SPACE_FILTER = "null_space_filter"  # Project to null space
    SPECTRAL_CLAMP = "spectral_clamp"  # Regularize ill-conditioned
    LAYER_SKIP = "layer_skip"  # Skip this layer entirely
    TSV_PRUNE = "tsv_prune"  # Keep only top singular vectors
    CURVATURE_CORRECT = "curvature_correct"  # Apply Riemannian correction


@dataclass(frozen=True)
class DiagnosticVector:
    """
    4D diagnostic state for a single layer.

    Each dimension is normalized to [0, 1] where:
    - 0 = minimal transformation effort needed
    - 1 = maximum transformation effort needed
    """

    # Overlap score (from RiemannianDensity)
    # High overlap indicates alpha scaling needed
    interference_score: float

    # Layer importance (from RefinementDensity)
    # High density = must preserve carefully
    importance_score: float

    # Numerical stability (from SpectralAnalysis)
    # High condition number = spectral clamping needed
    instability_score: float

    # Manifold complexity (from IntrinsicDimension)
    # High dimension = TSV pruning may help
    complexity_score: float

    @property
    def vector(self) -> list[float]:
        """Return as list for polytope operations."""
        return [
            self.interference_score,
            self.importance_score,
            self.instability_score,
            self.complexity_score,
        ]

    @property
    def magnitude(self) -> float:
        """L2 norm of diagnostic vector (total transformation effort)."""
        vec = self.vector
        return math.sqrt(sum(v * v for v in vec))

    @property
    def max_dimension(self) -> str:
        """Which dimension needs most attention."""
        dims = ["interference", "importance", "instability", "complexity"]
        vec = self.vector
        idx = vec.index(max(vec))
        return dims[idx]


@dataclass(frozen=True)
class PolytopeBounds:
    """Threshold configuration for transformation polytope.

    Derive from baseline measurements via from_baseline_metrics().
    """

    interference_threshold: float
    importance_threshold: float
    instability_threshold: float
    complexity_threshold: float
    magnitude_threshold: float
    high_instability_threshold: float
    high_interference_threshold: float

    @classmethod
    def from_baseline_metrics(
        cls,
        interference_samples: list[float],
        importance_samples: list[float] | None = None,
        instability_samples: list[float],
        complexity_samples: list[float],
        magnitude_samples: list[float],
        *,
        threshold_percentile: float = 0.75,
        high_percentile: float = 0.90,
    ) -> "PolytopeBounds":
        """Derive thresholds from baseline metric distributions."""
        if not all([interference_samples, instability_samples, complexity_samples, magnitude_samples]):
            raise ValueError("All metric sample lists required for calibration")

        def percentile(samples: list[float], p: float) -> float:
            sorted_s = sorted(samples)
            idx = int(p * (len(sorted_s) - 1))
            return sorted_s[idx]

        return cls(
            interference_threshold=percentile(interference_samples, threshold_percentile),
            importance_threshold=percentile(
                importance_samples or interference_samples, threshold_percentile
            ),
            instability_threshold=percentile(instability_samples, threshold_percentile),
            complexity_threshold=percentile(complexity_samples, threshold_percentile),
            magnitude_threshold=percentile(magnitude_samples, threshold_percentile),
            high_instability_threshold=percentile(instability_samples, high_percentile),
            high_interference_threshold=percentile(interference_samples, high_percentile),
        )

@dataclass
class TransformationTrigger:
    """A single transformation trigger from exceeding a threshold."""

    dimension: str
    value: float
    threshold: float
    intensity: float  # How far beyond threshold (0 = at threshold)
    transformation: TransformationType


@dataclass
class LayerTransformationResult:
    """
    Result of transformation analysis for a layer.

    Contains the transformations needed and recommended parameters.
    """

    # Diagnostic measurements
    diagnostics: DiagnosticVector

    # Transformations needed (empty if none needed)
    transformations: list[TransformationType] = field(default_factory=list)

    # Detailed triggers
    triggers: list[TransformationTrigger] = field(default_factory=list)

    # Recommended alpha adjustment
    recommended_alpha: float | None = None

    # Measurement confidence
    confidence: float = 1.0

    # Layer index (if per-layer analysis)
    layer: int | None = None

    @property
    def transformation_effort(self) -> float:
        """Total transformation effort (0 = direct merge, higher = more work)."""
        return self.diagnostics.magnitude

    @property
    def needs_spectral_clamping(self) -> bool:
        """Whether spectral clamping is needed."""
        return TransformationType.SPECTRAL_CLAMP in self.transformations

    @property
    def needs_alpha_reduction(self) -> bool:
        """Whether alpha reduction is needed."""
        return TransformationType.REDUCE_ALPHA in self.transformations


@dataclass
class ModelTransformationProfile:
    """
    Aggregate transformation profile across all layers.
    """

    per_layer: dict[int, LayerTransformationResult]

    # Layers by transformation intensity
    direct_merge_layers: list[int]  # No transformations needed
    light_transform_layers: list[int]  # Minor transformations
    heavy_transform_layers: list[int]  # Multiple transformations

    # All transformations needed
    all_transformations: list[TransformationType]

    # Summary metrics
    mean_interference: float
    mean_importance: float
    mean_instability: float
    mean_complexity: float

    @property
    def total_transformation_effort(self) -> float:
        """Total transformation effort across all layers."""
        return sum(r.transformation_effort for r in self.per_layer.values())


class SafetyPolytope:
    """Determines transformations needed for model merging."""

    def __init__(self, bounds: PolytopeBounds) -> None:
        self.bounds = bounds
        self._build_constraints()

    def _build_constraints(self) -> None:
        """Build the polytope constraint matrix."""
        self.A = [
            [1, 0, 0, 0],  # interference
            [0, 1, 0, 0],  # importance
            [0, 0, 1, 0],  # instability
            [0, 0, 0, 1],  # complexity
        ]

        self.b = [
            self.bounds.interference_threshold,
            self.bounds.importance_threshold,
            self.bounds.instability_threshold,
            self.bounds.complexity_threshold,
        ]

    def analyze_layer(
        self,
        diagnostics: DiagnosticVector,
        layer: int | None = None,
        base_alpha: float | None = None,
    ) -> LayerTransformationResult:
        """
        Analyze what transformations a layer needs.

        Args:
            diagnostics: 4D diagnostic vector for the layer
            layer: Optional layer index
            base_alpha: Base merge coefficient before adjustment

        Returns:
            LayerTransformationResult with transformations needed
        """
        triggers: list[TransformationTrigger] = []
        transformations: list[TransformationType] = []

        x = diagnostics.vector

        # Check each dimension against threshold
        constraint_values = []
        for row in self.A:
            val = sum(row[i] * x[i] for i in range(len(x)))
            constraint_values.append(val)

        dimension_names = ["interference", "importance", "instability", "complexity"]
        transformation_map = {
            "interference": TransformationType.NULL_SPACE_FILTER,
            "importance": TransformationType.REDUCE_ALPHA,
            "instability": TransformationType.SPECTRAL_CLAMP,
            "complexity": TransformationType.TSV_PRUNE,
        }

        for i, (val, threshold, name) in enumerate(
            zip(constraint_values, self.b, dimension_names)
        ):
            if val > threshold:
                intensity = (val - threshold) / (1.0 - threshold + 1e-6)
                triggers.append(
                    TransformationTrigger(
                        dimension=name,
                        value=float(val),
                        threshold=float(threshold),
                        intensity=float(min(1.0, intensity)),
                        transformation=transformation_map[name],
                    )
                )
                if transformation_map[name] not in transformations:
                    transformations.append(transformation_map[name])

        # Check overall magnitude
        magnitude = diagnostics.magnitude
        if magnitude > self.bounds.magnitude_threshold:
            triggers.append(
                TransformationTrigger(
                    dimension="magnitude",
                    value=float(magnitude),
                    threshold=self.bounds.magnitude_threshold,
                    intensity=float(
                        (magnitude - self.bounds.magnitude_threshold)
                        / self.bounds.magnitude_threshold
                    ),
                    transformation=TransformationType.LAYER_SKIP,
                )
            )
            if TransformationType.LAYER_SKIP not in transformations:
                transformations.append(TransformationType.LAYER_SKIP)

        # Compute recommended alpha only when a base alpha is provided
        recommended_alpha = (
            self._compute_adjusted_alpha(base_alpha, diagnostics, triggers)
            if base_alpha is not None
            else None
        )

        # Compute confidence based on how close to boundaries
        confidence = self._compute_confidence(diagnostics)

        return LayerTransformationResult(
            diagnostics=diagnostics,
            transformations=transformations,
            triggers=triggers,
            recommended_alpha=recommended_alpha,
            confidence=confidence,
            layer=layer,
        )

    def _compute_adjusted_alpha(
        self,
        base_alpha: float,
        diagnostics: DiagnosticVector,
        triggers: list[TransformationTrigger],
    ) -> float:
        """Compute alpha adjustment based on triggers."""
        if not triggers:
            return base_alpha

        alpha = base_alpha

        # Reduce alpha for interference triggers
        interference_triggers = [t for t in triggers if t.dimension == "interference"]
        for t in interference_triggers:
            alpha *= 1.0 - 0.3 * t.intensity

        # Reduce alpha for importance triggers
        importance_triggers = [t for t in triggers if t.dimension == "importance"]
        for t in importance_triggers:
            alpha *= 1.0 - 0.2 * t.intensity

        # Strongly reduce for instability
        instability_triggers = [t for t in triggers if t.dimension == "instability"]
        for t in instability_triggers:
            alpha *= 1.0 - 0.5 * t.intensity

        return max(0.1, min(0.95, alpha))

    def _compute_confidence(self, diagnostics: DiagnosticVector) -> float:
        """Compute confidence in the measurements."""
        x = diagnostics.vector

        distances = []
        for i in range(len(self.b)):
            row = self.A[i]
            constraint_val = sum(row[j] * x[j] for j in range(len(x)))
            distances.append(self.b[i] - constraint_val)

        normalized_distances = [
            distances[i] / (self.b[i] + 1e-6) for i in range(len(distances))
        ]
        min_distance = min(normalized_distances)

        if min_distance < 0:
            return max(0.3, 1.0 + min_distance)
        else:
            return min(1.0, 0.5 + 0.5 * min_distance)

    def analyze_model_pair(
        self,
        layer_diagnostics: dict[int, DiagnosticVector],
        base_alpha: float | None = None,
    ) -> ModelTransformationProfile:
        """
        Analyze transformations needed across all layers.

        Args:
            layer_diagnostics: Dict mapping layer index to diagnostic vector
            base_alpha: Base merge coefficient

        Returns:
            ModelTransformationProfile with aggregate analysis
        """
        per_layer: dict[int, LayerTransformationResult] = {}

        direct_merge_layers: list[int] = []
        light_transform_layers: list[int] = []
        heavy_transform_layers: list[int] = []

        all_transformations: set[TransformationType] = set()

        interference_sum = 0.0
        importance_sum = 0.0
        instability_sum = 0.0
        complexity_sum = 0.0

        for layer_idx, diag in sorted(layer_diagnostics.items()):
            result = self.analyze_layer(diag, layer=layer_idx, base_alpha=base_alpha)
            per_layer[layer_idx] = result

            # Categorize by transformation count
            n_transforms = len(result.transformations)
            if n_transforms == 0:
                direct_merge_layers.append(layer_idx)
            elif n_transforms <= 2:
                light_transform_layers.append(layer_idx)
            else:
                heavy_transform_layers.append(layer_idx)

            all_transformations.update(result.transformations)

            interference_sum += diag.interference_score
            importance_sum += diag.importance_score
            instability_sum += diag.instability_score
            complexity_sum += diag.complexity_score

        n_layers = len(layer_diagnostics)

        return ModelTransformationProfile(
            per_layer=per_layer,
            direct_merge_layers=direct_merge_layers,
            light_transform_layers=light_transform_layers,
            heavy_transform_layers=heavy_transform_layers,
            all_transformations=list(all_transformations),
            mean_interference=interference_sum / n_layers if n_layers else 0,
            mean_importance=importance_sum / n_layers if n_layers else 0,
            mean_instability=instability_sum / n_layers if n_layers else 0,
            mean_complexity=complexity_sum / n_layers if n_layers else 0,
        )


def create_diagnostic_vector(
    interference: float,
    refinement_density: float,
    condition_number: float,
    intrinsic_dimension: int,
    hidden_dim: int,
) -> DiagnosticVector:
    """
    Create a normalized diagnostic vector from raw measurements.

    Args:
        interference: Interference score [0, 1]
        refinement_density: Density score [0, 1]
        condition_number: Condition number from spectral analysis
        intrinsic_dimension: Estimated intrinsic dimension
        hidden_dim: Model hidden dimension (for normalization)

    Returns:
        DiagnosticVector with normalized scores
    """
    interference_score = min(1.0, max(0.0, interference))
    importance_score = min(1.0, max(0.0, refinement_density))

    if condition_number <= 1:
        instability_score = 0.0
    elif condition_number >= 1000:
        instability_score = 1.0
    else:
        instability_score = math.log10(condition_number) / 3.0

    if hidden_dim > 0:
        dim_ratio = intrinsic_dimension / hidden_dim
        complexity_score = min(1.0, dim_ratio)
    else:
        complexity_score = 0.5

    return DiagnosticVector(
        interference_score=float(interference_score),
        importance_score=float(importance_score),
        instability_score=float(instability_score),
        complexity_score=float(complexity_score),
    )


def format_transformation_report(profile: ModelTransformationProfile) -> str:
    """Format a human-readable transformation report."""
    lines = [
        "=" * 60,
        "MERGE TRANSFORMATION ANALYSIS",
        "=" * 60,
        "",
        f"Total Transformation Effort: {profile.total_transformation_effort:.2f}",
        "",
        "-" * 40,
        "Layer Classification",
        "-" * 40,
        f"  Direct Merge:       {len(profile.direct_merge_layers)} layers",
        f"  Light Transform:    {len(profile.light_transform_layers)} layers",
        f"  Heavy Transform:    {len(profile.heavy_transform_layers)} layers",
        "",
        "-" * 40,
        "Diagnostic Means",
        "-" * 40,
        f"  Interference: {profile.mean_interference:.3f}",
        f"  Importance:   {profile.mean_importance:.3f}",
        f"  Instability:  {profile.mean_instability:.3f}",
        f"  Complexity:   {profile.mean_complexity:.3f}",
    ]

    if profile.all_transformations:
        lines.extend(
            [
                "",
                "-" * 40,
                "Transformations Needed",
                "-" * 40,
            ]
        )
        for t in profile.all_transformations:
            lines.append(f"  • {t.value}")

    if profile.heavy_transform_layers:
        lines.extend(
            [
                "",
                "-" * 40,
                "Layers Needing Multiple Transformations",
                "-" * 40,
            ]
        )
        for layer_idx in profile.heavy_transform_layers:
            result = profile.per_layer[layer_idx]
            transforms = ", ".join(t.value for t in result.transformations)
            lines.append(f"  Layer {layer_idx}: {transforms}")

    lines.append("")
    return "\n".join(lines)


class SafetyVerdict(str, Enum):
    """Safety verdict classification for numerical stability."""

    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    CRITICAL = "critical"


__all__ = [
    "SafetyVerdict",
    "TransformationType",
    "DiagnosticVector",
    "PolytopeBounds",
    "TransformationTrigger",
    "LayerTransformationResult",
    "ModelTransformationProfile",
    "SafetyPolytope",
    "create_diagnostic_vector",
    "format_transformation_report",
]
