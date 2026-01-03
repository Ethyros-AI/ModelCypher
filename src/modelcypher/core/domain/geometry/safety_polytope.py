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

Defines a diagnostic polytope from four measured dimensions:
overlap/interference, importance, instability, and complexity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    log_scalar,
    machine_epsilon,
    sqrt_scalar,
)

logger = logging.getLogger(__name__)

# IEEE 754 float32 machine epsilon (2^-23)
# This is the smallest value where 1.0 + eps != 1.0 in float32.
_FLOAT32_MACHINE_EPS = 2.0 ** -23


class TransformationType(str, Enum):
    """Types of transformations that may be needed for merging."""

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
    # Higher overlap implies tighter null-space constraint.
    interference_score: float

    # Layer importance (from RefinementDensity)
    # High importance indicates preservation pressure.
    importance_score: float

    # Numerical stability (from SpectralAnalysis)
    # High condition number = spectral clamping needed.
    instability_score: float

    # Manifold complexity (from IntrinsicDimension)
    # High dimension = TSV pruning may help.
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
        _b = get_default_backend()
        return sqrt_scalar(sum(v * v for v in vec), _b)

    @property
    def max_dimension(self) -> str:
        """Which dimension needs most attention."""
        dims = ["interference", "importance", "instability", "complexity"]
        vec = self.vector
        idx = vec.index(max(vec))
        return dims[idx]


@dataclass(frozen=True)
class PolytopeBounds:
    """Boundary configuration for transformation polytope.

    Derived from baseline measurements via from_baseline_metrics().
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
        instability_samples: list[float],
        complexity_samples: list[float],
        magnitude_samples: list[float],
        importance_samples: list[float] | None = None,
    ) -> "PolytopeBounds":
        """Derive thresholds from baseline metric distributions."""
        if not all([interference_samples, instability_samples, complexity_samples, magnitude_samples]):
            raise ValueError("All metric sample lists required for calibration")

        def gap_threshold(samples: list[float]) -> float:
            sorted_s = sorted(samples)
            _b = get_default_backend()
            eps = machine_epsilon(_b, _b.array([1.0]))
            return find_magnitude_gap_threshold(sorted_s, eps=eps)

        def high_threshold(samples: list[float]) -> float:
            return max(samples) if samples else 0.0

        importance_list = importance_samples or interference_samples

        return cls(
            interference_threshold=gap_threshold(interference_samples),
            importance_threshold=gap_threshold(importance_list),
            instability_threshold=gap_threshold(instability_samples),
            complexity_threshold=gap_threshold(complexity_samples),
            magnitude_threshold=gap_threshold(magnitude_samples),
            high_instability_threshold=high_threshold(instability_samples),
            high_interference_threshold=high_threshold(interference_samples),
        )


@dataclass
class TransformationTrigger:
    """A single transformation trigger from exceeding a boundary."""

    dimension: str
    value: float
    threshold: float
    intensity: float  # How far beyond threshold (0 = at threshold)
    transformation: TransformationType


@dataclass
class LayerTransformationResult:
    """
    Result of transformation analysis for a layer.

    Contains the transformations needed and raw diagnostic measurements.
    """

    # Diagnostic measurements
    diagnostics: DiagnosticVector

    # Transformations needed (empty if none needed)
    transformations: list[TransformationType] = field(default_factory=list)

    # Detailed triggers
    triggers: list[TransformationTrigger] = field(default_factory=list)

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


@dataclass
class ModelTransformationProfile:
    """Aggregate transformation profile across all layers."""

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
    ) -> LayerTransformationResult:
        """
        Analyze what transformations a layer needs.

        Args:
            diagnostics: 4D diagnostic vector for the layer
            layer: Optional layer index

        Returns:
            LayerTransformationResult with transformations needed
        """
        triggers: list[TransformationTrigger] = []
        transformations: list[TransformationType] = []

        x = diagnostics.vector

        # Check each dimension against boundary
        constraint_values = []
        for row in self.A:
            val = sum(row[i] * x[i] for i in range(len(x)))
            constraint_values.append(val)

        dimension_names = ["interference", "importance", "instability", "complexity"]
        transformation_map = {
            "interference": TransformationType.NULL_SPACE_FILTER,
            # Importance implies preservation pressure; null-space filtering preserves target geometry.
            "importance": TransformationType.NULL_SPACE_FILTER,
            "instability": TransformationType.SPECTRAL_CLAMP,
            "complexity": TransformationType.TSV_PRUNE,
        }

        _b = get_default_backend()
        eps = machine_epsilon(_b, _b.array([1.0]))
        for val, threshold, name in zip(constraint_values, self.b, dimension_names):
            if val > threshold:
                denom = max(1.0 - threshold, eps)
                intensity = (val - threshold) / denom
                triggers.append(
                    TransformationTrigger(
                        dimension=name,
                        value=float(val),
                        threshold=float(threshold),
                        intensity=float(min(1.0, max(0.0, intensity))),
                        transformation=transformation_map[name],
                    )
                )
                if transformation_map[name] not in transformations:
                    transformations.append(transformation_map[name])

        # Check overall magnitude
        magnitude = diagnostics.magnitude
        if magnitude > self.bounds.magnitude_threshold:
            denom = max(self.bounds.magnitude_threshold, eps)
            triggers.append(
                TransformationTrigger(
                    dimension="magnitude",
                    value=float(magnitude),
                    threshold=self.bounds.magnitude_threshold,
                    intensity=float((magnitude - self.bounds.magnitude_threshold) / denom),
                    transformation=TransformationType.LAYER_SKIP,
                )
            )
            if TransformationType.LAYER_SKIP not in transformations:
                transformations.append(TransformationType.LAYER_SKIP)

        # Compute confidence based on how close to boundaries
        confidence = self._compute_confidence(diagnostics)

        return LayerTransformationResult(
            diagnostics=diagnostics,
            transformations=transformations,
            triggers=triggers,
            confidence=confidence,
            layer=layer,
        )

    def _compute_confidence(self, diagnostics: DiagnosticVector) -> float:
        """Compute confidence in the measurements."""
        x = diagnostics.vector

        distances = []
        for i in range(len(self.b)):
            row = self.A[i]
            constraint_val = sum(row[j] * x[j] for j in range(len(x)))
            distances.append(self.b[i] - constraint_val)

        _b = get_default_backend()
        eps = machine_epsilon(_b, _b.array([1.0]))
        normalized_distances = [
            distances[i] / max(self.b[i], eps) for i in range(len(distances))
        ]
        min_distance = min(normalized_distances)

        if min_distance < 0:
            return max(0.3, 1.0 + min_distance)
        return min(1.0, 0.5 + 0.5 * min_distance)

    def analyze_model_pair(
        self,
        layer_diagnostics: dict[int, DiagnosticVector],
    ) -> ModelTransformationProfile:
        """
        Analyze transformations needed across all layers.

        Args:
            layer_diagnostics: Dict mapping layer index to diagnostic vector

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
            result = self.analyze_layer(diag, layer=layer_idx)
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

    # Use float32 precision bounds for stability normalization.
    _b = get_default_backend()
    max_stable_condition = 1.0 / sqrt_scalar(_FLOAT32_MACHINE_EPS, _b)
    if condition_number <= 1.0:
        instability_score = 0.0
    else:
        log_cond = log_scalar(condition_number, _b)
        log_max = log_scalar(max_stable_condition, _b)
        instability_score = min(1.0, log_cond / log_max)

    if hidden_dim > 0:
        dim_ratio = intrinsic_dimension / hidden_dim
        complexity_score = min(1.0, max(0.0, dim_ratio))
    else:
        complexity_score = 0.0

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


# SafetyVerdict enum was REMOVED.
# Verdicts are subjective interpretations. The geometry IS what it is.
# Return raw measurements (interference, importance, instability, complexity).
# Callers interpret measurements relative to their own baselines.


__all__ = [
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
