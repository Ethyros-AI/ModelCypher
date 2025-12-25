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

"""
Safety Polytope: Unified merge-safety decision boundary.

Integrates four diagnostic dimensions into a convex safe operating region:
1. Volume/Overlap (RiemannianDensity) - Interference potential
2. Mass/Importance (RefinementDensity) - Layer significance
3. Stability (SpectralAnalysis) - Numerical conditioning
4. Complexity (IntrinsicDimension) - Manifold topology

The Safety Polytope defines the region in 4D diagnostic space where
merging is mathematically safe. Points outside require mitigation.

Mathematical Foundation:
    A merge is safe iff the diagnostic vector lies within the polytope:

    P = {x ∈ R^4 : Ax ≤ b}

    where A encodes the safety constraints and b the thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SafetyVerdict(str, Enum):
    """Overall safety classification."""

    SAFE = "safe"  # All diagnostics within bounds
    CAUTION = "caution"  # Minor violations, proceed with mitigations
    UNSAFE = "unsafe"  # Major violations, do not merge
    CRITICAL = "critical"  # Numerical instability risk


class MitigationType(str, Enum):
    """Types of mitigations available."""

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
    - 0 = ideal (no concern)
    - 1 = critical (maximum concern)
    """

    # Interference potential (from RiemannianDensity)
    # High overlap = high interference risk
    interference_score: float

    # Layer importance (from RefinementDensity)
    # High density = must preserve carefully
    importance_score: float

    # Numerical stability (from SpectralAnalysis)
    # High condition number = instability risk (normalized log scale)
    instability_score: float

    # Manifold complexity (from IntrinsicDimension)
    # High dimension = complex interactions (normalized)
    complexity_score: float

    @property
    def vector(self) -> np.ndarray:
        """Return as numpy array for polytope operations."""
        return np.array(
            [
                self.interference_score,
                self.importance_score,
                self.instability_score,
                self.complexity_score,
            ]
        )

    @property
    def magnitude(self) -> float:
        """L2 norm of diagnostic vector (overall risk)."""
        return float(np.linalg.norm(self.vector))

    @property
    def max_dimension(self) -> str:
        """Which dimension has highest concern."""
        dims = ["interference", "importance", "instability", "complexity"]
        idx = int(np.argmax(self.vector))
        return dims[idx]


@dataclass(frozen=True)
class PolytopeBounds:
    """
    Threshold configuration for safety polytope.

    Each bound defines the maximum acceptable value for that dimension.
    Values beyond the bound trigger mitigations or rejection.
    """

    # Maximum interference score before mitigation
    max_interference: float = 0.6

    # Maximum importance score for standard blending
    # Higher importance = more careful handling
    max_importance_for_blend: float = 0.7

    # Maximum instability before spectral clamping
    max_instability: float = 0.5

    # Maximum complexity before dimension reduction
    max_complexity: float = 0.8

    # Overall magnitude threshold
    max_magnitude: float = 1.2

    # Critical thresholds (triggers CRITICAL verdict)
    critical_instability: float = 0.9
    critical_interference: float = 0.9


@dataclass
class PolytopeViolation:
    """A single constraint violation."""

    dimension: str
    value: float
    threshold: float
    severity: float  # How far beyond threshold (0 = at threshold)
    mitigation: MitigationType


@dataclass
class SafetyPolytopeResult:
    """
    Result of safety polytope analysis for a layer or model.
    """

    # Overall verdict
    verdict: SafetyVerdict

    # Diagnostic vector
    diagnostics: DiagnosticVector

    # List of violations (empty if SAFE)
    violations: list[PolytopeViolation] = field(default_factory=list)

    # Recommended mitigations in priority order
    mitigations: list[MitigationType] = field(default_factory=list)

    # Adjusted alpha (if mitigation includes alpha reduction)
    recommended_alpha: float | None = None

    # Confidence in the verdict (based on diagnostic reliability)
    confidence: float = 1.0

    # Layer index (if per-layer analysis)
    layer: int | None = None

    @property
    def is_safe(self) -> bool:
        return self.verdict == SafetyVerdict.SAFE

    @property
    def needs_mitigation(self) -> bool:
        return self.verdict in (SafetyVerdict.CAUTION, SafetyVerdict.UNSAFE)

    @property
    def is_critical(self) -> bool:
        return self.verdict == SafetyVerdict.CRITICAL


@dataclass
class ModelSafetyProfile:
    """
    Aggregate safety profile across all layers.
    """

    per_layer: dict[int, SafetyPolytopeResult]

    # Aggregate statistics
    safe_layers: list[int]
    caution_layers: list[int]
    unsafe_layers: list[int]
    critical_layers: list[int]

    # Overall verdict (worst case across layers)
    overall_verdict: SafetyVerdict

    # Recommended actions
    global_mitigations: list[MitigationType]

    # Summary metrics
    mean_interference: float
    mean_importance: float
    mean_instability: float
    mean_complexity: float

    @property
    def mergeable(self) -> bool:
        """Whether the model pair is safe to merge with mitigations."""
        return self.overall_verdict != SafetyVerdict.CRITICAL


class SafetyPolytope:
    """
    Unified safety boundary for model merging decisions.

    Combines diagnostics from:
    - RiemannianDensityEstimator (interference)
    - RefinementDensityAnalyzer (importance)
    - spectral_analysis (stability)
    - IntrinsicDimensionEstimator (complexity)

    Into a single convex polytope that defines the safe operating region.
    """

    def __init__(self, bounds: PolytopeBounds | None = None) -> None:
        self.bounds = bounds or PolytopeBounds()

        # Build constraint matrix A and threshold vector b
        # Polytope is defined as {x : Ax <= b}
        self._build_constraints()

    def _build_constraints(self) -> None:
        """Build the polytope constraint matrix."""
        # Individual dimension constraints
        # x_i <= threshold_i for each dimension
        self.A = np.array(
            [
                [1, 0, 0, 0],  # interference <= max_interference
                [0, 1, 0, 0],  # importance <= max_importance
                [0, 0, 1, 0],  # instability <= max_instability
                [0, 0, 0, 1],  # complexity <= max_complexity
            ]
        )

        self.b = np.array(
            [
                self.bounds.max_interference,
                self.bounds.max_importance_for_blend,
                self.bounds.max_instability,
                self.bounds.max_complexity,
            ]
        )

    def check_layer(
        self,
        diagnostics: DiagnosticVector,
        layer: int | None = None,
        base_alpha: float = 0.5,
    ) -> SafetyPolytopeResult:
        """
        Check if a layer's diagnostics fall within the safety polytope.

        Args:
            diagnostics: 4D diagnostic vector for the layer
            layer: Optional layer index
            base_alpha: Base merge coefficient before adjustment

        Returns:
            SafetyPolytopeResult with verdict and mitigations
        """
        violations: list[PolytopeViolation] = []
        mitigations: list[MitigationType] = []

        x = diagnostics.vector

        # Check polytope constraints
        constraint_values = self.A @ x

        dimension_names = ["interference", "importance", "instability", "complexity"]
        mitigation_map = {
            "interference": MitigationType.NULL_SPACE_FILTER,
            "importance": MitigationType.REDUCE_ALPHA,
            "instability": MitigationType.SPECTRAL_CLAMP,
            "complexity": MitigationType.TSV_PRUNE,
        }

        for i, (val, threshold, name) in enumerate(zip(constraint_values, self.b, dimension_names)):
            if val > threshold:
                severity = (val - threshold) / (1.0 - threshold + 1e-6)
                violations.append(
                    PolytopeViolation(
                        dimension=name,
                        value=float(val),
                        threshold=float(threshold),
                        severity=float(min(1.0, severity)),
                        mitigation=mitigation_map[name],
                    )
                )
                if mitigation_map[name] not in mitigations:
                    mitigations.append(mitigation_map[name])

        # Check critical thresholds
        is_critical = (
            diagnostics.instability_score > self.bounds.critical_instability
            or diagnostics.interference_score > self.bounds.critical_interference
        )

        # Check overall magnitude
        magnitude = diagnostics.magnitude
        if magnitude > self.bounds.max_magnitude:
            violations.append(
                PolytopeViolation(
                    dimension="magnitude",
                    value=float(magnitude),
                    threshold=self.bounds.max_magnitude,
                    severity=float(
                        (magnitude - self.bounds.max_magnitude) / self.bounds.max_magnitude
                    ),
                    mitigation=MitigationType.LAYER_SKIP,
                )
            )
            if MitigationType.LAYER_SKIP not in mitigations:
                mitigations.append(MitigationType.LAYER_SKIP)

        # Determine verdict
        if is_critical:
            verdict = SafetyVerdict.CRITICAL
        elif len(violations) == 0:
            verdict = SafetyVerdict.SAFE
        elif any(v.severity > 0.5 for v in violations):
            verdict = SafetyVerdict.UNSAFE
        else:
            verdict = SafetyVerdict.CAUTION

        # Compute recommended alpha
        recommended_alpha = self._compute_adjusted_alpha(base_alpha, diagnostics, violations)

        # Compute confidence based on how close to boundaries
        confidence = self._compute_confidence(diagnostics)

        return SafetyPolytopeResult(
            verdict=verdict,
            diagnostics=diagnostics,
            violations=violations,
            mitigations=mitigations,
            recommended_alpha=recommended_alpha,
            confidence=confidence,
            layer=layer,
        )

    def _compute_adjusted_alpha(
        self,
        base_alpha: float,
        diagnostics: DiagnosticVector,
        violations: list[PolytopeViolation],
    ) -> float:
        """Compute alpha adjustment based on violations."""
        if not violations:
            return base_alpha

        # Start with base alpha
        alpha = base_alpha

        # Reduce alpha for interference violations
        interference_violations = [v for v in violations if v.dimension == "interference"]
        for v in interference_violations:
            # Higher interference = lower alpha (trust target more)
            alpha *= 1.0 - 0.3 * v.severity

        # Reduce alpha for importance violations
        importance_violations = [v for v in violations if v.dimension == "importance"]
        for v in importance_violations:
            # High importance source = might want higher alpha actually
            # But violation means we need to be careful
            alpha *= 1.0 - 0.2 * v.severity

        # Strongly reduce for instability
        instability_violations = [v for v in violations if v.dimension == "instability"]
        for v in instability_violations:
            alpha *= 1.0 - 0.5 * v.severity

        # Clamp to valid range
        return max(0.1, min(0.95, alpha))

    def _compute_confidence(self, diagnostics: DiagnosticVector) -> float:
        """
        Compute confidence in the verdict.

        Lower confidence when diagnostics are near boundaries.
        """
        x = diagnostics.vector

        # Distance to each boundary (positive = inside, negative = outside)
        distances = self.b - self.A @ x

        # Normalize distances by threshold values
        normalized_distances = distances / (self.b + 1e-6)

        # Confidence is based on minimum normalized distance
        # Far from boundaries = high confidence
        # Close to boundaries = lower confidence
        min_distance = float(np.min(normalized_distances))

        if min_distance < 0:
            # Outside polytope
            return max(0.3, 1.0 + min_distance)  # Decreases as we go further out
        else:
            # Inside polytope
            return min(1.0, 0.5 + 0.5 * min_distance)  # Increases toward center

    def analyze_model_pair(
        self,
        layer_diagnostics: dict[int, DiagnosticVector],
        base_alpha: float = 0.5,
    ) -> ModelSafetyProfile:
        """
        Analyze safety across all layers.

        Args:
            layer_diagnostics: Dict mapping layer index to diagnostic vector
            base_alpha: Base merge coefficient

        Returns:
            ModelSafetyProfile with aggregate analysis
        """
        per_layer: dict[int, SafetyPolytopeResult] = {}

        safe_layers: list[int] = []
        caution_layers: list[int] = []
        unsafe_layers: list[int] = []
        critical_layers: list[int] = []

        all_mitigations: set[MitigationType] = set()

        interference_sum = 0.0
        importance_sum = 0.0
        instability_sum = 0.0
        complexity_sum = 0.0

        for layer_idx, diag in sorted(layer_diagnostics.items()):
            result = self.check_layer(diag, layer=layer_idx, base_alpha=base_alpha)
            per_layer[layer_idx] = result

            if result.verdict == SafetyVerdict.SAFE:
                safe_layers.append(layer_idx)
            elif result.verdict == SafetyVerdict.CAUTION:
                caution_layers.append(layer_idx)
            elif result.verdict == SafetyVerdict.UNSAFE:
                unsafe_layers.append(layer_idx)
            else:
                critical_layers.append(layer_idx)

            all_mitigations.update(result.mitigations)

            interference_sum += diag.interference_score
            importance_sum += diag.importance_score
            instability_sum += diag.instability_score
            complexity_sum += diag.complexity_score

        n_layers = len(layer_diagnostics)

        # Determine overall verdict (worst case)
        if critical_layers:
            overall_verdict = SafetyVerdict.CRITICAL
        elif unsafe_layers:
            overall_verdict = SafetyVerdict.UNSAFE
        elif caution_layers:
            overall_verdict = SafetyVerdict.CAUTION
        else:
            overall_verdict = SafetyVerdict.SAFE

        return ModelSafetyProfile(
            per_layer=per_layer,
            safe_layers=safe_layers,
            caution_layers=caution_layers,
            unsafe_layers=unsafe_layers,
            critical_layers=critical_layers,
            overall_verdict=overall_verdict,
            global_mitigations=list(all_mitigations),
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
        interference: Interference score from InterferencePredictor [0, 1]
        refinement_density: Density score from RefinementDensityAnalyzer [0, 1]
        condition_number: Condition number from spectral analysis
        intrinsic_dimension: Estimated intrinsic dimension
        hidden_dim: Model hidden dimension (for normalization)

    Returns:
        DiagnosticVector with normalized scores
    """
    # Interference is already normalized
    interference_score = min(1.0, max(0.0, interference))

    # Refinement density: high density = high importance
    # Invert because higher density means more careful handling needed
    importance_score = min(1.0, max(0.0, refinement_density))

    # Condition number: log scale normalization
    # κ = 1 is perfect, κ > 100 is concerning, κ > 1000 is critical
    if condition_number <= 1:
        instability_score = 0.0
    elif condition_number >= 1000:
        instability_score = 1.0
    else:
        # Log scale: log10(1) = 0, log10(1000) = 3
        instability_score = np.log10(condition_number) / 3.0

    # Intrinsic dimension: relative to hidden dim
    # Low relative dimension = simple manifold
    # High relative dimension = complex manifold
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


def format_safety_report(profile: ModelSafetyProfile) -> str:
    """Format a human-readable safety report."""
    lines = [
        "=" * 60,
        "SAFETY POLYTOPE ANALYSIS",
        "=" * 60,
        "",
        f"Overall Verdict: {profile.overall_verdict.value.upper()}",
        f"Mergeable: {'Yes' if profile.mergeable else 'NO - CRITICAL ISSUES'}",
        "",
        "-" * 40,
        "Layer Classification",
        "-" * 40,
        f"  Safe:     {len(profile.safe_layers)} layers",
        f"  Caution:  {len(profile.caution_layers)} layers",
        f"  Unsafe:   {len(profile.unsafe_layers)} layers",
        f"  Critical: {len(profile.critical_layers)} layers",
        "",
        "-" * 40,
        "Diagnostic Means",
        "-" * 40,
        f"  Interference: {profile.mean_interference:.3f}",
        f"  Importance:   {profile.mean_importance:.3f}",
        f"  Instability:  {profile.mean_instability:.3f}",
        f"  Complexity:   {profile.mean_complexity:.3f}",
    ]

    if profile.global_mitigations:
        lines.extend(
            [
                "",
                "-" * 40,
                "Required Mitigations",
                "-" * 40,
            ]
        )
        for m in profile.global_mitigations:
            lines.append(f"  • {m.value}")

    if profile.unsafe_layers or profile.critical_layers:
        lines.extend(
            [
                "",
                "-" * 40,
                "Problem Layers",
                "-" * 40,
            ]
        )
        for layer_idx in profile.unsafe_layers + profile.critical_layers:
            result = profile.per_layer[layer_idx]
            lines.append(f"  Layer {layer_idx}: {result.verdict.value}")
            for v in result.violations:
                lines.append(f"    - {v.dimension}: {v.value:.3f} > {v.threshold:.3f}")

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SafetyVerdict",
    "MitigationType",
    "DiagnosticVector",
    "PolytopeBounds",
    "PolytopeViolation",
    "SafetyPolytopeResult",
    "ModelSafetyProfile",
    "SafetyPolytope",
    "create_diagnostic_vector",
    "format_safety_report",
]
