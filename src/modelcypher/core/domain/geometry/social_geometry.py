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
Social Geometry Analysis for Language Models.

This module probes the emergent "Social Manifold" in language models - the geometric
structure encoding power hierarchies, kinship relations, and formality gradients.

Key insight: Language models trained on human text absorb implicit social structures.
These structures manifest as geometric relationships in latent space:
- Power axis: slave → servant → citizen → noble → king
- Kinship axis: enemy → stranger → acquaintance → friend → family
- Formality axis: hey → hi → hello → greetings → salutations

Reference: Emergent Social Geometry (ModelCypher 2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class SocialAxis(str, Enum):
    """The three primary axes of social geometry."""

    POWER = "power"  # Hierarchy: low status → high status
    KINSHIP = "kinship"  # Social distance: hostile → intimate
    FORMALITY = "formality"  # Register: casual → formal


class SocialCategory(str, Enum):
    """Categories of social anchors."""

    POWER_HIERARCHY = "power_hierarchy"
    FORMALITY_REGISTER = "formality_register"
    KINSHIP_DISTANCE = "kinship_distance"
    STATUS_MARKERS = "status_markers"
    AGE_HIERARCHY = "age_hierarchy"


@dataclass(frozen=True)
class SocialAnchor:
    """A probe for social geometry measurement."""

    name: str
    prompt: str
    category: SocialCategory
    axis: SocialAxis
    level: int  # 1-5 scale within axis
    description: str = ""


# The Social Prime Atlas: 23 anchors across social dimensions
SOCIAL_PRIME_ATLAS: list[SocialAnchor] = [
    # Power Hierarchy (5 anchors, level 1-5)
    SocialAnchor(
        "slave",
        "The word slave represents",
        SocialCategory.POWER_HIERARCHY,
        SocialAxis.POWER,
        1,
        "Lowest power",
    ),
    SocialAnchor(
        "servant",
        "The word servant represents",
        SocialCategory.POWER_HIERARCHY,
        SocialAxis.POWER,
        2,
        "Low power",
    ),
    SocialAnchor(
        "citizen",
        "The word citizen represents",
        SocialCategory.POWER_HIERARCHY,
        SocialAxis.POWER,
        3,
        "Neutral power",
    ),
    SocialAnchor(
        "noble",
        "The word noble represents",
        SocialCategory.POWER_HIERARCHY,
        SocialAxis.POWER,
        4,
        "High power",
    ),
    SocialAnchor(
        "emperor",
        "The word emperor represents",
        SocialCategory.POWER_HIERARCHY,
        SocialAxis.POWER,
        5,
        "Highest power",
    ),
    # Formality Register (5 anchors, level 1-5)
    SocialAnchor(
        "hey",
        "The greeting hey represents",
        SocialCategory.FORMALITY_REGISTER,
        SocialAxis.FORMALITY,
        1,
        "Very casual",
    ),
    SocialAnchor(
        "hi",
        "The greeting hi represents",
        SocialCategory.FORMALITY_REGISTER,
        SocialAxis.FORMALITY,
        2,
        "Casual",
    ),
    SocialAnchor(
        "hello",
        "The greeting hello represents",
        SocialCategory.FORMALITY_REGISTER,
        SocialAxis.FORMALITY,
        3,
        "Neutral",
    ),
    SocialAnchor(
        "greetings",
        "The word greetings represents",
        SocialCategory.FORMALITY_REGISTER,
        SocialAxis.FORMALITY,
        4,
        "Formal",
    ),
    SocialAnchor(
        "salutations",
        "The word salutations represents",
        SocialCategory.FORMALITY_REGISTER,
        SocialAxis.FORMALITY,
        5,
        "Very formal",
    ),
    # Kinship Distance (5 anchors, level 1-5)
    SocialAnchor(
        "enemy",
        "The word enemy represents",
        SocialCategory.KINSHIP_DISTANCE,
        SocialAxis.KINSHIP,
        1,
        "Hostile",
    ),
    SocialAnchor(
        "stranger",
        "The word stranger represents",
        SocialCategory.KINSHIP_DISTANCE,
        SocialAxis.KINSHIP,
        2,
        "Unknown",
    ),
    SocialAnchor(
        "acquaintance",
        "The word acquaintance represents",
        SocialCategory.KINSHIP_DISTANCE,
        SocialAxis.KINSHIP,
        3,
        "Known",
    ),
    SocialAnchor(
        "friend",
        "The word friend represents",
        SocialCategory.KINSHIP_DISTANCE,
        SocialAxis.KINSHIP,
        4,
        "Close",
    ),
    SocialAnchor(
        "family",
        "The word family represents",
        SocialCategory.KINSHIP_DISTANCE,
        SocialAxis.KINSHIP,
        5,
        "Intimate",
    ),
    # Status Markers (4 anchors)
    SocialAnchor(
        "beggar",
        "The word beggar represents",
        SocialCategory.STATUS_MARKERS,
        SocialAxis.POWER,
        1,
        "Low status",
    ),
    SocialAnchor(
        "worker",
        "The word worker represents",
        SocialCategory.STATUS_MARKERS,
        SocialAxis.POWER,
        2,
        "Working class",
    ),
    SocialAnchor(
        "professional",
        "The word professional represents",
        SocialCategory.STATUS_MARKERS,
        SocialAxis.POWER,
        3,
        "Middle class",
    ),
    SocialAnchor(
        "wealthy",
        "The word wealthy represents",
        SocialCategory.STATUS_MARKERS,
        SocialAxis.POWER,
        5,
        "High status",
    ),
    # Age Hierarchy (4 anchors)
    SocialAnchor(
        "child",
        "The word child represents",
        SocialCategory.AGE_HIERARCHY,
        SocialAxis.POWER,
        1,
        "Young/dependent",
    ),
    SocialAnchor(
        "youth",
        "The word youth represents",
        SocialCategory.AGE_HIERARCHY,
        SocialAxis.POWER,
        2,
        "Adolescent",
    ),
    SocialAnchor(
        "adult",
        "The word adult represents",
        SocialCategory.AGE_HIERARCHY,
        SocialAxis.POWER,
        3,
        "Mature",
    ),
    SocialAnchor(
        "elder",
        "The word elder represents",
        SocialCategory.AGE_HIERARCHY,
        SocialAxis.POWER,
        4,
        "Respected senior",
    ),
]


@dataclass
class AxisOrthogonality:
    """Measures how independent the social axes are."""

    power_kinship: float  # 1.0 = perfectly orthogonal
    power_formality: float
    kinship_formality: float
    mean_orthogonality: float


@dataclass
class GradientConsistency:
    """Measures whether anchors form monotonic gradients along axes."""

    power_monotonic: bool
    power_correlation: float  # Spearman correlation with expected ordering
    kinship_monotonic: bool
    kinship_correlation: float
    formality_monotonic: bool
    formality_correlation: float


@dataclass
class PowerGradientResult:
    """Analysis of the power hierarchy axis."""

    power_axis_detected: bool
    power_direction: np.ndarray  # Unit vector pointing "up" in status
    status_correlation: float  # Correlation between activation position and expected status
    high_status_anchors: list[str]
    low_status_anchors: list[str]


@dataclass
class SocialGeometryReport:
    """Complete social geometry analysis report."""

    has_social_manifold: bool
    social_manifold_score: float  # 0-1, overall quality
    axis_orthogonality: AxisOrthogonality
    gradient_consistency: GradientConsistency
    power_gradient: PowerGradientResult
    principal_components_variance: list[float]
    anchor_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_social_manifold": self.has_social_manifold,
            "social_manifold_score": self.social_manifold_score,
            "axis_orthogonality": {
                "power_kinship": self.axis_orthogonality.power_kinship,
                "power_formality": self.axis_orthogonality.power_formality,
                "kinship_formality": self.axis_orthogonality.kinship_formality,
                "mean_orthogonality": self.axis_orthogonality.mean_orthogonality,
            },
            "gradient_consistency": {
                "power_monotonic": self.gradient_consistency.power_monotonic,
                "power_correlation": self.gradient_consistency.power_correlation,
                "kinship_monotonic": self.gradient_consistency.kinship_monotonic,
                "kinship_correlation": self.gradient_consistency.kinship_correlation,
                "formality_monotonic": self.gradient_consistency.formality_monotonic,
                "formality_correlation": self.gradient_consistency.formality_correlation,
            },
            "power_gradient": {
                "power_axis_detected": self.power_gradient.power_axis_detected,
                "status_correlation": self.power_gradient.status_correlation,
                "high_status_anchors": self.power_gradient.high_status_anchors,
                "low_status_anchors": self.power_gradient.low_status_anchors,
            },
            "principal_components_variance": self.principal_components_variance,
            "anchor_count": self.anchor_count,
        }


class SocialGeometryAnalyzer:
    """
    Analyzes the social geometry embedded in language model representations.

    Detects emergent social structure: power hierarchies, kinship relations,
    and formality gradients that models learn from training on human text.
    """

    def __init__(self, backend: "Backend"):
        self.backend = backend

    def _to_numpy(self, activations: dict[str, any]) -> tuple[list[str], np.ndarray]:
        """Convert activation dict to numpy matrix."""
        names = list(activations.keys())
        vectors = [self.backend.to_numpy(activations[n]) for n in names]
        return names, np.stack(vectors)

    def _compute_pca(self, X: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Compute PCA using numpy."""
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        X_pca = X_centered @ eigenvectors[:, :n_components]
        variance_explained = eigenvalues[:n_components] / eigenvalues.sum()
        return X_pca, variance_explained

    def _compute_axis_orthogonality(
        self,
        activations: dict[str, any],
    ) -> AxisOrthogonality:
        """Compute orthogonality between social axes."""

        def get_axis_vector(low_anchor: str, high_anchor: str) -> np.ndarray:
            low = self.backend.to_numpy(activations.get(low_anchor))
            high = self.backend.to_numpy(activations.get(high_anchor))
            if low is None or high is None:
                return np.zeros(1)
            return high - low

        def cosine_orthogonality(a: np.ndarray, b: np.ndarray) -> float:
            """1 - |cos(a, b)| gives orthogonality."""
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            cos = np.dot(a, b) / (norm_a * norm_b)
            return 1.0 - abs(cos)

        # Get axis direction vectors
        power_vec = get_axis_vector("slave", "emperor")
        kinship_vec = get_axis_vector("enemy", "family")
        formality_vec = get_axis_vector("hey", "salutations")

        pk = cosine_orthogonality(power_vec, kinship_vec)
        pf = cosine_orthogonality(power_vec, formality_vec)
        kf = cosine_orthogonality(kinship_vec, formality_vec)

        return AxisOrthogonality(
            power_kinship=pk,
            power_formality=pf,
            kinship_formality=kf,
            mean_orthogonality=(pk + pf + kf) / 3,
        )

    def _compute_gradient_consistency(
        self,
        names: list[str],
        X_pca: np.ndarray,
    ) -> GradientConsistency:
        """Check if axes form monotonic gradients."""

        # Define expected orderings
        power_order = ["slave", "servant", "citizen", "noble", "emperor"]
        kinship_order = ["enemy", "stranger", "acquaintance", "friend", "family"]
        formality_order = ["hey", "hi", "hello", "greetings", "salutations"]

        def check_monotonicity(order: list[str]) -> tuple[bool, float]:
            """Check if ordering is monotonic along PC1."""
            indices = [names.index(n) for n in order if n in names]
            if len(indices) < 3:
                return False, 0.0

            positions = X_pca[indices, 0]
            expected = np.arange(len(indices))

            # Spearman correlation
            from scipy.stats import spearmanr

            corr, _ = spearmanr(positions, expected)

            # Check monotonicity
            diffs = np.diff(positions)
            monotonic = np.all(diffs > 0) or np.all(diffs < 0)

            return monotonic, abs(corr) if not np.isnan(corr) else 0.0

        power_mono, power_corr = check_monotonicity(power_order)
        kinship_mono, kinship_corr = check_monotonicity(kinship_order)
        formality_mono, formality_corr = check_monotonicity(formality_order)

        return GradientConsistency(
            power_monotonic=power_mono,
            power_correlation=power_corr,
            kinship_monotonic=kinship_mono,
            kinship_correlation=kinship_corr,
            formality_monotonic=formality_mono,
            formality_correlation=formality_corr,
        )

    def _analyze_power_gradient(
        self,
        activations: dict[str, any],
        names: list[str],
        X_pca: np.ndarray,
    ) -> PowerGradientResult:
        """Analyze the power hierarchy axis specifically."""

        # Get power anchors
        power_anchors = [a for a in SOCIAL_PRIME_ATLAS if a.axis == SocialAxis.POWER]
        power_names = [a.name for a in power_anchors if a.name in names]
        power_levels = {a.name: a.level for a in power_anchors}

        if len(power_names) < 3:
            return PowerGradientResult(
                power_axis_detected=False,
                power_direction=np.zeros(1),
                status_correlation=0.0,
                high_status_anchors=[],
                low_status_anchors=[],
            )

        # Compute correlation between PC position and expected level
        indices = [names.index(n) for n in power_names]
        positions = X_pca[indices, 0]
        expected_levels = [power_levels[n] for n in power_names]

        correlation = np.corrcoef(positions, expected_levels)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Compute power direction vector
        low_status = [n for n in power_names if power_levels[n] <= 2]
        high_status = [n for n in power_names if power_levels[n] >= 4]

        if low_status and high_status:
            low_centroid = np.mean(
                [self.backend.to_numpy(activations[n]) for n in low_status], axis=0
            )
            high_centroid = np.mean(
                [self.backend.to_numpy(activations[n]) for n in high_status], axis=0
            )
            power_direction = high_centroid - low_centroid
            power_direction = power_direction / (np.linalg.norm(power_direction) + 1e-8)
        else:
            power_direction = np.zeros(1)

        return PowerGradientResult(
            power_axis_detected=abs(correlation) > 0.5,
            power_direction=power_direction,
            status_correlation=correlation,
            high_status_anchors=high_status,
            low_status_anchors=low_status,
        )

    def full_analysis(self, activations: dict[str, any]) -> SocialGeometryReport:
        """
        Run complete social geometry analysis.

        Args:
            activations: Dict mapping anchor names to activation vectors

        Returns:
            SocialGeometryReport with all metrics
        """
        names, X = self._to_numpy(activations)
        X_pca, variance = self._compute_pca(X, n_components=5)

        # Compute all metrics
        axis_ortho = self._compute_axis_orthogonality(activations)
        gradient = self._compute_gradient_consistency(names, X_pca)
        power = self._analyze_power_gradient(activations, names, X_pca)

        # Compute overall score
        # Weighted combination of:
        # - Axis orthogonality (30%)
        # - Gradient consistency (40%)
        # - Power detection (30%)
        ortho_score = axis_ortho.mean_orthogonality
        gradient_score = (
            gradient.power_correlation
            + gradient.kinship_correlation
            + gradient.formality_correlation
        ) / 3
        power_score = abs(power.status_correlation)

        social_score = 0.3 * ortho_score + 0.4 * gradient_score + 0.3 * power_score

        return SocialGeometryReport(
            has_social_manifold=social_score > 0.4,
            social_manifold_score=social_score,
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            power_gradient=power,
            principal_components_variance=variance.tolist(),
            anchor_count=len(names),
        )


__all__ = [
    "SocialAxis",
    "SocialCategory",
    "SocialAnchor",
    "SOCIAL_PRIME_ATLAS",
    "AxisOrthogonality",
    "GradientConsistency",
    "PowerGradientResult",
    "SocialGeometryReport",
    "SocialGeometryAnalyzer",
]
