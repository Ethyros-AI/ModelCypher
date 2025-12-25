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

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

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
    power_direction: "Array"  # Unit vector pointing "up" in status
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

    def _to_array(self, activations: dict[str, any]) -> tuple[list[str], "Array"]:
        """Convert activation dict to backend array matrix."""
        names = list(activations.keys())
        vectors = [self.backend.array(activations[n]) for n in names]
        # Stack vectors along axis 0
        reshaped = [self.backend.reshape(v, (1, -1)) for v in vectors]
        stacked = self.backend.concatenate(reshaped, axis=0)
        self.backend.eval(stacked)
        return names, stacked

    def _compute_pca(self, X: "Array", n_components: int = 5) -> tuple["Array", "Array"]:
        """Compute PCA using backend operations."""
        backend = self.backend

        # Center the data
        X_mean = backend.mean(X, axis=0)
        X_centered = X - X_mean

        # Compute covariance matrix: (X.T @ X) / (n - 1)
        n = X.shape[0]
        X_t = backend.transpose(X_centered)
        cov = backend.matmul(X_t, X_centered) / max(n - 1, 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = backend.eigh(cov)
        backend.eval(eigenvalues, eigenvectors)

        # Sort descending
        idx = backend.argsort(eigenvalues)
        # Reverse the indices
        idx_np = backend.to_numpy(idx)[::-1].tolist()
        eigenvalues_sorted = backend.array([float(backend.to_numpy(eigenvalues)[i]) for i in idx_np])
        eigenvectors_np = backend.to_numpy(eigenvectors)
        eigenvectors_sorted = backend.array([[eigenvectors_np[i, j] for j in idx_np] for i in range(eigenvectors_np.shape[0])])

        # Project data
        eigenvectors_subset = backend.array([[eigenvectors_np[i, idx_np[j]] for j in range(n_components)] for i in range(eigenvectors_np.shape[0])])
        X_pca = backend.matmul(X_centered, eigenvectors_subset)

        # Variance explained
        eigenvalues_np = backend.to_numpy(eigenvalues_sorted)
        total_var = sum(eigenvalues_np)
        variance_explained = backend.array([eigenvalues_np[i] / total_var for i in range(n_components)])

        backend.eval(X_pca, variance_explained)
        return X_pca, variance_explained

    def _compute_axis_orthogonality(
        self,
        activations: dict[str, any],
    ) -> AxisOrthogonality:
        """Compute orthogonality between social axes."""
        backend = self.backend

        def get_axis_vector(low_anchor: str, high_anchor: str) -> "Array":
            low_val = activations.get(low_anchor)
            high_val = activations.get(high_anchor)
            if low_val is None or high_val is None:
                return backend.zeros((1,))
            low = backend.array(low_val)
            high = backend.array(high_val)
            return high - low

        def cosine_orthogonality(a: "Array", b: "Array") -> float:
            """1 - |cos(a, b)| gives orthogonality."""
            norm_a = backend.norm(a)
            norm_b = backend.norm(b)
            backend.eval(norm_a, norm_b)
            norm_a_val = float(backend.to_numpy(norm_a))
            norm_b_val = float(backend.to_numpy(norm_b))
            if norm_a_val < 1e-8 or norm_b_val < 1e-8:
                return 0.0
            dot_prod = backend.sum(a * b)
            backend.eval(dot_prod)
            cos = float(backend.to_numpy(dot_prod)) / (norm_a_val * norm_b_val)
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
        X_pca: "Array",
    ) -> GradientConsistency:
        """Check if axes form monotonic gradients."""
        import math

        backend = self.backend

        # Define expected orderings
        power_order = ["slave", "servant", "citizen", "noble", "emperor"]
        kinship_order = ["enemy", "stranger", "acquaintance", "friend", "family"]
        formality_order = ["hey", "hi", "hello", "greetings", "salutations"]

        def check_monotonicity(order: list[str]) -> tuple[bool, float]:
            """Check if ordering is monotonic along PC1."""
            indices = [names.index(n) for n in order if n in names]
            if len(indices) < 3:
                return False, 0.0

            # Get positions from X_pca using backend slicing
            X_pca_np = backend.to_numpy(X_pca)
            positions = [float(X_pca_np[i, 0]) for i in indices]
            expected = list(range(len(indices)))

            # Spearman correlation (computed manually to avoid scipy)
            # Rank positions and expected values
            def rank(values):
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
                ranks = [0] * len(values)
                for rank_val, idx in enumerate(sorted_indices):
                    ranks[idx] = rank_val + 1
                return ranks

            pos_ranks = rank(positions)
            exp_ranks = rank(expected)

            # Spearman correlation = Pearson correlation of ranks
            n = len(positions)
            mean_pos = sum(pos_ranks) / n
            mean_exp = sum(exp_ranks) / n

            num = sum((pos_ranks[i] - mean_pos) * (exp_ranks[i] - mean_exp) for i in range(n))
            den_pos = math.sqrt(sum((pos_ranks[i] - mean_pos) ** 2 for i in range(n)))
            den_exp = math.sqrt(sum((exp_ranks[i] - mean_exp) ** 2 for i in range(n)))

            if den_pos < 1e-10 or den_exp < 1e-10:
                corr = 0.0
            else:
                corr = num / (den_pos * den_exp)

            # Check monotonicity
            diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

            return monotonic, abs(corr) if not math.isnan(corr) else 0.0

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
        X_pca: "Array",
    ) -> PowerGradientResult:
        """Analyze the power hierarchy axis specifically."""
        import math

        backend = self.backend

        # Get power anchors
        power_anchors = [a for a in SOCIAL_PRIME_ATLAS if a.axis == SocialAxis.POWER]
        power_names = [a.name for a in power_anchors if a.name in names]
        power_levels = {a.name: a.level for a in power_anchors}

        if len(power_names) < 3:
            return PowerGradientResult(
                power_axis_detected=False,
                power_direction=backend.zeros((1,)),
                status_correlation=0.0,
                high_status_anchors=[],
                low_status_anchors=[],
            )

        # Compute correlation between PC position and expected level
        indices = [names.index(n) for n in power_names]
        X_pca_np = backend.to_numpy(X_pca)
        positions = [float(X_pca_np[i, 0]) for i in indices]
        expected_levels = [power_levels[n] for n in power_names]

        # Pearson correlation
        n = len(positions)
        mean_pos = sum(positions) / n
        mean_exp = sum(expected_levels) / n
        num = sum((positions[i] - mean_pos) * (expected_levels[i] - mean_exp) for i in range(n))
        den_pos = math.sqrt(sum((positions[i] - mean_pos) ** 2 for i in range(n)))
        den_exp = math.sqrt(sum((expected_levels[i] - mean_exp) ** 2 for i in range(n)))

        if den_pos < 1e-10 or den_exp < 1e-10:
            correlation = 0.0
        else:
            correlation = num / (den_pos * den_exp)

        if math.isnan(correlation):
            correlation = 0.0

        # Compute power direction vector
        low_status = [n for n in power_names if power_levels[n] <= 2]
        high_status = [n for n in power_names if power_levels[n] >= 4]

        if low_status and high_status:
            # Use Fréchet mean for centroids (Riemannian center of mass)
            from modelcypher.core.domain.geometry.riemannian_geometry import (
                RiemannianGeometry,
            )

            rg = RiemannianGeometry(backend)

            # Compute low-status centroid via Fréchet mean
            low_vecs = [backend.reshape(backend.array(activations[n]), (1, -1)) for n in low_status]
            low_activations = backend.concatenate(low_vecs, axis=0)
            low_arr = backend.astype(low_activations, "float32")
            low_result = rg.frechet_mean(low_arr, max_iterations=50, tolerance=1e-5)
            backend.eval(low_result.mean)
            low_centroid = low_result.mean

            # Compute high-status centroid via Fréchet mean
            high_vecs = [backend.reshape(backend.array(activations[n]), (1, -1)) for n in high_status]
            high_activations = backend.concatenate(high_vecs, axis=0)
            high_arr = backend.astype(high_activations, "float32")
            high_result = rg.frechet_mean(high_arr, max_iterations=50, tolerance=1e-5)
            backend.eval(high_result.mean)
            high_centroid = high_result.mean

            # Direction vector in tangent space (approximation)
            power_direction = high_centroid - low_centroid
            norm = backend.norm(power_direction)
            backend.eval(norm)
            norm_val = float(backend.to_numpy(norm))
            if norm_val > 1e-8:
                power_direction = power_direction / norm
            else:
                power_direction = backend.zeros_like(power_direction)
        else:
            power_direction = backend.zeros((1,))

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
        names, X = self._to_array(activations)
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
            principal_components_variance=self.backend.to_numpy(variance).tolist(),
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
