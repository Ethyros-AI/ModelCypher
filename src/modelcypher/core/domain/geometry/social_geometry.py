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
- Power axis: slave → servant → citizen → noble → emperor
- Kinship axis: enemy → stranger → acquaintance → friend → family
- Formality axis: hey → hi → hello → greetings → salutations

Reference: Emergent Social Geometry (ModelCypher 2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.atlas_protocols import (
    axis_key,
)
from modelcypher.core.domain.geometry.atlas_registry import get_social_concepts
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_nan,
    power_iteration_eigh,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_cosine_batch,
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

_AXIS_POWER = "power"
_AXIS_KINSHIP = "kinship"
_AXIS_FORMALITY = "formality"


@dataclass(frozen=True)
class AxisOrthogonality:
    """Measures how independent the social axes are."""

    power_kinship: float  # 1.0 = exactly orthogonal
    power_formality: float
    kinship_formality: float
    mean_orthogonality: float


@dataclass(frozen=True)
class GradientConsistency:
    """Measures whether anchors form monotonic gradients along axes."""

    power_monotonic: bool
    power_correlation: float  # Geodesic correlation on rank ordering
    kinship_monotonic: bool
    kinship_correlation: float
    formality_monotonic: bool
    formality_correlation: float


@dataclass(frozen=True)
class PowerGradientResult:
    """Analysis of the power hierarchy axis."""

    power_axis_detected: bool
    power_direction: "Array"  # Unit vector pointing "up" in status
    status_correlation: float  # Geodesic correlation between position and expected status
    high_status_anchors: tuple[str, ...]
    low_status_anchors: tuple[str, ...]


@dataclass(frozen=True)
class SocialGeometryReport:
    """Complete social geometry analysis report."""

    social_manifold_score: float  # 0-1, overall quality
    axis_orthogonality: AxisOrthogonality
    gradient_consistency: GradientConsistency
    power_gradient: PowerGradientResult
    principal_components_variance: tuple[float, ...]
    anchor_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
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
                "high_status_anchors": list(self.power_gradient.high_status_anchors),
                "low_status_anchors": list(self.power_gradient.low_status_anchors),
            },
            "principal_components_variance": list(self.principal_components_variance),
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

    @staticmethod
    def _geodesic_correlation(
        backend: "Backend", values_a: list[float], values_b: list[float]
    ) -> float:
        if len(values_a) != len(values_b) or len(values_a) < 2:
            return 0.0
        a_arr = backend.array(values_a)
        b_arr = backend.array(values_b)
        mean_a = backend.mean(a_arr)
        mean_b = backend.mean(b_arr)
        centered_a = a_arr - mean_a
        centered_b = b_arr - mean_b
        centered_a_mat = backend.reshape(centered_a, (1, -1))
        centered_b_mat = backend.reshape(centered_b, (1, -1))
        cos_arr, _ = geodesic_pairwise_metrics(centered_a_mat, centered_b_mat, backend)
        backend.eval(cos_arr)
        if cos_arr.size == 0:
            return 0.0
        corr = float(backend.to_scalar(cos_arr[0]))
        return 0.0 if is_nan(corr, backend) else corr

    def _to_array(self, activations: dict[str, any]) -> tuple[list[str], "Array"]:
        """Convert activation dict to backend array matrix."""
        names = list(activations.keys())
        vectors = [self.backend.array(activations[n]) for n in names]
        # Stack vectors along axis 0
        reshaped = [self.backend.reshape(v, (1, -1)) for v in vectors]
        stacked = self.backend.concatenate(reshaped, axis=0)
        self.backend.eval(stacked)
        return names, stacked

    def _compute_pca(self, X: "Array", n_components: int | None = None) -> tuple["Array", "Array"]:
        """Compute PCA using backend operations.

        Args:
            X: Data matrix [n_samples, n_features]
            n_components: Number of components to retain. If None, auto-derives
                         using effective dimensionality: ceil(d_eff) where
                         d_eff = (Σλ)²/Σλ² (Rényi entropy-based effective rank).

        Returns:
            Tuple of (X_pca, variance_explained)
        """
        backend = self.backend

        # Center the data
        X_mean = backend.mean(X, axis=0)
        X_centered = X - X_mean

        # Compute covariance matrix: (X.T @ X) / (n - 1)
        n = X.shape[0]
        X_t = backend.transpose(X_centered)
        cov = backend.matmul(X_t, X_centered) / max(n - 1, 1)

        # Eigendecomposition (geodesic - GPU-only, float32 required)
        cov = backend.astype(cov, "float32")
        n_cov = int(cov.shape[0])
        eigenvalues, eigenvectors = power_iteration_eigh(backend, cov, k=n_cov)
        backend.eval(eigenvalues, eigenvectors)

        # power_iteration_eigh returns eigenvalues in descending order.
        # Auto-derive n_components using effective dimensionality if not specified
        # Formula: n_components = ceil(d_eff) where d_eff = (Σλ)²/Σλ²
        if n_components is None:
            n_components = self._effective_dim_components(eigenvalues)

        idx_top = slice(0, n_components)

        # Project data onto top components
        eigenvectors_subset = eigenvectors[:, idx_top]
        X_pca = backend.matmul(X_centered, eigenvectors_subset)

        # Variance explained
        total_var = backend.sum(eigenvalues)
        top_eigs = eigenvalues[idx_top]
        variance_explained = top_eigs / total_var

        backend.eval(X_pca, variance_explained)
        return X_pca, variance_explained

    def _effective_dim_components(self, eigenvalues: "Array") -> int:
        """Determine number of components using effective dimensionality.

        Formula: n_components = ceil(d_eff) where d_eff = (Σλ)² / Σλ²

        This is the Rényi entropy-based effective rank - a mathematically
        derived measure of intrinsic dimensionality, not an arbitrary threshold.

        Args:
            eigenvalues: Sorted eigenvalues (descending order)

        Returns:
            Number of components to retain, minimum 1.
        """
        backend = self.backend

        # Compute effective dimensionality: d_eff = (Σλ)² / Σλ²
        sum_eigenvals = backend.sum(eigenvalues)
        sum_sq_eigenvals = backend.sum(eigenvalues * eigenvalues)
        backend.eval(sum_eigenvals, sum_sq_eigenvals)

        sum_val = float(backend.to_scalar(sum_eigenvals))
        sum_sq_val = float(backend.to_scalar(sum_sq_eigenvals))

        # Use √(machine_epsilon) for numerical stability
        eps = sqrt_scalar(regularization_epsilon(backend, eigenvalues), backend)

        if sum_sq_val < eps:
            return 1

        d_eff = (sum_val * sum_val) / (sum_sq_val + eps)

        # n_components = ceil(d_eff)
        import math
        n_components = int(math.ceil(d_eff))

        # Clamp to valid range
        n_eigs = int(eigenvalues.shape[0])
        return max(1, min(n_components, n_eigs))

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
            cos_arr = geodesic_cosine_batch(a, backend.reshape(b, (1, -1)), backend)
            backend.eval(cos_arr)
            cos_val = float(backend.to_scalar(cos_arr))
            return 1.0 - abs(cos_val)

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

            # Use take + tolist() for O(1) extraction instead of O(n) scalar extractions
            pc1_col = X_pca[:, 0]
            gathered = backend.take(pc1_col, backend.array(indices))
            backend.eval(gathered)
            positions = [float(x) for x in backend.tolist(gathered)]
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

            # Spearman correlation using geodesic correlation on ranks
            corr = SocialGeometryAnalyzer._geodesic_correlation(backend, pos_ranks, exp_ranks)

            # Check monotonicity
            diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

            return monotonic, abs(corr) if not is_nan(corr, backend) else 0.0

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
        backend = self.backend

        # Get power anchors
        power_anchors = [
            a for a in get_social_concepts() if axis_key(a.axis) == _AXIS_POWER
        ]
        power_names = [a.id for a in power_anchors if a.id in names]
        power_levels = {a.id: a.level for a in power_anchors}

        if len(power_names) < 3:
            return PowerGradientResult(
                power_axis_detected=False,
                power_direction=backend.zeros((1,)),
                status_correlation=0.0,
                high_status_anchors=(),
                low_status_anchors=(),
            )

        # Compute correlation between PC position and expected level
        indices = [names.index(n) for n in power_names]
        # Use take + tolist() for O(1) extraction instead of O(n) scalar extractions
        pc1_col = X_pca[:, 0]
        gathered = backend.take(pc1_col, backend.array(indices))
        backend.eval(gathered)
        positions = [float(x) for x in backend.tolist(gathered)]
        expected_levels = [power_levels[n] for n in power_names]

        # Geodesic correlation between positions and expected levels
        correlation = SocialGeometryAnalyzer._geodesic_correlation(
            backend, positions, expected_levels
        )

        # Compute power direction vector
        low_status = [n for n in power_names if power_levels[n] <= 2]
        high_status = [n for n in power_names if power_levels[n] >= 4]

        if low_status and high_status:
            # Use Fréchet mean for centroids (Riemannian center of mass)
            from modelcypher.core.domain.geometry.riemannian_utils import (
                RiemannianGeometry,
            )

            rg = RiemannianGeometry(backend)

            # Compute low-status centroid via Fréchet mean
            low_vecs = [backend.reshape(backend.array(activations[n]), (1, -1)) for n in low_status]
            low_activations = backend.concatenate(low_vecs, axis=0)
            low_arr = backend.astype(low_activations, "float32")
            low_tol = regularization_epsilon(backend, low_arr)
            low_result = rg.frechet_mean(
                low_arr, tolerance=low_tol  # max_iterations auto-derived from n
            )
            backend.eval(low_result.mean)
            low_centroid = low_result.mean

            # Compute high-status centroid via Fréchet mean
            high_vecs = [backend.reshape(backend.array(activations[n]), (1, -1)) for n in high_status]
            high_activations = backend.concatenate(high_vecs, axis=0)
            high_arr = backend.astype(high_activations, "float32")
            high_tol = regularization_epsilon(backend, high_arr)
            high_result = rg.frechet_mean(
                high_arr, tolerance=high_tol  # max_iterations auto-derived from n
            )
            backend.eval(high_result.mean)
            high_centroid = high_result.mean

            # Direction vector in tangent space (approximation)
            power_direction = high_centroid - low_centroid
            norm = geodesic_norms(backend.reshape(power_direction, (1, -1)), backend)
            backend.eval(norm)
            norm_val = float(backend.to_scalar(norm[0]))
            div_eps = division_epsilon(backend, power_direction)
            if norm_val > div_eps:
                power_direction = power_direction / norm_val
            else:
                power_direction = backend.zeros_like(power_direction)
        else:
            power_direction = backend.zeros((1,))

        return PowerGradientResult(
            # Power axis detected if any measurable correlation exists
            power_axis_detected=abs(correlation) > 0,
            power_direction=power_direction,
            status_correlation=correlation,
            high_status_anchors=tuple(high_status),
            low_status_anchors=tuple(low_status),
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
        # n_components auto-derived via scree test (95% variance threshold)
        X_pca, variance = self._compute_pca(X)

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
            social_manifold_score=social_score,
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            power_gradient=power,
            # Use native tolist() for O(1) extraction
            principal_components_variance=tuple(float(x) for x in self.backend.tolist(variance)),
            anchor_count=len(names),
        )


__all__ = [
    "AxisOrthogonality",
    "GradientConsistency",
    "PowerGradientResult",
    "SocialGeometryReport",
    "SocialGeometryAnalyzer",
]
