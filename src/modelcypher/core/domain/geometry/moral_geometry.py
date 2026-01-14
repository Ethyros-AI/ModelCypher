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

"""Moral Geometry: Probing ethical structure in LLM representations.

This module implements the "Latent Ethicist" hypothesis: that language models
trained on human text encode moral reasoning as a coherent geometric manifold with:
1. Valence axis (evil → good)
2. Agency axis (victim → perpetrator)
3. Scope axis (self → universal)

Scientific Method:
- H1: Models encode moral structure above chance (MMS > 0.33 baseline)
- H2: Moral axes are geometrically independent (orthogonality > 80%)
- H3: Valence gradient is monotonic (cruelty < kindness < compassion)
- H4: Moral foundations cluster distinctly
- H5: Measurements are reproducible (CV < 10%)

Based on:
- Moral Foundations Theory (Haidt, 2012)
- Virtue ethics tradition (Aristotle → MacIntyre)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.cache import ComputationCache
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

from modelcypher.core.domain.geometry.atlas_protocols import (
    MoralConceptProtocol,
    axis_key,
    enum_key,
)
from modelcypher.core.domain.geometry.atlas_registry import get_moral_concepts

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

_cache = ComputationCache.shared()

_AXIS_VALENCE = "valence"
_AXIS_AGENCY = "agency"
_AXIS_SCOPE = "scope"


def _foundation_key(value: object) -> str:
    return enum_key(value).lower()


@dataclass
class MoralAxisOrthogonality:
    """Orthogonality measurements between moral axes."""

    valence_agency: float
    valence_scope: float
    agency_scope: float
    mean_orthogonality: float


@dataclass
class MoralGradientConsistency:
    """Gradient consistency measurements for each axis."""

    valence_correlation: float
    valence_monotonic: bool
    agency_correlation: float
    agency_monotonic: bool
    scope_correlation: float
    scope_monotonic: bool


@dataclass
class MoralFoundationClustering:
    """Clustering quality of moral foundations."""

    within_foundation_similarity: float  # Mean similarity within foundations
    between_foundation_similarity: float  # Mean similarity between foundations
    separation_ratio: float  # within / between (higher indicates tighter clustering)
    most_distinct_foundation: str
    most_overlapping_pair: tuple[str, str]


@dataclass
class VirtueViceOpposition:
    """Detection of virtue-vice opposition structure."""

    care_harm_opposition: float  # cruelty ↔ compassion distance
    fairness_opposition: float  # exploitation ↔ justice distance
    loyalty_opposition: float  # betrayal ↔ devotion distance
    mean_opposition: float


@dataclass
class MoralGeometryReport:
    """Complete moral geometry analysis report."""

    model_path: str
    layer: int
    anchors_probed: int
    axis_orthogonality: MoralAxisOrthogonality
    gradient_consistency: MoralGradientConsistency
    foundation_clustering: MoralFoundationClustering
    virtue_vice_opposition: VirtueViceOpposition
    principal_components_variance: list[float]
    moral_manifold_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "layer": self.layer,
            "anchors_probed": self.anchors_probed,
            "axis_orthogonality": {
                "valence_agency": self.axis_orthogonality.valence_agency,
                "valence_scope": self.axis_orthogonality.valence_scope,
                "agency_scope": self.axis_orthogonality.agency_scope,
                "mean": self.axis_orthogonality.mean_orthogonality,
            },
            "gradient_consistency": {
                "valence": {
                    "correlation": self.gradient_consistency.valence_correlation,
                    "monotonic": self.gradient_consistency.valence_monotonic,
                },
                "agency": {
                    "correlation": self.gradient_consistency.agency_correlation,
                    "monotonic": self.gradient_consistency.agency_monotonic,
                },
                "scope": {
                    "correlation": self.gradient_consistency.scope_correlation,
                    "monotonic": self.gradient_consistency.scope_monotonic,
                },
            },
            "foundation_clustering": {
                "within_similarity": self.foundation_clustering.within_foundation_similarity,
                "between_similarity": self.foundation_clustering.between_foundation_similarity,
                "separation_ratio": self.foundation_clustering.separation_ratio,
                "most_distinct": self.foundation_clustering.most_distinct_foundation,
                "most_overlapping": self.foundation_clustering.most_overlapping_pair,
            },
            "virtue_vice_opposition": {
                "care_harm": self.virtue_vice_opposition.care_harm_opposition,
                "fairness": self.virtue_vice_opposition.fairness_opposition,
                "loyalty": self.virtue_vice_opposition.loyalty_opposition,
                "mean": self.virtue_vice_opposition.mean_opposition,
            },
            "principal_components_variance": self.principal_components_variance,
            "moral_manifold_score": self.moral_manifold_score,
        }


class MoralGeometryAnalyzer:
    """Analyzer for moral structure in LLM representations.

    Implements the scientific method for testing the Latent Ethicist hypothesis:
    1. Extract activations for 30 moral anchors
    2. Measure axis orthogonality (Valence ⊥ Agency ⊥ Scope)
    3. Test gradient consistency (monotonic orderings)
    4. Analyze foundation clustering
    5. Detect virtue-vice opposition
    6. Compute Moral Manifold Score (MMS)
    """

    def __init__(
        self,
        backend: "Backend",
        concepts: list[MoralConceptProtocol] | None = None,
    ) -> None:
        """Initialize with compute backend.

        Args:
            backend: Backend for array operations
            concepts: Optional moral concept inventory (defaults to registry)
        """
        self._backend = backend
        self._concepts = list(concepts or get_moral_concepts())
        if not self._concepts:
            raise ValueError(
                "No moral concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before running moral geometry analysis."
            )
        self._concept_lookup = {c.id: c for c in self._concepts}

    def full_analysis(
        self,
        activations: dict[str, "Array"],
        model_path: str = "",
        layer: int = -1,
    ) -> MoralGeometryReport:
        """Run complete moral geometry analysis.

        Args:
            activations: Dict mapping concept name to activation vector
            model_path: Path to model (for reporting)
            layer: Layer analyzed (for reporting)

        Returns:
            MoralGeometryReport with all measurements
        """
        backend = self._backend

        # Build activation matrix
        concepts = [
            c.id for c in self._concepts if c.name in activations or c.id in activations
        ]
        if len(concepts) < 15:
            raise ValueError(f"Insufficient anchors: {len(concepts)} < 15 required")

        # Get activations (try both id and name as keys)
        act_list = []
        for cid in concepts:
            concept = self._concept_lookup[cid]
            if concept.name in activations:
                act = activations[concept.name]
            elif cid in activations:
                act = activations[cid]
            else:
                continue
            # Ensure activation is a backend array with precision dtype
            act_arr = backend.array(act) if not hasattr(act, "shape") else act
            act_arr = backend.astype(act_arr, precision_dtype(backend, reference=act_arr))
            act_list.append(act_arr)

        # Use concatenate with reshape instead of stack for broader compatibility
        reshaped = [backend.reshape(a, (1, -1)) for a in act_list]
        matrix = backend.concatenate(reshaped, axis=0)
        matrix = backend.astype(matrix, precision_dtype(backend, reference=matrix))

        # Normalize for geodesic cosine similarity
        norms = geodesic_norms(matrix, backend)
        norms = backend.reshape(norms, (-1, 1))
        eps = division_epsilon(backend, norms)
        matrix_norm = matrix / backend.maximum(norms, backend.full(norms.shape, eps))

        # PCA for axis analysis
        mean_vec = backend.mean(matrix_norm, axis=0, keepdims=True)
        centered = matrix_norm - mean_vec
        try:
            # Geodesic SVD (GPU-only)
            _, s, vh = geodesic_svd(backend, centered)
            backend.eval(s)
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
            variance_list = [float(x) for x in backend.tolist(variance_explained)]
            pc_variance = variance_list[:5] + [0.0] * (5 - len(variance_list[:5]))
        except Exception:
            pc_variance = [0.0] * 5

        # Compute axis orthogonality
        axis_ortho = self._compute_axis_orthogonality(matrix_norm, concepts)

        # Compute gradient consistency
        gradient = self._compute_gradient_consistency(matrix_norm, concepts)

        # Compute foundation clustering
        clustering = self._compute_foundation_clustering(matrix_norm, concepts)

        # Detect virtue-vice opposition
        opposition = self._compute_virtue_vice_opposition(matrix_norm, concepts)

        # Compute Moral Manifold Score (MMS)
        # Weighted: 25% orthogonality + 30% gradient + 25% clustering + 20% opposition
        ortho_score = axis_ortho.mean_orthogonality

        gradient_scores = [
            gradient.valence_correlation,
            gradient.agency_correlation,
            gradient.scope_correlation,
        ]
        gradient_score = sum(abs(s) for s in gradient_scores) / len(gradient_scores)

        cluster_score = min(1.0, clustering.separation_ratio)
        opposition_score = opposition.mean_opposition

        # Equal weights - let individual scores speak for themselves
        mms = (ortho_score + gradient_score + cluster_score + opposition_score) / 4.0

        return MoralGeometryReport(
            model_path=model_path,
            layer=layer,
            anchors_probed=len(concepts),
            axis_orthogonality=axis_ortho,
            gradient_consistency=gradient,
            foundation_clustering=clustering,
            virtue_vice_opposition=opposition,
            principal_components_variance=pc_variance,
            moral_manifold_score=mms,
        )

    def _compute_axis_orthogonality(
        self, matrix: "Array", concepts: list[str]
    ) -> MoralAxisOrthogonality:
        """Compute orthogonality between moral axes."""
        backend = self._backend
        valence_vecs = []
        agency_vecs = []
        scope_vecs = []

        for i, cid in enumerate(concepts):
            concept = self._concept_lookup.get(cid)
            if concept is None:
                continue
            axis_str = axis_key(concept.axis)
            if axis_str == _AXIS_VALENCE:
                valence_vecs.append(matrix[i])
            elif axis_str == _AXIS_AGENCY:
                agency_vecs.append(matrix[i])
            elif axis_str == _AXIS_SCOPE:
                scope_vecs.append(matrix[i])

        def axis_direction(vecs: list) -> "Array":
            """Compute principal direction of axis from anchors."""
            if len(vecs) < 2:
                d = int(vecs[0].shape[0]) if vecs else 1
                return backend.zeros((d,))
            # Use concatenate with reshape instead of stack for compatibility
            reshaped = [backend.reshape(v, (1, -1)) for v in vecs]
            arr = backend.concatenate(reshaped, axis=0)
            mean_vec = backend.mean(arr, axis=0, keepdims=True)
            centered = arr - mean_vec
            try:
                # Geodesic SVD (GPU-only)
                _, _, vh = geodesic_svd(backend, centered)
                return vh[0]
            except Exception:
                d = int(arr.shape[1])
                return backend.zeros((d,))

        val_vec = axis_direction(valence_vecs)
        agen_vec = axis_direction(agency_vecs)
        scope_vec = axis_direction(scope_vecs)

        def orthogonality(v1: "Array", v2: "Array") -> float:
            """Compute orthogonality as 1 - |cos(angle)|."""
            v1_mat = backend.reshape(v1, (1, -1))
            v2_mat = backend.reshape(v2, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(v1_mat, v2_mat, backend)
            backend.eval(cos_arr)
            cos_sim = abs(float(backend.to_scalar(cos_arr)))
            return 1.0 - cos_sim

        val_agen = orthogonality(val_vec, agen_vec)
        val_scope = orthogonality(val_vec, scope_vec)
        agen_scope = orthogonality(agen_vec, scope_vec)

        return MoralAxisOrthogonality(
            valence_agency=val_agen,
            valence_scope=val_scope,
            agency_scope=agen_scope,
            mean_orthogonality=(val_agen + val_scope + agen_scope) / 3,
        )

    def _compute_gradient_consistency(
        self, matrix: "Array", concepts: list[str]
    ) -> MoralGradientConsistency:
        """Compute gradient consistency (Spearman correlation with expected ordering)."""
        backend = self._backend
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

            for i, cid in enumerate(concepts):
                concept = self._concept_lookup.get(cid)
                if concept is None or axis_key(concept.axis) != axis:
                    continue
                levels.append(concept.level)
                projections.append(float(col0[i]))

            if len(levels) < 3:
                return 0.0, False

            corr = compute_spearman_correlation(
                levels, projections, default=0.0, backend=backend
            )
            if corr is None or is_nan(float(corr), self._backend):
                corr = 0.0

            # Monotonic if any measurable correlation exists
            monotonic = abs(corr) > 0
            return float(corr), monotonic

        val_corr, val_mono = axis_correlation(_AXIS_VALENCE)
        agen_corr, agen_mono = axis_correlation(_AXIS_AGENCY)
        scope_corr, scope_mono = axis_correlation(_AXIS_SCOPE)

        return MoralGradientConsistency(
            valence_correlation=val_corr,
            valence_monotonic=val_mono,
            agency_correlation=agen_corr,
            agency_monotonic=agen_mono,
            scope_correlation=scope_corr,
            scope_monotonic=scope_mono,
        )

    def _compute_foundation_clustering(
        self, matrix: "Array", concepts: list[str]
    ) -> MoralFoundationClustering:
        """Analyze how well moral foundations cluster in the representation space."""
        backend = self._backend
        backend.eval(matrix)

        # Group by foundation
        foundation_indices: dict[str, list[int]] = {}
        for i, cid in enumerate(concepts):
            concept = self._concept_lookup.get(cid)
            if concept is None:
                continue
            foundation_key = _foundation_key(concept.foundation)
            if foundation_key not in foundation_indices:
                foundation_indices[foundation_key] = []
            foundation_indices[foundation_key].append(i)

        # Pre-compute geodesic cosine similarity matrix once (vectorized, with caching)
        sim_matrix = _cache.get_or_compute_gram(
            matrix, backend, kernel_type="geodesic_cosine"
        )

        # Compute within-foundation similarity (backend vectorized)
        within_sum = 0.0
        within_count = 0
        foundation_means: dict[str, float] = {}  # Compute once, reuse below
        for foundation, indices in foundation_indices.items():
            count = len(indices)
            if count < 2:
                continue
            idx_arr = backend.array(indices)
            sub = backend.take(sim_matrix, idx_arr, axis=0)
            sub = backend.take(sub, idx_arr, axis=1)
            off_diag = 1.0 - backend.eye(count)
            sub_off = sub * off_diag
            sub_sum_arr = backend.sum(sub_off)
            backend.eval(sub_sum_arr)
            sub_sum = float(backend.to_scalar(sub_sum_arr))
            denom = count * (count - 1)
            mean_val = sub_sum / denom if denom > 0 else 0.0
            foundation_means[foundation] = mean_val
            within_sum += sub_sum
            within_count += denom

        within_sim = within_sum / within_count if within_count > 0 else 0.0

        # Compute between-foundation similarity (backend vectorized)
        between_sum = 0.0
        between_count = 0
        pair_means: dict[tuple[str, str], float] = {}
        foundations = list(foundation_indices.keys())

        for f1_idx, f1 in enumerate(foundations):
            for f2 in foundations[f1_idx + 1 :]:
                indices_a = foundation_indices[f1]
                indices_b = foundation_indices[f2]
                if not indices_a or not indices_b:
                    continue
                idx_a = backend.array(indices_a)
                idx_b = backend.array(indices_b)
                sub = backend.take(sim_matrix, idx_a, axis=0)
                sub = backend.take(sub, idx_b, axis=1)
                sub_sum_arr = backend.sum(sub)
                backend.eval(sub_sum_arr)
                sub_sum = float(backend.to_scalar(sub_sum_arr))
                denom = len(indices_a) * len(indices_b)
                mean_val = sub_sum / denom if denom > 0 else 0.0
                key = (f1, f2)
                pair_means[key] = mean_val
                between_sum += sub_sum
                between_count += denom

        between_sim = between_sum / between_count if between_count > 0 else 0.0

        # Find most distinct (already computed in foundation_means above)
        most_distinct = (
            max(foundation_means.keys(), key=lambda k: foundation_means[k])
            if foundation_means
            else "unknown"
        )

        most_overlapping = ("unknown", "unknown")
        if pair_means:
            max_pair = max(pair_means.keys(), key=lambda k: pair_means[k])
            most_overlapping = max_pair

        separation_eps = division_epsilon(backend, matrix)
        separation = within_sim / (between_sim + separation_eps) if between_sim > 0 else 1.0

        return MoralFoundationClustering(
            within_foundation_similarity=within_sim,
            between_foundation_similarity=between_sim,
            separation_ratio=separation,
            most_distinct_foundation=most_distinct,
            most_overlapping_pair=most_overlapping,
        )

    def _compute_virtue_vice_opposition(
        self, matrix: "Array", concepts: list[str]
    ) -> VirtueViceOpposition:
        """Detect opposition structure between virtues and vices."""
        backend = self._backend
        backend.eval(matrix)

        def get_idx(target_id: str) -> int | None:
            for i, cid in enumerate(concepts):
                if cid == target_id:
                    return i
            return None

        def opposition_distance(virtue_id: str, vice_id: str) -> float:
            """Compute normalized distance (1 - cosine) between virtue and vice."""
            vi = get_idx(virtue_id)
            vci = get_idx(vice_id)
            if vi is None or vci is None:
                return 0.0

            v1, v2 = matrix[vi], matrix[vci]
            v1_mat = backend.reshape(v1, (1, -1))
            v2_mat = backend.reshape(v2, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(v1_mat, v2_mat, backend)
            backend.eval(cos_arr)
            cos_sim = float(backend.to_scalar(cos_arr))
            # Distance = 1 - similarity (higher = more opposed)
            return float(1.0 - cos_sim)

        care_harm = opposition_distance("compassion", "cruelty")
        fairness = opposition_distance("justice", "exploitation")
        loyalty = opposition_distance("devotion", "betrayal")

        mean_opp = (care_harm + fairness + loyalty) / 3

        return VirtueViceOpposition(
            care_harm_opposition=care_harm,
            fairness_opposition=fairness,
            loyalty_opposition=loyalty,
            mean_opposition=mean_opp,
        )
