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
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

from modelcypher.core.domain.agents.moral_atlas import (
    ALL_MORAL_PROBES,
    MoralAxis,
    MoralFoundation,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


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
    separation_ratio: float  # within / between (higher = better clustering)
    most_distinct_foundation: str
    most_overlapping_pair: tuple[str, str]


@dataclass
class VirtueViceOpposition:
    """Detection of virtue-vice opposition structure."""

    care_harm_opposition: float  # cruelty ↔ compassion distance
    fairness_opposition: float  # exploitation ↔ justice distance
    loyalty_opposition: float  # betrayal ↔ devotion distance
    mean_opposition: float
    opposition_detected: bool  # True if mean > 0.5


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
    has_moral_manifold: bool
    verdict: str

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
                "detected": self.virtue_vice_opposition.opposition_detected,
            },
            "principal_components_variance": self.principal_components_variance,
            "moral_manifold_score": self.moral_manifold_score,
            "has_moral_manifold": self.has_moral_manifold,
            "verdict": self.verdict,
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

    def __init__(self, backend: "Backend") -> None:
        """Initialize with compute backend.

        Args:
            backend: Backend for array operations
        """
        self._backend = backend
        self._concept_lookup = {c.id: c for c in ALL_MORAL_PROBES}

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
        concepts = [c.id for c in ALL_MORAL_PROBES if c.name in activations or c.id in activations]
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
            # Ensure activation is a backend array and float32
            act_arr = backend.array(act) if not hasattr(act, "shape") else act
            act_arr = backend.astype(act_arr, "float32")
            act_list.append(act_arr)

        # Use concatenate with reshape instead of stack for better compatibility
        reshaped = [backend.reshape(a, (1, -1)) for a in act_list]
        matrix = backend.concatenate(reshaped, axis=0)
        matrix = backend.astype(matrix, "float32")

        # Normalize for cosine similarity
        norms = backend.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)

        # PCA for axis analysis
        mean_vec = backend.mean(matrix_norm, axis=0, keepdims=True)
        centered = matrix_norm - mean_vec
        try:
            _, s, vh = backend.svd(centered)
            backend.eval(s)
            s_np = backend.to_numpy(s)
            s_squared = [float(x) ** 2 for x in s_np.flatten()]
            total = sum(s_squared)
            variance_explained = [x / total for x in s_squared] if total > 0 else [0.0] * len(s_squared)
            pc_variance = variance_explained[:5] + [0.0] * (5 - len(variance_explained[:5]))
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

        mms = (
            0.25 * ortho_score
            + 0.30 * gradient_score
            + 0.25 * cluster_score
            + 0.20 * opposition_score
        )

        # Determine verdict
        has_manifold = mms > 0.40
        if mms > 0.55:
            verdict = "STRONG MORAL MANIFOLD - Clear valence/agency/scope axes detected."
        elif mms > 0.40:
            verdict = "MODERATE MORAL MANIFOLD - Some ethical structure detected."
        else:
            verdict = "WEAK MORAL MANIFOLD - Limited moral geometry found."

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
            has_moral_manifold=has_manifold,
            verdict=verdict,
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
            if concept.axis == MoralAxis.VALENCE:
                valence_vecs.append(matrix[i])
            elif concept.axis == MoralAxis.AGENCY:
                agency_vecs.append(matrix[i])
            elif concept.axis == MoralAxis.SCOPE:
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
                _, _, vh = backend.svd(centered)
                return vh[0]
            except Exception:
                d = int(arr.shape[1])
                return backend.zeros((d,))

        val_vec = axis_direction(valence_vecs)
        agen_vec = axis_direction(agency_vecs)
        scope_vec = axis_direction(scope_vecs)

        def orthogonality(v1: "Array", v2: "Array") -> float:
            """Compute orthogonality as 1 - |cos(angle)|."""
            n1 = backend.norm(v1)
            n2 = backend.norm(v2)
            backend.eval(n1, n2)
            n1_val = float(backend.to_numpy(n1))
            n2_val = float(backend.to_numpy(n2))
            if n1_val < 1e-8 or n2_val < 1e-8:
                return 0.0
            dot = backend.sum(v1 * v2)
            backend.eval(dot)
            cos_sim = abs(float(backend.to_numpy(dot)) / (n1_val * n2_val))
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
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        backend = self._backend

        def axis_correlation(axis: MoralAxis) -> tuple[float, bool]:
            """Compute correlation for a specific axis."""
            levels = []
            projections = []

            for i, cid in enumerate(concepts):
                concept = self._concept_lookup.get(cid)
                if concept is None or concept.axis != axis:
                    continue
                levels.append(concept.level)
                backend.eval(matrix)
                projections.append(float(backend.to_numpy(matrix[i, 0])) if matrix.shape[1] > 0 else 0.0)

            if len(levels) < 3:
                return 0.0, False

            corr = VectorMath.spearman_correlation(levels, projections)
            if corr is None or math.isnan(float(corr)):
                corr = 0.0

            monotonic = abs(corr) > 0.8
            return float(corr), monotonic

        val_corr, val_mono = axis_correlation(MoralAxis.VALENCE)
        agen_corr, agen_mono = axis_correlation(MoralAxis.AGENCY)
        scope_corr, scope_mono = axis_correlation(MoralAxis.SCOPE)

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
        foundation_indices: dict[MoralFoundation, list[int]] = {}
        for i, cid in enumerate(concepts):
            concept = self._concept_lookup.get(cid)
            if concept is None:
                continue
            if concept.foundation not in foundation_indices:
                foundation_indices[concept.foundation] = []
            foundation_indices[concept.foundation].append(i)

        def cosine_sim(v1: "Array", v2: "Array") -> float:
            n1 = float(backend.to_numpy(backend.norm(v1)))
            n2 = float(backend.to_numpy(backend.norm(v2)))
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0
            dot = float(backend.to_numpy(backend.sum(v1 * v2)))
            return dot / (n1 * n2)

        # Compute within-foundation similarity
        within_sims = []
        for foundation, indices in foundation_indices.items():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = cosine_sim(matrix[indices[i]], matrix[indices[j]])
                    within_sims.append(sim)

        within_sim = sum(within_sims) / len(within_sims) if within_sims else 0.0

        # Compute between-foundation similarity
        between_sims = []
        pair_sims: dict[tuple[str, str], list[float]] = {}
        foundations = list(foundation_indices.keys())

        for f1_idx, f1 in enumerate(foundations):
            for f2 in foundations[f1_idx + 1 :]:
                key = (f1.value, f2.value)
                pair_sims[key] = []
                for i1 in foundation_indices[f1]:
                    for i2 in foundation_indices[f2]:
                        sim = cosine_sim(matrix[i1], matrix[i2])
                        between_sims.append(sim)
                        pair_sims[key].append(sim)

        between_sim = sum(between_sims) / len(between_sims) if between_sims else 0.0

        # Find most distinct and most overlapping
        foundation_means: dict[str, float] = {}
        for f, indices in foundation_indices.items():
            if len(indices) >= 2:
                sims = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        sims.append(cosine_sim(matrix[indices[i]], matrix[indices[j]]))
                foundation_means[f.value] = sum(sims) / len(sims) if sims else 0.0

        most_distinct = (
            max(foundation_means.keys(), key=lambda k: foundation_means[k])
            if foundation_means
            else "unknown"
        )

        most_overlapping = ("unknown", "unknown")
        if pair_sims:
            max_pair = max(pair_sims.keys(), key=lambda k: sum(pair_sims[k]) / len(pair_sims[k]) if pair_sims[k] else 0.0)
            most_overlapping = max_pair

        separation = within_sim / (between_sim + 1e-8) if between_sim > 0 else 1.0

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
            n1 = float(backend.to_numpy(backend.norm(v1)))
            n2 = float(backend.to_numpy(backend.norm(v2)))
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0

            dot = float(backend.to_numpy(backend.sum(v1 * v2)))
            cos_sim = dot / (n1 * n2)
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
            opposition_detected=mean_opp > 0.5,
        )

