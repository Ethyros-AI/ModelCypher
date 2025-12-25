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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class CompositionCategory(Enum):
    MENTAL_PREDICATE = "mentalPredicate"
    ACTION = "action"
    EVALUATIVE = "evaluative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIED = "quantified"
    RELATIONAL = "relational"


@dataclass(frozen=True)
class CompositionProbe:
    phrase: str
    components: list[str]
    category: CompositionCategory


@dataclass(frozen=True)
class CompositionAnalysis:
    probe: CompositionProbe
    barycentric_weights: list[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: list[float]

    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3


@dataclass(frozen=True)
class ConsistencyResult:
    """Result of cross-model compositional consistency check.

    Note: Models are ALWAYS compatible. The consistency_score measures
    how similar the compositional structure is, not whether merge is possible.
    """

    probe_count: int
    analyses_a: list[CompositionAnalysis]
    analyses_b: list[CompositionAnalysis]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float
    high_consistency: bool  # True if consistency_score >= 0.5 (convenience flag)
    interpretation: str


class CompositionalProbes:
    """
    Compositional probe analysis for cross-model semantic structure verification.
    Ported from CompositionalProbes.swift.
    """

    STANDARD_PROBES = [
        CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I KNOW", ["I", "KNOW"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I FEEL", ["I", "FEEL"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I SEE", ["I", "SEE"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I HEAR", ["I", "HEAR"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("SOMEONE DO", ["SOMEONE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("PEOPLE DO", ["PEOPLE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("I SAY", ["I", "SAY"], CompositionCategory.ACTION),
        CompositionProbe("GOOD THINGS", ["GOOD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BAD THINGS", ["BAD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("GOOD PEOPLE", ["GOOD", "PEOPLE"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BEFORE NOW", ["BEFORE", "NOW"], CompositionCategory.TEMPORAL),
        CompositionProbe("AFTER THIS", ["AFTER", "THIS"], CompositionCategory.TEMPORAL),
        CompositionProbe(
            "A LONG TIME BEFORE", ["A_LONG_TIME", "BEFORE"], CompositionCategory.TEMPORAL
        ),
        CompositionProbe("ABOVE HERE", ["ABOVE", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("FAR FROM HERE", ["FAR", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("NEAR THIS", ["NEAR", "THIS"], CompositionCategory.SPATIAL),
        CompositionProbe("MUCH GOOD", ["MUCH_MANY", "GOOD"], CompositionCategory.QUANTIFIED),
        CompositionProbe("MANY PEOPLE", ["MUCH_MANY", "PEOPLE"], CompositionCategory.QUANTIFIED),
        CompositionProbe(
            "I WANT GOOD THINGS",
            ["I", "WANT", "GOOD", "SOMETHING"],
            CompositionCategory.MENTAL_PREDICATE,
        ),
        CompositionProbe(
            "SOMEONE DO BAD THINGS",
            ["SOMEONE", "DO", "BAD", "SOMETHING"],
            CompositionCategory.ACTION,
        ),
    ]

    @staticmethod
    def analyze_composition(
        composition_embedding: "Array",  # [D]
        component_embeddings: "Array",  # [N, D]
        probe: CompositionProbe,
        backend: "Backend | None" = None,
    ) -> CompositionAnalysis:
        b = backend or get_default_backend()

        # Ensure 1D comp
        if composition_embedding.ndim == 2 and composition_embedding.shape[0] == 1:
            composition_embedding = composition_embedding[0]

        d = composition_embedding.shape[0]
        n = component_embeddings.shape[0]

        if n == 0 or d == 0:
            return CompositionAnalysis(probe, [], float("inf"), 0.0, [])

        # Centroid
        centroid = b.mean(component_embeddings, axis=0)

        # Centroid similarity
        centroid_sim_arr = CompositionalProbes._cosine_similarity(
            composition_embedding, centroid, b
        )
        b.eval(centroid_sim_arr)
        centroid_sim = float(b.to_numpy(centroid_sim_arr).item())

        # Component angles
        angles = []
        for i in range(n):
            sim_arr = CompositionalProbes._cosine_similarity(
                composition_embedding, component_embeddings[i], b
            )
            b.eval(sim_arr)
            angles.append(float(b.to_numpy(sim_arr).item()))

        # Barycentric weights via normal equations
        # G = component_embeddings @ component_embeddings.T [N, N]
        # rhs = component_embeddings @ composition_embedding [N]
        G = b.matmul(component_embeddings, b.transpose(component_embeddings))  # [N, N]
        rhs = b.matmul(component_embeddings, composition_embedding)  # [N]

        # Regularize diagonal for stability
        eps = 1e-6
        G = G + b.eye(n) * eps

        weights_arr = b.solve(G, rhs)  # [N]
        b.eval(weights_arr)
        weights = b.to_numpy(weights_arr).tolist()

        # Calc residual
        reconstructed = b.matmul(weights_arr, component_embeddings)  # [D]
        diff = composition_embedding - reconstructed
        residual_norm_arr = b.norm(diff)
        b.eval(residual_norm_arr)
        residual_norm = float(b.to_numpy(residual_norm_arr).item())

        return CompositionAnalysis(
            probe=probe,
            barycentric_weights=weights,
            residual_norm=residual_norm,
            centroid_similarity=centroid_sim,
            component_angles=angles,
        )

    @staticmethod
    def _cosine_similarity(a: "Array", b_vec: "Array", backend: "Backend") -> "Array":
        # Returns scalar array
        dot = backend.dot(a, b_vec)
        norm_a = backend.norm(a)
        norm_b = backend.norm(b_vec)
        denom = norm_a * norm_b
        return backend.where(denom > 1e-9, dot / denom, backend.array(0.0))

    @staticmethod
    def check_consistency(
        analyses_a: list[CompositionAnalysis], analyses_b: list[CompositionAnalysis]
    ) -> ConsistencyResult:
        if len(analyses_a) != len(analyses_b) or not analyses_a:
            return ConsistencyResult(0, [], [], 0, 0, 0, False, "Insufficient data")

        n = len(analyses_a)

        # Collect weights
        weights_a: list[float] = []
        weights_b: list[float] = []
        for i in range(n):
            if len(analyses_a[i].barycentric_weights) == len(analyses_b[i].barycentric_weights):
                weights_a.extend(analyses_a[i].barycentric_weights)
                weights_b.extend(analyses_b[i].barycentric_weights)

        # Collect angles
        angles_a: list[float] = []
        angles_b: list[float] = []
        for i in range(n):
            if len(analyses_a[i].component_angles) == len(analyses_b[i].component_angles):
                angles_a.extend(analyses_a[i].component_angles)
                angles_b.extend(analyses_b[i].component_angles)

        bary_corr = CompositionalProbes._pearson(weights_a, weights_b)
        ang_corr = CompositionalProbes._pearson(angles_a, angles_b)

        score = 0.4 * max(0, bary_corr) + 0.6 * max(0, ang_corr)
        high_consistency = score >= 0.5 and ang_corr >= 0.4

        interpretation = "Low consistency"
        if score >= 0.8:
            interpretation = "Excellent consistency"
        elif score >= 0.6:
            interpretation = "Good consistency"
        elif score >= 0.4:
            interpretation = "Partial consistency"

        return ConsistencyResult(
            probe_count=n,
            analyses_a=analyses_a,
            analyses_b=analyses_b,
            barycentric_correlation=bary_corr,
            angular_correlation=ang_corr,
            consistency_score=score,
            high_consistency=high_consistency,
            interpretation=interpretation,
        )

    @staticmethod
    def _pearson(a: list[float], b_list: list[float], backend: "Backend | None" = None) -> float:
        if len(a) < 2 or len(b_list) < 2:
            return 0.0

        bk = backend or get_default_backend()
        # Create vectors
        va = bk.array(a)
        vb = bk.array(b_list)

        ma = bk.mean(va)
        mb = bk.mean(vb)

        da = va - ma
        db = vb - mb

        num = bk.sum(da * db)
        den = bk.sqrt(bk.sum(da**2) * bk.sum(db**2))
        bk.eval(num, den)

        den_val = float(bk.to_numpy(den).item())
        if den_val > 1e-9:
            return float(bk.to_numpy(num).item()) / den_val
        return 0.0
