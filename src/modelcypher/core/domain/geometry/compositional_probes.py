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

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class CompositionCategory(str, Enum):
    mental_predicate = "mentalPredicate"
    action = "action"
    evaluative = "evaluative"
    temporal = "temporal"
    spatial = "spatial"
    quantified = "quantified"
    relational = "relational"

    MENTAL_PREDICATE = mental_predicate
    ACTION = action
    EVALUATIVE = evaluative
    TEMPORAL = temporal
    SPATIAL = spatial
    QUANTIFIED = quantified
    RELATIONAL = relational


@dataclass
class CompositionProbe:
    """
    A compositional probe: a phrase and its component primes.
    """

    phrase: str
    components: list[str]
    category: CompositionCategory


@dataclass
class CompositionAnalysis:
    """
    Result of compositional structure analysis.
    """

    probe: CompositionProbe
    barycentric_weights: list[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: list[float]

    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3


@dataclass
class ConsistencyResult:
    """
    Result of cross-model compositional consistency check.
    """

    probe_count: int
    analyses_a: list[CompositionAnalysis]
    analyses_b: list[CompositionAnalysis]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float
    is_compatible: bool
    interpretation: str


class CompositionalProbes:
    """
    Compositional probe analysis for cross-model semantic structure verification.
    """

    STANDARD_PROBES = [
        # Mental predicates
        CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I KNOW", ["I", "KNOW"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I FEEL", ["I", "FEEL"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I SEE", ["I", "SEE"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I HEAR", ["I", "HEAR"], CompositionCategory.MENTAL_PREDICATE),
        # Actions
        CompositionProbe("SOMEONE DO", ["SOMEONE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("PEOPLE DO", ["PEOPLE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("I SAY", ["I", "SAY"], CompositionCategory.ACTION),
        # Evaluatives
        CompositionProbe("GOOD THINGS", ["GOOD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BAD THINGS", ["BAD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("GOOD PEOPLE", ["GOOD", "PEOPLE"], CompositionCategory.EVALUATIVE),
        # Temporal
        CompositionProbe("BEFORE NOW", ["BEFORE", "NOW"], CompositionCategory.TEMPORAL),
        CompositionProbe("AFTER THIS", ["AFTER", "THIS"], CompositionCategory.TEMPORAL),
        CompositionProbe(
            "A LONG TIME BEFORE", ["A_LONG_TIME", "BEFORE"], CompositionCategory.TEMPORAL
        ),
        # Spatial
        CompositionProbe("ABOVE HERE", ["ABOVE", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("FAR FROM HERE", ["FAR", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("NEAR THIS", ["NEAR", "THIS"], CompositionCategory.SPATIAL),
        # Quantified
        CompositionProbe("MUCH GOOD", ["MUCH_MANY", "GOOD"], CompositionCategory.QUANTIFIED),
        CompositionProbe("MANY PEOPLE", ["MUCH_MANY", "PEOPLE"], CompositionCategory.QUANTIFIED),
        # Complex
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
        composition_embedding: list[float],
        component_embeddings: list[list[float]],
        probe: CompositionProbe,
        backend: "Backend | None" = None,
    ) -> CompositionAnalysis:
        n = len(component_embeddings)
        d = len(composition_embedding)

        if n == 0 or d == 0:
            return CompositionAnalysis(probe, [], float("inf"), 0.0, [])

        b = backend or get_default_backend()
        # Vectors using backend for speed
        target = b.array(composition_embedding)
        basis = b.array(component_embeddings)

        # Centroid
        centroid = b.mean(basis, axis=0)

        # Centroid similarity
        centroid_sim = CompositionalProbes.cosine_similarity(target, centroid, b)

        # Component angles
        angles = []
        for i in range(n):
            sim = CompositionalProbes.cosine_similarity(target, basis[i], b)
            angles.append(sim)

        # Barycentric weights
        weights, residual = CompositionalProbes.compute_barycentric_weights(target, basis, b)

        return CompositionAnalysis(
            probe=probe,
            barycentric_weights=weights,
            residual_norm=residual,
            centroid_similarity=centroid_sim,
            component_angles=angles,
        )

    @staticmethod
    def compute_barycentric_weights(
        target: "Array", basis: "Array", backend: "Backend"
    ) -> tuple[list[float], float]:
        # Use Moore-Penrose Pseudo-Inverse for robust least squares solution.
        # target (d,) approx weights (n,) @ basis (n,d)
        # target = basis.T @ weights
        # weights = pinv(basis.T) @ target

        try:
            basis_t = backend.transpose(basis)
            pinv_basis_t = backend.pinv(basis_t)
            weights_vec = backend.matmul(pinv_basis_t, target)
            backend.eval(weights_vec)
            weights = backend.to_numpy(weights_vec).tolist()
            reconstructed = backend.matmul(weights_vec, basis)
            diff = target - reconstructed
            residual_arr = backend.norm(diff)
            backend.eval(residual_arr)
            residual = float(backend.to_numpy(residual_arr).item())
            return (weights, residual)
        except Exception:
            # CPU fallback via NumPy for pinv to avoid GPU-only limitations.
            basis_np = backend.to_numpy(basis)
            target_np = backend.to_numpy(target)
            weights_np = np.linalg.pinv(basis_np.T) @ target_np
            reconstructed_np = weights_np @ basis_np
            residual = float(np.linalg.norm(target_np - reconstructed_np))
            return (weights_np.tolist(), residual)

    @staticmethod
    def cosine_similarity(a: "Array", b_vec: "Array", backend: "Backend") -> float:
        dot_arr = backend.sum(a * b_vec)
        norm_a_arr = backend.norm(a)
        norm_b_arr = backend.norm(b_vec)
        backend.eval(dot_arr, norm_a_arr, norm_b_arr)
        dot = float(backend.to_numpy(dot_arr).item())
        norm_a = float(backend.to_numpy(norm_a_arr).item())
        norm_b = float(backend.to_numpy(norm_b_arr).item())
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def check_consistency(
        analyses_a: list[CompositionAnalysis], analyses_b: list[CompositionAnalysis]
    ) -> ConsistencyResult:
        if len(analyses_a) != len(analyses_b) or not analyses_a:
            return ConsistencyResult(0, [], [], 0.0, 0.0, 0.0, False, "Insufficient data")

        n = len(analyses_a)

        all_weights_a = []
        all_weights_b = []
        for i in range(n):
            all_weights_a.extend(analyses_a[i].barycentric_weights)
            all_weights_b.extend(analyses_b[i].barycentric_weights)

        all_angles_a = []
        all_angles_b = []
        for i in range(n):
            all_angles_a.extend(analyses_a[i].component_angles)
            all_angles_b.extend(analyses_b[i].component_angles)

        bary_corr = CompositionalProbes.pearson_correlation(all_weights_a, all_weights_b)
        ang_corr = CompositionalProbes.pearson_correlation(all_angles_a, all_angles_b)

        score = 0.4 * max(0.0, bary_corr) + 0.6 * max(0.0, ang_corr)
        is_compatible = score >= 0.5 and ang_corr >= 0.4

        if score >= 0.8:
            interp = "Excellent compositional consistency."
        elif score >= 0.6:
            interp = "Good compositional consistency."
        elif score >= 0.4:
            interp = "Partial compositional consistency."
        else:
            interp = "Low compositional consistency."

        return ConsistencyResult(
            probe_count=n,
            analyses_a=analyses_a,
            analyses_b=analyses_b,
            barycentric_correlation=bary_corr,
            angular_correlation=ang_corr,
            consistency_score=score,
            is_compatible=is_compatible,
            interpretation=interp,
        )

    @staticmethod
    def pearson_correlation(
        a: list[float], b_list: list[float], backend: "Backend | None" = None
    ) -> float:
        if len(a) != len(b_list) or len(a) < 2:
            return 0.0

        bk = backend or get_default_backend()
        arr_a = bk.array(a)
        arr_b = bk.array(b_list)

        mean_a = bk.mean(arr_a)
        mean_b = bk.mean(arr_b)

        da = arr_a - mean_a
        db = arr_b - mean_b

        sum_ab = bk.sum(da * db)
        sum_a2 = bk.sum(da * da)
        sum_b2 = bk.sum(db * db)
        bk.eval(sum_ab, sum_a2, sum_b2)

        sum_ab_val = float(bk.to_numpy(sum_ab).item())
        sum_a2_val = float(bk.to_numpy(sum_a2).item())
        sum_b2_val = float(bk.to_numpy(sum_b2).item())

        denom = math.sqrt(sum_a2_val * sum_b2_val)
        if denom > 1e-10:
            return sum_ab_val / denom
        if sum_a2_val == 0.0 and sum_b2_val == 0.0:
            return 1.0
        return 0.0

    @staticmethod
    def analyze_all_probes(
        prime_embeddings: dict[str, list[float]],
        composition_embeddings: dict[str, list[float]],
        probes: list[CompositionProbe] = STANDARD_PROBES,
    ) -> list[CompositionAnalysis]:
        analyses = []
        for probe in probes:
            if probe.phrase not in composition_embeddings:
                continue

            comp_embed = composition_embeddings[probe.phrase]

            component_embeds = []
            all_found = True
            for c in probe.components:
                if c in prime_embeddings:
                    component_embeds.append(prime_embeddings[c])
                else:
                    all_found = False
                    break

            if not all_found:
                continue

            analysis = CompositionalProbes.analyze_composition(comp_embed, component_embeds, probe)
            analyses.append(analysis)

        return analyses
