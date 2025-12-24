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

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable

import hashlib
import math

import numpy as np


class AnchorSet(str, Enum):
    semantic_primes = "semanticPrimes"
    computational_gates = "computationalGates"
    hybrid = "hybrid"
    custom = "custom"


@dataclass(frozen=True)
class FitPrediction:
    fit_score: float
    location_score: float
    direction_score: float
    rotation_penalty: float

    @property
    def is_compatible(self) -> bool:
        return self.fit_score >= 0.5

    @property
    def recommends_smoothing(self) -> bool:
        return self.fit_score < 0.7 or self.rotation_penalty > 0.3

    @property
    def assessment(self) -> str:
        if self.fit_score >= 0.9:
            return "excellent"
        if self.fit_score >= 0.7:
            return "good"
        if self.fit_score >= 0.5:
            return "moderate"
        if self.fit_score >= 0.3:
            return "poor"
        return "incompatible"


class CompositionStrategy(str, Enum):
    weight_blending = "weightBlending"
    attention_routing = "attentionRouting"
    sequential = "sequential"
    automatic = "automatic"


@dataclass(frozen=True)
class GeometricFingerprint:
    gram_hash: str
    gram_mean_off_diagonal: float
    gram_std_off_diagonal: float
    gram_spectral_radius: float
    gram_condition_number: float
    semantic_path_self_correlations: list[float] = field(default_factory=list)
    computational_path_self_correlations: list[float] = field(default_factory=list)
    estimated_rotation_complexity: float = 0.0
    effective_dimensionality: float = 0.0
    anchor_set: AnchorSet = AnchorSet.hybrid
    anchor_count: int = 0
    hidden_size: int = 0
    model_id: str = ""
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def predict_fit(
        self,
        other: GeometricFingerprint,
        location_weight: float = 0.4,
        direction_weight: float = 0.4,
        rotation_weight: float = 0.2,
    ) -> FitPrediction:
        mean_diff = abs(self.gram_mean_off_diagonal - other.gram_mean_off_diagonal)
        std_diff = abs(self.gram_std_off_diagonal - other.gram_std_off_diagonal)
        spectral_ratio = min(self.gram_spectral_radius, other.gram_spectral_radius) / max(
            self.gram_spectral_radius,
            other.gram_spectral_radius,
            1e-6,
        )
        location_score = max(0.0, 1.0 - mean_diff - 0.5 * std_diff) * spectral_ratio

        semantic_diff = _mean_abs_diff(
            self.semantic_path_self_correlations,
            other.semantic_path_self_correlations,
        )
        computational_diff = _mean_abs_diff(
            self.computational_path_self_correlations,
            other.computational_path_self_correlations,
        )
        direction_score = max(0.0, 1.0 - semantic_diff - computational_diff)

        rotation_penalty = (self.estimated_rotation_complexity + other.estimated_rotation_complexity) / 2.0

        fit_score = (
            location_weight * location_score
            + direction_weight * direction_score
            - rotation_weight * rotation_penalty
        )

        return FitPrediction(
            fit_score=max(0.0, min(1.0, fit_score)),
            location_score=location_score,
            direction_score=direction_score,
            rotation_penalty=rotation_penalty,
        )

    @staticmethod
    def suggest_composition_strategy(fingerprints: Iterable[GeometricFingerprint]) -> CompositionStrategy:
        items = list(fingerprints)
        if len(items) < 2:
            return CompositionStrategy.automatic

        avg_fit = 0.0
        comparisons = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                avg_fit += items[i].predict_fit(items[j]).fit_score
                comparisons += 1
        avg_fit = avg_fit / max(comparisons, 1)

        if avg_fit >= 0.8:
            return CompositionStrategy.weight_blending

        avg_direction = 0.0
        for fp in items:
            semantic = _safe_mean(fp.semantic_path_self_correlations)
            computational = _safe_mean(fp.computational_path_self_correlations)
            avg_direction += (semantic + computational) / 2.0
        avg_direction = avg_direction / max(len(items), 1)

        if avg_fit >= 0.5 and avg_direction < 0.6:
            return CompositionStrategy.attention_routing

        if avg_fit < 0.5:
            return CompositionStrategy.sequential

        return CompositionStrategy.automatic

    @staticmethod
    def gram_statistics(gram: list[float], n: int) -> tuple[float, float, str]:
        if len(gram) != n * n or n <= 1:
            return 0.0, 0.0, ""

        off_diag: list[float] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag.append(float(gram[i * n + j]))

        mean = _safe_mean(off_diag)
        variance = _safe_mean([(val - mean) ** 2 for val in off_diag])
        std = math.sqrt(variance)

        raw = np.asarray(gram, dtype=np.float32).tobytes()
        gram_hash = hashlib.sha256(raw).hexdigest()

        return mean, std, gram_hash

    @staticmethod
    def estimate_spectral_radius(gram: list[float], n: int, iterations: int = 50) -> float:
        if len(gram) != n * n or n <= 0:
            return 0.0

        rng = np.random.default_rng()
        v = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        lam = 0.0
        for _ in range(iterations):
            w = np.zeros(n, dtype=np.float64)
            for i in range(n):
                row_sum = 0.0
                for j in range(n):
                    row_sum += float(gram[i * n + j]) * v[j]
                w[i] = row_sum
            lam = float(np.dot(v, w))
            norm = float(np.linalg.norm(w))
            if norm <= 1e-10:
                break
            v = w / norm

        return float(abs(lam))

    @staticmethod
    def estimate_condition_number(gram: list[float], n: int, iterations: int = 50) -> float:
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n, max_iterations=iterations)
        if eigenvalues is None or not eigenvalues:
            return float("inf")
        max_eigen = max(eigenvalues)
        min_eigen = min(val for val in eigenvalues if val > 1e-12) if eigenvalues else 0.0
        if max_eigen <= 1e-12 or min_eigen <= 1e-12:
            return float("inf")
        return float(max_eigen / min_eigen)

    @staticmethod
    def estimate_effective_dimensionality(gram: list[float], n: int) -> float:
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n)
        if eigenvalues is None or not eigenvalues:
            return float(n)
        clamped = [max(0.0, val) for val in eigenvalues]
        sum_vals = sum(clamped)
        sum_sq = sum(val * val for val in clamped)
        if sum_sq <= 1e-12:
            return float(n)
        return float((sum_vals * sum_vals) / sum_sq)

    @staticmethod
    def symmetric_eigenvalues(
        gram: list[float],
        n: int,
        max_iterations: int = 64,
        tolerance: float = 1e-10,
    ) -> list[float] | None:
        if len(gram) != n * n or n <= 0:
            return None
        if n == 1:
            return [float(gram[0])]

        matrix = [float(val) for val in gram]

        def idx(i: int, j: int) -> int:
            return i * n + j

        for _ in range(max_iterations):
            max_off = 0.0
            p = 0
            q = 1
            for i in range(n):
                for j in range(i + 1, n):
                    value = abs(matrix[idx(i, j)])
                    if value > max_off:
                        max_off = value
                        p = i
                        q = j
            if max_off < tolerance:
                break

            app = matrix[idx(p, p)]
            aqq = matrix[idx(q, q)]
            apq = matrix[idx(p, q)]
            if apq == 0.0:
                continue

            phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
            c = math.cos(phi)
            s = math.sin(phi)

            for i in range(n):
                if i == p or i == q:
                    continue
                aip = matrix[idx(i, p)]
                aiq = matrix[idx(i, q)]
                new_aip = c * aip - s * aiq
                new_aiq = s * aip + c * aiq
                matrix[idx(i, p)] = new_aip
                matrix[idx(p, i)] = new_aip
                matrix[idx(i, q)] = new_aiq
                matrix[idx(q, i)] = new_aiq

            new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq
            new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
            matrix[idx(p, p)] = new_app
            matrix[idx(q, q)] = new_aqq
            matrix[idx(p, q)] = 0.0
            matrix[idx(q, p)] = 0.0

        return [matrix[idx(i, i)] for i in range(n)]


GeometricFingerprint.placeholder = GeometricFingerprint(
    gram_hash="placeholder",
    gram_mean_off_diagonal=0.5,
    gram_std_off_diagonal=0.1,
    gram_spectral_radius=1.0,
    gram_condition_number=10.0,
    semantic_path_self_correlations=[0.5] * 8,
    computational_path_self_correlations=[0.5] * 7,
    estimated_rotation_complexity=0.5,
    effective_dimensionality=8.0,
    anchor_set=AnchorSet.hybrid,
    anchor_count=131,
    hidden_size=4096,
    model_id="placeholder",
    computed_at=datetime.utcfromtimestamp(0),
)


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _mean_abs_diff(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    left = list(lhs)
    right = list(rhs)
    if not left or not right:
        return 0.0
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    total = sum(abs(left[i] - right[i]) for i in range(count))
    return total / count
