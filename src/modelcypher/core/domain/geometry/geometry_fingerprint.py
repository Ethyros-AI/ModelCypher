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

import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    atan2_scalar,
    cos_scalar,
    division_epsilon,
    regularization_epsilon,
    sin_scalar,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import safe_arithmetic_mean


class AnchorSet(str, Enum):
    semantic_primes = "semanticPrimes"
    computational_gates = "computationalGates"
    hybrid = "hybrid"
    custom = "custom"


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

    @staticmethod
    def gram_statistics(gram: list[float], n: int) -> tuple[float, float, str]:
        if len(gram) != n * n or n <= 1:
            return 0.0, 0.0, ""

        off_diag: list[float] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag.append(float(gram[i * n + j]))

        mean = safe_arithmetic_mean(off_diag)
        variance = safe_arithmetic_mean([(val - mean) ** 2 for val in off_diag])
        backend = get_default_backend()
        std = sqrt_scalar(variance, backend)
        raw_bytes = b"".join(struct.pack("<f", float(val)) for val in gram)
        gram_hash = hashlib.sha256(raw_bytes).hexdigest()

        return mean, std, gram_hash

    @staticmethod
    def estimate_spectral_radius(gram: list[float], n: int, iterations: int = 50) -> float:
        if len(gram) != n * n or n <= 0:
            return 0.0

        backend = get_default_backend()
        backend.random_seed(42)
        v = backend.random_normal((n,))
        norm_arr = backend.norm(v)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        if norm > 0:
            v = v / norm

        lam = 0.0
        for _ in range(iterations):
            w_values: list[float] = []
            for i in range(n):
                row_sum = 0.0
                for j in range(n):
                    v_val = float(backend.to_scalar(v[j]))
                    row_sum += float(gram[i * n + j]) * v_val
                w_values.append(row_sum)
            w = backend.array(w_values)
            backend.eval(w)

            dot_arr = backend.sum(v * w)
            norm_arr = backend.norm(w)
            backend.eval(dot_arr, norm_arr)
            lam = float(backend.to_scalar(dot_arr))
            norm = float(backend.to_scalar(norm_arr))
            eps = division_epsilon(backend, w)
            if norm <= eps:
                break
            v = w / norm

        return float(abs(lam))

    @staticmethod
    def estimate_condition_number(gram: list[float], n: int, iterations: int = 50) -> float:
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n, max_iterations=iterations)
        if eigenvalues is None or not eigenvalues:
            return float("inf")
        backend = get_default_backend()
        eig_eps = regularization_epsilon(backend, backend.array(eigenvalues))
        max_eigen = max(eigenvalues)
        min_eigen = min(val for val in eigenvalues if val > eig_eps) if eigenvalues else 0.0
        if max_eigen <= eig_eps or min_eigen <= eig_eps:
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
        backend = get_default_backend()
        eig_eps = regularization_epsilon(backend, backend.array(clamped))
        if sum_sq <= eig_eps:
            return float(n)
        return float((sum_vals * sum_vals) / sum_sq)

    @staticmethod
    def symmetric_eigenvalues(
        gram: list[float],
        n: int,
        max_iterations: int = 64,
        tolerance: float = 0.0,
    ) -> list[float] | None:
        if len(gram) != n * n or n <= 0:
            return None
        if n == 1:
            return [float(gram[0])]

        matrix = [float(val) for val in gram]
        backend = get_default_backend()
        tol = max(tolerance, regularization_epsilon(backend, backend.array(matrix)))

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
            if max_off < tol:
                break

            app = matrix[idx(p, p)]
            aqq = matrix[idx(q, q)]
            apq = matrix[idx(p, q)]
            if apq == 0.0:
                continue

            phi = 0.5 * atan2_scalar(2.0 * apq, aqq - app, backend)
            c = cos_scalar(phi, backend)
            s = sin_scalar(phi, backend)

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
