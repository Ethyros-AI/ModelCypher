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
    division_epsilon,
    power_iteration_eigh,
    precision_dtype,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import safe_arithmetic_mean
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)


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

        backend = get_default_backend()

        # Vectorized off-diagonal statistics - O(n²) GPU ops vs O(n²) Python loops
        gram_arr = backend.reshape(backend.array(gram), (n, n))
        mask = 1.0 - backend.eye(n)  # 1 for off-diagonal, 0 for diagonal
        off_diag_count = n * (n - 1)  # n² - n off-diagonal elements

        # Mean of off-diagonal elements
        off_diag_sum = backend.sum(gram_arr * mask)
        backend.eval(off_diag_sum)
        off_diag_sum_val = float(backend.to_scalar(off_diag_sum))
        mean = off_diag_sum_val / off_diag_count

        # Variance of off-diagonal elements
        centered = (gram_arr - mean) * mask
        variance_sum = backend.sum(centered * centered)
        backend.eval(variance_sum)
        variance_sum_val = float(backend.to_scalar(variance_sum))
        variance = variance_sum_val / off_diag_count
        std = sqrt_scalar(variance, backend)

        raw_bytes = b"".join(struct.pack("<f", float(val)) for val in gram)
        gram_hash = hashlib.sha256(raw_bytes).hexdigest()

        return mean, std, gram_hash

    @staticmethod
    def estimate_spectral_radius(gram: list[float], n: int) -> float:
        """Estimate spectral radius via power iteration with convergence-based stopping.

        Uses sqrt(machine_epsilon) as convergence tolerance, derived from dtype precision.
        Minimum iterations = ceil(log2(n)) + 1 (geometric convergence theory).
        Maximum iterations = 10 * n (conservative upper bound).
        """
        if len(gram) != n * n or n <= 0:
            return 0.0

        import math

        backend = get_default_backend()
        backend.random_seed(42)

        # Convert gram to matrix once - vectorized matmul is O(n²) GPU ops vs O(n²) Python loops
        gram_arr = backend.reshape(backend.array(gram), (n, n))
        backend.eval(gram_arr)

        v = backend.random_normal((n,))
        norm_arr = geodesic_norms(backend.reshape(v, (1, -1)), backend)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr[0]))
        if norm > 0:
            v = v / norm

        # Convergence parameters derived from theory
        # sqrt(float32 machine epsilon) for convergence tolerance
        tol = 2.0 ** -11.5
        min_iter = int(math.ceil(math.log2(max(2, n)))) + 1
        max_iter = 10 * n

        lam = 0.0
        lam_prev = 0.0
        for i in range(max_iter):
            # Vectorized matrix-vector multiply instead of O(n²) Python loops
            w = backend.matmul(gram_arr, v)
            backend.eval(w)

            v_mat = backend.reshape(v, (1, -1))
            w_mat = backend.reshape(w, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(v_mat, w_mat, backend)
            v_norm_arr = geodesic_norms(v_mat, backend)
            w_norm_arr = geodesic_norms(w_mat, backend)
            backend.eval(cos_arr, v_norm_arr, w_norm_arr)
            cos_val = float(backend.to_scalar(cos_arr[0]))
            v_norm_val = float(backend.to_scalar(v_norm_arr[0]))
            w_norm_val = float(backend.to_scalar(w_norm_arr[0]))
            lam = cos_val * v_norm_val * w_norm_val
            norm = w_norm_val
            eps = division_epsilon(backend, w)
            if norm <= eps:
                break

            # Convergence check: relative change in eigenvalue estimate
            if i >= min_iter and abs(lam - lam_prev) < tol * max(abs(lam), tol):
                break

            lam_prev = lam
            v = w / norm

        return float(abs(lam))

    @staticmethod
    def estimate_condition_number(gram: list[float], n: int) -> float:
        """Estimate condition number from eigenvalues.

        Convergence is handled internally by power_iteration_eigh.
        """
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n)
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

        backend = get_default_backend()
        eig_arr = backend.array(eigenvalues)
        zeros = backend.zeros_like(eig_arr)
        clamped = backend.maximum(eig_arr, zeros)

        sum_vals_arr = backend.sum(clamped)
        sum_sq_arr = backend.sum(clamped * clamped)
        backend.eval(sum_vals_arr, sum_sq_arr)

        sum_vals = float(backend.to_scalar(sum_vals_arr))
        sum_sq = float(backend.to_scalar(sum_sq_arr))

        eig_eps = regularization_epsilon(backend, clamped)
        if sum_sq <= eig_eps:
            return float(n)
        return float((sum_vals * sum_vals) / sum_sq)

    @staticmethod
    def symmetric_eigenvalues(gram: list[float], n: int) -> list[float] | None:
        """Compute eigenvalues of symmetric Gram matrix.

        Convergence is dtype-derived inside power_iteration_eigh -
        no arbitrary iteration count needed.
        """
        if len(gram) != n * n or n <= 0:
            return None
        if n == 1:
            return [float(gram[0])]

        backend = get_default_backend()
        matrix = backend.reshape(
            backend.array(gram, dtype=precision_dtype(backend)), (n, n)
        )
        eigenvalues, _ = power_iteration_eigh(backend, matrix, k=n)
        backend.eval(eigenvalues)
        eigen_list = backend.tolist(eigenvalues)
        if not isinstance(eigen_list, list):
            return [float(eigen_list)]
        return [float(val) for val in eigen_list]


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

    backend = get_default_backend()
    left_arr = backend.array(left[:count])
    right_arr = backend.array(right[:count])
    diff = backend.abs(left_arr - right_arr)
    mean_arr = backend.mean(diff)
    backend.eval(mean_arr)
    return float(backend.to_scalar(mean_arr))
