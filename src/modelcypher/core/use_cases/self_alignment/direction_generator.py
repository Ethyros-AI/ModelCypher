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

"""Direction generation for geometric self-alignment.

Generates candidate weight perturbations that aim to reduce manifold entropy
by aligning SVD ratios to fundamental constants {π, e, φ, √2, π/e}.

Strategies:
1. Random - baseline exploration in weight space
2. Constant-aligned - perturb singular values toward constant ratios
3. SVD gap-filling - target specific singular value pairs
4. Complexity-law - adjust to bring slope → e/π, intercept → π/e

All directions are designed to be compatible with null-space projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fundamental_constants import (
    FundamentalConstant,
    percent_error,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    geodesic_svd,
    machine_epsilon,
    safe_log_epsilon,
    sqrt_scalar,
    svd_rank_threshold,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class DirectionStrategy(Enum):
    """Strategy for generating weight perturbation directions."""

    RANDOM = "random"  # Random direction in weight space
    CONSTANT_ALIGNED = "constant_aligned"  # Align SVD ratios to constants
    SVD_GAP = "svd_gap"  # Fill gaps in SVD spectrum
    SPECTRAL_COMPRESS = "spectral_compress"  # Reduce spectral entropy


@dataclass
class DirectionResult:
    """Result of direction generation."""

    direction: "Array"  # The perturbation direction
    strategy: DirectionStrategy
    scale: float  # Direction magnitude (Frobenius norm)
    target_constant: Optional[FundamentalConstant] = None  # For constant-aligned
    target_ratio_indices: Optional[Tuple[int, int]] = None  # For SVD gap
    expected_entropy_reduction: float = 0.0  # Estimated entropy improvement


class DirectionGenerator:
    """Generate candidate weight perturbations for geometric self-alignment.

    The key insight: weight matrices have SVD structure, and perturbing
    singular values to match fundamental constant ratios reduces entropy.

    Usage:
        generator = DirectionGenerator(backend)
        directions = generator.generate(
            weights,
            strategies=[DirectionStrategy.CONSTANT_ALIGNED, DirectionStrategy.RANDOM],
        )
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

        # Target constants for alignment (ordered by significance)
        self._target_constants = [
            FundamentalConstant.PI_OVER_E,  # Most frequent in neural geometry
            FundamentalConstant.PHI,  # Self-similar recursion
            FundamentalConstant.SQRT2,  # Orthogonal projection
            FundamentalConstant.E_OVER_PI,  # Complexity-dimension slope
            FundamentalConstant.E,  # Information scaling
            FundamentalConstant.PI_OVER_2,  # Quarter rotation
        ]

    def generate(
        self,
        weights: "Array",
        strategies: Optional[List[DirectionStrategy]] = None,
    ) -> List[DirectionResult]:
        """Generate candidate perturbation directions.

        Args:
            weights: Weight matrix to perturb [out_dim, in_dim]
            strategies: List of strategies to use (default: all)

        Returns:
            List of DirectionResult with candidate directions
        """
        b = self._backend
        w = b.array(weights) if not hasattr(weights, "shape") else weights
        b.eval(w)

        if strategies is None:
            strategies = list(DirectionStrategy)

        directions: List[DirectionResult] = []

        # Compute SVD once for all constant-aligned strategies
        U, S, Vt = geodesic_svd(b, w)
        b.eval(U, S, Vt)

        n_sv = int(S.shape[0])
        if n_sv == 0:
            return directions

        # Convert singular values once for deterministic geometry decisions
        S_list = [float(b.to_scalar(S[k:k+1])) for k in range(n_sv)]
        max_s = max(S_list) if S_list else 0.0
        eps = float(machine_epsilon(b, S))
        sqrt_eps = sqrt_scalar(eps, b)
        rank_scale = svd_rank_threshold(b, S, n_sv)
        rank_threshold = max_s * rank_scale
        value_threshold = max(rank_threshold, eps)
        valid_indices = [i for i, s in enumerate(S_list) if s > value_threshold]

        if len(valid_indices) < 2:
            return directions

        for strategy in strategies:
            if strategy == DirectionStrategy.RANDOM:
                dirs = self._generate_random(w, sqrt_eps)
            elif strategy == DirectionStrategy.CONSTANT_ALIGNED:
                dirs = self._generate_constant_aligned(
                    U, S, Vt, S_list, valid_indices, sqrt_eps
                )
            elif strategy == DirectionStrategy.SVD_GAP:
                dirs = self._generate_svd_gap(
                    U, S, Vt, S_list, valid_indices, sqrt_eps
                )
            elif strategy == DirectionStrategy.SPECTRAL_COMPRESS:
                dirs = self._generate_spectral_compress(U, S, Vt, sqrt_eps)
            else:
                dirs = []

            directions.extend(dirs)

        return directions

    def _generate_random(
        self,
        weights: "Array",
        sqrt_eps: float,
    ) -> List[DirectionResult]:
        """Generate random perturbation directions.

        Random directions provide baseline exploration and can discover
        unexpected improvements. Normalized to unit Frobenius norm.
        """
        b = self._backend
        shape = b.shape(weights)
        results = []

        # Random direction with same shape as weights
        direction = b.random_normal(tuple(shape))
        b.eval(direction)

        # Normalize to unit Frobenius norm
        norm = b.sqrt(b.sum(direction * direction))
        b.eval(norm)
        norm_val = float(b.to_scalar(norm))

        weight_norm = b.sqrt(b.sum(weights * weights))
        b.eval(weight_norm)
        weight_norm_val = float(b.to_scalar(weight_norm))

        eps = float(machine_epsilon(b, direction))
        if norm_val > eps:
            direction = direction / norm_val

        # Minimal meaningful scale from machine epsilon and weight norm
        scale = sqrt_eps * max(1.0, weight_norm_val)
        scaled = direction * scale

        results.append(DirectionResult(
            direction=scaled,
            strategy=DirectionStrategy.RANDOM,
            scale=self._direction_norm(scaled),
        ))

        return results

    def _generate_constant_aligned(
        self,
        U: "Array",
        S: "Array",
        Vt: "Array",
        S_list: List[float],
        valid_indices: List[int],
        sqrt_eps: float,
    ) -> List[DirectionResult]:
        """Generate directions that align SVD ratios to fundamental constants.

        For each target constant, find the singular value pair that is closest
        but not matching, and generate a direction that moves it closer.
        """
        b = self._backend
        results = []
        if len(valid_indices) < 2:
            return results

        percent_eps = sqrt_eps * 100.0

        # Find best alignment targets for each constant (full numeric rank)
        targets = []
        for const in self._target_constants:
            # Find singular value pair closest to this constant
            best_pair = None
            best_error = float("inf")

            for idx_i, i in enumerate(valid_indices[:-1]):
                s_i = S_list[i]
                for j in valid_indices[idx_i + 1:]:
                    s_j = S_list[j]
                    if s_j <= 0.0:
                        continue

                    ratio = s_i / s_j
                    error = percent_error(ratio, const.value)

                    # Only consider pairs that aren't already aligned
                    if error <= percent_eps:
                        continue

                    if error < best_error:
                        best_error = error
                        best_pair = (i, j, ratio, const)

            if best_pair is not None:
                targets.append(best_pair)

        # Generate directions for all constants with mismatches
        for idx_i, idx_j, current_ratio, const in targets:
            target_ratio = const.value

            # Compute minimal L2 change under the ratio constraint
            s_i = S_list[idx_i]
            s_j = S_list[idx_j]
            denom = (target_ratio * target_ratio) + 1.0
            if denom <= 0.0:
                continue

            s_j_prime = (target_ratio * s_i + s_j) / denom
            s_i_prime = target_ratio * s_j_prime

            # Create modified S vector
            S_new_list = list(S_list)
            S_new_list[idx_i] = s_i_prime
            S_new_list[idx_j] = s_j_prime
            S_new = b.array(S_new_list)
            b.eval(S_new)

            # Reconstruct weight direction: W_new = U @ diag(S_new) @ Vt
            # Direction = W_new - W = U @ diag(S_new - S) @ Vt
            S_delta = S_new - S
            direction = b.matmul(U * S_delta, Vt)
            b.eval(direction)

            current_error = percent_error(current_ratio, target_ratio)
            new_ratio = s_i_prime / s_j_prime if s_j_prime > 0 else current_ratio
            new_error = percent_error(new_ratio, target_ratio)
            estimated_reduction = max(0.0, current_error - new_error)

            results.append(DirectionResult(
                direction=direction,
                strategy=DirectionStrategy.CONSTANT_ALIGNED,
                scale=self._direction_norm(direction),
                target_constant=const,
                target_ratio_indices=(idx_i, idx_j),
                expected_entropy_reduction=estimated_reduction,
            ))

        return results

    def _generate_svd_gap(
        self,
        U: "Array",
        S: "Array",
        Vt: "Array",
        S_list: List[float],
        valid_indices: List[int],
        sqrt_eps: float,
    ) -> List[DirectionResult]:
        """Generate directions that fill gaps in the SVD spectrum.

        Finds singular value pairs with unusually large gaps and smooths them.
        A smoother spectrum often indicates more coherent representation.
        """
        b = self._backend
        results = []
        if len(valid_indices) < 2:
            return results

        # Smooth adjacent log-ratio gaps across numeric rank
        sorted_indices = sorted(valid_indices)
        pair_logs: List[Tuple[int, int, float]] = []
        log_values: List[float] = []

        for idx in range(len(sorted_indices) - 1):
            i = sorted_indices[idx]
            j = sorted_indices[idx + 1]
            s_i = S_list[i]
            s_j = S_list[j]
            if s_j <= 0.0:
                continue

            ratio = s_i / s_j
            if ratio <= 0.0:
                continue

            log_ratio = abs(math.log(ratio))
            pair_logs.append((i, j, log_ratio))
            log_values.append(log_ratio)

        if not log_values:
            return results

        # Data-derived gap threshold (largest relative gap in log-ratios)
        sorted_logs = sorted(log_values, reverse=True)
        threshold = find_magnitude_gap_threshold(
            sorted_logs,
            eps=sqrt_eps,
            backend=b,
        )

        # Apply smoothing to gaps above threshold and beyond numerical noise
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        total_log_gap = 0.0

        for i, j, log_ratio in pair_logs:
            if log_ratio < max(threshold, sqrt_eps):
                continue
            s_i = S_list[i]
            s_j = S_list[j]
            target = math.sqrt(s_i * s_j)
            sums[i] = sums.get(i, 0.0) + target
            counts[i] = counts.get(i, 0) + 1
            sums[j] = sums.get(j, 0.0) + target
            counts[j] = counts.get(j, 0) + 1
            total_log_gap += log_ratio

        if not sums:
            return results

        # Create modified S vector (average targets per index)
        S_new_list = list(S_list)
        for idx, total in sums.items():
            S_new_list[idx] = total / counts[idx]
        S_new = b.array(S_new_list)
        b.eval(S_new)

        # Direction = U @ diag(S_new - S) @ Vt
        S_delta = S_new - S
        direction = b.matmul(U * S_delta, Vt)
        b.eval(direction)

        results.append(DirectionResult(
            direction=direction,
            strategy=DirectionStrategy.SVD_GAP,
            scale=self._direction_norm(direction),
            target_ratio_indices=None,
            expected_entropy_reduction=total_log_gap,
        ))

        return results

    def _generate_spectral_compress(
        self,
        U: "Array",
        S: "Array",
        Vt: "Array",
        sqrt_eps: float,
    ) -> List[DirectionResult]:
        """Generate directions that reduce spectral entropy.

        Spectral entropy measures the "spread" of singular values.
        Lower entropy = more concentrated spectrum = potentially more coherent.

        Strategy: Boost top singular values, suppress small ones.
        """
        b = self._backend
        results = []

        entropy_before = self._spectral_entropy(S)
        total = b.sum(S * S)
        b.eval(total)
        total_val = float(b.to_scalar(total))
        eps = float(machine_epsilon(b, S))

        if total_val <= eps:
            return results

        # Gradient descent direction for spectral entropy
        S_sq = S * S
        p = S_sq / total_val
        eps_log = safe_log_epsilon(b, S)
        p_safe = b.maximum(p, b.array(eps_log))
        log_p = b.log(p_safe)
        b.eval(log_p)

        entropy_scalar = b.array(entropy_before)
        descent = 2.0 * S * (entropy_scalar + log_p) / total_val
        b.eval(descent)

        descent_norm = self._direction_norm(descent)
        s_norm = self._direction_norm(S)
        if descent_norm <= eps:
            return results
        step = sqrt_eps * max(1.0, s_norm) / descent_norm

        S_new = S + descent * step
        S_new = b.maximum(S_new, b.array(0.0))
        b.eval(S_new)

        S_delta = S_new - S
        direction = b.matmul(U * S_delta, Vt)
        b.eval(direction)

        entropy_after = self._spectral_entropy(S_new)

        results.append(DirectionResult(
            direction=direction,
            strategy=DirectionStrategy.SPECTRAL_COMPRESS,
            scale=self._direction_norm(direction),
            expected_entropy_reduction=max(0.0, entropy_before - entropy_after),
        ))

        return results

    def _spectral_entropy(self, S: "Array") -> float:
        """Compute spectral entropy for singular values."""
        b = self._backend
        S_sq = S * S
        total = b.sum(S_sq)
        b.eval(total)
        total_val = float(b.to_scalar(total))
        eps = float(machine_epsilon(b, S))
        if total_val <= eps:
            return 0.0

        p = S_sq / total_val
        eps_log = safe_log_epsilon(b, S)
        p_safe = b.maximum(p, b.array(eps_log))
        log_p = b.log(p_safe)
        entropy = -b.sum(p * log_p)
        b.eval(entropy)
        return float(b.to_scalar(entropy))

    def _direction_norm(self, direction: "Array") -> float:
        """Compute Frobenius norm of a direction."""
        b = self._backend
        norm = b.sqrt(b.sum(direction * direction))
        b.eval(norm)
        return float(b.to_scalar(norm))


__all__ = [
    "DirectionGenerator",
    "DirectionResult",
    "DirectionStrategy",
]
