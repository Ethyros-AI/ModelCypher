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
    E,
    E_OVER_PI,
    PHI,
    PI,
    PI_OVER_E,
    SQRT2,
    FundamentalConstant,
    find_constant_match,
    percent_error,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_svd,
    machine_epsilon,
    regularization_epsilon,
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
    scale: float  # Recommended scale for this direction
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
            n_directions=10,
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
        n_directions: int = 10,
        strategies: Optional[List[DirectionStrategy]] = None,
        scale: float = 0.01,
    ) -> List[DirectionResult]:
        """Generate candidate perturbation directions.

        Args:
            weights: Weight matrix to perturb [out_dim, in_dim]
            n_directions: Number of directions to generate
            strategies: List of strategies to use (default: all)
            scale: Base scale for perturbations

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

        # Distribute directions across strategies
        per_strategy = max(1, n_directions // len(strategies))
        remainder = n_directions - per_strategy * len(strategies)

        for i, strategy in enumerate(strategies):
            count = per_strategy + (1 if i < remainder else 0)

            if strategy == DirectionStrategy.RANDOM:
                dirs = self._generate_random(w, count, scale)
            elif strategy == DirectionStrategy.CONSTANT_ALIGNED:
                dirs = self._generate_constant_aligned(w, U, S, Vt, count, scale)
            elif strategy == DirectionStrategy.SVD_GAP:
                dirs = self._generate_svd_gap(w, U, S, Vt, count, scale)
            elif strategy == DirectionStrategy.SPECTRAL_COMPRESS:
                dirs = self._generate_spectral_compress(w, U, S, Vt, count, scale)
            else:
                dirs = []

            directions.extend(dirs)

        return directions

    def _generate_random(
        self,
        weights: "Array",
        n: int,
        scale: float,
    ) -> List[DirectionResult]:
        """Generate random perturbation directions.

        Random directions provide baseline exploration and can discover
        unexpected improvements. Normalized to unit Frobenius norm.
        """
        b = self._backend
        shape = b.shape(weights)
        results = []

        for _ in range(n):
            # Random direction with same shape as weights
            direction = b.random_normal(tuple(shape))
            b.eval(direction)

            # Normalize to unit Frobenius norm
            norm = b.sqrt(b.sum(direction * direction))
            b.eval(norm)
            norm_val = float(b.to_scalar(norm))

            eps = machine_epsilon(b, direction)
            if norm_val > eps:
                direction = direction / norm_val

            results.append(DirectionResult(
                direction=direction * scale,
                strategy=DirectionStrategy.RANDOM,
                scale=scale,
            ))

        return results

    def _generate_constant_aligned(
        self,
        weights: "Array",
        U: "Array",
        S: "Array",
        Vt: "Array",
        n: int,
        scale: float,
    ) -> List[DirectionResult]:
        """Generate directions that align SVD ratios to fundamental constants.

        For each target constant, find the singular value pair that is closest
        but not matching, and generate a direction that moves it closer.
        """
        b = self._backend
        results = []
        n_sv = int(S.shape[0])

        if n_sv < 2:
            return results

        # Find best alignment targets for each constant
        targets = []
        for const in self._target_constants:
            # Find singular value pair closest to this constant
            best_pair = None
            best_error = float("inf")

            for i in range(min(n_sv - 1, 15)):  # Check first 15 pairs
                for j in range(i + 1, min(n_sv, i + 8)):  # Check up to gap 7
                    s_i = float(b.to_scalar(S[i:i+1]))
                    s_j = float(b.to_scalar(S[j:j+1]))

                    if s_j < machine_epsilon(b, S):
                        continue

                    ratio = s_i / s_j
                    error = percent_error(ratio, const.value)

                    # Only consider pairs that aren't already aligned
                    if 1.0 < error < best_error:
                        best_error = error
                        best_pair = (i, j, ratio, const)

            if best_pair is not None:
                targets.append(best_pair)

        # Sort by error (biggest gaps first = most room for improvement)
        targets.sort(key=lambda x: -x[2] if x[2] > 0 else 0)

        # Generate directions for top n targets
        for i, (idx_i, idx_j, current_ratio, const) in enumerate(targets[:n]):
            target_ratio = const.value

            # Compute how much to adjust S[i] and S[j]
            s_i = float(b.to_scalar(S[idx_i:idx_i+1]))
            s_j = float(b.to_scalar(S[idx_j:idx_j+1]))

            # Target: S[i] / S[j] = target_ratio
            # We can either increase S[i] or decrease S[j]
            # Choose the smaller adjustment
            option1_new_si = target_ratio * s_j
            option2_new_sj = s_i / target_ratio

            delta_si = option1_new_si - s_i
            delta_sj = option2_new_sj - s_j

            # Create modified S vector
            S_new = b.array([float(b.to_scalar(S[k:k+1])) for k in range(n_sv)])

            # Use the option with smaller relative change
            if abs(delta_si / s_i) < abs(delta_sj / s_j):
                # Adjust S[i]
                S_new = b.concatenate([
                    S_new[:idx_i],
                    b.array([s_i + delta_si * scale]),
                    S_new[idx_i+1:],
                ], axis=0)
            else:
                # Adjust S[j]
                S_new = b.concatenate([
                    S_new[:idx_j],
                    b.array([s_j + delta_sj * scale]),
                    S_new[idx_j+1:],
                ], axis=0)

            b.eval(S_new)

            # Reconstruct weight direction: W_new = U @ diag(S_new) @ Vt
            # Direction = W_new - W = U @ diag(S_new - S) @ Vt
            S_delta = S_new - S
            direction = b.matmul(U * S_delta, Vt)
            b.eval(direction)

            # Estimate entropy reduction (heuristic: larger error reduction = better)
            current_error = percent_error(current_ratio, target_ratio)
            estimated_reduction = current_error * 0.01  # Rough estimate

            results.append(DirectionResult(
                direction=direction,
                strategy=DirectionStrategy.CONSTANT_ALIGNED,
                scale=scale,
                target_constant=const,
                target_ratio_indices=(idx_i, idx_j),
                expected_entropy_reduction=estimated_reduction,
            ))

        return results

    def _generate_svd_gap(
        self,
        weights: "Array",
        U: "Array",
        S: "Array",
        Vt: "Array",
        n: int,
        scale: float,
    ) -> List[DirectionResult]:
        """Generate directions that fill gaps in the SVD spectrum.

        Finds singular value pairs with unusually large gaps and smooths them.
        A smoother spectrum often indicates more coherent representation.
        """
        b = self._backend
        results = []
        n_sv = int(S.shape[0])

        if n_sv < 3:
            return results

        # Compute consecutive ratios
        ratios = []
        for i in range(min(n_sv - 1, 20)):
            s_i = float(b.to_scalar(S[i:i+1]))
            s_next = float(b.to_scalar(S[i+1:i+2]))
            if s_next > machine_epsilon(b, S):
                ratio = s_i / s_next
                ratios.append((i, ratio))

        if not ratios:
            return results

        # Find largest gaps (ratios far from 1.0)
        ratios.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)

        # Generate directions to smooth the top n gaps
        for idx, (i, ratio) in enumerate(ratios[:n]):
            s_i = float(b.to_scalar(S[i:i+1]))
            s_next = float(b.to_scalar(S[i+1:i+2]))

            # Target: geometric mean of neighbors
            target = math.sqrt(s_i * s_next)

            # Adjust S[i] toward target
            delta = (target - s_i) * scale

            # Create modified S vector
            S_list = [float(b.to_scalar(S[k:k+1])) for k in range(n_sv)]
            S_list[i] = s_i + delta
            S_new = b.array(S_list)
            b.eval(S_new)

            # Direction = U @ diag(S_new - S) @ Vt
            S_delta = S_new - S
            direction = b.matmul(U * S_delta, Vt)
            b.eval(direction)

            results.append(DirectionResult(
                direction=direction,
                strategy=DirectionStrategy.SVD_GAP,
                scale=scale,
                target_ratio_indices=(i, i + 1),
            ))

        return results

    def _generate_spectral_compress(
        self,
        weights: "Array",
        U: "Array",
        S: "Array",
        Vt: "Array",
        n: int,
        scale: float,
    ) -> List[DirectionResult]:
        """Generate directions that reduce spectral entropy.

        Spectral entropy measures the "spread" of singular values.
        Lower entropy = more concentrated spectrum = potentially more coherent.

        Strategy: Boost top singular values, suppress small ones.
        """
        b = self._backend
        results = []
        n_sv = int(S.shape[0])

        if n_sv < 2:
            return results

        # Compute spectral entropy
        S_sq = S * S
        total = b.sum(S_sq)
        b.eval(total)
        total_val = float(b.to_scalar(total))

        if total_val < machine_epsilon(b, S):
            return results

        p = S_sq / total_val
        b.eval(p)

        for direction_idx in range(n):
            # Different compression strengths
            strength = 0.5 + 0.5 * (direction_idx / max(1, n - 1))

            # Boost top-k singular values, suppress rest
            k = max(1, n_sv // (4 + direction_idx))  # Varying k

            # Create modified S: S_new[i] = S[i] * (1 + delta) for i < k
            #                    S_new[i] = S[i] * (1 - delta) for i >= k
            S_list = []
            for i in range(n_sv):
                s_i = float(b.to_scalar(S[i:i+1]))
                if i < k:
                    S_list.append(s_i * (1 + scale * strength))
                else:
                    S_list.append(s_i * (1 - scale * strength * 0.5))

            S_new = b.array(S_list)
            b.eval(S_new)

            # Direction = U @ diag(S_new - S) @ Vt
            S_delta = S_new - S
            direction = b.matmul(U * S_delta, Vt)
            b.eval(direction)

            results.append(DirectionResult(
                direction=direction,
                strategy=DirectionStrategy.SPECTRAL_COMPRESS,
                scale=scale,
            ))

        return results


__all__ = [
    "DirectionGenerator",
    "DirectionResult",
    "DirectionStrategy",
]
