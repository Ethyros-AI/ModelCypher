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
Vector Math Utilities.

Provides common vector operations for geometry domain computations.
Supports both Python lists and MLX arrays as inputs.
"""

from __future__ import annotations

import math
from typing import Sequence

# Type alias for array-like inputs (list or MLX array)
ArrayLike = list[float] | Sequence[float]


def _to_list(arr: ArrayLike) -> list[float]:
    """Convert array-like to Python list, handling MLX arrays."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _len(arr: ArrayLike) -> int:
    """Get length of array-like, handling MLX arrays."""
    if hasattr(arr, "shape"):
        return arr.shape[0] if arr.shape else 0
    return len(arr)


class VectorMath:
    """Vector math utilities for dense vectors."""

    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> float | None:
        """Compute dot product of two vectors.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Dot product, or None if vectors are empty or different lengths.
        """
        len_a = _len(a)
        len_b = _len(b)
        if len_a != len_b or len_a == 0:
            return None

        a_list = _to_list(a)
        b_list = _to_list(b)
        return sum(x * y for x, y in zip(a_list, b_list))

    @staticmethod
    def l2_norm(a: ArrayLike) -> float | None:
        """Compute L2 norm of a vector.

        Args:
            a: Vector (list or MLX array)

        Returns:
            L2 norm, or None if vector is empty or all zeros.
        """
        if _len(a) == 0:
            return None

        a_list = _to_list(a)
        sum_squares = sum(x * x for x in a_list)
        if sum_squares <= 0:
            return None
        return math.sqrt(sum_squares)

    @staticmethod
    def l2_normalized(a: ArrayLike) -> list[float]:
        """Return L2-normalized vector.

        Args:
            a: Vector (list or MLX array)

        Returns:
            Normalized vector as Python list.
        """
        a_list = _to_list(a)
        norm = VectorMath.l2_norm(a_list)
        if norm is None or norm <= 0:
            return a_list
        inv_norm = 1.0 / norm
        return [x * inv_norm for x in a_list]

    @staticmethod
    def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float | None:
        """Compute cosine similarity between two vectors.

        Uses single-pass computation for efficiency.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Cosine similarity in [-1, 1], or None if vectors are invalid.
        """
        len_a = _len(a)
        len_b = _len(b)
        if len_a != len_b or len_a == 0:
            return None

        a_list = _to_list(a)
        b_list = _to_list(b)

        # Single-pass computation for efficiency
        dot_product = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0

        for i in range(len_a):
            x = float(a_list[i])
            y = float(b_list[i])
            dot_product += x * y
            norm_a_sq += x * x
            norm_b_sq += y * y

        if norm_a_sq <= 0 or norm_b_sq <= 0:
            return None

        return dot_product / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))

    @staticmethod
    def cosine_similarity_clamped(a: ArrayLike, b: ArrayLike) -> float:
        """Compute cosine similarity clamped to [0, 1].

        Useful when only non-negative similarity is meaningful.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Cosine similarity clamped to [0, 1], or 0.0 if invalid.
        """
        result = VectorMath.cosine_similarity(a, b)
        if result is None:
            return 0.0
        return max(0.0, min(1.0, result))

    @staticmethod
    def slerp(
        v0: ArrayLike,
        v1: ArrayLike,
        t: float,
        epsilon: float = 1e-6,
        interpolate_magnitude: bool = True,
    ) -> list[float] | None:
        """Spherical linear interpolation (SLERP) between two vectors.

        SLERP follows the geodesic (great circle arc) on the hypersphere,
        providing smoother interpolation than linear averaging for neural
        network weight merging.

        Formula: SLERP(v0, v1, t) = (sin((1-t)θ)/sinθ)v0 + (sin(tθ)/sinθ)v1
        where θ = arccos(v0·v1) is the angle between normalized vectors.

        Args:
            v0: First vector (list or MLX array)
            v1: Second vector (list or MLX array)
            t: Interpolation factor in [0, 1]. t=0 returns v0, t=1 returns v1.
            epsilon: Threshold for near-parallel detection. When angle < epsilon,
                     falls back to linear interpolation to avoid numerical issues.
            interpolate_magnitude: If True (default), interpolate magnitudes
                linearly. If False, return unit-normalized result.

        Returns:
            Interpolated vector as Python list, or None if vectors are invalid
            (empty, different lengths, or zero-norm).

        References:
            Shoemake, K. (1985). "Animating Rotation with Quaternion Curves."
            SIGGRAPH 1985, Computer Graphics, 19(3), 245-254.
        """
        len_v0 = _len(v0)
        len_v1 = _len(v1)
        if len_v0 != len_v1 or len_v0 == 0:
            return None

        v0_list = _to_list(v0)
        v1_list = _to_list(v1)

        # Compute magnitudes
        norm_v0 = VectorMath.l2_norm(v0_list)
        norm_v1 = VectorMath.l2_norm(v1_list)
        if norm_v0 is None or norm_v1 is None:
            return None

        # Normalize inputs
        inv_norm_v0 = 1.0 / norm_v0
        inv_norm_v1 = 1.0 / norm_v1
        v0_unit = [x * inv_norm_v0 for x in v0_list]
        v1_unit = [x * inv_norm_v1 for x in v1_list]

        # Compute dot product and clamp to [-1, 1] for numerical stability
        dot = sum(a * b for a, b in zip(v0_unit, v1_unit))
        dot = max(-1.0, min(1.0, dot))

        # Compute angle between vectors
        theta = math.acos(dot)

        # Handle near-parallel case (θ ≈ 0) - fall back to linear interpolation
        if theta < epsilon:
            result = [
                (1.0 - t) * v0_list[i] + t * v1_list[i] for i in range(len_v0)
            ]
            return result

        # Handle near-antipodal case (θ ≈ π) - SLERP is undefined
        # Use linear interpolation as fallback (not ideal but defined)
        if theta > math.pi - epsilon:
            result = [
                (1.0 - t) * v0_list[i] + t * v1_list[i] for i in range(len_v0)
            ]
            return result

        # SLERP formula: s0 * v0_unit + s1 * v1_unit
        sin_theta = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta

        result = [s0 * v0_unit[i] + s1 * v1_unit[i] for i in range(len_v0)]

        # Optionally rescale to interpolated magnitude
        if interpolate_magnitude:
            target_mag = (1.0 - t) * norm_v0 + t * norm_v1
            result = [x * target_mag for x in result]

        return result

    @staticmethod
    def slerp_batch(
        weights_a: dict[str, ArrayLike],
        weights_b: dict[str, ArrayLike],
        t: float,
        epsilon: float = 1e-6,
        interpolate_magnitude: bool = True,
    ) -> dict[str, list[float]]:
        """Apply SLERP to dictionaries of weight vectors (per-layer merging).

        Useful for merging model weights where each key corresponds to a layer.

        Args:
            weights_a: First model's weights as {layer_name: vector}
            weights_b: Second model's weights as {layer_name: vector}
            t: Interpolation factor in [0, 1]
            epsilon: Threshold for near-parallel detection
            interpolate_magnitude: Whether to interpolate magnitudes

        Returns:
            Merged weights as {layer_name: interpolated_vector}.
            Keys present in only one dict are included unchanged.
            Keys with incompatible vectors are skipped with a warning.
        """
        result: dict[str, list[float]] = {}
        all_keys = set(weights_a.keys()) | set(weights_b.keys())

        for key in all_keys:
            if key not in weights_a:
                # Only in weights_b
                result[key] = _to_list(weights_b[key])
            elif key not in weights_b:
                # Only in weights_a
                result[key] = _to_list(weights_a[key])
            else:
                # Present in both - apply SLERP
                merged = VectorMath.slerp(
                    weights_a[key],
                    weights_b[key],
                    t,
                    epsilon=epsilon,
                    interpolate_magnitude=interpolate_magnitude,
                )
                if merged is not None:
                    result[key] = merged
                else:
                    # Incompatible vectors - skip (caller should handle)
                    pass

        return result

    @staticmethod
    def _rankdata(values: list[float]) -> list[float]:
        """Compute average ranks for values (ties get averaged ranks)."""
        n = len(values)
        if n == 0:
            return []

        sorted_pairs = sorted(enumerate(values), key=lambda x: (x[1], x[0]))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j + 1
        return ranks

    @staticmethod
    def spearman_correlation(a: ArrayLike, b: ArrayLike) -> float | None:
        """Compute Spearman rank correlation (Pearson on ranks)."""
        len_a = _len(a)
        len_b = _len(b)
        if len_a != len_b or len_a < 2:
            return None

        a_list = _to_list(a)
        b_list = _to_list(b)

        rank_a = VectorMath._rankdata([float(v) for v in a_list])
        rank_b = VectorMath._rankdata([float(v) for v in b_list])

        mean_a = sum(rank_a) / len_a
        mean_b = sum(rank_b) / len_b

        num = 0.0
        den_a = 0.0
        den_b = 0.0
        for i in range(len_a):
            da = rank_a[i] - mean_a
            db = rank_b[i] - mean_b
            num += da * db
            den_a += da * da
            den_b += db * db

        if den_a <= 0.0 or den_b <= 0.0:
            return None
        return num / math.sqrt(den_a * den_b)


# Sparse vector operations (for dict-based vectors)
# Key type can be any hashable (str, int, tuple, etc.)
SparseVector = dict  # dict[K, float] where K is hashable


class SparseVectorMath:
    """Sparse vector math utilities for dict-based vectors.

    Works with any hashable key type (str, int, tuple, etc.).
    This is the canonical implementation - do not duplicate elsewhere.
    """

    @staticmethod
    def l2_norm(vector: SparseVector) -> float | None:
        """Compute L2 norm of a sparse vector.

        Args:
            vector: Dict mapping keys to float values.

        Returns:
            L2 norm, or None if vector is empty or all zeros.
        """
        if not vector:
            return None
        sum_squares = sum(v * v for v in vector.values())
        if sum_squares <= 0:
            return None
        return math.sqrt(sum_squares)

    @staticmethod
    def cosine_similarity(a: SparseVector, b: SparseVector) -> float | None:
        """Compute cosine similarity between sparse vectors.

        This is the canonical implementation for sparse cosine similarity.
        Works with any hashable key type (str for labels, int for indices).

        Args:
            a: First sparse vector as dict.
            b: Second sparse vector as dict.

        Returns:
            Cosine similarity in [-1, 1], or None if vectors are invalid.
        """
        if not a or not b:
            return None

        norm_a = SparseVectorMath.l2_norm(a)
        norm_b = SparseVectorMath.l2_norm(b)

        if norm_a is None or norm_b is None or norm_a <= 0 or norm_b <= 0:
            return None

        # Iterate over smaller dict for efficiency
        smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
        dot = 0.0
        for key, val in smaller.items():
            if key in larger:
                dot += val * larger[key]

        return dot / (norm_a * norm_b)
