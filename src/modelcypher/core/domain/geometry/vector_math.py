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
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return list(arr)


def _len(arr: ArrayLike) -> int:
    """Get length of array-like, handling MLX arrays."""
    if hasattr(arr, 'shape'):
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
