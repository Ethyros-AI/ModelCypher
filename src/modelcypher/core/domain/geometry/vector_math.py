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

Two implementations are provided:
- VectorMath: Pure Python fallback (always available)
- BackendVectorMath: GPU-accelerated via Backend protocol (preferred)

Use get_vector_math() to get the best available implementation.

NOTE: For cross-dimensional comparison of activation matrices (n_samples x n_features),
use CKA from modelcypher.core.domain.geometry.cka - it works via Gram matrices
which are dimension-independent. Single-vector operations require matching dimensions.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Sequence

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    sin_scalar,
    sqrt_scalar,
)

# math.pi is just a constant, no GPU acceleration needed
_PI = 3.141592653589793

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

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


def _to_scalar(val: Any) -> float:
    """Convert backend array scalar to Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    if hasattr(val, "tolist"):
        result = val.tolist()
        return float(result) if not isinstance(result, list) else float(result[0])
    return float(val)


def _angle_epsilon_from_values(values: list[float]) -> float:
    """Derive an angle tolerance from backend dtype precision."""
    backend = get_default_backend()
    ref = backend.array(values if values else [1.0])
    return division_epsilon(backend, ref)


class VectorMath:
    """Vector math utilities for dense vectors."""

    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> float:
        """Compute dot product of two vectors.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Dot product.

        Raises:
            ValueError: If vectors are empty or have different lengths.
        """
        len_a = _len(a)
        len_b = _len(b)
        if len_a == 0:
            raise ValueError("Cannot compute dot product of empty vectors")
        if len_a != len_b:
            raise ValueError(
                f"Dot product requires matching dimensions: got {len_a} vs {len_b}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

        a_list = _to_list(a)
        b_list = _to_list(b)
        return sum(x * y for x, y in zip(a_list, b_list))

    @staticmethod
    def l2_norm(a: ArrayLike) -> float:
        """Compute L2 norm of a vector.

        Args:
            a: Vector (list or MLX array)

        Returns:
            L2 norm (0.0 for zero vector).

        Raises:
            ValueError: If vector is empty.
        """
        if _len(a) == 0:
            raise ValueError("Cannot compute L2 norm of empty vector")

        a_list = _to_list(a)
        sum_squares = sum(x * x for x in a_list)
        _b = get_default_backend()
        return sqrt_scalar(max(0.0, sum_squares), _b)

    @staticmethod
    def l2_normalized(a: ArrayLike) -> list[float]:
        """Return L2-normalized vector.

        Args:
            a: Vector (list or MLX array)

        Returns:
            Normalized vector as Python list (unchanged if zero vector).

        Raises:
            ValueError: If vector is empty.
        """
        a_list = _to_list(a)
        norm = VectorMath.l2_norm(a_list)
        if norm <= 0:
            return a_list  # Return unchanged for zero vector
        inv_norm = 1.0 / norm
        return [x * inv_norm for x in a_list]

    @staticmethod
    def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
        """Compute cosine similarity between two vectors.

        Uses single-pass computation for efficiency.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Cosine similarity in [-1, 1].

        Raises:
            ValueError: If vectors are empty, have different lengths, or are zero.
        """
        len_a = _len(a)
        len_b = _len(b)
        if len_a == 0:
            raise ValueError("Cannot compute cosine similarity of empty vectors")
        if len_a != len_b:
            raise ValueError(
                f"Cosine similarity requires matching dimensions: got {len_a} vs {len_b}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

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

        eps = _angle_epsilon_from_values(a_list + b_list)
        if norm_a_sq <= eps or norm_b_sq <= eps:
            raise ValueError("Cannot compute cosine similarity of zero vector")

        _b = get_default_backend()
        return dot_product / (sqrt_scalar(norm_a_sq, _b) * sqrt_scalar(norm_b_sq, _b))

    @staticmethod
    def cosine_similarity_clamped(a: ArrayLike, b: ArrayLike) -> float:
        """Compute cosine similarity clamped to [0, 1].

        Useful when only non-negative similarity is meaningful.

        Args:
            a: First vector (list or MLX array)
            b: Second vector (list or MLX array)

        Returns:
            Cosine similarity clamped to [0, 1].

        Raises:
            ValueError: If vectors are empty, have different lengths, or are zero.
        """
        result = VectorMath.cosine_similarity(a, b)
        return max(0.0, min(1.0, result))

    @staticmethod
    def slerp(
        v0: ArrayLike,
        v1: ArrayLike,
        t: float,
        epsilon: float | None = None,
        interpolate_magnitude: bool = True,
    ) -> list[float]:
        """Spherical linear interpolation (SLERP) between two vectors.

        SLERP follows the geodesic (great circle arc) on the hypersphere.
        Useful for animation and visualization of smooth transitions.

        WARNING: Do NOT use SLERP for model weight merging. SLERP interpolates
        between vectors, destroying information from both. For model merging,
        use null space addition: target + null_space_projection(source - target).

        Formula: SLERP(v0, v1, t) = (sin((1-t)θ)/sinθ)v0 + (sin(tθ)/sinθ)v1
        where θ = arccos(v0·v1) is the angle between normalized vectors.

        Args:
            v0: First vector (list or MLX array)
            v1: Second vector (list or MLX array)
            t: Interpolation factor in [0, 1]. t=0 returns v0, t=1 returns v1.
            epsilon: Threshold for near-parallel detection. When angle < epsilon,
                     falls back to linear interpolation to avoid numerical issues.
                     If None, derived from dtype precision.
            interpolate_magnitude: If True (default), interpolate magnitudes
                linearly. If False, return unit-normalized result.

        Returns:
            Interpolated vector as Python list.

        Raises:
            ValueError: If vectors are empty, have different lengths, or are zero.

        References:
            Shoemake, K. (1985). "Animating Rotation with Quaternion Curves."
            SIGGRAPH 1985, Computer Graphics, 19(3), 245-254.
        """
        len_v0 = _len(v0)
        len_v1 = _len(v1)
        if len_v0 == 0:
            raise ValueError("Cannot SLERP empty vectors")
        if len_v0 != len_v1:
            raise ValueError(
                f"SLERP requires matching dimensions: got {len_v0} vs {len_v1}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

        v0_list = _to_list(v0)
        v1_list = _to_list(v1)

        if epsilon is None:
            epsilon = _angle_epsilon_from_values(v0_list)

        # Compute magnitudes (will raise if zero)
        norm_v0 = VectorMath.l2_norm(v0_list)
        norm_v1 = VectorMath.l2_norm(v1_list)
        if norm_v0 <= 0 or norm_v1 <= 0:
            raise ValueError("Cannot SLERP zero vectors")

        # Normalize inputs
        inv_norm_v0 = 1.0 / norm_v0
        inv_norm_v1 = 1.0 / norm_v1
        v0_unit = [x * inv_norm_v0 for x in v0_list]
        v1_unit = [x * inv_norm_v1 for x in v1_list]

        # Compute dot product and clamp to [-1, 1] for numerical stability
        dot = sum(a * b for a, b in zip(v0_unit, v1_unit))
        dot = max(-1.0, min(1.0, dot))

        # Compute angle between vectors
        _b = get_default_backend()
        theta = acos_scalar(dot, _b)

        # Handle near-parallel case (θ ≈ 0) - fall back to linear interpolation
        if theta < epsilon:
            result = [
                (1.0 - t) * v0_list[i] + t * v1_list[i] for i in range(len_v0)
            ]
            return result

        # Handle near-antipodal case (θ ≈ π) - SLERP is undefined
        # Use linear interpolation as fallback (not ideal but defined)
        if theta > _PI - epsilon:
            result = [
                (1.0 - t) * v0_list[i] + t * v1_list[i] for i in range(len_v0)
            ]
            return result

        # SLERP formula: s0 * v0_unit + s1 * v1_unit
        sin_theta = sin_scalar(theta, _b)
        s0 = sin_scalar((1.0 - t) * theta, _b) / sin_theta
        s1 = sin_scalar(t * theta, _b) / sin_theta

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
        epsilon: float | None = None,
        interpolate_magnitude: bool = True,
    ) -> dict[str, list[float]]:
        """Apply SLERP to dictionaries of vectors.

        WARNING: Do NOT use this for model weight merging. SLERP interpolates
        between vectors, destroying information from both. For model merging,
        use null space addition. This function is for visualization/animation.

        Args:
            weights_a: First dict of vectors as {name: vector}
            weights_b: Second dict of vectors as {name: vector}
            t: Interpolation factor in [0, 1]
            epsilon: Threshold for near-parallel detection. If None, derived from dtype.
            interpolate_magnitude: Whether to interpolate magnitudes

        Returns:
            Interpolated vectors as {name: interpolated_vector}.
            Keys present in only one dict are included unchanged.
        """
        result: dict[str, list[float]] = {}
        if epsilon is None:
            sample = next(iter(weights_a.values()), None)
            if sample is None:
                sample = next(iter(weights_b.values()), None)
            if sample is not None:
                epsilon = _angle_epsilon_from_values(_to_list(sample))
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
    def spearman_correlation(a: ArrayLike, b: ArrayLike) -> float:
        """Compute Spearman rank correlation (Pearson on ranks).

        Raises:
            ValueError: If vectors have different lengths, fewer than 2 elements,
                        or constant values (zero variance).
        """
        len_a = _len(a)
        len_b = _len(b)
        if len_a < 2:
            raise ValueError("Spearman correlation requires at least 2 elements")
        if len_a != len_b:
            raise ValueError(
                f"Spearman correlation requires matching dimensions: got {len_a} vs {len_b}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

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
            raise ValueError("Spearman correlation undefined for constant vectors (zero variance)")
        _b = get_default_backend()
        return num / sqrt_scalar(den_a * den_b, _b)


# Sparse vector operations (for dict-based vectors)
# Key type can be any hashable (str, int, tuple, etc.)
SparseVector = dict  # dict[K, float] where K is hashable


class SparseVectorMath:
    """Sparse vector math utilities for dict-based vectors.

    Works with any hashable key type (str, int, tuple, etc.).
    This is the canonical implementation - do not duplicate elsewhere.
    """

    @staticmethod
    def l2_norm(vector: SparseVector) -> float:
        """Compute L2 norm of a sparse vector.

        Args:
            vector: Dict mapping keys to float values.

        Returns:
            L2 norm (0.0 for zero vector).

        Raises:
            ValueError: If vector is empty (no keys).
        """
        if not vector:
            raise ValueError("Cannot compute L2 norm of empty sparse vector")
        sum_squares = sum(v * v for v in vector.values())
        _b = get_default_backend()
        return sqrt_scalar(max(0.0, sum_squares), _b)

    @staticmethod
    def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
        """Compute cosine similarity between sparse vectors.

        This is the canonical implementation for sparse cosine similarity.
        Works with any hashable key type (str for labels, int for indices).
        Sparse vectors with different key sets are valid - similarity is
        computed on the intersection (non-overlapping dimensions contribute 0).

        Args:
            a: First sparse vector as dict.
            b: Second sparse vector as dict.

        Returns:
            Cosine similarity in [-1, 1] (0.0 if no key overlap).

        Raises:
            ValueError: If either vector is empty or zero.
        """
        if not a or not b:
            raise ValueError("Cannot compute cosine similarity of empty sparse vectors")

        norm_a = SparseVectorMath.l2_norm(a)
        norm_b = SparseVectorMath.l2_norm(b)

        if norm_a <= 0 or norm_b <= 0:
            raise ValueError("Cannot compute cosine similarity of zero sparse vectors")

        # Iterate over smaller dict for efficiency
        smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
        dot = 0.0
        for key, val in smaller.items():
            if key in larger:
                dot += val * larger[key]

        return dot / (norm_a * norm_b)


class BackendVectorMath:
    """GPU-accelerated vector math using the Backend protocol.

    This class provides the same operations as VectorMath but uses
    Backend tensor operations for GPU acceleration. Use this for
    large vectors or batch operations.

    All operations work directly on Backend arrays without conversion
    to Python lists, enabling full GPU utilization.
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a Backend instance.

        Args:
            backend: Backend instance (MLXBackend, JAXBackend, etc.)
        """
        self.backend = backend
        # Cache finfo for numerical stability
        self._finfo = backend.finfo()

    def dot(self, a: Any, b: Any) -> float:
        """Compute dot product using backend operations.

        Args:
            a: First vector (Backend array or convertible)
            b: Second vector (Backend array or convertible)

        Returns:
            Dot product as Python float.

        Raises:
            ValueError: If arrays are invalid, have different shapes, or are empty.
        """
        # Convert to backend arrays if needed
        a_arr = self._ensure_array(a)
        b_arr = self._ensure_array(b)

        if a_arr is None or b_arr is None:
            raise ValueError("Cannot compute dot product of invalid arrays")

        shape_a = self.backend.shape(a_arr)
        shape_b = self.backend.shape(b_arr)

        if len(shape_a) != 1 or len(shape_b) != 1:
            raise ValueError("Dot product requires 1D arrays")
        if shape_a[0] == 0:
            raise ValueError("Cannot compute dot product of empty arrays")
        if shape_a[0] != shape_b[0]:
            raise ValueError(
                f"Dot product requires matching dimensions: got {shape_a[0]} vs {shape_b[0]}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

        result = self.backend.dot(a_arr, b_arr)
        self.backend.eval(result)
        return float(self.backend.to_scalar(result))

    def l2_norm(self, a: Any) -> float:
        """Compute L2 norm using backend operations.

        Args:
            a: Vector (Backend array or convertible)

        Returns:
            L2 norm as Python float (0.0 for zero vector).

        Raises:
            ValueError: If array is invalid or empty.
        """
        a_arr = self._ensure_array(a)
        if a_arr is None:
            raise ValueError("Cannot compute L2 norm of invalid array")

        shape = self.backend.shape(a_arr)
        if len(shape) != 1:
            raise ValueError("L2 norm requires 1D array")
        if shape[0] == 0:
            raise ValueError("Cannot compute L2 norm of empty array")

        result = self.backend.norm(a_arr)
        self.backend.eval(result)
        return max(0.0, float(self.backend.to_scalar(result)))

    def l2_normalized(self, a: Any) -> Any:
        """Return L2-normalized vector using backend operations.

        Args:
            a: Vector (Backend array or convertible)

        Returns:
            Normalized vector as Backend array.
        """
        a_arr = self._ensure_array(a)
        if a_arr is None:
            return a

        norm = self.backend.norm(a_arr)
        self.backend.eval(norm)
        norm_val = float(self.backend.to_scalar(norm))

        if norm_val <= self._finfo.eps:
            return a_arr

        return a_arr / norm

    def cosine_similarity(self, a: Any, b: Any) -> float:
        """Compute cosine similarity using backend operations.

        Args:
            a: First vector (Backend array or convertible)
            b: Second vector (Backend array or convertible)

        Returns:
            Cosine similarity in [-1, 1].

        Raises:
            ValueError: If arrays are invalid, have different shapes, are empty, or zero.
        """
        a_arr = self._ensure_array(a)
        b_arr = self._ensure_array(b)

        if a_arr is None or b_arr is None:
            raise ValueError("Cannot compute cosine similarity of invalid arrays")

        shape_a = self.backend.shape(a_arr)
        shape_b = self.backend.shape(b_arr)

        if len(shape_a) != 1 or len(shape_b) != 1:
            raise ValueError("Cosine similarity requires 1D arrays")
        if shape_a[0] == 0:
            raise ValueError("Cannot compute cosine similarity of empty arrays")
        if shape_a[0] != shape_b[0]:
            raise ValueError(
                f"Cosine similarity requires matching dimensions: got {shape_a[0]} vs {shape_b[0]}. "
                "For cross-dimensional comparison, use CKA on activation matrices."
            )

        # Compute norms
        norm_a = self.backend.norm(a_arr)
        norm_b = self.backend.norm(b_arr)
        self.backend.eval(norm_a, norm_b)

        norm_a_val = float(self.backend.to_scalar(norm_a))
        norm_b_val = float(self.backend.to_scalar(norm_b))

        if norm_a_val <= self._finfo.eps or norm_b_val <= self._finfo.eps:
            raise ValueError("Cannot compute cosine similarity of zero vector")

        # Compute dot product
        dot = self.backend.dot(a_arr, b_arr)
        self.backend.eval(dot)
        dot_val = float(self.backend.to_scalar(dot))

        return dot_val / (norm_a_val * norm_b_val)

    def slerp(
        self,
        v0: Any,
        v1: Any,
        t: float,
        epsilon: float | None = None,
        interpolate_magnitude: bool = True,
    ) -> Any | None:
        """Spherical linear interpolation using backend operations.

        GPU-accelerated SLERP for animation and visualization.
        Formula: SLERP(v0, v1, t) = (sin((1-t)θ)/sinθ)v0 + (sin(tθ)/sinθ)v1

        WARNING: Do NOT use for model weight merging. SLERP interpolates,
        destroying information. Use null space addition for merging.

        Args:
            v0: First vector (Backend array or convertible)
            v1: Second vector (Backend array or convertible)
            t: Interpolation factor in [0, 1]
            epsilon: Threshold for near-parallel detection. If None, uses
                     dtype-derived epsilon.
            interpolate_magnitude: If True, interpolate magnitudes linearly.

        Returns:
            Interpolated vector as Backend array, or None if invalid.
        """
        v0_arr = self._ensure_array(v0)
        v1_arr = self._ensure_array(v1)

        if v0_arr is None or v1_arr is None:
            return None

        if epsilon is None:
            epsilon = division_epsilon(self.backend, v0_arr)

        shape_v0 = self.backend.shape(v0_arr)
        shape_v1 = self.backend.shape(v1_arr)

        if shape_v0 != shape_v1 or len(shape_v0) != 1 or shape_v0[0] == 0:
            return None

        # Compute magnitudes
        norm_v0 = self.backend.norm(v0_arr)
        norm_v1 = self.backend.norm(v1_arr)
        self.backend.eval(norm_v0, norm_v1)

        norm_v0_val = float(self.backend.to_scalar(norm_v0))
        norm_v1_val = float(self.backend.to_scalar(norm_v1))

        if norm_v0_val <= self._finfo.eps or norm_v1_val <= self._finfo.eps:
            return None

        # Normalize inputs
        v0_unit = v0_arr / norm_v0
        v1_unit = v1_arr / norm_v1

        # Compute dot product and clamp to [-1, 1]
        dot = self.backend.dot(v0_unit, v1_unit)
        dot_clamped = self.backend.clip(dot, -1.0, 1.0)
        self.backend.eval(dot_clamped)
        dot_val = float(self.backend.to_scalar(dot_clamped))

        # Compute angle
        theta = acos_scalar(dot_val, self.backend)

        # Handle edge cases
        if theta < epsilon or theta > _PI - epsilon:
            # Near-parallel or near-antipodal: fall back to linear
            result = v0_arr * (1.0 - t) + v1_arr * t
            self.backend.eval(result)
            return result

        # SLERP formula using backend trig functions
        sin_theta = sin_scalar(theta, self.backend)
        s0 = sin_scalar((1.0 - t) * theta, self.backend) / sin_theta
        s1 = sin_scalar(t * theta, self.backend) / sin_theta

        result = v0_unit * s0 + v1_unit * s1

        # Optionally rescale to interpolated magnitude
        if interpolate_magnitude:
            target_mag = (1.0 - t) * norm_v0_val + t * norm_v1_val
            result = result * target_mag

        self.backend.eval(result)
        return result

    def slerp_matrix(
        self,
        m0: Any,
        m1: Any,
        t: float,
        epsilon: float | None = None,
    ) -> tuple[Any, dict[str, float | str]] | None:
        """Spherical linear interpolation for 2D matrices.

        Treats each matrix as a high-dimensional vector and applies SLERP.
        Useful for visualization of transformation paths, NOT model merging.

        WARNING: Do NOT use for model weight merging. SLERP interpolates,
        destroying information from both matrices. For merging, use null space
        addition: target + null_space_projection(source - target).

        For matrices M₀ and M₁:
        1. Flatten to vectors v₀, v₁
        2. Compute angle θ = arccos(v₀·v₁ / (||v₀|| ||v₁||))
        3. SLERP: v_merged = (sin((1-t)θ)/sinθ)v₀ + (sin(tθ)/sinθ)v₁
        4. Reshape back to matrix

        Args:
            m0: First matrix (Backend array, shape [m, n])
            m1: Second matrix (Backend array, shape [m, n])
            t: Interpolation factor in [0, 1]. t=0 returns m0, t=1 returns m1.
            epsilon: Threshold for near-parallel detection.

        Returns:
            Tuple of (interpolated_matrix, metrics), or None if invalid.
            Metrics include: angle_deg, interpolation_mode, magnitude_ratio.
        """
        m0_arr = self._ensure_array(m0)
        m1_arr = self._ensure_array(m1)

        if m0_arr is None or m1_arr is None:
            return None

        if epsilon is None:
            epsilon = division_epsilon(self.backend, m0_arr)

        shape_m0 = self.backend.shape(m0_arr)
        shape_m1 = self.backend.shape(m1_arr)

        if shape_m0 != shape_m1 or len(shape_m0) != 2:
            return None

        # Flatten matrices to vectors
        v0 = self.backend.reshape(m0_arr, (-1,))
        v1 = self.backend.reshape(m1_arr, (-1,))

        # Compute magnitudes (Frobenius norms)
        norm_v0 = self.backend.norm(v0)
        norm_v1 = self.backend.norm(v1)
        self.backend.eval(norm_v0, norm_v1)

        norm_v0_val = float(self.backend.to_scalar(norm_v0))
        norm_v1_val = float(self.backend.to_scalar(norm_v1))

        if norm_v0_val <= self._finfo.eps or norm_v1_val <= self._finfo.eps:
            return None

        # Normalize
        v0_unit = v0 / norm_v0
        v1_unit = v1 / norm_v1

        # Compute cosine similarity (dot product of unit vectors)
        dot = self.backend.dot(v0_unit, v1_unit)
        dot_clamped = self.backend.clip(dot, -1.0, 1.0)
        self.backend.eval(dot_clamped)
        dot_val = float(self.backend.to_scalar(dot_clamped))

        # Compute angle
        theta = acos_scalar(dot_val, self.backend)
        angle_deg = theta * 180.0 / _PI  # degrees conversion

        metrics: dict[str, float | str] = {
            "angle_deg": angle_deg,
            "magnitude_ratio": norm_v0_val
            / (norm_v1_val + division_epsilon(self.backend, v1)),
            "cosine_similarity": dot_val,
        }

        # Handle edge cases
        if theta < epsilon:
            # Near-identical: linear interpolation
            result_flat = v0 * (1.0 - t) + v1 * t
            metrics["interpolation_mode"] = "linear_parallel"
        elif theta > _PI - epsilon:
            # Near-antipodal: linear interpolation (SLERP undefined)
            result_flat = v0 * (1.0 - t) + v1 * t
            metrics["interpolation_mode"] = "linear_antipodal"
        else:
            # Standard SLERP
            sin_theta = sin_scalar(theta, self.backend)
            s0 = sin_scalar((1.0 - t) * theta, self.backend) / sin_theta
            s1 = sin_scalar(t * theta, self.backend) / sin_theta

            # Interpolate on unit sphere then rescale
            result_unit = v0_unit * s0 + v1_unit * s1

            # Interpolate magnitude linearly
            target_mag = (1.0 - t) * norm_v0_val + t * norm_v1_val
            result_flat = result_unit * target_mag
            metrics["interpolation_mode"] = "slerp"

        self.backend.eval(result_flat)

        # Reshape back to matrix
        result = self.backend.reshape(result_flat, shape_m0)
        self.backend.eval(result)

        return result, metrics

    def slerp_batch(
        self,
        weights_a: dict[str, Any],
        weights_b: dict[str, Any],
        t: float,
        epsilon: float | None = None,
        interpolate_magnitude: bool = True,
    ) -> dict[str, Any]:
        """Apply SLERP to dictionaries of vectors.

        GPU-accelerated batch SLERP for visualization/animation.

        WARNING: Do NOT use for model weight merging. SLERP interpolates,
        destroying information. Use null space addition for merging.

        Args:
            weights_a: First dict of arrays as {name: array}
            weights_b: Second dict of arrays as {name: array}
            t: Interpolation factor in [0, 1]
            epsilon: Threshold for near-parallel detection
            interpolate_magnitude: Whether to interpolate magnitudes

        Returns:
            Interpolated arrays as {name: interpolated_array}.
        """
        result: dict[str, Any] = {}
        all_keys = set(weights_a.keys()) | set(weights_b.keys())

        for key in all_keys:
            if key not in weights_a:
                result[key] = weights_b[key]
            elif key not in weights_b:
                result[key] = weights_a[key]
            else:
                merged = self.slerp(
                    weights_a[key],
                    weights_b[key],
                    t,
                    epsilon=epsilon,
                    interpolate_magnitude=interpolate_magnitude,
                )
                if merged is not None:
                    result[key] = merged

        return result

    def _ensure_array(self, data: Any) -> Any | None:
        """Convert data to Backend array if needed.

        Args:
            data: Input data (list, array, or Backend array)

        Returns:
            Backend array, or None if conversion fails.
        """
        if data is None:
            return None

        # Check if already a backend array (has shape attribute)
        if hasattr(data, "shape"):
            return data

        # Convert from list/sequence
        try:
            return self.backend.array(data)
        except (TypeError, ValueError):
            return None


def get_vector_math(backend: "Backend | None" = None) -> VectorMath | BackendVectorMath:
    """Get the best available vector math implementation.

    Args:
        backend: Optional Backend instance. If provided, returns
                 BackendVectorMath for GPU acceleration. If None,
                 returns the pure Python VectorMath.

    Returns:
        VectorMath or BackendVectorMath instance.

    Example:
        >>> from modelcypher.core.domain._backend import get_default_backend
        >>> backend = get_default_backend()
        >>> vm = get_vector_math(backend)
        >>> result = vm.slerp(v0, v1, 0.5)
    """
    if backend is not None:
        return BackendVectorMath(backend)
    return VectorMath()
