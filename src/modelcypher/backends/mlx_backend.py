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

"""MLX backend with SOTA performance optimizations.

Performance Features (MLX 0.30+):
- mx.compile: JIT compilation with kernel fusion (5x speedup)
- mx.fast.*: Fused Metal kernels (rms_norm, layer_norm, rope, scaled_dot_product_attention)
- mx.vmap: Auto-vectorization for batch operations (200x speedup vs loops)
- mx.async_eval: Pipeline parallelism for multi-layer analysis
- Streams: CPU/GPU parallelism for data loading + compute
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.backends.conversion_utils import (
    raise_numpy_disabled,
    to_list_with_eval,
    to_scalar_with_eval,
)
from modelcypher.ports.backend import Array, Backend, FloatInfo

if TYPE_CHECKING:
    pass


class MLXBackend(Backend):
    """MLX backend implementing the Backend protocol with lazy evaluation.

    Provides MLX-specific performance features including JIT compilation,
    vectorization, and fused Metal kernels. Operations return lazy arrays;
    callers must explicitly call eval() to trigger computation.
    """

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self.fast = mx.fast  # Expose mx.fast for fused kernels
        self._compiled_cache: dict[str, Callable] = {}

    # --- Array Creation (lazy - no eval) ---
    def array(self, data: Any, dtype: Any | None = None) -> Array:
        mapped_dtype = self._map_dtype(dtype)

        # Fast-path: already an MLX array
        module = type(data).__module__
        if module.startswith("mlx"):
            if mapped_dtype is None:
                return data
            # Avoid unnecessary copies when dtype already matches
            try:
                if getattr(data, "dtype", None) == mapped_dtype:
                    return data
            except Exception:
                pass
            return data.astype(mapped_dtype)

        # Numpy scalars -> Python scalars
        if module.startswith("numpy") and hasattr(data, "item") and not hasattr(data, "shape"):
            data = data.item()
            module = type(data).__module__

        # Fast-path: array-like objects (including numpy arrays without importing numpy).
        if hasattr(data, "__array__") and not module.startswith("mlx"):
            try:
                return self.mx.array(data, dtype=mapped_dtype)
            except RuntimeError as exc:
                if "bad_cast" not in str(exc):
                    raise
                if hasattr(data, "tolist"):
                    return self.mx.array(data.tolist(), dtype=mapped_dtype)
                raise

        # MLX doesn't accept nested lists of numpy scalars (e.g., numpy.float32)
        if isinstance(data, list) and data:
            first = data[0]
            while isinstance(first, list) and first:
                first = first[0]
            if hasattr(first, "item") and type(first).__module__.startswith("numpy"):
                def _coerce(value: Any) -> Any:
                    if isinstance(value, list):
                        return [_coerce(item) for item in value]
                    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
                        return value.item()
                    return value

                data = _coerce(data)

        try:
            return self.mx.array(data, dtype=mapped_dtype)
        except RuntimeError as exc:
            if "bad_cast" not in str(exc):
                raise
            # Large integers (> 2^31) can't be converted directly.
            # Convert via Python float first, use explicit float64 for precision
            fallback_dtype = mapped_dtype if mapped_dtype is not None else self.mx.float64
            if hasattr(data, "tolist"):
                data = data.tolist()

            def _to_float_recursive(val: Any) -> Any:
                if isinstance(val, list):
                    return [_to_float_recursive(v) for v in val]
                return float(val)

            if isinstance(data, list):
                float_data = _to_float_recursive(data)
                return self.mx.array(float_data, dtype=fallback_dtype)
            return self.mx.array([float(data)], dtype=fallback_dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.mx.zeros(shape, dtype=self._map_dtype(dtype))

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.mx.ones(shape, dtype=self._map_dtype(dtype))

    # --- Shape Manipulation (lazy - no eval) ---
    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.shape)

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.mx.reshape(array, shape)

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        return self.mx.squeeze(array, axis=axis) if axis is not None else self.mx.squeeze(array)

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        return self.mx.transpose(array, axes=axes) if axes else self.mx.transpose(array)

    # --- Core Operations (lazy - no eval) ---
    def matmul(self, lhs: Array, rhs: Array) -> Array:
        return self.mx.matmul(lhs, rhs)

    def sum(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.mx.sum(array, axis=axis, keepdims=keepdims)

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.mx.max(array, axis=axis, keepdims=keepdims)

    def sqrt(self, array: Array) -> Array:
        return self.mx.sqrt(array)

    def exp(self, array: Array) -> Array:
        return self.mx.exp(array)

    def log(self, array: Array) -> Array:
        return self.mx.log(array)

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return self.mx.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return self.mx.minimum(lhs, rhs)

    def add(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise addition."""
        return self.mx.add(lhs, rhs)

    def subtract(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise subtraction."""
        return self.mx.subtract(lhs, rhs)

    def multiply(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise multiplication."""
        return self.mx.multiply(lhs, rhs)

    def divide(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise division."""
        return self.mx.divide(lhs, rhs)

    def abs(self, array: Array) -> Array:
        return self.mx.abs(array)

    def astype(self, array: Array, dtype: Any) -> Array:
        # Ensure array is an MLX array before calling astype with MLX dtype
        # Numpy arrays have astype but don't understand MLX dtypes
        module = type(array).__module__
        if module.startswith("numpy") or not module.startswith("mlx"):
            array = self.mx.array(array)
        return array.astype(self._map_dtype(dtype))

    # --- Linear Algebra (eval to force compute) ---
    def svd(
        self,
        array: Array,
        compute_uv: bool = True,
        full_matrices: bool | None = None,
    ) -> tuple[Array, Array, Array] | Array:
        # Accept both compute_uv and full_matrices for compatibility
        # full_matrices=False is equivalent to compute_uv=True (compact SVD)
        # full_matrices=True means compute full U and Vt matrices
        if full_matrices is not None:
            # MLX's compute_uv means "return U, S, Vt" not "use full matrices"
            # We always want compact SVD, so compute_uv=True
            compute_uv = True

        # MLX SVD supports batched inputs; use device stream when available.
        try:
            result = self.mx.linalg.svd(array, compute_uv=compute_uv)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            result = self.mx.linalg.svd(array, compute_uv=compute_uv, stream=self.mx.cpu)
        if compute_uv:
            u, s, vt = result
            self.mx.eval(u, s, vt)
            return u, s, vt
        self.mx.eval(result)
        return result

    def floyd_warshall(self, dist: Array) -> Array:
        """Compute all-pairs shortest paths using Floyd-Warshall on device."""
        mx = self.mx
        dist_arr = dist if type(dist).__module__.startswith("mlx") else mx.array(dist)
        if dist_arr.ndim != 2 or dist_arr.shape[0] != dist_arr.shape[1]:
            raise ValueError("floyd_warshall requires a square [n, n] matrix")
        n = int(dist_arr.shape[0])
        if n <= 1:
            return dist_arr

        cache_key = f"floyd_warshall_{n}_{dist_arr.dtype}"
        compiled = self._compiled_cache.get(cache_key)
        if compiled is None:
            def _fw(mat: Array) -> Array:
                out = mat
                for k in range(n):
                    via = out[:, k : k + 1] + out[k : k + 1, :]
                    out = mx.minimum(out, via)
                return out

            compiled = self.compile(_fw)
            self._compiled_cache[cache_key] = compiled

        return compiled(dist_arr)

    def single_source_shortest_paths(self, dist: Array, source_index: int) -> Array:
        """Compute shortest paths from a single source using Dijkstra-style relaxation."""
        mx = self.mx
        dist_arr = dist if type(dist).__module__.startswith("mlx") else mx.array(dist)
        if dist_arr.ndim != 2 or dist_arr.shape[0] != dist_arr.shape[1]:
            raise ValueError("single_source_shortest_paths requires a square [n, n] matrix")
        n = int(dist_arr.shape[0])
        if n <= 1:
            return dist_arr[0] if n == 1 else dist_arr

        src = int(source_index)
        if src < 0 or src >= n:
            raise ValueError("source_index out of bounds")

        idx = mx.arange(n)
        dist_vec = dist_arr[src]
        visited = self.astype(idx == src, dist_arr.dtype)
        inf_val = mx.array(mx.finfo(dist_arr.dtype).max)

        for _ in range(n - 1):
            masked = dist_vec + visited * inf_val
            min_idx = mx.argmin(masked)
            is_min = idx == min_idx
            visited = mx.minimum(visited + self.astype(is_min, dist_arr.dtype), 1.0)

            row = mx.take(dist_arr, min_idx, axis=0)
            dist_at_min = mx.take(dist_vec, min_idx, axis=0)
            alt = dist_at_min + row
            dist_vec = mx.minimum(dist_vec, alt)

        return dist_vec

    # --- Quantization (lazy - no eval) ---
    def quantize(
        self,
        weight: Array,
        group_size: int,
        bits: int,
        mode: str,
    ) -> tuple[Array, Array, Array | None]:
        result = self.mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
        if len(result) == 2:
            weight_q, scales = result
            return weight_q, scales, None
        return result

    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        return self.mx.dequantize(
            weight,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    # --- Explicit Evaluation ---
    def eval(self, *arrays: Array) -> None:
        """Evaluate arrays - triggers kernel fusion and GPU execution."""
        self.mx.eval(*arrays)

    def clear_cache(self) -> None:
        """Clear Metal GPU memory cache to release lazy computations.

        Note: We intentionally do NOT call gc.collect() here. Triggering Python's
        garbage collector during MLX operations can cause segfaults due to reentrancy
        issues when the GC tries to finalize MLX arrays that are still in use.
        Let Python's GC run on its own schedule; mx.clear_cache() is sufficient
        for releasing GPU memory.
        """
        # Use mx.clear_cache() (not mx.metal.clear_cache which is deprecated)
        if hasattr(self.mx, "clear_cache"):
            self.mx.clear_cache()
        elif hasattr(self.mx, "metal") and hasattr(self.mx.metal, "clear_cache"):
            self.mx.metal.clear_cache()

    def create_causal_mask(self, seq_len: int, dtype: Any | None = None) -> Array:
        """Create additive causal attention mask for autoregressive models."""
        import mlx.nn as nn

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        if dtype is not None:
            mask = mask.astype(self._map_dtype(dtype))
        return mask

    def to_numpy(self, array: Array) -> Any:
        """DISABLED: CPU arrays are not permitted in ModelCypher.

        Use backend.tolist() or backend.to_scalar() for extracting values.
        Use backend.save_safetensors() for serialization.
        """
        raise_numpy_disabled()

    def to_scalar(self, array: Array) -> float | int:
        """Extract a scalar from a 0-d or single-element array.

        Faster than to_numpy().item() - uses MLX's native .item() directly,
        skipping numpy conversion entirely.

        Args:
            array: A scalar (0-d) or single-element array. Must be evaluated first.

        Returns:
            Python float or int.

        Raises:
            ValueError: If array has more than one element.
        """
        return to_scalar_with_eval(array, self.eval)

    def tolist(self, array: Array) -> list | float | int:
        """Convert array to nested Python lists.

        Uses MLX's native tolist() - MUCH faster than element-by-element to_scalar().
        """
        return to_list_with_eval(array, self.eval)

    @lru_cache(maxsize=8)
    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        """Return floating-point precision info for the given dtype.

        Derives numerical stability constants from the actual dtype precision.
        Cached for performance since dtype info is immutable.
        """
        resolved = self._map_dtype(dtype) or self.mx.float32
        finfo_dtype = self.mx.float32 if resolved == self.mx.bfloat16 else resolved
        if hasattr(self.mx, "finfo"):
            info = self.mx.finfo(finfo_dtype)
            # MLX finfo doesn't have 'tiny', compute from IEEE 754 standard
            # float32: smallest normal = 2^(-126) ≈ 1.175e-38
            # float16: smallest normal = 2^(-14) ≈ 6.1e-5
            if hasattr(info, 'tiny'):
                tiny = float(info.tiny)
            elif finfo_dtype == self.mx.float16:
                tiny = 6.103515625e-05  # 2^(-14)
            else:
                tiny = 1.1754943508222875e-38  # 2^(-126) for float32
            return FloatInfo(
                eps=float(info.eps),
                tiny=tiny,
                max=float(info.max),
                min=float(info.min),
            )
        return self._compute_finfo(finfo_dtype)

    # --- Array Creation (lazy - no eval) ---
    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        return self.mx.eye(n, m, dtype=self._map_dtype(dtype))

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        if stop is None:
            return self.mx.arange(start, dtype=self._map_dtype(dtype))
        return self.mx.arange(start, stop, step, dtype=self._map_dtype(dtype))

    def triu_indices(self, n: int, k: int = 0) -> tuple[Array, Array]:
        if n <= 0:
            empty = self.array([], dtype="int32")
            return empty, empty

        idx = self.mx.arange(n, dtype=self.mx.int32)
        row = self.mx.reshape(idx, (n, 1))
        col = self.mx.reshape(idx, (1, n))
        row_idx = self.mx.broadcast_to(row, (n, n))
        col_idx = self.mx.broadcast_to(col, (n, n))
        mask = (col_idx - row_idx) >= k

        row_flat = self.mx.reshape(row_idx, (-1,))
        col_flat = self.mx.reshape(col_idx, (-1,))
        mask_flat = self.mx.reshape(mask, (-1,))
        mask_int = self.astype(mask_flat, "int32")
        valid_count = self.mx.sum(mask_int)

        sentinel = n * n + 1
        sentinel_arr = self.mx.full(row_flat.shape, sentinel, dtype=row_flat.dtype)
        key = self.mx.where(mask_flat, row_flat * n + col_flat, sentinel_arr)
        sorted_idx = self.mx.argsort(key)
        self.eval(sorted_idx, valid_count)

        count = int(self.to_scalar(valid_count))
        if count == 0:
            empty = self.array([], dtype="int32")
            return empty, empty

        row_sorted = self.mx.take(row_flat, sorted_idx, axis=0)
        col_sorted = self.mx.take(col_flat, sorted_idx, axis=0)
        return row_sorted[:count], col_sorted[:count]

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.mx.diag(array, k=k)

    def dtype(self, array: Array) -> Any:
        """Return the dtype of an array."""
        return array.dtype

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return self.mx.full(shape, fill_value, dtype=self._map_dtype(dtype))

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        if dtype:
            return self.mx.ones_like(array, dtype=self._map_dtype(dtype))
        return self.mx.ones_like(array)

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        if dtype:
            return self.mx.zeros_like(array, dtype=self._map_dtype(dtype))
        return self.mx.zeros_like(array)

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return self.mx.linspace(start, stop, num, dtype=self._map_dtype(dtype))

    def meshgrid(self, *arrays: Array, indexing: str = "xy") -> list[Array]:
        return list(self.mx.meshgrid(*arrays, indexing=indexing))

    # --- Shape Manipulation (lazy - no eval) ---
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.mx.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.mx.concatenate(arrays, axis=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.mx.broadcast_to(array, shape)

    def tile(self, array: Array, reps: tuple[int, ...] | int) -> Array:
        return self.mx.tile(array, reps)

    def expand_dims(self, array: Array, axis: int | tuple[int, ...]) -> Array:
        return self.mx.expand_dims(array, axis=axis)

    # --- Reductions (lazy - no eval) ---
    def mean(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.mx.mean(array, axis=axis, keepdims=keepdims)

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.mx.min(array, axis=axis, keepdims=keepdims)

    def argmax(self, array: Array, axis: int | None = None) -> Array:
        return self.mx.argmax(array, axis=axis)

    def argmin(self, array: Array, axis: int | None = None) -> Array:
        return self.mx.argmin(array, axis=axis)

    def var(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.mx.var(array, axis=axis, keepdims=keepdims)

    def std(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.mx.std(array, axis=axis, keepdims=keepdims)

    def all(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.mx.all(array, axis=axis, keepdims=keepdims)

    def any(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.mx.any(array, axis=axis, keepdims=keepdims)

    # --- Element-wise Operations (lazy - no eval) ---
    def sign(self, array: Array) -> Array:
        return self.mx.sign(array)

    def isnan(self, array: Array) -> Array:
        return self.mx.isnan(array)

    def isinf(self, array: Array) -> Array:
        return self.mx.isinf(array)

    def isfinite(self, array: Array) -> Array:
        return self.mx.isfinite(array)

    def sin(self, array: Array) -> Array:
        return self.mx.sin(array)

    def cos(self, array: Array) -> Array:
        return self.mx.cos(array)

    def arccos(self, array: Array) -> Array:
        return self.mx.arccos(array)

    def arctan(self, array: Array) -> Array:
        return self.mx.arctan(array)

    def lgamma(self, array: Array) -> Array:
        # Lanczos approximation (vectorized) for log-gamma without NumPy.
        z = self.mx.array(array)
        coeffs = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.3234287776531,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
        g = 7.0
        half = 0.5
        one = 1.0
        pi = 3.141592653589793
        log_sqrt_two_pi = 0.9189385332046727  # 0.5*log(2*pi)

        def _lgamma_lanczos(x: Array) -> Array:
            x = x - one
            series = self.mx.full(x.shape, coeffs[0])
            for i, c in enumerate(coeffs[1:], start=1):
                series = series + (c / (x + i))
            t = x + g + half
            return (
                log_sqrt_two_pi
                + (x + half) * self.mx.log(t)
                - t
                + self.mx.log(series)
            )

        reflect = z < half
        z_reflect = one - z
        main = _lgamma_lanczos(self.mx.where(reflect, z_reflect, z))
        reflected = self.mx.log(pi) - self.mx.log(self.mx.sin(pi * z)) - main
        return self.mx.where(reflect, reflected, main)

    def clip(
        self, array: Array, min_val: float | Array | None, max_val: float | Array | None
    ) -> Array:
        return self.mx.clip(array, min_val, max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return self.mx.where(condition, x, y)

    def softmax(self, array: Array, axis: int = -1) -> Array:
        return self.mx.softmax(array, axis=axis)

    def cumsum(self, array: Array, axis: int | None = None) -> Array:
        return self.mx.cumsum(array, axis=axis)

    def floor(self, array: Array) -> Array:
        return self.mx.floor(array)

    def ceil(self, array: Array) -> Array:
        return self.mx.ceil(array)

    def log2(self, array: Array) -> Array:
        # MLX has native log2 support
        return self.mx.log2(array)

    def mod(self, lhs: Array, rhs: Array | float | int) -> Array:
        return lhs % rhs

    # --- Linear Algebra (lazy except CPU stream ops) ---
    def dot(self, a: Array, b: Array) -> Array:
        # MLX uses matmul for general case; for 1D vectors use sum of element-wise product
        if a.ndim == 1 and b.ndim == 1:
            return self.mx.sum(a * b)
        return self.mx.matmul(a, b)

    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        # Ensure array is MLX array
        if not hasattr(array, '__mlx_array__') and hasattr(array, '__array__'):
            array = self.mx.array(array)
        return self.mx.linalg.norm(array, axis=axis, keepdims=keepdims)

    def det(self, array: Array) -> Array:
        """Compute determinant via LU decomposition.

        MLX doesn't have linalg.det, so we compute it as:
        det(A) = det(U) * sign(permutation)
        where PA = LU and det(L) = 1 (unit diagonal).

        Supports batched matrices: if array is 3D [batch, n, n],
        returns 1D array of determinants [batch].
        """
        # Handle batched matrices (3D array)
        if len(array.shape) == 3:
            batch_size = int(array.shape[0])
            dets = []
            for i in range(batch_size):
                dets.append(self._det_single(array[i]))
            result = self.mx.stack(dets)
            self.mx.eval(result)
            return result

        return self._det_single(array)

    def _det_single(self, array: Array) -> Array:
        """Compute determinant of a single 2D matrix."""
        try:
            p, L, U = self.mx.linalg.lu(array)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            p, L, U = self.mx.linalg.lu(array, stream=self.mx.cpu)
        self.mx.eval(p, L, U)

        diag_U = self.mx.diag(U)
        det_U = self.mx.prod(diag_U)
        self.mx.eval(det_U)

        p_list = [int(x) for x in self.tolist(p)]
        n = len(p_list)
        swaps = 0
        seen = [False] * n
        for i in range(n):
            if seen[i]:
                continue
            j = i
            cycle_len = 0
            while not seen[j]:
                seen[j] = True
                j = p_list[j]
                cycle_len += 1
            swaps += cycle_len - 1

        sign = 1.0 if swaps % 2 == 0 else -1.0
        return det_U * sign

    def linalg_det(self, array: Array) -> Array:
        """Alias for det() for compatibility."""
        return self.det(array)

    def inv(self, array: Array) -> Array:
        """Compute the inverse of a square matrix.

        Uses backend native implementation (GPU acceleration where available).
        """
        return self.mx.linalg.inv(array)

    def matrix_sqrt_newton_schulz(self, A: Array, num_iters: int = 15) -> Array:
        """Compute matrix square root AND inverse square root via Newton-Schulz.
        
        Converges to (A^{1/2}, A^{-1/2}) for positive semi-definite A.
        Runs entirely on GPU.
        
        Algorithm:
            Y₀ = A / norm(A)
            Z₀ = I
            for k=0..N:
                T = (3I - ZₖYₖ)
                Yₖ₊₁ = ½ Yₖ T
                Zₖ₊₁ = ½ T Zₖ
            
            A^{1/2}   ≈ Y_final * sqrt(norm(A))
            A^{-1/2}  ≈ Z_final / sqrt(norm(A))
        """
        # Ensure float32 for stability
        A_f32 = self.astype(A, "float32")
        
        # Scaling to ensure convergence (spectral norm <= 1)
        padding = self.mx.array(1e-7, dtype=self.mx.float32)
        normA_val = self.mx.sqrt(self.mx.sum(A_f32 * A_f32)) + padding
        Y = A_f32 / normA_val
        
        shape = A.shape
        I = self.eye(shape[0], dtype="float32")
        Z = I
        
        three = self.mx.array(3.0, dtype=self.mx.float32)
        half = self.mx.array(0.5, dtype=self.mx.float32)
        
        # Iteration
        for _ in range(num_iters):
            ZY = self.mx.matmul(Z, Y)
            T = three * I - ZY
            
            Y_new = half * self.mx.matmul(Y, T)
            Z_new = half * self.mx.matmul(T, Z)
            
            Y = Y_new
            Z = Z_new
            
        # Rescale
        sqrtA = Y * self.mx.sqrt(normA_val)
        
        # Cast back if necessary
        if A.dtype != self.mx.float32:
            sqrtA = self.astype(sqrtA, A.dtype)
            
        self.mx.eval(sqrtA)
        return sqrtA

    def eigh(self, array: Array) -> tuple[Array, Array]:
        # MLX eigh only supports float32, float64, complex64 - convert others to float32
        if array.dtype in (self.mx.bfloat16, self.mx.float16):
            array = array.astype(self.mx.float32)
            self.mx.eval(array)
        try:
            eigenvalues, eigenvectors = self.mx.linalg.eigh(array)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            eigenvalues, eigenvectors = self.mx.linalg.eigh(array, stream=self.mx.cpu)
        self.mx.eval(eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    def eigvalsh(self, array: Array) -> Array:
        """Compute eigenvalues of symmetric/Hermitian matrix (values only, more efficient)."""
        # MLX 0.30+ has native eigvalsh - more efficient than eigh
        if array.dtype in (self.mx.bfloat16, self.mx.float16):
            array = array.astype(self.mx.float32)
            self.mx.eval(array)
        try:
            eigenvalues = self.mx.linalg.eigvalsh(array)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            eigenvalues = self.mx.linalg.eigvalsh(array, stream=self.mx.cpu)
        self.mx.eval(eigenvalues)
        return eigenvalues

    def solve(self, a: Array, b: Array) -> Array:
        # Force CPU for linalg.solve - GPU has stability issues with some matrices
        arr = self.mx.linalg.solve(a, b, stream=self.mx.cpu)
        self.mx.eval(arr)
        return arr

    def inv(self, array: Array) -> Array:
        try:
            arr = self.mx.linalg.inv(array)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            arr = self.mx.linalg.inv(array, stream=self.mx.cpu)
        self.mx.eval(arr)
        return arr

    def pinv(self, array: Array) -> Array:
        # MLX pinv requires float32/float64, not bfloat16
        # Cast to float32, compute, then cast back to original dtype
        original_dtype = array.dtype
        array_f32 = array.astype(self.mx.float32) if "bfloat" in str(original_dtype) else array
        try:
            arr = self.mx.linalg.pinv(array_f32)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            arr = self.mx.linalg.pinv(array_f32, stream=self.mx.cpu)
        self.mx.eval(arr)
        # Cast back to original dtype if needed
        if "bfloat" in str(original_dtype):
            arr = arr.astype(original_dtype)
            self.mx.eval(arr)
        return arr

    def cholesky(self, array: Array) -> Array:
        # MLX cholesky requires CPU stream - must eval
        arr = self.mx.linalg.cholesky(array, stream=self.mx.cpu)
        self.mx.eval(arr)
        return arr

    def trace(self, array: Array) -> Array:
        # MLX doesn't have direct trace - compute via diagonal sum
        return self.mx.sum(self.mx.diag(array))

    def qr(self, array: Array) -> tuple[Array, Array]:
        try:
            q, r = self.mx.linalg.qr(array)
        except Exception as exc:
            if "not yet supported on the GPU" not in str(exc):
                raise
            q, r = self.mx.linalg.qr(array, stream=self.mx.cpu)
        self.mx.eval(q, r)
        return q, r

    # --- Indexing (lazy - no eval) ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        if axis is not None:
            return self.mx.take(array, indices, axis=axis)
        return self.mx.take(array, indices)

    def take_along_axis(self, array: Array, indices: Array, axis: int) -> Array:
        """Take values from array along axis using indices array.

        MLX doesn't have native take_along_axis, so we implement it via
        flattening and linear index computation for the gather axis.
        """
        # Handle negative axis
        ndim = len(array.shape)
        if axis < 0:
            axis = ndim + axis

        # Ensure indices are int64 to avoid overflow on large arrays
        indices_int = indices.astype(self.mx.int64)

        # For 2D arrays along axis=1 (the common case), use efficient gather
        if ndim == 2 and axis == 1:
            n_rows = int(array.shape[0])
            n_cols = int(array.shape[1])
            n_gather = int(indices.shape[1])

            # Compute linear indices: row * n_cols + col_index
            row_offsets = self.mx.reshape(
                self.mx.arange(n_rows, dtype=self.mx.int64), (-1, 1)
            ) * n_cols
            linear_indices = row_offsets + indices_int
            flat_array = self.mx.reshape(array, (-1,))
            flat_indices = self.mx.reshape(linear_indices, (-1,))
            gathered = self.mx.take(flat_array, flat_indices)
            return self.mx.reshape(gathered, (n_rows, n_gather))

        # General case: loop over the gather dimension
        # Move gather axis to front for easier iteration
        array_t = self.mx.moveaxis(array, axis, 0)
        indices_t = self.mx.moveaxis(indices_int, axis, 0)

        # Stack results along axis 0
        results = []
        for i in range(int(indices_t.shape[0])):
            gathered = self.mx.take(array_t, indices_t[i], axis=0)
            results.append(gathered)

        result = self.mx.stack(results, axis=0)
        return self.mx.moveaxis(result, 0, axis)

    def put_along_axis(
        self, array: Array, indices: Array, values: Array, axis: int | None = None
    ) -> Array:
        indices_int = indices.astype(self.mx.int32)
        return self.mx.put_along_axis(array, indices_int, values, axis=axis)

    # --- Sorting (lazy - no eval) ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return self.mx.sort(array, axis=axis)

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return self.mx.argsort(array, axis=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        return self.mx.argpartition(array, kth=kth, axis=axis)

    def partition(self, array: Array, kth: int, axis: int = -1) -> Array:
        """Partition array elements around kth element along axis.

        Parameters
        ----------
        array : Array
            Input array.
        kth : int
            Element index to partition around.
        axis : int, optional
            Axis along which to partition. Default is -1.

        Returns
        -------
        Array
            Partitioned array where elements less than kth are before it.
        """
        # O(n) partition algorithm
        return self.mx.partition(array, kth=kth, axis=axis)

    def nonzero(self, array: Array) -> tuple[Array, ...]:
        """Find indices of non-zero elements.

        MLX doesn't have native nonzero, so we implement it using argsort.

        Parameters
        ----------
        array : Array
            Input array.

        Returns
        -------
        tuple[Array, ...]
            Tuple of arrays, one for each dimension, containing indices
            of non-zero elements.
        """
        # Flatten array to work with 1D indices
        flat = self.mx.reshape(array, (-1,))
        n_total = flat.size

        # Create mask for non-zero elements
        mask = flat != 0

        # Count non-zero elements
        n_nonzero = int(self.mx.sum(mask.astype(self.mx.int32)))

        if n_nonzero == 0:
            # Return empty arrays for each dimension
            ndim = len(array.shape)
            return tuple(self.mx.array([], dtype=self.mx.int32) for _ in range(ndim))

        # Use argsort on mask to get non-zero indices at the end
        # False=0, True=1, so argsort puts True values last
        sorted_indices = self.mx.argsort(mask.astype(self.mx.int32))
        nonzero_flat_indices = sorted_indices[-n_nonzero:]

        # Sort to maintain original order
        nonzero_flat_indices = self.mx.sort(nonzero_flat_indices)

        # Unravel flat indices to multi-dimensional indices
        shape = array.shape
        result: list[Array] = []
        remaining = nonzero_flat_indices

        for dim in reversed(range(len(shape))):
            dim_size = shape[dim]
            result.append(remaining % dim_size)
            remaining = remaining // dim_size

        return tuple(reversed(result))

    # --- Random (lazy - no eval) ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.mx.random.normal(shape=shape, dtype=self._map_dtype(dtype) or self.mx.float32)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array:
        return self.mx.random.uniform(
            low=low,
            high=high,
            shape=shape or (1,),
            dtype=self._map_dtype(dtype) or self.mx.float32,
        )

    def random_randint(self, low: int, high: int, shape: tuple[int, ...] | None = None) -> Array:
        return self.mx.random.randint(low, high, shape=shape or (1,))

    def random_seed(self, seed: int) -> None:
        self.mx.random.seed(seed)

    def random_categorical(self, logits: Array, num_samples: int = 1) -> Array:
        """Sample from categorical distribution defined by logits."""
        return self.mx.random.categorical(logits, num_samples=num_samples)

    def randperm(self, n: int) -> Array:
        """Generate a random permutation of integers from 0 to n-1."""
        return self.mx.random.permutation(n)

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        # Handle string dtype names
        if isinstance(dtype, str):
            # Normalize MLX-style dtype strings (e.g., "mlx.core.float32" -> "float32")
            if dtype.startswith("mlx.core."):
                dtype = dtype.replace("mlx.core.", "")
            dtype_map = {
                "float32": self.mx.float32,
                "float64": self.mx.float64,  # MLX supports float64 (CPU fallback)
                "float16": self.mx.float16,
                "bfloat16": self.mx.bfloat16,
                "int32": self.mx.int32,
                "int64": self.mx.int64,
                "int16": self.mx.int16,
                "int8": self.mx.int8,
                "uint8": self.mx.uint8,
                "bool": self.mx.bool_,
            }
            return dtype_map.get(dtype, dtype)
        # Handle dtype objects by name (covers numpy/jax dtypes without importing them)
        name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
        name = name.replace("mlx.core.", "")
        dtype_map = {
            "float32": self.mx.float32,
            "float64": self.mx.float64,
            "float16": self.mx.float16,
            "bfloat16": self.mx.bfloat16,
            "int32": self.mx.int32,
            "int64": self.mx.int64,
            "int16": self.mx.int16,
            "int8": self.mx.int8,
            "uint8": self.mx.uint8,
            "bool": self.mx.bool_,
        }
        return dtype_map.get(name, dtype)

    def _compute_finfo(self, dtype: Any) -> FloatInfo:
        """Compute finfo from backend ops when no native finfo is available."""
        one = self.mx.array(1.0, dtype=dtype)
        two = self.mx.array(2.0, dtype=dtype)

        eps = self.mx.array(1.0, dtype=dtype)
        while True:
            half = eps / two
            test = one + half
            self.mx.eval(test)
            if test.item() == 1.0:
                break
            eps = half
        eps_val = float(eps.item())

        tiny = self.mx.array(1.0, dtype=dtype)
        while True:
            half = tiny / two
            self.mx.eval(half)
            if half.item() == 0.0:
                break
            tiny = half
        tiny_val = float(tiny.item())

        max_val = self.mx.array(1.0, dtype=dtype)
        while True:
            doubled = max_val * two
            self.mx.eval(doubled)
            if not bool(self.mx.isfinite(doubled).item()):
                break
            max_val = doubled
        max_scalar = float(max_val.item())

        return FloatInfo(
            eps=eps_val,
            tiny=tiny_val,
            max=max_scalar,
            min=-max_scalar,
        )
        return dtype

    # =========================================================================
    # SOTA PERFORMANCE APIs (MLX 0.30+)
    # =========================================================================

    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        """JIT-compile a function for kernel fusion.

        Parameters
        ----------
        fun : Callable
            Function to compile.
        inputs : list, optional
            List of arrays to capture as implicit inputs.
        outputs : list, optional
            List of arrays to capture as implicit outputs.
        shapeless : bool, optional
            If True, do not recompile on shape changes. Default is False.

        Returns
        -------
        Callable
            Compiled function with fused kernels.
        """
        return self.mx.compile(fun, inputs=inputs, outputs=outputs, shapeless=shapeless)

    def vmap(
        self,
        fun: Callable,
        in_axes: int | tuple | None = 0,
        out_axes: int | tuple | None = 0,
    ) -> Callable:
        """Vectorize a function over batch dimension.

        Parameters
        ----------
        fun : Callable
            Function to vectorize.
        in_axes : int, tuple, or None, optional
            Axis of each input to vectorize over. None means do not vectorize.
            Default is 0.
        out_axes : int, tuple, or None, optional
            Where to place the batch axis in outputs. Default is 0.

        Returns
        -------
        Callable
            Vectorized function that processes batches efficiently.
        """
        return self.mx.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def value_and_grad(self, fun: Callable, argnums: int | list[int] = 0) -> Callable:
        """Return a function that computes both value and gradient of fun."""
        return self.mx.value_and_grad(fun, argnums=argnums)

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays without blocking.

        Parameters
        ----------
        *arrays : Array
            Arrays to evaluate asynchronously.

        Notes
        -----
        Returns immediately while GPU work continues in the background.
        Use for overlapping CPU preparation with GPU computation.
        """
        self.mx.async_eval(*arrays)

    # --- Fused Metal Kernels (mx.fast.*) ---

    def rms_norm(
        self,
        x: Array,
        weight: Array | None,
        eps: float = 1e-5,
        stream: Any | None = None,
    ) -> Array:
        """Apply RMS normalization using fused kernel.

        Parameters
        ----------
        x : Array
            Input array to normalize.
        weight : Array or None
            Scaling weights.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.
        stream : Any, optional
            Stream or device for execution. Default is None.

        Returns
        -------
        Array
            RMS-normalized output.
        """
        return self.mx.fast.rms_norm(x, weight, eps, stream=stream)

    def layer_norm(
        self,
        x: Array,
        weight: Array | None,
        bias: Array | None,
        eps: float = 1e-5,
        stream: Any | None = None,
    ) -> Array:
        """Apply layer normalization using fused kernel.

        Parameters
        ----------
        x : Array
            Input array to normalize.
        weight : Array or None
            Scaling weights.
        bias : Array or None
            Bias terms.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.
        stream : Any, optional
            Stream or device for execution. Default is None.

        Returns
        -------
        Array
            Layer-normalized output.
        """
        return self.mx.fast.layer_norm(x, weight, bias, eps, stream=stream)

    def rope(
        self,
        x: Array,
        dims: int,
        traditional: bool = False,
        base: float | None = 10000.0,
        scale: float = 1.0,
        offset: int | Array = 0,
        freqs: Array | None = None,
        stream: Any | None = None,
    ) -> Array:
        """Apply rotary position embeddings using fused kernel.

        Parameters
        ----------
        x : Array
            Input array.
        dims : int
            Number of dimensions to apply RoPE to.
        traditional : bool, optional
            Use traditional RoPE formulation. Default is False.
        base : float or None, optional
            Base for frequency computation. Default is 10000.0.
        scale : float, optional
            Scaling factor. Default is 1.0.
        offset : int or Array, optional
            Position offset. Default is 0.
        freqs : Array or None, optional
            Optional precomputed frequencies. If set, base must be None.
        stream : Any, optional
            Stream or device for execution. Default is None.

        Returns
        -------
        Array
            Output with rotary position embeddings applied.
        """
        if freqs is not None and base is not None:
            raise ValueError("rope() expects base=None when freqs is provided")
        return self.mx.fast.rope(
            x,
            dims,
            traditional=traditional,
            base=base,
            scale=scale,
            offset=offset,
            freqs=freqs,
            stream=stream,
        )

    def scaled_dot_product_attention(
        self,
        q: Array,
        k: Array,
        v: Array,
        scale: float,
        mask: Array | str | None = None,
        sinks: Array | None = None,
        stream: Any | None = None,
    ) -> Array:
        """Compute scaled dot-product attention using fused kernel.

        Parameters
        ----------
        q : Array
            Query array.
        k : Array
            Key array.
        v : Array
            Value array.
        scale : float
            Scaling factor for attention scores.
        mask : Array, str, or None, optional
            Attention mask. Default is None.
        sinks : Array or None, optional
            Attention sinks. Default is None.
        stream : Any, optional
            Stream or device for execution. Default is None.

        Returns
        -------
        Array
            Attention output.
        """
        return self.mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask, sinks=sinks, stream=stream
        )

    # --- Stream Management for CPU/GPU Parallelism ---

    def new_stream(self, device: str = "gpu") -> Any:
        """Create a new stream for parallel execution.

        Args:
            device: "gpu" or "cpu"

        Returns:
            Stream object for use with stream= parameter
        """
        if device == "cpu":
            return self.mx.cpu
        return self.mx.gpu

    def synchronize(self) -> None:
        """Synchronize all streams (wait for all GPU work to complete)."""
        self.mx.synchronize()

    # --- File I/O (Native Backend Serialization) ---

    def save_safetensors(
        self, path: str, weights: dict[str, Any], metadata: dict[str, str] | None = None
    ) -> None:
        """Save weights to safetensors using MLX native I/O.

        Args:
            path: File path to save to.
            weights: Dictionary of weight name -> array.
            metadata: Optional dictionary of string metadata to include.
        """
        # Ensure all arrays are MLX arrays
        mlx_weights = {}
        for key, value in weights.items():
            if hasattr(value, "__module__") and "mlx" in type(value).__module__:
                # Already an MLX array - ensure it's a GPU-compatible dtype
                if value.dtype == self.mx.float64:
                    mlx_weights[key] = self.mx.astype(value, self.mx.float32)
                else:
                    mlx_weights[key] = value
            else:
                # Convert to float32 by default (GPU-compatible)
                try:
                    if isinstance(value, list):
                        float_list = [float(v) for v in value]
                        mlx_weights[key] = self.mx.array(float_list, dtype=self.mx.float32)
                    else:
                        mlx_weights[key] = self.mx.astype(self.array(value), self.mx.float32)
                except (RuntimeError, OverflowError):
                    # Handle large integers by converting via float first
                    if isinstance(value, list):
                        float_list = [float(v) for v in value]
                        mlx_weights[key] = self.mx.array(float_list, dtype=self.mx.float32)
                    else:
                        mlx_weights[key] = self.mx.array([float(value)], dtype=self.mx.float32)
        if metadata:
            self.mx.save_safetensors(path, mlx_weights, metadata=metadata)
        else:
            self.mx.save_safetensors(path, mlx_weights)

    def load_safetensors(self, path: str) -> dict[str, Any]:
        """Load weights from safetensors using MLX native I/O.

        Args:
            path: File path to load from.

        Returns:
            Dictionary of weight name -> MLX array.
        """
        return self.mx.load(path)

    # --- Model Operations ---

    def load_model(
        self, path: str, adapter_path: str | None = None
    ) -> tuple[Any, Any]:
        """Load a model and tokenizer using mlx_lm.

        Args:
            path: Path to model directory.
            adapter_path: Optional path to LoRA adapter.

        Returns:
            Tuple of (model, tokenizer).
        """
        from mlx_lm import load

        if adapter_path:
            model, tokenizer = load(path, adapter_path=adapter_path)
        else:
            model, tokenizer = load(path)
        return model, tokenizer

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """Generate text using mlx_lm.generate.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        from mlx_lm import generate

        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, **kwargs)

    def get_embed_tokens(self, model: Any) -> Any:
        """Get the embedding matrix from a model.

        Args:
            model: Model object from load_model.

        Returns:
            Embedding weight matrix [vocab_size, hidden_dim].
        """
        base = getattr(model, "model", model)
        return base.embed_tokens.weight

    def get_hidden_dim(self, model: Any) -> int:
        """Get the hidden dimension of a model.

        Args:
            model: Model object from load_model.

        Returns:
            Hidden dimension size.
        """
        embed = self.get_embed_tokens(model)
        return int(embed.shape[1])

    def get_num_layers(self, model: Any) -> int:
        """Get the number of transformer layers in a model.

        Args:
            model: Model object from load_model.

        Returns:
            Number of layers.
        """
        base = getattr(model, "model", model)
        return len(base.layers)

    def encode_tokens(self, tokenizer: Any, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            tokenizer: Tokenizer object from load_model.
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        return tokenizer.encode(text)

    def decode_tokens(self, tokenizer: Any, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokenizer: Tokenizer object from load_model.
            token_ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        return tokenizer.decode(token_ids)

    # --- Activation Collection ---

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect hidden state activations from model layers.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompts: List of input prompts.
            layer_indices: Optional specific layers to collect (None = all).

        Returns:
            Dictionary mapping layer index to activations [batch, seq, hidden].
        """
        base = getattr(model, "model", model)
        n_layers = len(base.layers)
        if layer_indices is None:
            layer_indices = list(range(n_layers))

        activations: dict[int, list[Any]] = {i: [] for i in layer_indices}

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = self.mx.array([tokens])

            # Embedding
            hidden = base.embed_tokens(input_ids)
            self.mx.eval(hidden)

            # Forward through layers
            for layer_idx, layer in enumerate(base.layers):
                hidden = layer(hidden, mask=None, cache=None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                self.mx.eval(hidden)

                if layer_idx in layer_indices:
                    activations[layer_idx].append(hidden)

        # Stack activations per layer
        result = {}
        for layer_idx, acts in activations.items():
            if acts:
                result[layer_idx] = self.mx.concatenate(acts, axis=0)
                self.mx.eval(result[layer_idx])

        return result

    def trace_norm_trajectory(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> list[float]:
        """Trace the norm of hidden states through all layers.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompt: Input prompt.

        Returns:
            List of norms, one per layer (including embedding).
        """
        tokens = tokenizer.encode(prompt)
        input_ids = self.mx.array([tokens])

        base = getattr(model, "model", model)

        # Embedding
        hidden = base.embed_tokens(input_ids)
        self.mx.eval(hidden)

        norms = [float(self.mx.sqrt(self.mx.sum(hidden * hidden)).item())]

        # Each layer
        for layer in base.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            self.mx.eval(hidden)
            norms.append(float(self.mx.sqrt(self.mx.sum(hidden * hidden)).item()))

        return norms

    # --- Neural Network Operations ---

    def silu(self, array: Any) -> Any:
        """SiLU (Swish) activation function: x * sigmoid(x)."""
        from mlx import nn
        return nn.silu(array)

    # --- Extended Activation Collection ---

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect post-embedding activation for a text input."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)
        self.mx.eval(h)

        # Mean pool over sequence
        pooled = self.mx.mean(h, axis=(0, 1))
        self.mx.eval(pooled)
        return pooled

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer MLP intermediate activations."""
        from mlx import nn

        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])
        activations: dict[int, Any] = {}

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            # Get layer components
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            # Attention
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                attn_out = attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            else:
                attn_out = self.mx.zeros_like(h)
            h = h + attn_out

            # Post-attention norm
            if hasattr(layer, "post_attention_layernorm"):
                h_post = layer.post_attention_layernorm(h)
            elif hasattr(layer, "ln_2"):
                h_post = layer.ln_2(h)
            elif hasattr(layer, "ffn_norm"):
                h_post = layer.ffn_norm(h)
            else:
                h_post = h

            # MLP intermediate
            ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                elif hasattr(ff_module, "fc1"):
                    intermediate = ff_module.fc1(h_post)
                else:
                    intermediate = None

                if intermediate is not None:
                    pooled = self.mx.mean(intermediate, axis=(0, 1))
                    self.mx.eval(pooled)
                    activations[layer_idx] = pooled

                # Complete MLP forward
                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        return activations

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
        """Collect per-layer attention Q, K, V activations."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])
        q_activations: dict[int, Any] = {}
        k_activations: dict[int, Any] = {}
        v_activations: dict[int, Any] = {}

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            # Get layer norm
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "q_proj"):
                q = attn.q_proj(h_norm)
                k = attn.k_proj(h_norm)
                v = attn.v_proj(h_norm)
                self.mx.eval(q, k, v)

                q_activations[layer_idx] = self.mx.mean(q, axis=(0, 1))
                k_activations[layer_idx] = self.mx.mean(k, axis=(0, 1))
                v_activations[layer_idx] = self.mx.mean(v, axis=(0, 1))

            # Forward through layer
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result

        return q_activations, k_activations, v_activations

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect logits for the last token position."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        logits = model(input_ids)
        self.mx.eval(logits)

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            last_logits = logits[0, :]
        else:
            last_logits = logits

        self.mx.eval(last_logits)
        return last_logits

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect hidden + intermediate + gate + embedding activations in batch."""
        from mlx import nn
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        if not texts:
            return ProbeActivationBatch(hidden=[], intermediate=[], gate=[], embedding=[])

        # Tokenize all texts
        all_token_ids = [tokenizer.encode(text) for text in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        hidden_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        intermediate_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        gate_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        embedding_results: list[Any] = []
        all_tensors: list[Any] = []

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        # Compute mask for proper mean pooling
        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = self.mx.array(seq_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]

        # Embedding activations
        pooled_embeddings = self.mx.sum(h * mask[:, :, None], axis=1) / denom
        for i in range(batch_size):
            embedding_results.append(pooled_embeddings[i])
        all_tensors.append(pooled_embeddings)

        for layer_idx, layer in enumerate(base.layers):
            # Detect layer type
            is_lfm2 = (
                hasattr(layer, "operator_norm")
                and hasattr(layer, "ffn_norm")
                and hasattr(layer, "feed_forward")
            )

            if is_lfm2:
                h_norm = layer.operator_norm(h)
                if hasattr(layer, "self_attn"):
                    causal = pos[:, None] >= pos[None, :]
                    attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
                    attn_mask = attn_mask[:, None, :, :]
                    attn_out = layer.self_attn(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            # MLP computation
            intermediate = None
            gate = None
            mlp_out = None

            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "down_proj"):
                        mlp_out = ff_module.down_proj(intermediate)
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "w2"):
                        mlp_out = ff_module.w2(intermediate)
                elif hasattr(ff_module, "fc1"):
                    intermediate = ff_module.fc1(h_post)

            if intermediate is not None:
                pooled_intermediate = self.mx.sum(intermediate * mask[:, :, None], axis=1) / denom
                for i in range(batch_size):
                    intermediate_results[i][layer_idx] = pooled_intermediate[i]
                all_tensors.append(pooled_intermediate)

            if gate is not None:
                pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                for i in range(batch_size):
                    gate_results[i][layer_idx] = pooled_gate[i]
                all_tensors.append(pooled_gate)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    mlp_out = self.mx.zeros_like(h)
            h = h + mlp_out

            # Hidden state
            pooled_hidden = self.mx.sum(h * mask[:, :, None], axis=1) / denom
            for i in range(batch_size):
                hidden_results[i][layer_idx] = pooled_hidden[i]
            all_tensors.append(pooled_hidden)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return ProbeActivationBatch(
            hidden=hidden_results,
            intermediate=intermediate_results,
            gate=gate_results,
            embedding=embedding_results,
        )

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer hidden activations for multiple texts."""
        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors = []

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result

            for i in range(batch_size):
                seq_len = len(all_token_ids[i])
                pooled = self.mx.mean(h[i, :seq_len, :], axis=0)
                results[i][layer_idx] = pooled
                all_tensors.append(pooled)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer intermediate activations for multiple texts."""
        from mlx import nn

        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors = []

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                attn_out = attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            else:
                attn_out = self.mx.zeros_like(h)
            h = h + attn_out

            if hasattr(layer, "post_attention_layernorm"):
                h_post = layer.post_attention_layernorm(h)
            elif hasattr(layer, "ln_2"):
                h_post = layer.ln_2(h)
            elif hasattr(layer, "ffn_norm"):
                h_post = layer.ffn_norm(h)
            else:
                h_post = h

            ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                elif hasattr(ff_module, "fc1"):
                    intermediate = ff_module.fc1(h_post)
                else:
                    intermediate = None

                if intermediate is not None:
                    for i in range(batch_size):
                        seq_len = len(all_token_ids[i])
                        pooled = self.mx.mean(intermediate[i, :seq_len, :], axis=0)
                        results[i][layer_idx] = pooled
                        all_tensors.append(pooled)

                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer gate (pre-SiLU) activations for multiple texts."""
        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors: list[Any] = []

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = self.mx.array(seq_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]

        for layer_idx, layer in enumerate(base.layers):
            is_lfm2 = (
                hasattr(layer, "operator_norm")
                and hasattr(layer, "ffn_norm")
                and hasattr(layer, "feed_forward")
            )

            if is_lfm2:
                h_norm = layer.operator_norm(h)
                if hasattr(layer, "self_attn"):
                    causal = pos[:, None] >= pos[None, :]
                    attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
                    attn_mask = attn_mask[:, None, :, :]
                    attn_out = layer.self_attn(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            if ff_module is not None:
                if hasattr(ff_module, "gate_proj"):
                    gate = ff_module.gate_proj(h_post)
                    pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                    for i in range(batch_size):
                        results[i][layer_idx] = pooled_gate[i]
                    all_tensors.append(pooled_gate)
                elif hasattr(ff_module, "w1"):
                    gate = ff_module.w1(h_post)
                    pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                    for i in range(batch_size):
                        results[i][layer_idx] = pooled_gate[i]
                    all_tensors.append(pooled_gate)

                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = h + (result[0] if isinstance(result, tuple) else result)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect full trajectory activations for manifold mapping."""
        from mlx import nn
        from modelcypher.ports.activation_provider import TrajectoryActivations

        if not texts:
            return TrajectoryActivations(
                positions={},
                velocities={},
                intermediate_positions={},
                embedding_positions=self.mx.zeros((0, 1)),
                q_positions={},
                k_positions={},
                v_positions={},
                gate_positions={},
                text_lengths=[],
                total_tokens=0,
                n_texts=0,
            )

        all_token_ids = [tokenizer.encode(t) for t in texts]
        text_lengths = [len(ids) for ids in all_token_ids]
        total_tokens = sum(text_lengths)
        n_texts = len(texts)

        max_len = max(text_lengths)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)

        base = getattr(model, "model", model)
        h = base.embed_tokens(input_ids)

        # Embedding positions
        embedding_positions_list = []
        for i in range(n_texts):
            seq_len = text_lengths[i]
            embedding_positions_list.append(h[i, :seq_len, :])
        embedding_positions = self.mx.concatenate(embedding_positions_list, axis=0)

        positions: dict[int, Any] = {}
        velocities: dict[int, Any] = {}
        intermediate_positions: dict[int, Any] = {}
        q_positions: dict[int, Any] = {}
        k_positions: dict[int, Any] = {}
        v_positions: dict[int, Any] = {}
        gate_positions: dict[int, Any] = {}
        all_tensors: list[Any] = [embedding_positions]

        seq_lengths_arr = self.mx.array(text_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < seq_lengths_arr[:, None]
        causal = pos[:, None] >= pos[None, :]
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
        attn_mask = attn_mask[:, None, :, :]

        for layer_idx, layer in enumerate(base.layers):
            is_lfm2 = (
                hasattr(layer, "operator_norm")
                and hasattr(layer, "ffn_norm")
                and hasattr(layer, "feed_forward")
            )

            if is_lfm2:
                h_norm = layer.operator_norm(h)
                if hasattr(layer, "self_attn"):
                    attn_module = layer.self_attn
                    attn_out = attn_module(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                    attn_module = None
                else:
                    attn_out = self.mx.zeros_like(h)
                    attn_module = None
                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                attn_module = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn_module is not None:
                    attn_out = attn_module(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            # Q/K/V extraction
            if attn_module is not None and hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(h_norm)
                k = attn_module.k_proj(h_norm)
                v = attn_module.v_proj(h_norm)

                q_pos_list, k_pos_list, v_pos_list = [], [], []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    q_pos_list.append(q[i, :seq_len, :])
                    k_pos_list.append(k[i, :seq_len, :])
                    v_pos_list.append(v[i, :seq_len, :])

                layer_q_pos = self.mx.concatenate(q_pos_list, axis=0)
                layer_k_pos = self.mx.concatenate(k_pos_list, axis=0)
                layer_v_pos = self.mx.concatenate(v_pos_list, axis=0)

                q_positions[layer_idx] = layer_q_pos
                k_positions[layer_idx] = layer_k_pos
                v_positions[layer_idx] = layer_v_pos
                all_tensors.extend([layer_q_pos, layer_k_pos, layer_v_pos])

            # Intermediate and gate extraction
            intermediate = None
            gate = None
            mlp_out = None

            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "down_proj"):
                        mlp_out = ff_module.down_proj(intermediate)
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "w2"):
                        mlp_out = ff_module.w2(intermediate)
                elif hasattr(ff_module, "fc1"):
                    intermediate = ff_module.fc1(h_post)

            if intermediate is not None:
                int_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    int_pos_list.append(intermediate[i, :seq_len, :])
                layer_int_pos = self.mx.concatenate(int_pos_list, axis=0)
                intermediate_positions[layer_idx] = layer_int_pos
                all_tensors.append(layer_int_pos)

            if gate is not None:
                gate_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    gate_pos_list.append(gate[i, :seq_len, :])
                layer_gate_pos = self.mx.concatenate(gate_pos_list, axis=0)
                gate_positions[layer_idx] = layer_gate_pos
                all_tensors.append(layer_gate_pos)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    mlp_out = self.mx.zeros_like(h)
            h = h + mlp_out

            # Hidden state positions and velocities
            layer_positions_list = []
            layer_velocities_list = []

            for i in range(n_texts):
                seq_len = text_lengths[i]
                text_positions = h[i, :seq_len, :]
                layer_positions_list.append(text_positions)

                if seq_len >= 2:
                    text_velocities = text_positions[1:, :] - text_positions[:-1, :]
                    layer_velocities_list.append(text_velocities)

            if layer_positions_list:
                layer_positions = self.mx.concatenate(layer_positions_list, axis=0)
                positions[layer_idx] = layer_positions
                all_tensors.append(layer_positions)

            if layer_velocities_list:
                layer_velocities = self.mx.concatenate(layer_velocities_list, axis=0)
                velocities[layer_idx] = layer_velocities
                all_tensors.append(layer_velocities)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return TrajectoryActivations(
            positions=positions,
            velocities=velocities,
            intermediate_positions=intermediate_positions,
            embedding_positions=embedding_positions,
            q_positions=q_positions,
            k_positions=k_positions,
            v_positions=v_positions,
            gate_positions=gate_positions,
            text_lengths=text_lengths,
            total_tokens=total_tokens,
            n_texts=n_texts,
        )
