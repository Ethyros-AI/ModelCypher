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

from modelcypher.backends.safe_gpu import SafeGPU
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
        self.safe = SafeGPU(mx)
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

    def abs(self, array: Array) -> Array:
        return self.mx.abs(array)

    def astype(self, array: Array, dtype: Any) -> Array:
        # Ensure array is an MLX array before calling astype with MLX dtype
        # Numpy arrays have astype but don't understand MLX dtypes
        module = type(array).__module__
        if module.startswith("numpy") or not module.startswith("mlx"):
            array = self.mx.array(array)
        return array.astype(self._map_dtype(dtype))

    # --- Linear Algebra (requires eval for CPU stream ops) ---
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

        # MLX SVD requires CPU stream - must eval before returning
        result = self.mx.linalg.svd(array, compute_uv=compute_uv, stream=self.mx.cpu)
        if compute_uv:
            u, s, vt = result
            self.safe.eval(u, s, vt)
            return u, s, vt
        self.safe.eval(result)
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
        self.safe.eval(*arrays)

    def clear_cache(self) -> None:
        """Clear Metal GPU memory cache to release lazy computations."""
        import gc

        gc.collect()
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
        """Convert to a host list - requires eval first.

        Note: Returns a Python list, not a numpy array. Use backend operations
        for array math. This method exists for serialization/interop only.
        """
        self.safe.eval(array)
        # Handle bfloat16 which some protocols may not support
        if array.dtype == self.mx.bfloat16:
            array = array.astype(self.mx.float32)
            self.safe.eval(array)
        return array.tolist()

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
        if hasattr(array, "shape"):
            self.safe.eval(array)
        if hasattr(array, "item"):
            return array.item()
        return float(array)

    def tolist(self, array: Array) -> list | float | int:
        """Convert array to nested Python lists.

        Uses MLX's native tolist() - MUCH faster than element-by-element to_scalar().
        """
        self.safe.eval(array)
        return array.tolist()

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
        """
        p, L, U = self.mx.linalg.lu(array, stream=self.mx.cpu)
        self.safe.eval(p, L, U)

        diag_U = self.mx.diag(U)
        det_U = self.mx.prod(diag_U)
        self.safe.eval(det_U)

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

    def eigh(self, array: Array) -> tuple[Array, Array]:
        # MLX eigh requires CPU stream - must eval
        # MLX eigh only supports float32, float64, complex64 - convert others to float32
        if array.dtype in (self.mx.bfloat16, self.mx.float16):
            array = array.astype(self.mx.float32)
            self.safe.eval(array)
        eigenvalues, eigenvectors = self.mx.linalg.eigh(array, stream=self.mx.cpu)
        self.safe.eval(eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    def solve(self, a: Array, b: Array) -> Array:
        # MLX solve requires CPU stream - must eval
        arr = self.mx.linalg.solve(a, b, stream=self.mx.cpu)
        self.safe.eval(arr)
        return arr

    def inv(self, array: Array) -> Array:
        # MLX inv requires CPU stream - must eval
        arr = self.mx.linalg.inv(array, stream=self.mx.cpu)
        self.safe.eval(arr)
        return arr

    def pinv(self, array: Array) -> Array:
        # MLX pinv requires float32/float64, not bfloat16
        # Cast to float32, compute, then cast back to original dtype
        original_dtype = array.dtype
        array_f32 = array.astype(self.mx.float32) if "bfloat" in str(original_dtype) else array
        # MLX pinv requires CPU stream - must eval
        arr = self.mx.linalg.pinv(array_f32, stream=self.mx.cpu)
        self.safe.eval(arr)
        # Cast back to original dtype if needed
        if "bfloat" in str(original_dtype):
            arr = arr.astype(original_dtype)
            self.safe.eval(arr)
        return arr

    def cholesky(self, array: Array) -> Array:
        # MLX cholesky requires CPU stream - must eval
        arr = self.mx.linalg.cholesky(array, stream=self.mx.cpu)
        self.safe.eval(arr)
        return arr

    def trace(self, array: Array) -> Array:
        # MLX doesn't have direct trace - compute via diagonal sum
        return self.mx.sum(self.mx.diag(array))

    def qr(self, array: Array) -> tuple[Array, Array]:
        # MLX QR requires CPU stream - must eval
        q, r = self.mx.linalg.qr(array, stream=self.mx.cpu)
        self.safe.eval(q, r)
        return q, r

    # --- Indexing (lazy - no eval) ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        if axis is not None:
            return self.mx.take(array, indices, axis=axis)
        return self.mx.take(array, indices)

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
            self.safe.eval(test)
            if test.item() == 1.0:
                break
            eps = half
        eps_val = float(eps.item())

        tiny = self.mx.array(1.0, dtype=dtype)
        while True:
            half = tiny / two
            self.safe.eval(half)
            if half.item() == 0.0:
                break
            tiny = half
        tiny_val = float(tiny.item())

        max_val = self.mx.array(1.0, dtype=dtype)
        while True:
            doubled = max_val * two
            self.safe.eval(doubled)
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

    def rms_norm(self, x: Array, weight: Array, eps: float = 1e-5) -> Array:
        """Apply RMS normalization using fused kernel.

        Parameters
        ----------
        x : Array
            Input array to normalize.
        weight : Array
            Scaling weights.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.

        Returns
        -------
        Array
            RMS-normalized output.
        """
        return self.mx.fast.rms_norm(x, weight, eps)

    def layer_norm(
        self, x: Array, weight: Array | None, bias: Array | None, eps: float = 1e-5
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

        Returns
        -------
        Array
            Layer-normalized output.
        """
        return self.mx.fast.layer_norm(x, weight, bias, eps)

    def rope(
        self,
        x: Array,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
        offset: int = 0,
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
        base : float, optional
            Base for frequency computation. Default is 10000.0.
        scale : float, optional
            Scaling factor. Default is 1.0.
        offset : int, optional
            Position offset. Default is 0.

        Returns
        -------
        Array
            Output with rotary position embeddings applied.
        """
        return self.mx.fast.rope(
            x, dims, traditional=traditional, base=base, scale=scale, offset=offset
        )

    def scaled_dot_product_attention(
        self,
        q: Array,
        k: Array,
        v: Array,
        scale: float,
        mask: Array | None = None,
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
        mask : Array or None, optional
            Attention mask. Default is None.

        Returns
        -------
        Array
            Attention output.
        """
        return self.mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

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
